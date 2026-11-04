import argparse
import contextlib
import copy
import io

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, BertModel

from . import config
from .relation_injection import RelationInjectionModule
from .upstream import aerialvg, backbone, bertwarper, misc, transformer, utils


class BertEncoder(bertwarper.BertModelWarper):
    def __init__(self, bert_model):
        nn.Module.__init__(self)
        self.config = bert_model.config
        self.embeddings = bert_model.embeddings
        self.encoder = bert_model.encoder
        self.pooler = bert_model.pooler
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids):
        extended_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.size(), dtype=self.embeddings.word_embeddings.weight.dtype
        )
        embedding = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        output = self.encoder(
            embedding, attention_mask=extended_mask, head_mask=[None] * self.config.num_hidden_layers,
            encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=False,
            output_attentions=False, output_hidden_states=False, return_dict=True,
        )
        return {'last_hidden_state': output[0]}


def special_token_masks(input_ids, special_tokens):
    batch, length = input_ids.shape
    special = torch.zeros((batch, length), device=input_ids.device).bool()
    for token in special_tokens:
        special |= input_ids == token
    positions = torch.nonzero(special)
    attention_mask = torch.eye(length, device=input_ids.device).bool().unsqueeze(0).repeat(batch, 1, 1)
    position_ids = torch.zeros((batch, length), device=input_ids.device)
    previous = 0
    for i in range(positions.shape[0]):
        row, col = positions[i]
        if col == 0 or col == length - 1:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous + 1:col + 1, previous + 1:col + 1] = True
            position_ids[row, previous + 1:col + 1] = torch.arange(0, col - previous, device=input_ids.device)
        previous = col
    return attention_mask, position_ids.to(torch.long)


class Pairception(nn.Module):
    def __init__(self, parses=None):
        super().__init__()
        args = config.detector_args()
        self.parses = parses
        self.backbone = backbone.build_backbone(args)
        self.transformer = transformer.build_transformer(args)
        self.relation_injection_module = RelationInjectionModule(args.hidden_dim, config.INJECTION_DIM,
                                                                 config.MAX_PHRASE_PAIRS)
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_type)
        self.bert = BertEncoder(BertModel.from_pretrained(args.text_encoder_type))
        hidden_dim = args.hidden_dim
        self.feat_map = nn.Linear(self.bert.config.hidden_size, hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        self.downsample_conv = aerialvg.SharedKernelConv(kernel_size=3)
        self.special_tokens = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '.', '?'])

        channels = self.backbone.num_channels
        input_proj = [nn.Sequential(nn.Conv2d(c, hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim))
                      for c in channels]
        input_proj.append(nn.Sequential(nn.Conv2d(channels[-1], hidden_dim, kernel_size=3, stride=2, padding=1),
                                        nn.GroupNorm(32, hidden_dim)))
        self.input_proj = nn.ModuleList(input_proj)

        class_embed = utils.ContrastiveEmbed()
        bbox_embed = utils.MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        self.bbox_embed = nn.ModuleList([bbox_embed for _ in range(args.dec_layers)])
        self.class_embed = nn.ModuleList([class_embed for _ in range(args.dec_layers)])
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.enc_out_bbox_embed = copy.deepcopy(bbox_embed)
        self.transformer.enc_out_class_embed = copy.deepcopy(class_embed)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def load_detector(self):
        missing, unexpected = self.load_state_dict(official_state(), strict=False)
        if any(not key.startswith('relation_injection_module.') for key in missing) or any(
                not key.startswith(('relation_transformer.', 'upsample_conv.')) for key in unexpected):
            raise RuntimeError(f'Unexpected AerialVG checkpoint keys: missing={missing}, unexpected={unexpected}')
        self.requires_grad_(False)
        self.relation_injection_module.requires_grad_(True)
        return self

    def encode_text(self, captions, device):
        max_len = config.DETECTOR['max_text_len']
        tokenized = self.tokenizer(captions, padding='longest', return_tensors='pt').to(device)
        attention, position_ids = special_token_masks(tokenized['input_ids'], self.special_tokens)
        attention, position_ids = attention[:, :max_len, :max_len], position_ids[:, :max_len]
        text = self.bert(input_ids=tokenized['input_ids'][:, :max_len], attention_mask=attention,
                         token_type_ids=tokenized['token_type_ids'][:, :max_len], position_ids=position_ids)
        encoded = self.feat_map(text['last_hidden_state'])
        text_dict = {
            'encoded_text': encoded,
            'text_token_mask': tokenized['attention_mask'][:, :max_len].bool(),
            'position_ids': position_ids,
            'text_self_attention_masks': attention,
        }
        offsets = [encoding.offsets[:encoded.shape[1]] for encoding in tokenized.encodings]
        return text_dict, offsets

    @torch.no_grad()
    def detect(self, samples, captions):
        text_dict, offsets = self.encode_text(captions, samples.tensors.device)
        text = text_dict['encoded_text']
        features, poss = self.backbone(samples)
        srcs, masks = [], []
        for level, feature in enumerate(features):
            src, mask = feature.decompose()
            srcs.append(self.input_proj[level](src))
            masks.append(mask)
        src = self.input_proj[len(features)](features[-1].tensors)
        mask = F.interpolate(samples.mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        srcs.append(src)
        masks.append(mask)
        poss.append(self.backbone[1](misc.NestedTensor(src, mask)).to(src.dtype))
        hs, reference, _, _, _ = self.transformer(
            srcs, masks, None, poss, None, None, text_dict,
            sim_parm=config.HCA_WEIGHT, up_conv=None, down_conv=self.downsample_conv,
        )
        boxes = (self.bbox_embed[-1](hs[-1]) + misc.inverse_sigmoid(reference[-2])).sigmoid()
        logits = self.class_embed[-1](hs[-1], text_dict)
        index = logits.sigmoid().max(dim=-1).values.topk(config.TOP_K, dim=1).indices

        def pick(tensor):
            return tensor.gather(1, index[..., None].expand(-1, -1, tensor.shape[-1]))

        return {'queries': pick(hs[-1]), 'boxes': pick(boxes), 'anchors': pick(reference[-2]),
                'logits': pick(logits), 'text': text, 'text_dict': text_dict, 'offsets': offsets}

    def forward(self, samples, captions, hflip=None):
        candidates = self.detect(samples, captions)
        missing = [caption for caption in captions if caption not in self.parses]
        if missing:
            raise KeyError(f'No parse for {missing[0]!r}; run python -m pairception.parse')
        queries, records = self.relation_injection_module(
            candidates['queries'], candidates['boxes'], candidates['text'], candidates['text_dict']['text_token_mask'],
            candidates['offsets'], [self.parses[caption] for caption in captions],
            [False] * len(captions) if hflip is None else hflip,
        )
        boxes = (self.bbox_embed[-1](queries) + misc.inverse_sigmoid(candidates['anchors'])).sigmoid()
        scores = self.class_embed[-1](queries, candidates['text_dict']).max(dim=-1).values
        return {'boxes': boxes, 'scores': scores, 'base_boxes': candidates['boxes'], 'queries': queries,
                'records': records}


def official_state():
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(config.official_checkpoint(), map_location='cpu', weights_only=True)['model']


class Batch(misc.NestedTensor):
    @property
    def device(self):
        return self.tensors.device


def _upstream_masks(tokenized, special_tokens, tokenizer):
    attention_mask, position_ids = special_token_masks(tokenized['input_ids'], special_tokens)
    return attention_mask, position_ids, None


def official_aerialvg():
    aerialvg.BertModelWarper = BertEncoder
    aerialvg.generate_masks_with_special_tokens_and_transfer_map = _upstream_masks
    with contextlib.redirect_stdout(io.StringIO()):
        model = aerialvg.build_aerialvg(config.detector_args())
    model.load_state_dict(official_state())
    return model.requires_grad_(False)
