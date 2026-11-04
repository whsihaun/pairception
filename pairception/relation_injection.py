import torch
import torch.nn.functional as F
from torch import nn


class PairwiseMergingBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def additive(self, first, second):
        return first + second

    def forward(self, first, second):
        return self.norm(self.additive(first, second) + first * second)


class QueryMerging(PairwiseMergingBlock):
    def additive(self, first, second):
        return F.gelu(first + second)


class SpatialMerging(PairwiseMergingBlock):
    pass


class RelationMerging(PairwiseMergingBlock):
    pass


def canonical_boxes(boxes, hflip):
    if not hflip:
        return boxes
    return torch.stack((1 - boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]), -1)


def geometry(a, b):
    delta = b[..., :2] - a[..., :2]
    ratio = (b[..., 2:].clamp_min(1e-8) / a[..., 2:].clamp_min(1e-8)).log()
    lo = torch.maximum(a[..., :2] - a[..., 2:] / 2, b[..., :2] - b[..., 2:] / 2)
    hi = torch.minimum(a[..., :2] + a[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2)
    intersection = (hi - lo).clamp_min(0).prod(-1)
    union = a[..., 2:].prod(-1) + b[..., 2:].prod(-1) - intersection
    return torch.cat((delta, ratio, delta.norm(dim=-1, keepdim=True),
                      (intersection / union.clamp_min(1e-8))[..., None]), -1)


def span_indices(offsets, span):
    return [i for i, (s, e) in enumerate(offsets) if e > s and s < span[1] and e > span[0]]


def pool_span(text, offsets, span):
    ids = span_indices(offsets, span)
    if not ids:
        raise ValueError(f'Visible noun phrase has no text token: {span}')
    return text[ids].mean(0)


def visible_phrase_pairs(hypotheses, offsets, limit):
    visible_end = max((end for start, end in offsets if end > start), default=0)
    pairs = []
    for target, auxiliaries in hypotheses:
        if target[1] > visible_end:
            continue
        for auxiliary in auxiliaries:
            if auxiliary[1] <= visible_end and (target, auxiliary) not in pairs:
                pairs.append((target, auxiliary))
    return pairs[:limit]


class RelationInjectionModule(nn.Module):
    def __init__(self, dim=256, injection_dim=128, max_phrase_pairs=12):
        super().__init__()
        self.dim = dim
        self.injection_dim = injection_dim
        self.max_phrase_pairs = max_phrase_pairs
        self.target_query = nn.Linear(dim, injection_dim)
        self.auxiliary_query = nn.Linear(dim, injection_dim)
        self.relation = nn.Linear(dim, injection_dim)
        self.phrase_pair_condition = nn.Linear(2 * dim, injection_dim)
        self.relation_token_attention = nn.Linear(injection_dim, 1)
        self.box_geometry = nn.Sequential(
            nn.Linear(6, injection_dim), nn.GELU(), nn.Linear(injection_dim, injection_dim)
        )
        self.query_merging = QueryMerging(injection_dim)
        self.spatial_merging = SpatialMerging(injection_dim)
        self.relation_merging = RelationMerging(injection_dim)
        self.pair_logit = nn.Sequential(
            nn.Linear(injection_dim, injection_dim), nn.GELU(), nn.Linear(injection_dim, 1)
        )
        self.phrase_relevance = nn.Sequential(
            nn.Linear(4 * dim, injection_dim), nn.GELU(), nn.Linear(injection_dim, 1)
        )
        self.injection_gate = nn.Linear(dim + injection_dim, dim)
        self.injection_out = nn.Linear(injection_dim, dim, bias=False)

        nn.init.zeros_(self.injection_out.weight)
        nn.init.zeros_(self.phrase_relevance[-1].weight)
        nn.init.zeros_(self.phrase_relevance[-1].bias)

    def _relation_text(self, text, text_mask, offsets, target_span, auxiliary_span):
        target_text = pool_span(text, offsets, target_span)
        auxiliary_text = pool_span(text, offsets, auxiliary_span)
        condition = F.gelu(self.phrase_pair_condition(torch.cat((target_text, auxiliary_text), dim=-1)))
        token_state = self.relation(text)
        attention_state = F.gelu(token_state + condition) + token_state * condition
        logits = self.relation_token_attention(attention_state).squeeze(-1)

        eligible = text_mask.bool().clone()
        valid_offsets = text_mask.new_zeros(text.shape[0], dtype=torch.bool)
        for index, (start, end) in enumerate(offsets[: text.shape[0]]):
            valid_offsets[index] = end > start
        eligible &= valid_offsets
        for index in span_indices(offsets, target_span) + span_indices(offsets, auxiliary_span):
            if index < eligible.numel():
                eligible[index] = False
        if not bool(eligible.any()):
            return None
        weights = torch.softmax(logits.masked_fill(~eligible, float('-inf')), dim=0)
        relation_text = torch.sum(weights[:, None] * text, dim=0)
        return target_text, auxiliary_text, relation_text, eligible, logits

    def _pair_state(self, pair, boxes, relation_text):
        pair_geometry = geometry(boxes[:, None], boxes[None, :])
        relation_state = self.relation(relation_text)[None, None, :]
        spatial_relation = self.spatial_merging(relation_state, self.box_geometry(pair_geometry))
        merged = self.relation_merging(pair, spatial_relation)
        return merged, self.pair_logit(merged).squeeze(-1)

    def forward(self, queries, boxes, text_features, text_mask, offsets, hypotheses, hflip):
        count = queries.shape[1]
        injected_queries, batch_records = [], []
        for batch_index in range(queries.shape[0]):
            query = queries[batch_index]
            canonical = canonical_boxes(boxes[batch_index], bool(hflip[batch_index]))
            text = text_features[batch_index]
            sample_mask = text_mask[batch_index]
            sample_offsets = offsets[batch_index]

            target = self.target_query(query)[:, None, :]
            auxiliary = self.auxiliary_query(query)[None, :, :]
            pair = self.query_merging(target, auxiliary)
            records, expected_states, relevance_logits = [], [], []
            for target_span, auxiliary_span in visible_phrase_pairs(hypotheses[batch_index], sample_offsets,
                                                                    self.max_phrase_pairs):
                evidence = self._relation_text(text, sample_mask, sample_offsets, target_span, auxiliary_span)
                if evidence is None:
                    continue
                target_text, auxiliary_text, relation_text, eligible, token_logits = evidence
                merged, pair_logits = self._pair_state(pair, canonical, relation_text)
                diagonal = torch.eye(count, dtype=torch.bool, device=query.device)
                auxiliary_weights = torch.softmax(pair_logits.masked_fill(diagonal, float('-inf')), dim=-1)
                expected_states.append(torch.sum(auxiliary_weights[..., None] * merged, dim=1))
                relevance = self.phrase_relevance(
                    torch.cat((target_text, auxiliary_text, relation_text, query.mean(0)))
                ).squeeze(-1)
                relevance_logits.append(relevance)
                records.append({
                    'target_span': target_span,
                    'auxiliary_span': auxiliary_span,
                    'pair_logits': pair_logits,
                    'relevance_logit': relevance,
                    'relation_token_logits': token_logits,
                    'relation_token_mask': eligible.detach(),
                    'relation_token_offsets': tuple(sample_offsets[:token_logits.numel()]),
                })

            if expected_states:
                relevance_weights = torch.softmax(torch.stack(relevance_logits), dim=0)
                relation_state = torch.sum(relevance_weights[:, None, None] * torch.stack(expected_states), dim=0)
            else:
                relation_state = query.new_zeros((count, self.injection_dim))
            gate = torch.sigmoid(self.injection_gate(torch.cat((query, relation_state), dim=-1)))
            injected_queries.append(query + gate * self.injection_out(relation_state))
            batch_records.append(records)
        return torch.stack(injected_queries), batch_records
