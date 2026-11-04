import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _directory(name, default):
    return Path(os.environ.get(f'PAIRCEPTION_{name}_DIR', ROOT / default))


def data_dir():
    return _directory('DATA', 'data/AerialVG')


def checkpoint_dir():
    return _directory('CHECKPOINT', 'checkpoints')


def output_dir():
    return _directory('OUTPUT', 'outputs')


def cache_dir():
    return _directory('CACHE', 'cache')


def official_checkpoint():
    return checkpoint_dir() / 'aerialvg.pth'


def pairception_checkpoint(seed):
    return checkpoint_dir() / f'pairception_seed{seed}.pt'


def bert_dir():
    return cache_dir() / 'bert-base-uncased'


DETECTOR = dict(
    backbone='swin_T_224_1k', position_embedding='sine', pe_temperatureH=20, pe_temperatureW=20,
    return_interm_indices=[1, 2, 3], backbone_freeze_keywords=None, dilation=False, use_checkpoint=False,
    hidden_dim=256, dropout=0.0, nheads=8, num_queries=900, dim_feedforward=2048, enc_layers=6, dec_layers=6,
    pre_norm=False, query_dim=4, transformer_activation='relu', num_patterns=0, num_feature_levels=4,
    enc_n_points=4, dec_n_points=4, two_stage_type='standard', two_stage_bbox_embed_share=False,
    two_stage_class_embed_share=False, dec_pred_bbox_embed_share=True, embed_init_tgt=True,
    use_text_enhancer=True, use_fusion_layer=True, use_transformer_ckpt=False, use_text_cross_attention=True,
    text_dropout=0.0, fusion_dropout=0.0, fusion_droppath=0.1, max_text_len=256, sub_sentence_present=True,
    dn_labelbook_size=91, dn_box_noise_scale=1.0, dn_label_noise_ratio=0.5,
    relation_num_layers=3, topk=15, relation_weight=0.4,
)
HCA_WEIGHT = 0.3


def detector_args():
    return SimpleNamespace(**DETECTOR, text_encoder_type=str(bert_dir()))


TOP_K = 15
INJECTION_DIM = 128
MAX_PHRASE_PAIRS = 12

SEED = 42
EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-5
LR_DROP = 4
LR_GAMMA = 0.1
CLIP_MAX_NORM = 0.1
NUM_WORKERS = 2
IOU_THRESHOLD = 0.5
LOSS_WEIGHTS = dict(
    loss_cls=1.0, loss_l1=5.0, loss_giou=2.0,
    loss_pair=0.1, loss_phrase=0.05, loss_rel_token=0.05,
)

TRAIN_SCALES = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
EVAL_SCALE = 800
MAX_SIZE = 1333
