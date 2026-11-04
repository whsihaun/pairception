import argparse
import json

import torch
from tqdm import tqdm

from . import config, parse
from .data import AerialVG, loader
from .model import Batch, Pairception, official_aerialvg
from .upstream import box_ops, require_kernel

MODELS = ('pairception', 'aerialvg', 'frozen_detector')


@torch.no_grad()
def run(predict, data, device, desc):
    top1 = top5 = top15 = iou_sum = 0.0
    for samples, annotations in tqdm(data, desc=desc, dynamic_ncols=True):
        scores, boxes = predict(samples.to(device), [annotation['caption'] for annotation in annotations])
        order = scores.topk(config.TOP_K, dim=1).indices
        boxes = boxes.gather(1, order[..., None].expand(-1, -1, 4))
        for ranked, annotation in zip(boxes, annotations):
            iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(ranked),
                                  box_ops.box_cxcywh_to_xyxy(annotation['target_box'].to(device)))[0][:, 0]
            hit = iou >= config.IOU_THRESHOLD
            top1 += float(hit[0])
            top5 += float(hit[:5].any())
            top15 += float(hit.any())
            iou_sum += float(iou[0])
    count = len(data.dataset)
    return {'top1': top1 / count, 'top5': top5 / count, 'top15': top15 / count, 'miou': iou_sum / count}


def predictor(name, seed):
    if name == 'aerialvg':
        model = official_aerialvg()

        def predict(samples, captions):
            outputs = model(Batch(samples.tensors, samples.mask), captions=captions)
            return outputs['pred_logits'].max(dim=-1).values.sigmoid(), outputs['pred_boxes']
    elif name == 'frozen_detector':
        model = Pairception().load_detector()

        def predict(samples, captions):
            candidates = model.detect(samples, captions)
            return candidates['logits'].max(dim=-1).values, candidates['boxes']
    else:
        model = Pairception(parse.load()).load_detector()
        state = torch.load(config.pairception_checkpoint(seed), map_location='cpu', weights_only=True)
        model.relation_injection_module.load_state_dict(state['relation_injection_module'])

        def predict(samples, captions):
            outputs = model(samples, captions)
            return outputs['scores'], outputs['boxes']
    return model, predict


def evaluate(name, split, seed, device, limit=None):
    if device.type == 'cuda':
        require_kernel()
    model, predict = predictor(name, seed)
    model.to(device).eval()
    metrics = run(predict, loader(AerialVG(split, model.tokenizer, filtered=False, limit=limit), shuffle=False),
                  device, split)
    tag = f'pairception_seed{seed}' if name == 'pairception' else name
    config.output_dir().mkdir(parents=True, exist_ok=True)
    (config.output_dir() / f'{tag}_{split}.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, default='pairception',
                        help='frozen_detector is AerialVG w/o relation module')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--seed', type=int, default=config.SEED, help='training seed of the Pairception checkpoint')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    metrics = evaluate(args.model, args.split, args.seed, torch.device(args.device))
    print('  '.join(f'{label} {100 * metrics[key]:.2f}'
                    for label, key in (('Top-1', 'top1'), ('Top-5', 'top5'), ('Top-15', 'top15'), ('mIoU', 'miou'))))


if __name__ == '__main__':
    main()
