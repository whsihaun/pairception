import json
import random
import re

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import config
from .upstream import misc, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, img, annotation):
        if random.random() < self.p:
            img, annotation = T.hflip(img, annotation)
            annotation['hflip'] = True
        return img, annotation


def train_transforms():
    return T.Compose([RandomHorizontalFlip(), T.RandomResize(config.TRAIN_SCALES, max_size=config.MAX_SIZE),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transforms():
    return T.Compose([T.RandomResize([config.EVAL_SCALE], max_size=config.MAX_SIZE), T.ToTensor(),
                      T.Normalize(MEAN, STD)])


def read_split(split):
    with (config.data_dir() / 'annotation' / f'vg_{split}_odvg.jsonl').open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def _caption(record):
    caption = record['grounding'].get('caption')
    return caption if isinstance(caption, str) else ''


def _phrases(record):
    return [region['phrase'] if isinstance(region.get('phrase'), str) else '' for region in record['grounding']['regions']]


def phrase_spans(caption, phrases):
    occurrences, used, spans = {}, {}, []
    for phrase in phrases:
        key = phrase.casefold()
        if not phrase.strip():
            spans.append(None)
            continue
        if key not in occurrences:
            occurrences[key] = [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), caption, flags=re.IGNORECASE)]
        index = used.get(key, 0)
        if index < len(occurrences[key]):
            spans.append(occurrences[key][index])
            used[key] = index + 1
        else:
            spans.append(None)
    return spans


def _spans_reach_tokens(tokenizer, caption, spans):
    encoding = tokenizer(caption, padding=False, truncation=True, max_length=config.DETECTOR['max_text_len'],
                         return_tensors='pt')
    for span in spans:
        if span is None:
            return False
        start, end = span
        first = next((t for c in range(start, end) if (t := encoding.char_to_token(0, c)) is not None), None)
        last = next((t for c in range(end - 1, start - 1, -1) if (t := encoding.char_to_token(0, c)) is not None), None)
        if first is None or last is None or first > last or first >= config.DETECTOR['max_text_len']:
            return False
    return True


class AerialVG(Dataset):
    def __init__(self, split, tokenizer, filtered, limit=None):
        records = read_split(split)[:limit]
        if filtered:
            records = [r for r in records
                       if _spans_reach_tokens(tokenizer, _caption(r), phrase_spans(_caption(r), _phrases(r)))]
        else:
            records = [r for r in records if _caption(r).strip()]
        self.records = records
        self.transforms = train_transforms() if split == 'train' else eval_transforms()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        caption = _caption(record)
        regions = record['grounding']['regions']
        spans = phrase_spans(caption, _phrases(record))
        image = Image.open(config.data_dir() / 'images' / record['filename']).convert('RGB')
        annotation = {
            'boxes': torch.as_tensor([region['bbox'] for region in regions], dtype=torch.float32).reshape(-1, 4),
            'caption': caption,
            'spans': torch.as_tensor([span if span is not None else (-1, -1) for span in spans], dtype=torch.int64),
            'relations': [region['realation'] for region in regions[1:]],
            'hflip': False,
        }
        image, annotation = self.transforms(image, annotation)
        annotation['target_box'] = annotation['boxes'][0:1]
        annotation['auxiliary_boxes'] = annotation['boxes'][1:]
        return image, annotation


def loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle, drop_last=shuffle,
                      num_workers=config.NUM_WORKERS, collate_fn=misc.collate_fn)
