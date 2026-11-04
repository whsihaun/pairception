import math

import torch
import torch.nn.functional as F

from . import config
from .upstream import box_ops


def _span_overlap(first, second):
    first_start, first_end = int(first[0]), int(first[1])
    second_start, second_end = int(second[0]), int(second[1])
    if min(first_start, second_start) < 0:
        return 0.0
    intersection = max(0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return intersection / max(union, 1)


def _positive_set_ce(logits, positives, valid):
    numerator = torch.logsumexp(logits.masked_fill(~positives, float('-inf')).flatten(), 0)
    denominator = torch.logsumexp(logits.masked_fill(~valid, float('-inf')).flatten(), 0)
    return denominator - numerator


def _ious(boxes, target_box):
    iou, _ = box_ops.box_iou(
        box_ops.box_cxcywh_to_xyxy(boxes), box_ops.box_cxcywh_to_xyxy(target_box.reshape(1, 4))
    )
    return iou[:, 0]


def target_losses(outputs, annotations):
    boxes = outputs['boxes']
    log_probs = F.log_softmax(outputs['scores'], dim=1)
    target = torch.stack([annotation['target_box'].reshape(4) for annotation in annotations])
    expanded_target = target[:, None, :].expand_as(boxes)
    flat_boxes = box_ops.box_cxcywh_to_xyxy(boxes.reshape(-1, 4))
    flat_target = box_ops.box_cxcywh_to_xyxy(expanded_target.reshape(-1, 4))
    giou = box_ops.generalized_box_iou_pairwise(flat_boxes, flat_target).reshape(boxes.shape[:2])
    l1 = (boxes - expanded_target).abs().sum(-1)
    match = (5.0 * l1 + 2.0 * (1.0 - giou)).detach().argmin(dim=1)

    terms = []
    for batch_index in range(boxes.shape[0]):
        positive = _ious(boxes[batch_index], target[batch_index]) >= config.IOU_THRESHOLD
        positive[match[batch_index]] = True
        terms.append(-torch.logsumexp(log_probs[batch_index, positive], dim=0))
    rows = torch.arange(boxes.shape[0], device=boxes.device)
    return {
        'loss_cls': torch.stack(terms).mean(),
        'loss_l1': l1[rows, match].mean(),
        'loss_giou': (1.0 - giou[rows, match]).mean(),
    }


def _aligned_auxiliary_indices(record, annotation, overlap_threshold=0.5):
    region_spans = annotation['spans']
    if _span_overlap(record['target_span'], region_spans[0]) < overlap_threshold:
        return []
    return [index for index, span in enumerate(region_spans[1:])
            if _span_overlap(record['auxiliary_span'], span) >= overlap_threshold]


def _literal_relation_spans(expression, phrase):
    needle = phrase.strip().casefold()
    if not needle:
        return []
    haystack = expression.casefold()
    found = []
    cursor = 0
    while (start := haystack.find(needle, cursor)) >= 0:
        end = start + len(needle)
        left_ok = start == 0 or not (haystack[start - 1].isalnum() or haystack[start - 1] == '_')
        right_ok = end == len(haystack) or not (haystack[end].isalnum() or haystack[end] == '_')
        if left_ok and right_ok:
            found.append((start, end))
        cursor = start + 1
    return found


def relation_token_loss(outputs, annotations):
    terms = []
    for records, annotation in zip(outputs['records'], annotations):
        expression = annotation['caption']
        for record in records:
            for auxiliary_index in _aligned_auxiliary_indices(record, annotation):
                spans = _literal_relation_spans(expression, annotation['relations'][auxiliary_index])
                logits = record['relation_token_logits']
                eligible = record['relation_token_mask'].bool()
                offsets = record['relation_token_offsets']
                log_probs = F.log_softmax(logits.masked_fill(~eligible, float('-inf')), dim=0)
                candidate_losses = []
                for span_start, span_end in spans:
                    token_indices = [index for index, (start, end) in enumerate(offsets)
                                     if start < span_end and span_start < end]
                    if token_indices:
                        candidates = torch.as_tensor(token_indices, device=logits.device)
                        visible = candidates[eligible[candidates]]
                        if visible.numel():
                            candidate_losses.append(-log_probs[visible].mean() - math.log(len(visible)))
                if not candidate_losses:
                    continue
                choices = torch.stack(candidate_losses)
                terms.append(math.log(len(candidate_losses)) - torch.logsumexp(-choices, dim=0))
    anchor = outputs['queries'].sum() * 0.0
    return torch.stack(terms).mean() if terms else anchor


def relation_losses(outputs, annotations):
    base_boxes = outputs['base_boxes']
    pair_terms, relevance_logits, relevance_labels = [], [], []
    for batch_index, (records, annotation) in enumerate(zip(outputs['records'], annotations)):
        target_hit = _ious(base_boxes[batch_index], annotation['target_box']) >= config.IOU_THRESHOLD
        count = base_boxes.shape[1]
        valid = ~torch.eye(count, dtype=torch.bool, device=base_boxes.device)
        for record in records:
            aligned = _aligned_auxiliary_indices(record, annotation)
            relevance_logits.append(record['relevance_logit'])
            relevance_labels.append(float(bool(aligned)))
            for auxiliary_index in aligned:
                auxiliary_hit = (_ious(base_boxes[batch_index], annotation['auxiliary_boxes'][auxiliary_index])
                                 >= config.IOU_THRESHOLD)
                positive = target_hit[:, None] & auxiliary_hit[None, :] & valid
                if bool(positive.any()):
                    pair_terms.append(_positive_set_ce(record['pair_logits'], positive, valid))
    anchor = outputs['queries'].sum() * 0.0
    pair_loss = torch.stack(pair_terms).mean() if pair_terms else anchor
    if relevance_logits:
        logits = torch.stack(relevance_logits)
        phrase_loss = F.binary_cross_entropy_with_logits(logits, logits.new_tensor(relevance_labels))
    else:
        phrase_loss = anchor
    return {'loss_pair': pair_loss, 'loss_phrase': phrase_loss,
            'loss_rel_token': relation_token_loss(outputs, annotations)}


def losses(outputs, annotations):
    terms = {**target_losses(outputs, annotations), **relation_losses(outputs, annotations)}
    return sum(terms[name] * config.LOSS_WEIGHTS[name] for name in terms)
