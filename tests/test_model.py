import random
import unittest

import torch
from PIL import Image

from pairception.data import RandomHorizontalFlip
from pairception.losses import losses
from pairception.relation_injection import RelationInjectionModule
from pairception.upstream import transforms

CAPTION = 'red car left of blue bus'
OFFSETS = [(0, 0), (0, 3), (4, 7), (8, 12), (13, 15), (16, 20), (21, 24), (0, 0)]
HYPOTHESES = [((0, 7), [(16, 24)])]
BOXES = torch.tensor([[[0.2, 0.5, 0.1, 0.1], [0.6, 0.5, 0.1, 0.1], [0.4, 0.2, 0.1, 0.1]]])


def inputs(module):
    torch.manual_seed(0)
    return dict(queries=torch.randn(1, 3, 8), boxes=BOXES, text_features=torch.randn(1, len(OFFSETS), 8),
                text_mask=torch.ones(1, len(OFFSETS), dtype=torch.bool), offsets=[OFFSETS], hypotheses=[HYPOTHESES],
                hflip=[False])


class RelationInjectionTest(unittest.TestCase):
    def test_starts_as_identity(self):
        module = RelationInjectionModule(dim=8, injection_dim=4)
        x = inputs(module)
        queries, records = module(**x)
        self.assertTrue(torch.equal(queries, x['queries']))
        self.assertEqual([(r['target_span'], r['auxiliary_span']) for r in records[0]], [((0, 7), (16, 24))])

    def test_update_differs_per_candidate(self):
        module = RelationInjectionModule(dim=8, injection_dim=4)
        torch.nn.init.normal_(module.injection_out.weight)
        x = inputs(module)
        delta = module(**x)[0] - x['queries']
        self.assertFalse(torch.allclose(delta[0, 0], delta[0, 1]))

    def test_losses_train_only_the_module(self):
        module = RelationInjectionModule(dim=8, injection_dim=4)
        torch.nn.init.normal_(module.injection_out.weight)
        head = torch.nn.Linear(8, 4).requires_grad_(False)
        x = inputs(module)
        queries, records = module(**x)
        outputs = {'boxes': (head(queries) + torch.logit(BOXES)).sigmoid(), 'scores': queries.sum(-1),
                   'base_boxes': BOXES, 'queries': queries, 'records': records}
        annotations = [{'target_box': BOXES[0, :1], 'auxiliary_boxes': BOXES[0, 1:2], 'caption': CAPTION,
                        'relations': ['left of'], 'spans': torch.tensor([[0, 7], [16, 24]])}]
        loss = losses(outputs, annotations)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(head.weight.grad)
        for name in ('pair_logit.2.weight', 'phrase_relevance.2.weight', 'relation_token_attention.weight',
                     'injection_out.weight'):
            gradient = dict(module.named_parameters())[name].grad
            self.assertTrue(gradient is not None and gradient.abs().sum() > 0, name)


class FlipTest(unittest.TestCase):
    def test_flip_matches_upstream_and_is_recorded(self):
        image = Image.new('RGB', (100, 50))
        annotation = {'boxes': torch.tensor([[10.0, 5.0, 30.0, 25.0]]), 'hflip': False}
        for seed in range(6):
            random.seed(seed)
            _, ours = RandomHorizontalFlip()(image, annotation)
            after_ours = random.random()
            random.seed(seed)
            _, upstream = transforms.RandomHorizontalFlip()(image, annotation)
            self.assertEqual(after_ours, random.random())
            self.assertTrue(torch.equal(ours['boxes'], upstream['boxes']))
            self.assertEqual(ours['hflip'], not torch.equal(ours['boxes'], annotation['boxes']))
