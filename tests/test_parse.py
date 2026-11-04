import unittest
from types import SimpleNamespace

from pairception.parse import entities, propose
from pairception.relation_injection import visible_phrase_pairs

CAPTION = 'A red car is parked left of a blue bus.'
WORDS = [(1, 'A', 'DET', 3, 'det'), (2, 'red', 'ADJ', 3, 'amod'), (3, 'car', 'NOUN', 5, 'nsubj:pass'),
         (4, 'is', 'AUX', 5, 'aux:pass'), (5, 'parked', 'VERB', 0, 'root'), (6, 'left', 'ADV', 5, 'advmod'),
         (7, 'of', 'ADP', 10, 'case'), (8, 'a', 'DET', 10, 'det'), (9, 'blue', 'ADJ', 10, 'amod'),
         (10, 'bus', 'NOUN', 6, 'obl'), (11, '.', 'PUNCT', 5, 'punct')]


def document():
    words, cursor = [], 0
    for index, text, upos, head, deprel in WORDS:
        start = CAPTION.index(text, cursor)
        cursor = start + len(text)
        words.append(SimpleNamespace(id=index, text=text, upos=upos, head=head, deprel=deprel,
                                     start_char=start, end_char=cursor))
    return SimpleNamespace(sentences=[SimpleNamespace(words=words)])


class ParseTest(unittest.TestCase):
    def test_each_noun_phrase_is_tried_as_the_target_phrase(self):
        car, bus = (0, 9), (28, 38)
        self.assertEqual([CAPTION[s:e] for s, e in (car, bus)], ['A red car', 'a blue bus'])
        self.assertEqual(propose(entities(document())), [(car, [bus]), (bus, [car])])

    def test_phrases_beyond_the_encoded_text_are_dropped(self):
        hypotheses = propose(entities(document()))
        self.assertEqual(visible_phrase_pairs(hypotheses, [(0, 1), (2, 5), (6, 9), (10, 12)], limit=12), [])
        self.assertEqual(visible_phrase_pairs(hypotheses, [(0, 1), (35, 38)], limit=1), [((0, 9), (28, 38))])
