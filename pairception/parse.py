import json
from typing import NamedTuple

from tqdm import tqdm

from . import config
from .data import read_split

PROCESSORS = {'tokenize': 'ewt', 'mwt': 'ewt', 'pos': 'ewt_nocharlm', 'lemma': 'ewt_nocharlm',
              'depparse': 'ewt_nocharlm'}
BATCH_SIZE = 32


class Entity(NamedTuple):
    span: tuple
    head: tuple
    dependency: str


def _expand_entity(sentence, head):
    words = {word.id: word for word in sentence.words}
    ids = {head.id}
    allowed = {'det', 'amod', 'compound', 'nummod', 'flat', 'fixed'}
    while True:
        extra = {word.id for word in sentence.words if word.head in ids and word.deprel.split(':')[0] in allowed}
        if extra <= ids:
            break
        ids |= extra
    members = [words[index] for index in ids]
    return Entity(
        (min(word.start_char for word in members), max(word.end_char for word in members)),
        (head.start_char, head.end_char),
        head.deprel,
    )


def entities(document):
    result = []
    seen = set()
    for sentence in document.sentences:
        for word in sentence.words:
            if word.upos not in {'NOUN', 'PROPN'}:
                continue
            entity = _expand_entity(sentence, word)
            if (entity.span, entity.head) not in seen:
                result.append(entity)
                seen.add((entity.span, entity.head))
            contains_other_noun = any(
                other.id != word.id and other.upos in {'NOUN', 'PROPN'}
                and entity.span[0] <= other.start_char and other.end_char <= entity.span[1]
                for other in sentence.words
            )
            if contains_other_noun or word.deprel.split(':')[0] in {'compound', 'flat', 'fixed'}:
                nested = Entity((word.start_char, word.end_char), (word.start_char, word.end_char), word.deprel)
                if (nested.span, nested.head) not in seen:
                    result.append(nested)
                    seen.add((nested.span, nested.head))
    return sorted(result, key=lambda entity: (entity.span[0], entity.span[1], entity.head))


def propose(entities, max_targets=3, max_auxiliaries=8):
    if not entities:
        return []
    preferred = sorted(entities, key=lambda entity: (
        not (entity.dependency.startswith('nsubj') or entity.dependency == 'root'), entity.span[0], entity.span[1]))
    targets = list(dict.fromkeys([entities[0]] + preferred))[:max_targets]
    hypotheses = []
    for target in targets:
        auxiliaries = [entity.span for entity in entities if entity != target][:max_auxiliaries]
        if auxiliaries:
            hypotheses.append((target.span, auxiliaries))
    return hypotheses


def pipeline():
    import stanza
    from stanza.pipeline.core import DownloadMethod

    return stanza.Pipeline('en', dir=str(config.cache_dir() / 'stanza'), package=None, processors=PROCESSORS,
                           use_gpu=False, download_method=DownloadMethod.REUSE_RESOURCES, verbose=False)


def load():
    path = config.cache_dir() / 'parses.json'
    if not path.is_file():
        raise FileNotFoundError(f'{path} is missing; run python -m pairception.parse')
    return {caption: [(tuple(target), [tuple(auxiliary) for auxiliary in auxiliaries])
                      for target, auxiliaries in hypotheses]
            for caption, hypotheses in json.loads(path.read_text()).items()}


def main():
    captions = list(dict.fromkeys(
        record['grounding']['caption'] for split in ('train', 'val', 'test') for record in read_split(split)
        if isinstance(record['grounding'].get('caption'), str) and record['grounding']['caption'].strip()
    ))
    nlp = pipeline()
    parses = {}
    batches = [captions[start:start + BATCH_SIZE] for start in range(0, len(captions), BATCH_SIZE)]
    for batch in tqdm(batches, desc='parse', dynamic_ncols=True):
        for caption, document in zip(batch, nlp.bulk_process(batch)):
            parses[caption] = propose(entities(document))
    path = config.cache_dir() / 'parses.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parses))


if __name__ == '__main__':
    main()
