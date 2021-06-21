import copy
import more_itertools
import random
import re
from blingfire import text_to_sentences
from collections import deque

from eventful_stories.data.contraction_utils import CONTRACTIONS_LIST
from eventful_stories.data.coref_events import CorefEventExtractor

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


class CorefEventExamples:

    def __init__(self):
        self.event_extractor = CorefEventExtractor()

    def process_examples(self, reader,
                         dummy: bool = False,
                         dummy_max_examples: int = 10000,
                         contractions: bool = False):

        def cleanup_text(text):
            if contractions:
                for e, r in CONTRACTIONS_LIST:
                    text = text.replace(e, r)

            text = text.replace("\t", " ")
            text = text.replace("\n", " ")

            if text.startswith('"'):
                text = text[1:]
            if text.endswith('"'):
                text = text[:-1]

            text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()

            return text

        for i, episode in enumerate(reader):

            if not dummy or i < dummy_max_examples:

                if 'id' in episode:
                    id = f"{episode['id']}"
                else:
                    id = f"{i}"

                text = episode['text']

                if contractions:
                    text = cleanup_text(text)

                processed_events = self.event_extractor(text)

                example = {
                    "id": f"{id}-{i}",
                    "title": f"{episode['title']}",
                    # "text": text,
                    "sentences": processed_events["sentences"],
                    "sentences_corefs": processed_events["sentences_corefs"],
                    "events": processed_events["events"],
                    "coreference_glosses": processed_events["coreference_glosses"],
                    "coreference_mentions": processed_events["coreference_mentions"],
                    "coreference_types": processed_events["coreference_types"]
                }

                print(example)
                yield example
