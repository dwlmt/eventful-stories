import more_itertools
import spacy
from allennlp.common.util import lazy_groups_of
from allennlp.predictors import Predictor
from allennlp_models.pretrained import load_predictor
from spacy.tokens import Doc
from typing import List

MAX_SRL_STRINGS = 5

exclude_verbs = {"is", "was", "were", "are", "be", "´s", "´re", "´ll", "can", "could", "must", "may", "have to",
                 "has to", "had to", "will", "would", "has", "have", "had", "do", "does", "did"}


# This code is adapted from https://neurosys.com/article/how-to-make-an-effective-coreference-resolution-model/

def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
    spans = [doc[span[0]:span[1] + 1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
                         if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices


def replace_mention(document: Doc, coref: List[int], resolved: List[str], mention_text: str):
    final_token = document[coref[1]]

    resolved[coref[0]] = mention_text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved


def get_cluster_noun(doc: Doc, cluster: List[List[int]], noun_index: int):
    head_idx = noun_index
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end + 1]
    return head_span, [head_start, head_end]


def get_ner(doc: Doc, cluster: List[List[int]], noun_index: int):
    head_idx = noun_index
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end + 1]
    ents = head_span.ents
    if not ents:
        return None
    return ents[0].label_


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def replace_corefs(document, clusters, cluster_offset):
    coref_dict = {}
    coref_type_dict = {}
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

    for i, cluster in enumerate(clusters):

        noun_indices = get_span_noun_indices(document, cluster)

        entity_id = f"ent{cluster_offset}"
        ner = None

        if noun_indices:

            for n in noun_indices:

                if ner is None:
                    ner = get_ner(document, cluster, n)

                    if ner is not None:
                        cluster_offset += 1

                if ner is not None:

                    coref_type_dict[entity_id] = ner

                    mention_span, mention = get_cluster_noun(document, cluster, n)

                    if entity_id not in coref_dict:
                        coref_dict[entity_id] = {}

                    if mention_span.text not in coref_dict[entity_id]:
                        coref_dict[entity_id][mention_span.text] = 1
                    else:
                        coref_dict[entity_id][mention_span.text] += 1

            for coref in cluster:
                # coref_text = document[coref[0]:coref[1]+1].text
                # if coref != mention and
                if not is_containing_other_spans(coref, all_spans):
                    replace_mention(document, coref, resolved, entity_id)

    return resolved, coref_dict, coref_type_dict, cluster_offset


class CorefEventExtractor:

    def __init__(self, spacy_model_name="en_core_web_trf", coref_model_name: str = "coref-spanbert",
                 event_model_name: str = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
                 event_batch_size: int = 64,
                 num_sentences_batch: int = 64):

        self.nlp = spacy.load(spacy_model_name)
        self.coref = load_predictor(coref_model_name)
        self.coref._model = self.coref._model.cuda()
        self.event = Predictor.from_path(event_model_name, cuda_device=0)
        self.event_batch_size = event_batch_size
        # self.coref_max_tokens = coref_max_tokens
        self.num_sentences_batch = num_sentences_batch

    def __call__(self, text: str, **kwargs):

        # print(f"TEXT: {text}")

        results_dict = {}

        if text is not None and len(text) > 0:
            doc = self.nlp(text)
            # print(f"DOC {doc}")

            all_sentences = []
            for sent in doc.sents:
                # print(token.text)
                sentence = []
                for token in sent:
                    sentence.append(token.text)

                all_sentences.append(sentence)

            updated_doc_sentences = []
            coref_dict = {}
            coref_type_dict = {}
            cluster_offset = 0
            for curr_sentences in list(more_itertools.chunked(all_sentences, self.num_sentences_batch)):

                tokens = list(more_itertools.flatten(curr_sentences))
                curr_sentence_lengthes = [len(s) for s in curr_sentences]
                #print(f"TOKENS: {tokens}")
                clusters = self.coref.predict_tokenized(tokens).get("clusters")
                #print(f"Clusters: {clusters}")

                batch_doc_nlp = self.nlp.pipe([" ".join(tokens)])
                # batch_doc =  self.nlp(" ".join(tokens))

                for batch_doc in batch_doc_nlp:
                    #print(f"BATCH DOC: {batch_doc}")

                    updated_doc_batch, coref_dict_batch, coref_type_dict_batch, off = replace_corefs(
                        batch_doc, clusters, cluster_offset)

                    cluster_offset = off

                    updated_batch_sentences = []
                    offset = 0
                    for s_len in curr_sentence_lengthes:
                        sentence_tokens = updated_doc_batch[offset: offset + s_len]

                        updated_batch_sentences.append(" ".join(sentence_tokens))

                        offset += s_len

                    print(f"UPDATED BATCH SENTENCES: {updated_batch_sentences}")

                    #cluster_offset += len(clusters)
                    coref_dict = {**coref_dict, **coref_dict_batch}
                    coref_type_dict = {**coref_type_dict, **coref_type_dict_batch}

                    print(f"COREF: {coref_dict}")
                    print(f"COREF TYPEs: {coref_type_dict}")

                    updated_doc_sentences += updated_batch_sentences
                    # print(f"UPDATED DOC: {updated_doc_tokens}")
                    # print(f"COREFERENCES: {coref_dict}")

            coref_glosses_flat = []
            for k, v in coref_dict.items():
                for m in v.keys():
                    coref_glosses_flat.append({"id": k, "gloss": m, "count": v[m]})

            coref_type_flat = []
            for k, v in coref_type_dict.items():
                coref_type_flat.append({"id": k, "type": v})

            sentences = []

            # print(f"SPACY UPDATED DOC: {updated_doc}")
            for i, sentence in enumerate(doc.sents):
                sentences.append(sentence.text)

            coreference_mention_dict = {}
            for k in coref_dict.keys():
                coreference_mention_dict[k] = []

            sentences_coref = []

            for i, sentence in enumerate(updated_doc_sentences):
                # print(f"SENTENCE: {sentence}")
                sentence_text = sentence
                for k in coref_dict.keys():
                    if k in sentence_text:
                        coreference_mention_dict[k].append(i)
                    sentences_coref.append(sentence)

            coreference_mention_flat = []
            for k, v in coreference_mention_dict.items():
                for m in v:
                    coreference_mention_flat.append({"id": k, "seq_num": m})

            def make_srl_string(words: List[str], tags: List[str]) -> str:
                frame = []
                chunk = []

                for (token, tag) in zip(words, tags):
                    if tag.startswith("I-"):
                        chunk.append(token)
                    else:
                        if chunk:
                            # frame.append("[" + " ".join(chunk) + "]")
                            frame.append(" ".join(chunk))
                            chunk = []

                        if tag.startswith("B-"):
                            # chunk.append(tag[2:] + ": " + token)
                            chunk.append(f"<{tag[2:].replace('ARG', 'A')}>" + " " + token)
                        elif tag == "O":
                            pass  # frame.append(token)

                if chunk:
                    # frame.append("[" + " ".join(chunk) + "]")
                    frame.append(" ".join(chunk))

                return " ".join(frame)

            srl_flat = []
            sentences_coref_dict_list = []
            for s in sentences_coref:
                sentences_coref_dict_list.append({"sentence": s})

            i = 0
            for batch in lazy_groups_of(sentences_coref_dict_list, self.event_batch_size):
                all_srl_results = self.event.predict_batch_json(batch)

                for srl_result in all_srl_results:

                    count = 0
                    seen_verbs_set = set()
                    words = srl_result["words"]
                    for v in srl_result["verbs"]:

                        if count < MAX_SRL_STRINGS:

                            verb = v["verb"]
                            # print(v)

                            if verb not in seen_verbs_set and verb not in exclude_verbs:

                                srl_string = f"{make_srl_string(words, v['tags'])}"

                                if ("A0" in srl_string or "A1" in srl_string):
                                    srl_flat.append({"seq_num": i, "verb": verb, "text": srl_string})
                                    print(f"{srl_flat[-1]}")

                                    count += 1

                            seen_verbs_set.add(verb)

                    i += i

            # print(f"SRL: {srl_flat}")

            results_dict["sentences"] = sentences
            results_dict["sentences_corefs"] = sentences_coref
            results_dict["events"] = srl_flat
            results_dict["coreference_glosses"] = coref_glosses_flat
            results_dict["coreference_types"] = coref_type_flat
            results_dict["coreference_mentions"] = coreference_mention_flat

        print(f"RESULTS: {results_dict}")
        return results_dict
