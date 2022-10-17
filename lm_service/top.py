import logging

import requests

from lm_service.linking import (
    EntityLinker,
    iterate_over_linkers,
    link_unlinked_entities,
)
from lm_service.text import normalize_text, phrases_to_triples

logger = logging.getLogger(__name__)


# only v2 is supported by the API
api_spec = {
    "v1": {
        "url": "https://bern.korea.ac.kr/plain",
        "text_field": "sample_text",
    },
    "v2": {"url": "http://bern2.korea.ac.kr/plain", "text_field": "text"},
}


def query_bern(text, version="v2"):
    url = api_spec[version]["url"]
    text_field = api_spec[version]["text_field"]
    return requests.post(url, json={text_field: text}, verify=False).json()


# def link(sentences):
#     es = []
#     for s in sentences:
#         es.append(link_phrase(s))
#     return es
#
#
# def link_phrase(s):
#     entities = query_bern(s)
#     if not entities.get("denotations"):
#         return {
#             "text": entities["text"],
#             # "text_sha256": hashlib.sha256(
#             #     entities["text"].encode("utf-8")
#             # ).hexdigest(),
#         }
#     else:
#         e = []
#         for entity in entities["denotations"]:
#             other_ids = [
#                 eid for eid in entity["id"] if not eid.startswith("BERN")
#             ]
#             entity_type = entity["obj"]
#             entity_name = entities["text"][
#                 entity["span"]["begin"] : entity["span"]["end"]
#             ]
#             bern_id = [eid for eid in entity["id"] if eid.startswith("BERN")]
#             e.append(
#                 {
#                     "entity_id": bern_id[0] if bern_id else entity_name,
#                     "other_ids": other_ids,
#                     "entity_type": entity_type,
#                     "entity": entity_name,
#                 }
#             )
#         return {
#             "entities": e,
#             "text": entities["text"],
#             # "text_sha256": hashlib.sha256(
#             #     entities["text"].encode("utf-8")
#             # ).hexdigest(),
#         }


def text_to_rel_graph(text, nlp, rules):
    phrases = normalize_text(text, nlp)

    global_triples, map_muindex_candidate, ecl = phrases_to_triples(
        phrases, nlp, rules, window_size=2
    )

    if "entityLinker" not in nlp.pipe_names:
        nlp.add_pipe("entityLinker", last=True)

    phrase_entities_foos: dict = {
        EntityLinker.BERN_V2: lambda p: query_bern(p, "v2")["annotations"],
        EntityLinker.SPACY_NAIVE_WIKI: lambda p: nlp(p)._.linkedEntities,
    }

    map_eindex_entity, map_c2e = iterate_over_linkers(
        phrases=phrases,
        ecl=ecl,
        map_muindex_candidate=map_muindex_candidate,
        phrase_entities_foos=phrase_entities_foos,
    )

    map_eindex_entity, map_c2e = link_unlinked_entities(
        map_eindex_entity, map_c2e, map_muindex_candidate
    )

    # eindex_set = set(map_eindex_entity.keys()) & set(y for _, y in map_c2e)

    map_eindex_entity_str = {
        k: v.to_dict(skip_defaults=True) for k, v in map_eindex_entity.items()
    }

    return map_eindex_entity_str, map_c2e, map_muindex_candidate
