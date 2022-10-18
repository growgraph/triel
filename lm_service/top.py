import dataclasses
import logging

import requests
from dataclass_wizard import JSONWizard

from lm_service.linking import (
    Entity,
    EntityLinker,
    iterate_over_linkers,
    link_unlinked_entities,
)
from lm_service.onto import MuIndex, SimplifiedCandidate
from lm_service.text import normalize_text, phrases_to_triples

logger = logging.getLogger(__name__)


@dataclasses.dataclass(repr=False, frozen=True, eq=True)
class RELResponse(JSONWizard):
    """
    represents a token in dep tree
    """

    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    triples: dict[MuIndex, tuple[MuIndex, MuIndex, MuIndex]]
    eindex_entity: dict[str, Entity]
    muindex_eindex: list[tuple[MuIndex, str]]
    muindex_candidate: dict[str, SimplifiedCandidate]


# only v2 is supported by the API
api_spec = {
    "v1": {
        "url": "https://bern.korea.ac.kr/plain",
        "text_field": "sample_text",
    },
    "v2": {"url": "http://bern2.korea.ac.kr/plain", "text_field": "text"},
}


def to_dict(obj):
    if isinstance(obj, dict):
        return {to_dict(k): to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    elif isinstance(obj, MuIndex):
        return obj.to_str()
    elif dataclasses.is_dataclass(obj):
        return obj.to_dict()
    else:
        return obj


def query_bern(text, version="v2"):
    url = api_spec[version]["url"]
    text_field = api_spec[version]["text_field"]
    return requests.post(url, json={text_field: text}, verify=False).json()


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

    map_muindex_candidate_simplified = {
        k: v.to_simplified() for k, v in map_muindex_candidate.items()
    }

    return {
        "triples": global_triples,
        "eindex_entity": map_eindex_entity,
        "muindex_eindex": map_c2e,
        "muindex_candidate": map_muindex_candidate_simplified,
    }
    # TODO
    # return RELResponse(
    #     triples=global_triples,
    #     eindex_entity=map_eindex_entity,
    #     muindex_eindex=map_c2e,
    #     muindex_candidate=map_muindex_candidate_simplified,
    # )
