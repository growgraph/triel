from __future__ import annotations

import dataclasses
import logging
from collections import deque

import requests
from dataclass_wizard import JSONWizard

from lm_service.hash import hashme
from lm_service.linking import (
    Entity,
    EntityLinker,
    iterate_over_linkers,
    link_unlinked_entities,
)
from lm_service.onto import MuIndex, SimplifiedCandidate
from lm_service.text import normalize_text, phrases_to_triples

logger = logging.getLogger(__name__)


class UnknownCastTripleVersion(Exception):
    pass


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

    phrase_entities_foos: dict = {
        EntityLinker.BERN_V2: lambda p: query_bern(p, "v2")["annotations"],
        EntityLinker.SPACY_NAIVE_WIKI: lambda p: nlp(p)._.linkedEntities,
    }

    if (
        EntityLinker.SPACY_NAIVE_WIKI in phrase_entities_foos
        and "entityLinker" not in nlp.pipe_names
    ):
        nlp.add_pipe("entityLinker", last=True)

    map_eindex_entity, map_c2e = iterate_over_linkers(
        phrases=phrases,
        ecl=ecl,
        map_muindex_candidate=map_muindex_candidate,
        phrase_entities_foos=phrase_entities_foos,
    )

    if (
        EntityLinker.SPACY_NAIVE_WIKI in phrase_entities_foos
        and "entityLinker" in nlp.pipe_names
    ):
        nlp.remove_pipe("entityLinker")

    map_eindex_entity, map_c2e = link_unlinked_entities(
        map_eindex_entity, map_c2e, map_muindex_candidate
    )

    map_muindex_candidate_simplified = {
        k: v.to_simplified() for k, v in map_muindex_candidate.items()
    }

    return RELResponse(
        triples=global_triples,
        eindex_entity=map_eindex_entity,
        muindex_eindex=map_c2e,
        muindex_candidate=map_muindex_candidate_simplified,
    )


def cast_triple(
    item: tuple[
        SimplifiedCandidate,
        tuple[SimplifiedCandidate, SimplifiedCandidate, SimplifiedCandidate],
    ],
    cast_triple_version: str = "v0",
):
    mu, (s, r, t) = item
    if cast_triple_version == "v0":
        result = {
            "mu": mu,
            "source": s,
            "relation": r,
            "target": t,
        }
    elif cast_triple_version == "v1":
        result = {
            "triple_index": mu,
            "triple": tuple(  # type: ignore
                [
                    s.set_role("source"),
                    r.set_role("relation"),
                    t.set_role("target"),
                ]
            ),
        }
    else:
        raise UnknownCastTripleVersion("")
    return result


def cast_response_to_unfolded(response: RELResponse, **kwargs):
    muc = response.muindex_candidate
    map_eindex_entity = response.eindex_entity

    deq: deque[
        tuple[
            MuIndex,
            tuple[
                MuIndex | SimplifiedCandidate,
                MuIndex | SimplifiedCandidate,
                MuIndex | SimplifiedCandidate,
            ],
        ]
    ] = deque(
        response.triples.items()
    )  # type: ignore

    triples_upd = []

    # extend muc with meta candidates
    # create {{} , source:{}, target:{} {}}
    while deq:
        mu, tri = deq.popleft()
        tri_sub = [muc[t] if t in muc else t for t in tri]
        if all(t in muc.values() for t in tri_sub):
            if mu in muc:
                mu_sub = muc[mu]
            else:
                s = "".join(t.hash for t in tri_sub)
                mu_sub = SimplifiedCandidate(hash=hashme(s))
                muc[mu] = mu_sub
            triples_upd += [(mu_sub, tri_sub)]
        else:
            deq.append((mu, tuple(tri_sub)))  # type: ignore

    mu_ei = response.muindex_eindex

    mu_ei_grounded = []
    for mu, ei in mu_ei:
        if mu not in muc:
            logger.error(
                f"index {mu} should be in muc. muc keys(): {muc.keys()}"
            )
        if ei not in map_eindex_entity:
            logger.error(
                f"index {mu} should be in muc. muc keys():"
                f" {map_eindex_entity.keys()}"
            )
        mu_ei_grounded += [
            {
                "mention": muc[mu],
                "entity": map_eindex_entity[ei],
            }
        ]

    triples_upd = [cast_triple(x, **kwargs) for x in triples_upd]  # type: ignore
    r = to_dict({"triples": triples_upd, "map_mention_entity": mu_ei_grounded})
    return r
