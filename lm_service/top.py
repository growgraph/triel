from __future__ import annotations

import dataclasses
import logging
from collections import deque

from dataclass_wizard import JSONWizard
from graph_cast.util.timer import Timer
from suthing import Report, Return, secureit, timeit

from lm_service.hash import hashme
from lm_service.linking import (
    Entity,
    EntityLinkerFailed,
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


def text_to_rel_graph(text, nlp, rules, elm):
    phrases = normalize_text(text, nlp)

    deco_phrases_to_triples = timeit(secureit(phrases_to_triples))
    ret_triple: Return = deco_phrases_to_triples(
        phrases, nlp, rules, window_size=2
    )

    global_triples, map_muindex_candidate, ecl = ret_triple.ret

    ret_linking = iterate_over_linkers(
        phrases=phrases,
        ecl=ecl,
        map_muindex_candidate=map_muindex_candidate,
        entity_linker_manager=elm,
    )

    map_eindex_entity, map_c2e = ret_linking.ret

    map_eindex_entity, map_c2e = link_unlinked_entities(
        map_eindex_entity, map_c2e, map_muindex_candidate
    )

    map_muindex_candidate_simplified = {
        k: v.to_simplified() for k, v in map_muindex_candidate.items()
    }

    rel = RELResponse(
        triples=global_triples,
        eindex_entity=map_eindex_entity,
        muindex_eindex=map_c2e,
        muindex_candidate=map_muindex_candidate_simplified,
    )
    r0 = Return(ret=rel)
    r0.update([ret_linking, ret_triple])
    return r0


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
                    s.get_copy_with_role("source"),
                    r.get_copy_with_role("relation"),
                    t.get_copy_with_role("target"),
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

    connected_mus = set(
        [x for t in response.triples.values() for x in t]
        + list(response.triples.keys())
    )

    for mu, ei in mu_ei:
        if mu in connected_mus:
            try:
                mu_ei_grounded += [
                    {
                        "mention": muc[mu],
                        "entity": map_eindex_entity[ei],
                    }
                ]
            except Exception as e:
                logger.error(
                    f"Exception in top.cast_response_to_unfolded : {e}"
                )
                logger.error(f" mu = {mu}, ei = {ei}")

    triples_upd = [cast_triple(x, **kwargs) for x in triples_upd]  # type: ignore
    r = to_dict({"triples": triples_upd, "map_mention_entity": mu_ei_grounded})
    return r
