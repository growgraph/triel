from __future__ import annotations

import dataclasses
import logging
from collections import deque

from dataclass_wizard import JSONWizard
from suthing import profile

from lm_service.hash import hashme
from lm_service.linking.onto import Entity
from lm_service.linking.util import iterate_over_linkers
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


@profile
def text_to_rel_graph(text, nlp, rules, elm, **kwargs):
    phrases = normalize_text(text, nlp)

    global_triples, map_muindex_candidate, ecl = phrases_to_triples(
        phrases, nlp, rules, window_size=2, **kwargs
    )

    map_eindex_entity, map_c2e = iterate_over_linkers(
        phrases=phrases,
        ecl=ecl,
        map_muindex_candidate=map_muindex_candidate,
        entity_linker_manager=elm,
        **kwargs,
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
) -> dict:
    mu, (s, r, t) = item
    result: dict
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
    ] = deque(response.triples.items())  # type: ignore

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
                logger.error(f"Exception in top.cast_response_to_unfolded : {e}")
                logger.error(f" mu = {mu}, ei = {ei}")

    triples_upd: list[dict] = [cast_triple(x, **kwargs) for x in triples_upd]  # type: ignore

    metamus = {t["triple_index"].hash for t in triples_upd}  # type: ignore
    all_mus = {t.hash for item in triples_upd for t in item["triple"]}  # type: ignore
    top_level_metamus = metamus - all_mus

    top_level_mention = [{"hash": h} for h in top_level_metamus]

    r = to_dict(
        {
            "triples": triples_upd,
            "map_mention_entity": mu_ei_grounded,
            "top_level_mention": top_level_mention,
        }
    )
    return r
