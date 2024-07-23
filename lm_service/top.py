from __future__ import annotations

import dataclasses
import logging
from collections import deque

import networkx as nx
from dataclass_wizard import JSONWizard
from suthing import profile

from lm_service.coref import stitch_coreference
from lm_service.hash import hashme
from lm_service.linking.onto import Entity
from lm_service.linking.util import (
    iterate_over_linkers,
    link_unlinked_entities,
    map_mentions_to_entities,
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
    muindex_candidate: dict[MuIndex, SimplifiedCandidate]


@profile
def text_to_rel_graph(text, nlp, rules, elm, **kwargs):
    phrases = normalize_text(text, nlp)

    global_triples, map_muindex_candidate = phrases_to_triples(
        phrases, nlp, rules, **kwargs
    )

    edges_chain_token_global, edges_chain_order_global = stitch_coreference(
        nlp=nlp, phrases_for_coref=phrases, window_size=2
    )

    entity_pack = iterate_over_linkers(
        phrases=phrases,
        entity_linker_manager=elm,
        **kwargs,
    )

    # chain2entities
    map_eindex_entity, map_c2e = map_mentions_to_entities(
        phrases, entity_pack, map_muindex_candidate
    )

    map_muindex_candidate_simplified = {
        k: v.to_simplified() for k, v in map_muindex_candidate.items()
    }

    phrase_candidates: dict[int, list[tuple]] = {}
    for mu, v in map_muindex_candidate.items():
        if mu.phrase in phrase_candidates:
            phrase_candidates[mu.phrase] += [(mu, v)]
        else:
            phrase_candidates[mu.phrase] = [(mu, v)]

    # chain -> entities

    graph_chain_tokens = nx.DiGraph()
    graph_chain_tokens.add_edges_from(edges_chain_token_global)

    chain_mu_edges = []

    for c in graph_chain_tokens.nodes:
        if graph_chain_tokens.in_degree(c) == 0:
            tokens_equivalence_class = list(graph_chain_tokens.successors(c))
            iphrases = set(x for token in tokens_equivalence_class for x, _ in token)
            for iphrase in iphrases:
                for mu, candidate in phrase_candidates[iphrase]:
                    if any(
                        set(candidate.stokens) & set(tokens)
                        for tokens in tokens_equivalence_class
                    ):
                        chain_mu_edges += [(c, mu)]

    graph_chain_mus = nx.DiGraph()
    graph_chain_mus.add_edges_from(chain_mu_edges)

    graph_c2e = nx.DiGraph()
    graph_c2e.add_edges_from(map_c2e)
    c2e_edges_extra = []

    for c in graph_chain_mus.nodes:
        if graph_chain_mus.in_degree(c) == 0:
            es = set(
                [
                    e
                    for mu in graph_chain_mus.successors(c)
                    if mu in graph_c2e.nodes
                    for e in graph_c2e.successors(mu)
                ]
            )
            for mu in graph_chain_mus.successors(c):
                c2e_edges_extra += [(mu, e) for e in es]

    graph_c2e.add_edges_from(c2e_edges_extra)
    map_c2e = list(graph_c2e.edges())

    map_eindex_entity_extra, map_c2e_extra = link_unlinked_entities(
        map_c2e, map_muindex_candidate
    )

    map_c2e += map_c2e_extra

    map_eindex_entity_total = {**map_eindex_entity, **map_eindex_entity_extra}

    return RELResponse(
        triples=global_triples,
        eindex_entity=map_eindex_entity_total,
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

    r = {
        "triples": triples_upd,
        "map_mention_entity": mu_ei_grounded,
        "top_level_mention": top_level_mention,
    }
    return r
