from __future__ import annotations

import logging
from collections import defaultdict, deque
from itertools import product

import networkx as nx
from suthing import profile

from lm_service.coref import stitch_coreference
from lm_service.hash import hashme
from lm_service.linking.onto import EntityLinker
from lm_service.linking.util import (
    iterate_over_linkers,
    link_unlinked_entities,
    map_mentions_to_entities,
)
from lm_service.onto import MuIndex, SimplifiedCandidate
from lm_service.response.onto import (
    Entity,
    REELResponse,
    REELResponseEntity,
    REELResponseRedux,
    Triple,
    TripleFormal,
)
from lm_service.text import normalize_text, phrases_to_triples

logger = logging.getLogger(__name__)


@profile
def text_to_graph_mentions_entities(text, nlp, rules, elm, **kwargs):
    phrases = normalize_text(text, nlp)

    triples_dict, map_muindex_candidate = phrases_to_triples(
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
    map_eindex_entity, map_c2e, ee_edges = map_mentions_to_entities(
        phrases, entity_pack, map_muindex_candidate
    )

    # candidate entity equivalences
    len(ee_edges)

    list_muindex_candidate_simplified = [
        (k, v.to_simplified()) for k, v in map_muindex_candidate.items()
    ]

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

    list_triples = [(k, v) for k, v in triples_dict.items()]

    return REELResponse(
        triples=list_triples,
        _muindex_candidate=list_muindex_candidate_simplified,
        eindex_entity=map_eindex_entity_total,
        muindex_eindex=map_c2e,
    )


def cast_response_redux(response: REELResponse) -> REELResponseRedux:
    muc = response.muindex_candidate
    map_eindex_entity = response.eindex_entity

    deq: deque[
        tuple[
            MuIndex,
            tuple[
                MuIndex,
                MuIndex,
                MuIndex,
            ],
        ]
    ] = deque(response.triples)

    triples_upd: list[
        tuple[
            SimplifiedCandidate,
            tuple[SimplifiedCandidate, SimplifiedCandidate, SimplifiedCandidate],
        ]
    ] = []

    # extend muc with meta candidates
    while deq:
        mu, tri = deq.popleft()
        if all(t in muc for t in tri):
            tri_sub = (
                muc[tri[0]].get_copy_with_role("source"),
                muc[tri[1]].get_copy_with_role("relation"),
                muc[tri[2]].get_copy_with_role("target"),
            )
            if mu in muc:
                mu_sub = muc[mu]
            else:
                s = "".join(t.hash for t in tri_sub)
                mu_sub = SimplifiedCandidate(hash=hashme(s))
                muc[mu] = mu_sub
            triples_upd += [(mu_sub, tri_sub)]
        else:
            deq.append((mu, tri))

    mu_ei = response.muindex_eindex

    mu_ei_grounded = []

    connected_mus = set(
        [x for _, t in response.triples for x in t] + [m for m, _ in response.triples]
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

    # triples_upd = [TripleExplicit(mu=x[0], source=x[1][0], relation=x[1][1], target=x[1][2]) for x in triples_upd]
    triples_upd_tri = [Triple(triple_index=x[0], triple=x[1]) for x in triples_upd]

    metamus = {t.triple_index.hash for t in triples_upd_tri}
    all_mus = {t.hash for item in triples_upd_tri for t in item.triple}
    top_level_metamus = metamus - all_mus

    top_level_mention = [{"hash": h} for h in top_level_metamus]

    return REELResponseRedux(
        triples=triples_upd_tri,
        map_mention_entity=mu_ei_grounded,
        top_level_mention=top_level_mention,
    )


def cast_response_entity_representation(response: REELResponse) -> REELResponseEntity:
    map_eindex_entity = response.eindex_entity

    map_muindex_eindexes: defaultdict[MuIndex, list[str]] = defaultdict(list)

    for mu_index, e_index in response.muindex_eindex:
        map_muindex_eindexes[mu_index] += [e_index]

    acc: list[TripleFormal] = []

    deq: deque[
        tuple[
            MuIndex,
            tuple[
                MuIndex,
                MuIndex,
                MuIndex,
            ],
        ]
    ] = deque(response.triples)
    predicate_compound_entity_map: defaultdict[MuIndex, list] = defaultdict(list)
    predicates = set(tri[1] for _, tri in response.triples)
    while deq:
        mu, tri = deq.popleft()
        s, p, o = tri
        if (s in predicates and s not in predicate_compound_entity_map) or (
            o in predicates and o not in predicate_compound_entity_map
        ):
            deq.append((mu, tri))
        else:
            if s in predicates:
                s_eindexes = predicate_compound_entity_map[s]
            else:
                s_eindexes = map_muindex_eindexes[s]

            if o in predicates:
                o_eindexes = predicate_compound_entity_map[o]
            else:
                o_eindexes = map_muindex_eindexes[o]

            for es, ep, eo in product(
                o_eindexes,
                map_muindex_eindexes[p],
                s_eindexes,
            ):
                acc += [TripleFormal(subject=es, predicate=ep, object=eo)]
                original_form = f"s:{es}, p:{ep}, o:{eo}"
                e = Entity(
                    id=hashme(original_form),
                    linker_type=EntityLinker.META,
                    original_form=original_form,
                    ent_db_type="_",
                )

                predicate_compound_entity_map[p] += [e.hash]
                map_eindex_entity[e.hash] = e

    return REELResponseEntity(triples=acc, entities=list(map_eindex_entity.values()))
