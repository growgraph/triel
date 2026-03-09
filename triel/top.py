from __future__ import annotations

import logging
from collections import defaultdict, deque
from itertools import product

import networkx as nx
from suthing import Timer, profile

from triel.coref import stitch_coreference
from triel.coref_adapter import CorefAdapterError
from triel.hash import hashme
from triel.linking.onto import EntityLinker
from triel.linking.util import (
    iterate_over_linkers,
    link_unlinked_entities,
    map_mentions_to_entities,
)
from triel.onto import MuIndex, SimplifiedCandidate
from triel.response.onto import (
    Entity,
    REELResponse,
    REELResponseEntity,
    REELResponseRedux,
    Triple,
    TripleFormal,
)
from triel.text import normalize_text, phrases_to_triples

logger = logging.getLogger(__name__)


def _coref_chain_signatures(
    edges_chain_token_global: set[tuple[MuIndex, tuple[MuIndex, ...]]] | set[tuple],
) -> set[frozenset[tuple]]:
    chain_map: dict[MuIndex, set[tuple]] = defaultdict(set)
    for chain_id, mention_tokens in edges_chain_token_global:
        chain_map[chain_id].add(tuple(mention_tokens))
    return {frozenset(mentions) for mentions in chain_map.values()}


def _log_dual_run_coref_diff(
    *,
    text_hash: str,
    primary_backend: str,
    shadow_backend: str,
    primary_edges_chain_token,
    shadow_edges_chain_token,
    primary_edges_chain_order,
    shadow_edges_chain_order,
) -> None:
    primary_signatures = _coref_chain_signatures(set(primary_edges_chain_token))
    shadow_signatures = _coref_chain_signatures(set(shadow_edges_chain_token))
    added_chains = len(shadow_signatures - primary_signatures)
    removed_chains = len(primary_signatures - shadow_signatures)
    edge_drift = abs(len(shadow_edges_chain_token) - len(primary_edges_chain_token))
    order_edge_drift = abs(
        len(shadow_edges_chain_order) - len(primary_edges_chain_order)
    )
    logger.info(
        "coref_dual_run text_hash=%s primary=%s shadow=%s "
        "added_chains=%s removed_chains=%s edge_drift=%s order_edge_drift=%s",
        text_hash,
        primary_backend,
        shadow_backend,
        added_chains,
        removed_chains,
        edge_drift,
        order_edge_drift,
    )


@profile
def text_to_graph_mentions_entities(text, nlp, rules, elm, ix_phrases=None, **kwargs):
    coref_resolver = kwargs.pop("coref_resolver", None)
    coref_shadow_resolver = kwargs.pop("coref_shadow_resolver", None)
    nlp_shadow = kwargs.pop("nlp_shadow", None)
    coref_dual_run_enabled = kwargs.pop("coref_dual_run_enabled", False)
    phrases = normalize_text(text, nlp)
    text_hash = hashme(text)

    if ix_phrases is not None:
        if ix_phrases:
            phrases = [phrases[ix] for ix in ix_phrases]

    triples_dict, map_muindex_candidate = phrases_to_triples(
        phrases, nlp, rules, **kwargs
    )
    with Timer() as t_coref:
        try:
            edges_chain_token_global, edges_chain_order_global = stitch_coreference(
                nlp=nlp,
                phrases_for_coref=phrases,
                window_size=2,
                coref_resolver=coref_resolver,
            )
        except CorefAdapterError as e:
            backend = (
                coref_resolver.backend.value
                if coref_resolver is not None
                else "unknown"
            )
            logger.warning(
                "coref_primary_failed text_hash=%s backend=%s error=%s",
                text_hash,
                backend,
                e,
            )
            edges_chain_token_global, edges_chain_order_global = set(), []
        except Exception as e:
            backend = (
                coref_resolver.backend.value
                if coref_resolver is not None
                else "unknown"
            )
            logger.warning(
                "coref_primary_unexpected_error text_hash=%s backend=%s error=%s",
                text_hash,
                backend,
                e,
            )
            edges_chain_token_global, edges_chain_order_global = set(), []

    if coref_dual_run_enabled and coref_shadow_resolver is not None:
        with Timer() as t_coref_shadow:
            try:
                shadow_nlp = nlp_shadow if nlp_shadow is not None else nlp
                shadow_edges_chain_token, shadow_edges_chain_order = stitch_coreference(
                    nlp=shadow_nlp,
                    phrases_for_coref=phrases,
                    window_size=2,
                    coref_resolver=coref_shadow_resolver,
                )
                _log_dual_run_coref_diff(
                    text_hash=text_hash,
                    primary_backend=(
                        coref_resolver.backend.value
                        if coref_resolver is not None
                        else "unknown"
                    ),
                    shadow_backend=coref_shadow_resolver.backend.value,
                    primary_edges_chain_token=edges_chain_token_global,
                    shadow_edges_chain_token=shadow_edges_chain_token,
                    primary_edges_chain_order=edges_chain_order_global,
                    shadow_edges_chain_order=shadow_edges_chain_order,
                )
                logger.info(
                    "coref_dual_run_timing text_hash=%s shadow_backend=%s elapsed=%.4f",
                    text_hash,
                    coref_shadow_resolver.backend.value,
                    t_coref_shadow.elapsed,
                )
            except CorefAdapterError as e:
                logger.warning(
                    "coref_shadow_failed text_hash=%s backend=%s error=%s",
                    text_hash,
                    coref_shadow_resolver.backend.value,
                    e,
                )
            except Exception as e:
                logger.warning(
                    "coref_shadow_unexpected_error text_hash=%s backend=%s error=%s",
                    text_hash,
                    coref_shadow_resolver.backend.value,
                    e,
                )

    with Timer() as t_el:
        entity_pack = iterate_over_linkers(
            phrases=phrases,
            entity_linker_manager=elm,
            **kwargs,
        )

    with Timer() as t_rest:
        # chain2entities
        map_eindex_entity, map_c2e, ee_edges = map_mentions_to_entities(
            phrases, entity_pack, map_muindex_candidate
        )

        # TODO to use in the future releases
        # candidate entity equivalences :
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
                iphrases = set(
                    x for token in tokens_equivalence_class for x, _ in token
                )
                for iphrase in iphrases:
                    # some candidates do not pass the filters (do not form triples)
                    # but are references to while co-referencing
                    if iphrase in phrase_candidates:
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

    logger.info(
        f"coref: {t_coref.elapsed:.2f}s; el: {t_el.elapsed:.2f}s; "
        f"rest: {t_rest.elapsed:.2f}s; for text '{text[:20]}...'"
    )
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

    triples_set: set[TripleFormal] = set()

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
    predicate_compound_entity_map: defaultdict[MuIndex, set] = defaultdict(set)
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
                s_eindexes = list(predicate_compound_entity_map[s])
            else:
                s_eindexes = map_muindex_eindexes[s]

            if o in predicates:
                o_eindexes = list(predicate_compound_entity_map[o])
            else:
                o_eindexes = map_muindex_eindexes[o]

            for es, ep, eo in product(
                s_eindexes,
                map_muindex_eindexes[p],
                o_eindexes,
            ):
                triples_set.add(TripleFormal(subject=es, predicate=ep, object=eo))
                original_form = f"s:{es}, p:{ep}, o:{eo}"
                e = Entity(
                    id=hashme(original_form),
                    linker_type=EntityLinker.META,
                    original_form=original_form,
                    ent_db_type="_",
                )

                predicate_compound_entity_map[p].add(e.hash)
                map_eindex_entity[e.hash] = e

    set_present_entities: list[str] = sorted(
        set([t.subject for t in triples_set])
        | set([t.object for t in triples_set])
        | set([t.predicate for t in triples_set])
    )
    entities: list[Entity] = [map_eindex_entity[k] for k in set_present_entities]

    # strip linker_type

    for e in entities:
        e.linker_type = EntityLinker.NA
    return REELResponseEntity(triples=sorted(triples_set), entities=entities)
