from __future__ import annotations

import logging
from collections import defaultdict, deque
from itertools import product

import networkx as nx

from lm_service.onto import Candidate, MuIndex, Relation, TokenIndexT
from lm_service.piles import CandidatePile, ExtCandidateList
from lm_service.preprocessing import normalize_input_text, transform_advcl
from lm_service.relation import (
    align_relation_to_target,
    form_triples,
    graph_to_candidate_pile,
    graph_to_maps,
    text_to_compound_index_graph,
    text_to_coref_sourcetarget,
)
from lm_service.util import plot_graph

logger = logging.getLogger(__name__)


def normalize_text(text, nlp, head=None) -> list[str]:
    phrases_original = normalize_input_text(text, terminal_full_stop=True)
    if head is not None:
        phrases_original = phrases_original[:head]
    phrases = [transform_advcl(nlp, p) for p in phrases_original]
    return phrases


def phrases_to_triples_stage_a(
    phrases,
    nlp,
    rules,
    plot_path=None,
):
    """
    stage a : basis triples
    :param phrases:
    :param nlp:
    :param rules:
    :param plot_path:
    :return:
    """
    (
        striples,
        striples_meta,
        candidate_depot,
        relations,
        megagraph,
    ) = phrases_to_basis_triples(nlp, rules, phrases, plot_path)

    # mnemonics : ecl ~ ExtCandidateList()
    ext_cand_list = candidate_depot.unfold_conjunction()
    for r in relations:
        r.normalize()
    return striples, striples_meta, relations, ext_cand_list, megagraph


def phrases_to_triples(
    phrases,
    nlp,
    rules,
    window_size,
    plot_path=None,
) -> tuple[
    dict[MuIndex, tuple[MuIndex, MuIndex, MuIndex]],
    dict[MuIndex, Candidate],
    ExtCandidateList,
]:
    (
        striples,
        striples_meta,
        relations,
        ext_cand_list,
        megagraph,
    ) = phrases_to_triples_stage_a(phrases, nlp, rules, plot_path)

    global_ecl = ExtCandidateList()

    window_size = min([window_size, len(phrases)])
    nmax = len(phrases) - window_size + 1
    for i in range(nmax):
        fragment = " ".join(phrases[i : i + window_size])
        ext_cand_list.set_filter(lambda x: i <= x[0] < i + window_size)
        ncp = text_to_coref_sourcetarget(
            nlp, fragment, ext_cand_list, initial_phrase_index=i
        )

        for key, candidate_list in ncp.items():
            for c in candidate_list:
                global_ecl.append(key, c)

    global_ecl.filter_out_pronouns()

    # mnemonics : fun ~ fundamental
    fun_candidates: dict[MuIndex, Candidate] = dict()
    for key, item_list in global_ecl:
        for j, item in enumerate(item_list):
            fun_candidates[MuIndex(False, *key, j)] = item

    # iphrase -> fundamental triple
    fundamental_triples_aux: defaultdict[
        int, list[tuple[MuIndex, MuIndex, MuIndex]]
    ] = defaultdict(list)

    # triple_index -> fundamental triple
    fundamental_triples: dict[
        MuIndex, tuple[MuIndex, MuIndex, MuIndex]
    ] = dict()

    # iphrase -> meta triple
    meta_triples_aux: defaultdict[
        int, list[tuple[MuIndex, MuIndex, MuIndex]]
    ] = defaultdict(list)

    # triple_index -> meta triple
    meta_triples: dict[MuIndex, tuple[MuIndex, MuIndex, MuIndex]] = dict()
    relation_ecd: defaultdict[TokenIndexT, dict[int, Relation]] = defaultdict(
        dict
    )

    for s, r, t in striples:
        current_relation = relations[r]
        if isinstance(r, str):
            iphrase = 0
        elif isinstance(r, tuple):
            iphrase = r[0]
        else:
            raise TypeError("Unknown TokenIndexT subtype")
        for srunning in range(len(global_ecl[s])):

            if current_relation.has_prepositions():
                relation_redux = [
                    align_relation_to_target(relations[r], tprime, megagraph)
                    for tprime in global_ecl[t]
                ]

                rt = list(
                    zip(
                        (r.approximate_hash_int() for r in relation_redux),
                        range(len(relation_redux)),
                    )
                )

                rt_map = dict(rt)

                for k, v in rt_map.items():
                    relation_ecd[r][k] = relation_redux[v]
            else:
                relation_index = relations[r].approximate_hash_int()
                rt = [
                    (relation_index, trunning)
                    for trunning in range(len(global_ecl[t]))
                ]
                relation_ecd[r][relation_index] = relations[r]

            fundamental_triples_aux[iphrase] += [
                (
                    MuIndex(False, *s, srunning),
                    MuIndex(False, *r, rrunning),
                    MuIndex(False, *t, trunning),
                )
                for rrunning, trunning in rt
            ]

    for r in relations.sroots:
        if r not in relation_ecd:
            relation_ecd[r][relations[r].approximate_hash_int()] = relations[r]

    for iphrase, list_item in fundamental_triples_aux.items():
        for k_tri, tri in enumerate(list_item):
            fundamental_triples[MuIndex(True, iphrase, "000", k_tri)] = tri

    relation_triple_map: dict[MuIndex, MuIndex] = {}
    for mu_tri, (_, r, _) in fundamental_triples.items():
        relation_triple_map[r] = mu_tri

    deq_striples_meta = deque(striples_meta)
    deq_len = len(deq_striples_meta)
    deq_len_original = deq_len
    cnt = 0

    while deq_striples_meta:
        if len(deq_striples_meta) < deq_len:
            cnt = 0
            deq_len_original = len(deq_striples_meta)
        else:
            cnt += 1
        deq_len = len(deq_striples_meta)

        if cnt > deq_len_original + 5:
            failing_deq = list(deq_striples_meta)
            failing_phrases = sorted(set([r[0] for _, r, _ in failing_deq]))
            logger.error(
                " the following meta-triples could not be resolved [phrase"
                f" numbers]: {failing_phrases}"
            )
            logger.error(f" Dangling metatriples : {failing_deq}")
            for iphrase in failing_phrases:  # type: ignore
                logger.error(f" failing phrase : <B>{phrases[iphrase]}<E>")
            break
            # raise ValueError(f"Deq is stuck in a loop: {deq_striples_meta}")

        s, r, t = deq_striples_meta.pop()

        if s in relations.sroots:
            r_current_index = relations[s].approximate_hash_int()
            if MuIndex(False, *s, r_current_index) in relation_triple_map:
                sources_mu = [
                    relation_triple_map[MuIndex(False, *s, r_current_index)]
                ]
            else:
                deq_striples_meta.appendleft((s, r, t))
                continue
        else:
            sources_mu = [
                MuIndex(False, *s, j) for j in range(len(global_ecl[s]))
            ]
        if t in relations.sroots:
            r_current_index = relations[t].approximate_hash_int()

            if MuIndex(False, *t, r_current_index) in relation_triple_map:
                targets_mu = [
                    relation_triple_map[MuIndex(False, *t, r_current_index)]
                ]
            else:
                deq_striples_meta.appendleft((s, r, t))
                continue
        else:
            targets_mu = [
                MuIndex(False, *t, j) for j in range(len(global_ecl[t]))
            ]

        if isinstance(r, str):
            iphrase = 0
        elif isinstance(r, tuple):
            iphrase = r[0]
        else:
            raise TypeError("Unexpected TokenIndexT composition")
        current_phrase_index = iphrase
        k_tri_offset_meta = len(meta_triples_aux[current_phrase_index])

        for sprime, tprime in product(sources_mu, targets_mu):
            meta_triples_aux[current_phrase_index] += [
                (
                    sprime,
                    MuIndex(False, *r, relations[r].approximate_hash_int()),
                    tprime,
                )
            ]

        for k_tri, tri in enumerate(meta_triples_aux[current_phrase_index]):
            # start count per phrase: # fund triples + # meta triples added  + current
            k_tri_offset = (
                len(fundamental_triples_aux[tri[1].phrase]) + k_tri_offset_meta
            )
            meta_tri_index = MuIndex(
                True, tri[1].phrase, "000", k_tri + k_tri_offset
            )
            meta_triples[meta_tri_index] = tri
            # update relation -> triple map
            relation_triple_map[tri[1]] = meta_tri_index
    global_triples = {**fundamental_triples, **meta_triples}

    candidate_likes = set(
        [
            candlike
            for item in global_triples.values()
            for candlike in item
            if not candlike.meta
        ]
    )

    # dict : MuIndex -> Candidate
    mu_index_candidate_map: dict[MuIndex, Candidate] = dict()
    for candlike_like in candidate_likes:
        basic_index = (candlike_like.phrase, candlike_like.token)
        if (
            basic_index in relation_ecd
            and candlike_like.running in relation_ecd[basic_index]
        ):
            mu_index_candidate_map[candlike_like] = relation_ecd[basic_index][
                candlike_like.running
            ]
        elif basic_index in global_ecl and candlike_like.running < len(
            global_ecl[basic_index]
        ):
            mu_index_candidate_map[candlike_like] = global_ecl[basic_index][
                candlike_like.running
            ]
        else:
            raise IndexError(
                f"Fundamental MuIndex {candlike_like} not in global_ecl and"
                " not in relation_ecd"
            )
    return global_triples, mu_index_candidate_map, ext_cand_list


def phrases_to_basis_triples(
    nlp, rules, phrases, plot_path=None
) -> tuple[
    list[tuple[TokenIndexT, TokenIndexT, TokenIndexT]],
    set[tuple[TokenIndexT, TokenIndexT, TokenIndexT]],
    CandidatePile,
    CandidatePile,
    nx.DiGraph,
]:
    """
    accumulate prospective triples
    :param nlp:
    :param rules:
    :param phrases:
    :param plot_path:
    :return:
    """

    striples = []
    striples_meta = set()
    candidate_depot = CandidatePile()
    relations = CandidatePile()
    megagraph = nx.DiGraph()

    for k, phrase in enumerate(phrases):
        (
            graph_relabeled,
            rdoc,
            map_tree_subtree_index,
        ) = text_to_compound_index_graph(nlp, phrase, initial_phrase_index=k)

        if plot_path is not None:
            plot_graph(graph_relabeled, plot_path, f"phrase_{k}_full")

        pile, candidate_depot0, mod_graph = graph_to_candidate_pile(
            graph_relabeled, rules
        )

        (
            sources_per_relation,
            targets_per_relation,
            rel_sources_per_relation,
            rel_targets_per_relation,
            g_undirected,
        ) = graph_to_maps(mod_graph=mod_graph, pile=pile)

        striples0, striples0_meta = form_triples(
            pile,
            sources_per_relation,
            targets_per_relation,
            rel_sources_per_relation,
            rel_targets_per_relation,
            g_undirected,
        )

        relations += pile.relations
        striples += striples0
        striples_meta |= striples0_meta
        candidate_depot += candidate_depot0

        megagraph.add_nodes_from(graph_relabeled.nodes(data=True))
        megagraph.add_edges_from(graph_relabeled.edges())

    # mnemonics : prefix `s` stands for str or compound index
    return striples, striples_meta, candidate_depot, relations, megagraph


def cast_simplified_triples_table(global_triples, map_mu_index_triple):
    global_triples_txt = {}
    for mu_key, tri in global_triples.items():

        def foo(mu: MuIndex):
            if mu.meta:
                return (
                    "(meta)"
                    + map_mu_index_triple[
                        global_triples[mu][1]
                    ].project_to_text_str()
                )
            else:
                return map_mu_index_triple[mu].project_to_text_str()

        tri_txt = tuple([foo(mu) for mu in tri])
        global_triples_txt[mu_key] = tri_txt
    return global_triples_txt
