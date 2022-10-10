from __future__ import annotations

import logging
from collections import defaultdict, deque
from itertools import product

from lm_service.onto import Candidate, MuIndex, TokenIndexT
from lm_service.piles import CandidatePile, ExtCandidateList
from lm_service.preprocessing import normalize_input_text, transform_advcl
from lm_service.relation import (
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
    ) = phrases_to_basis_triples(nlp, rules, phrases, plot_path)

    # mnemonics : ecl ~ ExtCandidateList()
    ecl = candidate_depot.unfold_conjunction()
    return striples, striples_meta, relations, ecl


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
    striples, striples_meta, relations, ecl = phrases_to_triples_stage_a(
        phrases, nlp, rules, plot_path
    )

    global_ecl = ExtCandidateList()

    window_size = min([window_size, len(phrases)])
    nmax = len(phrases) - window_size + 1
    for i in range(nmax):
        fragment = " ".join(phrases[i : i + window_size])
        ecl.set_filter(lambda x: i <= x[0] < i + window_size)
        ncp = text_to_coref_sourcetarget(
            nlp, fragment, ecl, initial_phrase_index=i
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

    for s, r, t in striples:
        for srunning, trunning in product(
            range(len(global_ecl[s])), range(len(global_ecl[t]))
        ):
            if isinstance(r, str):
                ip = 0
            elif isinstance(r, tuple):
                ip = r[0]
            else:
                raise TypeError("Unknown TokenIndexT subtype")
            fundamental_triples_aux[ip] += [
                (
                    MuIndex(False, *s, srunning),
                    MuIndex(False, *r, 0),
                    MuIndex(False, *t, trunning),
                )
            ]

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

        if cnt > deq_len_original:
            failing_deq = list(deq_striples_meta)
            failing_phrases = sorted(set([r[0] for _, r, _ in failing_deq]))
            logger.error(
                "Following meta-triples could not be resolved :"
                f" {failing_phrases}"
            )
            logger.error(f" Dangling metatriples : {failing_deq}")
            for ip in failing_phrases:  # type: ignore
                logger.error(f" failing phrase : <B>{phrases[ip]}<E>")
            # raise ValueError(f"Deq is stuck in a loop: {deq_striples_meta}")

        s, r, t = deq_striples_meta.pop()

        if s in relations.sroots:
            if MuIndex(False, *s, 0) in relation_triple_map:
                sources_mu = [relation_triple_map[MuIndex(False, *s, 0)]]
            else:
                deq_striples_meta.appendleft((s, r, t))
                continue
        else:
            sources_mu = [
                MuIndex(False, *s, j) for j in range(len(global_ecl[s]))
            ]
        if t in relations.sroots:
            if MuIndex(False, *t, 0) in relation_triple_map:
                targets_mu = [relation_triple_map[MuIndex(False, *t, 0)]]
            else:
                deq_striples_meta.appendleft((s, r, t))
                continue
        else:
            targets_mu = [
                MuIndex(False, *t, j) for j in range(len(global_ecl[t]))
            ]

        if isinstance(r, str):
            ip = 0
        elif isinstance(r, tuple):
            ip = r[0]
        else:
            raise TypeError("Unexpected TokenIndexT composition")
        current_phrase_index = ip
        k_tri_offset_meta = len(meta_triples_aux[current_phrase_index])

        for sprime, tprime in product(sources_mu, targets_mu):
            meta_triples_aux[current_phrase_index] += [
                (sprime, MuIndex(False, *r, 0), tprime)
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
        if basic_index in global_ecl:
            mu_index_candidate_map[candlike_like] = global_ecl[basic_index][
                candlike_like.running
            ]
        elif basic_index in relations.sroots:
            mu_index_candidate_map[candlike_like] = relations[basic_index]
        else:
            raise ValueError(
                "Fundamental MuIndex not in ExtCandPile and not in relations"
            )
    return global_triples, mu_index_candidate_map, ecl


def phrases_to_basis_triples(
    nlp, rules, phrases, plot_path=None
) -> tuple[
    list[tuple[TokenIndexT, TokenIndexT, TokenIndexT]],
    set[tuple[TokenIndexT, TokenIndexT, TokenIndexT]],
    CandidatePile,
    CandidatePile,
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
    for k, phrase in enumerate(phrases):
        (
            graph_relabeled,
            rdoc,
            map_tree_subtree_index,
        ) = text_to_compound_index_graph(nlp, phrase, initial_phrase_index=k)

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

        if plot_path is not None:
            plot_graph(graph_relabeled, plot_path, f"phrase_{k}_full")

    # mnemonics : prefix `s` stands for str or compound index
    return striples, striples_meta, candidate_depot, relations


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
