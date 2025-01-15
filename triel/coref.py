from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

import networkx as nx
import spacy
from spacy.tokens import Doc

from triel.graph import (
    get_subtree_wrapper,
    phrase_to_deptree,
    relabel_nodes_and_key,
)
from triel.onto import AbsToken, Candidate, ChainIndex, Token, TokenIndexT
from triel.piles import ExtCandidateList
from triel.relation import logger
from triel.util import plot_graph


def graph_component_maps(
    graph: nx.DiGraph, initial_phrase_index=0
) -> dict[int, tuple[int, int]]:
    roots = [n for n, d in graph.in_degree() if d == 0]

    map_tree_subtree_index = {}
    sum_nodes = 0
    for sg, r in enumerate(sorted(roots)):
        acc = get_subtree_wrapper(graph, r)
        for i in acc:
            map_tree_subtree_index[i] = (
                sg + initial_phrase_index,
                i - sum_nodes,
            )
        sum_nodes += len(acc)

    return map_tree_subtree_index


def render_coref_graph(rdoc: Doc) -> nx.DiGraph:
    """
    render super graph using coreferee package

    :param rdoc:
    :return: coref graph is a Tree of 4 levels:
        root -> chain -> blank -> token
        chain corresponds to one co-referenced entity, like [("they"), ("Mary", "Peter")]

            - there can be only one root,
            - 0 or many chains per root
            - 1 or many blanks per chain
            - 1 or many tokens per blank

        NB: most specific mention is encoded by blank attribute `most_specific`
    """

    chains = rdoc._.coref_chains if rdoc._.coref_chains is not None else []
    # nodes for coref graph
    vs_coref = []

    # edges for coref graph
    es_coref = []

    vertex_counter = 0
    coref_root = (-1, vertex_counter)

    # add root
    vs_coref += [
        (
            coref_root,
            {
                "label": f"{coref_root}-*-coref-root",
                "tag_": "coref",
                "dep_": "root",
            },
        )
    ]
    vertex_counter += 1

    for jchain, chain in enumerate(chains):
        coref_chain = (-1, vertex_counter)
        chain_state: dict[str, Any] = {
            "tag_": "coref",
            "dep_": "chain",
            "chain": jchain,
        }
        chain_state["label"] = (
            f"{coref_chain}-*-{chain_state['tag_']}-{chain_state['dep_']}"
        )

        vs_coref += [(coref_chain, chain_state)]
        es_coref.append((coref_root, coref_chain))
        vertex_counter += 1
        for kth, x in enumerate(chain.mentions):
            coref_blank = (-1, vertex_counter)
            blank_state: dict[str, Any] = {
                "tag_": "coref",
                "dep_": "blank"
                + ("*" if kth == chain.most_specific_mention_index else ""),
                "most_specific": kth == chain.most_specific_mention_index,
                "chain": jchain,
            }
            blank_state["label"] = (
                f"{coref_blank}-*-{blank_state['tag_']}-{blank_state['dep_']}"
            )

            vs_coref += [(coref_blank, blank_state)]
            es_coref.append((coref_chain, coref_blank))
            vertex_counter += 1
            for y in x.token_indexes:
                vs_coref += [
                    (
                        y,
                        {"label": f"{y}-{rdoc[y].text}-{rdoc[y].tag_}-{rdoc[y].dep_}"},
                    )
                ]
                es_coref.append((coref_blank, y))

    coref_graph = nx.DiGraph()

    coref_graph.add_nodes_from(vs_coref)
    coref_graph.add_edges_from(es_coref)
    return coref_graph


def render_coref_candidate_map(
    coref_graph: nx.DiGraph,
) -> tuple[list[tuple[int, tuple[int, ...]]], list[tuple[int, int]]]:
    """
    from coref graph (root -> chain -> blank -> token)
        here tau is a tuple of tokens

    render two maps
        1. tau -> chain
        2. chain -> tau

    :param coref_graph:
    :return:
    """

    # only one root - guaranteed if coref_graph is produced by render_coref_graph
    root = next(n for n, d in coref_graph.in_degree if d == 0)

    # chain is an equivalence class of tokens' groups

    edges_chain_token: list[tuple[int, tuple[int, ...]]] = list()

    map_chain_int = {
        v_chain: jchain
        for jchain, v_chain in enumerate(sorted(coref_graph.successors(root)))
    }
    for v_chain in coref_graph.successors(root):
        for v_blank in sorted(coref_graph.successors(v_chain)):
            tau = tuple(sorted(coref_graph.successors(v_blank)))
            edges_chain_token += [(map_chain_int[v_chain], tau)]

    edges_chain_order: list[tuple[int, int]] = []
    from itertools import combinations

    for n in coref_graph.nodes():
        # pick bottom level nodes (tokens)
        if isinstance(n, int):
            taus = list(coref_graph.predecessors(n))
            if len(taus) > 1:
                for tau_a, tau_b in combinations(taus, 2):
                    succ_a = list(coref_graph.successors(tau_a))
                    succ_b = list(coref_graph.successors(tau_b))
                    chain_a = list(coref_graph.predecessors(tau_a))
                    chain_b = list(coref_graph.predecessors(tau_b))
                    if chain_a and chain_b:
                        chain_a_int = map_chain_int[chain_a[0]]
                        chain_b_int = map_chain_int[chain_b[0]]
                        if all(item in succ_b for item in succ_a):
                            # chain_a in chain_b
                            edges_chain_order += [(chain_a_int, chain_b_int)]
                        if all(item in succ_a for item in succ_b):
                            # chain_b in chain_a
                            edges_chain_order += [(chain_b_int, chain_a_int)]

    return edges_chain_token, edges_chain_order


def render_coref_maps_wrapper(
    rdoc, initial_phrase_index=None, map_tree_subtree_index=None, **kwargs
) -> tuple[list[tuple[int, tuple[int, ...]]], list[tuple[int, int]]]:
    coref_graph = render_coref_graph(rdoc)
    plot_path = kwargs.pop("plot_path", None)

    if plot_path is not None:
        fname = "coref_graph" + (
            "" if initial_phrase_index is None else f"_{initial_phrase_index}"
        )
        plot_graph(coref_graph, plot_path, fname)

    if map_tree_subtree_index is not None:
        coref_graph = nx.relabel_nodes(coref_graph, map_tree_subtree_index)
    (edges_chain_token, edges_chain_order) = render_coref_candidate_map(coref_graph)
    return edges_chain_token, edges_chain_order


def sub_coreference(
    map_subbable_to_chain: defaultdict[TokenIndexT, list[TokenIndexT]],
    map_chain_to_most_specific: defaultdict[TokenIndexT, list[TokenIndexT]],
    x,
) -> list[TokenIndexT]:
    """

        from two maps

            token -> [chain]
            chain -> [token]

        render a composition
            token -> [token]
                most specific

    :param map_subbable_to_chain:
    :param map_chain_to_most_specific:
    :param x:
    :return:
    """
    if x in map_subbable_to_chain:
        chains = map_subbable_to_chain[x]
        chains = [c for c in chains if c in map_chain_to_most_specific]
        if chains:
            chains = sorted(chains, key=lambda y: len(map_chain_to_most_specific[y]))
            r = map_chain_to_most_specific[chains[0]]
            if x in r:
                return [x]
            else:
                return [
                    z
                    for r0 in r
                    for z in sub_coreference(
                        map_subbable_to_chain, map_chain_to_most_specific, r0
                    )
                ]
        else:
            return list()
    else:
        return list()


def coref_candidates(
    ext_candidate_list: ExtCandidateList,
    map_subbable_to_chain: defaultdict[TokenIndexT, list[TokenIndexT]],
    map_chain_to_most_specific: defaultdict[TokenIndexT, list[TokenIndexT]],
    token_dict: dict[TokenIndexT, Token],
) -> defaultdict[TokenIndexT, list[Candidate]]:
    map_token_specific_token = {
        i: sub_coreference(map_subbable_to_chain, map_chain_to_most_specific, i)
        for i in map_subbable_to_chain
    }

    map_trunc = {k: v for k, v in map_token_specific_token.items() if [k] != v}

    all_coref_i = set(map_trunc.keys()) | {
        i for subl in map_trunc.values() for i in subl
    }

    map_icoref_source_target: dict[TokenIndexT, tuple[TokenIndexT, Candidate]] = {}

    # stoken -> atomic candidate
    for sroot, candidates in ext_candidate_list:
        for sigma_candidate in candidates:
            for k in all_coref_i:
                if k in sigma_candidate.stokens:
                    map_icoref_source_target[k] = sroot, deepcopy(sigma_candidate)
                elif (
                    k not in map_icoref_source_target
                ):  # case when coref pointer is not a source/target candidate
                    ac = Candidate().from_tokens([token_dict[k]])
                    map_icoref_source_target[k] = k, ac

    # map (iroot, coref_index) -> clean atomic candidate
    deq: deque[tuple[TokenIndexT, Candidate]] = deque()
    for sroot, candidates in ext_candidate_list:
        for sigma_candidate in candidates:
            deq.append((sroot, sigma_candidate))

    # ecl stands for extCandidateList
    ecl_like: defaultdict[TokenIndexT, list[Candidate]] = defaultdict(list)
    cnt = 0
    max_cnt = max([len(map_icoref_source_target) ** 2, len(deq) ** 2])
    while deq and cnt < max_cnt:
        cnt += 1
        sroot, sigma_candidate = deq.popleft()
        # indices to substitute using co-reference
        candidate_ix_subs = set(map_trunc) & set(sigma_candidate.stokens)

        # suppose we have 1->2; 2->3 substitution, first do 2->3 sub
        map_trunc_local_uniq: dict[TokenIndexT, list[TokenIndexT]] = {
            k: v for k, v in map_trunc.items() if k in candidate_ix_subs
        }
        domain = [x for sublist in map_trunc_local_uniq.values() for x in sublist]
        map_trunc_local_uniq = {
            k: v for k, v in map_trunc_local_uniq.items() if k not in domain
        }

        # perform all independent subs
        if map_trunc_local_uniq:
            sorted_wrt_number_coref = sorted(
                map_trunc_local_uniq.items(), key=lambda item: len(item[1])
            )
            sub = sorted_wrt_number_coref[0][0]
            iy_subs = map_trunc[sub]
            # coreference might contain references to several tokens
            for y in iy_subs:
                (
                    iroot_new,
                    sigma_candidate_substitution,
                ) = map_icoref_source_target[y]
                sigma_copy = deepcopy(sigma_candidate)
                # do not substitute if sigma already contains parts of proposed sub
                if not (
                    set(sigma_copy.stokens) & set(sigma_candidate_substitution.stokens)
                ):
                    # replace sub with sigma_candidate_substitution view
                    # the view is a subtree starting from token y and onwards
                    sub_tree_cand = sigma_candidate_substitution.from_subtree(y)
                    sigma_copy.replace_token_with_acandidate(sub, sub_tree_cand)
                deq.append((sroot, sigma_copy))
        else:
            try:
                sigma_new = sigma_candidate.normalize().sort_index()
            except:
                logger.warning(
                    "sigma_candidate.normalize().sort_index() failed for for"
                    f" {sigma_candidate}"
                )
                sigma_new = sigma_candidate
            ecl_like[sroot] += [sigma_new]
    return ecl_like


def stitch_coreference(
    phrases_for_coref: list[str], nlp: spacy.Language, window_size: int, plot_path=None
):
    """
    go over phrases with stride 1 and window `window` to identify coreferences;
    then identify chains (clusters) corresponding to the same concept

    :param phrases_for_coref:
    :param nlp:
    :param window_size:
    :param plot_path:
    :return:
    """
    window_size = min([window_size, len(phrases_for_coref)])
    nmax = len(phrases_for_coref) - window_size + 1
    acc_chain_token = []
    acc_chain_order = []
    for i in range(nmax):
        fragment = " ".join(phrases_for_coref[i : i + window_size])
        edges_chain_token, edges_chain_order = text_to_coref_classes(
            nlp, fragment, initial_phrase_index=i, plot_path=plot_path
        )
        acc_chain_token += edges_chain_token
        acc_chain_order += edges_chain_order

    g = nx.DiGraph()
    g.add_edges_from(acc_chain_token)

    merged_chains = set()
    for n in g.nodes():
        if g.in_degree(n) > 1:
            merger = tuple(sorted(g.predecessors(n), reverse=True))
            merged_chains.add(merger)

    merger_map = {u: v for u, v in merged_chains}

    # merge equivalent chains / clusters
    edges_chain_token_global = set(
        [(merger_map[u] if u in merger_map else u, v) for u, v in acc_chain_token]
    )

    edges_chain_order_global = list(
        {
            (
                merger_map[u] if u in merger_map else u,
                merger_map[v] if v in merger_map else v,
            )
            for u, v in acc_chain_order
        }
    )
    return edges_chain_token_global, edges_chain_order_global


def text_to_coref_classes(
    nlp, text, initial_phrase_index, **kwargs
) -> tuple[
    list[tuple[ChainIndex, tuple[TokenIndexT, ...]]],
    list[tuple[ChainIndex, ChainIndex]],
]:
    """
    :param nlp:
    :param text:
    :param initial_phrase_index:

    :return:
    """
    (
        graph_relabeled,
        rdoc,
        map_tree_subtree_index,
    ) = text_to_compound_index_graph(nlp, text, initial_phrase_index)

    # coref maps

    (edges_chain_token, edges_chain_order) = render_coref_maps_wrapper(
        rdoc, initial_phrase_index, **kwargs
    )
    edges_chain_tokenit: list[tuple[ChainIndex, tuple[TokenIndexT, ...]]] = [
        (
            (initial_phrase_index, f"c_{k}"),
            tuple([map_tree_subtree_index[vv] for vv in v]),
        )
        for k, v in edges_chain_token
    ]

    edges_chaint_order: list[tuple[ChainIndex, ChainIndex]] = [
        ((initial_phrase_index, f"c_{a}"), (initial_phrase_index, f"c_{b}"))
        for a, b in edges_chain_order
    ]
    return edges_chain_tokenit, edges_chaint_order


def text_to_compound_index_graph(
    nlp, text, initial_phrase_index, single_phrase_mode=False
):
    rdoc, graph = phrase_to_deptree(nlp, text)

    if single_phrase_mode and nx.number_weakly_connected_components(graph) > 1:
        components = sorted(nx.weakly_connected_components(graph), key=lambda x: len(x))
        sg = nx.subgraph(graph, components[-1])
        logger.warning(
            f" with single_phrase_mode from text <fail>{text}<fail> only"
            f" largest component [representing {sg.size()}/{graph.size()}] was"
            " kept."
        )
        graph = sg

    # cast index to compound index
    map_tree_subtree_index = graph_component_maps(graph, initial_phrase_index)

    map_tree_subtree_index = {
        k: AbsToken.ituple2stuple(v) for k, v in map_tree_subtree_index.items()
    }
    graph_relabeled = relabel_nodes_and_key(graph, map_tree_subtree_index, "s")
    return graph_relabeled, rdoc, map_tree_subtree_index
