from __future__ import annotations

import logging
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

import networkx as nx
from spacy.tokens import Doc

from lm_service.graph import get_subtree_wrapper
from lm_service.onto import (
    Candidate,
    Token,
    TokenIndexT,
    partition_conjunctive_wrapper,
)
from lm_service.piles import CandidatePile, ExtCandidateList

logger = logging.getLogger(__name__)


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
                "dep_": "blank" + (
                    "*" if kth == chain.most_specific_mention_index else ""
                ),
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
                        {
                            "label": f"{y}-{rdoc[y].text}-{rdoc[y].tag_}-{rdoc[y].dep_}"
                        },
                    )
                ]
                es_coref.append((coref_blank, y))

    coref_graph = nx.DiGraph()

    coref_graph.add_nodes_from(vs_coref)
    coref_graph.add_edges_from(es_coref)
    return coref_graph


def render_coref_candidate_map(
    coref_graph: nx.DiGraph,
) -> tuple[defaultdict[int, list[int]], defaultdict[int, list[int]]]:
    """
    from coref graph (root -> chain -> blank -> token)
    render two maps
        token -> [chain]
        chain -> [token] # most specific tokens for the chain
    :param coref_graph:
    :return:
    """

    # only one root - guaranteed if coref_graph is produced by render_coref_graph
    root = next(n for n, d in coref_graph.in_degree if d == 0)

    map_chain_to_most_specific = defaultdict(list)
    map_subbable_to_chain = defaultdict(list)

    for v_chain in coref_graph.successors(root):
        for v_blank in coref_graph.successors(v_chain):
            blank_props = coref_graph.nodes[v_blank]
            for v in coref_graph.successors(v_blank):
                map_subbable_to_chain[v].append(v_chain)
                if blank_props["most_specific"]:
                    map_chain_to_most_specific[v_chain].append(v)

    return map_subbable_to_chain, map_chain_to_most_specific


def render_coref_maps_wrapper(
    rdoc, map_tree_subtree_index=None
) -> tuple[defaultdict[int, list[int]], defaultdict[int, list[int]]]:
    coref_graph = render_coref_graph(rdoc)
    if map_tree_subtree_index is not None:
        coref_graph = nx.relabel_nodes(coref_graph, map_tree_subtree_index)
    (
        map_subbable_to_chain,
        map_chain_to_most_specific,
    ) = render_coref_candidate_map(coref_graph)
    return map_subbable_to_chain, map_chain_to_most_specific


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
            chains = sorted(
                chains, key=lambda y: len(map_chain_to_most_specific[y])
            )
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
        i: sub_coreference(
            map_subbable_to_chain, map_chain_to_most_specific, i
        )
        for i in map_subbable_to_chain
    }

    map_trunc = {k: v for k, v in map_token_specific_token.items() if [k] != v}

    all_coref_i = set(map_trunc.keys()) | {
        i for subl in map_trunc.values() for i in subl
    }

    map_icoref_source_target: dict[
        TokenIndexT, tuple[TokenIndexT, Candidate]
    ] = {}

    # stoken -> atomic candidate
    for sroot, candidates in ext_candidate_list:
        for sigma_candidate in candidates:
            for k in all_coref_i:
                if k in sigma_candidate.stokens:
                    map_icoref_source_target[k] = sroot, deepcopy(
                        sigma_candidate
                    )
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
        domain = [
            x for sublist in map_trunc_local_uniq.values() for x in sublist
        ]
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
                    set(sigma_copy.stokens)
                    & set(sigma_candidate_substitution.stokens)
                ):
                    # replace sub with sigma_candidate_substitution view
                    # the view is a subtree starting from token y and onwards
                    sub_tree_cand = sigma_candidate_substitution.from_subtree(
                        y
                    )
                    sigma_copy.replace_token_with_acandidate(
                        sub, sub_tree_cand
                    )
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
