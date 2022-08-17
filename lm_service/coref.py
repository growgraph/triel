from __future__ import annotations

import logging
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

import networkx as nx
from spacy.tokens import Doc

from lm_service.onto import Candidate
from lm_service.piles import partition_conjunctive_wrapper

logger = logging.getLogger(__name__)


def render_coref_graph(rdoc: Doc, graph: nx.DiGraph) -> nx.DiGraph:
    """
    render super graph using coreferee package

    :param rdoc:
    :param graph:
    :return: coref graph is a Tree of 4 levels:
        root -> chain -> blank -> token
        chain corresponds to one co-referenced entity, like [("they"), ("Mary", "Peter")]

            - there can be only one root,
            - 0 or many chains per root
            - 1 or many blanks per chain
            - 1 or many tokens per blank

        NB: most specific mention is encodedin in blank attribute `most_specific`
    """

    chains = rdoc._.coref_chains if rdoc._.coref_chains is not None else []
    # nodes for coref graph
    vs_coref = []

    # edges for coref graph
    es_coref = []

    # mention_nodes = []

    # map : chain_id -> most_specific_mention_id
    concept_specific_blank: dict[int, int] = dict()

    vertex_counter = max([token.i for token in rdoc]) + 1
    coref_root = vertex_counter

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
        coref_chain = vertex_counter
        chain_state: dict[str, Any] = {
            "tag_": "coref",
            "dep_": "chain",
            "chain": jchain,
        }
        chain_state[
            "label"
        ] = f"{coref_chain}-*-{chain_state['tag_']}-{chain_state['dep_']}"

        vs_coref += [(coref_chain, chain_state)]
        es_coref.append((coref_root, coref_chain))
        vertex_counter += 1
        for kth, x in enumerate(chain.mentions):
            coref_blank = vertex_counter
            blank_state: dict[str, Any] = {
                "tag_": "coref",
                "dep_": "blank"
                + ("*" if kth == chain.most_specific_mention_index else ""),
                "most_specific": kth == chain.most_specific_mention_index,
                "chain": jchain,
            }
            blank_state[
                "label"
            ] = f"{coref_blank}-*-{blank_state['tag_']}-{blank_state['dep_']}"

            vs_coref += [(coref_blank, blank_state)]
            es_coref.append((coref_chain, coref_blank))
            vertex_counter += 1
            for y in x.token_indexes:
                vs_coref += [(y, graph.nodes[y])]
                es_coref.append((coref_blank, y))

    chs = {
        j: [[graph.nodes[x]["lower"] for x in item] for item in k.mentions]
        for j, k in enumerate(chains)
    }
    logger.info(f"{chs}")
    chs = {
        j: [
            graph.nodes[x]["lower"]
            for x in k.mentions[k.most_specific_mention_index]
        ]
        for j, k in enumerate(chains)
    }
    logger.info(f"specifics {chs}")

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
    rdoc, graph
) -> tuple[defaultdict[int, list[int]], defaultdict[int, list[int]]]:

    coref_graph = render_coref_graph(rdoc, graph)
    (
        map_subbable_to_chain,
        map_chain_to_most_specific,
    ) = render_coref_candidate_map(coref_graph)
    return map_subbable_to_chain, map_chain_to_most_specific


def sub_coreference(
    map_subbable_to_chain: defaultdict[int, list[int]],
    map_chain_to_most_specific: defaultdict[int, list[int]],
    x,
) -> list[int]:
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
    dep_tree: nx.DiGraph,
    candidate_depot,
    map_subbable_to_chain,
    map_chain_to_most_specific,
    token_dict,
    unfold_conjunction=True,
) -> dict[int, list[Candidate]]:
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

    map_icoref_source_target = {}

    ncp: defaultdict[int, list] = defaultdict(list)
    # unfold conjunction
    for c in candidate_depot:
        if unfold_conjunction:
            ncp[c.root.i].extend(partition_conjunctive_wrapper(c, dep_tree))
        else:
            ncp[c.root.i] = [c]

    # itoken -> atomic candidate
    for iroot, sigmas in ncp.items():
        for sigma in sigmas:
            for k in all_coref_i:
                if k in sigma.itokens:
                    map_icoref_source_target[k] = iroot, deepcopy(sigma)
                elif k not in map_icoref_source_target:
                    ac = Candidate()
                    ac.append(token_dict[k])
                    map_icoref_source_target[k] = k, ac

    # map (iroot, coref_index) -> clean atomic candidate
    deq: deque = deque()
    for iroot, sigmas in ncp.items():
        for sigma in sigmas:
            deq.append((iroot, sigma))

    ncp2: defaultdict[int, list] = defaultdict(list)
    cnt = 0
    max_cnt = max([len(map_icoref_source_target) ** 2, len(deq) ** 2])
    while deq and cnt < max_cnt:
        cnt += 1
        iroot, sigma = deq.popleft()
        candidate_ix_subs = set(map_trunc) & set(sigma.itokens)
        if candidate_ix_subs:
            for sub in candidate_ix_subs:
                iy_subs = map_trunc[sub]
                for y in iy_subs:
                    s2 = deepcopy(sigma)
                    iroot_new, sigma_sub = map_icoref_source_target[y]
                    s2.replace_token_with_acandidate(sub, sigma_sub)
                    deq.append((iroot, s2))
        else:
            ncp2[iroot] += [sigma]
    return ncp2
