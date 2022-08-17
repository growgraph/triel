from __future__ import annotations

import hashlib
import logging
from collections import deque
from copy import deepcopy
from itertools import product
from typing import List, Tuple

import networkx as nx
import pandas as pd

from lm_service.coref import render_coref_candidate_map, render_coref_graph
from lm_service.folding import get_flag

# import pygraphviz as pgv
from lm_service.graph import excise_node, phrase_to_deptree
from lm_service.onto import (
    ACandidateKind,
    CandidatePile,
    CandidateType,
    Relation,
    Source,
    SourceOrTarget,
    SRTPile,
    Target,
    Token,
    TripleCandidate,
    partition_conjunctive_wrapper,
)

logger = logging.getLogger(__name__)

"""
    relation extraction module based on spacy.
    For each phrase:
    1. cast the dependency tree as a metagraph (unify noun chunk) 
    2. identify relation candidates in the metagraph based of tags
    3. for each relation candidate identify a pool of candidate sources and targets
    4. choose closest / best sources/ target candidates for each relation to form (relation, source, target) triples    

"""


class SimplifiedConjunctionHypothesisFailure(Exception):
    pass


def find_candidates_bfs(
    graph: nx.DiGraph,
    q: deque[int],
    candidate_pile: CandidatePile,
    how: ACandidateKind,
    **kwargs,
):
    """

    :param graph: please provide a copy graph, find_candidates_bfs potentially modifies graph structure
    :param q: deque to keep track of vertices while traversing
    :param candidate_pile: pile to accumulate candidates
    :param how:
    :return:
    """
    # VB, VBP, VBZ, VBD, VBG, VBN
    # auxiliary verb uses of be, have, do, and get: AUX
    # the infinitive to is PART

    foo_map = {
        ACandidateKind.RELATION: find_relation_subtree_dfs,
        ACandidateKind.SOURCE_TARGET: find_sourcetarget_subtree_dfs,
    }

    foo_map_class = {
        ACandidateKind.RELATION: Relation,
        ACandidateKind.SOURCE_TARGET: SourceOrTarget,
    }

    foo = foo_map[how]
    if not q:
        return

    current_vertex = q.popleft()
    current_relation = foo_map_class[how]()  # type: ignore

    successors = set(graph.successors(current_vertex))
    if current_vertex not in candidate_pile.tokens:
        foo(graph, deque([(current_vertex, 0)]), current_relation, **kwargs)  # type: ignore

    if not current_relation.empty:  # type: ignore
        current_relation.sort_index()  # type: ignore
        candidate_pile.append(current_relation)  # type: ignore

    q.extend(successors & set(graph.nodes))
    find_candidates_bfs(graph, q, candidate_pile, how, **kwargs)


def find_relation_subtree_dfs(
    graph: nx.DiGraph,
    q: deque[tuple[int, int]],
    current_relation: CandidateType,
):
    """

    :param graph:
    :param q: (!) the initial call should have only a single vertex in q
    :param current_relation:
    :return:
    """

    if not q:
        return
    current_vertex, level = q.pop()
    # print(current_vertex, end="->")
    if (
        current_relation.empty
        and level > 0
        or (
            not current_relation.empty
            and level - current_relation.max_level() > 1
        )
    ):
        return

    vtoken = Token(_level=level, **graph.nodes[current_vertex])

    len_current_relation = len(current_relation)

    if level == 0 and current_relation.empty:
        if vtoken.tag_.startswith("VB") and vtoken.dep_ != "amod":
            current_relation.append(vtoken)
    elif level > 0 and not current_relation.empty:
        # auxpass or aux
        if (vtoken.tag_.startswith("VB") or (vtoken.tag_ == "MD")) and (
            "aux" in vtoken.dep_
        ):
            current_relation.append(vtoken)
        # elif vtoken.tag_.startswith("VB") and vtoken.dep_ == "advcl":
        #     current_relation.append(vtoken)
        elif (vtoken.tag_ == "IN" and vtoken.dep_ == "prep") or (
            vtoken.tag_ == "IN" and vtoken.dep_ == "agent"
        ):
            current_relation.append(vtoken)

    successors = list(graph.successors(current_vertex))

    if len(current_relation) > len_current_relation:
        if level > 0:
            excise_node(graph, current_vertex)
        for v in successors:
            q.append((v, level + 1))
            find_relation_subtree_dfs(graph, q, current_relation)


def find_sourcetarget_subtree_dfs(
    graph: nx.DiGraph,
    q: deque[tuple[int, int]],
    source_target_cand: CandidateType,
    rules=None,
):
    """

    :param graph:
    :param q: (!) the initial call should have only a single vertex in q
    :param source_target_cand: source or target
    :param rules:
    :return:
    """

    if not q:
        return
    # current_vertex, level = q.popleft()
    current_vertex, level = q.pop()
    # print(current_vertex, end="->")

    if (
        source_target_cand.empty
        and level > 0
        or (
            not source_target_cand.empty
            and (
                (level - source_target_cand.max_level() > 1)
                or (source_target_cand.root.dep_ == "ccomp")
            )
        )
    ):
        return

    vtoken = Token(_level=level, **graph.nodes[current_vertex])

    len_current_relation = len(source_target_cand)

    if level == 0 and source_target_cand.empty:
        if (("NN" in vtoken.tag_) or (vtoken.tag_ == "PRP")) or (
            vtoken.dep_ == "ccomp"
        ):
            source_target_cand.append(vtoken)
    elif level > 0 and not source_target_cand.empty:
        vflag = get_flag(vtoken.__dict__, rules)
        if vflag:
            source_target_cand.append(vtoken)

    # q.extend((v, level + 1) for v in graph.successors(current_vertex))

    successors = list(graph.successors(current_vertex))

    if len(source_target_cand) > len_current_relation:
        if level > 0:
            excise_node(graph, current_vertex)
        for v in successors:
            q.append((v, level + 1))
            find_sourcetarget_subtree_dfs(graph, q, source_target_cand, rules)

    # if len(current_st) > len_current_relation:
    #     excise_node(graph, current_vertex)
    #     find_st_subtree_dfs(graph, q, current_st, rules)


def maybe_source(n) -> bool:
    return (("NN" in n["tag_"]) or (n["tag_"] == "PRP")) and (
        n["dep_"] != "pobj"
    )


def maybe_target(n) -> bool:
    return (
        ("NN" in n["tag_"]) or (n["dep_"] == "pobj") or (n["dep_"] == "ccomp")
    )


def check_condition(graph, s, foo_condition) -> bool:
    logger.debug(f" {s} : {id(graph)} : {graph.nodes[s]}")
    flag = [foo_condition(graph.nodes[s])]
    if "leaf" in graph.nodes[s]:
        leaf = graph.nodes[s]["leaf"]
        flag += [foo_condition(prop) for n, prop in leaf.nodes if n != s]
    return any(flag)


def graph_to_candidate_pile(
    graph: nx.DiGraph, rules
) -> tuple[SRTPile, CandidatePile, nx.DiGraph]:
    """

    :param graph:
    :param rules:
    :return: tuple[SRTPile, nx.DiGraph]
        a tuple of SourceRelationTarget Pile and a modified graph
            (that does not contain vertices that became parts of Candidates,
            so only one node (root note) per candidate is preserved)
    """
    roots = [n for n, d in graph.in_degree() if d == 0]
    relation_pile = CandidatePile()

    mgraph = deepcopy(graph)

    find_candidates_bfs(
        mgraph,
        deque(roots),
        relation_pile,
        ACandidateKind.RELATION,
    )

    source_target_pile = CandidatePile()
    find_candidates_bfs(
        mgraph,
        deque(roots),
        source_target_pile,
        ACandidateKind.SOURCE_TARGET,
        rules=rules,
    )

    source_candidates, target_candidates = sieve_sources_targets(
        source_target_pile
    )

    logger.info(f" relations: {relation_pile}")

    return (
        SRTPile(
            relations=relation_pile,
            sources=source_candidates,
            targets=target_candidates,
        ),
        source_target_pile,
        mgraph,
    )


def generate_extra_graphs(
    graph: nx.DiGraph,
) -> tuple[nx.Graph, nx.DiGraph, nx.Graph]:
    g_undirected = graph.to_undirected()
    g_reversed = graph.reverse()

    nx.set_edge_attributes(g_reversed, values=-1, name="weight")

    g_weighted = graph.copy()

    nx.set_edge_attributes(g_weighted, values=1, name="weight")

    # distance wrt to weights defined in this way will be 0 for the same level
    # +1 or -1 for level k to k+1 and vice versa
    g_weighted.add_weighted_edges_from(
        [
            (u, v, g_reversed.edges[u, v]["weight"])
            for u, v in g_reversed.edges
        ],
        weight="weight",
    )

    return g_undirected, g_reversed, g_weighted


def compute_distances(
    graph: nx.DiGraph,
    g_undirected: nx.Graph,
    g_weighted: nx.Graph,
    pile: CandidatePile,
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:

    # (u, v) : distance
    distance_directed: dict[tuple[int, int], int] = {
        (c.root.i, v): ll
        for c in pile
        for v, ll in nx.shortest_path_length(graph, c.root.i).items()
    }

    # (u, v) : distance
    distance_undirected: dict[tuple[int, int], int] = {
        (c.root.i, v): ll
        for c in pile
        for v, ll in nx.shortest_path_length(g_undirected, c.root.i).items()
    }

    # compute distances
    # gextra | dag paths
    paths = {c.root.i: nx.shortest_path(g_weighted, c.root.i) for c in pile}

    # g_original | (root, vertex) : weight
    distance_levels: dict[tuple[int, int], int] = {
        (r, v): nx.path_weight(g_weighted, pp, "weight")
        for r, batch in paths.items()
        for v, pp in batch.items()
    }

    return distance_undirected, distance_directed, distance_levels


def derive_sources_per_relaton(
    relation_candidate_roots: list[int],
    source_candidate_roots: list[int],
    distance_undirected: dict[tuple[int, int], int],
    distance_levels: dict[tuple[int, int], int],
    pile_sources_roots: list[Token],
) -> dict[int, set[int]]:
    # find sources per relation; sources may be up the tree, using undirected graph
    # for each relation find source candidates
    # a. close to relation on the tree
    # b. negative cost preferred (close in reverse direction),
    # c. add penalty if dep is attr for given node

    # distance_undirected, distance_directed, distance_levels

    # find targets per relation; targets are down the tree
    udist_to_sources: list[tuple] = [
        (r, s, distance_undirected[r, s])
        for r in relation_candidate_roots
        for s in source_candidate_roots
        if (r, s) in distance_undirected
    ]

    ldist_to_sources: list[tuple] = [
        (r, s, distance_levels[r, s])
        for r in relation_candidate_roots
        for s in source_candidate_roots
        if (r, s) in distance_levels
    ]

    udm = pd.DataFrame(udist_to_sources, columns=["r", "s", "ud"]).sort_values(
        ["r", "s"]
    )
    ldm = pd.DataFrame(ldist_to_sources, columns=["r", "s", "ld"]).sort_values(
        ["r", "s"]
    )
    decision = pd.merge(udm, ldm, on=["r", "s"], how="outer")

    # for each source add penalty if its dep_ is "attr" or "dobj"
    targetlike_penalty = pd.DataFrame(
        [(s.i, int(s.dep_ in ["attr", "dobj"])) for s in pile_sources_roots],
        columns=["s", "syn_penalty"],
    )
    decision = decision.merge(targetlike_penalty, how="left", on="s")

    decision["mcost"] = (
        decision["ud"] + decision["ld"] + decision["syn_penalty"]
    )

    # source decisions are based on mcost

    s = decision.set_index(["r", "s"])["mcost"]

    # mask minimum values of mcost per relation
    mask = s.eq(s.groupby(level=0).transform("min"))

    s = s[mask].reset_index(level=1)["s"]
    sources_per_relation = s.groupby(level=0).apply(set).to_dict()
    return sources_per_relation


def derive_targets_per_relaton(
    relation_candidate_roots: list[int],
    target_candidate_roots: list[int],
    distance_directed: dict[tuple[int, int], int],
) -> dict[int, set[int]]:

    # find targets per relation; targets are down the tree
    dist_to_targets: dict[int, dict[int, int]] = {
        r: {
            t: distance_directed[r, t]
            for t in target_candidate_roots
            if (r, t) in distance_directed
        }
        for r in relation_candidate_roots
    }

    min_dists: dict[int, int] = {
        r: min(item.values()) if item else -1
        for r, item in dist_to_targets.items()
    }

    targets_per_relation: dict[int, set[int]] = {
        r: {t for t, d in item.items() if d == min_dists[r]} if item else set()
        for r, item in dist_to_targets.items()
    }
    return targets_per_relation


def graph_to_triples(
    graph, rules
) -> tuple[list[TripleCandidate], CandidatePile]:
    """
    find triplets in a dep graph:
        a. find relation candidates
        b. find source candidates
        c. find target candidates

    :param graph: nx.Digraph
    :param rules:

    :return:
    """
    (
        pile,
        sources_per_relation,
        targets_per_relation,
        source_target_depot,
        g_undirected,
    ) = graph_to_maps(graph, rules)

    triples = form_triples(
        pile, sources_per_relation, targets_per_relation, g_undirected
    )

    # unfold conjunction
    triples_unfolded = []
    for tri in triples:
        sources = partition_conjunctive_wrapper(tri.source, graph)
        targets = partition_conjunctive_wrapper(tri.target, graph)
        triples_unfolded += [
            TripleCandidate(source=s, target=t, relation=tri.relation)
            for s, t in product(sources.candidates, targets.candidates)
        ]

    source_target_depot_unfolded = CandidatePile()
    for c in source_target_depot:
        c_unfolded = partition_conjunctive_wrapper(c, graph)
        for c2 in c_unfolded:
            source_target_depot_unfolded.append(c2)
    return triples_unfolded, source_target_depot_unfolded


def graph_to_maps(
    graph, rules
) -> tuple[
    SRTPile, dict[int, set[int]], dict[int, set[int]], CandidatePile, nx.Graph
]:
    """
    derive maps
        a. relation -> source
        b. relation -> target

    :param graph: nx.Digraph
    :param rules:

    :return:
    """

    pile, candidate_depot, mod_raph = graph_to_candidate_pile(graph, rules)

    # create relevant graphs for distance calculations : undirected, reversed ...
    g_undirected, g_reversed, g_weighted = generate_extra_graphs(mod_raph)

    (
        distance_undirected,
        distance_directed,
        distance_levels,
    ) = compute_distances(
        graph=mod_raph,
        g_undirected=g_undirected,
        g_weighted=g_weighted,
        pile=pile.relations,
    )
    # distance_undirected, distance_directed

    target_candidate_roots = pile.targets.iroots
    source_candidate_roots = pile.sources.iroots
    relation_candidate_roots = pile.relations.iroots

    targets_per_relation = derive_targets_per_relaton(
        relation_candidate_roots, target_candidate_roots, distance_directed
    )

    sources_per_relation = derive_sources_per_relaton(
        relation_candidate_roots,
        source_candidate_roots,
        distance_undirected,
        distance_levels,
        pile.sources.roots,
    )
    return (
        pile,
        sources_per_relation,
        targets_per_relation,
        candidate_depot,
        g_undirected,
    )


def form_triples(
    pile: SRTPile,
    sources_per_relation: dict[int, set[int]],
    targets_per_relation: dict[int, set[int]],
    g_undirected: nx.Graph,
    relaxed=False,
) -> list[TripleCandidate]:
    triples = []

    for iroot in pile.relations.iroots:
        sources = sources_per_relation[iroot]
        targets = list(targets_per_relation[iroot] - set(sources))

        if sources and targets:
            for s, t in product(sources, targets):
                path = nx.shortest_path(g_undirected, s, t)
                # to make sure relation is in dep tree path between the source and the target
                if s != t and iroot in path:
                    triples += [
                        TripleCandidate(
                            source=pile.sources[s],
                            relation=pile.relations[iroot],
                            target=pile.targets[t],
                        )
                    ]
        elif not sources and relaxed:
            triples += [
                TripleCandidate(
                    source=Source(),
                    relation=pile.relations[iroot],
                    target=pile.targets[t],
                )
                for t in targets
            ]
        elif not targets and relaxed:
            triples += [
                TripleCandidate(
                    source=pile.sources[s],
                    relation=pile.relations[iroot],
                    target=Target(),
                )
                for s in sources
            ]

    return triples


def sieve_sources_targets(
    pile: CandidatePile,
) -> tuple[CandidatePile, CandidatePile]:
    sources, targets = CandidatePile(), CandidatePile()
    for c in pile:
        if not c.root.dep_ == "pobj":
            sources.append(c)
        if True:
            targets.append(c)
    return sources, targets


# def doc_to_chunks(rdoc):
#     """
#
#     :param rdoc:
#     :return: (root, start, end) NB: last token is at end-1
#     """
#     acc = []
#     for chunk in rdoc.noun_chunks:
#         acc += [(chunk.root.i, chunk.start, chunk.end)]
#     return acc


# TODO WIP
def phrase_to_relations(
    phrase, nlp, rules, filter_pronouns=True
) -> Tuple[
    nx.DiGraph,
    nx.DiGraph,
    List[TripleCandidate],
    List[Tuple[str, str, str]],
]:
    logging.info(f"{phrase}")

    rdoc, graph = phrase_to_deptree(nlp, phrase)

    # chunks = doc_to_chunks(rdoc)

    triples, source_target_depot = graph_to_triples(graph, rules)

    # coref_graph = render_mstar_graph(rdoc, graph)

    coref_graph = render_coref_graph(rdoc, graph)

    map_subbable_to_blank, map_blank_to_starred = render_coref_candidate_map(
        coref_graph
    )

    # triples_expanded: list[TripleCandidate] = []
    #
    # for triple in triples:
    #     s_candidates = expand_candidate(
    #         triple.source, metagraph=metagraph, coref_graph=coref_graph
    #     )
    #     t_candidates = expand_candidate(
    #         triple.target, metagraph=metagraph, coref_graph=coref_graph
    #     )
    #
    #     triples_expanded += [
    #         TripleCandidate(source=sp, relation=triple.relation, target=tp)
    #         for sp, tp in product(s_candidates, t_candidates)
    #     ]

    # if filter_pronouns:
    #     triples_expanded = [
    #         tri
    #         for tri in triples_expanded
    #         if graph.nodes[tri.source]["tag_"] != "PRP"
    #         and graph.nodes[tri.target]["tag_"] != "PRP"
    #     ]

    triples_proj = [tri.project_to_text() for tri in triples]
    return graph, coref_graph, triples, triples_proj


def add_hash(triples_expanded, graph):
    agg = []

    for tri in triples_expanded:
        # TODO
        source_txt = tri.relation.project_to_text_str()
        target_txt = tri.relation.project_to_text_str()
        relation_txt = tri.relation.project_to_text_str()

        subitem = [
            {
                "hash": hashlib.sha256(source_txt.encode("utf-8")).hexdigest(),
                "text": source_txt,
                "type": "source",
            },
            {
                "hash": hashlib.sha256(
                    relation_txt.encode("utf-8")
                ).hexdigest(),
                "text": relation_txt,
                "type": "relation",
            },
            {
                "hash": hashlib.sha256(target_txt.encode("utf-8")).hexdigest(),
                "text": target_txt,
                "type": "target",
            },
        ]

        item = {"triple": subitem}

        item["triple_meta"] = {
            "hash": hashlib.sha256(
                (
                    hashlib.sha256(source_txt.encode("utf-8")).hexdigest()
                    + hashlib.sha256(relation_txt.encode("utf-8")).hexdigest()
                    + hashlib.sha256(target_txt.encode("utf-8")).hexdigest()
                ).encode("utf-8")
            ).hexdigest(),
            "type": "meta",
        }

        agg += [item]
    return agg
