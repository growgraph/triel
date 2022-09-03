from __future__ import annotations

import hashlib
import logging
from collections import deque
from copy import deepcopy
from itertools import product
from typing import List, Tuple

import networkx as nx
import pandas as pd

from lm_service.coref import (
    coref_candidates,
    graph_component_maps,
    render_coref_maps_wrapper,
)
from lm_service.folding import get_flag
from lm_service.graph import (
    excise_node,
    phrase_to_deptree,
    relabel_nodes_and_key,
)
from lm_service.onto import (
    ACandidateKind,
    CandidateType,
    Relation,
    Source,
    SourceOrTarget,
    Target,
    Token,
    TokenIndexT,
    TripleCandidate,
    apply_map,
)
from lm_service.piles import CandidatePile, SRTPile

logger = logging.getLogger(__name__)

"""
    relation extraction module based on spacy.
    For each phrase:
    1. cast the dependency tree as a metagraph (unify noun chunk) 
    2. identify relation candidates in the metagraph based of tags
    3. for each relation candidate identify a pool of candidate sources and targets
    4. choose closest / best sources/ target candidates for each relation to form (relation, source, target) triples    

"""


def is_compound_index_graph(g):
    n = next(iter(g.nodes))
    return isinstance(n, tuple)


class SimplifiedConjunctionHypothesisFailure(Exception):
    pass


def find_candidates_bfs(
    graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    deq: deque[int],
    candidate_pile: CandidatePile,
    how: ACandidateKind,
    robust_mode=False,
    **kwargs,
):
    """

    :param graph: please provide a copy graph, find_candidates_bfs potentially modifies graph structure
    :param original_graph: to assign edges for candi ates correctly
    :param deq: deque to keep track of vertices while traversing
    :param candidate_pile: pile to accumulate candidates
    :param how:
    :param robust_mode: if True, try to resolve errors
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

    # token_transform = token_transforms[
    #     "compound" if is_compound_index(graph) else "int"
    # ]

    foo_func = foo_map[how]
    if not deq:
        return

    current_vertex = deq.popleft()
    current_candidate = foo_map_class[how]()  # type: ignore

    # if token_transform(current_vertex) not in candidate_pile.tokens:
    if current_vertex not in candidate_pile.tokens:
        foo_func(graph, original_graph, deque([(current_vertex, 0)]), current_candidate, **kwargs)  # type: ignore

    if not current_candidate.empty:  # type: ignore
        candidate_pile.append(current_candidate.clean_dangling_edges(robust_mode).sort_index())  # type: ignore

    deq.extend(graph.successors(current_vertex))
    find_candidates_bfs(
        graph,
        original_graph,
        deq,
        candidate_pile,
        how,
        robust_mode=robust_mode,
        **kwargs,
    )


def find_relation_subtree_dfs(
    graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    deq: deque[tuple[int, int]],
    current_relation: CandidateType,
):
    """

    :param graph:
    :param original_graph:
    :param deq: (!) the initial call should have only a single vertex in q
    :param current_relation:
    :return:
    """

    if not deq:
        return
    current_vertex, level = deq.pop()
    if (
        current_relation.empty
        and level > 0
        or (
            not current_relation.empty
            and level - current_relation.max_level() > 1
        )
    ):
        return

    vtoken = Token(
        **graph.nodes[current_vertex],
        _level=level,
        successors=original_graph.successors(current_vertex),
        predecessors=original_graph.predecessors(current_vertex),
    )

    len_current_relation = len(current_relation)

    # TODO externalize logic using get_flag_advanced()

    if level == 0 and current_relation.empty:
        if vtoken.tag_.startswith("VB") and (
            vtoken.dep_ != "amod"
            or (vtoken.tag_ == "VBN" and vtoken.dep_ == "acl")
        ):
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
            if current_relation.sroot < vtoken.s:
                current_relation.append(vtoken)

    successors = list(graph.successors(current_vertex))

    if len(current_relation) > len_current_relation:
        if level > 0:
            excise_node(graph, current_vertex)
        for v in successors:
            deq.append((v, level + 1))
            find_relation_subtree_dfs(
                graph, original_graph, deq, current_relation
            )


def find_sourcetarget_subtree_dfs(
    graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    deq: deque[tuple[int, int]],
    source_target_cand: CandidateType,
    rules=None,
):
    """

    :param graph:
    :param original_graph:
    :param deq: (!) the initial call should have only a single vertex in q
    :param source_target_cand: source or target
    :param rules:
    :return:
    """

    if not deq:
        return
    current_vertex, level = deq.pop()
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

    vtoken = Token(
        **graph.nodes[current_vertex],
        _level=level,
        successors=original_graph.successors(current_vertex),
        predecessors=original_graph.predecessors(current_vertex),
    )

    len_current_relation = len(source_target_cand)

    if level == 0 and source_target_cand.empty:
        if (("NN" in vtoken.tag_) or (vtoken.tag_ == "PRP")) or (
            vtoken.dep_ == "ccomp"
        ):
            source_target_cand.append(vtoken)
    elif level > 0 and not source_target_cand.empty:
        if get_flag(vtoken.__dict__, rules):
            source_target_cand.append(vtoken)

    successors = list(graph.successors(current_vertex))

    if len(source_target_cand) > len_current_relation:
        if level > 0:
            excise_node(graph, current_vertex)
        for v in successors:
            deq.append((v, level + 1))
            find_sourcetarget_subtree_dfs(
                graph, original_graph, deq, source_target_cand, rules
            )


def graph_to_candidate_pile(
    graph: nx.DiGraph, rules, robust_mode=False
) -> tuple[SRTPile, CandidatePile, nx.DiGraph]:
    """

    :param graph:
    :param rules:
    :param robust_mode:
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
        graph,
        deque(roots),
        relation_pile,
        ACandidateKind.RELATION,
        robust_mode=True,
    )

    candidate_pile = CandidatePile()
    find_candidates_bfs(
        mgraph,
        graph,
        deque(roots),
        candidate_pile,
        ACandidateKind.SOURCE_TARGET,
        rules=rules,
        robust_mode=robust_mode,
    )

    source_candidates, target_candidates = sieve_sources_targets(
        candidate_pile
    )

    logger.info(f" relations: {relation_pile}")

    return (
        SRTPile(
            relations=relation_pile,
            sources=source_candidates,
            targets=target_candidates,
        ),
        candidate_pile,
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
    indices_of_interest: list[TokenIndexT],
) -> tuple[
    dict[tuple[TokenIndexT, TokenIndexT], int],
    dict[tuple[TokenIndexT, TokenIndexT], int],
    dict[tuple[TokenIndexT, TokenIndexT], int],
]:

    # (u, v) : distance
    distance_directed: dict[tuple[TokenIndexT, TokenIndexT], int] = {
        (i, v): ll
        for i in indices_of_interest
        for v, ll in nx.shortest_path_length(graph, i).items()
    }

    # (u, v) : distance
    distance_undirected: dict[tuple[TokenIndexT, TokenIndexT], int] = {
        (i, v): ll
        for i in indices_of_interest
        for v, ll in nx.shortest_path_length(g_undirected, i).items()
    }

    # compute distances
    # gextra | dag paths
    paths = {i: nx.shortest_path(g_weighted, i) for i in indices_of_interest}

    # g_original | (root, vertex) : weight
    distance_levels: dict[tuple[TokenIndexT, TokenIndexT], int] = {
        (r, v): nx.path_weight(g_weighted, pp, "weight")
        for r, batch in paths.items()
        for v, pp in batch.items()
    }

    return distance_undirected, distance_directed, distance_levels


def derive_sources_per_relation(
    relation_candidate_roots: list[TokenIndexT],
    source_candidate_roots: list[TokenIndexT],
    distance_undirected: dict[tuple[TokenIndexT, TokenIndexT], int],
    distance_levels: dict[tuple[TokenIndexT, TokenIndexT], int],
    pile_sources_roots: list[Token],
) -> dict[TokenIndexT, set[TokenIndexT]]:
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
        [(s.s, int(s.dep_ in ["attr", "dobj"])) for s in pile_sources_roots],
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
    relation_candidate_roots: list[TokenIndexT],
    target_candidate_roots: list[TokenIndexT],
    distance_directed: dict[tuple[TokenIndexT, TokenIndexT], int],
) -> dict[TokenIndexT, set[TokenIndexT]]:

    # find targets per relation; targets are down the tree
    dist_to_targets: dict[TokenIndexT, dict[TokenIndexT, int]] = {
        r: {
            t: distance_directed[r, t]
            for t in target_candidate_roots
            if (r, t) in distance_directed
        }
        for r in relation_candidate_roots
    }

    min_dists: dict[TokenIndexT, int] = {
        r: min(item.values()) if item else -1
        for r, item in dist_to_targets.items()
    }

    targets_per_relation: dict[TokenIndexT, set[TokenIndexT]] = {
        r: {t for t, d in item.items() if d == min_dists[r]} if item else set()
        for r, item in dist_to_targets.items()
    }
    return targets_per_relation


def graph_to_triples(
    graph,
    map_subbable_to_chain_str,
    map_chain_to_most_specific_str,
    rules,
) -> list[TripleCandidate]:
    """
    find triplets in a dep graph:
        a. find relation candidates
        b. find source candidates
        c. find target candidates

    :param graph: nx.Digraph
    :param map_subbable_to_chain_str:
    :param map_chain_to_most_specific_str:
    :param rules:

    :return:
    """

    # derive
    # i) pile of source, relation, target
    # ii) depot of candidates (mixed)
    # iii) modified graph to use for distance computation
    pile, candidate_depot, mod_graph = graph_to_candidate_pile(graph, rules)

    tokens = [
        Token(
            **graph.nodes[i],
            successors=graph.successors(i),
            predecessors=graph.predecessors(i),
        )
        for i in graph.nodes()
    ]

    token_dict = {t.s: t for t in tokens}

    (
        sources_per_relation,
        targets_per_relation,
        g_undirected,
    ) = graph_to_maps(mod_graph=mod_graph, pile=pile)

    itriples = form_triples(
        pile, sources_per_relation, targets_per_relation, g_undirected
    )

    # ncp : dict[str, list[Candidate]]
    # for each root -> a list of relevant candidates
    ncp = coref_candidates(
        candidate_depot,
        map_subbable_to_chain_str,
        map_chain_to_most_specific_str,
        token_dict,
        unfold_conjunction=True,
    )

    triples = []
    # expand using coref and conj maps
    for tri in itriples:
        s, r, t = tri
        for sprime, tprime in product(ncp[s], ncp[t]):
            triples += [
                TripleCandidate(
                    source=sprime, target=tprime, relation=pile.relations[r]
                )
            ]

    triples = sorted(
        triples,
        key=lambda x: (x.source.sroot, x.relation.sroot, x.target.sroot),
    )

    return triples


def graph_to_maps(
    mod_graph, pile
) -> tuple[
    dict[TokenIndexT, set[TokenIndexT]],
    dict[TokenIndexT, set[TokenIndexT]],
    nx.Graph,
]:
    """

    derive maps
        a. relation -> source
        b. relation -> target

    :param mod_graph: nx.Digraph
    :param pile:

    :return:
    """

    # create relevant graphs for distance calculations : undirected, reversed ...
    g_undirected, g_reversed, g_weighted = generate_extra_graphs(mod_graph)

    relation_indices = [c.root.s for c in pile.relations]

    (
        distance_undirected,
        distance_directed,
        distance_levels,
    ) = compute_distances(
        graph=mod_graph,
        g_undirected=g_undirected,
        g_weighted=g_weighted,
        indices_of_interest=relation_indices,
    )

    # token_transform = token_transforms[
    #     "compound" if is_compound_index(mod_graph) else "int"
    # ]
    # distance_undirected_str = to_string_keys(
    #     distance_undirected, token_transform
    # )
    # distance_directed_str = to_string_keys(distance_directed, token_transform)
    # distance_levels_str = to_string_keys(distance_levels, token_transform)

    target_candidate_roots = pile.targets.sroots
    source_candidate_roots = pile.sources.sroots
    relation_candidate_roots = pile.relations.sroots

    targets_per_relation: dict[
        TokenIndexT, set[TokenIndexT]
    ] = derive_targets_per_relaton(
        relation_candidate_roots, target_candidate_roots, distance_directed
    )

    sources_per_relation: dict[
        TokenIndexT, set[TokenIndexT]
    ] = derive_sources_per_relation(
        relation_candidate_roots,
        source_candidate_roots,
        distance_undirected,
        distance_levels,
        pile.sources.roots,
    )
    return (
        sources_per_relation,
        targets_per_relation,
        g_undirected,
    )


def form_triples(
    pile: SRTPile,
    sources_per_relation: dict[TokenIndexT, set[TokenIndexT]],
    targets_per_relation: dict[TokenIndexT, set[TokenIndexT]],
    g_undirected: nx.Graph,
    relaxed=False,
) -> list:
    triples = []

    for sroot in pile.relations.sroots:
        sources = sources_per_relation[sroot]
        targets = list(targets_per_relation[sroot] - set(sources))

        if sources and targets:
            for s, t in product(sources, targets):
                # si, ti, srooti = map(int, (s, t, sroot))
                path = nx.shortest_path(g_undirected, s, t)
                # to make sure relation is in dep tree path between the source and the target
                if s != t and sroot in path:
                    triples += [(s, sroot, t)]
        elif not sources and relaxed:
            triples += [
                TripleCandidate(
                    source=Source(),
                    relation=pile.relations[sroot],
                    target=pile.targets[t],
                )
                for t in targets
            ]
        elif not targets and relaxed:
            triples += [
                TripleCandidate(
                    source=pile.sources[s],
                    relation=pile.relations[sroot],
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


def phrase_to_triples(
    phrase, nlp, rules, filter_pronouns=True
) -> Tuple[List[TripleCandidate], List[Tuple[str, str, str]], nx.DiGraph]:
    # work this out

    logger.info(f"{phrase}")

    rdoc, graph = phrase_to_deptree(nlp, phrase)

    # graph_relabeled = nx.relabel_nodes(graph, map_tree_subtree_index)

    map_tree_subtree_index = graph_component_maps(graph)

    graph_relabeled = relabel_nodes_and_key(graph, map_tree_subtree_index, "s")

    # coref maps
    (
        map_subbable_to_chain,
        map_chain_to_most_specific,
    ) = render_coref_maps_wrapper(rdoc)

    (map_subbable_to_chain_str, map_chain_to_most_specific_str,) = apply_map(
        [map_subbable_to_chain, map_chain_to_most_specific],
        map_tree_subtree_index,
    )

    triples = graph_to_triples(
        graph_relabeled,
        map_subbable_to_chain_str,
        map_chain_to_most_specific_str,
        rules,
    )

    triples = [tri.normalize_relation() for tri in triples]
    if filter_pronouns:
        triples = [tri for tri in triples if not tri.has_pronouns()]
    triples_proj = [tri.project_to_text() for tri in triples]
    return triples, triples_proj, graph


def add_hash(triples_expanded):
    agg = []

    for tri in triples_expanded:
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
