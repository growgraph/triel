from __future__ import annotations

import hashlib
import logging
from collections import defaultdict, deque
from copy import deepcopy
from itertools import product

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
    AbsToken,
    ACandidateKind,
    Candidate,
    CandidateType,
    Relation,
    SourceOrTarget,
    Token,
    TokenIndexT,
    apply_map,
)
from lm_service.piles import CandidatePile, ExtCandidateList, SRTPile

logger = logging.getLogger(__name__)

"""
    relation extraction module based on spacy.
    For each phrase:
    1. cast the dependency tree as a metagraph (unify noun chunk) 
    2. identify relation candidates in the metagraph based of tags
    3. for each relation candidate identify a pool of candidate sources and targets
    4. choose closest / best sources/ target candidates for each relation to form (relation, source, target) triples    

"""


def find_candidates_bfs(
    graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    deq: deque[int],
    candidate_pile: CandidatePile,
    how: ACandidateKind,
    robust_mode=False,
    rules=None,
):
    """

    :param graph: please provide a copy graph, find_candidates_bfs potentially modifies graph structure
    :param original_graph: to assign edges for candi ates correctly
    :param deq: deque to keep track of vertices while traversing
    :param candidate_pile: pile to accumulate candidates
    :param how:
    :param robust_mode: if True, try to resolve errors
    :param rules: if True, try to resolve errors
    :return:
    """
    # VB, VBP, VBZ, VBD, VBG, VBN
    # auxiliary verb uses of be, have, do, and get: AUX
    # the infinitive to is PART

    foo_map_class = {
        ACandidateKind.RELATION: Relation,
        ACandidateKind.SOURCE_TARGET: SourceOrTarget,
    }

    rules_current = (
        rules["sourcetarget"]
        if how == ACandidateKind.SOURCE_TARGET
        else rules["relation"]
    )
    if not deq:
        return

    current_vertex = deq.popleft()
    current_candidate = foo_map_class[how]()  # type: ignore

    if current_vertex not in candidate_pile.tokens:
        find_subtree_dfs(graph, original_graph, deque([(current_vertex, 0)]), current_candidate, rules_current)  # type: ignore

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
        rules=rules,
    )


def find_subtree_dfs(
    graph: nx.DiGraph,
    original_graph: nx.DiGraph,
    deq: deque[tuple[int, int]],
    current_candidate: CandidateType,
    rules=None,
):
    """

    :param graph:
    :param original_graph:
    :param deq: (!) the initial call should have only a single vertex in q
    :param current_candidate:
    :param rules:
    :return:
    """

    if not deq:
        return
    current_vertex, level = deq.pop()
    if current_candidate.empty:
        if level > 0:
            return
    else:
        if level - current_candidate.max_level() > 1:
            return

    vtoken = Token(
        **graph.nodes[current_vertex],
        _level=level,
        successors=original_graph.successors(current_vertex),
        predecessors=original_graph.predecessors(current_vertex),
    )

    len_current_relation = len(current_candidate)

    if level == 0 and current_candidate.empty:
        if get_flag(vtoken.__dict__, rules["primary"]):
            current_candidate.append(vtoken)
    elif level > 0 and not current_candidate.empty:
        if get_flag(vtoken.__dict__, rules["secondary"]):
            current_candidate.append(vtoken)
        elif "secondary_preceding" in rules and get_flag(
            vtoken.__dict__, rules["secondary_preceding"]
        ):
            if current_candidate.sroot < vtoken.s:
                current_candidate.append(vtoken)

    successors = list(graph.successors(current_vertex))

    if len(current_candidate) > len_current_relation:
        if level > 0:
            excise_node(graph, current_vertex)
        for v in successors:
            deq.append((v, level + 1))
            find_subtree_dfs(
                graph, original_graph, deq, current_candidate, rules
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
    candidate_pile = CandidatePile()

    mgraph = deepcopy(graph)

    find_candidates_bfs(
        mgraph,
        graph,
        deque(roots),
        relation_pile,
        ACandidateKind.RELATION,
        rules=rules,
        robust_mode=True,
    )

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
    g_reversed: nx.DiGraph,
    indices_of_interest: list[TokenIndexT],
) -> tuple[
    dict[tuple[TokenIndexT, TokenIndexT], int],
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
    distance_reversed: dict[tuple[TokenIndexT, TokenIndexT], int] = {
        (i, v): ll
        for i in indices_of_interest
        for v, ll in nx.shortest_path_length(g_reversed, i).items()
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

    return (
        distance_undirected,
        distance_directed,
        distance_reversed,
        distance_levels,
    )


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
    decision = decision[decision["ud"] != 0]

    decision["mcost"] = (
        decision["ud"] + decision["ld"] + decision["syn_penalty"]
    )

    # source decisions are based on mcost

    mask_mcost = decision.groupby("r")["mcost"].transform("min")

    decision = decision[decision["mcost"].eq(mask_mcost)]

    mask_ud = decision.groupby("r")["ud"].transform("min")

    decision = decision[decision["ud"].eq(mask_ud)]

    sources_per_relation = (
        decision.groupby("r").apply(lambda x: set(x["s"])).to_dict()
    )
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
        r: {t for t, d in item.items() if d == min_dists[r] and t != r}
        if item
        else set()
        for r, item in dist_to_targets.items()
    }
    return targets_per_relation


def realign_prepositions(r: Relation, t: SourceOrTarget, graph: nx.DiGraph):
    """
    remove prepositions that on the path from relation to target
    :param r:
    :param t:
    :param graph:
    :return:
    """
    r = deepcopy(r)
    preposition_tokens = [
        t.s for t in r.tokens if t.dep_ == "prep" and t.tag_ == "IN"
    ]
    for prep in preposition_tokens:
        path = nx.shortest_path(graph, r.sroot, t.sroot)
        if prep not in path:
            r.drop_tokens([prep])
    return r


def derive_relations_per_relation(
    distance_directed, distance_reversed, relation_candidate_roots
):
    dist_rr: dict[TokenIndexT, dict[TokenIndexT, int]] = {
        r: {
            t: distance_directed[r, t]
            for t in relation_candidate_roots
            if t != r and (r, t) in distance_directed
        }
        for r in relation_candidate_roots
    }

    min_dists: dict[TokenIndexT, int] = {
        r: min(item.values()) if item else -1 for r, item in dist_rr.items()
    }

    rel_targets_per_relation: dict[TokenIndexT, set[TokenIndexT]] = {
        r: {t for t, d in item.items() if d == min_dists[r] and t != r}
        if item
        else set()
        for r, item in dist_rr.items()
    }

    dist_rr_rev: dict[TokenIndexT, dict[TokenIndexT, int]] = {
        r: {
            t: distance_reversed[r, t]
            for t in relation_candidate_roots
            if t != r and (r, t) in distance_reversed
        }
        for r in relation_candidate_roots
    }

    min_dists: dict[TokenIndexT, int] = {
        r: min(item.values()) if item else -1
        for r, item in dist_rr_rev.items()
    }

    rel_sources_per_relation: dict[TokenIndexT, set[TokenIndexT]] = {
        r: {t for t, d in item.items() if d == min_dists[r] and t != r}
        if item
        else set()
        for r, item in dist_rr_rev.items()
    }

    return rel_sources_per_relation, rel_targets_per_relation


def graph_to_maps(
    mod_graph, pile
) -> tuple[
    dict[TokenIndexT, set[TokenIndexT]],
    dict[TokenIndexT, set[TokenIndexT]],
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
        distance_reversed,
        distance_levels,
    ) = compute_distances(
        graph=mod_graph,
        g_undirected=g_undirected,
        g_weighted=g_weighted,
        g_reversed=g_reversed,
        indices_of_interest=relation_indices,
    )

    target_candidate_roots = pile.targets.sroots
    source_candidate_roots = pile.sources.sroots
    relation_candidate_roots = pile.relations.sroots

    (
        rel_sources_per_relation,
        rel_targets_per_relation,
    ) = derive_relations_per_relation(
        distance_directed=distance_directed,
        distance_reversed=distance_reversed,
        relation_candidate_roots=relation_candidate_roots,
    )

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

    # make sure sources and targets are disjoint per relation
    for r in set(sources_per_relation) & set(targets_per_relation):
        if sources_per_relation[r] & targets_per_relation[r]:
            # treat the case when sources and targets have non-trivial overlap
            # rank wrt to
            #   (a) contains subj - likely subj
            #   (b) contains obj - likely target
            #   (c) further in the phrase - likely subj

            ranked = [
                (
                    s,
                    int("subj" in mod_graph.nodes[s]["dep_"]),
                    -int("obj" in mod_graph.nodes[s]["dep_"]),
                    -int(s[1]),
                )
                for s in sources_per_relation[r] | targets_per_relation[r]
            ]
            ranked = sorted(ranked, key=lambda x: x[1:], reverse=True)
            subj_flag = [x[1] for x in ranked] + [0]
            # take all subj-like, otherwise take the first element (it is likely to have one subj per relation)
            index_non_subj = subj_flag.index(0)
            if index_non_subj == 0:
                index_non_subj = 1

            sources_per_relation[r] = {x for x, *y in ranked[:index_non_subj]}
            targets_per_relation[r] = {x for x, *y in ranked[index_non_subj:]}
    return (
        sources_per_relation,
        targets_per_relation,
        rel_sources_per_relation,
        rel_targets_per_relation,
        g_undirected,
    )


def form_triples(
    pile: SRTPile,
    sources_per_relation: dict[TokenIndexT, set[TokenIndexT]],
    targets_per_relation: dict[TokenIndexT, set[TokenIndexT]],
    rel_sources_per_relation: dict[TokenIndexT, set[TokenIndexT]],
    rel_targets_per_relation: dict[TokenIndexT, set[TokenIndexT]],
    g_undirected: nx.Graph,
    relaxed=False,
) -> tuple[
    list[tuple[TokenIndexT, TokenIndexT, TokenIndexT]],
    set[tuple[TokenIndexT, TokenIndexT, TokenIndexT]],
]:
    triples = []
    meta_triples = set()
    relation_sroots = set(pile.relations.sroots)

    for sroot in pile.relations.sroots:
        sources = sources_per_relation[sroot]
        targets = targets_per_relation[sroot]
        if not sources:
            sources = rel_sources_per_relation[sroot]
        if not targets:
            targets = rel_targets_per_relation[sroot]

        if sources and targets:
            for s, t in product(sources, targets):
                path = nx.shortest_path(g_undirected, s, t)
                # current relation is the only relation in dep tree path between the source and the target
                flag_only_relation_in_path = (
                    (set(path) - {s, t}) & relation_sroots
                ) == {sroot}

                if s != t:
                    if flag_only_relation_in_path:
                        if (
                            s in pile.sources.sroots
                            and t in pile.targets.sroots
                            # s not in relation_sroots
                            # and t not in relation_sroots
                        ):
                            triples += [(s, sroot, t)]
                        else:
                            # case when source or target was empty, subbed from rel_****ts_per_relation
                            meta_triples |= {(s, sroot, t)}
                    else:
                        # there is another relation in path
                        # decide whether it replaces source or target
                        # options : path ~ [s, rA, r0, rB, t]
                        if sroot in path:
                            index_sroot = path.index(sroot)
                            if set(path[:index_sroot]) & relation_sroots:
                                if rel_sources_per_relation[sroot]:
                                    # TODO clear up next-iter hack
                                    # currently only first element is used (in practice there should be only one)
                                    s = next(
                                        iter(rel_sources_per_relation[sroot])
                                    )
                                else:
                                    s = (0, "nil")
                            if set(path[index_sroot + 1 :]) & relation_sroots:
                                if rel_targets_per_relation[sroot]:
                                    t = next(
                                        iter(rel_targets_per_relation[sroot])
                                    )
                                else:
                                    t = (0, "nil")
                            if s != (0, "nil") and t != (0, "nil"):
                                meta_triples |= {(s, sroot, t)}
        # elif not sources and relaxed:
        #     triples += [
        #         (
        #             None,
        #             sroot,
        #             t,
        #         )
        #         for t in targets
        #     ]
        # elif not targets and relaxed:
        #     triples += [(s, sroot, None) for s in sources]

    return triples, meta_triples


def sieve_sources_targets(
    pile: CandidatePile,
) -> tuple[CandidatePile, CandidatePile]:
    sources, targets = CandidatePile(), CandidatePile()
    for c in pile:
        if True:
            # if not c.root.dep_ == "pobj":
            sources.append(c)
        if True:
            targets.append(c)
    return sources, targets


def text_to_compound_index_graph(nlp, text, initial_phrase_index):
    rdoc, graph = phrase_to_deptree(nlp, text)

    # cast index to compound index
    map_tree_subtree_index = graph_component_maps(graph, initial_phrase_index)

    map_tree_subtree_index = {
        k: AbsToken.ituple2stuple(v) for k, v in map_tree_subtree_index.items()
    }
    graph_relabeled = relabel_nodes_and_key(graph, map_tree_subtree_index, "s")
    return graph_relabeled, rdoc, map_tree_subtree_index


def text_to_coref_sourcetarget(
    nlp, text, ext_candidate_list: ExtCandidateList, initial_phrase_index
) -> defaultdict[TokenIndexT, list[Candidate]]:
    """

    :param nlp:
    :param text:
    :param  ext_candidate_list: ExtCandidateList
    :param  initial_phrase_index
    :return:
    """
    (
        graph_relabeled,
        rdoc,
        map_tree_subtree_index,
    ) = text_to_compound_index_graph(nlp, text, initial_phrase_index)

    # coref maps
    (
        map_subbable_to_chain,
        map_chain_to_most_specific,
    ) = render_coref_maps_wrapper(rdoc)

    (map_subbable_to_chain_str, map_chain_to_most_specific_str,) = apply_map(
        [map_subbable_to_chain, map_chain_to_most_specific],
        map_tree_subtree_index,
    )

    tokens = [
        Token(
            **graph_relabeled.nodes[i],
            successors=graph_relabeled.successors(i),
            predecessors=graph_relabeled.predecessors(i),
        )
        for i in graph_relabeled.nodes()
    ]

    token_dict = {t.s: t for t in tokens}

    # ncp : dict[TokenIndexT, list[Candidate]]
    # for each root -> a list of relevant candidates
    ncp = coref_candidates(
        ext_candidate_list,
        map_subbable_to_chain_str,
        map_chain_to_most_specific_str,
        token_dict,
    )
    return ncp


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
