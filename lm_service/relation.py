from __future__ import annotations
import logging
import hashlib
from copy import deepcopy
from collections import deque

import pandas as pd
from itertools import product
import networkx as nx

from spacy.tokens import Doc

from lm_service.folding import fold_graph_top, get_flag

# import pygraphviz as pgv
from lm_service.graph import phrase_to_deptree, excise_node

from typing import List, Tuple

from lm_service.onto import (
    ACandidate,
    ACandidateType,
    Token,
    Relation,
    SourceOrTarget,
    TripleCandidate,
    ACandidatePile,
    ACandidateKind,
    CandidatePile,
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


class CorefGraph:
    def __init__(
        self, graph: nx.DiGraph, root: int, map_specific: dict[int, int]
    ):
        self.graph: nx.DiGraph = graph
        self.root: int = root
        self.map_specific: dict[int:int] = map_specific


def render_coref_graph(rdoc: Doc, graph: nx.DiGraph, full=False):

    chains = rdoc._.coref_chains if rdoc._.coref_chains is not None else []
    vs_coref = []
    es_coref = []

    mention_nodes = []
    chain_specific_mention = dict()
    concept_specific_blank = dict()

    coref_root = max([token.i for token in rdoc]) + 1
    jc = coref_root + 1

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

    for jchain, chain in enumerate(chains):
        coref_chain = jc
        chain_specific_mention[coref_chain] = chain[
            chain.most_specific_mention_index
        ]
        vs_coref += [
            (
                coref_chain,
                {
                    "label": f"{coref_chain}-*-coref-chain",
                    "tag_": "coref",
                    "dep_": "chain",
                    "chain": jchain,
                },
            )
        ]
        es_coref.append((coref_root, coref_chain))
        jc += 1
        for kth, x in enumerate(chain.mentions):
            coref_blank = jc
            if kth == chain.most_specific_mention_index:
                concept_specific_blank[coref_chain] = coref_blank
            vs_coref += [
                (
                    coref_blank,
                    {
                        "label": f"{coref_blank}-*-coref-blank",
                        "tag_": "coref",
                        "dep_": "blank",
                        "chain": jchain,
                    },
                )
            ]
            es_coref.append((coref_chain, coref_blank))
            jc += 1
            mention_nodes.extend(x.token_indexes)
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

    if full:
        coref_graph = deepcopy(graph)
    else:
        coref_graph = nx.DiGraph()

    coref_graph.add_nodes_from(vs_coref)
    coref_graph.add_edges_from(es_coref)
    return coref_graph, mention_nodes, concept_specific_blank


def render_mstar_graph(rdoc: Doc, graph: nx.DiGraph) -> nx.DiGraph:
    coref_graph, mention_nodes, concept_specific_blank = render_coref_graph(
        rdoc, graph
    )

    for m in mention_nodes:
        # find m_star
        blanks = list(coref_graph.predecessors(m))
        blank_metrics = []
        for b in blanks:
            # one concept per blank
            c0 = [concept for concept in coref_graph.predecessors(b)][0]
            best_blank_per_concept = concept_specific_blank[c0]
            specific_mentions = list(
                coref_graph.successors(best_blank_per_concept)
            )
            blank_metrics += [
                (
                    best_blank_per_concept,
                    propotion_of_pronouns(coref_graph, specific_mentions),
                )
            ]
        blank_metrics = sorted(blank_metrics, key=lambda item: item[1])
        coref_graph.nodes[m]["m*"] = list(
            coref_graph.successors(blank_metrics[0][0])
        )

    return coref_graph


def propotion_of_pronouns(graph, mentions):
    return sum(
        [graph.nodes[m]["tag_"].startswith("PRP") for m in mentions]
    ) / len(mentions)


def find_candidates_bfs(
    graph: nx.DiGraph,
    q: deque[int],
    candidate_pile: ACandidatePile,
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
        ACandidateKind.SOURCE_TARGET: find_st_subtree_dfs,
    }

    foo_map_class = {
        ACandidateKind.RELATION: Relation,
        ACandidateKind.SOURCE_TARGET: SourceOrTarget,
    }

    foo = foo_map[how]
    if not q:
        return

    current_vertex = q.popleft()
    current_relation = foo_map_class[how]()

    successors = set(graph.successors(current_vertex))
    if current_vertex not in candidate_pile.tokens:
        foo(graph, deque([(current_vertex, 0)]), current_relation, **kwargs)

    if not current_relation.empty:
        current_relation.sort()
        candidate_pile.append(current_relation)

    q.extend(successors & set(graph.nodes))
    # print("\n")
    find_candidates_bfs(graph, q, candidate_pile, how, **kwargs)


def find_relation_subtree_dfs(
    graph: nx.DiGraph, q: deque[(int, int)], current_relation: ACandidateType
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

    vtoken = Token(**graph.nodes[current_vertex], **{"_level": level})

    len_current_relation = len(current_relation)

    if level == 0 and current_relation.empty:
        if vtoken.tag_.startswith("VB") and vtoken.dep_ != "amod":
            current_relation.append(vtoken)
            if vtoken.dep_ == "acl":
                current_relation.passive = True
    elif level > 0 and not current_relation.empty:
        # auxpass or aux
        if (vtoken.tag_.startswith("VB") or (vtoken.tag_ == "MD")) and (
            "aux" in vtoken.dep_
        ):
            current_relation.prepend(vtoken)
            # current_relation.passive = True
        # elif vtoken.tag_.startswith("VB") and vtoken.dep_ == "advcl":
        #     current_relation.append(vtoken)
        #     current_relation.passive = True
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


def find_st_subtree_dfs(
    graph: nx.DiGraph,
    q: deque[(int, int)],
    current_st: ACandidateType,
    rules=None,
):
    """

    :param graph:
    :param q: (!) the initial call should have only a single vertex in q
    :param current_st: source or target
    :param rules:
    :return:
    """

    if not q:
        return
    # current_vertex, level = q.popleft()
    current_vertex, level = q.pop()
    # print(current_vertex, end="->")

    # print(
    #     current_vertex,
    #     graph.nodes[current_vertex],
    #     level,
    #     current_st.max_level(),
    # )
    if (
        current_st.empty
        and level > 0
        or (
            not current_st.empty
            and (
                (level - current_st.max_level() > 1)
                or (current_st._tokens[0].dep_ == "ccomp")
            )
        )
    ):
        return

    vtoken = Token(**graph.nodes[current_vertex], **{"_level": level})

    len_current_relation = len(current_st)

    if level == 0 and current_st.empty:
        if (("NN" in vtoken.tag_) or (vtoken.tag_ == "PRP")) or (
            vtoken.dep_ == "ccomp"
        ):
            current_st.append(vtoken)
    elif level > 0 and not current_st.empty:
        vflag = get_flag(vtoken.__dict__, rules)
        if vflag:
            current_st.append(vtoken)

    # q.extend((v, level + 1) for v in graph.successors(current_vertex))

    successors = list(graph.successors(current_vertex))

    if len(current_st) > len_current_relation:
        if level > 0:
            excise_node(graph, current_vertex)
        for v in successors:
            q.append((v, level + 1))
            find_st_subtree_dfs(graph, q, current_st, rules)

    # if len(current_st) > len_current_relation:
    #     excise_node(graph, current_vertex)
    #     find_st_subtree_dfs(graph, q, current_st, rules)


def find_relation_candidates_obsolete(graph: nx.DiGraph) -> ACandidatePile:
    r_candidates = []
    for v in graph.nodes():
        cand = Relation()

        if graph.nodes[v]["tag_"].startswith("VB"):
            # TODO check graph.nodes[v]["dep_"] != "aux"
            vtoken = Token(**graph.nodes[v])
            if vtoken.tag_ == "VBN" and vtoken.dep_ == "acl":
                cand.passive = True

            if len(list(graph.successors(v))) > 0:
                cand._tokens = [vtoken]
            for w in graph.successors(v):
                wtoken = Token(**graph.nodes[w])

                if wtoken.tag_.startswith("VB"):
                    if (
                        vtoken.tag_ == "VBN"
                        and
                        # VBN or VBZ
                        (
                            "VB" in wtoken.tag_
                            # auxpass or aux
                            and "aux" in wtoken.dep_
                        )
                    ):
                        cand._tokens = [wtoken] + cand._tokens
                        cand.passive = True
                    elif (
                        vtoken.tag_ == "VBZ"
                        and wtoken.tag_ == "VBN"
                        and wtoken.dep_ == "advcl"
                    ):
                        cand._tokens = cand._tokens + [wtoken]
                        cand.passive = True
                if (wtoken.tag_ == "IN" and wtoken.dep_ == "prep") or (
                    wtoken.tag_ == "IN" and wtoken.dep_ == "agent"
                ):
                    if any([t.tag_ == "IN" for t in cand._tokens]):
                        cand2 = deepcopy(cand)
                        for j, t in enumerate(cand2._tokens):
                            if t.tag_ == "IN":
                                cand2._tokens[j] = wtoken
                        r_candidates += [cand2]
                    else:
                        cand._tokens = cand._tokens + [wtoken]
        if cand.tokens:
            r_candidates += [cand]
    for j, r in enumerate(r_candidates):
        r.r0 = j

    return ACandidatePile(candidates=r_candidates)


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


def graph_to_candidate_pile(graph: nx.DiGraph, rules) -> CandidatePile:
    roots = [n for n, d in graph.in_degree() if d == 0]
    relation_pile = ACandidatePile()
    source_target_pile = ACandidatePile()

    find_candidates_bfs(
        graph,
        deque(roots),
        relation_pile,
        ACandidateKind.RELATION,
    )
    find_candidates_bfs(
        graph,
        deque(roots),
        source_target_pile,
        ACandidateKind.SOURCE_TARGET,
        rules=rules,
    )
    source_candidates, target_candidates = sieve_sources_targets(
        source_target_pile
    )

    logger.info(f" relations: {relation_pile}")
    # for r in relation_pile.candidates:
    #     logger.info(
    #         f" relations: {[graph.nodes[r0]['lower'] for r0 in r.tokens]}"
    #     )
    # logger.info(
    #     " sources:"
    #     f" {source_candidates} {[graph.nodes[r]['lower'] for r in source_candidates]}"
    # )
    # logger.info(
    #     " targets:"
    #     f" {target_candidates} {[graph.nodes[r]['lower'] for r in target_candidates]}"
    # )

    return CandidatePile(
        relations=relation_pile,
        sources=source_candidates,
        targets=target_candidates,
    )


def compute_distances(graph: nx.DiGraph, pile: ACandidatePile):

    g_undirected = graph.to_undirected()
    g_reversed = graph.reverse()

    nx.set_edge_attributes(g_reversed, values=-1, name="weight")

    g_original = graph.copy()
    nx.set_edge_attributes(g_original, values=1, name="weight")

    g_original.add_weighted_edges_from(
        [(u, v, g_reversed.edges[u, v]["weight"]) for u, v in g_reversed.edges],
        weight="weight",
    )

    # compute distances
    # g_original | dag paths
    # paths = {c.root.i: nx.shortest_path(g_original, c.root.i) for c in pile.candidates}

    # g_original | (root, vertex) : weight
    # path_weights = {
    #     (r, v): nx.path_weight(g_original, pp, "weight")
    #     for r, batch in paths.items() for v, pp in batch.items()
    # }

    # distance_directed = {
    #     c.root.i: nx.shortest_path_length(graph, c.root.i) for c in pile.candidates
    # }

    # # g_original | (root, vertex) : weight
    distance_directed = {
        (c.root.i, v): ll
        for c in pile.candidates
        for v, ll in nx.shortest_path_length(graph, c.root.i).items()
    }

    # dm = pd.DataFrame.from_dict(distance_directed).sort_index(axis=0)

    # distance_reverse = {r: nx.shortest_path_length(greverse, r) for r in rs}
    # rdm = pd.DataFrame.from_dict(distance_reverse).sort_index(axis=0)

    # distance_undirected = {
    #     r: nx.shortest_path_length(g_undirected, r) for r in pile.tokens
    # }

    distance_undirected = {
        (c.root.i, v): ll
        for c in pile.candidates
        for v, ll in nx.shortest_path_length(g_undirected, c.root.i).items()
    }

    # undirected graph distance matrix
    udm = pd.DataFrame(
        distance_undirected.values(), index=distance_undirected.keys()
    ).sort_index()

    # weighted graph distance matrix
    wdm = pd.DataFrame(
        distance_directed.values(), index=distance_directed.keys()
    ).sort_index()

    return g_undirected, distance_directed


def graph_to_relations(graph, rules):
    """
    find triplets in a dep graph:
        a. find relation candidates
        b. find source candidates
        c. find target candidates

    :param graph: nx.Digraph
    :param rules: nx.Digraph

    :return:
    """

    triples = []

    pile = graph_to_candidate_pile(graph, rules)

    # create relevant graphs for distance calculations : undirected, reversed ...

    undirected, distance_directed = compute_distances(graph, pile.relations)

    # WIP

    target_per_relation = dict()
    sources_per_relation = dict()

    target_candidates = pile.targets.tokens
    source_candidates = pile.sources.tokens

    # find targets per relation; targets are down the tree
    for r_parent, rels in pile.relations.map.items():
        dist_r_parent = []
        for r in rels:
            dist = distance_directed[r]
            # for r, dist in distance_directed.items():
            # find min distance to source candidate on the tree wrt relation r
            # target could be the same as r (if subgraph is hiding in r)
            dist_to_targets = [
                (r, target, dist[target])
                for target in target_candidates
                if target in dist and target not in rels
            ]
            dist_r_parent += dist_to_targets

        if dist_r_parent:
            min_dist = min([d for _, _, d in dist_r_parent])
            # find all such targets
            target_per_relation[r_parent] = set(
                [
                    target
                    for relation_part, target, d_rk in dist_r_parent
                    if d_rk == min_dist
                ]
            )

        if not dist_r_parent:
            target_per_relation[r_parent] = set()
            logger.error(f" relation {r_parent} has not target candidates")
            # raise RelationHasNoTargetCandidatesError(f" relation {r} has not target candidates")

    udm_source = list(set(source_candidates) & set(udm.index))
    wdm_source = list(set(source_candidates) & set(wdm.index))

    # find sources per relation; sources may be up the tree, using undirected graph
    # for each relation find source candidates
    # a. close to relation on the tree
    # b. negative cost preferred (close in reverse direction),
    # c. add penalty if dep is attr for given node
    for r_parent, rels in pile.relations.map.items():
        try:
            undirected_to_source = udm.loc[udm_source, rels].unstack()
        except ValueError:
            sources_per_relation[r_parent] = []
            continue

        try:
            cost_to_source = wdm.loc[wdm_source, rels].unstack()
        except ValueError:
            sources_per_relation[r_parent] = []
            continue

        decision = pd.concat(
            [
                undirected_to_source.rename("undirected"),
                cost_to_source.rename("cost"),
            ],
            axis=1,
        )

        # if candidate is "attr" or "dobj" add penalty (because they are likely to be targets
        decision["syn_penalty"] = pd.Series(
            decision.index.map(
                lambda x: int(graph.nodes[x[1]]["dep_"] in ["attr", "dobj"])
            ),
            index=decision.index,
        )

        decision["mcost"] = (
            decision["undirected"] + decision["cost"] + decision["syn_penalty"]
        )

        decision = decision.sort_values(
            ["mcost", "undirected", "cost", "syn_penalty"],
            ascending=[True, True, True, True],
        )
        top_row = decision.iloc[0]
        mask = (decision == top_row).all(axis=1)
        sources_per_relation[r_parent] = (
            decision[mask].index.get_level_values(1).tolist()
        )

    for r in pile.relations:
        sources = sources_per_relation[r.r0]
        targets = set(target_per_relation[r.r0]) - set(
            sources_per_relation[r.r0]
        )

        for s, t in product(sources, targets):
            path = nx.shortest_path(undirected, s, t)
            if any([token in path for token in r.tokens]) and s != t:
                triples += [TripleCandidate(s, r, t)]
                logger.info(
                    f" {graph.nodes[s]['lower']},"
                    f" {r.project_to_text_str(graph)},"
                    f" {graph.nodes[t]['lower']}"
                )
    return triples


def sieve_sources_targets(
    pile: ACandidatePile,
) -> tuple[ACandidatePile, ACandidatePile]:
    sources, targets = ACandidatePile(), ACandidatePile()
    for c in pile:
        if not c.root.dep_ == "pobj":
            sources.append(c)
        if True:
            targets.append(c)
    return sources, targets


def yield_star_nodes(graph, node_list):
    """
    yield most specific mentions for any mentions, given a coref graph
    :param graph:
    :param node_list:
    :return:
    """
    nlist = set()
    for n in node_list:
        if "m*" in graph.nodes[n] and n in graph.nodes[n]["m*"]:
            nlist |= {n}
        else:
            nlist |= yield_star_nodes(graph, graph.nodes[n]["m*"])
    return nlist


def expand_mstar(candidates, coref_graph):
    candidates_out = set()
    for c in candidates:
        if c in coref_graph.nodes():
            candidates_out |= yield_star_nodes(
                coref_graph, coref_graph.nodes[c]["m*"]
            )
        else:
            candidates_out |= {c}
    return list(candidates_out)


def expand_candidate(candidate_token: int, metagraph, coref_graph):

    # t = st.tree
    # [(t.nodes[n]["lower"], t.nodes[n]["tag_"], t.nodes[n]["dep_"]) for n in t.nodes() if "lower" in t.nodes[n]]

    candidates = [candidate_token]
    if metagraph.nodes[candidate_token]["leaf"].is_compound():
        candidates = metagraph.nodes[candidate_token]["leaf"].compute_conj()
    candidates = expand_mstar(candidates, coref_graph)
    return candidates


def doc_to_chunks(rdoc):
    """

    :param rdoc:
    :return: (root, start, end) NB: last token is at end-1
    """
    acc = []
    for chunk in rdoc.noun_chunks:
        acc += [(chunk.root.i, chunk.start, chunk.end)]
    return acc


def phrase_to_relations(
    phrase, nlp, rules, filter_pronouns=True
) -> Tuple[
    nx.DiGraph,
    nx.DiGraph,
    nx.DiGraph,
    List[TripleCandidate],
    List[Tuple[str, str, str]],
]:
    logging.info(f"{phrase}")

    rdoc, graph = phrase_to_deptree(nlp, phrase)

    # chunks = doc_to_chunks(rdoc)

    triples = graph_to_relations(graph, rules)
    # _, triples, triples_projected, metagraph = graph_to_relations2(graph, rules)

    coref_graph = render_mstar_graph(rdoc, graph)

    triples_expanded = []

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

    if filter_pronouns:
        triples_expanded = [
            tri
            for tri in triples_expanded
            if graph.nodes[tri.source]["tag_"] != "PRP"
            and graph.nodes[tri.target]["tag_"] != "PRP"
        ]

    triples_proj = [tri.project_to_text(graph) for tri in triples_expanded]
    return graph, coref_graph, metagraph, triples_expanded, triples_proj


def add_hash(triples_expanded, graph):
    agg = []

    for tri in triples_expanded:
        source_txt = ACandidate.concretize(tri.source, graph)
        target_txt = ACandidate.concretize(tri.target, graph)
        relation_txt = tri.relation.project_to_text_str(graph)

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
