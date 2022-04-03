from copy import deepcopy
import spacy
from typing import Optional, Dict
import pandas as pd
from itertools import product
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from lm_service.folded import Leaf
from spacy import Language
from spacy.tokens import Doc
import logging

# import pygraphviz as pgv

logger = logging.getLogger(__name__)

"""
    relation extraction module based on spacy.
    For each phrase:
    1. cast the dependency tree as a metagraph (unify noun chunk) 
    2. identify relation candidates in the metagraph based of tags
    3. for each relation candidate identify a pool of candidate sources and targets
    4. choose closest / best sources/ target candidates for each relation to form (relation, source, target) triples    

"""


class RelationHasNoTargetCandidatesError(Exception):
    pass


class CorefGraph:
    def __init__(self, graph: nx.DiGraph, root: int, map_specific: Dict[int, int]):
        self.graph: nx.DiGraph = graph
        self.root: int = root
        self.map_specific: Dict[int:int] = map_specific


def dep_tree_from_phrase(nlp: Language, document: str) -> (nx.DiGraph, Doc):
    """
    given nlp and a phrase (string) - yield spacy doc and a digraph representing syn parsing
    :param nlp:
    :param document:
    :return:
    """
    graph = nx.DiGraph()

    rdoc = nlp(document)
    vs = [
        (
            token.i,
            {
                "lower": token.lower_,
                "dep": token.dep_,
                "tag": token.tag_,
                "lemma": token.lemma_,
                "label": f"{token.i}-{token.lower_}-{token.dep_}-{token.tag_}",
            },
        )
        for token in rdoc
    ]
    # root = [v[0] for v in vs if v[1]["dep"] == "ROOT"][0]
    # FYI https://spacy.io/docs/api/token
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    es = []
    for token in rdoc:
        for child in token.children:
            es.append((token.i, child.i))

    graph.add_nodes_from(vs)
    graph.add_edges_from(es)

    return rdoc, graph


def render_coref_graph(rdoc: Doc, graph: nx.DiGraph) -> CorefGraph:
    """

    :param rdoc:
    :param graph:
    :return: Di
    """
    chains = rdoc._.coref_chains
    vs_coref = []
    es_coref = []
    jc = len(rdoc)

    chain_specific_mention = dict()
    coref_root = jc
    vs_coref += [
        (
            coref_root,
            {"label": f"{coref_root}-*-coref-root", "tag": "coref", "dep": "root"},
        )
    ]
    jc += 1

    for jchain, chain in enumerate(chains):
        coref_chain = jc
        chain_specific_mention[coref_chain] = chain[chain.most_specific_mention_index]
        vs_coref += [
            (
                coref_chain,
                {
                    "label": f"{coref_chain}-*-coref-chain",
                    "tag": "coref",
                    "dep": "chain",
                    "chain": jchain,
                },
            )
        ]
        es_coref.append((coref_root, coref_chain))
        jc += 1
        for x in chain.mentions:
            coref_blank = jc
            vs_coref += [
                (
                    coref_blank,
                    {
                        "label": f"{coref_blank}-*-coref-blank",
                        "tag": "coref",
                        "dep": "blank",
                        "chain": jchain,
                    },
                )
            ]
            es_coref.append((coref_chain, coref_blank))
            jc += 1
            for y in x.token_indexes:
                es_coref.append((coref_blank, y))
                jc += 1

    coref_graph = deepcopy(graph)
    coref_graph.add_nodes_from(vs_coref)
    coref_graph.add_edges_from(es_coref)
    cg = CorefGraph(coref_graph, coref_root, chain_specific_mention)
    return cg


def render_coref_graph_reduced(rdoc: Doc, graph: nx.DiGraph) -> nx.DiGraph:
    """
    render a dag of type (root -> "concept" -> blank -> mentions)
    # "concept" -> blank always 1:1

    :param rdoc:
    :param graph:
    :return: Di
    """
    chains = rdoc._.coref_chains
    vs_coref = []
    es_coref = []
    jc = len(rdoc)

    chain_specific_mention = dict()
    concept_specific_blank = dict()

    coref_root = jc
    vs_coref += [
        (
            coref_root,
            {"label": f"{coref_root}-*-coref-root", "tag": "coref", "dep": "root"},
        )
    ]
    jc += 1

    mention_nodes = []

    for jchain, chain in enumerate(chains):
        coref_chain = jc
        chain_specific_mention[coref_chain] = chain[chain.most_specific_mention_index]
        vs_coref += [
            (
                coref_chain,
                {
                    "label": f"{coref_chain}-*-coref-chain",
                    "tag": "coref",
                    "dep": "chain",
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
                        "tag": "coref",
                        "dep": "blank",
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

    coref_graph = nx.DiGraph()
    coref_graph.add_nodes_from(vs_coref)
    coref_graph.add_edges_from(es_coref)

    for m in mention_nodes:
        # find m_star
        blanks = list(coref_graph.predecessors(m))
        blank_metrics = []
        for b in blanks:
            # one concept per blank
            c0 = [concept for concept in coref_graph.predecessors(b)][0]
            best_blank_per_concept = concept_specific_blank[c0]
            specific_mentions = list(coref_graph.successors(best_blank_per_concept))
            blank_metrics += [
                (
                    best_blank_per_concept,
                    propotion_of_pronouns(coref_graph, specific_mentions),
                )
            ]
        blank_metrics = sorted(blank_metrics, key=lambda item: item[1])
        coref_graph.nodes[m]["m*"] = list(coref_graph.successors(blank_metrics[0][0]))

    return coref_graph


def propotion_of_pronouns(graph, mentions):
    return sum([graph.nodes[m]["tag"].startswith("PRP") for m in mentions]) / len(
        mentions
    )


def fold_graph(
    graph: nx.DiGraph, u, v, metagraph: nx.DiGraph, local_root, subgraph, rules
) -> nx.DiGraph:
    """

    :param graph: original directed graph
    :param u:
    :param v:
    :param metagraph:
    :param local_root:
    :param subgraph:
    :param rules:
    :return:
    """
    vprops = graph.nodes[v]
    vflag = get_flag(vprops, rules)
    if vflag:
        # add vertex to subgraph
        subgraph.add_node(v, **vprops)
        if u != -1:
            subgraph.add_edge(u, v)
    else:
        # u -> v and vflag is False (v should be folded)
        subgraph = nx.DiGraph()
        subgraph.add_node(v, **vprops)
        # add subgraph as a node to new_graph
        metagraph.add_node(v, gg=subgraph, **vprops)
        if local_root is not None:
            metagraph.add_edge(local_root, v)
        local_root = v
    # if u != -1:
    #     logger.debug(
    #         f" {u} : {graph.nodes[u]['lower']} : {v} : {graph.nodes[v]['lower']} : {vflag}"
    #     )
    for w in graph.successors(v):
        metagraph = fold_graph(graph, v, w, metagraph, local_root, subgraph, rules)
    return metagraph


def fold_graph_v2(
    graph: nx.DiGraph, metagraph: nx.DiGraph, u, v, local_root, rules
) -> nx.DiGraph:

    vprops = graph.nodes[v]
    vflag = get_flag(vprops, rules)

    if vflag and local_root is not None and u is not None:
        subgraph = metagraph.nodes[local_root]["leaf"]
        subgraph.add_node(v, **vprops)
        subgraph.add_edge(u, v)
    else:
        metagraph.add_node(v, **vprops)
        metagraph.nodes[v]["leaf"] = Leaf(v)
        if local_root is not None:
            metagraph.add_edge(local_root, v)
        local_root = v

    for w in graph.successors(v):
        metagraph = fold_graph_v2(graph, metagraph, v, w, local_root, rules)
    return metagraph


def get_flag(props, rules):
    conclusion = []
    for r in rules:
        flag = []
        for subrule in r:
            if "how" not in subrule:
                flag.append(props[subrule["key"]] == subrule["value"])
            elif subrule["how"] == "contains":
                flag.append(subrule["value"] in props[subrule["key"]])
        conclusion += [all(flag)]
    return any(conclusion)


def fold_graph_multi(guide_graph: nx.DiGraph, u, v, flag_u, graph0, rules):
    vprops = guide_graph.nodes[v]
    flag_v = get_flag(vprops, rules)
    print(f"{u} {v} {flag_u} {flag_v} {id(graph0)}")

    if not flag_u and flag_v:
        working_graph = graph0.nodes[u]["g*"]
    else:
        working_graph = graph0

    working_graph.add_node(v, **guide_graph.nodes[v])
    working_graph.add_edge(u, v)

    new_graph = nx.DiGraph()
    new_graph2 = nx.DiGraph()
    new_graph.add_node(v, **guide_graph.nodes[v])
    working_graph.nodes[v]["g*"] = new_graph

    new_graph2.add_node(v, **guide_graph.nodes[v])
    new_graph.nodes[v]["g*"] = new_graph2

    for w in guide_graph.neighbors(v):
        fold_graph_multi(guide_graph, v, w, flag_v, working_graph, rules)


def find_relation_candidates(graph):
    r_candidates = [
        v
        for v in graph.nodes()
        if graph.nodes[v]["tag"][:2] == "VB"
        # and graph.nodes[v]["tag"] != "VBN"
        and graph.nodes[v]["dep"] != "aux"
    ]
    return r_candidates


def maybe_source(n) -> bool:
    return (("NN" in n["tag"]) or (n["tag"] == "PRP")) and (n["dep"] != "pobj")


def maybe_target(n) -> bool:
    return ("NN" in n["tag"]) or (n["dep"] == "pobj") or (n["dep"] == "ccomp")


def check_condition(graph, s, foo_condition) -> bool:
    logger.debug(f" {s} : {id(graph)} : {graph.nodes[s]}")
    flag = [foo_condition(graph.nodes[s])]
    if "gg" in graph.nodes[s]:
        logger.debug(
            f" enter gg:  {id(graph.nodes[s]['gg'])} : {graph.nodes[s]['gg'].nodes()}"
        )
        subgraph = graph.nodes[s]["gg"]
        flag += [
            check_condition(subgraph, n, foo_condition)
            for n in subgraph.nodes
            if n != s
        ]
    return any(flag)


# def target_condition(graph, t):
#     flag = [("NN" in graph.nodes[t]["tag"]) or (graph.nodes[t]["dep"] == "pobj")]
#     if "gg" in graph.nodes[t]:
#         sgraph = graph.nodes[t]["gg"]
#         for n in sgraph.nodes:
#             flag += [source_condition(sgraph, n)]
#     return any(flag)
#     # return ("NN" in graph.nodes[t]["tag"]) or (graph.nodes[t]["dep"] == "pobj")
#     # and not any([graph.nodes[x]["lower"] == "of" for x in graph.predecessors(t)])
#     # )


def parse_first_level_relations(graph):
    """

    :param graph: nx.Digraph is potentially a metagraph, so each source or target might be split into many

    :return:
    """

    relations = []
    rs = find_relation_candidates(graph)
    source_candidates = [
        i for i in graph.nodes if check_condition(graph, i, maybe_source)
    ]
    target_candidates = [
        i for i in graph.nodes if check_condition(graph, i, maybe_target)
    ]
    logger.info(f" relations: {rs} {[graph.nodes[r]['lower'] for r in rs]}")
    logger.info(
        f" sources: {source_candidates} {[graph.nodes[r]['lower'] for r in source_candidates]}"
    )
    logger.info(
        f" targets: {target_candidates} {[graph.nodes[r]['lower'] for r in target_candidates]}"
    )
    undirected = graph.to_undirected()
    greverse = graph.reverse()
    nx.set_edge_attributes(greverse, values=-1, name="weight")

    gextra = graph.copy()
    nx.set_edge_attributes(gextra, values=1, name="weight")

    gextra.add_weighted_edges_from(
        [(u, v, greverse.edges[u, v]["weight"]) for u, v in greverse.edges],
        weight="weight",
    )

    paths = {r: nx.shortest_path(gextra, r) for r in rs}
    path_weights = {
        r: {v: nx.path_weight(gextra, pp, "weight") for v, pp in batch.items()}
        for r, batch in paths.items()
    }

    distance_directed = {r: nx.shortest_path_length(graph, r) for r in rs}
    # dm = pd.DataFrame.from_dict(distance_directed).sort_index(axis=0)

    # distance_reverse = {r: nx.shortest_path_length(greverse, r) for r in rs}
    # rdm = pd.DataFrame.from_dict(distance_reverse).sort_index(axis=0)

    distance_undirected = {r: nx.shortest_path_length(undirected, r) for r in rs}
    udm = pd.DataFrame.from_dict(distance_undirected).sort_index(axis=0)

    wdm = pd.DataFrame.from_dict(path_weights).sort_index(axis=0)

    t_cand = dict()
    s_cand = dict()

    # pick target, targets are down the tree
    for r, dist in distance_directed.items():
        # find min distance to source candidate on the tree wrt relation r
        min_dist = min([dist[k] for k in target_candidates if k in dist and k != r])
        # rarely it could target could be the same as r (if subgraph is hiding in r)
        min_dist = min([dist[k] for k in target_candidates if k in dist and k != r])
        # find all such targets
        t_cand[r] = [k for k in target_candidates if k in dist and dist[k] == min_dist]
        if not t_cand[r]:
            logger.error(f" relation {r} has not target candidates")
            # raise RelationHasNoTargetCandidatesError(f" relation {r} has not target candidates")

    # pick sources; sources might be up the tree, using undirected graph
    for r in rs:
        # for each relation find source candidates
        #  close to relation on the tree, negative cost preferred (in reverse direction), penalty if dep is attr
        undirected_to_source = udm.loc[source_candidates, r]
        cost_to_source = wdm.loc[source_candidates, r]
        decision = pd.concat(
            [undirected_to_source.rename("undirected"), cost_to_source.rename("cost")],
            axis=1,
        )

        decision["syn_penalty"] = pd.Series(
            decision.index.map(
                lambda x: int(graph.nodes[x]["dep"] in ["attr", "dobj"])
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
        s_cand[r] = decision[mask].index.tolist()

    for r in rs:
        sources = sorted(s_cand[r])

        # targets = [t for t in t_cand[r] if t not in s_cand[r]]
        targets = sorted(t_cand[r])

        sources = [s for s in sources if s != targets[-1]]
        targets = [t for t in targets if t != sources[0]]

        logger.info(
            f" relations: {[graph.nodes[s]['lower'] for s in sources]} "
            f"{graph.nodes[r]['lower']} {[graph.nodes[t]['lower'] for t in targets]}"
        )
        for s, t in product(sources, targets):
            path = nx.shortest_path(undirected, s, t)
            if r in path and s != t:
                relations += [(s, r, t)]
                logger.info(
                    f" {graph.nodes[s]['lower']}, {graph.nodes[r]['lower']}, {graph.nodes[t]['lower']}"
                )
    return relations


def graph_to_metagraph(graph: nx.DiGraph, root, rules) -> nx.DiGraph:
    metagraph = nx.DiGraph()
    u, v = -1, root
    metagraph = fold_graph(graph, u, v, metagraph, None, nx.DiGraph(), rules)
    return metagraph


def phrase_to_relations(graph: nx.DiGraph, rules):

    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    metas = []
    relations = []
    mg = nx.DiGraph()
    for root in roots:
        metagraph = graph_to_metagraph(graph, root, rules)
        relations += parse_first_level_relations(metagraph)
        mg.update(metagraph)

    def project(x):
        return graph.nodes[x]["lemma"]

    relations_proj = [[project(u) for u in item] for item in relations]
    return graph, relations, relations_proj, mg


def plot_graph(graph, fields=()):
    if "_id" in fields:
        wfields = [x for x in fields if x != "_id"]
    else:
        wfields = fields

    for n in graph.nodes:
        if "_id" in fields:
            ffs = [str(n)]
        else:
            ffs = []
        ffs += [graph.nodes[n][f] if f in graph.nodes[n] else "0" for f in wfields]
        graph.nodes[n]["label"] = "_".join(ffs)
    a = to_agraph(graph)
    a.layout("dot")
    # graph.draw('file.png')
    # return Image(a.draw(format="png", prog="dot"))
