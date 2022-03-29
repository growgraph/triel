from copy import deepcopy
import spacy
import pandas as pd
from itertools import product
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
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


def add_coref(rdoc: Doc, graph: nx.DiGraph) -> nx.DiGraph:
    chains = rdoc._.coref_chains
    es_coref = []
    for j, chain in enumerate(chains):
        # print(chain.most_specific_mention_index)
        # print([x.token_indexes for x in chain.mentions])
        jc = j + len(rdoc)
        for x in chain.mentions:
            for y in x.token_indexes:
                es_coref.append((jc, y))
    coref_graph = deepcopy(graph)
    coref_graph.add_edges_from(es_coref)
    return coref_graph


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
    conclusion = []
    for r in rules:
        flag = []
        for subrule in r:
            if "how" not in subrule:
                if graph.nodes[v][subrule["key"]] == subrule["value"]:
                    flag.append(True)
                else:
                    flag.append(False)
            elif subrule["how"] == "contains":
                if subrule["value"] in graph.nodes[v][subrule["key"]]:
                    flag.append(True)
                else:
                    flag.append(False)
        conclusion += [all(flag)]
    if any(conclusion):
        # add vertex to subgraph
        subgraph.add_node(v, **graph.nodes[v])
        if u != -1:
            subgraph.add_edge(u, v)
    else:
        # u -> v and v is not of subtype (does not go to subgraph)
        subgraph = nx.DiGraph()
        subgraph.add_node(v, **graph.nodes[v])
        # add subgraph as a node to new_graph
        metagraph.add_node(v, gg=subgraph, **graph.nodes[v])
        if local_root is not None:
            metagraph.add_edge(local_root, v)
        local_root = v
    if u != -1:
        logger.debug(
            f" {u} : {graph.nodes[u]['lower']} : {v} : {graph.nodes[v]['lower']} : {any(conclusion)}"
        )
    for w in graph.successors(v):
        metagraph = fold_graph(graph, v, w, metagraph, local_root, subgraph, rules)
    return metagraph


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


# def graph_to_subgraphs(graph: nx.DiGraph) -> nx.DiGraph:
#     """
#
#     :param graph:
#     :return: list of connected components
#     """
#     roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
#     metas = []
#     relations = []
#     for root in roots:
#         graph


def phrase_to_relations(graph: nx.DiGraph, rules):
    # _, graph = dep_tree_from_phrase(nlp, document)

    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    metas = []
    relations = []
    for root in roots:
        metagraph = graph_to_metagraph(graph, root, rules)
        relations += parse_first_level_relations(metagraph)
        metas += [metagraph]

    def project(x):
        return graph.nodes[x]["lemma"]

    relations_proj = [[project(u) for u in item] for item in relations]
    return graph, relations, relations_proj


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
