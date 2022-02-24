import spacy
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import logging

# import pygraphviz as pgv
# from IPython.display import Image

logger = logging.getLogger(__name__)

"""
    relation extraction module based on spacy.
    For each phrase:
    1. cast the dependency tree as a metagraph (unify noun chunk) 
    2. identify relation candidates in the metagraph based of tags
    3. for each relation candidate identify a pool of candidate sources and targets
    4. choose closest / best sources/ target candidates for each relation to form (relation, source, target) triples    

"""


def dep_tree_from_phrase(nlp, document):
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
                "label": f"{token.i}-{token.lower_}",
            },
        )
        for token in rdoc
    ]
    # root = [v[0] for v in vs if v[1]["dep"] == "ROOT"][0]
    # FYI https://spacy.io/docs/api/token
    es = []
    for token in rdoc:
        for child in token.children:
            es.append((token.i, child.i))

    graph.add_nodes_from(vs)
    graph.add_edges_from(es)
    return graph


def prune(graph, u, v, metagraph, local_root, subgraph, rules):
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
    for w in graph.successors(v):
        prune(graph, v, w, metagraph, local_root, subgraph, rules)


def find_relation_candidates(graph):
    r_candidates = [
        v
        for v in graph.nodes()
        if graph.nodes[v]["tag"][:2] == "VB" and graph.nodes[v]["tag"] != "VBN"
    ]
    return r_candidates


def find_target(graph, relation_candidate):
    target_candidates = []
    for s, t, how in nx.edge_dfs(graph, relation_candidate, orientation="original"):
        if "NN" in graph.nodes[t]["tag"] and not any(
            [graph.nodes[x]["lower"] == "of" for x in graph.predecessors(t)]
        ):
            target_candidates += [
                (t, nx.shortest_path_length(graph, relation_candidate, t))
            ]
    return target_candidates


def find_source(graph, relation_candidate):
    source_candidates = []
    for s, t, how in nx.edge_dfs(graph, relation_candidate, orientation="reverse"):
        if (
            "NN"
            in graph.nodes[s]["tag"]
            # or "PRP" in graph.nodes[s]["tag"]
        ) and graph.nodes[s]["dep"] != "pobj":
            source_candidates += [
                (s, nx.shortest_path_length(graph, s, relation_candidate))
            ]
    return source_candidates


def extra_manipulation(graph, source_candidates, target_candidates):
    logger.debug(f" lens : {len(source_candidates)}, {len(target_candidates)}")
    logger.debug(f" {source_candidates}, {target_candidates}")
    for j, (c, distance) in enumerate(target_candidates):
        if graph.nodes[c]["dep"] == "nsubj":
            target_candidates.pop(j)
            source_candidates += [(c, distance)]

    if source_candidates:
        source = sorted(source_candidates, key=lambda x: x[-1])[0]
    else:
        source = None
    if target_candidates:
        target = sorted(target_candidates, key=lambda x: x[-1])[0]
    else:
        target = None
    return source, target


def parse_first_level_relations(graph):
    """

    :param graph: graph is potentially a metagraph, so each source or target might be split into many

    :return:
    """
    relations = []
    rs = find_relation_candidates(graph)
    for r in rs:
        sources = find_source(graph, r)
        targets = find_target(graph, r)
        logger.debug(f" rel: {r}, source: {sources}, target: {targets}")
        s, t = extra_manipulation(graph, sources, targets)
        if s is not None and t is not None:
            relations += [(r, s[0], t[0])]
            logger.debug(
                f" {graph.nodes[s[0]]['lower']}, {graph.nodes[r]['lower']}, {graph.nodes[t[0]]['lower']}"
            )
    return relations


def phrase_to_relations(nlp, document, rules):
    graph = dep_tree_from_phrase(nlp, document)

    if nx.number_weakly_connected_components(graph) > 1:
        logger.info(
            f" number of connected components : {nx.number_weakly_connected_components(graph)}"
        )
        graph = max(
            (graph.subgraph(c) for c in nx.weakly_connected_components(graph)), key=len
        )

    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if len(roots) == 1:
        root = roots[0]
    else:
        logger.debug(" graph is still not (connected and tree)")
        raise ValueError("graph is not (connected and tree)")

    metagraph = nx.DiGraph()
    u, v = -1, root
    prune(graph, u, v, metagraph, None, None, rules)
    relations = parse_first_level_relations(metagraph)

    def project(x):
        return metagraph.nodes[x]["lemma"]

    relations_proj = [[project(u) for u in item] for item in relations]
    return metagraph, relations, relations_proj


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
