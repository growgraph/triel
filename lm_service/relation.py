import spacy
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from copy import deepcopy
from spacy import Language
from spacy.tokens import Doc
import pandas as pd
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


def prune(
    graph: nx.DiGraph, u, v, metagraph: nx.DiGraph, local_root, subgraph, rules
) -> (nx.DiGraph, nx.DiGraph):
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
        metagraph, subgraph = prune(graph, v, w, metagraph, local_root, subgraph, rules)
    return metagraph, subgraph


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


def source_condition(graph, s):
    return ("NN" in graph.nodes[s]["tag"]) and (graph.nodes[s]["dep"] != "pobj")
    # or "PRP" in graph.nodes[s]["tag"])


def target_condition(graph, t):
    return "NN" in graph.nodes[t]["tag"]
    # and not any([graph.nodes[x]["lower"] == "of" for x in graph.predecessors(t)])
    # )


def parse_first_level_relations(graph):
    """

    :param graph: graph is potentially a metagraph, so each source or target might be split into many

    :return:
    """

    relations = []
    rs = find_relation_candidates(graph)
    source_candidates = [i for i in graph.nodes if source_condition(graph, i)]
    target_candidates = [i for i in graph.nodes if target_condition(graph, i)]
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
    # distance_reverse = {r: nx.shortest_path_length(greverse, r) for r in rs}
    distance_undirected = {r: nx.shortest_path_length(undirected, r) for r in rs}

    dm = pd.DataFrame.from_dict(distance_directed).sort_index(axis=0)
    # rdm = pd.DataFrame.from_dict(distance_reverse).sort_index(axis=0)
    udm = pd.DataFrame.from_dict(distance_undirected).sort_index(axis=0)
    wdm = pd.DataFrame.from_dict(path_weights).sort_index(axis=0)

    dm_targets = dm.loc[target_candidates].copy()
    t_cand = dict()
    s_cand = dict()

    for r in rs:
        min_dist = dm_targets.loc[target_candidates, r].min()
        t_cand[r] = dm_targets.loc[dm_targets[r] == min_dist, r].index.to_list()

    for r in rs:
        # source candidate, close on the graph, negative cost preferred (in reverse direction), penalty if dep is attr
        undirected_to_source = udm.loc[source_candidates, r]
        cost_to_source = wdm.loc[source_candidates, r]
        decision = pd.concat(
            [undirected_to_source.rename("undirected"), cost_to_source.rename("cost")],
            axis=1,
        )
        decision["mcost"] = decision["cost"] + decision["undirected"]

        decision["syn_penalty"] = pd.Series(
            decision.index.map(lambda x: int(graph.nodes[x]["dep"] in ["attr"])),
            index=decision.index,
        )
        decision = decision.sort_values(
            ["mcost", "cost", "undirected", "syn_penalty"],
            ascending=[True, True, True, True],
        )
        top_row = decision.iloc[0]
        mask = (decision == top_row).all(axis=1)
        s_cand[r] = decision[mask].index.tolist()

    from itertools import product

    for r in rs:
        sources = s_cand[r]
        targets = [t for t in t_cand[r] if t not in s_cand[r]]
        for s, t in product(sources, targets):
            relations += [(s, r, t)]
            logger.debug(
                f" {graph.nodes[s]['lower']}, {graph.nodes[r]['lower']}, {graph.nodes[t]['lower']}"
            )
    return relations


def phrase_to_relations(nlp, document, rules):
    _, graph = dep_tree_from_phrase(nlp, document)

    # _, graph = dep_tree_from_phrase(nlp, document)

    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    metas = []
    for root in roots:
        metagraph = nx.DiGraph()
        u, v = -1, root
        metagraph, _ = prune(graph, u, v, metagraph, None, None, rules)
        metas += [metagraph]

        relations = parse_first_level_relations(metagraph)

    def project(x):
        return graph.nodes[x]["lemma"]

    relations_proj = [[project(u) for u in item] for item in relations]
    return graph, relations, relations_proj


def phrase_to_relations_old(nlp, document, rules):
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
