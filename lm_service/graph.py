from __future__ import annotations

from itertools import product
from typing import Any

import networkx as nx
from spacy import Language
from spacy.tokens import Doc


def excise_node(graph, u):
    graph.add_edges_from(product(graph.predecessors(u), graph.successors(u)))
    graph.remove_node(u)


def phrase_to_deptree(nlp: Language, document: str) -> tuple[Doc, nx.DiGraph]:
    """
    given nlp and a phrase (string) - yield spacy doc and a digraph representing syn parsing
    :param nlp:
    :param document:
    :return:
    """
    graph = nx.DiGraph()

    rdoc = nlp(document)

    keys_to_pick = [
        "i",
        "dep_",
        "tag_",
        "lower_",
        "lemma_",
        "text",
        "ent_iob",
        "idx",
    ]

    map_keys = {"i": "s", "lower_": "lower", "lemma_": "lemma"}

    vs = [
        (
            token.i,
            {
                map_keys[k] if k in map_keys else k: token.__getattribute__(k)
                for k in keys_to_pick
            },
        )
        for token in rdoc
    ]

    # add label
    for i, v in vs:
        v["label"] = f"{v['s']}-{v['lower']}-{v['dep_']}-{v['tag_']}"

    # root = [v[0] for v in vs if v[1]["dep_"] == "ROOT"][0]
    # FYI https://www.spacy.io/docs/api/token
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    es = []
    for token in rdoc:
        for child in token.children:
            es.append((token.i, child.i))

    graph.add_nodes_from(vs)
    graph.add_edges_from(es)

    return rdoc, graph


def get_subtree_wrapper(graph: nx.DiGraph, v):
    acc: list[Any] = []
    get_subtree(graph, v, acc)
    return acc


def get_subtree(graph: nx.DiGraph, v, acc):
    acc += [v]
    for w in graph.successors(v):
        get_subtree(graph, w, acc)


def relabel_nodes_and_key(g, map_tree_subtree_index, key="s"):
    nx.set_node_attributes(g, map_tree_subtree_index, name=key)
    graph_relabeled = nx.relabel_nodes(g, map_tree_subtree_index)
    return graph_relabeled
