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
    vs = [
        (
            token.i,
            {
                "i": token.i,
                "dep_": token.dep_,
                "tag_": token.tag_,
                "lower": token.lower_,
                "lemma": token.lemma_,
                "ent_iob": token.ent_iob,
                "text": token.text,
                "label": f"{token.i}-{token.lower_}-{token.dep_}-{token.tag_}",
            },
        )
        for token in rdoc
    ]
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


def transform_advcl(nlp: Language, phrase, debug=False):
    """
    it is assumed there are no fullstops
    :param nlp:
    :param phrase:
    :return:
    """

    rdoc, graph = phrase_to_deptree(nlp, phrase)
    advcls = [
        u
        for u in graph.nodes()
        if graph.nodes[u]["tag_"] == "VBN"
        and graph.nodes[u]["dep_"] == "advcl"
    ]

    for advcl in advcls:
        root = next(iter(graph.predecessors(advcl)))

        succs = sorted(graph.successors(root))
        advcl_index = succs.index(advcl)

        if advcl_index < len(succs) - 1:
            next_ = succs[advcl_index + 1]

            if (
                graph.nodes[next_]["dep_"] == "punct"
                and graph.nodes[next_]["tag_"] == ","
            ):
                if len(list(graph.successors(next_))) != 0:
                    raise Exception(f" `,` punct has successors")
                graph.remove_node(next_)

        # refresh succs just in case: advcl has the same index
        succs = sorted(graph.successors(root))
        jstar = None
        for j in range(advcl_index, len(succs)):
            if graph.nodes[succs[j]]["tag_"] == "NN":
                jstar = j
                break

        if jstar is not None:
            next_ = succs[jstar]

            adv_nodes = sorted(get_subtree_wrapper(graph, advcl))
            next_nodes = sorted(get_subtree_wrapper(graph, next_))
            root_nodes = sorted(get_subtree_wrapper(graph, root))

            if (
                list(range(adv_nodes[0], adv_nodes[0] + len(adv_nodes)))
                != adv_nodes
            ):
                raise Exception(" subtree nodes are not a sequence")
            if (
                list(range(next_nodes[0], next_nodes[0] + len(next_nodes)))
                != next_nodes
            ):
                raise Exception(" subtree nodes are not a sequence")

            i0 = root_nodes.index(adv_nodes[0])
            ilast = root_nodes.index(next_nodes[-1])
            index_remap = root_nodes[i0 : ilast + 1]
            full_index_shifted = (
                index_remap[len(adv_nodes) :] + index_remap[: len(adv_nodes)]
            )
            mapping = dict(zip(full_index_shifted, index_remap))
            if i0 == 0:
                s = graph.nodes[adv_nodes[0]]["text"]
                graph.nodes[adv_nodes[0]]["text"] = s[0].lower() + s[1:]

            graph = nx.relabel_nodes(graph, mapping)

    phrase_rep = [graph.nodes[i]["text"] for i in sorted(graph.nodes)]
    phrase_rep[0] = phrase_rep[0][0].capitalize() + phrase_rep[0][1:]
    transformed_phrase = " ".join(phrase_rep)
    if debug:
        return transformed_phrase, graph
    else:
        return transformed_phrase


def get_subtree_wrapper(graph: nx.DiGraph, v):
    acc: list[Any] = []
    get_subtree(graph, v, acc)
    return acc


def get_subtree(graph: nx.DiGraph, v, acc):
    acc += [v]
    for w in graph.successors(v):
        get_subtree(graph, w, acc)
