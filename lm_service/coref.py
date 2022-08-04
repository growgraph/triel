from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import networkx as nx
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


class CorefGraph:
    def __init__(
        self, graph: nx.DiGraph, root: int, map_specific: dict[int, int]
    ):
        self.graph: nx.DiGraph = graph
        self.root: int = root
        self.map_specific: dict[int, int] = map_specific


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
        dd: dict[str, Any] = {
            "label": f"{coref_chain}-*-coref-chain",
            "tag_": "coref",
            "dep_": "chain",
            "chain": jchain,
        }
        vs_coref += [(coref_chain, dd)]
        es_coref.append((coref_root, coref_chain))
        jc += 1
        for kth, x in enumerate(chain.mentions):
            coref_blank = jc
            if kth == chain.most_specific_mention_index:
                concept_specific_blank[coref_chain] = coref_blank

            ddd: dict[str, Any] = {
                "label": f"{coref_blank}-*-coref-blank",
                "tag_": "coref",
                "dep_": "blank",
                "chain": jchain,
            }
            vs_coref += [(coref_blank, ddd)]
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
