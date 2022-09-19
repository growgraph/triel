import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path

import coreferee
import networkx as nx
import spacy
import yaml

from lm_service.coref import (
    graph_component_maps,
    render_coref_graph,
    render_coref_maps_wrapper,
)
from lm_service.graph import phrase_to_deptree, relabel_nodes_and_key
from lm_service.preprocessing import normalize_input_text, transform_advcl
from lm_service.relation import graph_to_candidate_pile

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    phrases = normalize_input_text(text, terminal_full_stop=True)

    def test_indexing(self):
        fragment = self.phrases[:2]
        fragment = [transform_advcl(self.nlp, p) for p in fragment]

        fragment_joined = " ".join(fragment)
        graphs = {}
        for j, doc in enumerate(fragment):
            rdoc, graph = phrase_to_deptree(self.nlp, doc)
            graphs[j] = graph

        rdoc, graph = phrase_to_deptree(self.nlp, fragment_joined)
        map_tree_subtree_index = graph_component_maps(graph)

        graph_relabeled = relabel_nodes_and_key(
            graph, map_tree_subtree_index, "s"
        )
        for j, graph in graphs.items():
            for n in graph.nodes:
                self.assertEqual(
                    graph.nodes[n]["lemma"],
                    graph_relabeled.nodes[(j, n)]["lemma"],
                )


if __name__ == "__main__":
    unittest.main()
