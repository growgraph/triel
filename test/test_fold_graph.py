import pkgutil
import yaml
import sys
import os
import unittest
import networkx as nx
import spacy
import logging
from pathlib import Path
from networkx.drawing.nx_agraph import to_agraph
from lm_service.relation import fold_graph

nl_data = {
    "directed": True,
    "multigraph": False,
    "graph": {},
    "nodes": [
        {
            "lower": "cheops",
            "dep": "nsubj",
            "tag": "NNP",
            "lemma": "CHEOPS",
            "label": "0-cheops-nsubj-NNP",
            "id": 0,
        },
        {
            "lower": "(",
            "dep": "punct",
            "tag": "-LRB-",
            "lemma": "(",
            "label": "1-(-punct--LRB-",
            "id": 1,
        },
        {
            "lower": "characterising",
            "dep": "nmod",
            "tag": "NNP",
            "lemma": "CHaracterising",
            "label": "2-characterising-nmod-NNP",
            "id": 2,
        },
        {
            "lower": "exoplanets",
            "dep": "compound",
            "tag": "NNPS",
            "lemma": "ExOPlanets",
            "label": "3-exoplanets-compound-NNPS",
            "id": 3,
        },
        {
            "lower": "satellite",
            "dep": "appos",
            "tag": "NNP",
            "lemma": "Satellite",
            "label": "4-satellite-appos-NNP",
            "id": 4,
        },
        {
            "lower": ")",
            "dep": "punct",
            "tag": "-RRB-",
            "lemma": ")",
            "label": "5-)-punct--RRB-",
            "id": 5,
        },
        {
            "lower": "is",
            "dep": "ROOT",
            "tag": "VBZ",
            "lemma": "be",
            "label": "6-is-ROOT-VBZ",
            "id": 6,
        },
        {
            "lower": "a",
            "dep": "det",
            "tag": "DT",
            "lemma": "a",
            "label": "7-a-det-DT",
            "id": 7,
        },
        {
            "lower": "european",
            "dep": "amod",
            "tag": "JJ",
            "lemma": "european",
            "label": "8-european-amod-JJ",
            "id": 8,
        },
        {
            "lower": "space",
            "dep": "compound",
            "tag": "NN",
            "lemma": "space",
            "label": "9-space-compound-NN",
            "id": 9,
        },
        {
            "lower": "telescope",
            "dep": "attr",
            "tag": "NN",
            "lemma": "telescope",
            "label": "10-telescope-attr-NN",
            "id": 10,
        },
        {
            "lower": "to",
            "dep": "aux",
            "tag": "TO",
            "lemma": "to",
            "label": "11-to-aux-TO",
            "id": 11,
        },
        {
            "lower": "determine",
            "dep": "advcl",
            "tag": "VB",
            "lemma": "determine",
            "label": "12-determine-advcl-VB",
            "id": 12,
        },
        {
            "lower": "the",
            "dep": "det",
            "tag": "DT",
            "lemma": "the",
            "label": "13-the-det-DT",
            "id": 13,
        },
        {
            "lower": "size",
            "dep": "dobj",
            "tag": "NN",
            "lemma": "size",
            "label": "14-size-dobj-NN",
            "id": 14,
        },
        {
            "lower": "of",
            "dep": "prep",
            "tag": "IN",
            "lemma": "of",
            "label": "15-of-prep-IN",
            "id": 15,
        },
        {
            "lower": "known",
            "dep": "amod",
            "tag": "VBN",
            "lemma": "know",
            "label": "16-known-amod-VBN",
            "id": 16,
        },
        {
            "lower": "extrasolar",
            "dep": "amod",
            "tag": "JJ",
            "lemma": "extrasolar",
            "label": "17-extrasolar-amod-JJ",
            "id": 17,
        },
        {
            "lower": "planets",
            "dep": "pobj",
            "tag": "NNS",
            "lemma": "planet",
            "label": "18-planets-pobj-NNS",
            "id": 18,
        },
        {
            "lower": ",",
            "dep": "punct",
            "tag": ",",
            "lemma": ",",
            "label": "19-,-punct-,",
            "id": 19,
        },
        {
            "lower": "which",
            "dep": "nsubj",
            "tag": "WDT",
            "lemma": "which",
            "label": "20-which-nsubj-WDT",
            "id": 20,
        },
        {
            "lower": "will",
            "dep": "aux",
            "tag": "MD",
            "lemma": "will",
            "label": "21-will-aux-MD",
            "id": 21,
        },
        {
            "lower": "allow",
            "dep": "relcl",
            "tag": "VB",
            "lemma": "allow",
            "label": "22-allow-relcl-VB",
            "id": 22,
        },
        {
            "lower": "the",
            "dep": "det",
            "tag": "DT",
            "lemma": "the",
            "label": "23-the-det-DT",
            "id": 23,
        },
        {
            "lower": "estimation",
            "dep": "dobj",
            "tag": "NN",
            "lemma": "estimation",
            "label": "24-estimation-dobj-NN",
            "id": 24,
        },
        {
            "lower": "of",
            "dep": "prep",
            "tag": "IN",
            "lemma": "of",
            "label": "25-of-prep-IN",
            "id": 25,
        },
        {
            "lower": "their",
            "dep": "poss",
            "tag": "PRP$",
            "lemma": "their",
            "label": "26-their-poss-PRP$",
            "id": 26,
        },
        {
            "lower": "mass",
            "dep": "pobj",
            "tag": "NN",
            "lemma": "mass",
            "label": "27-mass-pobj-NN",
            "id": 27,
        },
        {
            "lower": ",",
            "dep": "punct",
            "tag": ",",
            "lemma": ",",
            "label": "28-,-punct-,",
            "id": 28,
        },
        {
            "lower": "density",
            "dep": "conj",
            "tag": "NN",
            "lemma": "density",
            "label": "29-density-conj-NN",
            "id": 29,
        },
        {
            "lower": ",",
            "dep": "punct",
            "tag": ",",
            "lemma": ",",
            "label": "30-,-punct-,",
            "id": 30,
        },
        {
            "lower": "composition",
            "dep": "conj",
            "tag": "NN",
            "lemma": "composition",
            "label": "31-composition-conj-NN",
            "id": 31,
        },
        {
            "lower": "and",
            "dep": "cc",
            "tag": "CC",
            "lemma": "and",
            "label": "32-and-cc-CC",
            "id": 32,
        },
        {
            "lower": "their",
            "dep": "poss",
            "tag": "PRP$",
            "lemma": "their",
            "label": "33-their-poss-PRP$",
            "id": 33,
        },
        {
            "lower": "formation",
            "dep": "conj",
            "tag": "NN",
            "lemma": "formation",
            "label": "34-formation-conj-NN",
            "id": 34,
        },
    ],
    "links": [
        {"source": 0, "target": 1},
        {"source": 0, "target": 4},
        {"source": 0, "target": 5},
        {"source": 4, "target": 2},
        {"source": 4, "target": 3},
        {"source": 6, "target": 0},
        {"source": 6, "target": 10},
        {"source": 6, "target": 12},
        {"source": 10, "target": 7},
        {"source": 10, "target": 8},
        {"source": 10, "target": 9},
        {"source": 12, "target": 11},
        {"source": 12, "target": 14},
        {"source": 14, "target": 13},
        {"source": 14, "target": 15},
        {"source": 15, "target": 18},
        {"source": 18, "target": 16},
        {"source": 18, "target": 17},
        {"source": 18, "target": 19},
        {"source": 18, "target": 22},
        {"source": 22, "target": 20},
        {"source": 22, "target": 21},
        {"source": 22, "target": 24},
        {"source": 24, "target": 23},
        {"source": 24, "target": 25},
        {"source": 25, "target": 27},
        {"source": 27, "target": 26},
        {"source": 27, "target": 28},
        {"source": 27, "target": 29},
        {"source": 29, "target": 30},
        {"source": 29, "target": 31},
        {"source": 31, "target": 32},
        {"source": 31, "target": 34},
        {"source": 34, "target": 33},
    ],
}

graph = nx.node_link_graph(nl_data)


class TestMetagraph(unittest.TestCase):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    def test_fold_graph(self):
        fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
        rules = yaml.load(fp, Loader=yaml.FullLoader)

        metagraph = nx.DiGraph()
        unit_graph = nx.DiGraph()

        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]

        u, v = -1, roots[0]
        metagraph = fold_graph(graph, u, v, metagraph, None, unit_graph, rules)

        dot = to_agraph(metagraph)
        metagraph_name = "test_fold_graph"
        dot.layout("dot")
        dot.draw(path=os.path.join(self.path, f"./figs/{metagraph_name}.png"), format="png", prog="dot")
        dot.draw(path=os.path.join(self.path, f"./figs/{metagraph_name}.pdf"), format="pdf", prog="dot")

        self.assertEqual(len(metagraph.nodes()), 16)
        size_ggs = [1, 1, 1, 1, 1, 1, 1, 4, 2, 7, 1, 2, 6, 2, 2, 2]
        self.assertEqual([len(metagraph.nodes[n]["gg"].nodes()) for n in sorted(metagraph.nodes())], size_ggs)


if __name__ == "__main__":
    unittest.main()
