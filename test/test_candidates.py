"""
test candidate extraction
"""

import logging
import os
import pkgutil
import sys
import unittest
from collections import deque
from pathlib import Path

import spacy
import yaml

from lm_service.graph import phrase_to_deptree
from lm_service.onto import ACandidateKind, Relation, SourceOrTarget
from lm_service.piles import CandidatePile
from lm_service.preprocessing import normalize_input_text
from lm_service.relation import (
    find_candidates_bfs,
    find_relation_subtree_dfs,
    find_sourcetarget_subtree_dfs,
)

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")

    phrases = normalize_input_text(text, terminal_full_stop=False)
    documents = [
        "The medium was affected by the near-field radiation",
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets,"
        " which will allow the estimation of their mass, density,"
        " composition and their formation.",
        "He treated her unfairly.",
    ]

    # @unittest.skip("")
    def test_relation_candidates(self):

        piles = []
        for document in self.documents:
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            roots = [n for n, d in graph.in_degree() if d == 0]
            rp = CandidatePile()
            find_candidates_bfs(
                graph, deque(roots), rp, ACandidateKind.RELATION
            )
            piles += [rp]

        self.assertEqual(
            [len(rp) for rp in piles],
            [1, 3, 1],
        )

        self.assertEqual(
            {
                k: [c.lemmas for c in p._candidates]
                for k, p in enumerate(piles)
            },
            {
                0: [["be", "affect", "by"]],
                1: [["be"], ["determine"], ["will", "allow"]],
                2: [["treat"]],
            },
        )

    def test_st_candidates(self):

        piles = []
        for document in self.documents:
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            roots = [n for n, d in graph.in_degree() if d == 0]
            rp = CandidatePile()
            find_candidates_bfs(
                graph,
                deque(roots),
                rp,
                ACandidateKind.SOURCE_TARGET,
                rules=self.rules,
            )
            piles += [rp]

        self.assertEqual(
            [len(rp) for rp in piles],
            [2, 5, 2],
        )

        self.assertEqual(
            {
                k: [c.lemmas for c in p._candidates]
                for k, p in enumerate(piles)
            },
            {
                0: [
                    ["the", "medium"],
                    ["the", "near", "-", "field", "radiation"],
                ],
                1: [
                    ["CHEOPS", "(", ")"],
                    ["a", "european", "space", "telescope"],
                    ["CHaracterising", "ExOPlanets", "Satellite"],
                    ["the", "size", "of", "know", "extrasolar", "planet"],
                    [
                        "the",
                        "estimation",
                        "of",
                        "their",
                        "mass",
                        ",",
                        "density",
                        ",",
                        "composition",
                        "and",
                        "their",
                        "formation",
                    ],
                ],
                2: [["he"], ["she"]],
            },
        )

    def test_relation_subtree_dfs(self):

        piles = []
        vertices_of_interest = [3, 22, 1]
        vertices_of_interest = [deque([(x, 0)]) for x in vertices_of_interest]
        for deq, document in zip(vertices_of_interest, self.documents):
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            cr = Relation()
            find_relation_subtree_dfs(graph, deq, cr)
            cr.sort_index()
            piles += [cr]

        self.assertEqual(
            [len(rp) for rp in piles],
            [3, 2, 1],
        )

        self.assertEqual(
            {k: p.itokens for k, p in enumerate(piles)},
            {0: [2, 3, 4], 1: [21, 22], 2: [1]},
        )

    def test_st_subtree_dfs(self):
        piles = []
        vertices_of_interest = [9, 24, 2]
        vertices_of_interest = [deque([(x, 0)]) for x in vertices_of_interest]
        for deq, document in zip(vertices_of_interest, self.documents):
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            st = SourceOrTarget()
            find_sourcetarget_subtree_dfs(graph, deq, st, rules=self.rules)
            st.sort_index()
            piles += [st]

        self.assertEqual(
            [len(rp) for rp in piles],
            [5, 12, 1],
        )

        self.assertEqual(
            {k: p.itokens for k, p in enumerate(piles)},
            {
                0: [5, 6, 7, 8, 9],
                1: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                2: [2],
            },
        )

    def test_consecutive_candidates(self):
        """
        the essence of this test is to demonstrate that when a relation or a source/target are identified,
        the subgraph corresponding to it is excised, leaving only the root node
        :return:
        """

        sizes = []
        for document in self.documents:
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            roots = [n for n, d in graph.in_degree() if d == 0]
            relation_pile = CandidatePile()
            source_target_pile = CandidatePile()
            sa = len(graph.nodes)
            find_candidates_bfs(
                graph,
                deque(roots),
                relation_pile,
                ACandidateKind.RELATION,
            )
            sb = len(graph.nodes)
            find_candidates_bfs(
                graph,
                deque(roots),
                source_target_pile,
                ACandidateKind.SOURCE_TARGET,
                rules=self.rules,
            )
            sc = len(graph.nodes)
            sizes += [(sa, sb, sc)]
        self.assertEqual(
            sizes,
            [(10, 8, 3), (36, 35, 12), (5, 5, 5)],
        )


if __name__ == "__main__":
    unittest.main()
