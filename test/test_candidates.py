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
from lm_service.onto import (
    ACandidateKind,
    Candidate,
    Relation,
    SourceOrTarget,
    Token,
)
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

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
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

    def test_relation_candidates(self):

        piles = []
        for document in self.documents:
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            ograph = graph.copy()
            roots = [n for n, d in graph.in_degree() if d == 0]
            rp = CandidatePile()
            find_candidates_bfs(
                graph, ograph, deque(roots), rp, ACandidateKind.RELATION
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
            ograph = graph.copy()
            roots = [n for n, d in graph.in_degree() if d == 0]
            rp = CandidatePile()
            find_candidates_bfs(
                graph,
                ograph,
                deque(roots),
                rp,
                ACandidateKind.SOURCE_TARGET,
                rules=self.rules,
            )
            piles += [rp.sort_index()]

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
            ograph = graph.copy()
            cr = Relation()
            find_relation_subtree_dfs(graph, ograph, deq, cr)
            cr.clean_dangling_edges().sort_index()
            piles += [cr]

        self.assertEqual(
            [len(rp) for rp in piles],
            [3, 2, 1],
        )

        self.assertEqual(
            {k: p.stokens for k, p in enumerate(piles)},
            {0: ["002", "003", "004"], 1: ["021", "022"], 2: ["001"]},
        )

    def test_st_subtree_dfs(self):
        piles = []
        vertices_of_interest = [9, 24, 2]
        vertices_of_interest = [deque([(x, 0)]) for x in vertices_of_interest]
        for deq, document in zip(vertices_of_interest, self.documents):
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            original_graph = graph.copy()
            st = SourceOrTarget()
            find_sourcetarget_subtree_dfs(
                graph, original_graph, deq, st, rules=self.rules
            )
            st.sort_index()
            piles += [st]

        self.assertEqual(
            [len(rp) for rp in piles],
            [5, 12, 1],
        )

        self.assertEqual(
            {k: p.stokens for k, p in enumerate(piles)},
            {
                0: ["005", "006", "007", "008", "009"],
                1: [
                    "023",
                    "024",
                    "025",
                    "026",
                    "027",
                    "028",
                    "029",
                    "030",
                    "031",
                    "032",
                    "033",
                    "034",
                ],
                2: ["002"],
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
            ograph = graph.copy()
            roots = [n for n, d in graph.in_degree() if d == 0]
            relation_pile = CandidatePile()
            source_target_pile = CandidatePile()
            sa = len(graph.nodes)
            find_candidates_bfs(
                graph,
                ograph,
                deque(roots),
                relation_pile,
                ACandidateKind.RELATION,
            )
            sb = len(graph.nodes)
            find_candidates_bfs(
                graph,
                ograph,
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

    def test_sort_index_tree(self):
        tokens = [
            Token(**{"i": 2, "text": "a0", "successors": {1, 0}}),
            Token(**{"i": 1, "text": "a1", "predecessors": {2}}),
            Token(
                **{
                    "i": 0,
                    "text": "a2",
                    "predecessors": {2},
                    "successors": {15},
                }
            ),
            Token(
                **{
                    "i": 15,
                    "text": "b0",
                    "predecessors": {2},
                    "successors": {16, 17},
                }
            ),
            Token(**{"i": 16, "text": "b1", "predecessors": {15}}),
            Token(**{"i": 17, "text": "b2", "predecessors": {15}}),
        ]
        ac = Candidate().from_tokens(tokens)

        ac.sort_index()

        self.assertEqual(
            ac.stokens, ["000", "015", "016", "017", "001", "002"]
        )


if __name__ == "__main__":
    unittest.main()
