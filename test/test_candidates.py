import pkgutil
import logging
from collections import deque
import yaml
import sys
import os
import unittest
import spacy
from pathlib import Path
import coreferee
from lm_service.preprocessing import normalize_input_text
from lm_service.graph import transform_advcl
from lm_service.graph import dep_tree_from_phrase
from lm_service.onto import (
    Relation,
    ACandidatePile,
    ACandidateKind,
    SourceOrTarget,
)
from lm_service.relation import (
    find_candidates_bfs,
    find_relation_subtree_dfs,
    find_st_subtree_dfs,
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
    nlp.add_pipe("coreferee")

    phrases = normalize_input_text(text, terminal_full_stop=False)
    documents = [
        "The medium was affected by the near-field radiation",
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets,"
        " which will allow the estimation of their mass, density,"
        " composition and their formation.",
    ]

    @unittest.skip("")
    def test_relation_candidates(self):

        piles = []
        for document in self.documents:
            rdoc, graph = dep_tree_from_phrase(self.nlp, document)
            roots = [n for n, d in graph.in_degree() if d == 0]
            rp = ACandidatePile()
            find_candidates_bfs(
                graph, deque(roots), rp, ACandidateKind.RELATION
            )
            piles += [rp]

        self.assertEqual(
            [len(rp) for rp in piles],
            [1, 3],
        )

        self.assertEqual(
            {
                k: [[t.lemma for t in c._tokens] for c in p.candidates]
                for k, p in enumerate(piles)
            },
            {
                0: [["be", "affect", "by"]],
                1: [["be"], ["determine"], ["will", "allow"]],
            },
        )

    @unittest.skip("")
    def test_st_candidates(self):

        piles = []
        for document in self.documents:
            rdoc, graph = dep_tree_from_phrase(self.nlp, document)
            roots = [n for n, d in graph.in_degree() if d == 0]
            rp = ACandidatePile()
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
            [2, 5],
        )

        self.assertEqual(
            {
                k: [[t.lemma for t in c._tokens] for c in p.candidates]
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
            },
        )

    def test_relation_subtree_dfs(self):

        piles = []
        vertices_of_interest = [deque([(3, 0)]), deque([(22, 0)])]
        for deq, document in zip(vertices_of_interest, self.documents):
            rdoc, graph = dep_tree_from_phrase(self.nlp, document)
            cr = Relation()
            find_relation_subtree_dfs(graph, deq, cr)
            cr.sort()
            piles += [cr]

        self.assertEqual(
            [len(rp) for rp in piles],
            [3, 2],
        )

        self.assertEqual(
            {k: p.tokens for k, p in enumerate(piles)},
            {0: [2, 3, 4], 1: [21, 22]},
        )

    def test_st_subtree_dfs(self):
        piles = []
        vertices_of_interest = [deque([(9, 0)]), deque([(24, 0)])]
        for deq, document in zip(vertices_of_interest, self.documents):
            rdoc, graph = dep_tree_from_phrase(self.nlp, document)
            st = SourceOrTarget()
            find_st_subtree_dfs(graph, deq, st, rules=self.rules)
            st.sort()
            piles += [st]

        self.assertEqual(
            [len(rp) for rp in piles],
            [5, 12],
        )

        self.assertEqual(
            {k: p.tokens for k, p in enumerate(piles)},
            {
                0: [5, 6, 7, 8, 9],
                1: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            },
        )

    def test_consecutive_candidates(self):

        sizes = []
        for document in self.documents:
            rdoc, graph = dep_tree_from_phrase(self.nlp, document)
            roots = [n for n, d in graph.in_degree() if d == 0]
            relation_pile = ACandidatePile()
            source_target_pile = ACandidatePile()
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
            [(10, 8, 3), (36, 35, 12)],
        )


if __name__ == "__main__":
    unittest.main()
