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
from lm_service.relation import (
    graph_to_relations,
    parse_relations_advanced,
    add_hash,
)
from lm_service.preprocessing import normalize_input_text
from lm_service.graph import transform_advcl
from lm_service.graph import dep_tree_from_phrase
from lm_service.onto import Relation, ACandidatePile
from lm_service.relation import (
    find_relation_candidates_obsolete,
    find_relation_candidates_bfs,
)

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    add_dict_rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    phrases = normalize_input_text(text, terminal_full_stop=False)

    # @unittest.skip("")
    def test_relation_candidates(self):
        documents = [
            "The medium was affected by the radiation",
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass, density,"
            " composition and their formation.",
        ]
        piles = []
        for document in documents:
            rdoc, graph = dep_tree_from_phrase(self.nlp, document)
            roots = [n for n, d in graph.in_degree() if d == 0]
            rp = ACandidatePile()
            find_relation_candidates_bfs(graph, deque(roots), rp)
            piles += [rp]

        self.assertEqual(
            [len(rp) for rp in piles],
            [1, 3],
        )

        self.assertEqual(
            {k: [c.tokens for c in p.candidates] for k, p in enumerate(piles)},
            # piles[0].candidates[0].tokens,
            {0: [[2, 3, 4]], 1: [[6], [12], [21, 22]]},
        )

    @unittest.skip("")
    def test_source_candidates(self):
        document = (
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass, density,"
            " composition and their formation. Launched on 18 December 2019, it"
            " is the first Small-class mission in ESA's Cosmic Vision science"
            " programme."
        )
        rdoc, graph = dep_tree_from_phrase(self.nlp, document)
        rs = find_relation_candidates_obsolete(graph)

        self.assertEqual(
            rs.candidates[0].tokens,
            [2, 3, 4],
        )


if __name__ == "__main__":
    unittest.main()
