import logging
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import spacy
from graph_cast.util import ResourceHandler

from lm_service.onto import Candidate, Token

logger = logging.getLogger(__name__)


class TestOps(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    cand_json = ResourceHandler.load("test.data", "candidate_conj.json")
    c = Candidate.from_dict(cand_json)

    nlp = spacy.load("en_core_web_trf")

    def test_replace_with_candidate(self):
        tokens = [
            Token(
                **{
                    "i": 7,
                    "lower": "his",
                    "text": "his",
                    "dep_": "poss",
                    "predecessors": {8},
                }
            ),
            Token(
                **{"i": 8, "lower": "dog", "text": "dog", "successors": {7}}
            ),
        ]
        ac = Candidate().from_tokens(tokens)

        tokens_b = [
            Token(**{"i": 15, "lower": "john", "text": "John"}),
        ]
        bc = Candidate().from_tokens(tokens_b)

        ac.replace_token_with_acandidate("007", bc)
        ac.sort_index()
        self.assertEqual(ac.stokens, ["008", "008a", "015"])
        self.assertEqual(ac.token("008").successors, {"008a"})

    def test_replace_top(self):
        tokens = [
            Token(
                **{
                    "i": 7,
                    "lower": "he",
                    "text": "he",
                }
            ),
        ]
        ac = Candidate().from_tokens(tokens)

        tokens_b = [
            Token(**{"i": 15, "lower": "john", "text": "John"}),
        ]
        bc = Candidate().from_tokens(tokens_b)

        ac.replace_token_with_acandidate("007", bc)
        self.assertEqual(ac._index_vec, ["015"])

    def test_from_subtree(self):
        tokens = [
            Token(**{"i": 0, "text": "a0", "successors": {1, 2}}),
            Token(**{"i": 1, "text": "a1", "predecessors": {0}}),
            Token(
                **{
                    "i": 2,
                    "text": "a2",
                    "predecessors": {0},
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

        bc = ac.from_subtree("002")

        self.assertEqual(bc.stokens, ["002", "015", "016", "017"])

    def test_replace_with_candidate(self):
        root_candidate = (
            Candidate()
            .from_tokens(
                [
                    self.c.token(s)
                    for s in [Token.i2s(i + 23) for i in range(6)]
                ]
            )
            .clean_dangling_edges()
        )
        child_a = (
            Candidate()
            .from_tokens(
                [
                    self.c.token(s)
                    for s in [Token.i2s(i + 31) for i in range(2)]
                ]
            )
            .clean_dangling_edges()
        )
        c_prime = deepcopy(root_candidate)
        c_prime.replace_token_with_acandidate(i="027", ac=child_a)

        self.assertEqual(c_prime.token("025").successors, {"031"})


if __name__ == "__main__":
    unittest.main()
