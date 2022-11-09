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

    cand_json = {
        "r0": 6,
        "_tokens": {
            "024": {
                "text": "estimation",
                "s": "024",
                "dep_": "dobj",
                "tag_": "NN",
                "lower": "estimation",
                "lemma": "estimation",
                "ent_iob": 2,
                "_level": 0,
                "label": "24-estimation-dobj-NN",
                "predecessors": [],
                "successors": ["025", "023"],
            },
            "023": {
                "text": "the",
                "s": "023",
                "dep_": "det",
                "tag_": "DT",
                "lower": "the",
                "lemma": "the",
                "ent_iob": 2,
                "_level": 1,
                "label": "23-the-det-DT",
                "predecessors": ["024"],
                "successors": [],
            },
            "025": {
                "text": "of",
                "s": "025",
                "dep_": "prep",
                "tag_": "IN",
                "lower": "of",
                "lemma": "of",
                "ent_iob": 2,
                "_level": 1,
                "label": "25-of-prep-IN",
                "predecessors": ["024"],
                "successors": ["027"],
            },
            "027": {
                "text": "mass",
                "s": "027",
                "dep_": "pobj",
                "tag_": "NN",
                "lower": "mass",
                "lemma": "mass",
                "ent_iob": 2,
                "_level": 2,
                "label": "27-mass-pobj-NN",
                "predecessors": ["025"],
                "successors": ["028", "026", "029"],
            },
            "026": {
                "text": "their",
                "s": "026",
                "dep_": "poss",
                "tag_": "PRP$",
                "lower": "their",
                "lemma": "their",
                "ent_iob": 2,
                "_level": 3,
                "label": "26-their-poss-PRP$",
                "predecessors": ["027"],
                "successors": [],
            },
            "028": {
                "text": ",",
                "s": "028",
                "dep_": "punct",
                "tag_": ",",
                "lower": ",",
                "lemma": ",",
                "ent_iob": 2,
                "_level": 3,
                "label": "28-,-punct-,",
                "predecessors": ["027"],
                "successors": [],
            },
            "029": {
                "text": "density",
                "s": "029",
                "dep_": "conj",
                "tag_": "NN",
                "lower": "density",
                "lemma": "density",
                "ent_iob": 2,
                "_level": 3,
                "label": "29-density-conj-NN",
                "predecessors": ["027"],
                "successors": ["030", "031"],
            },
            "030": {
                "text": ",",
                "s": "030",
                "dep_": "punct",
                "tag_": ",",
                "lower": ",",
                "lemma": ",",
                "ent_iob": 2,
                "_level": 4,
                "label": "30-,-punct-,",
                "predecessors": ["029"],
                "successors": [],
            },
            "031": {
                "text": "composition",
                "s": "031",
                "dep_": "conj",
                "tag_": "NN",
                "lower": "composition",
                "lemma": "composition",
                "ent_iob": 2,
                "_level": 4,
                "label": "31-composition-conj-NN",
                "predecessors": ["029"],
                "successors": ["034", "032"],
            },
            "032": {
                "text": "and",
                "s": "032",
                "dep_": "cc",
                "tag_": "CC",
                "lower": "and",
                "lemma": "and",
                "ent_iob": 2,
                "_level": 5,
                "label": "32-and-cc-CC",
                "predecessors": ["031"],
                "successors": [],
            },
            "034": {
                "s": 34,
                "text": "formation",
                "s": "034",
                "dep_": "conj",
                "tag_": "NN",
                "lower": "formation",
                "lemma": "formation",
                "ent_iob": 2,
                "_level": 5,
                "label": "34-formation-conj-NN",
                "predecessors": ["031"],
                "successors": ["033"],
            },
            "033": {
                "text": "their",
                "s": "033",
                "dep_": "poss",
                "tag_": "PRP$",
                "lower": "their",
                "lemma": "their",
                "ent_iob": 2,
                "_level": 6,
                "label": "33-their-poss-PRP$",
                "predecessors": ["034"],
                "successors": [],
            },
        },
        "_indexVec": [
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
        "_root": "024",
    }

    c = Candidate.from_dict(cand_json)

    nlp = spacy.load("en_core_web_trf")

    def test_replace_with_candidate(self):
        tokens = [
            Token(
                **{
                    "s": 7,
                    "lower": "his",
                    "text": "his",
                    "dep_": "poss",
                    "predecessors": {8},
                }
            ),
            Token(
                **{"s": 8, "lower": "dog", "text": "dog", "successors": {7}}
            ),
        ]
        ac = Candidate().from_tokens(tokens)

        tokens_b = [
            Token(**{"s": 15, "lower": "john", "text": "John"}),
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
                    "s": 7,
                    "lower": "he",
                    "text": "he",
                }
            ),
        ]
        ac = Candidate().from_tokens(tokens)

        tokens_b = [
            Token(**{"s": 15, "lower": "john", "text": "John"}),
        ]
        bc = Candidate().from_tokens(tokens_b)

        ac.replace_token_with_acandidate("007", bc)
        self.assertEqual(ac._index_vec, ["015"])

    def test_from_subtree(self):
        tokens = [
            Token(**{"s": 0, "text": "a0", "successors": {1, 2}}),
            Token(**{"s": 1, "text": "a1", "predecessors": {0}}),
            Token(
                **{
                    "s": 2,
                    "text": "a2",
                    "predecessors": {0},
                    "successors": {15},
                }
            ),
            Token(
                **{
                    "s": 15,
                    "text": "b0",
                    "predecessors": {2},
                    "successors": {16, 17},
                }
            ),
            Token(**{"s": 16, "text": "b1", "predecessors": {15}}),
            Token(**{"s": 17, "text": "b2", "predecessors": {15}}),
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
