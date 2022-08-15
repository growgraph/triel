import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path

import spacy
import yaml

from lm_service.graph import phrase_to_deptree, transform_advcl
from lm_service.onto import Candidate, CandidatePile, SourceOrTarget, Token
from lm_service.relation import (
    graph_to_candidate_pile,
    partition_conjunctive_dfs,
    partition_conjunctive_wrapper,
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

    conjunctive_target = {
        "r0": 4,
        "_tokens": {
            24: {
                "i": 24,
                "text": "estimation",
                "dep_": "dobj",
                "tag_": "NN",
                "lower": "estimation",
                "lemma": "estimation",
                "ent_iob": 2,
                "_level": 0,
                "label": "24-estimation-dobj-NN",
            },
            25: {
                "i": 25,
                "text": "of",
                "dep_": "prep",
                "tag_": "IN",
                "lower": "of",
                "lemma": "of",
                "ent_iob": 2,
                "_level": 1,
                "label": "25-of-prep-IN",
            },
            27: {
                "i": 27,
                "text": "mass",
                "dep_": "pobj",
                "tag_": "NN",
                "lower": "mass",
                "lemma": "mass",
                "ent_iob": 2,
                "_level": 2,
                "label": "27-mass-pobj-NN",
            },
            26: {
                "i": 26,
                "text": "their",
                "dep_": "poss",
                "tag_": "PRP$",
                "lower": "their",
                "lemma": "their",
                "ent_iob": 2,
                "_level": 3,
                "label": "26-their-poss-PRP$",
            },
            28: {
                "i": 28,
                "text": ",",
                "dep_": "punct",
                "tag_": ",",
                "lower": ",",
                "lemma": ",",
                "ent_iob": 2,
                "_level": 3,
                "label": "28-,-punct-,",
            },
            29: {
                "i": 29,
                "text": "density",
                "dep_": "conj",
                "tag_": "NN",
                "lower": "density",
                "lemma": "density",
                "ent_iob": 2,
                "_level": 3,
                "label": "29-density-conj-NN",
            },
            30: {
                "i": 30,
                "text": ",",
                "dep_": "punct",
                "tag_": ",",
                "lower": ",",
                "lemma": ",",
                "ent_iob": 2,
                "_level": 4,
                "label": "30-,-punct-,",
            },
            31: {
                "i": 31,
                "text": "composition",
                "dep_": "conj",
                "tag_": "NN",
                "lower": "composition",
                "lemma": "composition",
                "ent_iob": 2,
                "_level": 4,
                "label": "31-composition-conj-NN",
            },
            32: {
                "i": 32,
                "text": "and",
                "dep_": "cc",
                "tag_": "CC",
                "lower": "and",
                "lemma": "and",
                "ent_iob": 2,
                "_level": 5,
                "label": "32-and-cc-CC",
            },
            34: {
                "i": 34,
                "text": "formation",
                "dep_": "conj",
                "tag_": "NN",
                "lower": "formation",
                "lemma": "formation",
                "ent_iob": 2,
                "_level": 5,
                "label": "34-formation-conj-NN",
            },
            33: {
                "i": 33,
                "text": "their",
                "dep_": "poss",
                "tag_": "PRP$",
                "lower": "their",
                "lemma": "their",
                "ent_iob": 2,
                "_level": 6,
                "label": "33-their-poss-PRP$",
            },
        },
        "_indexSet": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        "_root": 24,
    }

    documents = {
        "cheops0": "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets,"
        " which will allow the estimation of their mass, density,"
        " composition and their formation.",
        "coref": "Although he was very busy with his work, Peter Brown had had enough of it. "
        "He and his wife decided they needed a holiday. "
        "They travelled to Spain because they loved the country very much.",
    }

    def test_acandidate_insert_end(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x+3}"}) for x in range(3)]
        ac = Candidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac.insert_at(4, tokens_to_add)
        self.assertEqual(ac._index_set, [3, 4, 5, 15, 17])

    def test_acandidate_insert(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x+3}"}) for x in range(3)]
        ac = Candidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac.insert_at(1, tokens_to_add)
        self.assertEqual(ac._index_set, [3, 15, 17, 4, 5])

    def test_acandidate_insert_with_token_index(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x + 3}"}) for x in range(3)]
        ac = Candidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac.insert_at(5, tokens_to_add, token_index=True)
        self.assertEqual(ac._index_set, [3, 4, 15, 17, 5])

    def test_acandidate_replace(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x + 3}"}) for x in range(3)]
        ac = Candidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac.replace_token_with_tokens(4, tokens_to_add)
        self.assertEqual(ac._index_set, [3, 15, 17, 5])

    def test_acandidate_replace_acandidate(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x + 3}"}) for x in range(3)]
        ac = Candidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac2 = Candidate()
        for t in tokens_to_add:
            ac2.append(t)

        ac.replace_token_with_acandidate(4, ac2)
        self.assertEqual(ac._index_set, [3, 15, 17, 5])

    def test_split_conj(self):
        lens = dict()
        for key in ["coref", "cheops0"]:
            lens[key] = {}
            apile = CandidatePile()
            rdoc, graph = phrase_to_deptree(self.nlp, self.documents[key])
            pile, mgraph = graph_to_candidate_pile(graph, self.rules)
            lens[key]["was"] = len(pile.sources)

            for c in pile.sources:
                accum = partition_conjunctive_wrapper(c, graph)
                accum = (
                    accum.sort_index().drop_punct().drop_cc().drop_articles()
                )
                apile += accum
            text = apile.project_to_text()
            lens[key]["became"] = len(apile)
            print(text)
        self.assertEqual(
            lens,
            {
                "coref": {"was": 9, "became": 10},
                "cheops0": {"was": 5, "became": 8},
            },
        )


if __name__ == "__main__":
    unittest.main()
