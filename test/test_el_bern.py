import pkgutil
import unittest

import spacy
import yaml

from lm_service.linking import link_candidate_entity, normalize_bern_entity
from lm_service.text import normalize_text, phrases_to_basis_triples


class MyTestCase(unittest.TestCase):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    def test_bern(self):
        text = "Diabetic ulcers are related to burns."
        phs = normalize_text(text, self.nlp)

        (
            striples,
            striples_meta,
            candidate_depot,
            relations,
        ) = phrases_to_basis_triples(self.nlp, self.rules, phs)

        ecl = candidate_depot.unfold_conjunction()

        r_bern = {
            "annotations": [
                {
                    "id": ["mesh:D017719"],
                    "is_neural_normalized": True,
                    "mention": "Diabetic ulcers",
                    "obj": "disease",
                    "prob": 0.9999968409538269,
                    "span": {"begin": 0, "end": 15},
                },
                {
                    "id": ["mesh:D002056"],
                    "is_neural_normalized": False,
                    "mention": "burns",
                    "obj": "disease",
                    "prob": 0.9982181191444397,
                    "span": {"begin": 31, "end": 36},
                },
            ],
            "text": "Diabetic ulcers are related to burns.",
            "timestamp": "Tue Sep 20 16:11:48 +0000 2022",
        }

        print(ecl)

        # phrase index set
        ix_current_phrase = 0

        bern_normalized = [
            normalize_bern_entity(item) for item in r_bern["annotations"]
        ]
        map_c2e = link_candidate_entity(
            bern_normalized, ecl, ix_current_phrase
        )

        print(map_c2e)

        # r = query_bern(phrase, "v2")
        # print(r)
        # print(len(r["entities"]))


if __name__ == "__main__":
    unittest.main()
