import pkgutil
import unittest

import coreferee
import spacy
import yaml

from lm_service.linking import (
    iterate_linking_over_phrases,
    link_candidate_entity,
    normalize_bern_entity,
)
from lm_service.onto import MuIndex
from lm_service.text import normalize_text, phrases_to_basis_triples


class MyTestCase(unittest.TestCase):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    response_bern = {
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

        # phrase index set
        ix_current_phrase = 0

        bern_normalized = [
            normalize_bern_entity(item)
            for item in self.response_bern["annotations"]
        ]

        bern_normalized = dict(
            zip(
                [(ix_current_phrase, j) for j in range(len(bern_normalized))],
                [x for _, x in bern_normalized if x],
            )
        )

        map_c2e = link_candidate_entity(bern_normalized, ecl)

        self.assertEqual(
            map_c2e,
            {
                MuIndex(False, 0, "001", 0): (0, 0),
                MuIndex(False, 0, "005", 0): (0, 1),
            },
        )

    def test_iterate_linking_over_phrases(self):
        foo_link = lambda p: self.response_bern["annotations"]
        text = "Diabetic ulcers are related to burns."
        phrases = normalize_text(text, self.nlp)

        (
            striples,
            striples_meta,
            candidate_depot,
            relations,
        ) = phrases_to_basis_triples(self.nlp, self.rules, phrases)

        ecl = candidate_depot.unfold_conjunction()

        entities_index_e_map = {}
        map_c2e = {}
        entities_index_e_map, map_c2e = iterate_linking_over_phrases(
            phrases=phrases,
            ecl=ecl,
            entities_index_e_map=entities_index_e_map,
            map_c2e=map_c2e,
            foo=foo_link,
        )

        entities_index_e_map_ref, map_c2e_ref = (
            {
                (0, 0): {
                    "linker_type": "bern",
                    "ent_type": "disease",
                    "ent_db_type": "mesh",
                    "id": "D017719",
                    "confidence": 0.9999968409538269,
                    "mention": "Diabetic ulcers",
                },
                (0, 1): {
                    "linker_type": "bern",
                    "ent_type": "disease",
                    "ent_db_type": "mesh",
                    "id": "D002056",
                    "confidence": 0.9982181191444397,
                    "mention": "burns",
                },
            },
            {
                MuIndex(meta=False, phrase=0, token="001", running=0): (0, 0),
                MuIndex(meta=False, phrase=0, token="005", running=0): (0, 1),
            },
        )

        self.assertEqual(entities_index_e_map, entities_index_e_map_ref)
        self.assertEqual(map_c2e, map_c2e_ref)


if __name__ == "__main__":
    unittest.main()
