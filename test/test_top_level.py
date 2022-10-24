import pkgutil
import unittest
from pprint import pprint

import coreferee
import spacy
import yaml

from lm_service.top import cast_response_to_unfolded, text_to_rel_graph


class TestREL(unittest.TestCase):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    def test_iterate_linking_bern(self):
        text = "Diabetic ulcers are related to burns."
        # text = (
        #     "Thousands of exoplanets have been discovered by the end of the"
        #     " 2010s; some have minimum mass measurements from the radial"
        #     " velocity method while others that are seen to transit their"
        #     " parent stars have measures of their physical size."
        # )
        response = text_to_rel_graph(text, self.nlp, self.rules)
        response_jsonlike = cast_response_to_unfolded(
            response, cast_triple_version="v1"
        )
        rj_ref = {
            "triples": [
                {
                    "triple_index": {
                        "hash": "36dc4e2a7d2eae685d047bca0bc144bc9d95049b"
                    },
                    "triple": [
                        {
                            "hash": "c621319201d349cb4f42aabedbea2c73a1419b98",
                            "text": "diabetic ulcers",
                            "role": "source",
                        },
                        {
                            "hash": "dda96135ac461d989729db27e63bdf3f88b724e3",
                            "text": "is related to",
                            "role": "relation",
                        },
                        {
                            "hash": "0547fec2c8f9153e2ea5619090155c87fddf806b",
                            "text": "burns",
                            "role": "target",
                        },
                    ],
                }
            ],
            "map_mention_entity": [
                {
                    "mention": {
                        "hash": "c621319201d349cb4f42aabedbea2c73a1419b98",
                        "text": "diabetic ulcers",
                        "role": "source",
                    },
                    "entity": {
                        "linker_type": "BERN_V2",
                        "ent_db_type": "mesh",
                        "id": "D017719",
                        "hash": "BERN_V2/mesh/D017719",
                        "ent_type": "disease",
                    },
                },
                {
                    "mention": {
                        "hash": "0547fec2c8f9153e2ea5619090155c87fddf806b",
                        "text": "burns",
                        "role": "target",
                    },
                    "entity": {
                        "linker_type": "BERN_V2",
                        "ent_db_type": "mesh",
                        "id": "D002056",
                        "hash": "BERN_V2/mesh/D002056",
                        "ent_type": "disease",
                    },
                },
                {
                    "mention": {
                        "hash": "c621319201d349cb4f42aabedbea2c73a1419b98",
                        "text": "diabetic ulcers",
                        "role": "source",
                    },
                    "entity": {
                        "linker_type": "SPACY_NAIVE_WIKI",
                        "ent_db_type": "wikidata",
                        "id": "Q6452285",
                        "hash": "SPACY_NAIVE_WIKI/wikidata/Q6452285",
                        "original_form": "ulcer",
                        "description": "type of cutaneous condition",
                    },
                },
                {
                    "mention": {
                        "hash": "0547fec2c8f9153e2ea5619090155c87fddf806b",
                        "text": "burns",
                        "role": "target",
                    },
                    "entity": {
                        "linker_type": "SPACY_NAIVE_WIKI",
                        "ent_db_type": "wikidata",
                        "id": "Q170518",
                        "hash": "SPACY_NAIVE_WIKI/wikidata/Q170518",
                        "original_form": "burns",
                        "description": (
                            "injury to flesh or skin, often caused by"
                            " excessive heat"
                        ),
                    },
                },
                {
                    "mention": {
                        "hash": "dda96135ac461d989729db27e63bdf3f88b724e3",
                        "text": "is related to",
                        "role": "relation",
                    },
                    "entity": {
                        "linker_type": "LOCAL_NON_EL",
                        "ent_db_type": "ent_db_type_local_gg",
                        "id": "dda96135ac461d989729db27e63bdf3f88b724e3",
                        "hash": "LOCAL_NON_EL/ent_db_type_local_gg/dda96135ac461d989729db27e63bdf3f88b724e3",
                        "original_form": "is related to",
                    },
                },
            ],
        }
        for k in response_jsonlike:
            item = response_jsonlike[k]
            ref_item = rj_ref[k]
            for x, y in zip(item, ref_item):
                if x != y:
                    pprint(x)
                    pprint(y)
                self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
