import pkgutil
import unittest

import coreferee
import spacy
import yaml

from lm_service.top import text_to_rel_graph, to_dict


class TestREL(unittest.TestCase):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    def test_iterate_linking_bern(self):
        text = "Diabetic ulcers are related to burns."
        response = text_to_rel_graph(text, self.nlp, self.rules)

        rd = to_dict(response)

        rd_ref = {
            "triples": {"1|0|0|0": ["0|0|1|0", "0|0|2|9", "0|0|5|0"]},
            "eindex_entity": {
                "BERN_V2/mesh/D017719": {
                    "linker_type": "BERN_V2",
                    "ent_db_type": "mesh",
                    "id": "D017719",
                    "hash": "BERN_V2/mesh/D017719",
                    "ent_type": "disease",
                },
                "BERN_V2/mesh/D002056": {
                    "linker_type": "BERN_V2",
                    "ent_db_type": "mesh",
                    "id": "D002056",
                    "hash": "BERN_V2/mesh/D002056",
                    "ent_type": "disease",
                },
                "SPACY_NAIVE_WIKI/wikidata/Q6452285": {
                    "linker_type": "SPACY_NAIVE_WIKI",
                    "ent_db_type": "wikidata",
                    "id": "Q6452285",
                    "hash": "SPACY_NAIVE_WIKI/wikidata/Q6452285",
                    "original_form": "ulcer",
                    "description": "type of cutaneous condition",
                },
                "SPACY_NAIVE_WIKI/wikidata/Q170518": {
                    "linker_type": "SPACY_NAIVE_WIKI",
                    "ent_db_type": "wikidata",
                    "id": "Q170518",
                    "hash": "SPACY_NAIVE_WIKI/wikidata/Q170518",
                    "original_form": "burns",
                    "description": (
                        "injury to flesh or skin, often caused by excessive"
                        " heat"
                    ),
                },
                "LOCAL_NON_EL/ent_db_type_local_gg/44afc2df2816ef50ecd4f847": {
                    "linker_type": "LOCAL_NON_EL",
                    "ent_db_type": "ent_db_type_local_gg",
                    "id": "44afc2df2816ef50ecd4f847",
                    "hash": "LOCAL_NON_EL/ent_db_type_local_gg/44afc2df2816ef50ecd4f847",
                    "original_form": "is related to",
                },
            },
            "muindex_eindex": [
                ["0|0|1|0", "BERN_V2/mesh/D017719"],
                ["0|0|5|0", "BERN_V2/mesh/D002056"],
                ["0|0|1|0", "SPACY_NAIVE_WIKI/wikidata/Q6452285"],
                ["0|0|5|0", "SPACY_NAIVE_WIKI/wikidata/Q170518"],
                [
                    "0|0|2|9",
                    "LOCAL_NON_EL/ent_db_type_local_gg/44afc2df2816ef50ecd4f847",
                ],
            ],
            "muindex_candidate": {
                "0|0|1|0": {
                    "hash": "f531ac82d224ed6fe5bb5487",
                    "text": "diabetic ulcers",
                },
                "0|0|2|9": {
                    "hash": "44afc2df2816ef50ecd4f847",
                    "text": "is related to",
                },
                "0|0|5|0": {
                    "hash": "0e037ba9e99471d472ef9edc",
                    "text": "burns",
                },
            },
        }
        self.assertEqual(rd, rd_ref)


if __name__ == "__main__":
    unittest.main()
