import pkgutil
import unittest

import coreferee
import spacy
import yaml

from lm_service.linking import EntityLinker
from lm_service.onto import MuIndex
from lm_service.top import text_to_rel_graph


class TestREL(unittest.TestCase):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    def test_iterate_linking_bern(self):
        text = "Diabetic ulcers are related to burns."
        map_eindex_entity, map_c2e, _ = text_to_rel_graph(
            text, self.nlp, self.rules
        )

        map_eindex_entity_ref, map_c2e_ref = (
            {
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
            [
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    "BERN_V2/mesh/D017719",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="005", running=0),
                    "BERN_V2/mesh/D002056",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    "SPACY_NAIVE_WIKI/wikidata/Q6452285",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="005", running=0),
                    "SPACY_NAIVE_WIKI/wikidata/Q170518",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="002", running=9),
                    "LOCAL_NON_EL/ent_db_type_local_gg/44afc2df2816ef50ecd4f847",
                ),
            ],
        )

        self.assertEqual(map_eindex_entity, map_eindex_entity_ref)
        self.assertEqual(map_c2e, map_c2e_ref)


if __name__ == "__main__":
    unittest.main()
