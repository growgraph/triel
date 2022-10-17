import pkgutil
import unittest

import coreferee
import spacy
import yaml

from lm_service.linking import (
    EntityLinker,
    iterate_linking_over_phrases,
    iterate_over_linkers,
    link_candidate_entity,
    normalize_bern_entity,
    phrase_to_spacy_basic_entities,
)
from lm_service.onto import MuIndex
from lm_service.text import (
    normalize_text,
    phrases_to_basis_triples,
    phrases_to_triples,
)


class TestEL(unittest.TestCase):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    response_bern = {
        "annotations": [
            {
                "id": ["mesh:D017719"],
                "is_neural_normalized": True,
                "obj": "disease",
                "span": {"begin": 0, "end": 15},
            },
            {
                "id": ["mesh:D002056"],
                "is_neural_normalized": False,
                "obj": "disease",
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
            _,
        ) = phrases_to_basis_triples(self.nlp, self.rules, phs)

        ecl = candidate_depot.unfold_conjunction()

        # phrase index set

        bern_normalized = [
            normalize_bern_entity(item)
            for item in self.response_bern["annotations"]
        ]

        bern_normalized = {e.hash: span for e, span in bern_normalized}

        map_c2e = link_candidate_entity(bern_normalized, ecl, ix_phrases=(0,))

        self.assertEqual(
            map_c2e,
            [
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    "BERN_V2/mesh/D017719",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="005", running=0),
                    "BERN_V2/mesh/D002056",
                ),
            ],
        )

    def test_iterate_linking_bern(self):
        foo_link = lambda p: self.response_bern["annotations"]
        text = "Diabetic ulcers are related to burns."
        phrases = normalize_text(text, self.nlp)

        (
            striples,
            striples_meta,
            candidate_depot,
            relations,
            _,
        ) = phrases_to_basis_triples(self.nlp, self.rules, phrases)

        ecl = candidate_depot.unfold_conjunction()

        map_eindex_entity = {}
        map_c2e = []
        map_eindex_entity, map_c2e = iterate_linking_over_phrases(
            phrases=phrases,
            ecl=ecl,
            map_eindex_entity=map_eindex_entity,
            map_c2e=map_c2e,
            link_foo=foo_link,
        )

        map_eindex_entity_str = {
            k: v.to_dict(skip_defaults=True)
            for k, v in map_eindex_entity.items()
        }

        map_eindex_entity_str_ref, map_c2e_ref = (
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
            ],
        )

        self.assertEqual(map_eindex_entity_str, map_eindex_entity_str_ref)
        self.assertEqual(map_c2e, map_c2e_ref)

    def test_iterate_naive_wiki_linking(self):

        text = "Diabetic ulcers are related to burns."

        phrases = normalize_text(text, self.nlp)

        if "entityLinker" not in self.nlp.pipe_names:
            self.nlp.add_pipe("entityLinker", last=True)

        foo_link = lambda p: self.nlp(p)._.linkedEntities

        (
            striples,
            striples_meta,
            candidate_depot,
            relations,
            _,
        ) = phrases_to_basis_triples(self.nlp, self.rules, phrases)

        ecl = candidate_depot.unfold_conjunction()

        map_eindex_entity = {}
        map_c2e = []
        map_eindex_entity, map_c2e = iterate_linking_over_phrases(
            phrases=phrases,
            ecl=ecl,
            map_eindex_entity=map_eindex_entity,
            map_c2e=map_c2e,
            link_foo=foo_link,
            etype=EntityLinker.SPACY_NAIVE_WIKI,
        )

        map_eindex_entity_str = {
            k: v.to_dict(skip_defaults=True)
            for k, v in map_eindex_entity.items()
        }

        map_eindex_entity_ref, map_c2e_ref = (
            {
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
            },
            [
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    "SPACY_NAIVE_WIKI/wikidata/Q6452285",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="005", running=0),
                    "SPACY_NAIVE_WIKI/wikidata/Q170518",
                ),
            ],
        )

        self.assertEqual(map_eindex_entity_str, map_eindex_entity_ref)
        self.assertEqual(map_c2e, map_c2e_ref)

    def test_iterate_spacy(self):

        text = (
            "Cheops ( CHaracterising ExOPlanets Satellite ) is a European"
            " space telescope to determine the size of known extrasolar"
            " planets , which will allow the estimation of their mass ,"
            " density , composition and their formation."
        )

        phrases = normalize_text(text, self.nlp)

        (
            striples,
            striples_meta,
            candidate_depot,
            relations,
            _,
        ) = phrases_to_basis_triples(self.nlp, self.rules, phrases)

        ecl = candidate_depot.unfold_conjunction()

        map_eindex_entity = {}
        map_c2e = []

        map_eindex_entity, map_c2e = iterate_linking_over_phrases(
            phrases=phrases,
            ecl=ecl,
            map_eindex_entity=map_eindex_entity,
            map_c2e=map_c2e,
            link_foo=phrase_to_spacy_basic_entities,
            link_foo_kwargs={"nlp": self.nlp},
            etype=EntityLinker.SPACY_BASIC,
        )

        map_eindex_entity_str = {
            k: v.to_dict(skip_defaults=True)
            for k, v in map_eindex_entity.items()
        }

        entities_index_e_map_ref, map_c2e_ref = (
            {
                "SPACY_BASIC/basic/92298dabd25734eab4386b6a": {
                    "linker_type": "SPACY_BASIC",
                    "ent_db_type": "basic",
                    "id": "92298dabd25734eab4386b6a",
                    "hash": "SPACY_BASIC/basic/92298dabd25734eab4386b6a",
                    "ent_type": "386",
                    "original_form": "cheops",
                },
                "SPACY_BASIC/basic/636a58bcdfd3c8e3450c0bbe": {
                    "linker_type": "SPACY_BASIC",
                    "ent_db_type": "basic",
                    "id": "636a58bcdfd3c8e3450c0bbe",
                    "hash": "SPACY_BASIC/basic/636a58bcdfd3c8e3450c0bbe",
                    "ent_type": "381",
                    "original_form": "european",
                },
            },
            [
                (
                    MuIndex(meta=False, phrase=0, token="000", running=0),
                    "SPACY_BASIC/basic/92298dabd25734eab4386b6a",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="010", running=0),
                    "SPACY_BASIC/basic/636a58bcdfd3c8e3450c0bbe",
                ),
            ],
        )

        self.assertEqual(map_eindex_entity_str, entities_index_e_map_ref)
        self.assertEqual(map_c2e, map_c2e_ref)

    def test_iterate_over_linkers(self):

        text = "Diabetic ulcers are related to burns."

        phrases = normalize_text(text, self.nlp)

        if "entityLinker" not in self.nlp.pipe_names:
            self.nlp.add_pipe("entityLinker", last=True)

        global_triples, map_muindex_candidate, ecl = phrases_to_triples(
            phrases, self.nlp, self.rules, window_size=2
        )

        phrase_entities_foos: dict = {
            EntityLinker.BERN_V2: lambda p: self.response_bern["annotations"],
            EntityLinker.SPACY_NAIVE_WIKI: lambda p: self.nlp(
                p
            )._.linkedEntities,
        }

        map_eindex_entity, map_c2e = iterate_over_linkers(
            phrases=phrases,
            ecl=ecl,
            map_muindex_candidate=map_muindex_candidate,
            phrase_entities_foos=phrase_entities_foos,
        )

        map_eindex_entity_str = {
            k: v.to_dict(skip_defaults=True)
            for k, v in map_eindex_entity.items()
        }

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

        self.assertEqual(map_eindex_entity_str, map_eindex_entity_ref)
        self.assertEqual(map_c2e, map_c2e_ref)


if __name__ == "__main__":
    unittest.main()
