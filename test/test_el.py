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
                "prob": 0.9999968409538269,
                "span": {"begin": 0, "end": 15},
            },
            {
                "id": ["mesh:D002056"],
                "is_neural_normalized": False,
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
            [
                (MuIndex(False, 0, "001", 0), (0, 0)),
                (MuIndex(False, 0, "005", 0), (0, 1)),
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

        entities_index_e_map_ref, map_c2e_ref = (
            {
                (0, 0): {
                    "linker_type": EntityLinker.BERN_V2,
                    "ent_type": "disease",
                    "ent_db_type": "mesh",
                    "id": "D017719",
                    "confidence": 0.9999968409538269,
                },
                (0, 1): {
                    "linker_type": EntityLinker.BERN_V2,
                    "ent_type": "disease",
                    "ent_db_type": "mesh",
                    "id": "D002056",
                    "confidence": 0.9982181191444397,
                },
            },
            [
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    (0, 0),
                ),
                (
                    MuIndex(meta=False, phrase=0, token="005", running=0),
                    (0, 1),
                ),
            ],
        )

        self.assertEqual(map_eindex_entity, entities_index_e_map_ref)
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
        ) = phrases_to_basis_triples(self.nlp, self.rules, phrases)

        ecl = candidate_depot.unfold_conjunction()

        entities_index_e_map = {}
        map_c2e = []
        entities_index_e_map, map_c2e = iterate_linking_over_phrases(
            phrases=phrases,
            ecl=ecl,
            map_eindex_entity=entities_index_e_map,
            map_c2e=map_c2e,
            link_foo=foo_link,
            etype=EntityLinker.SPACY_NAIVE_WIKI,
        )

        entities_index_e_map_ref, map_c2e_ref = (
            {
                (0, 0): {
                    "linker_type": EntityLinker.SPACY_NAIVE_WIKI,
                    "ent_db_type": "wikidata",
                    "id": "Q6452285",
                    "original": "ulcer",
                    "ent_type": None,
                    "desc": "type of cutaneous condition",
                },
                (0, 1): {
                    "linker_type": EntityLinker.SPACY_NAIVE_WIKI,
                    "ent_db_type": "wikidata",
                    "id": "Q170518",
                    "original": "burns",
                    "ent_type": None,
                    "desc": (
                        "injury to flesh or skin, often caused by excessive"
                        " heat"
                    ),
                },
            },
            [
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    (0, 0),
                ),
                (
                    MuIndex(meta=False, phrase=0, token="005", running=0),
                    (0, 1),
                ),
            ],
        )

        self.assertEqual(entities_index_e_map, entities_index_e_map_ref)
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

        entities_index_e_map_ref, map_c2e_ref = (
            {
                (0, 0): {
                    "linker_type": EntityLinker.SPACY_BASIC,
                    "ent_db_type": "basic",
                    "ent_type": 386,
                },
                (0, 1): {
                    "linker_type": EntityLinker.SPACY_BASIC,
                    "ent_db_type": "basic",
                    "ent_type": 381,
                },
            },
            [
                (
                    MuIndex(meta=False, phrase=0, token="000", running=0),
                    (0, 0),
                ),
                (
                    MuIndex(meta=False, phrase=0, token="010", running=0),
                    (0, 1),
                ),
            ],
        )

        self.assertEqual(map_eindex_entity, entities_index_e_map_ref)
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

        entities_index_e_map, map_c2e = iterate_over_linkers(
            phrases=phrases,
            ecl=ecl,
            map_muindex_candidate=map_muindex_candidate,
            phrase_entities_foos=phrase_entities_foos,
        )

        entities_index_e_map_ref, map_c2e_ref = (
            {
                (0, 0): {
                    "linker_type": EntityLinker.BERN_V2,
                    "ent_type": "disease",
                    "ent_db_type": "mesh",
                    "id": "D017719",
                    "confidence": 0.9999968409538269,
                },
                (0, 1): {
                    "linker_type": EntityLinker.BERN_V2,
                    "ent_type": "disease",
                    "ent_db_type": "mesh",
                    "id": "D002056",
                    "confidence": 0.9982181191444397,
                },
                (0, 2): {
                    "linker_type": EntityLinker.SPACY_NAIVE_WIKI,
                    "ent_db_type": "wikidata",
                    "id": "Q6452285",
                    "original": "ulcer",
                    "ent_type": None,
                    "desc": "type of cutaneous condition",
                },
                (0, 3): {
                    "linker_type": EntityLinker.SPACY_NAIVE_WIKI,
                    "ent_db_type": "wikidata",
                    "id": "Q170518",
                    "original": "burns",
                    "ent_type": None,
                    "desc": (
                        "injury to flesh or skin, often caused by excessive"
                        " heat"
                    ),
                },
                (0, 4): {
                    "linker_type": EntityLinker.LOCAL_NON_EL,
                    "ent_db_type": "ent_db_type_local_gg",
                    "id": "is related to",
                    "confidence": 0.0,
                },
            },
            [
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    (0, 0),
                ),
                (
                    MuIndex(meta=False, phrase=0, token="005", running=0),
                    (0, 1),
                ),
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    (0, 2),
                ),
                (
                    MuIndex(meta=False, phrase=0, token="005", running=0),
                    (0, 3),
                ),
                (
                    MuIndex(meta=False, phrase=0, token="002", running=0),
                    (0, 4),
                ),
            ],
        )

        self.assertEqual(entities_index_e_map, entities_index_e_map_ref)
        self.assertEqual(map_c2e, map_c2e_ref)


if __name__ == "__main__":
    unittest.main()
