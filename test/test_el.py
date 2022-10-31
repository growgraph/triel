import pkgutil
import unittest
from collections import defaultdict

import coreferee
import spacy
import yaml

from lm_service.linking import (
    EntityLinker,
    EntityLinkerManager,
    PhraseMapper,
    iterate_over_linkers,
    link_candidate_entity,
    link_over_phrases,
)
from lm_service.onto import MuIndex
from lm_service.text import (
    normalize_text,
    phrases_to_basis_triples,
    phrases_to_triples,
)
from lm_service.top import to_dict


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

    r_bern_multiphrase = {
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
            {
                "id": ["mesh:D009369"],
                "is_neural_normalized": False,
                "mention": "tumour",
                "obj": "disease",
                "prob": 0.9999922513961792,
                "span": {"begin": 59, "end": 65},
            },
            {
                "id": ["mesh:D001120"],
                "is_neural_normalized": False,
                "mention": "arginine",
                "obj": "drug",
                "prob": 0.9819279909133911,
                "span": {"begin": 93, "end": 101},
            },
        ]
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
            EntityLinkerManager._normalize_bern_entity(item)
            for item in self.response_bern["annotations"]
        ]

        pm = PhraseMapper([text])

        bern_normalized_ip: defaultdict[
            int, list[tuple[str, tuple[int, int]]]
        ] = defaultdict(list)
        for e, span in bern_normalized:
            ip, (ia, ib) = pm.span(span)
            bern_normalized_ip[ip] += [(e.hash, (ia, ib))]

        map_c2e = link_candidate_entity(bern_normalized_ip, ecl)

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

        elm = EntityLinkerManager(
            {
                EntityLinker.BERN_V2: {
                    "url": "http://bern2.korea.ac.kr/plain",
                    "text_field": "text",
                }
            }
        )
        elm.set_linker_type(EntityLinker.BERN_V2)

        map_eindex_entity, map_c2e = link_over_phrases(
            phrases=phrases, ecl=ecl, elm=elm
        )

        map_eindex_entity_str = to_dict(map_eindex_entity)

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

    @unittest.skip("obsolete")
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
        map_eindex_entity, map_c2e = link_over_phrases(
            phrases=phrases,
            ecl=ecl,
            map_eindex_entity=map_eindex_entity,
            map_c2e=map_c2e,
            etype=EntityLinker.SPACY_NAIVE_WIKI,
        )

        map_eindex_entity_str = to_dict(map_eindex_entity)

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

    @unittest.skip("obsolete")
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

        map_eindex_entity, map_c2e = link_over_phrases(
            phrases=phrases,
            ecl=ecl,
            link_foo_kwargs={"nlp": self.nlp},
            etype=EntityLinker.SPACY_BASIC,
        )

        map_eindex_entity_str = to_dict(map_eindex_entity)

        map_eindex_entity_ref, map_c2e_ref = (
            {
                "SPACY_BASIC/basic/e50064e29b1d2f3fe19cd61ba0b6b7144069c90f": {
                    "linker_type": "SPACY_BASIC",
                    "ent_db_type": "basic",
                    "id": "e50064e29b1d2f3fe19cd61ba0b6b7144069c90f",
                    "hash": "SPACY_BASIC/basic/e50064e29b1d2f3fe19cd61ba0b6b7144069c90f",
                    "ent_type": "386",
                    "original_form": "cheops",
                },
                "SPACY_BASIC/basic/c93ca73f8d770c25597bb877021545d719bf1e4d": {
                    "linker_type": "SPACY_BASIC",
                    "ent_db_type": "basic",
                    "id": "c93ca73f8d770c25597bb877021545d719bf1e4d",
                    "hash": "SPACY_BASIC/basic/c93ca73f8d770c25597bb877021545d719bf1e4d",
                    "ent_type": "381",
                    "original_form": "european",
                },
            },
            [
                (
                    MuIndex(meta=False, phrase=0, token="000", running=0),
                    "SPACY_BASIC/basic/e50064e29b1d2f3fe19cd61ba0b6b7144069c90f",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="010", running=0),
                    "SPACY_BASIC/basic/c93ca73f8d770c25597bb877021545d719bf1e4d",
                ),
            ],
        )

        self.assertEqual(map_eindex_entity_str, map_eindex_entity_ref)
        self.assertEqual(map_c2e, map_c2e_ref)

    def test_iterate_over_linkers(self):

        text = "Diabetic ulcers are related to burns."

        phrases = normalize_text(text, self.nlp)

        if "entityLinker" not in self.nlp.pipe_names:
            self.nlp.add_pipe("entityLinker", last=True)

        global_triples, map_muindex_candidate, ecl = phrases_to_triples(
            phrases, self.nlp, self.rules, window_size=2
        )

        elm = EntityLinkerManager(
            {
                EntityLinker.BERN_V2: {
                    "url": "http://bern2.korea.ac.kr/plain",
                    "text_field": "text",
                }
            },
        )

        map_eindex_entity, map_c2e = iterate_over_linkers(
            phrases=phrases,
            ecl=ecl,
            map_muindex_candidate=map_muindex_candidate,
            elm=elm,
        )

        map_eindex_entity_str = to_dict(map_eindex_entity)

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
                "LOCAL_NON_EL/ent_db_type_local_gg/dda96135ac461d989729db27e63bdf3f88b724e3": {
                    "linker_type": "LOCAL_NON_EL",
                    "ent_db_type": "ent_db_type_local_gg",
                    "id": "dda96135ac461d989729db27e63bdf3f88b724e3",
                    "hash": "LOCAL_NON_EL/ent_db_type_local_gg/dda96135ac461d989729db27e63bdf3f88b724e3",
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
                    MuIndex(meta=False, phrase=0, token="002", running=9),
                    "LOCAL_NON_EL/ent_db_type_local_gg/dda96135ac461d989729db27e63bdf3f88b724e3",
                ),
            ],
        )

        self.assertEqual(map_eindex_entity_str, map_eindex_entity_ref)
        self.assertEqual(map_c2e, map_c2e_ref)

    def test_phrasemapper(self):
        pretext = (
            "Diabetic ulcers are related to burns. Autophagy maintains tumour"
            " growth through circulating arginine."
        )
        phrases = normalize_text(pretext, self.nlp)
        text = " ".join(phrases)
        pm = PhraseMapper(phrases, " ")
        i = 39
        ip, m = pm(i)
        self.assertEqual(text[i : i + 9], phrases[ip][m : m + 9])

    def test_qb(self):
        pretext = (
            "Diabetic ulcers are related to burns. Autophagy maintains tumour"
            " growth through circulating arginine."
        )
        phrases = normalize_text(pretext, self.nlp)

        (
            striples,
            striples_meta,
            candidate_depot,
            relations,
            _,
        ) = phrases_to_basis_triples(self.nlp, self.rules, phrases)

        ecl = candidate_depot.unfold_conjunction()

        text = " ".join(phrases)

        elm = EntityLinkerManager(
            {
                EntityLinker.BERN_V2: {
                    "url": "http://bern2.korea.ac.kr/plain",
                    "text_field": "text",
                }
            }
        )

        elm.set_linker_type(EntityLinker.BERN_V2)
        response_bern = elm.query(text)
        bern_normalized = elm.normalize(response_bern)

        pm = PhraseMapper(phrases)

        bern_normalized_ip: defaultdict[
            int, list[tuple[str, tuple[int, int]]]
        ] = defaultdict(list)
        for e, span in bern_normalized:
            ip, (ia, ib) = pm.span(span)
            bern_normalized_ip[ip] += [(e.hash, (ia, ib))]

        map_c2e = link_candidate_entity(bern_normalized_ip, ecl)

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
                (
                    MuIndex(meta=False, phrase=1, token="003", running=0),
                    "BERN_V2/mesh/D009369",
                ),
                (
                    MuIndex(meta=False, phrase=1, token="006", running=0),
                    "BERN_V2/mesh/D001120",
                ),
            ],
        )

    def test_fishing(self):
        text = "Diabetic ulcers are related to skin burns."
        response = {
            "date": "2022-10-27T21:26:19.815994Z",
            "entities": [
                {
                    "confidence_score": 0.5669,
                    "domains": ["Medicine"],
                    "offsetEnd": 15,
                    "offsetStart": 0,
                    "rawName": "Diabetic ulcers",
                    "wikidataId": "Q2078852",
                    "wikipediaExternalRef": 3120850,
                },
                {
                    "confidence_score": 0.4572,
                    "domains": ["Engineering", "Chemistry"],
                    "offsetEnd": 41,
                    "offsetStart": 31,
                    "rawName": "skin burns",
                    "wikidataId": "Q170518",
                    "wikipediaExternalRef": 233082,
                },
            ],
            "global_categories": [
                {
                    "category": "Hazards of outdoor recreation",
                    "page_id": 69268770,
                    "source": "wikipedia-en",
                    "weight": 0.14285714285714288,
                },
                {
                    "category": "Medical emergencies",
                    "page_id": 741752,
                    "source": "wikipedia-en",
                    "weight": 0.14285714285714288,
                },
                {
                    "category": (
                        "Skin conditions resulting from physical factors"
                    ),
                    "page_id": 19985316,
                    "source": "wikipedia-en",
                    "weight": 0.14285714285714288,
                },
                {
                    "category": "Burns",
                    "page_id": 44260517,
                    "source": "wikipedia-en",
                    "weight": 0.14285714285714288,
                },
                {
                    "category": "Acute pain",
                    "page_id": 49648845,
                    "source": "wikipedia-en",
                    "weight": 0.14285714285714288,
                },
                {
                    "category": "Heat transfer",
                    "page_id": 23011909,
                    "source": "wikipedia-en",
                    "weight": 0.14285714285714288,
                },
                {
                    "category": "Necrosis",
                    "page_id": 36631009,
                    "source": "wikipedia-en",
                    "weight": 0.14285714285714288,
                },
            ],
            "language": {"conf": 1.0, "lang": "en"},
            "nbest": False,
            "runtime": 78,
            "software": "entity-fishing",
            "text": "Diabetic ulcers are related to skin burns.",
            "version": "0.0.5",
        }
        phs = normalize_text(text, self.nlp)

        (
            striples,
            striples_meta,
            candidate_depot,
            relations,
            _,
        ) = phrases_to_basis_triples(self.nlp, self.rules, phs)

        ecl = candidate_depot.unfold_conjunction()

        elm = EntityLinkerManager(
            {
                EntityLinker.FISHING: {
                    "url": "http://",
                    "text_field": "text",
                }
            }
        )

        elm.set_linker_type(EntityLinker.FISHING)

        normalized = elm.normalize(response)

        pm = PhraseMapper([text])

        normalized_ip: defaultdict[
            int, list[tuple[str, tuple[int, int]]]
        ] = defaultdict(list)
        for e, span in normalized:
            ip, (ia, ib) = pm.span(span)
            normalized_ip[ip] += [(e.hash, (ia, ib))]

        map_c2e = link_candidate_entity(normalized_ip, ecl)

        self.assertEqual(
            map_c2e,
            [
                (
                    MuIndex(meta=False, phrase=0, token="001", running=0),
                    "FISHING/wikidataId/Q2078852",
                ),
                (
                    MuIndex(meta=False, phrase=0, token="006", running=0),
                    "FISHING/wikidataId/Q170518",
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
