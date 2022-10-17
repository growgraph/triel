import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path
from pprint import pprint

import coreferee
import spacy
import yaml
from reference.distances import reference_distance

from lm_service.coref import graph_component_maps, render_coref_maps_wrapper
from lm_service.graph import phrase_to_deptree, relabel_nodes_and_key
from lm_service.linking import (
    EntityLinker,
    ent_db_type_local_gg,
    iterate_linking_over_phrases,
    link_unlinked_entities,
)
from lm_service.onto import AbsToken, MuIndex, apply_map
from lm_service.phrase import graph_to_triples
from lm_service.preprocessing import normalize_input_text
from lm_service.relation import (
    compute_distances,
    generate_extra_graphs,
    graph_to_candidate_pile,
)
from lm_service.text import (
    cast_simplified_triples_table,
    normalize_text,
    phrases_to_triples,
)

logger = logging.getLogger(__name__)


class TestTriples(unittest.TestCase):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    phrases = normalize_input_text(text, terminal_full_stop=False)

    documents = {
        "near-field": "The medium was affected by the near-field radiation",
        "cheops0_trunc": (
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass"
        ),
        "cheops0": (
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass, density,"
            " composition and their formation."
        ),
        "coref": (
            "Although he was very busy with his work, Peter Brown had had"
            " enough of it. He and his wife decided they needed a holiday."
            " They travelled to Spain because they loved the country very"
            " much."
        ),
        "cheops_ext": (
            "Cheops ( CHaracterising ExOPlanets Satellite ) is a European"
            " space telescope to determine the size of known extrasolar"
            " planets , which will allow the estimation of their mass ,"
            " density , composition and their formation. It is the first Small"
            " class mission in ESA 's Cosmic Vision science programme Launched"
            " on 18 December 2019"
        ),
        "photometric": (
            "Cheops measures photometric signals with a precision limited by"
            " stellar photon noise of 150 ppm min for a 9th magnitude star."
            " This corresponds to the transit of an Earth sized planet"
            " orbiting a star of 0 . 9 R in 60 days  detected with a S"
            " Ntransit > 10 ( 100 ppm transit depth )"
        ),
        "thousands": (
            "Thousands of exoplanets have been discovered by the end of the"
            " 2010s; some have minimum mass measurements from the radial"
            " velocity method while others that are seen to transit their"
            " parent stars have measures of their physical size."
        ),
    }

    def test_distances(self):
        distance_check = {}
        for key, document in self.documents.items():
            rdoc, graph0 = phrase_to_deptree(self.nlp, document)

            # cast index to compound index
            map_tree_subtree_index = graph_component_maps(graph0)
            map_tree_subtree_index = {
                k: AbsToken.ituple2stuple(v)
                for k, v in map_tree_subtree_index.items()
            }

            graph_relabeled = relabel_nodes_and_key(
                graph0, map_tree_subtree_index, "s"
            )

            pile, _, mod_graph = graph_to_candidate_pile(
                graph_relabeled, self.rules
            )

            g_undirected, g_reversed, g_weighted = generate_extra_graphs(
                graph_relabeled
            )
            relation_indices = [c.root.s for c in pile.relations]
            (
                distance_undirected,
                distance_directed,
                distance_reversed,
                distance_levels,
            ) = compute_distances(
                mod_graph,
                g_undirected=g_undirected,
                g_weighted=g_weighted,
                g_reversed=g_reversed,
                indices_of_interest=relation_indices,
            )

            distance_check.update(
                {
                    key: {
                        "undirected": distance_undirected,
                        "directed": distance_directed,
                        "levels": distance_levels,
                    }
                }
            )

        self.assertEqual(distance_check, reference_distance)

    def test_relation(self):
        documents = {
            key: self.documents[key]
            for key in [
                # "near-field",
                # "cheops0_trunc",
                "cheops_ext",
                # "photometric",
                # "thousands",
            ]
        }
        acc_triples = []
        triples_projected = {}

        for key, doc in documents.items():
            rdoc, graph = phrase_to_deptree(self.nlp, doc)

            # cast index to compound index
            map_tree_subtree_index = graph_component_maps(graph)
            map_tree_subtree_index = {
                k: AbsToken.ituple2stuple(v)
                for k, v in map_tree_subtree_index.items()
            }
            graph_relabeled = relabel_nodes_and_key(
                graph, map_tree_subtree_index, "s"
            )

            # coref maps
            (
                map_subbable_to_chain,
                map_chain_to_most_specific,
            ) = render_coref_maps_wrapper(rdoc)

            (
                map_subbable_to_chain_str,
                map_chain_to_most_specific_str,
            ) = apply_map(
                [map_subbable_to_chain, map_chain_to_most_specific],
                map_tree_subtree_index,
            )

            triples = graph_to_triples(
                graph_relabeled,
                map_subbable_to_chain_str,
                map_chain_to_most_specific_str,
                self.rules,
            )
            triples = [tri.normalize_relation() for tri in triples]
            acc_triples += triples

            triples_projected[key] = [tri.project_to_text() for tri in triples]

        reference = {
            "near-field": [("medium", "isAffectedBy", "nearFieldRadiation")],
            "cheops0_trunc": [
                ("CHEOPS", "is", "europeanSpaceTelescope"),
                (
                    "europeanSpaceTelescope",
                    "determines",
                    "sizeOfKnownExtrasolarPlanets",
                ),
            ],
            "cheops_ext": [
                ("Cheops", "is", "europeanSpaceTelescope"),
                (
                    "europeanSpaceTelescope",
                    "determines",
                    "sizeOfKnownExtrasolarPlanets",
                ),
                (
                    "europeanSpaceTelescope",
                    "is",
                    "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
                ),
                (
                    "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
                    "LaunchesOn",
                    "18December2019",
                ),
            ],
            "photometric": [
                ("Cheops", "measures", "photometricSignals"),
                (
                    "Cheops",
                    "measuresWith",
                    "precisionOf150PpmMinFor9thMagnitudeStar",
                ),
                (
                    "precisionOf150PpmMinFor9thMagnitudeStar",
                    "limitsBy",
                    "stellarPhotonNoise",
                ),
                ("This", "correspondsTo", "transitOfEarthSizedPlanet"),
                ("transitOfEarthSizedPlanet", "orbits", "star"),
                ("transitOfEarthSizedPlanet", "orbitsOf", "09RIn60Day"),
            ],
            "thousands": [
                ("thousandOfExoplanets", "isDiscoveredBy", "endOf2010"),
                (
                    "some",
                    "has",
                    "minimumMassMeasurementsFromRadialVelocityMethod",
                ),
                ("others", "isSeen", "toTransitParentStarsOfOthers"),
                ("others", "has", "measuresOfPhysicalSizeOfOthers"),
            ],
        }

        for k in triples_projected:
            if triples_projected[k] != reference[k]:
                print(k)
                pprint(triples_projected[k])
                pprint(reference[k])
            self.assertEqual(triples_projected[k], reference[k])

    def test_text_to_relations(self):
        text = (
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass, density,"
            " composition and their formation. Launched on 18 December 2019,"
            " it is the first Small-class mission in ESA's Cosmic Vision"
            " science programme.The small satellite features an optical"
            " Ritchey-Chrétien telescope with an aperture of 30 cm, mounted on"
            " a standard small satellite platform. It was placed into a"
            " Sun-synchronous orbit of about 700 km altitude. Thousands of"
            " exoplanets have been discovered by the end of the 2010s; some"
            " have minimum mass measurements from the radial velocity method"
            " while others that are seen to transit their parent stars have"
            " measures of their physical size."
        )

        phrases = normalize_text(text, self.nlp)

        global_triples, map_mu_index_triple, _ = phrases_to_triples(
            phrases, self.nlp, self.rules, window_size=2
        )

        global_triples_txt = cast_simplified_triples_table(
            global_triples, map_mu_index_triple
        )

        reference = {
            MuIndex(meta=True, phrase=0, token="000", running=0): (
                "CHEOPS",
                "is",
                "europeanSpaceTelescope",
            ),
            MuIndex(meta=True, phrase=0, token="000", running=1): (
                "europeanSpaceTelescope",
                "determines",
                "sizeOfKnownExtrasolarPlanets",
            ),
            MuIndex(meta=True, phrase=1, token="000", running=0): (
                "CHEOPS",
                "is",
                "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
            ),
            MuIndex(meta=True, phrase=1, token="000", running=1): (
                "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
                "launchesOn",
                "18December2019",
            ),
            MuIndex(meta=True, phrase=2, token="000", running=0): (
                "smallSatellite",
                "features",
                "opticalRitcheyChretienTelescopeWithApertureOf30Cm",
            ),
            MuIndex(meta=True, phrase=2, token="000", running=1): (
                "opticalRitcheyChretienTelescopeWithApertureOf30Cm",
                "mountsOn",
                "standardSmallSatellitePlatform",
            ),
            MuIndex(meta=True, phrase=3, token="000", running=0): (
                "smallSatellite",
                "isPlacedInto",
                "SunSynchronousOrbitOf700KmAltitude",
            ),
            MuIndex(meta=True, phrase=4, token="000", running=0): (
                "some",
                "has",
                "minimumMassMeasurementsFromRadialVelocityMethod",
            ),
            MuIndex(meta=True, phrase=4, token="000", running=1): (
                "thousandOfExoplanets",
                "isDiscoveredBy",
                "endOf2010S",
            ),
            MuIndex(meta=True, phrase=4, token="000", running=2): (
                "others",
                "has",
                "measuresOfPhysicalSizeOfOthers",
            ),
            MuIndex(meta=True, phrase=4, token="000", running=3): (
                "others",
                "isSeen",
                "toTransitParentStarsOfOthers",
            ),
            MuIndex(meta=True, phrase=0, token="000", running=2): (
                "(meta)determines",
                "allows",
                "estimationOfDensity",
            ),
            MuIndex(meta=True, phrase=0, token="000", running=3): (
                "(meta)determines",
                "allows",
                "estimationOfComposition",
            ),
            MuIndex(meta=True, phrase=0, token="000", running=4): (
                "(meta)determines",
                "allows",
                "estimationOfMassOfKnownExtrasolarPlanets",
            ),
            MuIndex(meta=True, phrase=0, token="000", running=5): (
                "(meta)determines",
                "allows",
                "estimationOfFormationOfKnownExtrasolarPlanets",
            ),
        }

        self.assertEqual(global_triples_txt, reference)

    def test_text_linking(self):
        text = "Diabetic ulcers are related to burns."

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

        phrases = normalize_text(text, self.nlp)

        global_triples, map_muindex_candidate, ecl = phrases_to_triples(
            phrases, self.nlp, self.rules, window_size=2
        )

        foo_link = lambda p: response_bern["annotations"]

        map_eindex_entity = {}
        map_c2e = []
        map_eindex_entity, map_c2e = iterate_linking_over_phrases(
            phrases=phrases,
            ecl=ecl,
            map_eindex_entity=map_eindex_entity,
            map_c2e=map_c2e,
            link_foo=foo_link,
        )

        map_eindex_entity, map_c2e = link_unlinked_entities(
            map_eindex_entity, map_c2e, map_muindex_candidate
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
                    MuIndex(meta=False, phrase=0, token="002", running=9),
                    "LOCAL_NON_EL/ent_db_type_local_gg/44afc2df2816ef50ecd4f847",
                ),
            ],
        )

        self.assertEqual(map_eindex_entity_str, map_eindex_entity_ref)
        self.assertEqual(map_c2e, map_c2e_ref)


if __name__ == "__main__":
    unittest.main()
