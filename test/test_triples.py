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
from reference.distances import ref_distance

from lm_service.coref import graph_component_maps, render_coref_maps_wrapper
from lm_service.graph import phrase_to_deptree, relabel_nodes_and_key
from lm_service.onto import AToken, apply_map
from lm_service.phrase import graph_to_triples
from lm_service.preprocessing import normalize_input_text, transform_advcl
from lm_service.relation import (
    compute_distances,
    form_triples,
    generate_extra_graphs,
    graph_to_candidate_pile,
    graph_to_maps,
)

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):

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

    reference_distance = ref_distance

    def test_distances(self):
        distance_check = {}
        for key, document in self.documents.items():
            rdoc, graph0 = phrase_to_deptree(self.nlp, document)

            # cast index to compound index
            map_tree_subtree_index = graph_component_maps(graph0)
            map_tree_subtree_index = {
                k: AToken.ituple2stuple(v)
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

        self.assertEqual(distance_check, self.reference_distance)

    def test_relation(self):
        documents = {
            key: self.documents[key]
            for key in [
                "near-field",
                "cheops0_trunc",
                "cheops_ext",
                "photometric",
                "thousands",
            ]
        }
        acc_triples = []
        triples_projected = {}

        for key, doc in documents.items():
            rdoc, graph = phrase_to_deptree(self.nlp, doc)

            # cast index to compound index
            map_tree_subtree_index = graph_component_maps(graph)
            map_tree_subtree_index = {
                k: AToken.ituple2stuple(v)
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

        # tri = acc_triples[0]
        # tt = tri.target
        # tt.sort_index()

        reference = {
            "near-field": [("medium", "wasAffectedBy", "nearFieldRadiation")],
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
                ("thousandOfExoplanets", "haveWasDiscoveredBy", "endOf2010"),
                (
                    "some",
                    "has",
                    "minimumMassMeasurementsFromRadialVelocityMethod",
                ),
                ("others", "wasSeen", "toTransitParentStarsOfOthers"),
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
        from lm_service.text import text_to_triples

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

        global_triples, map_mu_index_triple = text_to_triples(
            text, self.nlp, self.rules, window_size=2
        )

        from lm_service.text import cast_simplified_triples_table

        global_triples_txt = cast_simplified_triples_table(
            global_triples, map_mu_index_triple
        )

        reference = [
            ("CHEOPS", "is", "europeanSpaceTelescope"),
            (
                "europeanSpaceTelescope",
                "determine",
                "sizeOfKnownExtrasolarPlanets",
            ),
            (
                "CHEOPS",
                "is",
                "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
            ),
            (
                "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
                "launchedOn",
                "18December2019",
            ),
            (
                "smallSatellite",
                "features",
                "opticalRitcheyChretienTelescopeWithApertureOf30Cm",
            ),
            (
                "opticalRitcheyChretienTelescopeWithApertureOf30Cm",
                "mountedOn",
                "standardSmallSatellitePlatform",
            ),
            (
                "smallSatellite",
                "wasPlacedInto",
                "SunSynchronousOrbitOf700KmAltitude",
            ),
            (
                "some",
                "have",
                "minimumMassMeasurementsFromRadialVelocityMethod",
            ),
            ("thousandOfExoplanets", "haveBeenDiscoveredBy", "endOf2010S"),
            ("others", "have", "measuresOfPhysicalSizeOfOthers"),
            ("others", "areSeen", "toTransitParentStarsOfOthers"),
            ("(meta)determine", "willAllow", "estimationOfDensity"),
            ("(meta)determine", "willAllow", "estimationOfComposition"),
            (
                "(meta)determine",
                "willAllow",
                "estimationOfMassOfKnownExtrasolarPlanets",
            ),
            (
                "(meta)determine",
                "willAllow",
                "estimationOfFormationOfKnownExtrasolarPlanets",
            ),
        ]

        self.assertEqual(global_triples_txt, reference)


if __name__ == "__main__":
    unittest.main()
