import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path

import coreferee
import spacy
import yaml

from lm_service.graph import phrase_to_deptree, transform_advcl
from lm_service.preprocessing import normalize_input_text
from lm_service.relation import (
    compute_distances,
    generate_extra_graphs,
    graph_to_candidate_pile,
    graph_to_triples,
)

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    phrases = normalize_input_text(text, terminal_full_stop=False)

    documents = {
        "near-field": "The medium was affected by the near-field radiation",
        "cheops0_trunc": "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets,"
        " which will allow the estimation of their mass",
        "cheops0": "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets,"
        " which will allow the estimation of their mass, density,"
        " composition and their formation.",
        "coref": "Although he was very busy with his work, Peter Brown had had enough of it. "
        "He and his wife decided they needed a holiday. "
        "They travelled to Spain because they loved the country very much.",
        "cheops_ext": "Cheops ( CHaracterising ExOPlanets Satellite ) is a European space "
        "telescope to determine the size of known extrasolar planets , "
        "which will allow the estimation of their mass , density , composition and their formation. "
        "It is the first Small class mission in ESA 's Cosmic Vision "
        "science programme Launched on 18 December 2019",
        "photometric": "Cheops measures photometric signals with a precision limited by stellar photon "
        "noise of 150 ppm min for a 9th magnitude star. "
        "This corresponds to the transit of an Earth sized planet "
        "orbiting a star of 0 . 9 R in 60 days "
        " detected with a S Ntransit > 10 ( 100 ppm transit depth )",
    }

    def test_consecutive_candidates(self):

        for document in self.documents.values():
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            cp, _, mgraph = graph_to_candidate_pile(graph, rules=self.rules)

    def test_distances(self):
        for document in self.documents.values():
            rdoc, graph0 = phrase_to_deptree(self.nlp, document)
            pile, _, graph = graph_to_candidate_pile(graph0, self.rules)
            g_undirected, g_reversed, g_weighted = generate_extra_graphs(graph)
            relation_indices = [c.root.i for c in pile.relations]
            (
                distance_undirected,
                distance_directed,
                distance_levels,
            ) = compute_distances(
                graph,
                g_undirected=g_undirected,
                g_weighted=g_weighted,
                indices_of_interest=relation_indices,
            )

    # @unittest.skip("")
    def test_relation(self):
        documents = {
            key: self.documents[key]
            for key in [
                "near-field",
                "cheops0_trunc",
                "cheops_ext",
                "photometric",
            ]
        }
        acc_triples = []
        triples_projected = {}
        for key, doc in documents.items():
            rdoc, graph = phrase_to_deptree(self.nlp, doc)

            triples = graph_to_triples(rdoc, graph, self.rules)
            triples = [tri.normalize_relation() for tri in triples]
            acc_triples += triples

            triples_projected[key] = [tri.project_to_text() for tri in triples]

        # NB known problems in cheops_ext
        reference = {
            "near-field": [("medium", "wasAffectedBy", "nearFieldRadiation")],
            "cheops0_trunc": [
                ("CHEOPS", "is", "europeanSpaceTelescope"),
                (
                    "europeanSpaceTelescope",
                    "determines",
                    "sizeOfKnownExtrasolarPlanets",
                ),
                (
                    "europeanSpaceTelescope",
                    "allows",
                    "estimationOfMassOfKnownExtrasolarPlanets",
                ),
            ],
            "cheops_ext": [
                ("Cheops", "is", "europeanSpaceTelescope"),
                (
                    "europeanSpaceTelescope",
                    "is",
                    "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
                ),
                (
                    "europeanSpaceTelescope",
                    "determines",
                    "sizeOfKnownExtrasolarPlanets",
                ),
                (
                    "europeanSpaceTelescope",
                    "allows",
                    "estimationOfMassOfKnownExtrasolarPlanets",
                ),
                (
                    "europeanSpaceTelescope",
                    "allows",
                    "estimationOfDensityExtrasolarPlanetsOfKnown",
                ),
                (
                    "europeanSpaceTelescope",
                    "allows",
                    "estimationOfCompositionExtrasolarPlanetsOfKnown",
                ),
            ],
            "photometric": [
                ("Cheops", "measuresWith", "photometricSignals"),
                (
                    "Cheops",
                    "measuresWith",
                    "precisionOf150PpmMinFor9thMagnitudeStar",
                ),
                (
                    "100PpmTransitDepth",
                    "correspondsTo",
                    "transitOfSizedPlanet",
                ),
            ],
        }
        for k in triples_projected:
            self.assertEqual(triples_projected[k], reference[k])


if __name__ == "__main__":
    unittest.main()
