import logging
import os
import pkgutil
import sys
import unittest
from itertools import product
from pathlib import Path

import coreferee
import spacy
import yaml

from lm_service.coref import (
    render_coref_candidate_map,
    render_coref_graph,
    sub_coreference,
)
from lm_service.graph import phrase_to_deptree, transform_advcl
from lm_service.onto import Candidate, Token, TripleCandidate
from lm_service.preprocessing import normalize_input_text
from lm_service.relation import (
    add_hash,
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
        documents = [
            self.documents[key] for key in ["near-field", "cheops0_trunc"]
        ]
        acc_triples = []
        triples_projected = []
        for doc in documents:
            rdoc, graph = phrase_to_deptree(self.nlp, doc)

            triples = graph_to_triples(rdoc, graph, self.rules)
            triples = [
                tri.drop_articles().drop_amod_vbn().normalize_relation()
                for tri in triples
            ]
            acc_triples += triples

            triples_projected += [tri.project_to_text() for tri in triples]

        self.assertEqual(
            triples_projected,
            [
                ("medium", "wasAffectedBy", "nearFieldRadiation"),
                ("CHEOPS", "is", "europeanSpaceTelescope"),
                (
                    "europeanSpaceTelescope",
                    "determines",
                    "sizeOfExtrasolarPlanets",
                ),
                (
                    "europeanSpaceTelescope",
                    "allows",
                    "estimationOfMassOfSizeOfExtrasolarPlanets",
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
