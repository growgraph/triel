import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path

import spacy
import yaml

from lm_service.graph import phrase_to_deptree, transform_advcl
from lm_service.preprocessing import normalize_input_text
from lm_service.relation import (
    add_hash,
    compute_distances,
    generate_extra_graphs,
    graph_to_candidate_pile,
    graph_to_relations,
    phrase_to_relations,
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

    documents = [
        "The medium was affected by the near-field radiation",
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets,"
        " which will allow the estimation of their mass, density,"
        " composition and their formation.",
    ]

    def test_consecutive_candidates(self):

        for document in self.documents:
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            cp = graph_to_candidate_pile(graph, rules=self.rules)

    def test_distances(self):
        for document in self.documents:
            rdoc, graph = phrase_to_deptree(self.nlp, document)
            pile = graph_to_candidate_pile(graph, self.rules)
            g_undirected, g_reversed, g_weighted = generate_extra_graphs(graph)
            (
                distance_undirected,
                distance_directed,
                distance_levels,
            ) = compute_distances(
                graph,
                g_undirected=g_undirected,
                g_weighted=g_weighted,
                pile=pile.relations,
            )

    # @unittest.skip("")
    def test_relation(self):
        document = self.phrases[0]
        rdoc, graph = phrase_to_deptree(self.nlp, document)

        # mg, r, triples_projected, _ = graph_to_relations(graph, self.rules)
        triples = graph_to_relations(graph, self.rules)
        [tri.project_to_text() for tri in triples]
        print("")
        # self.assertEqual(
        #     triples_projected,
        #     [
        #         ("CHEOPS", "be", "telescope"),
        #         ("telescope", "determine", "size"),
        #         # here it should be rather (("telescope", "determine", "size"), "allow", "estimation")
        #         ("telescope", "allow", "estimation"),
        #     ],
        # )

    @unittest.skip("")
    def test_relation_advanced(self):
        nmax = 3
        window_size = 2

        phrases = [transform_advcl(self.nlp, p) for p in self.phrases[:nmax]]
        agg = []
        for i in range(nmax):
            fragment = ". ".join(phrases[i : i + window_size])
            (
                graph,
                coref_graph,
                metagraph,
                triples_expanded,
                triples_proj,
            ) = phrase_to_relations(fragment, self.nlp, self.rules)
            r = add_hash(triples_expanded, graph)
            agg.extend(r)

        # pprint(agg)
        # self.assertEqual(
        #     triples_projected,
        #     [
        #         ("CHEOPS", "be", "telescope"),
        #         ("telescope", "determine", "size"),
        #         ("size", "allow", "estimation"),
        #     ],
        # )


if __name__ == "__main__":
    unittest.main()
