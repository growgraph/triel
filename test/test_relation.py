import pkgutil
import yaml
import sys
import os
import unittest
import spacy
import logging
from collections import deque
from pathlib import Path
import coreferee
from lm_service.relation import (
    graph_to_relations,
    parse_relations_advanced,
    add_hash,
    sieve_sources_targets,
)
from lm_service.preprocessing import normalize_input_text
from lm_service.graph import transform_advcl
from lm_service.graph import dep_tree_from_phrase
from lm_service.onto import (
    Relation,
    ACandidatePile,
    ACandidateKind,
    SourceOrTarget,
)
from lm_service.relation import (
    find_candidates_bfs,
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

        sizes = []
        for document in self.documents:
            rdoc, graph = dep_tree_from_phrase(self.nlp, document)
            roots = [n for n, d in graph.in_degree() if d == 0]
            relation_pile = ACandidatePile()
            source_target_pile = ACandidatePile()
            find_candidates_bfs(
                graph,
                deque(roots),
                relation_pile,
                ACandidateKind.RELATION,
            )
            find_candidates_bfs(
                graph,
                deque(roots),
                source_target_pile,
                ACandidateKind.SOURCE_TARGET,
                rules=self.rules,
            )
            sources, targets = sieve_sources_targets(source_target_pile)
            sizes += [(len(sources), len(targets))]
        self.assertEqual(
            sizes,
            [(1, 2), (5, 5)],
        )

    @unittest.skip("")
    def test_relation(self):
        document = self.phrases[0]
        rdoc, graph = dep_tree_from_phrase(self.nlp, document)

        mg, r, triples_projected, _ = graph_to_relations(graph, self.rules)
        self.assertEqual(
            triples_projected,
            [
                ("CHEOPS", "be", "telescope"),
                ("telescope", "determine", "size"),
                # here it should be rather (("telescope", "determine", "size"), "allow", "estimation")
                ("telescope", "allow", "estimation"),
            ],
        )

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
            ) = parse_relations_advanced(fragment, self.nlp, self.rules)
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
