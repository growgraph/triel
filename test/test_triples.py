import logging
import os
import pkgutil
import sys
import unittest
from copy import deepcopy
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

    def test_relation(self):
        documents = [
            self.documents[key] for key in ["near-field", "cheops0_trunc"]
        ]
        acc_triples = []
        triples_projected = []
        for doc in documents:
            rdoc, graph = phrase_to_deptree(self.nlp, doc)

            triples, _ = graph_to_triples(graph, self.rules)
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
                ("europeanSpaceTelescope", "allows", "estimationOfTheirMass"),
            ],
        )

    def test_relation_advanced(self):
        rdoc, graph = phrase_to_deptree(self.nlp, self.documents["coref"])
        token_dict = {i: Token(**graph.nodes[i]) for i in graph.nodes()}

        coref_graph = render_coref_graph(rdoc, graph)
        (
            map_subbable_to_chain,
            map_chain_to_most_specific,
        ) = render_coref_candidate_map(coref_graph)

        map_token_specific_token = {
            i: sub_coreference(
                map_subbable_to_chain, map_chain_to_most_specific, i
            )
            for i in map_subbable_to_chain
        }

        map_trunc = {
            k: v for k, v in map_token_specific_token.items() if [k] != v
        }

        all_coref_i = set(map_trunc.keys()) | set(
            [i for subl in map_trunc.values() for i in subl]
        )
        map_icoref_source_target = {}

        triples, source_target_depot = graph_to_triples(graph, self.rules)
        triples = [tri.drop_articles().normalize_relation() for tri in triples]

        for s in source_target_depot:
            for k in all_coref_i:
                if k in s.itokens:
                    map_icoref_source_target[k] = deepcopy(s)
                elif k not in map_icoref_source_target:
                    ac = Candidate()
                    ac.append(token_dict[k])
                    map_icoref_source_target[k] = ac

        triples_expanded = []

        for tri in triples:
            source_ix_subs = set(map_trunc) & set(tri.source.itokens)
            new_sources = []
            for sub in source_ix_subs:
                iy_subs = map_trunc[sub]
                for y in iy_subs:
                    s = deepcopy(tri.source)
                    s.replace_token_with_acandidate(
                        sub, map_icoref_source_target[y]
                    )
                    new_sources.append(s)
            target_ix_subs = set(map_trunc) & set(tri.target.itokens)
            new_targets = []
            for sub in target_ix_subs:
                iy_subs = map_trunc[sub]
                for y in iy_subs:
                    t = deepcopy(tri.target)
                    t.replace_token_with_acandidate(
                        sub, map_icoref_source_target[y]
                    )
                    new_targets.append(t)
            triples_expanded += [
                TripleCandidate(source=s, target=t, relation=tri.relation)
                for s, t in product(
                    new_sources if new_sources else [tri.source],
                    new_targets if new_targets else [tri.target],
                )
            ]

        triples_projected = [tri.project_to_text() for tri in triples_expanded]

        print(triples_projected)

        # self.assertEqual(
        #     triples_projected,
        #     [
        #         ("CHEOPS", "be", "telescope"),
        #     ],
        # )


if __name__ == "__main__":
    unittest.main()
