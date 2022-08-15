import logging
import os
import pkgutil
import sys
import unittest
from copy import deepcopy
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
from lm_service.onto import Candidate, Token
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
            cp, mgraph = graph_to_candidate_pile(graph, rules=self.rules)

    def test_distances(self):
        for document in self.documents.values():
            rdoc, graph0 = phrase_to_deptree(self.nlp, document)
            pile, graph = graph_to_candidate_pile(graph0, self.rules)
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

            triples = graph_to_relations(graph, self.rules)
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

        triples = graph_to_relations(graph, self.rules)
        triples = [tri.drop_articles().normalize_relation() for tri in triples]

        for tri in triples:
            s = tri.source
            t = tri.target
            print(s, t)
            for k in all_coref_i:
                print(k, token_dict[k])
                if k in s.itokens:
                    map_icoref_source_target[k] = deepcopy(s)
                elif k in t.itokens:
                    map_icoref_source_target[k] = deepcopy(t)
                elif k not in map_icoref_source_target:
                    ac = Candidate()
                    ac.append(token_dict[k])
                    map_icoref_source_target[k] = ac

        triples_projected = []

        for tri in triples:
            subs_source = set(map_trunc) & set(tri.source.itokens)
            subs_target = set(map_trunc) & set(tri.target.itokens)
            for sub in subs_source:
                subbers = map_trunc[sub]
                if len(subbers) > 1:
                    print("###")
                print(f"before {tri.source} {id(tri.source)}")
                # WIP
                tri.source.replace_token_with_acandidate(
                    sub, map_icoref_source_target[subbers[0]]
                )
                print(f"after {tri.source} {id(tri.source)}")
            for sub in subs_target:
                subbers = map_trunc[sub]
                print(f"before {tri.target}")
                tri.target.replace_token_with_acandidate(
                    sub, map_icoref_source_target[subbers[0]]
                )
                print(f"after {tri.target}")

        triples_projected += [tri.project_to_text() for tri in triples]

        print(triples_projected)

        # self.assertEqual(
        #     triples_projected,
        #     [
        #         ("CHEOPS", "be", "telescope"),
        #     ],
        # )


if __name__ == "__main__":
    unittest.main()
