import logging
import pkgutil
import sys
import unittest

import coreferee
import spacy
import yaml

from lm_service.coref import (
    coref_candidates,
    render_coref_candidate_map,
    render_coref_graph,
    render_coref_maps_wrapper,
    sub_coreference,
)
from lm_service.graph import phrase_to_deptree, transform_advcl
from lm_service.onto import Candidate, Token, to_string
from lm_service.relation import graph_to_candidate_pile

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR, stream=sys.stdout)


class TestCoref(unittest.TestCase):
    phrase = (
        "Although he was very busy with his work, Peter Brown had had enough of it. "
        "He and his wife decided they needed a holiday. "
        "They travelled to Spain because they loved the country very much."
    )

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    def test_coref(self):
        doc = self.nlp(self.phrase)

        gt_chains = (
            [[1], [6], [10], [17], [19]],
            [[7], [15]],
            [[17, 20], [22], [27], [32]],
            [[30], [35]],
        )
        for chain, gt in zip(doc._.coref_chains, gt_chains):
            logger.info(chain.most_specific_mention_index)
            logger.info(f"{chain.mentions}, {gt}")
            self.assertTrue(
                all([x.token_indexes == y for x, y in zip(chain.mentions, gt)])
            )
        doc._.coref_chains.print()

    def test_coref_substitution(self):
        doc = self.phrase
        rdoc, graph = phrase_to_deptree(self.nlp, doc)
        coref_graph = render_coref_graph(rdoc, graph)
        (
            map_subbable_to_chain,
            map_chain_to_most_specific,
        ) = render_coref_candidate_map(coref_graph)

        map_subbable_to_chain_str, map_chain_to_most_specific_str = to_string(
            [map_subbable_to_chain, map_chain_to_most_specific]
        )

        map_token_specific_token = {
            i: sub_coreference(
                map_subbable_to_chain_str, map_chain_to_most_specific_str, i
            )
            for i in map_subbable_to_chain_str
        }

        self.assertEqual(
            map_token_specific_token,
            {
                "001": ["010"],
                "006": ["010"],
                "010": ["010"],
                "017": ["010"],
                "019": ["010"],
                "007": ["007"],
                "015": ["007"],
                "020": ["020"],
                "022": ["010", "020"],
                "027": ["010", "020"],
                "032": ["010", "020"],
                "030": ["030"],
                "035": ["030"],
            },
        )

    def test_substitution_in_depot(self):
        rdoc, graph = phrase_to_deptree(self.nlp, self.phrase)

        pile, candidate_depot, mod_graph = graph_to_candidate_pile(
            graph, self.rules
        )

        tokens = [
            Token(
                **graph.nodes[i],
                successors=graph.successors(i),
                predecessors=graph.predecessors(i),
            )
            for i in graph.nodes()
        ]

        token_dict = {t.s: t for t in tokens}

        (
            map_subbable_to_chain,
            map_chain_to_most_specific,
        ) = render_coref_maps_wrapper(rdoc, graph)

        map_subbable_to_chain_str = to_string(map_subbable_to_chain)
        map_chain_to_most_specific_str = to_string(map_chain_to_most_specific)

        ncp = coref_candidates(
            graph,
            candidate_depot,
            map_subbable_to_chain_str,
            map_chain_to_most_specific_str,
            token_dict,
            unfold_conjunction=True,
        )

        ncp_test = {k: [vv.stokens for vv in v] for k, v in ncp.items()}
        self.assertEqual(
            ncp_test,
            {
                "010": [["009", "010"]],
                "023": [["023"]],
                "030": [["030"]],
                "025": [["025"]],
                "017": [["009", "010"], ["020", "020a", "009", "010"]],
                "027": [["009", "010"], ["020", "020a", "009", "010"]],
                "001": [["009", "010"]],
                "022": [["009", "010"], ["020", "020a", "009", "010"]],
                "032": [["009", "010"], ["020", "020a", "009", "010"]],
                "035": [["030"]],
                "007": [["007", "007a", "009", "010"]],
                "015": [["007", "007a", "009", "010"]],
            },
        )


if __name__ == "__main__":
    unittest.main()
