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
from lm_service.onto import Candidate, Token
from lm_service.relation import graph_to_candidate_pile
from lm_service.util import to_string

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR, stream=sys.stdout)


class TestCoref(unittest.TestCase):
    phrase = (
        "Although he was very busy with his work, Peter Brown had had enough of it. "
        "He and his wife decided they needed a holiday. "
        "They travelled to Spain because they loved the country very much."
    )

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
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
                "1": ["10"],
                "6": ["10"],
                "10": ["10"],
                "17": ["10"],
                "19": ["10"],
                "7": ["7"],
                "15": ["7"],
                "20": ["20"],
                "22": ["10", "20"],
                "27": ["10", "20"],
                "32": ["10", "20"],
                "30": ["30"],
                "35": ["30"],
            },
        )

    # @unittest.skip("")
    def test_substitution_in_depot(self):
        rdoc, graph = phrase_to_deptree(self.nlp, self.phrase)

        pile, candidate_depot, mod_graph = graph_to_candidate_pile(
            graph, self.rules
        )

        token_dict = {
            str(i): Token(
                **graph.nodes[i],
                successors=set(graph.successors(i)),
                predecessors=set(graph.predecessors(i)),
            )
            for i in graph.nodes()
        }
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

        ncp_test = {k: [vv.itokens for vv in v] for k, v in ncp.items()}
        self.assertEqual(
            ncp_test,
            {
                "10": [["10", "9"]],
                "23": [["23"]],
                "30": [["30"]],
                "25": [["25"]],
                "17": [["10", "9"], ["20", "20a", "10", "9"]],
                "27": [["10", "9"], ["20", "20a", "10", "9"]],
                "1": [["10", "9"]],
                "22": [["10", "9"], ["20", "20a", "10", "9"]],
                "32": [["10", "9"], ["20", "20a", "10", "9"]],
                "35": [["30"]],
                "7": [["7", "7a", "10", "9"]],
                "15": [["7", "7a", "10", "9"]],
            },
        )


if __name__ == "__main__":
    unittest.main()
