import logging
import pkgutil
import sys
import unittest

import coreferee
import spacy
import yaml

from lm_service.coref import (
    render_coref_candidate_map,
    render_coref_graph,
    sub_coreference,
)
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

logging.basicConfig(level=logging.ERROR, stream=sys.stdout)


class TestCoref(unittest.TestCase):
    phrase = (
        "Although he was very busy with his work, Peter had had enough of it. "
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
            [[1], [6], [9], [16], [18]],
            [[7], [14]],
            [[16, 19], [21], [26], [31]],
            [[29], [34]],
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

        map_token_specific_token = {
            i: sub_coreference(
                map_subbable_to_chain, map_chain_to_most_specific, i
            )
            for i in map_subbable_to_chain
        }

        self.assertEqual(
            map_token_specific_token,
            {
                1: [9],
                6: [9],
                9: [9],
                16: [9],
                18: [9],
                7: [7],
                14: [7],
                19: [19],
                21: [9, 19],
                26: [9, 19],
                31: [9, 19],
                29: [29],
                34: [29],
            },
        )


if __name__ == "__main__":
    unittest.main()
