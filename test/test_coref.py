import logging
import os
import pathlib
import pkgutil
import sys
import unittest

import coreferee
import spacy
import yaml

from lm_service.piles import ExtCandidateList
from lm_service.relation import text_to_coref_sourcetarget
from lm_service.text import phrases_to_triples_stage_a

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR, stream=sys.stdout)


class TestCoref(unittest.TestCase):
    figs_folder = "./.figs"
    current_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), figs_folder
    )
    pathlib.Path(current_path).mkdir(parents=True, exist_ok=True)

    path = pathlib.Path(__file__).parent
    fig_path = os.path.join(path, figs_folder)

    phrases = (
        (
            "Although he was very busy with his work, Peter Brown had had"
            " enough of it."
        ),
        "He and his wife decided they needed a holiday.",
        "They travelled to Spain because they loved the country very much.",
    )

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    # def test_coref(self):
    #     doc = self.nlp(self.phrase)
    #
    #     gt_chains = (
    #         [[1], [6], [10], [17], [19]],
    #         [[7], [15]],
    #         [[17, 20], [22], [27], [32]],
    #         [[30], [35]],
    #     )
    #     for chain, gt in zip(doc._.coref_chains, gt_chains):
    #         logger.info(chain.most_specific_mention_index)
    #         logger.info(f"{chain.mentions}, {gt}")
    #         self.assertTrue(
    #             all([x.token_indexes == y for x, y in zip(chain.mentions, gt)])
    #         )
    #     doc._.coref_chains.print()
    #
    # def test_coref_substitution(self):
    #     doc = self.phrase
    #     rdoc, graph = phrase_to_deptree(self.nlp, doc)
    #
    #     (
    #         map_subbable_to_chain,
    #         map_chain_to_most_specific,
    #     ) = render_coref_maps_wrapper(rdoc)
    #
    #     map_tree_subtree_index = graph_component_maps(graph)
    #
    #     map_subbable_to_chain_str, map_chain_to_most_specific_str = apply_map(
    #         [map_subbable_to_chain, map_chain_to_most_specific],
    #         map_tree_subtree_index,
    #     )
    #
    #     map_token_specific_token = {
    #         i: sub_coreference(
    #             map_subbable_to_chain_str, map_chain_to_most_specific_str, i
    #         )
    #         for i in map_subbable_to_chain_str
    #     }
    #
    #     self.assertEqual(
    #         map_token_specific_token,
    #         {
    #             (0, 1): [(0, 10)],
    #             (0, 6): [(0, 10)],
    #             (0, 10): [(0, 10)],
    #             (1, 0): [(0, 10)],
    #             (1, 2): [(0, 10)],
    #             (0, 7): [(0, 7)],
    #             (0, 15): [(0, 7)],
    #             (1, 3): [(1, 3)],
    #             (1, 5): [(0, 10), (1, 3)],
    #             (2, 0): [(0, 10), (1, 3)],
    #             (2, 5): [(0, 10), (1, 3)],
    #             (2, 3): [(2, 3)],
    #             (2, 8): [(2, 3)],
    #         },
    #     )

    def test_substitution_in_depot(self):
        (
            striples,
            striples_meta,
            relations,
            ext_cand_list,
            megagraph,
        ) = phrases_to_triples_stage_a(
            self.phrases, self.nlp, self.rules, plot_path=self.fig_path
        )

        global_ecl = ExtCandidateList()

        window_size = 5
        window_size = min([window_size, len(self.phrases)])
        nmax = len(self.phrases) - window_size + 1
        for i in range(nmax):
            fragment = " ".join(self.phrases[i : i + window_size])
            ext_cand_list.set_filter(lambda x: i <= x[0] < i + window_size)
            ncp = text_to_coref_sourcetarget(
                self.nlp, fragment, ext_cand_list, initial_phrase_index=i
            )

            for key, candidate_list in ncp.items():
                for c in candidate_list:
                    global_ecl.append(key, c)

        global_ecl.filter_out_pronouns()

        ncp_ref = [
            ((0, "001"), [[(0, "009"), (0, "010")]]),
            ((0, "007"), [[(0, "007"), (0, "007a"), (0, "009"), (0, "010")]]),
            ((0, "010"), [[(0, "009"), (0, "010")]]),
            ((0, "015"), [[(0, "007"), (0, "007a"), (0, "009"), (0, "010")]]),
            (
                (1, "000"),
                [
                    [(0, "009"), (0, "010")],
                    [(1, "003"), (1, "003a"), (0, "009"), (0, "010")],
                ],
            ),
            (
                (1, "005"),
                [
                    [(0, "009"), (0, "010")],
                    [(1, "003"), (1, "003a"), (0, "009"), (0, "010")],
                ],
            ),
            ((1, "008"), [[(1, "008")]]),
            (
                (2, "000"),
                [
                    [(0, "009"), (0, "010")],
                    [(1, "003"), (1, "003a"), (0, "009"), (0, "010")],
                ],
            ),
            ((2, "003"), [[(2, "003")]]),
            (
                (2, "005"),
                [
                    [(0, "009"), (0, "010")],
                    [(1, "003"), (1, "003a"), (0, "009"), (0, "010")],
                ],
            ),
            ((2, "008"), [[(2, "003")]]),
        ]
        ncp_test = [(k, [vv.stokens for vv in ncp[k]]) for k in sorted(ncp)]
        self.assertEqual(ncp_test, ncp_ref)


if __name__ == "__main__":
    unittest.main()
