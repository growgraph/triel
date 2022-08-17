import logging
import pkgutil
import sys
import unittest
from collections import defaultdict
from copy import deepcopy

import coreferee
import spacy
import yaml

from lm_service.coref import (
    render_coref_candidate_map,
    render_coref_graph,
    sub_coreference,
)
from lm_service.graph import phrase_to_deptree, transform_advcl
from lm_service.onto import (
    Candidate,
    CandidatePile,
    Token,
    partition_conjunctive_wrapper,
)
from lm_service.relation import graph_to_maps

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

        map_token_specific_token = {
            i: sub_coreference(
                map_subbable_to_chain, map_chain_to_most_specific, i
            )
            for i in map_subbable_to_chain
        }

        self.assertEqual(
            map_token_specific_token,
            {
                1: [10],
                6: [10],
                10: [10],
                17: [10],
                19: [10],
                7: [7],
                15: [7],
                20: [20],
                22: [10, 20],
                27: [10, 20],
                32: [10, 20],
                30: [30],
                35: [30],
            },
        )

    # @unittest.skip("")
    def test_substitution_in_depot(self):
        rdoc, graph = phrase_to_deptree(self.nlp, self.phrase)
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

        (
            pile,
            sources_per_relation,
            targets_per_relation,
            candidate_depot,
            g_undirected,
        ) = graph_to_maps(graph, self.rules)

        ncp = defaultdict(list)
        # unfold conjunction
        for c in candidate_depot:
            ncp[c.root.i].extend(partition_conjunctive_wrapper(c, graph))

        # itoken -> atomic candidate
        for iroot, sigmas in ncp.items():
            for sigma in sigmas:
                for k in all_coref_i:
                    if k in sigma.itokens:
                        map_icoref_source_target[k] = iroot, deepcopy(sigma)
                    elif k not in map_icoref_source_target:
                        ac = Candidate()
                        ac.append(token_dict[k])
                        map_icoref_source_target[k] = k, ac

        # map (iroot, coref_index) -> clean atomic candidate
        from collections import deque

        q = deque()
        for iroot, sigmas in ncp.items():
            for sigma in sigmas:
                q.append((iroot, sigma))

        ncp2 = defaultdict(list)
        cnt = 0
        while q and cnt < len(map_icoref_source_target) ** 2:
            cnt += 1
            iroot, sigma = q.popleft()
            candidate_ix_subs = set(map_trunc) & set(sigma.itokens)
            if candidate_ix_subs:
                for sub in candidate_ix_subs:
                    iy_subs = map_trunc[sub]
                    for j, y in enumerate(iy_subs):
                        s2 = deepcopy(sigma)
                        iroot_new, sigma_sub = map_icoref_source_target[y]
                        s2.replace_token_with_acandidate(sub, sigma_sub)
                        q.append((iroot, s2))
            else:
                ncp2[iroot] += [sigma]
        ncp2_test = {k: [vv.itokens for vv in v] for k, v in ncp2.items()}
        self.assertEqual(
            ncp2_test,
            {
                10: [[9, 10]],
                23: [[23]],
                30: [[30]],
                25: [[24, 25]],
                17: [[9, 10], [20, 33, 9, 10]],
                27: [[9, 10], [20, 33, 9, 10]],
                1: [[9, 10]],
                22: [[9, 10], [20, 33, 9, 10]],
                32: [[9, 10], [20, 33, 9, 10]],
                35: [[34, 30]],
                7: [[7, 20, 9, 10]],
                15: [[7, 20, 9, 10]],
            },
        )


if __name__ == "__main__":
    unittest.main()
