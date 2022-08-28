import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path

import coreferee
import spacy
import yaml

from lm_service.coref import render_coref_maps_wrapper
from lm_service.graph import phrase_to_deptree, transform_advcl
from lm_service.preprocessing import normalize_input_text
from lm_service.relation import graph_to_candidate_pile

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    nlp = spacy.load("en_core_web_trf")

    phrases = normalize_input_text(text, terminal_full_stop=True)

    def test_indexing(self):
        fragment = self.phrases[:4]
        fragment_joined = " ".join(fragment)
        for doc in fragment:
            rdoc, graph = phrase_to_deptree(self.nlp, doc)
            pile, candidate_depot, mod_graph = graph_to_candidate_pile(
                graph, self.rules
            )

        rdoc, graph = phrase_to_deptree(self.nlp, fragment_joined)
        (
            map_subbable_to_chain,
            map_chain_to_most_specific,
        ) = render_coref_maps_wrapper(rdoc, graph)

        # write function i_joined -> s_meta (index over phrases "p#001"
        # verify it


if __name__ == "__main__":
    unittest.main()
