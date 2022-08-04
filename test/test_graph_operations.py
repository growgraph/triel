import logging
import unittest

import spacy

from lm_service.graph import excise_node, phrase_to_deptree

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):

    nlp = spacy.load("en_core_web_trf")

    phrases = [
        "Peter would have caught the fish with a fishing rod, if not the"
        " darkness. ",
    ]

    def test_excise(self):
        rdoc, nx_graph = phrase_to_deptree(self.nlp, self.phrases[0])

        excise_node(nx_graph, 9)

        self.assertEqual(
            len(nx_graph.edges),
            14,
        )

        self.assertEqual(
            len(nx_graph.nodes),
            15,
        )


if __name__ == "__main__":
    unittest.main()
