import pkgutil
import yaml
import sys
import os
import unittest
import spacy
import logging
from pathlib import Path
from lm_service.relation import graph_to_relations
from lm_service.graph import dep_tree_from_phrase

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    add_dict_rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_sm")

    phrases = text.split(".")

    def test_relation(self):
        document = self.phrases[0]
        rdoc, graph = dep_tree_from_phrase(self.nlp, document)

        mg, r, triples_projected, _ = graph_to_relations(graph, self.add_dict_rules)
        self.assertEqual(
            triples_projected,
            [
                ("CHEOPS", "be", "telescope"),
                ("telescope", "determine", "size"),
                ("size", "allow", "estimation"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
