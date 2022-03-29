import pkgutil
import yaml
import sys
import unittest
import spacy
import logging
from lm_service.relation import phrase_to_relations, dep_tree_from_phrase

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    add_dict_rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open("./data/cheops.txt", "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_sm")
    # nlp = spacy.load("en_core_web_trf")

    phrases = text.split(".")

    def test_relation(self):
        document = self.phrases[0]
        _, graph = dep_tree_from_phrase(self.nlp, document)

        mg, r, rproj = phrase_to_relations(graph, self.add_dict_rules)
        self.assertEqual(
            rproj,
            [
                ["CHEOPS", "be", "telescope"],
                ["CHEOPS", "determine", "size"],
                ["size", "allow", "estimation"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
