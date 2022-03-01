import pkgutil
import yaml
import unittest
import spacy
from lm_service.relation import phrase_to_relations


class TestR(unittest.TestCase):
    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    add_dict_rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open("./data/cheops.txt", "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_sm")
    # nlp = spacy.load("en_core_web_trf")

    phrases = text.split(".")

    def test_relation(self):
        document = self.phrases[0]
        mg, r, rproj = phrase_to_relations(self.nlp, document, self.add_dict_rules)
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
