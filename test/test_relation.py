import spacy
import unittest
import yaml
from lm_service.relation import phrase_to_relations


class TestR(unittest.TestCase):
    with open("../lm_service/config/prune_noun_compound.yaml") as file:
        add_dict_rules = yaml.load(file, Loader=yaml.FullLoader)

    with open("./data/cheops.txt", "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_sm")

    phrases = text.split(".")

    def test_relation(self):
        document = self.phrases[0]
        acc = []
        # for j, document in enumerate(phrases[:]):
        #     print(j, document)
        mg, r, rproj = phrase_to_relations(self.nlp, document, self.add_dict_rules)
        self.assertEqual(
            rproj,
            [
                ["be", "CHEOPS", "telescope"],
                ["determine", "telescope", "size"],
                ["allow", "size", "estimation"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
