import logging
import pkgutil
import sys
import unittest
from pathlib import Path

import spacy
import yaml
from graph_cast.util import ResourceHandler

from lm_service.folding import get_flag
from lm_service.onto import Candidate, Token

logger = logging.getLogger(__name__)


class TestFlag(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    cand_json = ResourceHandler.load("test.data", "candidate_conj.json")
    c = Candidate.from_dict(cand_json)

    nlp = spacy.load("en_core_web_trf")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules_v2 = yaml.load(fp, Loader=yaml.FullLoader)

    def test_flag(self):
        t = Token(**{"i": 7, "text": "his", "dep_": "conj", "tag_": "VBG"})
        flag = get_flag(t.__dict__, self.rules_v2)
        self.assertEqual(flag, False)


if __name__ == "__main__":
    unittest.main()
