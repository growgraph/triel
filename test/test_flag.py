import logging
import pkgutil
import sys
import unittest
from pathlib import Path

import spacy
import yaml

from lm_service.folding import get_flag
from lm_service.onto import Token

logger = logging.getLogger(__name__)


class TestFlag(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules_v2 = yaml.load(fp, Loader=yaml.FullLoader)

    def test_flag(self):
        t = Token(**{"s": 7, "text": "his", "dep_": "conj", "tag_": "VBG"})
        flag = get_flag(t.__dict__, self.rules_v2["sourcetarget"]["secondary"])
        self.assertEqual(flag, True)


if __name__ == "__main__":
    unittest.main()
