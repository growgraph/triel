"""
test candidate extraction
"""

import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path
from pprint import pprint

import spacy
import yaml

from lm_service.preprocessing import normalize_input_text, pivot_around_advcl

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    figs_folder = "./figs"
    figs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), figs_folder)
    Path(figs_path).mkdir(parents=True, exist_ok=True)

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, "./data/cheops.txt"), "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")

    def test_normalize_input_text(self):
        documents = [
            (
                "The program is freely available at"
                " \\url{http://graphics.med.yale.edu/cgi-bin/lib_comp.pl}."
            ),
            (
                "Launched on 18 December 2019, it is the first Small-class"
                " mission in ESA's Cosmic Vision science programme."
            ),
        ]
        for d in documents:
            phrases = normalize_input_text(d, terminal_full_stop=False)
            print(len(phrases))
            pprint(phrases)

    def test_transform_advcl(self):
        phrase = (
            "Launched on 18 December 2019, "
            "it is the first Small-class mission in "
            "ESA's Cosmic Vision science programme."
        )
        out = pivot_around_advcl(self.nlp, phrase)
        self.assertEqual(
            out,
            [
                "It is the first Small - class mission in ESA 's Cosmic Vision"
                " science programme launched on 18 December 2019 ."
            ],
        )


if __name__ == "__main__":
    unittest.main()
