"""
test candidate extraction
"""

import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path

import spacy
import yaml

from lm_service.graph import phrase_to_deptree
from lm_service.preprocessing import normalize_input_text, transform_advcl
from lm_service.util import plot_graph

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    figs_folder = "./figs"
    figs_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), figs_folder
    )
    Path(figs_path).mkdir(parents=True, exist_ok=True)

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")

    phrases = normalize_input_text(text, terminal_full_stop=False)
    documents = [
        "The medium was affected by the near-field radiation",
        (
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass, density,"
            " composition and their formation."
        ),
        (
            "Launched on 18 December 2019, it is the first Small-class mission"
            " in ESA's Cosmic Vision science programme."
        ),
    ]

    def test_transform_advcl(self):
        phrase = (
            "Launched on 18 December 2019, "
            "it is the first Small-class mission in "
            "ESA's Cosmic Vision science programme."
        )
        out = transform_advcl(self.nlp, phrase)
        self.assertEqual(
            out,
            [
                "It is the first Small - class mission in ESA 's Cosmic Vision"
                " science programme launched on 18 December 2019 ."
            ],
        )


if __name__ == "__main__":
    unittest.main()
