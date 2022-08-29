import logging
import sys
import unittest

from lm_service.preprocessing import (
    normalize_input_text,
    split_tokens_into_phrases,
)

logger = logging.getLogger(__name__)


class TestPreprocessing(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    phrase = (
        "This corresponds to the transit of an Earth-sized planet "
        "orbiting a star of 0.9 R☉ in 60 days detected "
        "with a S/Ntransit >10 (100 ppm transit depth). "
        "For example, an Earth-size transit across a G star creates an 80 ppm depth. "
        "The different science objectives require 500 separate target pointings. "
        "Assuming 1 hour per pointing the mission duration is estimated at 1175 days or 3.2 years."
    )

    def test_consecutive_candidates(self):
        out = normalize_input_text(self.phrase)
        print(out)
        self.assertEqual(
            out,
            [
                "This corresponds to the transit of an Earth-sized planet orbiting a star of 0.9 R "
                "in 60 days detected with a S/Ntransit > 10 ( 100 ppm transit depth ) .",
                "For example , an Earth-size transit across a G star creates an 80 ppm depth .",
                "The different science objectives require 500 separate target pointings .",
                "Assuming 1 hour per pointing the mission duration is estimated at 1175 days or 3.2 years .",
            ],
        )


if __name__ == "__main__":
    unittest.main()
