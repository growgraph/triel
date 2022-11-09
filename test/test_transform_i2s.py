"""
test candidate extraction
"""

import logging
import sys
import unittest

from lm_service.onto import AbsToken, apply_map, to_string

logger = logging.getLogger(__name__)


class TestTransforms(unittest.TestCase):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    def test_relation_candidates(self):
        i = 15
        s = AbsToken.i2s(i)
        itup = (0, 15)
        stup = AbsToken.ituple2stuple(itup)
        self.assertEqual(s, "015")
        self.assertEqual(stup, (0, "015"))

    def test_to_string(self):
        example = {(0, 10): [(1, 3)]}
        r = to_string(example, AbsToken.ituple2stuple)
        self.assertEqual(r, {(0, "010"): [(1, "003")]})


if __name__ == "__main__":
    unittest.main()
