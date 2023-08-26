import logging
import sys
import unittest

from pylatexenc.latex2text import LatexNodes2Text

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR, stream=sys.stdout)


class TestLEnc(unittest.TestCase):
    texts = [
        # "Check here: \href{http://github.com/A4Bio/RFold}{http://github.com/A4Bio/RFold}.",
        "We get 10 \pm 12",
        "We get $10 \pm 12$",
    ]

    def test_substitution_in_depot(self):
        for lt in self.texts:
            rtext = LatexNodes2Text().latex_to_text(lt)
            print(rtext)
            # self.assertEqual(ncp_test, ncp_ref)


if __name__ == "__main__":
    unittest.main()
