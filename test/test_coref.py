import coreferee
import sys
import spacy
import unittest
import logging

logger = logging.getLogger(__name__)


class TestCoref(unittest.TestCase):
    logging.basicConfig(level=logging.ERROR, stream=sys.stdout)

    def test_coref(self):
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("coreferee")
        doc = nlp(
            "Although he was very busy with his work, Peter had had enough of it. "
            "He and his wife decided they needed a holiday. "
            "They travelled to Spain because they loved the country very much."
        )

        gt_chains = (
            [[1], [6], [9], [16], [18]],
            [[7], [14]],
            [[16, 19], [21], [26], [31]],
            [[29], [34]],
        )
        for chain, gt in zip(doc._.coref_chains, gt_chains):
            logger.info(chain.most_specific_mention_index)
            logger.info(f"{chain.mentions}, {gt}")
            self.assertTrue(
                all([x.token_indexes == y for x, y in zip(chain.mentions, gt)])
            )
        doc._.coref_chains.print()


if __name__ == "__main__":
    unittest.main()
