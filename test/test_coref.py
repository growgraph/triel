import coreferee
import spacy
import unittest


class TestCoref(unittest.TestCase):
    def test_coref(self):
        # nlp = spacy.load("en_core_web_trf")
        nlp = spacy.load("en_core_web_sm")
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
            print(chain.most_specific_mention_index)
            print(chain.mentions, gt)
            self.assertTrue(
                all([x.token_indexes == y for x, y in zip(chain.mentions, gt)])
            )
        doc._.coref_chains.print()


if __name__ == "__main__":
    unittest.main()
