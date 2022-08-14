import logging
import os
import pkgutil
import sys
import unittest
from pathlib import Path

import spacy
import yaml

from lm_service.coref import (
    render_coref_candidate_map,
    render_coref_graph,
    sub_coreference,
)
from lm_service.graph import phrase_to_deptree, transform_advcl
from lm_service.onto import ACandidate, Token
from lm_service.preprocessing import normalize_input_text
from lm_service.relation import (
    add_hash,
    compute_distances,
    generate_extra_graphs,
    graph_to_candidate_pile,
    graph_to_relations,
    phrase_to_relations,
)

logger = logging.getLogger(__name__)


class TestR(unittest.TestCase):

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    path = Path(__file__).parent

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    with open(os.path.join(path, f"./data/cheops.txt"), "r") as f:
        text = f.read()

    nlp = spacy.load("en_core_web_trf")

    def test_acandidate_insert_end(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x+3}"}) for x in range(3)]
        ac = ACandidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac.insert_at(4, tokens_to_add)
        self.assertEqual(ac._index_set, [3, 4, 5, 15, 17])

    def test_acandidate_insert(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x+3}"}) for x in range(3)]
        ac = ACandidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac.insert_at(1, tokens_to_add)
        self.assertEqual(ac._index_set, [3, 15, 17, 4, 5])

    def test_acandidate_insert_with_token_index(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x + 3}"}) for x in range(3)]
        ac = ACandidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac.insert_at(5, tokens_to_add, token_index=True)
        self.assertEqual(ac._index_set, [3, 4, 15, 17, 5])

    def test_acandidate_replace(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x + 3}"}) for x in range(3)]
        ac = ACandidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac.replace_token_with_tokens(4, tokens_to_add)
        self.assertEqual(ac._index_set, [3, 15, 17, 5])

    def test_acandidate_replace_acandidate(self):
        tokens = [Token(**{"i": x + 3, "text": f"a{x + 3}"}) for x in range(3)]
        ac = ACandidate()
        for t in tokens:
            ac.append(t)

        tokens_to_add = [Token(**{"i": x, "text": f"b{x}"}) for x in [15, 17]]

        ac2 = ACandidate()
        for t in tokens_to_add:
            ac2.append(t)

        ac.replace_token_with_acandidate(4, ac2)
        self.assertEqual(ac._index_set, [3, 15, 17, 5])


if __name__ == "__main__":
    unittest.main()
