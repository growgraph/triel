import logging
import os
import pathlib
import pkgutil
import sys

import pytest
import yaml

from lm_service.piles import ExtCandidateList
from lm_service.relation import text_to_coref_sourcetarget
from lm_service.text import phrases_to_triples_stage_a

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.ERROR, stream=sys.stdout)

figs_folder = "./.figs"
current_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), figs_folder)
pathlib.Path(current_path).mkdir(parents=True, exist_ok=True)

path = pathlib.Path(__file__).parent
fig_path = os.path.join(path, figs_folder)

fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
rules = yaml.load(fp, Loader=yaml.FullLoader)


@pytest.fixture(scope="module")
def phrases():
    return (
        "Although he was very busy with his work, Peter Brown had had enough of it.",
        "He and his wife decided they needed a holiday.",
        "They travelled to Spain because they loved the country very much.",
    )


def test_substitution_in_depot(nlp_fixture, phrases):
    striples, striples_meta, relations, ext_cand_list, megagraph = (
        phrases_to_triples_stage_a(phrases, nlp_fixture, rules, plot_path=fig_path)
    )

    global_ecl = ExtCandidateList()

    window_size = 5
    window_size = min([window_size, len(phrases)])
    nmax = len(phrases) - window_size + 1
    for i in range(nmax):
        fragment = " ".join(phrases[i : i + window_size])
        ext_cand_list.set_filter(lambda x: i <= x[0] < i + window_size)
        ncp = text_to_coref_sourcetarget(
            nlp_fixture, fragment, ext_cand_list, initial_phrase_index=i
        )

        for key, candidate_list in ncp.items():
            for c in candidate_list:
                global_ecl.append(key, c)

    global_ecl.filter_out_pronouns()

    ncp_ref = [
        ((0, "001"), [[(0, "009"), (0, "010")]]),
        ((0, "007"), [[(0, "007"), (0, "007a"), (0, "009"), (0, "010")]]),
        ((0, "010"), [[(0, "009"), (0, "010")]]),
        ((0, "015"), [[(0, "007"), (0, "007a"), (0, "009"), (0, "010")]]),
        (
            (1, "000"),
            [
                [(0, "009"), (0, "010")],
                [(1, "003"), (1, "003a"), (0, "009"), (0, "010")],
            ],
        ),
        (
            (1, "005"),
            [
                [(0, "009"), (0, "010")],
                [(1, "003"), (1, "003a"), (0, "009"), (0, "010")],
            ],
        ),
        ((1, "008"), [[(1, "008")]]),
        (
            (2, "000"),
            [
                [(0, "009"), (0, "010")],
                [(1, "003"), (1, "003a"), (0, "009"), (0, "010")],
            ],
        ),
        ((2, "003"), [[(2, "003")]]),
        (
            (2, "005"),
            [
                [(0, "009"), (0, "010")],
                [(1, "003"), (1, "003a"), (0, "009"), (0, "010")],
            ],
        ),
        ((2, "008"), [[(2, "003")]]),
    ]
    ncp_test = [(k, [vv.stokens for vv in ncp[k]]) for k in sorted(ncp)]
    assert ncp_test == ncp_ref
