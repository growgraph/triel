import logging

import pytest

from triel.onto import Candidate, Token

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def candidate():
    j = {
        "r0": 6,
        "_tokens": {
            (0, "024"): {
                "text": "estimation",
                "s": (0, "024"),
                "dep_": "dobj",
                "tag_": "NN",
                "lower": "estimation",
                "lemma": "estimation",
                "ent_iob": 2,
                "_level": 0,
                "label": "24-estimation-dobj-NN",
                "predecessors": [],
                "successors": [(0, "025"), (0, "023")],
            },
            (0, "023"): {
                "text": "the",
                "s": (0, "023"),
                "dep_": "det",
                "tag_": "DT",
                "lower": "the",
                "lemma": "the",
                "ent_iob": 2,
                "_level": 1,
                "label": "23-the-det-DT",
                "predecessors": [(0, "024")],
                "successors": [],
            },
            (0, "025"): {
                "text": "of",
                "s": (0, "025"),
                "dep_": "prep",
                "tag_": "IN",
                "lower": "of",
                "lemma": "of",
                "ent_iob": 2,
                "_level": 1,
                "label": "25-of-prep-IN",
                "predecessors": [(0, "024")],
                "successors": [(0, "027")],
            },
            (0, "027"): {
                "text": "mass",
                "s": (0, "027"),
                "dep_": "pobj",
                "tag_": "NN",
                "lower": "mass",
                "lemma": "mass",
                "ent_iob": 2,
                "_level": 2,
                "label": "27-mass-pobj-NN",
                "predecessors": [(0, "025")],
                "successors": [(0, "028"), (0, "026"), (0, "029")],
            },
            (0, "026"): {
                "text": "their",
                "s": (0, "026"),
                "dep_": "poss",
                "tag_": "PRP$",
                "lower": "their",
                "lemma": "their",
                "ent_iob": 2,
                "_level": 3,
                "label": "26-their-poss-PRP$",
                "predecessors": [(0, "027")],
                "successors": [],
            },
            (0, "028"): {
                "text": ",",
                "s": (0, "028"),
                "dep_": "punct",
                "tag_": ",",
                "lower": ",",
                "lemma": ",",
                "ent_iob": 2,
                "_level": 3,
                "label": "28-,-punct-,",
                "predecessors": [(0, "027")],
                "successors": [],
            },
            (0, "029"): {
                "text": "density",
                "s": (0, "029"),
                "dep_": "conj",
                "tag_": "NN",
                "lower": "density",
                "lemma": "density",
                "ent_iob": 2,
                "_level": 3,
                "label": "29-density-conj-NN",
                "predecessors": [(0, "027")],
                "successors": [(0, "030"), (0, "031")],
            },
            (0, "030"): {
                "text": ",",
                "s": (0, "030"),
                "dep_": "punct",
                "tag_": ",",
                "lower": ",",
                "lemma": ",",
                "ent_iob": 2,
                "_level": 4,
                "label": "30-,-punct-,",
                "predecessors": [(0, "029")],
                "successors": [],
            },
            (0, "031"): {
                "text": "composition",
                "s": (0, "031"),
                "dep_": "conj",
                "tag_": "NN",
                "lower": "composition",
                "lemma": "composition",
                "ent_iob": 2,
                "_level": 4,
                "label": "31-composition-conj-NN",
                "predecessors": [(0, "029")],
                "successors": [(0, "034"), (0, "032")],
            },
            (0, "032"): {
                "text": "and",
                "s": (0, "032"),
                "dep_": "cc",
                "tag_": "CC",
                "lower": "and",
                "lemma": "and",
                "ent_iob": 2,
                "_level": 5,
                "label": "32-and-cc-CC",
                "predecessors": [(0, "031")],
                "successors": [],
            },
            (0, "034"): {
                "text": "formation",
                "s": (0, "034"),
                "dep_": "conj",
                "tag_": "NN",
                "lower": "formation",
                "lemma": "formation",
                "ent_iob": 2,
                "_level": 5,
                "label": "34-formation-conj-NN",
                "predecessors": [(0, "031")],
                "successors": [(0, "033")],
            },
            (0, "033"): {
                "text": "their",
                "s": (0, "033"),
                "dep_": "poss",
                "tag_": "PRP$",
                "lower": "their",
                "lemma": "their",
                "ent_iob": 2,
                "_level": 6,
                "label": "33-their-poss-PRP$",
                "predecessors": [(0, "034")],
                "successors": [],
            },
        },
        "_indexVec": [
            (0, "023"),
            (0, "024"),
            (0, "025"),
            (0, "026"),
            (0, "027"),
            (0, "028"),
            (0, "029"),
            (0, "030"),
            (0, "031"),
            (0, "032"),
            (0, "033"),
            (0, "034"),
        ],
        "_root": (0, "024"),
    }
    c = Candidate.from_dict(j)
    return c


def test_replace_with_candidate():
    tokens = [
        Token(
            **{
                "s": (0, "007"),
                "lower": "his",
                "text": "his",
                "dep_": "poss",
                "predecessors": {(0, "008")},
            }
        ),
        Token(
            **{
                "s": (0, "008"),
                "lower": "dog",
                "text": "dog",
                "successors": {(0, "007")},
            }
        ),
    ]
    ac = Candidate().from_tokens(tokens)

    tokens_b = [
        Token(**{"s": (0, "015"), "lower": "john", "text": "John"}),
    ]
    bc = Candidate().from_tokens(tokens_b)

    ac.replace_token_with_acandidate((0, "007"), bc)
    ac.sort_index()
    assert ac.stokens == [(0, "008"), (0, "008a"), (0, "015")]
    assert ac.token((0, "008")).successors == {(0, "008a")}


def test_replace_top():
    tokens = [
        Token(
            **{
                "s": 7,
                "lower": "he",
                "text": "he",
            }
        ),
    ]
    ac = Candidate().from_tokens(tokens)

    tokens_b = [
        Token(**{"s": 15, "lower": "john", "text": "John"}),
    ]
    bc = Candidate().from_tokens(tokens_b)

    ac.replace_token_with_acandidate("007", bc)
    assert ac._index_vec == ["015"]


def test_from_subtree():
    tokens = [
        Token(**{"s": 0, "text": "a0", "successors": {1, 2}}),
        Token(**{"s": 1, "text": "a1", "predecessors": {0}}),
        Token(
            **{
                "s": 2,
                "text": "a2",
                "predecessors": {0},
                "successors": {15},
            }
        ),
        Token(
            **{
                "s": 15,
                "text": "b0",
                "predecessors": {2},
                "successors": {16, 17},
            }
        ),
        Token(**{"s": 16, "text": "b1", "predecessors": {15}}),
        Token(**{"s": 17, "text": "b2", "predecessors": {15}}),
    ]
    ac = Candidate().from_tokens(tokens)

    bc = ac.from_subtree("002")

    assert bc.stokens == ["002", "015", "016", "017"]


def test_replace_with_candidate_extra(candidate):
    root_candidate = (
        Candidate()
        .from_tokens(
            [candidate.token((0, s)) for s in [Token.i2s(i + 23) for i in range(6)]]
        )
        .clean_dangling_edges()
    )
    child_a = (
        Candidate()
        .from_tokens(
            [candidate.token((0, s)) for s in [Token.i2s(i + 31) for i in range(2)]]
        )
        .clean_dangling_edges()
    )

    root_candidate.replace_token_with_acandidate(i=(0, "027"), ac=child_a)
    assert root_candidate.token((0, "025")).successors == {(0, "031")}
