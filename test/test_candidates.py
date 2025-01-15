"""
test candidate extraction
"""

import logging
from collections import deque

from triel.graph import phrase_to_deptree
from triel.onto import (
    ACandidateKind,
    Candidate,
    Relation,
    SourceOrTarget,
    Token,
)
from triel.piles import CandidatePile
from triel.relation import find_candidates_bfs, find_subtree_dfs

logger = logging.getLogger(__name__)


def test_relation_candidates(nlp_fixture, rules_v3, documents):
    piles = []
    for document in documents:
        rdoc, graph = phrase_to_deptree(nlp_fixture, document)
        ograph = graph.copy()
        roots = [n for n, d in graph.in_degree() if d == 0]
        rp = CandidatePile()
        find_candidates_bfs(
            graph,
            ograph,
            deque(roots),
            rp,
            ACandidateKind.RELATION,
            rules=rules_v3,
        )
        piles += [rp]

    reference = {
        0: [["can", "secrete", "in"], ["be", "able", "to", "suppress"]],
        1: [["be", "affect", "by"]],
        2: [["be"], ["to", "determine"], ["will", "allow"]],
        3: [["treat"]],
    }
    derived = {k: [c.lemmas for c in p.candidates] for k, p in enumerate(piles)}
    assert [len(rp) for rp in piles] == [2, 1, 3, 1]
    assert derived == reference


def test_st_candidates(documents, nlp_fixture, rules_v3):
    piles = []
    for document in documents:
        rdoc, graph = phrase_to_deptree(nlp_fixture, document)
        ograph = graph.copy()
        roots = [n for n, d in graph.in_degree() if d == 0]
        rp = CandidatePile()
        find_candidates_bfs(
            graph,
            ograph,
            deque(roots),
            rp,
            ACandidateKind.SOURCE_TARGET,
            rules=rules_v3,
        )
        piles += [rp.sort_index()]

    assert [len(rp) for rp in piles] == [5, 2, 5, 2]

    assert {k: [c.lemmas for c in p.candidates] for k, p in enumerate(piles)} == {
        0: [
            ["TAMs"],
            [
                "a",
                "number",
                "of",
                "immunosuppressive",
                "cytokine",
                ",",
                "such",
                "as",
                "il-6",
                ",",
                "tgf",
                "-",
                "β",
                ",",
                "and",
                "il-10",
            ],
            ["the", "TME"],
            ["t", "-", "cell", "function"],
            ["cd8", "+"],
        ],
        1: [["the", "medium"], ["the", "near", "-", "field", "radiation"]],
        2: [
            ["CHEOPS", "(", ")"],
            ["a", "european", "space", "telescope"],
            ["CHaracterising", "ExOPlanets", "Satellite"],
            ["the", "size", "of", "know", "extrasolar", "planet"],
            [
                "the",
                "estimation",
                "of",
                "their",
                "mass",
                ",",
                "density",
                ",",
                "composition",
                "and",
                "their",
                "formation",
            ],
        ],
        3: [["he"], ["she"]],
    }


def test_relation_subtree_dfs(documents, nlp_fixture, rules_v3):
    piles = []
    vertices_of_interest = [3, 22, 1]
    vertices_of_interest = [deque([(x, 0)]) for x in vertices_of_interest]
    for deq, document in zip(vertices_of_interest, documents[1:]):
        rdoc, graph = phrase_to_deptree(nlp_fixture, document)
        ograph = graph.copy()
        cr = Relation()
        find_subtree_dfs(graph, ograph, deq, cr, rules=rules_v3["relation"])
        cr.clean_dangling_edges().sort_index()
        piles += [cr]

    assert [len(rp) for rp in piles] == [3, 2, 1]

    assert {k: p.stokens for k, p in enumerate(piles)} == {
        0: ["002", "003", "004"],
        1: ["021", "022"],
        2: ["001"],
    }


def test_st_subtree_dfs(documents, nlp_fixture, rules_v3):
    piles = []
    vertices_of_interest = [9, 24, 2]
    vertices_of_interest = [deque([(x, 0)]) for x in vertices_of_interest]
    for deq, document in zip(vertices_of_interest, documents[1:]):
        rdoc, graph = phrase_to_deptree(nlp_fixture, document)
        original_graph = graph.copy()
        st = SourceOrTarget()
        find_subtree_dfs(
            graph,
            original_graph,
            deq,
            st,
            rules=rules_v3["source_target"],
        )
        st.sort_index()
        piles += [st]

    assert [len(rp) for rp in piles] == [5, 12, 1]

    assert {k: p.stokens for k, p in enumerate(piles)} == {
        0: ["005", "006", "007", "008", "009"],
        1: [
            "023",
            "024",
            "025",
            "026",
            "027",
            "028",
            "029",
            "030",
            "031",
            "032",
            "033",
            "034",
        ],
        2: ["002"],
    }


def test_consecutive_candidates(documents, nlp_fixture, rules_v3):
    """
    this test demonstrates that when a relation or a source/target are identified,
    the subgraph corresponding to it is excised, leaving only the root node
    :return:
    """

    sizes = []
    for document in documents:
        rdoc, graph = phrase_to_deptree(nlp_fixture, document)
        ograph = graph.copy()
        roots = [n for n, d in graph.in_degree() if d == 0]
        relation_pile = CandidatePile()
        source_target_pile = CandidatePile()
        sa = len(graph.nodes)
        find_candidates_bfs(
            graph,
            ograph,
            deque(roots),
            relation_pile,
            ACandidateKind.RELATION,
            rules=rules_v3,
        )
        sb = len(graph.nodes)
        find_candidates_bfs(
            graph,
            ograph,
            deque(roots),
            source_target_pile,
            ACandidateKind.SOURCE_TARGET,
            rules=rules_v3,
        )
        sc = len(graph.nodes)
        sizes += [(sa, sb, sc)]
    assert sizes == [(35, 30, 10), (10, 8, 3), (36, 34, 11), (5, 5, 5)]


def test_sort_index_tree(nlp_fixture, rules_v3, documents):
    tokens = [
        Token(**{"s": 2, "text": "a0", "successors": {1, 0}}),
        Token(**{"s": 1, "text": "a1", "predecessors": {2}}),
        Token(
            **{
                "s": 0,
                "text": "a2",
                "predecessors": {2},
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

    ac.sort_index()

    assert ac.stokens == ["000", "015", "016", "017", "001", "002"]
