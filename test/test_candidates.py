"""
test candidate extraction
"""

import logging
from collections import deque

from lm_service.graph import phrase_to_deptree
from lm_service.onto import (
    ACandidateKind,
    Candidate,
    Relation,
    SourceOrTarget,
    Token,
)
from lm_service.piles import CandidatePile
from lm_service.relation import find_candidates_bfs, find_subtree_dfs

logger = logging.getLogger(__name__)


def test_relation_candidates(nlp_fixture, rules, documents):
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
            rules=rules,
        )
        piles += [rp]

    reference = {
        0: [["be", "affect", "by"]],
        1: [["be"], ["determine"], ["will", "allow"]],
        2: [["treat"]],
    }
    derived = {k: [c.lemmas for c in p.candidates] for k, p in enumerate(piles)}
    assert [len(rp) for rp in piles] == [1, 3, 1]
    assert derived == reference


def test_st_candidates(documents, nlp_fixture, rules):
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
            rules=rules,
        )
        piles += [rp.sort_index()]

    assert [len(rp) for rp in piles] == [2, 5, 2]

    assert {k: [c.lemmas for c in p.candidates] for k, p in enumerate(piles)} == {
        0: [
            ["the", "medium"],
            ["the", "near", "-", "field", "radiation"],
        ],
        1: [
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
        2: [["he"], ["she"]],
    }


def test_relation_subtree_dfs(documents, nlp_fixture, rules):
    piles = []
    vertices_of_interest = [3, 22, 1]
    vertices_of_interest = [deque([(x, 0)]) for x in vertices_of_interest]
    for deq, document in zip(vertices_of_interest, documents):
        rdoc, graph = phrase_to_deptree(nlp_fixture, document)
        ograph = graph.copy()
        cr = Relation()
        find_subtree_dfs(graph, ograph, deq, cr, rules=rules["relation"])
        cr.clean_dangling_edges().sort_index()
        piles += [cr]

    assert [len(rp) for rp in piles] == [3, 2, 1]

    assert {k: p.stokens for k, p in enumerate(piles)} == {
        0: ["002", "003", "004"],
        1: ["021", "022"],
        2: ["001"],
    }


def test_st_subtree_dfs(documents, nlp_fixture, rules):
    piles = []
    vertices_of_interest = [9, 24, 2]
    vertices_of_interest = [deque([(x, 0)]) for x in vertices_of_interest]
    for deq, document in zip(vertices_of_interest, documents):
        rdoc, graph = phrase_to_deptree(nlp_fixture, document)
        original_graph = graph.copy()
        st = SourceOrTarget()
        find_subtree_dfs(
            graph,
            original_graph,
            deq,
            st,
            rules=rules["sourcetarget"],
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


def test_consecutive_candidates(documents, nlp_fixture, rules):
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
            rules=rules,
        )
        sb = len(graph.nodes)
        find_candidates_bfs(
            graph,
            ograph,
            deque(roots),
            source_target_pile,
            ACandidateKind.SOURCE_TARGET,
            rules=rules,
        )
        sc = len(graph.nodes)
        sizes += [(sa, sb, sc)]
    assert sizes == [(10, 8, 3), (36, 35, 12), (5, 5, 5)]


def test_sort_index_tree(nlp_fixture, rules, documents):
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
