from __future__ import annotations

import dataclasses
from collections import deque
from copy import deepcopy

import networkx as nx

from lm_service.onto import Candidate, CandidateType, SourceOrTarget, Token


@dataclasses.dataclass(repr=False)
class CandidatePile:
    """
    pile of candidates of one type
    """

    iroot_to_candidate: dict[int, CandidateType] = dataclasses.field(default_factory=dict)  # type: ignore
    _candidates: list[CandidateType] = dataclasses.field(default_factory=list)  # type: ignore

    def __post_init__(self):
        for j, r in enumerate(self._candidates):
            r.r0 = j
            self.iroot_to_candidate[r.root.s] = r

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, key) -> CandidateType:
        """

        :return: relation index in pile : relation tokens
        """

        return self.iroot_to_candidate[key]

    def __repr__(self):
        return str(self.map)

    def __iter__(self):
        for r in self.candidates:
            yield r

    def __iadd__(self, rp: CandidatePile):
        """
        usage: pile_a += pile_b
        :param rp:
        :return:
        """
        for c in rp:
            self.append(c)
        return self

    @property
    def candidates(self):
        """

        :return: relation index in pile : relation tokens
        """
        return self._candidates

    @property
    def map(self) -> dict[int, list[int]]:
        """

        :return: relation index in pile : relation tokens
        """
        return {r.r0: [x for x in r.stokens] for r in self.candidates}

    @property
    def roots(self) -> list[Token]:
        return [r.root for r in self.candidates]

    @property
    def iroots(self) -> list[str]:
        return [r.root.s for r in self.candidates]

    @property
    def tokens(self) -> set[str]:
        return set([x for r in self.candidates for x in r.stokens])

    def append(self, r: CandidateType, index=None):
        r.r0 = len(self.candidates)
        self.iroot_to_candidate[r.root.s if index is None else index] = r
        self._candidates += [r]

    def project_to_text(self):
        return [c.project_to_text_str() for c in self._candidates]

    def drop_amod_vbn(self):
        new = deepcopy(self)
        new._candidates = [c.drop_amod_vbn() for c in new._candidates]
        return new

    def drop_cc(self):
        new = deepcopy(self)
        new._candidates = [c.drop_cc() for c in new._candidates]
        return new

    def drop_punct(self):
        new = deepcopy(self)
        new._candidates = [c.drop_punct() for c in new._candidates]
        return new

    def drop_articles(self):
        new = deepcopy(self)
        new._candidates = [c.drop_articles() for c in new._candidates]
        return new

    def normalize(self):
        new = deepcopy(self)
        new._candidates = [c.normalize() for c in new._candidates]
        return new

    def clean_dangling_edges(self):
        new = deepcopy(self)
        new._candidates = [c.clean_dangling_edges() for c in new._candidates]
        return new

    def sort_index(self):
        new = deepcopy(self)
        new._candidates = [c.sort_index() for c in new._candidates]
        return new

    # def unfold_conjunction(self, graph):
    #     apile = CandidatePile()
    #     for c in self._candidates:
    #         accum = partition_conjunctive_wrapper(c, graph)
    #         accum = accum.drop_punct().drop_cc().sort_index()
    #         apile += accum
    #     return apile


@dataclasses.dataclass
class SRTPile:
    sources: CandidatePile
    targets: CandidatePile
    relations: CandidatePile


def partition_conjunctive_dfs(
    c: CandidateType,
    graph: nx.DiGraph,
    deq: deque[tuple[int, int]],
    current_cand,
    accumulist: list[tuple[int, Candidate]],
    iparent0: int = -1,
):
    """
    partition candidate into conjunctive pieces used DFS (depth first search)

    :param c: the original candidate that potentially contains multiple conj pieces
    :param graph:
    :param deq: (!) the initial call should have only a single vertex in q
    :param current_cand: candidate to accumulate the conjunctive piece
    :param accumulist: list that accumulates [(iparent0, transformed Candidate)]
    :param iparent0: for each Candidate iparent0 is the index of parent graph vertex

    :return:
    """

    if not deq:
        return
    itoken, iparent = deq.pop()
    stoken = Token.i2s(itoken)

    if c.token(stoken).dep_ == "conj":
        current_cand = Candidate()
        iparent0 = iparent
    current_cand.append(c.token(stoken))

    if len(current_cand) == 1:
        accumulist.append((iparent0, current_cand))

    successors = [
        x for x in graph.successors(itoken) if Token.i2s(x) in c.stokens
    ]

    for v in successors:
        deq.append((v, itoken))
        partition_conjunctive_dfs(
            c, graph, deq, current_cand, accumulist, iparent0
        )


def partition_conjunctive_wrapper(
    candidate: CandidateType, graph: nx.DiGraph
) -> CandidatePile:
    """

    :param candidate:
    :param graph:
    :return:
    """

    # init partition_conjunctive_dfs parameters
    deq: deque = deque()

    # queue starts with a root
    deq.append((candidate.root.i, -1))

    cand: SourceOrTarget = SourceOrTarget()
    accumulist: list[tuple[int, Candidate]] = []
    # NB check last arg (!)
    partition_conjunctive_dfs(candidate, graph, deq, cand, accumulist, -1)

    # dangling edges appear during partition
    accumulist = [(x, y.clean_dangling_edges()) for x, y in accumulist]

    accumulist = sorted(accumulist, key=lambda x: x[0])
    acc = CandidatePile()

    (_, root_candidate), clauses = accumulist[0], accumulist[1:]

    acc.append(root_candidate)

    for _, candidate in clauses:
        iparent, _ = clauses[0]
        sparent = Token.i2s(iparent)
        c_prime = deepcopy(root_candidate)
        c_prime.replace_token_with_acandidate(i=sparent, ac=candidate)
        acc.append(
            c_prime.drop_cc()
            .drop_punct()
            .drop_articles()
            .clean_dangling_edges()
            .sort_index()
        )
    return acc
