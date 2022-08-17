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
            self.iroot_to_candidate[r.root.i] = r

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
        return {r.r0: [x for x in r.itokens] for r in self.candidates}

    @property
    def roots(self) -> list[Token]:
        return [r.root for r in self.candidates]

    @property
    def iroots(self) -> list[int]:
        return [r.root.i for r in self.candidates]

    @property
    def tokens(self) -> set[int]:
        return set([x for r in self.candidates for x in r.itokens])

    def append(self, r: CandidateType, index=None):
        r.r0 = len(self.candidates)
        self.iroot_to_candidate[r.root.i if index is None else index] = r
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

    def sort_index(self):
        new = deepcopy(self)
        new._candidates = [c.sort_index() for c in new._candidates]
        return new

    def unfold_conjunction(self, graph):
        apile = CandidatePile()
        for c in self._candidates:
            accum = partition_conjunctive_wrapper(c, graph)
            accum = accum.sort_index().drop_punct().drop_cc()
            apile += accum
        return apile


@dataclasses.dataclass
class SRTPile:
    sources: CandidatePile
    targets: CandidatePile
    relations: CandidatePile


def partition_conjunctive_dfs(
    c: CandidateType,
    graph: nx.DiGraph,
    q: deque[tuple[int, int]],
    current_cand,
    accumulist: list[tuple[int, Candidate]],
    iparent0: int = -1,
):
    """
    :param c:
    :param graph:
    :param q: (!) the initial call should have only a single vertex in q
    :param current_cand:
    :param accumulist: list that accumulates [(iparent0, transformed Candidate)]
    :param iparent0: for each Candidate iparent0 is the index of parent graph vertex

    :return:
    """

    if not q:
        return
    itoken, iparent = q.pop()

    if c.token(itoken).dep_ == "conj":
        current_cand = Candidate()
        iparent0 = iparent
    current_cand.append(c.token(itoken))

    if len(current_cand) == 1:
        accumulist.append((iparent0, current_cand))

    successors = sorted(
        [i for i in graph.successors(itoken) if i in c.itokens]
    )

    for v in successors:
        q.append((v, itoken))
        partition_conjunctive_dfs(
            c, graph, q, current_cand, accumulist, iparent0
        )


def partition_conjunctive_wrapper(
    c: CandidateType, graph: nx.DiGraph
) -> CandidatePile:
    """

    :param c:
    :param graph:
    :return:
    """
    deq: deque = deque()
    deq.append((c.root.i, -1))
    cand: SourceOrTarget = SourceOrTarget()
    accumulist: list[tuple[int, Candidate]] = list()
    partition_conjunctive_dfs(c, graph, deq, cand, accumulist)

    accumulist = sorted(accumulist, key=lambda x: x[0])
    acc = CandidatePile()

    root_candidate = accumulist[0][1]

    prefix_candidate = Candidate()
    if len(accumulist) > 1:
        iparent = accumulist[1][0]
        for t in root_candidate.tokens:
            if t.i < iparent:
                prefix_candidate.append(t)

    for j, (k, c) in enumerate(accumulist):
        if j > 0 and not prefix_candidate.empty:
            c_prime = prefix_candidate + c
            acc.append(c_prime.sort_index().drop_cc().drop_punct())
        else:
            acc.append(c.sort_index().drop_cc().drop_punct())
    return acc
