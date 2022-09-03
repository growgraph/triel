from __future__ import annotations

import dataclasses
from collections import deque
from copy import deepcopy

from lm_service.onto import (
    Candidate,
    CandidateType,
    SourceOrTarget,
    Token,
    TokenIndexT,
)


@dataclasses.dataclass(repr=False)
class CandidatePile:
    """
    pile of candidates of one type
    """

    _root_to_candidate: dict[TokenIndexT, CandidateType] = dataclasses.field(default_factory=dict)  # type: ignore

    def __len__(self) -> int:
        return len(self._root_to_candidate)

    def __getitem__(self, key) -> CandidateType:
        """

        :return: relation index in pile : relation tokens
        """

        return self._root_to_candidate[key]

    def __repr__(self):
        s = f""
        for k, v in self._root_to_candidate.items():
            s += f"{k} : {v.__repr__()} \n"
        return s

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
        return self._root_to_candidate.values()

    @property
    def roots(self) -> list[Token]:
        return [r.root for r in self.candidates]

    @property
    def sroots(self) -> list[TokenIndexT]:
        return list(self._root_to_candidate.keys())

    @property
    def tokens(self) -> set[TokenIndexT]:
        return set([x for r in self.candidates for x in r.stokens])

    def append(self, r: CandidateType):
        self._root_to_candidate[r.root.s] = r

    def project_to_text(self):
        return [
            c.project_to_text_str() for c in self._root_to_candidate.values()
        ]

    def drop_amod_vbn(self):
        new = deepcopy(self)
        new._root_to_candidate = {
            k: c.drop_amod_vbn() for k, c in new._root_to_candidate.items()
        }
        return new

    def drop_cc(self):
        new = deepcopy(self)
        new._root_to_candidate = {
            k: c.drop_cc() for k, c in new._root_to_candidate.items()
        }
        return new

    def drop_punct(self):
        new = deepcopy(self)
        new._root_to_candidate = {
            k: c.drop_punct() for k, c in new._root_to_candidate.items()
        }
        return new

    def drop_articles(self):
        new = deepcopy(self)
        new._root_to_candidate = {
            k: c.drop_articles() for k, c in new._root_to_candidate.items()
        }
        return new

    def normalize(self):
        new = deepcopy(self)
        new._root_to_candidate = {
            k: c.normalize() for k, c in new._root_to_candidate.items()
        }
        return new

    def clean_dangling_edges(self):
        new = deepcopy(self)
        new._root_to_candidate = {
            k: c.clean_dangling_edges()
            for k, c in new._root_to_candidate.items()
        }
        return new

    def sort_index(self):
        new = deepcopy(self)
        new._root_to_candidate = {
            k: c.sort_index() for k, c in new._root_to_candidate.items()
        }
        return new

    def promote_to_metaindex(self, i: int):
        """
        change index in all Candidates from s for f"{i}#{s}"
        :param i:
        :return:
        """

    # TODO move conj here
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
    deq: deque[tuple[TokenIndexT, TokenIndexT]],
    current_cand,
    accumulist: list[tuple[TokenIndexT, Candidate]],
    sparent0: TokenIndexT,
):
    """
    partition candidate into conjunctive pieces used DFS (depth first search)

    :param c: the original candidate that potentially contains multiple conj pieces
    :param deq: (!) the initial call should have only a single vertex in q
    :param current_cand: candidate to accumulate the conjunctive piece
    :param accumulist: list that accumulates [(iparent0, transformed Candidate)]
    :param sparent0: for each Candidate sparent0 is the index of parent graph vertex (NB : "/" < "[0,9]")

    :return:
    """

    if not deq:
        return
    stoken, sparent = deq.pop()

    if c.token(stoken).dep_ == "conj":
        current_cand = SourceOrTarget()
        sparent0 = sparent
    current_cand.append(c.token(stoken))

    if len(current_cand) == 1:
        accumulist.append((sparent0, current_cand))

    successors = [s for s in c.token(stoken).successors]

    for v in successors:
        deq.append((v, stoken))
        partition_conjunctive_dfs(
            c,
            deq,
            current_cand,
            accumulist,
            sparent0,
        )


def partition_conjunctive_wrapper(
    candidate: CandidateType,
) -> list[Candidate]:
    """

    :param candidate:
    :return:
    """

    # init partition_conjunctive_dfs parameters
    deq: deque[tuple[TokenIndexT, TokenIndexT]] = deque()

    # queue starts with a root
    deq.append((candidate.root.s, candidate.root.prior_s()))

    cand: SourceOrTarget = SourceOrTarget()
    accumulist: list[tuple[TokenIndexT, Candidate]] = []

    partition_conjunctive_dfs(
        c=candidate,
        deq=deq,
        current_cand=cand,
        accumulist=accumulist,
        sparent0=candidate.root.prior_s(),
    )

    # dangling edges appear during partition
    accumulist = [(x, y.clean_dangling_edges()) for x, y in accumulist]

    accumulist = sorted(accumulist, key=lambda x: x[0])
    acc: list[Candidate] = []

    (_, root_candidate), clauses = accumulist[0], accumulist[1:]

    acc.append(root_candidate)

    for _, candidate in clauses:
        sparent, _ = clauses[0]
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
