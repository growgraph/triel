from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Set, Dict
from typing import TypeVar


class RelationHasNoTargetCandidatesError(Exception):
    pass


ACandidateType = TypeVar("ACandidateType", bound="ACandidate")


class ACandidate:
    def __init__(self):
        self.r0: int | None = None  # position in a CandidatePile
        self.root: Token | None = None  # index of root token
        self.passive: bool = False
        self._tokens: list[Token] = list()
        self.added: bool = False

    def __len__(self) -> int:
        return len(self._tokens)

    def max_level(self) -> int:
        return 0 if self.empty else max(t._level for t in self._tokens)

    @staticmethod
    def concretize(x, graph):
        # return lemma if not entity, otherwise return text
        return (
            graph.nodes[x]["lemma"]
            if not graph.nodes[x]["ent_iob"] not in (0, 2)
            else graph.nodes[x]["text"]
        )

    @property
    def tokens(self):
        return [t.i for t in self._tokens]

    @property
    def empty(self):
        return len(self._tokens) == 0

    @property
    def contains_vb(self):
        return any(t.tag_.startswith("VB") for t in self._tokens)

    def project_to_text(self, graph):
        ll = [ACandidate.concretize(r, graph) for r in self.tokens]
        return ll

    def project_to_text_str(self, graph):
        ll = self.project_to_text(graph)
        txt = "".join([ll[0]] + [x.capitalize() for x in ll[1:]])
        return txt

    def append(self, token: Token):
        if self.empty:
            self.root = token
        self._tokens += [token]

    def prepend(self, token: Token):
        self._tokens = [token] + self._tokens

    def __repr__(self):
        content = [
            f" {t.i} : {t.lower} : {t.tag_} : {t.dep_}" for t in self._tokens
        ]
        return (
            f"{self.__class__.__name__} tokens : (" + " |".join(content) + ")"
        )

    def sort(self):
        self._tokens = sorted(self._tokens, key=lambda x: x.i)


class Token:
    """
    represents a token in dep tree
    """

    def __init__(self, i, **kwargs):
        self.i: int = i
        self.dep_: str = kwargs.get("dep_", "")
        self.tag_: str = kwargs.get("tag_", "")
        self.lower: str = kwargs.get("lower", "")
        self.lemma: str = kwargs.get("lemma", "")
        self._level: int = kwargs.get("_level", 0)

    def __repr__(self):
        content = [f" {k} : {v}" for k, v in self.__dict__.items()]
        return f"Token fields:" + " |".join(content)


class ACandidateKind(Enum):
    RELATION = 1
    SOURCE_TARGET = 2
    SOURCE = 3
    TARGET = 4


class Relation(ACandidate):
    pass


class SourceOrTarget(ACandidate):
    pass


class Source(ACandidate):
    pass


class Target(ACandidate):
    pass


@dataclass
class TripleCandidate:
    source: Source
    relation: Relation
    target: Target

    def project_to_text(self, graph):
        s = ACandidate.concretize(self.source, graph)
        t = ACandidate.concretize(self.target, graph)
        r = self.relation.project_to_text_str(graph)
        return s, r, t


class ACandidatePile:
    def __init__(self, candidates: List[ACandidateType] | None = None):
        self.iroot_to_candidate = {}
        self._candidates: List[ACandidateType] = []
        if candidates is not None:
            for c in candidates:
                self.append(c)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, key) -> ACandidateType:
        """

        :return: relation index in pile : relation tokens
        """

        return self.iroot_to_candidate[key]

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
        return {r.r0: [x for x in r.tokens] for r in self.candidates}

    @property
    def roots(self) -> list[Token]:
        return [r.root for r in self.candidates]

    @property
    def iroots(self) -> list[int]:
        return [r.root.i for r in self.candidates]

    @property
    def tokens(self) -> Set[int]:
        return set([x for r in self.candidates for x in r.tokens])

    def __repr__(self):
        return str(self.map)

    def __iter__(self):
        for r in self.candidates:
            yield r

    def append(self, r: ACandidateType):
        r.r0 = len(self.candidates)
        self.iroot_to_candidate[r.root.i] = r
        self._candidates += [r]

    def __iadd__(self, rp: ACandidatePile):
        """
        usage: pile_a += pile_b
        :param rp:
        :return:
        """
        for c in rp:
            self.append(c)
        return self


@dataclass
class CandidatePile:
    sources: ACandidatePile
    targets: ACandidatePile
    relations: ACandidatePile
