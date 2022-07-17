from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Set, Dict
from typing import TypeVar


class RelationHasNoTargetCandidatesError(Exception):
    pass


ACandidateType = TypeVar("ACandidateType", bound="ACandidate")


class ACandidate:
    def __init__(self):
        self.r0: Optional[int] = None
        self.passive: bool = False
        self._tokens: List[Token] = list()
        self.added: bool = False

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

    def __repr__(self):
        content = [f" {k} : {v}" for k, v in self.__dict__.items()]
        return f"Token fields:" + " |".join(content)


# @dataclass
class Relation(ACandidate):
    pass


# @dataclass
class Source(ACandidate):
    pass


# @dataclass
class Target(ACandidate):
    pass


@dataclass
class TripleCandidate:
    source: int
    relation: Relation
    target: int

    def project_to_text(self, graph):
        s = ACandidate.concretize(self.source, graph)
        t = ACandidate.concretize(self.target, graph)
        r = self.relation.project_to_text_str(graph)
        return s, r, t


class ACandidatePile:
    def __init__(self, candidates: List[Relation] | None = None):
        self.candidates: List[Relation] = (
            [] if candidates is None else candidates
        )

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def map(self) -> Dict[int, List[int]]:
        return {r.r0: [x for x in r.tokens] for r in self.candidates}

    @property
    def tokens(self) -> Set[int]:
        return set([x for r in self.candidates for x in r.tokens])

    def __repr__(self):
        return str(self.map)

    def __iter__(self):
        for r in self.candidates:
            yield r

    def append(self, r: Relation):
        r.r0 = len(self.candidates)
        self.candidates += [r]

    def __iadd__(self, rp: ACandidatePile):
        self.candidates += rp.candidates
        return self


@dataclass
class CandidatePile:
    sources: ACandidatePile
    targets: ACandidatePile
    relations: ACandidatePile
