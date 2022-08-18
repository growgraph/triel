from __future__ import annotations

import dataclasses
from copy import deepcopy
from enum import Enum
from typing import TypeVar

from dataclass_wizard import JSONWizard
from lemminflect import getInflection, getLemma


class MissingTokenInACandidate(Exception):
    pass


class RelationHasNoTargetCandidatesError(Exception):
    pass


class InsertingExistingTokens(Exception):
    pass


CandidateType = TypeVar("CandidateType", bound="Candidate")


@dataclasses.dataclass(repr=False)
class Token(JSONWizard):
    """
    represents a token in dep tree
    """

    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    i: int
    text: str
    predecessors: set[int] = dataclasses.field(default_factory=set)
    successors: set[int] = dataclasses.field(default_factory=set)
    dep_: str = ""
    tag_: str = ""
    lower: str = ""
    lemma: str = ""
    ent_iob: str = ""
    _level: int = 0
    label: str = ""

    def __repr__(self):
        content = [f" {k} : {v}" for k, v in self.__dict__.items()]
        return f"Token fields:" + " |".join(content)


@dataclasses.dataclass(repr=False)
class Candidate(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    r0: int | None = None  # position in CandidatePile
    _tokens: dict[int, Token] = dataclasses.field(default_factory=dict)
    _index_set: list[int] = dataclasses.field(default_factory=list)
    _root: int | None = None

    def __add__(self, other: Candidate):
        new = deepcopy(self)
        for c in other.tokens:
            new.append(c)
        return new

    def __iadd__(self, other: Candidate):
        for c in other.tokens:
            self.append(c)
        return self

    def __len__(self) -> int:
        return len(self._tokens)

    def __repr__(self):
        content = []
        for i in self.itokens:
            t = self._tokens[i]
            content += [f" {t.i} : {t.lower} : {t.tag_} : {t.dep_}"]

        return (
            f"{self.__class__.__name__} tokens : (" + " |".join(content) + ")"
        )

    def max_level(self) -> int:
        return (
            0 if self.empty else max(t._level for t in self._tokens.values())
        )

    @property
    def root(self):
        return self._tokens[self._root]

    @property
    def itokens(self):
        return self._index_set

    def token(self, i):
        if i in self._index_set:
            return self._tokens[i]
        else:
            raise MissingTokenInACandidate(
                f"token {i} not present in ACandidate containing {self.itokens}"
            )

    @property
    def tokens(self):
        return (self.token(i) for i in self.itokens)

    @property
    def empty(self):
        return len(self._tokens) == 0

    @property
    def contains_vb(self):
        return any(t.tag_.startswith("VB") for t in self._tokens.values())

    def project_to_text(self):
        """
            see https://spacy.io/api/token#attributes
            if entity - return text, otherwise return lemma
        :return:
        """
        pp = []
        for i in self.itokens:
            x = self.token(i)
            if x.dep_ == "punct":
                continue
            if x.ent_iob in (0, 2):
                pp += [x.text]
            else:
                pp += [x.lemma]
        return pp

    def project_to_text_str(self):
        ll = self.project_to_text()
        if ll:
            txt = "".join([ll[0]] + [x.capitalize() for x in ll[1:]])
        else:
            txt = ""
        return txt

    def append(self, token: Token):
        if self.empty:
            self._root = token.i
        self._tokens[token.i] = token
        self._index_set.append(token.i)

    def drop_tokens(self, drop_indices):
        dropping = {}
        for i in drop_indices:
            dropping[i] = self._tokens.pop(i)
        self._index_set = [i for i in self._index_set if i not in drop_indices]
        return dropping

    def sort_index(self):
        self._index_set = sorted(self._index_set)
        return self

    def insert_at(self, j: int, tokens: list[Token], token_index=False):
        """

        :param j:
        :param tokens:
        :param token_index: treat j as token index if true, token position in ACandidate if false
        :return:
        """
        # if set(t.i for t in tokens) & set(self.itokens):
        #     raise InsertingExistingTokens(f"{[t.i for t in tokens]} vs {self.itokens}")

        if token_index:
            if j in self._index_set:
                j = self._index_set.index(j)
            else:
                raise MissingTokenInACandidate(
                    f"token {j} not in ACandidate {self.itokens}"
                )
        self._index_set = (
            self._index_set[:j] + [t.i for t in tokens] + self._index_set[j:]
        )
        for t in tokens:
            self._tokens[t.i] = t

    def replace_token_with_acandidate(self, i: int, ac: Candidate):
        if self.token(i).dep_ == "poss":
            predecessor_i = next(iter(self.token(i).predecessors))
            of_token = Token(
                **{
                    "i": max(self.itokens) + 99,
                    "lower": "of",
                    "text": "of",
                    "lemma": "of",
                    "dep_": "prep",
                    "tag_": "IN",
                    "successors": {ac.root.i},
                    "predecessors": {predecessor_i},
                }
            )
            # operate on pred i
            self.token(predecessor_i).successors.remove(i)
            self.token(predecessor_i).successors.add(of_token.i)

            position = len(self) + 1
            token_index = False
        else:
            position = i
            token_index = True
        # TODO
        # ac.root.predecessor = of_token.i or position ??
        self.insert_at(position, ac.tokens, token_index=token_index)
        j = self._index_set.index(i)
        self._index_set = self._index_set[:j] + self._index_set[j + 1 :]
        del self._tokens[i]

    def print(self):
        content = []
        for i in self.itokens:
            t = self._tokens[i]
            content += [f" {t.text}"]

        return (
            f"{self.__class__.__name__} tokens : (" + " |".join(content) + ")"
        )

    @property
    def lemmas(self):
        return [self._tokens[k].lemma for k in self.itokens]

    def drop_articles(self):
        # t.dep_ == "det" or t.tag_ != "DT"
        drop_aux_indices = [
            j for j, t in self._tokens.items() if t.dep_ == "det"
        ]
        self.drop_tokens(drop_aux_indices)
        return self

    def drop_amod_vbn(self):
        drop_aux_indices = [
            j
            for j, t in self._tokens.items()
            if (t.dep_ == "amod" and t.tag_ == "VBN")
        ]
        self.drop_tokens(drop_aux_indices)
        return self

    def drop_cc(self):
        drop_aux_indices = [
            j
            for j, t in self._tokens.items()
            if (t.dep_ == "cc" and t.tag_ == "CC")
        ]
        self.drop_tokens(drop_aux_indices)
        return self

    def drop_punct(self):
        drop_aux_indices = [
            j for j, t in self._tokens.items() if t.dep_ == "punct"
        ]
        self.drop_tokens(drop_aux_indices)
        return self


class ACandidateKind(Enum):
    RELATION = 1
    SOURCE_TARGET = 2
    SOURCE = 3
    TARGET = 4


@dataclasses.dataclass(repr=False)
class Relation(Candidate):
    @property
    def passive(self):
        return any([t.dep_ == "auxpass" for t in self._tokens.values()])

    def normalize(self):
        if self.passive:
            # find auxpass, inflect it to was
            for i in self.itokens:
                token = self.token(i)
                if token.dep_ == "auxpass":
                    lemmas = getLemma(token.text, upos="VERB")
                    if lemmas:
                        inflected = getInflection(lemmas[0], tag="VBD")
                        if inflected:
                            token.text = inflected[0]
        else:
            # drop all aux
            drop_aux_indices = [
                j for j, t in self._tokens.items() if t.dep_ == "aux"
            ]
            self.drop_tokens(drop_aux_indices)
            # inflect remaining VBs
            for i, t in self._tokens.items():
                if t.tag_.startswith("VB"):
                    lemmas = getLemma(t.text, upos="VERB")
                    if lemmas:
                        inflected = getInflection(lemmas[0], tag="VBZ")
                        if inflected:
                            self._tokens[i].text = inflected[0]


@dataclasses.dataclass(repr=False)
class SourceOrTarget(Candidate):
    pass


@dataclasses.dataclass(repr=False)
class Source(SourceOrTarget):
    pass


@dataclasses.dataclass(repr=False)
class Target(SourceOrTarget):
    pass


@dataclasses.dataclass(repr=False)
class TripleCandidate(JSONWizard):
    source: Source
    relation: Relation
    target: Target

    def project_to_text(self):
        return (
            self.source.project_to_text_str(),
            self.relation.project_to_text_str(),
            self.target.project_to_text_str(),
        )

    def drop_amod_vbn(self):
        new = deepcopy(self)
        new.source.drop_amod_vbn()
        new.target.drop_amod_vbn()
        return new

    def drop_cc(self):
        new = deepcopy(self)
        new.source.drop_cc()
        new.target.drop_cc()
        return new

    def drop_punct(self):
        new = deepcopy(self)
        new.source.drop_punct()
        new.target.drop_punct()
        return new

    def drop_articles(self):
        new = deepcopy(self)
        new.source.drop_articles()
        new.target.drop_articles()
        return new

    def normalize_relation(self):
        self.relation.normalize()
        return self

    def __repr__(self):
        s = f"\t{self.source.__repr__()}\n\t{self.relation.__repr__()}\n\t{self.target.__repr__()}\n"
        return s
