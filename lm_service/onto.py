from __future__ import annotations

import dataclasses
from copy import deepcopy
from enum import Enum
from typing import List, Set, TypeVar

from lemminflect import getAllInflections, getInflection, getLemma


class MissingTokeninACandidate(Exception):
    pass


class RelationHasNoTargetCandidatesError(Exception):
    pass


class InsertingExistingTokens(Exception):
    pass


ACandidateType = TypeVar("ACandidateType", bound="ACandidate")


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
        self.text: str = kwargs.get("text", "")
        self.ent_iob: str = kwargs.get("ent_iob", 0)
        self._level: int = kwargs.get("_level", 0)

    def __repr__(self):
        content = [f" {k} : {v}" for k, v in self.__dict__.items()]
        return f"Token fields:" + " |".join(content)


class ACandidate:
    def __init__(self):
        self.r0: int | None = None  # position in a CandidatePile
        self._tokens: dict[int, Token] = dict()
        self._index_set: list[int] = list()
        self.added: bool = False
        self._root: int

    def __len__(self) -> int:
        return len(self._tokens)

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
            raise MissingTokeninACandidate(
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

    def __repr__(self):
        content = []
        for i in self.itokens:
            t = self._tokens[i]
            content += [f" {t.i} : {t.lower} : {t.tag_} : {t.dep_}"]

        return (
            f"{self.__class__.__name__} tokens : (" + " |".join(content) + ")"
        )

    def drop_tokens(self, drop_indices):
        dropping = {}
        for i in drop_indices:
            dropping[i] = self._tokens.pop(i)
        self._index_set = [i for i in self._index_set if i not in drop_indices]
        return dropping

    def sort_index(self):
        self._index_set = sorted(self._index_set)

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
                raise MissingTokeninACandidate(
                    f"token {j} not in ACandidate {self.itokens}"
                )
        self._index_set = (
            self._index_set[:j] + [t.i for t in tokens] + self._index_set[j:]
        )
        for t in tokens:
            self._tokens[t.i] = t

    def replace_token_with_tokens(self, i: int, tokens: list[Token]):
        if self.token(i).dep_ == "poss":
            tokens = [
                Token(
                    **{
                        "i": max(self.itokens) + 13,
                        "text": "of",
                        "lemma": "of",
                        "dep_": "prep",
                        "tag_": "IN",
                    }
                )
            ] + tokens
            self.insert_at(len(self) + 1, tokens, token_index=False)
        else:
            self.insert_at(i, tokens, token_index=True)
        j = self._index_set.index(i)
        self._index_set = self._index_set[:j] + self._index_set[j + 1 :]
        del self._tokens[i]

    def replace_token_with_acandidate(self, i: int, ac: ACandidate):
        self.replace_token_with_tokens(i, list(ac.tokens))

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


class ACandidateKind(Enum):
    RELATION = 1
    SOURCE_TARGET = 2
    SOURCE = 3
    TARGET = 4


class Relation(ACandidate):
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


class SourceOrTarget(ACandidate):
    def drop_articles(self):
        # t.dep_ == "det" or t.tag_ != "DT"
        drop_aux_indices = [
            j for j, t in self._tokens.items() if t.dep_ == "det"
        ]
        self.drop_tokens(drop_aux_indices)

    def drop_amod_vbn(self):
        drop_aux_indices = [
            j
            for j, t in self._tokens.items()
            if (t.dep_ == "amod" and t.tag_ == "VBN")
        ]
        self.drop_tokens(drop_aux_indices)


class Source(SourceOrTarget):
    pass


class Target(SourceOrTarget):
    pass


@dataclasses.dataclass
class TripleCandidate:
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

    def drop_articles(self):
        new = deepcopy(self)
        new.source.drop_articles()
        new.target.drop_articles()
        return new

    def normalize_relation(self):
        self.relation.normalize()
        return self

    def __repr__(self):
        s = f"\t{self.source}\n\t{self.relation}\n\t{self.target}\n"
        return s


class ACandidatePile:
    def __init__(self, candidates: List[ACandidateType] | None = None):
        self.iroot_to_candidate: dict[int, ACandidateType] = {}
        self._candidates: list[ACandidateType] = []
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
        return {r.r0: [x for x in r.itokens] for r in self.candidates}

    @property
    def roots(self) -> list[Token]:
        return [r.root for r in self.candidates]

    @property
    def iroots(self) -> list[int]:
        return [r.root.i for r in self.candidates]

    @property
    def tokens(self) -> Set[int]:
        return set([x for r in self.candidates for x in r.itokens])

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


@dataclasses.dataclass
class CandidatePile:
    sources: ACandidatePile
    targets: ACandidatePile
    relations: ACandidatePile
