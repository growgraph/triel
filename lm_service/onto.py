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


class RequestedIndexDoesNotExist(Exception):
    pass


n_extra_token = 1000

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
    dep_: str = ""
    tag_: str = ""
    lower: str = ""
    lemma: str = ""
    ent_iob: str = ""
    _level: int = 0
    label: str = ""
    predecessors: set[int] = dataclasses.field(default_factory=set)
    successors: set[int] = dataclasses.field(default_factory=set)

    def __repr__(self):
        content = [f" {k} : {v}" for k, v in self.__dict__.items()]
        return f"Token fields:" + " |".join(content)


@dataclasses.dataclass(repr=False)
class Candidate(JSONWizard):
    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    r0: int | None = None  # position in CandidatePile
    _tokens: dict[int, Token] = dataclasses.field(default_factory=dict)
    _index_vec: list[int] = dataclasses.field(default_factory=list)
    _root: int | None = None

    def __add__(self, other: Candidate):
        new = deepcopy(self)
        for c in other.tokens:
            new.append(c)
        return new

    def __len__(self) -> int:
        return len(self._tokens)

    def __repr__(self):
        content = []
        for i in self.itokens:
            t = self._tokens[i]
            str_succ = ", ".join([f"{s}" for s in t.successors])
            str_pred = ", ".join([f"{s}" for s in t.predecessors])
            content += [
                f" {t.i} : {t.lower} : {t.tag_} : {t.dep_} : pred < {str_pred} : succ > {str_succ}"
            ]

        return (
            f"{self.__class__.__name__} tokens : (" + " |".join(content) + ")"
        )

    def from_tokens(self, tokens: list[Token]):
        for t in tokens:
            self.append(t)
        self.clean_dangling_edges()
        return self

    def clean_dangling_edges(self):
        present = set(self.itokens)
        for k, t in self._tokens.items():
            t.successors &= present
            t.predecessors &= present
        self._recompute_root()
        return self

    def max_level(self) -> int:
        return (
            0 if self.empty else max(t._level for t in self._tokens.values())
        )

    def _recompute_root(self):
        roots = [
            k for k, v in self._tokens.items() if len(v.predecessors) == 0
        ]
        if len(roots) != 1:
            print(self)
            raise ValueError(
                f" candidate has {len(roots)} roots, should be one"
            )
        else:
            self._root = next(iter(roots))

    @property
    def iroot(self):
        return self._root

    @property
    def root(self):
        return self._tokens[self._root]

    @property
    def itokens(self):
        return self._index_vec

    def token(self, i, index=False):
        if index:
            if i < len(self):
                return self._tokens[self._index_vec[i]]
            else:
                raise RequestedIndexDoesNotExist(
                    f" size of Candidate {len(self)}, requesting index {i}"
                )
        else:
            if i in self._index_vec:
                return self._tokens[i]
            else:
                raise MissingTokenInACandidate(
                    f"token {i} not present in ACandidate containing {self.itokens}"
                )

    def view_tokens(self, ifrom=None, ito=None):
        if ifrom in self._index_vec:
            iifrom = self._index_vec.index(ifrom)
        else:
            iifrom = None
        if ito in self._index_vec:
            iito = self._index_vec.index(ito) + 1
        else:
            iito = None
        return [self.token(i) for i in self.itokens[iifrom:iito]]

    def from_subtree(self, i: int):
        acc: list[int] = []
        self._pick_successors(i, acc)
        return (
            Candidate()
            .from_tokens([self.token(j) for j in self._index_vec if j in acc])
            .clean_dangling_edges()
        )

    def _pick_successors(self, i, acc):
        acc += [i]
        for j in self.token(i).successors:
            self._pick_successors(j, acc)

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
        self._index_vec.append(token.i)

    def drop_tokens(self, drop_indices):
        """
        NB: make it consistent wrt to graph operations
        """
        for i in drop_indices:
            self.remove(i)

    def sort_index(self):
        self._index_vec = sorted(self._index_vec)
        return self

    def _sort_wrt_tree(
        self,
        j: int,
        sorter: dict[int | None, tuple[float, int | None, int | None]],
    ):
        succs = self.token(j).successors

        # "of" exception => if j is "of" then force order
        if self.token(j).lower == "of" and self.token(j).tag_ == "IN":
            sorted_succs = [j] + sorted(succs)
        else:
            sorted_succs = sorted(list(succs) + [j])

        value, left, right = sorter[j]
        if left in sorter:
            value_left, _, _ = sorter[left]
        else:
            value_left = value - (len(sorted_succs) + 1)
        if right in sorter:
            value_right, _, _ = sorter[right]
        else:
            value_right = value + (len(sorted_succs) + 1)

        delta = (value_right - value_left) / (len(sorted_succs) + 1)

        values = [
            value_left + (k + 1) * delta for k in range(len(sorted_succs))
        ]
        working_sorted = [left] + sorted_succs + [right]  # type: ignore
        for a, b, c, value in zip(
            working_sorted, working_sorted[1:], working_sorted[2:], values
        ):
            sorter[b] = value, a, c
        for s in succs:
            self._sort_wrt_tree(s, sorter)

    def sort_index_tree(self):
        proposed_sorter = {self.iroot: (self.iroot, None, None)}
        self._sort_wrt_tree(self.iroot, sorter=proposed_sorter)
        self._index_vec = sorted(proposed_sorter, key=proposed_sorter.get)
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
            if j in self._index_vec:
                j = self._index_vec.index(j)
            else:
                raise MissingTokenInACandidate(
                    f"token {j} not in ACandidate {self.itokens}"
                )
        self._index_vec = (
            self._index_vec[:j] + [t.i for t in tokens] + self._index_vec[j:]
        )
        for t in tokens:
            self._tokens[t.i] = t

    def insert_before(self, ac: Candidate, j: int):
        """
            extend self with ac candidate
            such that jpred -> j becomes jpred -> ac.root -> j
            NB: likely should be followed by _recompute_root()

            first token of ac will be placed at position j in self


        :param ac:
        :param j:

        :return:
        """

        if j not in self._index_vec:
            raise MissingTokenInACandidate(
                f"token {j} not in ACandidate {self.itokens}"
            )

        jindex = self._index_vec.index(j)

        # update _index_vec
        self._index_vec = (
            self._index_vec[:jindex] + ac.itokens + self._index_vec[jindex:]
        )

        # update _tokens
        for t in ac.tokens:
            self._tokens[t.i] = t

        # update upward edges (NB: should be 1-step iteration)
        for pred in self.token(j).predecessors:
            self.token(pred).successors |= {ac.root.i}
            ac.root.predecessors |= {pred}

        ac.root.successors |= {j}
        self.token(j).predecessors |= {ac.root.i}
        # ac._recompute_root()

    def extend_with_candidate(
        self,
        ac: Candidate,
        j: int,
        token_index=False,
        succ=None,
        pred=None,
    ):
        """
            extend self with ac candidate
            NB: likely should be followed by _recompute_root()

            first token of ac will be placed at position j in self

            if succ is given ac becomes a successor of j from self
            if pred is given ac becomes a predecessor of j from self

        :param ac:
        :param j:
        :param token_index: interpret j as token id if true, rather than a position in self._index_set
        :param succ: index from self, ac becomes successor of succ
        :param pred: index from self, ac becomes predecessor of pred
                    (currently only becoming a pred of root of self makes sense)

        :return:
        """
        if succ and pred:
            raise ValueError(" both succ and pred were provided")

        # attach ac either as a successor of succ or pred
        if pred:
            if pred not in self.itokens:
                raise ValueError("pred not in self.itokens")
            if self.token(pred).predecessors:
                raise ValueError(
                    "Candidate token can have only one predecessor"
                )
        if succ and succ not in self.itokens:
            raise ValueError("succ not in self.itokens")

        if token_index:
            if j in self._index_vec:
                j = self._index_vec.index(j)
            else:
                raise MissingTokenInACandidate(
                    f"token {j} not in ACandidate {self.itokens}"
                )

        self._index_vec = (
            self._index_vec[:j] + ac.itokens + self._index_vec[j:]
        )
        for t in ac.tokens:
            self._tokens[t.i] = t

        if succ is not None:
            self.token(succ).successors |= {ac.root.i}
            ac.root.predecessors = {succ}

        # in that case ac replaces at the root (the only possible case)
        if pred is not None:
            self.token(pred).predecessors |= {ac.root.i}
            ac.root.successors = {pred}

    def replace_token_with_acandidate(self, i: int, ac: Candidate):
        """
        replace is a combination of remove and insert
        :param i:
        :param ac:
        :return:
        """

        ac = deepcopy(ac)
        if self.token(i).dep_ == "poss":
            of_token = Token(
                **{
                    "i": max(self.itokens + ac.itokens) + n_extra_token,
                    "lower": "of",
                    "text": "of",
                    "lemma": "of",
                    "dep_": "prep",
                    "tag_": "IN",
                }
            )
            nc = Candidate().from_tokens([of_token])
            ac = deepcopy(ac)
            ac.insert_before(nc, j=ac.root.i)
            ac._recompute_root()
        self.insert_before(ac, j=i)

        self.remove(i)
        self.clean_dangling_edges().sort_index_tree()

    def remove(self, i: int):
        # edges
        for pred in self.token(i).predecessors:
            self.token(pred).successors |= self.token(i).successors

        for succ in self.token(i).successors:
            self.token(succ).predecessors |= self.token(i).predecessors

        for pred in self.token(i).predecessors:
            self.token(pred).successors -= {i}

        for succ in self.token(i).successors:
            self.token(succ).predecessors -= {i}

        del self._tokens[i]

        j = self._index_vec.index(i)
        self._index_vec = self._index_vec[:j] + self._index_vec[j + 1 :]

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
    def normalize(self):
        return (
            self.drop_cc()
            .drop_punct()
            .drop_articles()
            .clean_dangling_edges()
            .sort_index_tree()
        )


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
