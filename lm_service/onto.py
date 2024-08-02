from __future__ import annotations

import dataclasses
import logging
import string
from abc import ABC
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import TypeVar

import networkx as nx
from dataclass_wizard import JSONWizard
from dataclass_wizard.enums import DateTimeTo
from lemminflect import getInflection, getLemma

from lm_service.hash import hashme


class BaseDataclass(JSONWizard, JSONWizard.Meta):
    marshal_date_time_as = DateTimeTo.ISO_FORMAT
    key_transform_with_dump = "SNAKE"
    skip_defaults = True


class MissingTokenInACandidate(Exception):
    pass


class RelationHasNoTargetCandidatesError(Exception):
    pass


class InsertingExistingTokens(Exception):
    pass


class RequestedIndexDoesNotExist(Exception):
    pass


TokenIndexT = tuple[int, str]
ChainIndex = tuple[int, str]


logger = logging.getLogger(__name__)


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


CandidateType = TypeVar("CandidateType", bound="Candidate")
TokenType = TypeVar("TokenType", bound="AbsToken")


class ConfigToken:
    representation_leading_zero = 3


@dataclasses.dataclass(repr=False)
class AbsToken(JSONWizard, ABC):
    """
    represents a token in dep tree
    """

    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    text: str = ""
    dep_: str = ""
    tag_: str = ""
    lower: str = ""
    lemma: str = ""
    ent_iob: int = 0
    _level: int = 0
    label: str = ""
    idx: int = 0
    idx_eot: int = 0
    ent_id: int = 0
    ent_kb_id: int = 0
    ent_type: int = 0
    ent_id_: str = ""
    ent_kb_id_: str = ""
    ent_type_: str = ""

    def __post_init__(self):
        if not self.lemma and self.text:
            self.lemma = self.text
        if not self.lower and self.text:
            self.lower = self.text.lower()

    def __repr__(self):
        content = [f" {k} : {v}" for k, v in self.__dict__.items()]
        return "Token fields:" + " |".join(content)

    @classmethod
    def i2s(cls, i) -> str:
        """
        cast integer index `i` to string index s used in Token
        :param i:
        :return:
        """

        if isinstance(i, int):
            return f"{i:0{ConfigToken.representation_leading_zero}}"
        elif isinstance(i, str):
            return i
        raise TypeError(f" Token.i2s received i={i} of type {type(i)}, int expected.")

    @classmethod
    def ituple2stuple(cls, i) -> tuple[int, str]:
        """
        cast integer index `i` to string index s used in Token
        :param i:
        :return:
        """

        if isinstance(i, tuple):
            if len(i) == 2:
                return i[0], cls.i2s(i[1])
            raise ValueError(f"expected tuple of len 2, {i} received")
        raise TypeError(
            f" Token.ituple2stuple received i={i} of type {type(i)}, tuple" " expected."
        )


@dataclasses.dataclass(repr=False)
class Token(AbsToken):
    """
    represents a token in dep tree
    """

    s: TokenIndexT = (0, "")
    predecessors: set[TokenIndexT] = dataclasses.field(default_factory=set)
    successors: set[TokenIndexT] = dataclasses.field(default_factory=set)

    def __post_init__(self):
        if isinstance(self.s, int):
            self.s = self.i2s(self.s)
            self.predecessors = (
                set(self.i2s(i) for i in self.predecessors)
                if self.predecessors
                else set()
            )
            self.successors = (
                set(self.i2s(i) for i in self.successors) if self.successors else set()
            )
        elif isinstance(self.s, tuple):
            if not self.s[1]:
                self.s = self.ituple2stuple(self.s)
            self.predecessors = (
                set(self.ituple2stuple(i) for i in self.predecessors)
                if self.predecessors
                else set()
            )
            self.successors = (
                set(self.ituple2stuple(i) for i in self.successors)
                if self.successors
                else set()
            )

    def __repr__(self):
        content = [f" {k} : {v}" for k, v in self.__dict__.items()]
        return "Token fields:" + " |".join(content)

    def prior_s(self):
        if isinstance(self.s, tuple):
            return self.s[0], "/"
        elif isinstance(self.s, str):
            return "/"
        else:
            raise TypeError(f"Unexpected TokenIndexT subtype: {type(self.s)}")


@dataclasses.dataclass(repr=False)
class AbsCandidate(BaseDataclass, ABC):
    def project_to_text_str(self):
        pass

    def drop_amod_vbn(self):
        return self

    def drop_cc(self):
        return self

    def drop_punct(self):
        return self

    def drop_articles(self):
        return self

    def normalize(self):
        return self

    def has_pronoun(self):
        return self


@dataclasses.dataclass(repr=False)
class CandidateReference(AbsCandidate, JSONWizard):
    sroot: tuple[int, int]

    def project_to_text_str(self):
        return f"{self.sroot}"


@dataclasses.dataclass(eq=True, order=True)
class SimplifiedCandidate(BaseDataclass):
    hash: str
    text: str | None = None
    role: str | None = None

    def get_copy_with_role(self, role: str):
        c = dataclasses.replace(self)
        c.role = role
        return c


@dataclasses.dataclass(repr=False)
class Candidate(AbsCandidate):
    _tokens: dict[TokenIndexT, Token] = dataclasses.field(default_factory=dict)
    _index_vec: list[TokenIndexT] = dataclasses.field(default_factory=list)
    _root: TokenIndexT | None = None

    def __add__(self, other: Candidate):
        new = deepcopy(self)
        for c in other.tokens:
            new.append(c)
        return new

    def __len__(self) -> int:
        return len(self._tokens)

    def hashme(self) -> str:
        original_form = " ".join(self.project_to_text()).lower()
        return hashme(original_form)

    def to_simplified(self):
        return SimplifiedCandidate(
            hash=self.hashme(), text=" ".join(self.project_to_text()).lower()
        )

    def __repr__(self):
        content = []
        for s in self.stokens:
            t = self._tokens[s]
            str_succ = ", ".join([f"{s}" for s in t.successors])
            str_pred = ", ".join([f"{s}" for s in t.predecessors])
            content += [
                f" \t {t.s} : {t.lower} : {t.tag_} : {t.dep_} : pred <"
                f" {str_pred} : succ > {str_succ}"
            ]

        return f"{self.__class__.__name__} tokens : (\n\t" + "|\n\t".join(content) + ")"

    def from_tokens(self, tokens: list[Token]):
        for t in tokens:
            self.append(t)
        self.clean_dangling_edges()
        return self

    def clean_dangling_edges(self, robust_mode=False):
        present = set(self.stokens)
        for k, t in self._tokens.items():
            t.successors &= present
            t.predecessors &= present
        self._recompute_root(robust_mode)
        return self

    def max_level(self) -> int:
        return 0 if self.empty else max(t._level for t in self._tokens.values())

    def _recompute_root(self, robust_mode=True):
        roots = [k for k, v in self._tokens.items() if len(v.predecessors) == 0]
        if len(roots) > 1:
            logger.error(f" {len(roots)} roots: dumping self")
            logger.error(
                f" {sorted(self.stokens)} :"
                f" {' '.join([self.token(x).text for x in sorted(self.stokens)])}"
            )
            if robust_mode:
                logger.error(" robust_mode picking a root with a smaller index")
                root = sorted(roots)
                acc = []
                self._pick_successors(root[0], acc)
                to_drop = set(self._tokens) - set(acc)
                self.drop_tokens(to_drop)
                self._recompute_root(robust_mode)
            else:
                raise ValueError(
                    f" candidate has {len(roots)} roots {roots} | candidate"
                    f" <{' '.join([self.token(x).text for x in sorted(self.stokens)])}>,"
                    " should be one"
                )
        elif len(roots) == 1:
            self._root = next(iter(roots))
        else:
            self._root = None

    @property
    def sroot(self):
        return self._root

    @property
    def root(self):
        return self._tokens[self._root]

    @property
    def stokens(self):
        return self._index_vec

    def token(self, i, index=False):
        if index:
            if i < len(self):
                return self._tokens[self._index_vec[i]]
            else:
                raise RequestedIndexDoesNotExist(
                    f" size of {self.__class__.__name__} obj {len(self)},"
                    f" requesting index {i}"
                )
        else:
            if i in self._index_vec:
                return self._tokens[i]
            else:
                raise MissingTokenInACandidate(
                    f"token {i} not present in"
                    f" {self.__class__.__name__} containing {self.stokens}"
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
        return [self.token(s) for s in self.stokens[iifrom:iito]]

    def from_subtree(self, i: TokenIndexT):
        acc: list[TokenIndexT] = []
        self._pick_successors(i, acc)
        return (
            Candidate()
            .from_tokens([self.token(s) for s in self._index_vec if s in acc])
            .clean_dangling_edges()
        )

    def _pick_successors(self, i, acc):
        acc += [i]
        for j in self.token(i).successors:
            self._pick_successors(j, acc)

    @property
    def tokens(self):
        return (self.token(s) for s in self.stokens)

    @property
    def empty(self):
        return len(self._tokens) == 0

    def project_to_text(self):
        """
            see https://spacy.io/api/token#attributes
            if entity - return text, otherwise return lemma
        :return:
        """
        pp = []
        for s in self.stokens:
            token = self.token(s)
            if token.ent_iob in (0, 2):
                pp += [token.text]
            else:
                pp += [token.lemma]
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
            self._root = token.s
        self._tokens[token.s] = token
        self._index_vec.append(token.s)

    def drop_tokens(self, drop_indices):
        """
        NB: consistent wrt to graph operations
        """
        for i in drop_indices:
            self.remove(i)

    def _sort_wrt_tree(
        self,
        j: TokenIndexT,
        sorter: dict[
            TokenIndexT | None,
            tuple[float, float],
        ],
    ):
        """
        sort tokens respecting the tree order: within each node sort succs wrt to token order
        important to correctly account for sorting order when substitution (due to coref) occur
        :param j:
        :param sorter:
        :return:
        """
        # TODO write a test
        #
        successors = self.token(j).successors

        # "of" exception => if j is "of" then force order
        if self.token(j).lower == "of" and self.token(j).tag_ == "IN":
            sorted_succs = [j] + sorted(successors)
        else:
            sorted_succs = sorted(list(successors) + [j])

        bnd_a, bnd_b = sorter[j]

        interval = (bnd_b - bnd_a) / len(sorted_succs)

        values = [bnd_a + k * interval for k in range(len(sorted_succs) + 1)]

        for b, ba, bb in zip(
            sorted_succs,
            values,
            values[1:],
        ):
            sorter[b] = ba, bb
        for s in successors:
            self._sort_wrt_tree(s, sorter)

    def sort_index(self):
        if self._root is not None:
            proposed_sorter = {self.sroot: (0, len(self))}
            self._sort_wrt_tree(self.sroot, sorter=proposed_sorter)
            self._index_vec = [
                x
                for x, _ in sorted(proposed_sorter.items(), key=lambda item: item[1][0])
            ]
        return self

    def insert_before(self, ac: Candidate, s: TokenIndexT):
        """
            extend self with ac candidate
            such that jpred -> j becomes jpred -> ac.root -> j

            first token of ac will be placed at position j in self


        :param ac:
        :param s:

        :return:
        """
        ac = deepcopy(ac)
        if s not in self._index_vec:
            raise MissingTokenInACandidate(
                f"token {s} not in ACandidate {self.stokens}"
            )

        jindex = self._index_vec.index(s)

        ac_root = ac.sroot

        # update _index_vec
        self._index_vec = (
            self._index_vec[:jindex] + ac.stokens + self._index_vec[jindex:]
        )

        # update _tokens
        for t in ac.tokens:
            self._tokens[t.s] = t

        # update upward edges (NB: should be only one predecessor) : maybe check for that?
        for pred in self.token(s).predecessors:
            self.token(pred).successors |= {ac_root}
            self.token(pred).successors -= {s}
            self.token(ac_root).predecessors |= {pred}

        self.token(ac_root).successors |= {s}
        self.token(s).predecessors = {ac.sroot}
        self._recompute_root()

    def replace_token_with_acandidate(self, i: TokenIndexT, ac: Candidate):
        """
        replace is a combination of remove and insert
        :param i:
        :param ac:
        :return:
        """

        ac = deepcopy(ac)
        if self.token(i).dep_ == "poss":
            try:
                of_index = next(iter(self.token(i).predecessors))
            except StopIteration:
                of_index = i

            # compute suffix
            derived_indices = [
                y for _, y in self.token(of_index).successors if not is_int(y)
            ]
            if derived_indices:
                max_index = max(x[-1] for x in derived_indices)
                current_index = chr(ord(max_index) + 1)
            else:
                current_index = "a"

            if isinstance(i, tuple):
                s0 = (of_index[0], of_index[1][:] + current_index)
            else:
                s0 = of_index[:] + current_index

            of_token = Token(
                s=s0,
                lower="of",
                text="of",
                lemma="of",
                dep_="prep",
                tag_="IN",
            )
            nc = Candidate().from_tokens([of_token])
            ac = deepcopy(ac)
            ac.insert_before(nc, s=ac.root.s)
        self.insert_before(ac, s=i)

        self.remove(i)
        self.clean_dangling_edges().sort_index()

    def remove_subtree_keeping_root(self, i: TokenIndexT):
        subtree_ix: list[TokenIndexT] = []
        self._pick_successors(i, subtree_ix)

        # exclude root
        subtree_ix = subtree_ix[1:]

        for ix in subtree_ix:
            del self._tokens[ix]
        self._index_vec = list(self._tokens)

        set_subtree_ix = set(subtree_ix)
        for t in self.tokens:
            t.successors -= set_subtree_ix
            t.predecessors -= set_subtree_ix

    def replace_subtree_with_acandidate(self, i: TokenIndexT, ac: Candidate):
        """
        replace is a combination of remove and insert
        :param i:
        :param ac:
        :return:
        """

        # TODO check for non overlapping TokenIndexT
        self.remove_subtree_keeping_root(i)
        self.replace_token_with_acandidate(i, ac)

    def remove(self, i: TokenIndexT):
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

    @property
    def lemmas(self):
        return [self._tokens[k].lemma for k in self.stokens]

    def drop_articles(self):
        # t.dep_ == "det" or t.tag_ != "DT"
        drop_aux_indices = [j for j, t in self._tokens.items() if t.dep_ == "det"]
        self.drop_tokens(drop_aux_indices)
        return self

    def drop_amod_vbn(self):
        drop_aux_indices = [
            j for j, t in self._tokens.items() if (t.dep_ == "amod" and t.tag_ == "VBN")
        ]
        self.drop_tokens(drop_aux_indices)
        return self

    def drop_cc(self):
        drop_aux_indices = [
            j for j, t in self._tokens.items() if (t.dep_ == "cc" and t.tag_ == "CC")
        ]
        self.drop_tokens(drop_aux_indices)
        return self

    def drop_pronouns(self):
        drop_aux_indices = [j for j, t in self._tokens.items() if t.tag_[:3] == "PRP"]
        self.drop_tokens(drop_aux_indices)
        return self

    def drop_punct(self):
        drop_aux_indices = [
            j
            for j, t in self._tokens.items()
            if t.dep_ == "punct"
            and (t.tag_ in string.punctuation or t.text in string.punctuation)
        ]
        self.drop_tokens(drop_aux_indices)
        return self

    def normalize(self):
        pass

    def to_nx_graph(self, offset=0, use_successors=True):
        # i -> s, inverse to _index_set

        s2index = {s: j + offset for j, s in enumerate(self._index_vec)}

        g = nx.DiGraph()

        vertex_desc = [
            (
                s2index[s],
                self.token(s).__dict__,
                [
                    s2index[ss]
                    for ss in (
                        self.token(s).successors
                        if use_successors
                        else self.token(s).predecessors
                    )
                ],
            )
            for j, s in enumerate(self._index_vec)
        ]
        g.add_nodes_from((i, props) for i, props, _ in vertex_desc)
        g.add_edges_from(
            (i, j) if use_successors else (j, i) for i, _, es in vertex_desc for j in es
        )
        return g

    def unfold_conjuction(self) -> list[Candidate]:
        return partition_conjunctive_wrapper(self)


class ACandidateKind(Enum):
    RELATION = 1
    SOURCE_TARGET = 2
    SOURCE = 3
    TARGET = 4


@dataclasses.dataclass(repr=False)
class Relation(Candidate):
    @property
    def passive(self):
        flags = [t.dep_ in ("auxpass", "ccomp", "acl") for t in self._tokens.values()]
        return any(flags)

    def has_prepositions(self):
        return any(t.dep_ == "prep" and t.tag_ == "IN" for t in self.tokens)

    def approximate_hash_int(self):
        return sum(int(x) for _, x in self.stokens)

    def normalize(self):
        """
        TODO extend to perfect tense
        :return:
        """

        # drop all aux
        drop_aux_indices = [j for j, t in self._tokens.items() if t.dep_ == "aux"]
        self.drop_tokens(drop_aux_indices)

        if self.passive:
            # find auxpass, inflect it to was
            for s in self.stokens:
                token = self.token(s)
                if token.dep_ in ("auxpass", "ccomp", "acl", "relcl"):
                    lemmas = getLemma(token.text, upos="VERB")
                    if lemmas:
                        # -> "was"
                        # inflected = getInflection(lemmas[0], tag="VBD")
                        # -> "is"
                        if token.dep_ in ("auxpass", "relcl"):
                            inflected = getInflection(lemmas[0], tag="VBZ")
                        elif token.dep_ == "acl":
                            inflected = getInflection(lemmas[0], tag="VBD")
                        else:
                            inflected = []
                        if inflected:
                            token.text = inflected[0]
        else:
            # inflect remaining VBs
            for s, t in self._tokens.items():
                if t.tag_.startswith("VB"):
                    lemmas = getLemma(t.text, upos="VERB")
                    if lemmas:
                        inflected = getInflection(lemmas[0], tag="VBZ")
                        if inflected:
                            self._tokens[s].text = inflected[0]
        return self


@dataclasses.dataclass(repr=False)
class SourceOrTarget(Candidate):
    def normalize(self):
        return (
            self.drop_cc()
            .drop_punct()
            .drop_articles()
            .drop_pronouns()
            .clean_dangling_edges()
            .sort_index()
        )

    def has_pronoun(self):
        return any(t for t in self._tokens.values() if t.tag_ == "PRP")


@dataclasses.dataclass(repr=False)
class Source(SourceOrTarget):
    pass


@dataclasses.dataclass(repr=False)
class Target(SourceOrTarget):
    pass


@dataclasses.dataclass(repr=False)
class TripleCandidate(JSONWizard):
    source: Source | CandidateReference
    relation: Relation
    target: Target | CandidateReference

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

    def has_pronouns(self):
        return self.source.has_pronoun() or self.source.has_pronoun()

    def __repr__(self):
        s = f"\t{self.source.__repr__()}\n\t{self.relation.__repr__()}\n\t{self.target.__repr__()}\n"
        return s


def to_string(obj, func):
    if isinstance(obj, dict):
        return {func(k): to_string(item, func) for k, item in obj.items()}
    if isinstance(obj, list):
        return [to_string(item, func) for item in obj]
    else:
        return func(obj)


def to_string_keys(obj, func):
    if isinstance(obj, dict):
        return {to_string_keys(k, func): item for k, item in obj.items()}
    if isinstance(obj, (list, tuple)):
        return tuple(to_string_keys(item, func) for item in obj)
    else:
        return func(obj)


def apply_map(obj, mapper):
    if isinstance(obj, dict):
        return {
            apply_map(k, mapper): apply_map(item, mapper) for k, item in obj.items()
        }
    if isinstance(obj, list):
        return tuple(apply_map(item, mapper) for item in obj)
    else:
        return mapper[obj] if obj in mapper else obj


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
        # it is a choice to remove subtree rather than a token
        # c_prime.replace_token_with_acandidate(i=sparent, ac=candidate)
        c_prime.replace_subtree_with_acandidate(i=sparent, ac=candidate)
        acc.append(
            c_prime.drop_cc()
            .drop_punct()
            .drop_articles()
            .clean_dangling_edges()
            .sort_index()
        )
    return acc


@dataclasses.dataclass(eq=True, frozen=True, order=True)
class MuIndex(JSONWizard):
    """
    Candidate index in a collections of phrases.

    meta - flag, true if MuIndex points to a triple
    phrase - phrase number
    token - token index within phrase
    running - in case there are several `candidates` (conjunction),
            that exist under the same token root
    """

    meta: bool
    phrase: int
    token: str
    running: int

    def to_tuple(self):
        return self.meta, self.phrase, self.token, self.running

    def to_str(self):
        return "|".join(f"{int(x)}" for x in self.to_tuple())

    def __hash__(self):
        return hash((self.meta, self.phrase, self.token, self.running))
