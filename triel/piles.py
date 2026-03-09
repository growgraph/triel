from __future__ import annotations

import dataclasses
from collections import defaultdict
from copy import deepcopy
from typing import Callable

import pandas as pd

from triel.onto import (
    Candidate,
    Relation,
    SourceOrTarget,
    Token,
    TokenIndexT,
)


@dataclasses.dataclass(repr=False)
class CandidatePile:
    """
    pile of candidates of one type
    TokenIndexT -> Candidate (SourceOrTarget | Relation)
    """

    _root_to_candidate: dict[TokenIndexT, SourceOrTarget | Relation] = (
        dataclasses.field(default_factory=dict)
    )

    def __len__(self) -> int:
        return len(self._root_to_candidate)

    def __getitem__(self, key) -> Candidate:
        """

        :return: relation index in pile : relation tokens
        """

        return self._root_to_candidate[key]

    def __repr__(self):
        s = ""
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

    def append(self, r: Candidate):
        self._root_to_candidate[r.root.s] = r

    def project_to_text(self):
        return [c.project_to_text_str() for c in self._root_to_candidate.values()]

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
        r2c = {k: c.normalize() for k, c in new._root_to_candidate.items()}
        r2c = {k: v for k, v in r2c.items() if v._root is not None}
        new._root_to_candidate = r2c
        return new

    def clean_dangling_edges(self):
        new = deepcopy(self)
        new._root_to_candidate = {
            k: c.clean_dangling_edges() for k, c in new._root_to_candidate.items()
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

    def unfold_conjunction(self) -> ExtCandidateList:
        dd_pile: ExtCandidateList = ExtCandidateList()
        for sroot, c in self._root_to_candidate.items():
            for c_unfolded in c.unfold_conjuction():
                dd_pile[sroot] += [c_unfolded]
        return dd_pile

    def dump_to_table(self):
        arr = [
            (
                *k,
                min([vv.idx for vv in v.tokens]),
                max([vv.idx_eot for vv in v.tokens]),
                " ".join(v.project_to_text()),
            )
            for k, v in self._root_to_candidate.items()
        ]
        return pd.DataFrame(
            arr,
            columns=pd.Index(["iphrase", "itoken", "a", "b", "text"]),
        )


class ExtCandidateList:
    """
    ext list of candidates

    necessary because under the same root there can be several candidate (e.g. conjunction)
    TokenIndexT -> [Candidate] (SourceOrTarget | Relation)
    """

    def __init__(self):
        self._filter: None | Callable = None
        self._root_to_lists: defaultdict[TokenIndexT, list[Candidate]] = defaultdict(
            list
        )

    def __len__(self) -> int:
        return sum(len(x) for x in self._root_to_lists.values())

    def __contains__(self, item):
        return item in self._root_to_lists

    def set_filter(self, foo: Callable):
        # eg consider only candidates from a phrase range
        # self.set_filter(lambda x: i <= x[0] < i + window_size)
        self._filter = foo

    def __setitem__(self, key, value):
        self._root_to_lists[key] = value

    def __getitem__(self, item):
        return self._root_to_lists[item]

    def select(self, ip):
        return (
            self._root_to_lists[k] for k in self._root_to_lists.keys() if k[0] == ip
        )

    def append(self, key: TokenIndexT, value: Candidate):
        if tuple(value.stokens) not in {
            tuple(x.stokens) for x in self._root_to_lists[key]
        }:
            self._root_to_lists[key] += [value]

    def __iter__(self):
        if self._filter is None:
            return ((k, v) for k, v in self._root_to_lists.items())
        else:
            return ((k, v) for k, v in self._root_to_lists.items() if self._filter(k))

    def filter_out_pronouns(self):
        for k, vlist in self._root_to_lists.items():
            self._root_to_lists[k] = [item for item in vlist if not item.has_pronoun()]

    def get_phrase(self, iphrase):
        return sorted(
            [(k, v) for k, v in self._root_to_lists.items() if k[0] == iphrase],
            key=lambda x: x[0][1],
        )


@dataclasses.dataclass
class SRTPile:
    sources: CandidatePile
    targets: CandidatePile
    relations: CandidatePile
