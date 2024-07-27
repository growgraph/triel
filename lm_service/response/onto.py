from __future__ import annotations

import dataclasses

from lm_service.linking.onto import Entity
from lm_service.onto import BaseDataclass, MuIndex, SimplifiedCandidate


class UnknownCastTripleVersion(Exception):
    pass


@dataclasses.dataclass(repr=False, frozen=True, eq=True)
class RELResponse(BaseDataclass):
    """
    represents a token in dep tree
    """

    triples: dict[MuIndex, tuple[MuIndex, MuIndex, MuIndex]]
    eindex_entity: dict[str, Entity]
    muindex_eindex: list[tuple[MuIndex, str]]
    muindex_candidate: dict[MuIndex, SimplifiedCandidate]


@dataclasses.dataclass(repr=False, frozen=True, eq=True)
class TripleExplicit(BaseDataclass):
    """
    represents a token in dep tree
    """

    mu: SimplifiedCandidate
    source: SimplifiedCandidate
    relation: SimplifiedCandidate
    target: SimplifiedCandidate


@dataclasses.dataclass(repr=False, frozen=True, eq=True)
class Triple(BaseDataclass):
    """
    represents a token in dep tree
    """

    triple_index: SimplifiedCandidate
    triple: tuple[SimplifiedCandidate, SimplifiedCandidate, SimplifiedCandidate]


@dataclasses.dataclass(repr=False, frozen=True, eq=True)
class RELResponseSimplified(BaseDataclass):
    """
    represents a token in dep tree
    """

    triples: list[Triple]
    map_mention_entity: list[dict]
    top_level_mention: list[dict]
