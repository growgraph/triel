from __future__ import annotations

import dataclasses

from lm_service.linking.onto import Entity, EntityHash
from lm_service.onto import BaseDataclass, MuIndex, SimplifiedCandidate


class UnknownCastTripleVersion(Exception):
    pass


@dataclasses.dataclass(frozen=True, eq=True)
class TripleExplicit(BaseDataclass):
    mu: SimplifiedCandidate
    source: SimplifiedCandidate
    relation: SimplifiedCandidate
    target: SimplifiedCandidate


@dataclasses.dataclass(frozen=True, eq=True)
class TripleFormal(BaseDataclass):
    object: EntityHash
    subject: EntityHash
    predicate: EntityHash


@dataclasses.dataclass(frozen=True, eq=True)
class Triple(BaseDataclass):
    """
    represents a token in dep tree
    """

    triple_index: SimplifiedCandidate
    triple: tuple[SimplifiedCandidate, SimplifiedCandidate, SimplifiedCandidate]


@dataclasses.dataclass(frozen=True, eq=True)
class REELResponse(BaseDataclass):
    """
    relation extraction / entity linking response
    """

    triples: list[tuple[MuIndex, tuple[MuIndex, MuIndex, MuIndex]]]
    eindex_entity: dict[str, Entity]
    muindex_eindex: list[tuple[MuIndex, str]]
    _muindex_candidate: list[tuple[MuIndex, SimplifiedCandidate]]

    # @property
    # def triples(self):
    #     return {k: v for k, v in self._triples}

    @property
    def muindex_candidate(self):
        return {k: v for k, v in self._muindex_candidate}


@dataclasses.dataclass(frozen=True, eq=True)
class REELResponseRedux(BaseDataclass):
    """
    represents a token in dep tree
    """

    triples: list[Triple]
    map_mention_entity: list[dict]
    top_level_mention: list[dict]


@dataclasses.dataclass(frozen=True, eq=True)
class REELResponseEntity(BaseDataclass):
    """
    represents a token in dep tree
    """

    triples: list[TripleFormal]
    entities: list[Entity]
