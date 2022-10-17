from __future__ import annotations

import dataclasses
from enum import Enum
from hashlib import blake2b

import numpy as np
from dataclass_wizard import JSONWizard
from spacy.tokens import Token

from lm_service.onto import Candidate, MuIndex
from lm_service.piles import ExtCandidateList


class EntityCandidateAlignmentError(Exception):
    pass


class EntityLinker(str, Enum):
    BERN_V2 = "BERN_V2"
    SPACY_NAIVE_WIKI = "SPACY_NAIVE_WIKI"
    SPACY_BASIC = "SPACY_BASIC"
    LOCAL_NON_EL = "LOCAL_NON_EL"

    def __repr__(self):
        return self.name


ent_db_type_local_gg = "ent_db_type_local_gg"
blake2b_digest_size = 12


@dataclasses.dataclass(repr=False)
class Entity(JSONWizard):
    """
    represents a token in dep tree
    """

    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    linker_type: EntityLinker
    ent_db_type: str
    id: str
    hash: str = ""
    ent_type: str | None = None
    original_form: str | None = None
    description: str | None = None

    def __post_init__(self):
        self.hash = f"{self.linker_type}/{self.ent_db_type}/{self.id}"


def interval_inclusion_metric(x, y):
    xa, xb = x
    ya, yb = y
    int_a = max([xa, ya])
    int_b = min([xb, yb])
    int_size = max([0, int_b - int_a])
    return (xb - xa) / int_size if int_size > 0 else 0


def normalize_bern_entity(item) -> tuple[Entity | None, tuple | None]:
    if len(item["id"]) > 0:
        item_spec = item["id"][0].split(":")
        db_type, item_id = item_spec
        return Entity(
            linker_type=EntityLinker.BERN_V2,
            ent_type=item["obj"],
            ent_db_type=db_type,
            id=item_id,
        ), (item["span"]["begin"], item["span"]["end"])
    else:
        return None, None


def normalize_naive_wiki_entity_linker(
    item,
) -> tuple[Entity | None, tuple | None]:
    """
    spacy-entity-linker package

    :param item:
    :return:
    """
    if item.get_url():
        span = item.get_span()
        item_id = item.get_url().split("/")[-1]

        try:
            ee = next(iter(item.get_categories()))
            ent_type = ee.get_url().split("/")[-1]
        except:
            ent_type = None
        return Entity(
            linker_type=EntityLinker.SPACY_NAIVE_WIKI,
            ent_db_type="wikidata",
            ent_type=ent_type,
            id=item_id,
            description=item.get_description(),
            original_form=item.get_original_alias(),
        ), (span.start_char, span.end_char)

    else:
        return None, None


def phrase_to_spacy_basic_entities(phrase=None, rdoc=None, nlp=None):
    if rdoc is None:
        rdoc = nlp(phrase)
    ents0 = (item for item in rdoc if item.ent_type != 0)
    ents_split = []
    ent_item = []
    for t in ents0:
        # 2 means ent_iob_ == "B", beginning of entity
        if t.ent_iob == 3:
            ents_split += [ent_item]
            ent_item = [t]
        else:
            ent_item += [t]
    if ent_item:
        ents_split += [ent_item]
    ents_split = [x for x in ents_split if x]

    return ents_split


def normalize_spacy_basic(
    item: list[Token],
) -> tuple[Entity | None, tuple | None]:
    """

    :param item:
    :return:
    """
    if item:
        span = item[0].idx, item[-1].idx + len(item[-1].text)
        original_form = " ".join([x.text for x in item]).lower()
        e_id = blake2b(
            original_form.encode("utf-8"), digest_size=blake2b_digest_size
        ).hexdigest()
        ent_type = str(item[0].ent_type)

        return (
            Entity(
                linker_type=EntityLinker.SPACY_BASIC,
                ent_db_type="basic",
                ent_type=ent_type,
                original_form=original_form,
                id=e_id,
            ),
            span,
        )
    else:
        return None, None


def link_unlinked_entities(
    map_eindex_entity: dict[str, Entity],
    map_c2e: list[tuple[MuIndex, str]],
    map_muindex_candidate: dict[MuIndex, Candidate],
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param map_eindex_entity:
    :param map_muindex_candidate:
    :param map_c2e:

    :return:
        i_e -> e ; i_e -> i_mu
    """

    # create entities for unlinked candidates (for some candidates entities were not found)

    mentions_not_in_entities = set(map_muindex_candidate) - set(
        c for c, e in map_c2e
    )

    for i_mu in mentions_not_in_entities:
        c = map_muindex_candidate[i_mu]
        original_form = " ".join(c.project_to_text()).lower()
        e_id = blake2b(
            original_form.encode("utf-8"), digest_size=blake2b_digest_size
        ).hexdigest()

        new_entity = Entity(
            linker_type=EntityLinker.LOCAL_NON_EL,
            ent_db_type=ent_db_type_local_gg,
            id=e_id,
            original_form=original_form,
        )
        map_c2e += [(i_mu, new_entity.hash)]
        map_eindex_entity[new_entity.hash] = new_entity

    return map_eindex_entity, map_c2e


def link_candidate_entity(ec_spans: dict, ecl: ExtCandidateList, ix_phrases):
    """
        NB: in future (iphrase, i_ent): ent
            will also be used for linking used
    :param ec_spans: (iphrase, i_ent): (span_beg, span_end)
    :param ecl:
    :param ix_phrases: phrases which to link
    :return:
    """

    # pick phrase indices
    i_e = list(ec_spans.keys())
    # c_index : (iphrase, sindex, cand_subindex, token_index) : [spans]

    map_c2e = []
    ecl.set_filter(lambda x: x[0] in ix_phrases)

    for k, cand_list in ecl:
        for n, candidate in enumerate(cand_list):
            dist = np.array(
                [
                    [
                        interval_inclusion_metric((t.idx, t.idx_eot), int_ec)
                        for t in candidate.tokens
                    ]
                    for k, int_ec in ec_spans.items()
                ]
            )
            if np.sum((dist > 0) & (dist < 1)) > 0:
                raise EntityCandidateAlignmentError(
                    "Entity indices and candidate indices are not aligned."
                )
            (ec_ixs,) = np.where((dist > 0.8).any(axis=1))
            # map current candidate to entity index
            map_c2e += [(MuIndex(False, *k, n), i_e[i]) for i in ec_ixs]

    # e_index : (str)
    # c_index : (iphrase, sindex, cand_subindex)
    # 1 -> n : cand -> entity (could be easily generalizable)
    return map_c2e


entity_normalized_foo_map = {
    EntityLinker.BERN_V2: normalize_bern_entity,
    EntityLinker.SPACY_NAIVE_WIKI: normalize_naive_wiki_entity_linker,
    EntityLinker.SPACY_BASIC: normalize_spacy_basic,
}


def iterate_linking_over_phrases(
    phrases,
    ecl,
    link_foo,
    link_foo_kwargs=None,
    map_eindex_entity=None,
    map_c2e=None,
    etype=EntityLinker.BERN_V2,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param phrases:
    :param ecl:
    :param link_foo: function, maps a phrase to a list of entities
    :param link_foo_kwargs:
    :param map_eindex_entity:
    :param map_c2e:
    :param etype:
    :return:
        s_e -> e ; i_c -> i_se
    """
    if link_foo_kwargs is None:
        link_foo_kwargs = dict()
    if map_eindex_entity is None:
        map_eindex_entity = dict()
    if map_c2e is None:
        map_c2e = dict()

    foo_normalize = entity_normalized_foo_map[etype]

    for ix_current_phrase, phrase in enumerate(phrases):
        response = link_foo(phrase, **link_foo_kwargs)

        # entities + spans
        entity_pack_current = [foo_normalize(item) for item in response]
        spans, ents = [], []
        for e, span in entity_pack_current:
            if e is not None:
                spans += [span]
                ents += [e]

        entities_index_e_map_current = {e.hash: e for e in ents}

        ei_span_map = {e.hash: span for e, span in zip(ents, spans)}

        map_c2e_current = link_candidate_entity(
            ei_span_map, ecl, (ix_current_phrase,)
        )

        map_c2e.extend(map_c2e_current)
        map_eindex_entity.update(entities_index_e_map_current)

    return map_eindex_entity, map_c2e


def iterate_over_linkers(
    phrases: list[str],
    ecl: ExtCandidateList,
    map_muindex_candidate: dict[MuIndex, Candidate],
    phrase_entities_foos: dict,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:

    map_eindex_entity: dict[str, Entity] = {}
    map_c2e: list[tuple[MuIndex, str]] = []

    for link_mode, link_foo in phrase_entities_foos.items():
        map_eindex_entity, map_c2e = iterate_linking_over_phrases(
            phrases=phrases,
            ecl=ecl,
            map_eindex_entity=map_eindex_entity,
            map_c2e=map_c2e,
            link_foo=link_foo,
            etype=link_mode,
        )

    map_eindex_entity, map_c2e = link_unlinked_entities(
        map_eindex_entity, map_c2e, map_muindex_candidate
    )
    return map_eindex_entity, map_c2e
