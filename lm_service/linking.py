from __future__ import annotations

from enum import Enum

import numpy as np

from lm_service.onto import Candidate, MuIndex
from lm_service.piles import ExtCandidateList


class EntityCandidateAlignmentError(Exception):
    pass


class EntityLinker(str, Enum):
    BERN_V2 = "BERN_V2"
    SPACY_NAIVE_WIKI = "SPACY_NAIVE_EL"
    SPACY_BASIC = "SPACY_BASIC"
    LOCAL_NON_EL = "LOCAL_NON_EL"


ent_db_type_local_gg = "ent_db_type_local_gg"


def interval_inclusion_metric(x, y):
    xa, xb = x
    ya, yb = y
    int_a = max([xa, ya])
    int_b = min([xb, yb])
    int_size = max([0, int_b - int_a])
    return (xb - xa) / int_size if int_size > 0 else 0


def normalize_bern_entity(item) -> tuple[dict | None, tuple | None]:
    if len(item["id"]) > 0:
        item_spec = item["id"][0].split(":")
        db_type, item_id = item_spec
        return {
            "linker_type": EntityLinker.BERN_V2,
            "ent_type": item["obj"],
            "ent_db_type": db_type,
            "id": item_id,
            "confidence": item["prob"],
        }, (item["span"]["begin"], item["span"]["end"])
    else:
        return None, None


def normalize_naive_wiki_entity_linker(
    item,
) -> tuple[dict | None, tuple | None]:
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
        return {
            "linker_type": EntityLinker.SPACY_NAIVE_WIKI,
            "ent_db_type": "wikidata",
            "id": item_id,
            "original": item.get_original_alias(),
            "ent_type": ent_type,
            "desc": item.get_description(),
        }, (span.start_char, span.end_char)
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


def normalize_spacy_basic(item) -> tuple[dict | None, tuple | None]:
    """

    :param item:
    :return:
    """
    if item:
        span = item[0].idx, item[-1].idx + len(item[-1].text)
        ent_type = item[0].ent_type

        return {
            "linker_type": EntityLinker.SPACY_BASIC,
            "ent_db_type": "basic",
            "ent_type": ent_type,
        }, span
    else:
        return None, None


def link_unlinked_entities(
    map_eindex_entity: dict[tuple[int, int], dict],
    map_c2e: list[tuple[MuIndex, tuple[int, int]]],
    map_muindex_candidate: dict[MuIndex, Candidate],
) -> tuple[dict[tuple[int, int], dict], list[tuple[MuIndex, tuple[int, int]]]]:
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
        s = " ".join(c.project_to_text())
        new_entity = {
            "linker_type": EntityLinker.LOCAL_NON_EL,
            "ent_db_type": ent_db_type_local_gg,
            "id": s,
            "confidence": 0.0,
        }
        max_ent = max(y for x, y in map_eindex_entity if x == i_mu.phrase)
        i_ent = (i_mu.phrase, max_ent + 1)
        map_c2e += [(i_mu, i_ent)]
        map_eindex_entity[i_ent] = new_entity

    return map_eindex_entity, map_c2e


def link_candidate_entity(ec_spans: dict, ecl: ExtCandidateList):
    """
        NB: in future (iphrase, i_ent): ent
            will also be used for linking used
    :param ec_spans: (iphrase, i_ent): (span_beg, span_end)
    :param ecl:
    :return:
    """

    # pick phrase indices
    i_e = list(ec_spans.keys())
    ix_phrases = set(k[0] for k in i_e)
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

    # e_index : (iphrase, eindex)
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
) -> tuple[dict[tuple[int, int], dict], list[tuple[MuIndex, tuple[int, int]]]]:
    """

    :param phrases:
    :param ecl:
    :param link_foo: function, maps a phrase to a list of entities
    :param link_foo_kwargs:
    :param map_eindex_entity:
    :param map_c2e:
    :param etype:
    :return:
        i_e -> e ; i_e -> i_mu
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
        entity_pack_current = [(x, y) for x, y in entity_pack_current if x]
        spans = [y for _, y in entity_pack_current]
        entity_normalized_current = [x for x, _ in entity_pack_current]

        existing_ents = [
            k[1] for k in map_eindex_entity if k[0] == ix_current_phrase
        ]

        e_index_max = max(existing_ents) + 1 if existing_ents else 0

        entities_index = [
            (ix_current_phrase, j + e_index_max)
            for j in range(len(entity_normalized_current))
        ]

        entities_index_e_map_current = dict(
            zip(entities_index, entity_normalized_current)
        )

        ei_span_map = dict(zip(entities_index, spans))

        map_c2e_current = link_candidate_entity(ei_span_map, ecl)

        map_c2e.extend(map_c2e_current)
        map_eindex_entity.update(entities_index_e_map_current)

    return map_eindex_entity, map_c2e


def iterate_over_linkers(
    phrases: list[str],
    ecl: ExtCandidateList,
    map_muindex_candidate: dict[MuIndex, Candidate],
    phrase_entities_foos: dict,
) -> tuple[dict[tuple[int, int], dict], list[tuple[MuIndex, tuple[int, int]]]]:

    map_eindex_entity: dict[tuple[int, int], dict] = {}
    map_c2e: list[tuple[MuIndex, tuple[int, int]]] = []

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
