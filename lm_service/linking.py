from __future__ import annotations

import numpy as np

from lm_service.onto import MuIndex
from lm_service.piles import ExtCandidateList


class EntityCandidateAlignmentError(Exception):
    pass


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
            "linker_type": "bern",
            "ent_type": item["obj"],
            "ent_db_type": db_type,
            "id": item_id,
            "confidence": item["prob"],
            "mention": item["mention"],
        }, (item["span"]["begin"], item["span"]["end"])
    else:
        return None, None


def link_candidate_entity(ec_spans: dict, ecl: ExtCandidateList):
    """
        NB: in futere (iphrase, i_ent): ent
            will also be used for linking used
    :param ec_spans: (iphrase, i_ent): (span_beg, span_end)
    :param ecl:
    :return:
    """

    # pick phrase indices
    i_e = list(ec_spans.keys())
    ix_phrases = set(k[0] for k in i_e)
    # c_index : (iphrase, sindex, cand_subindex, token_index) : [spans]

    cand_entity = []
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
            cand_entity += [(MuIndex(False, *k, n), i_e[i]) for i in ec_ixs]

    # e_index : (iphrase, eindex)
    # c_index : (iphrase, sindex, cand_subindex)
    # 1 -> n : cand -> entity (could be easily generalizable)
    map_c2e = dict(cand_entity)
    return map_c2e


def iterate_linking_over_phrases(
    phrases, ecl, foo, entities_index_e_map, map_c2e, etype="bern"
) -> tuple[dict, dict]:
    """

    :param phrases:
    :param ecl:
    :param foo:
    :param entities_index_e_map:
    :param map_c2e:
    :param etype:
    :return:
        i_e -> e ; i_e -> i_mu
    """
    entity_normalized_foo_map = {"bern": normalize_bern_entity}
    foo_normalize = entity_normalized_foo_map[etype]

    for ix_current_phrase, phrase in enumerate(phrases):
        response = foo(phrase)

        # entities + spans
        entity_pack_current = [foo_normalize(item) for item in response]
        entity_pack_current = [(x, y) for x, y in entity_pack_current if x]
        spans = [y for _, y in entity_pack_current]
        entity_normalized_current = [x for x, _ in entity_pack_current]

        existing_ents = [
            k[0] for k in entities_index_e_map if k[0] == ix_current_phrase
        ]

        e_index_max = max(existing_ents) if existing_ents else 0

        entities_index = [
            (ix_current_phrase, j + e_index_max)
            for j in range(len(entity_normalized_current))
        ]

        entities_index_e_map_current = dict(
            zip(entities_index, entity_normalized_current)
        )

        ei_span_map = dict(zip(entities_index, spans))

        map_c2e_current = link_candidate_entity(ei_span_map, ecl)

        map_c2e.update(map_c2e_current)
        entities_index_e_map.update(entities_index_e_map_current)

    return entities_index_e_map, map_c2e
