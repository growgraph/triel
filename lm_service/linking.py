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


def normalize_bern_entity(item) -> dict | None:
    if len(item["id"]) > 0:
        item_spec = item["id"][0].split(":")
        db_type, item_id = item_spec
        return {
            "linker_type": "bern",
            "ent_type": item["obj"],
            "ent_db_type": db_type,
            "id": item_id,
            "confidence": item["prob"],
            "span": (item["span"]["begin"], item["span"]["end"]),
        }
    else:
        return None


def link_candidate_entity(
    ents_normalized, ecl: ExtCandidateList, ix_current_phrase
):
    ec_spans = {
        (ix_current_phrase, n): doc["span"]
        for n, doc in enumerate(ents_normalized)
    }

    # c_index : (iphrase, sindex, cand_subindex, token_index) : [spans]
    # e_index : (iphrase, eindex) :

    cand_entity = []
    ecl.set_filter(lambda x: x[0] == ix_current_phrase)
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
            cand_entity += [
                (MuIndex(False, *k, n), (ix_current_phrase, ec_ix))
                for ec_ix in ec_ixs
            ]

    # e_index : (iphrase, eindex)
    # c_index : (iphrase, sindex, cand_subindex)
    # 1 -> n : cand -> entity (could be easily generalizable)
    map_c2e = dict(cand_entity)
    return map_c2e


def iterate_linking_over_phrases(phrases, ecl, foo, etype="bern"):
    for ix_current_phrase, phrase in enumerate(phrases):
        response = foo(phrase)

        entity_normalized_foo_map = {"bern": normalize_bern_entity}
        foo_normalize = (
            entity_normalized_foo_map[etype]
            if etype in entity_normalized_foo_map
            else None
        )

        entity_normalized = [foo_normalize(item) for item in response]

        entity_normalized = [x for x in entity_normalized if x]

        map_c2e = link_candidate_entity(
            entity_normalized, ecl, ix_current_phrase
        )
        return entity_normalized, map_c2e
