import numpy as np

from lm_service.piles import ExtCandidateList


def interval_inclusion_metric(x, y):
    xa, xb = x
    ya, yb = y
    int_a = max([xa, ya])
    int_b = min([xb, yb])
    int_size = max([0, int_b - int_a])
    return (xb - xa) / int_size if int_size > 0 else 0


def normalize_bern_entity(item):
    return {
        "linker_type": "bern",
        "ent_type": item["obj"],
        "id": item["id"][0],
    }, (item["span"]["begin"], item["span"]["end"])


def link_candidate_entity(
    ents_normalized, ecl: ExtCandidateList, ix_current_phrase
):
    ec_spans = {
        (ix_current_phrase, n): span
        for n, (doc, span) in enumerate(ents_normalized)
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
                raise ValueError(
                    "Entity indices and candidate indices are not aligned."
                )
            (ec_ixs,) = np.where((dist > 0.8).any(axis=1))
            cand_entity += [
                ((*k, n), (ix_current_phrase, ec_ix)) for ec_ix in ec_ixs
            ]

    # e_index : (iphrase, eindex)
    # c_index : (iphrase, sindex, cand_subindex)
    # 1 -> n : cand -> entity (could be easily generalizable)
    map_c2e = dict(cand_entity)
    return map_c2e
