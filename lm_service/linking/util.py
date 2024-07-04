from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import partial

import numpy as np
from pathos.multiprocessing import ProcessPool
from suthing import profile
from wordfreq import zipf_frequency

from lm_service.hash import hashme
from lm_service.linking.onto import (
    Entity,
    EntityCandidateAlignmentError,
    EntityLinker,
    EntityLinkerFailed,
    EntityLinkerManager,
    PhraseMapper,
    ent_db_type_gg_verbatim,
    interval_inclusion_metric,
    logger,
)
from lm_service.onto import Candidate, MuIndex
from lm_service.piles import ExtCandidateList


def phrase_to_spacy_basic_entities(phrase=None, rdoc=None, nlp=None):
    # TODO add to EntityLinkerManager
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


def link_unlinked_entities(
    map_eindex_entity: dict[str, Entity],
    map_c2e: list[tuple[MuIndex, str]],
    map_muindex_candidate: dict[MuIndex, Candidate],
    n_token_min=3,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param map_eindex_entity:
    :param map_muindex_candidate:
    :param map_c2e:
    :param n_token_min: min number of tokens to compute hash (could be more if they are infrequent)

    :return:
        i_e -> e ; i_e -> i_mu
    """

    mentions_not_in_entities = set(map_muindex_candidate) - set(c for c, e in map_c2e)

    for i_mu in mentions_not_in_entities:
        c = map_muindex_candidate[i_mu]
        lemmatized = [x.lemma for x in c.tokens]

        zipf_freqs = [
            zipf_frequency(
                word,
                "en",
            )
            for word in lemmatized
        ]
        index_zipf_metric = sorted(enumerate(zipf_freqs), key=lambda x: x[1])
        n_tokens = max(
            [
                len([item for item in index_zipf_metric if item[1] < 5.6]),
                n_token_min,
            ]
        )
        lucky_indices = [k for k, _ in index_zipf_metric[:n_tokens]]
        least_frequent = [w for j, w in enumerate(lemmatized) if j in lucky_indices]
        least_frequent_phrase = " ".join(least_frequent)
        e_id = hashme(" ".join(least_frequent))

        new_entity = Entity(
            linker_type=EntityLinker.GG,
            ent_db_type=ent_db_type_gg_verbatim,
            id=e_id,
            original_form=least_frequent_phrase,
        )
        map_c2e += [(i_mu, new_entity.hash)]
        map_eindex_entity[new_entity.hash] = new_entity

    return map_eindex_entity, map_c2e


def link_candidate_entity(
    phrase_to_ent_spans: defaultdict[int, list[tuple[str, tuple[int, int]]]],
    ecl: ExtCandidateList,
):
    """

    :param phrase_to_ent_spans: iphrase : (i_ent, (span_beg, span_end))
    :param ecl:
    :return: MuIndex(iphrase, sindex, cand_subindex, token_index) : e_index (str)
    """

    map_candidate2entity = []
    # filter on phrase
    ecl.set_filter(lambda x: x[0] in phrase_to_ent_spans.keys())

    for (iphrase, stoken), cand_list in ecl:
        ec_spans_phrase = phrase_to_ent_spans[iphrase]
        for jcandidate, candidate in enumerate(cand_list):
            dist = np.array(
                [
                    [
                        interval_inclusion_metric((t.idx, t.idx_eot), int_ec)
                        for t in candidate.tokens
                    ]
                    for _, int_ec in ec_spans_phrase
                ]
            )
            if dist.size > 0:
                if np.sum((dist > 0) & (dist < 1)) > 0:
                    raise EntityCandidateAlignmentError(
                        "Entity indices and candidate indices are not aligned."
                    )
                (ec_ixs,) = np.where((dist > 0.8).any(axis=1))
                # map current candidate to entity index
                map_candidate2entity += [
                    (
                        MuIndex(False, iphrase, stoken, jcandidate),
                        ec_spans_phrase[i][0],
                    )
                    for i in ec_ixs
                ]

    return map_candidate2entity


def iterate_over_linkers(
    phrases: list[str],
    ecl: ExtCandidateList,
    map_muindex_candidate: dict[MuIndex, Candidate],
    entity_linker_manager: EntityLinkerManager,
    **kwargs,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    map_eindex_entity: dict[str, Entity] = dict()
    map_c2e: list[tuple[MuIndex, str]] = []
    entity_pack_per_phrase: defaultdict[int, list[tuple[str, tuple[int, int]]]] = (
        defaultdict(list)
    )

    sep = " "
    pm = PhraseMapper(phrases, sep)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    text = sep.join(phrases)

    responses = map_linkers(
        text=text, entity_linker_manager=entity_linker_manager, **kwargs
    )

    entity_pack = []
    for link_mode, r in zip(entity_linker_manager.linker_types, responses):
        epack = entity_linker_manager.normalize(r, link_mode, text, **kwargs)
        entity_pack.extend(epack)

    for entity in entity_pack:
        span = entity.a, entity.b
        try:
            ip, (ia, ib) = pm.span(span)
            entity_pack_per_phrase[ip] += [(entity.hash, (ia, ib))]
        except ValueError as ex:
            logger.error(
                f"{ex} : span (mapping) for {entity.hash} was not computed" " correctly"
            )

    map_c2e += link_candidate_entity(entity_pack_per_phrase, ecl)
    map_eindex_entity = {
        **map_eindex_entity,
        **{e.hash: e for e in entity_pack},
    }

    map_eindex_entity, map_c2e = link_unlinked_entities(
        map_eindex_entity, map_c2e, map_muindex_candidate
    )

    map_c2e += link_candidate_entity(entity_pack_per_phrase, ecl)

    map_eindex_entity = {
        **map_eindex_entity,
        **{e.hash: e for e in entity_pack},
    }

    return map_eindex_entity, map_c2e


@profile(_argnames="link_simple")
def link_simple(
    link_mode: EntityLinker, text: str, elm: EntityLinkerManager, **kwargs
) -> dict:
    """

    :param text:
    :param link_mode:
    :param elm:
    """

    try:
        entity_pack = elm.query(text, link_mode)
    except EntityLinkerFailed as e:
        logging.error(f"EntityLinkerFailed es {e}")
        entity_pack = dict()
    return entity_pack


@profile
def map_linkers(text: str, entity_linker_manager: EntityLinkerManager, **kwargs):
    with ProcessPool() as pool:
        responses = pool.map(
            partial(
                link_simple,
                text=text,
                elm=entity_linker_manager,
                **kwargs,
            ),
            entity_linker_manager.linker_types,
        )
    return responses
