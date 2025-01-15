from __future__ import annotations

import logging
import operator
import os
from collections import defaultdict
from functools import partial

import numpy as np
from pathos.multiprocessing import ProcessPool
from suthing import profile
from wordfreq import zipf_frequency

from lm_service.linking.onto import (
    Entity,
    EntityLinker,
    EntityLinkerFailed,
    EntityLinkerManager,
    LocalEntity,
    PhraseMapper,
    ent_db_type_gg_verbatim,
    interval_overlap_metric,
)
from lm_service.linking.score import ScoreMapper
from lm_service.onto import Candidate, MuIndex

logger = logging.getLogger(__name__)


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
    map_c2e: list[tuple[MuIndex, str]],
    map_muindex_candidate: dict[MuIndex, Candidate],
    n_token_min=3,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param map_muindex_candidate:
    :param map_c2e:
    :param n_token_min: min number of tokens to compute hash (could be more if they are infrequent)

    :return:
        i_e -> e ; i_e -> i_mu
    """

    map_c2e_extra: list[tuple[MuIndex, str]] = []
    map_eindex_entity_extra: dict[str, Entity] = dict()

    mentions_not_in_entities = sorted(
        set(map_muindex_candidate) - set(c for c, e in map_c2e)
    )

    for i_mu in mentions_not_in_entities:
        c = map_muindex_candidate[i_mu]
        c = c.normalize()
        if c._root is None:
            continue
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
        least_frequent_phrase = "_".join(least_frequent)
        # e_id = hashme(" ".join(least_frequent))
        e_id = "_".join(least_frequent)

        new_entity = Entity(
            linker_type=EntityLinker.GG,
            ent_db_type=ent_db_type_gg_verbatim,
            id=e_id,
            original_form=least_frequent_phrase,
        )
        map_c2e_extra += [(i_mu, new_entity.hash)]
        map_eindex_entity_extra[new_entity.hash] = new_entity

    return map_eindex_entity_extra, map_c2e_extra


def link_candidate_entity(
    phrase_mapper: PhraseMapper,
    muindex_candidate,
    entities_local,
    score_mapper: ScoreMapper | None = None,
    overlap_thr=0.8,
) -> tuple[list[tuple[MuIndex, str]], list[tuple[str, str, float]]]:
    """

    :param phrase_mapper: PhraseMapper : (i_ent, (span_beg, span_end))
    :param muindex_candidate: MuIndex -> Candidate
    :param entities_local:
    :param score_mapper:
    :param overlap_thr:

    :return: MuIndex(meta, phrase, token, running) : e_index (str)
    """

    prio_entities, entity_edges = process_entities(entities_local, score_mapper)

    phrase_to_ent_spans: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    phrase_to_ents: defaultdict[int, list[LocalEntity]] = defaultdict(list)
    map_candidate2entity = []

    missed_mentions = []

    for entity in prio_entities:
        span = entity.a, entity.b
        ip, (ia, ib) = phrase_mapper.span(span)
        phrase_to_ent_spans[ip] += [(ia, ib)]
        phrase_to_ents[ip] += [entity]

    phrase_candidates: dict[int, list[tuple]] = {}
    for k, v in muindex_candidate.items():
        if k.phrase in phrase_candidates:
            phrase_candidates[k.phrase] += [(k, v)]
        else:
            phrase_candidates[k.phrase] = [(k, v)]

    for iphrase, candidates in phrase_candidates.items():
        for mu, candidate in candidates:
            stoken = mu.token
            dist = np.array(
                [
                    [
                        interval_overlap_metric((t.idx, t.idx_eot), (ea, eb))
                        for t in candidate.tokens
                    ]
                    for ea, eb in phrase_to_ent_spans[iphrase]
                ]
            )
            (ec_ixs,) = np.where((dist > overlap_thr).any(axis=1))
            if not ec_ixs.tolist():
                logger.debug(
                    f"ip: {iphrase} st: {stoken} dmax: {dist.max()}"
                    f"a: {min([t.idx for t in candidate.tokens])} "
                    f"b: {max([t.idx_eot for t in candidate.tokens])} "
                    f"|{' '.join([t.text for t in candidate.tokens])}|"
                )
                missed_mentions += [mu]

            map_candidate2entity += [
                (
                    MuIndex(False, iphrase, stoken, mu.running),
                    phrase_to_ents[iphrase][j].hash,
                )
                for j in ec_ixs
            ]

    return map_candidate2entity, entity_edges


def iterate_over_linkers(
    phrases: list[str],
    entity_linker_manager: EntityLinkerManager,
    sep=" ",
    **kwargs,
) -> list[LocalEntity]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    text = sep.join(phrases)

    responses = map_linkers(
        text=text, entity_linker_manager=entity_linker_manager, **kwargs
    )

    entity_pack = []
    for link_mode, r in zip(entity_linker_manager.linker_types, responses):
        epack = entity_linker_manager.normalize(r, link_mode, text, **kwargs)
        entity_pack.extend(epack)

    return entity_pack


def map_mentions_to_entities(
    phrases,
    entity_pack: list[LocalEntity],
    map_muindex_candidate: dict[MuIndex, Candidate],
    sep=" ",
):
    pm = PhraseMapper(phrases, sep)

    map_c2e: list[tuple[MuIndex, str]]
    ee_edges: list[tuple[str, str, float]]

    map_c2e, ee_edges = link_candidate_entity(pm, map_muindex_candidate, entity_pack)

    normalized_entities = set([Entity.from_local_entity(e) for e in entity_pack])

    map_eindex_entity: dict[str, Entity] = {e.hash: e for e in normalized_entities}

    return map_eindex_entity, map_c2e, ee_edges


@profile(_argnames="link_simple")
def link_simple(
    link_mode: EntityLinker, text: str, elm: EntityLinkerManager, **kwargs
) -> dict | None:
    """

    :param text:
    :param link_mode:
    :param elm:
    """

    try:
        entity_pack = elm.query(text, link_mode)
        return entity_pack
    except EntityLinkerFailed as e:
        logging.error(f"EntityLinkerFailed as {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None


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
    responses = [r for r in responses if r is not None]
    return responses


def process_entity_cluster(
    cluster: list[LocalEntity], score_mapper: ScoreMapper | None = None
) -> tuple[LocalEntity, list[tuple[str, str, float]]]:
    # pick leading candidate

    if score_mapper is not None:
        score_obj = [score_mapper(e.linker_type, e.score) for e in cluster]
    else:
        score_obj = [e.score for e in cluster]

    # TODO: replace by a generic solution
    # motivation: ent_db_type='NA' are least preferred
    # example: LocalEntity(linker_type=<EntityLinker.BERN_V2: 'BERN_V2'>, ent_db_type='NA',
    #                      id='cell_type:tams', hash='BERN_V2.NA.cell_type:tams',
    #                      ent_type='cell_type', original_form=None, description=None, a=0, b=4, score=0.9)
    ent_type = [e.ent_db_type for e in cluster]
    score_multiplier = [(0.5 if t == "NA" else 1.0) for t in ent_type]
    score_obj = [s * m for m, s in zip(score_multiplier, score_obj)]

    # maybe added extra factors into decision

    # metric0 = [
    #     (e.b - e.a, score) for e, score in zip(cluster, score_obj)
    # ]
    # max_size = max([s for s, _ in metric0])
    # score_obj = [score + 0.2*size/max_size for size, score in metric0]

    index, value = max(enumerate(score_obj), key=operator.itemgetter(1))

    principal_entity = cluster[index]
    candidates = cluster[:index] + cluster[index + 1 :]

    # render entity equivalences
    entity_edges = []
    for e in candidates:
        xs = principal_entity.a, principal_entity.b
        ys = e.a, e.b
        weight = interval_overlap_metric(xs, ys)
        entity_edges += [(principal_entity.hash, e.hash, weight)]
    return principal_entity, entity_edges


def render_entity_clusters(entity_pack: list[LocalEntity]) -> list[list[LocalEntity]]:
    clusters: list[list[LocalEntity]] = []

    if entity_pack:
        current_cluster: list[LocalEntity] = [entity_pack[0]]
        pnt = 1
        while pnt < len(entity_pack):
            xs = current_cluster[-1].a, current_cluster[-1].b
            ys = entity_pack[pnt].a, entity_pack[pnt].b
            if interval_overlap_metric(xs, ys) > 0:
                current_cluster += [entity_pack[pnt]]
            else:
                clusters += [current_cluster]
                current_cluster = [entity_pack[pnt]]
            pnt += 1
        clusters += [current_cluster]
        return clusters
    else:
        return []


def process_entities(
    entity_pack: list[LocalEntity], score_mapper: ScoreMapper | None = None
) -> tuple[list[LocalEntity], list[tuple[str, str, float]]]:
    """
    take a list of entities, cast them to clusters, pick one principal entity per cluster

    return a list of principal members and a list edges, connecting each principal entity
        to members of its cluster

    :param entity_pack:
    :param score_mapper:
    :return:
    """
    entity_pack = sorted(entity_pack, key=lambda x: (x.a, x.b))

    clusters = render_entity_clusters(entity_pack)

    edges = []
    principal_entities = []
    for entity_cluster in clusters:
        e, edges0 = process_entity_cluster(entity_cluster, score_mapper)
        principal_entities += [e]
        edges += edges0

    return principal_entities, edges


def render_mention_entity_clusters(
    entity_pack: list[LocalEntity],
) -> list[list[LocalEntity]]:
    """
    transform a list of entities into a list of entity clusters, based on overlap

    :param entity_pack:
    :return:
    """

    pnt = 0
    current_cluster: list[LocalEntity] = []
    clusters: list[list[LocalEntity]] = []

    while pnt < len(entity_pack):
        if current_cluster:
            xs = current_cluster[-1].a, current_cluster[-1].b
            ys = entity_pack[pnt].a, entity_pack[pnt].b
            if interval_overlap_metric(xs, ys) > 0:
                current_cluster += [entity_pack[pnt]]
            else:
                clusters += [current_cluster]
                current_cluster = []
        else:
            current_cluster += [entity_pack[pnt]]
        pnt += 1
    return clusters
