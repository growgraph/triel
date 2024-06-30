from __future__ import annotations

import dataclasses
import logging
import os
from collections import defaultdict
from enum import Enum
from functools import partial

import numpy as np
import requests
from pathos.pools import ProcessPool
from suthing import profile
from wordfreq import zipf_frequency

from lm_service.hash import hashme
from lm_service.onto import BaseDataclass, Candidate, MuIndex
from lm_service.piles import ExtCandidateList

logger = logging.getLogger(__name__)


class EntityCandidateAlignmentError(Exception):
    pass


class EntityLinkerKindNotSpecified(Exception):
    pass


class EntityLinkerTypeNotAvailable(Exception):
    pass


class EntityLinkerFailed(Exception):
    pass


class EntityLinker(str, Enum):
    BERN_V2 = "BERN_V2"
    FISHING = "FISHING"
    SPACY_BASIC = "SPACY_BASIC"
    GG = "GG"
    PELINKER = "PELINKER"


ent_db_type_gg_verbatim = "verbatim"


@dataclasses.dataclass
class Entity(BaseDataclass):
    linker_type: EntityLinker
    ent_db_type: str
    id: str
    hash: str = ""
    ent_type: str | None = None
    original_form: str | None = None
    description: str | None = None

    def __post_init__(self):
        self.hash = f"{self.linker_type}.{self.ent_db_type}.{self.id}"

    def as_dict(self):
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
            if v
        }

    def to_dict(self):
        return self.as_dict()


def interval_inclusion_metric(x, y):
    xa, xb = x
    ya, yb = y
    int_a = max([xa, ya])
    int_b = min([xb, yb])
    int_size = max([0, int_b - int_a])
    return (xb - xa) / int_size if int_size > 0 else 0


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


@dataclasses.dataclass
class EntityLinkerManager(BaseDataclass):
    linkers: list[APISpec] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self):
        self._map_linker_ix = {
            EntityLinker(item.keyword): j for j, item in enumerate(self.linkers)
        }

    @property
    def linker_types(self):
        return sorted(self._map_linker_ix.keys())

    def __getitem__(self, key):
        return self.linkers[self._map_linker_ix[key]]

    def query(self, text, link_mode):
        return EntityLinkerManager.query0(
            text,
            link_mode,
            self[link_mode].text_field,
            self[link_mode].url,
            self[link_mode].extra_args,
        )

    def query_and_normalize(
        self, text, link_mode, **kwargs
    ) -> list[tuple[Entity, tuple]]:
        r = self.query(text, link_mode)
        epack = self.normalize(r, link_mode, **kwargs)
        return epack

    @staticmethod
    def query0(text, kind, text_field, url, extras):
        q = requests.post(
            url,
            json={text_field: text, **extras},
            verify=False,
        )
        if q.status_code != 200:
            raise EntityLinkerFailed(
                f" EntityLinker.{kind} failed, possibly down, code" f" {q.status_code}"
            )
        return q.json()

    def normalize(self, response, link_mode, **kwargs) -> list[tuple[Entity, tuple]]:
        if link_mode == EntityLinker.BERN_V2:
            if "annotations" in response:
                ents = response["annotations"]
                normalized = [
                    EntityLinkerManager._normalize_bern_entity(
                        item, prob_thr=self[link_mode].threshold, **kwargs
                    )
                    for item in ents
                ]
            else:
                normalized = []
        elif link_mode == EntityLinker.FISHING:
            if "entities" in response:
                ents = response["entities"]
                normalized = [
                    EntityLinkerManager._normalize_fishing_entity(item) for item in ents
                ]
            else:
                normalized = []
        elif link_mode == EntityLinker.PELINKER:
            if "entities" in response:
                ents = response["entities"]
                normalized = [
                    EntityLinkerManager._normalize_pelinker_entity(item)
                    for item in ents
                ]
            else:
                normalized = []
        else:
            normalized = []

        entity_pack = [(x, y) for x, y in normalized if x is not None and y is not None]
        return entity_pack

    @staticmethod
    def _normalize_pelinker_entity(
        item, prob_thr=0.8, **kwargs
    ) -> tuple[None | Entity, tuple | None]:
        id0 = item["entity"]
        item_spec = id0.split(".")
        ent_db_type, item_id = item_spec
        ent_type = None
        return (
            Entity(
                linker_type=EntityLinker.PELINKER,
                ent_type=ent_type,
                ent_db_type=ent_db_type,
                id=item_id,
                original_form=item["entity_label"],
            ),
            (item["a"], item["b"]),
        )

    @staticmethod
    def _normalize_bern_entity(
        item, prob_thr=0.8, **kwargs
    ) -> tuple[None | Entity, tuple | None]:
        """

        :param item:
        :param prob_thr:
        :return:
        """

        try:
            _ids = item.pop("id")
            id0 = _ids.pop()
        except KeyError:
            logger.warning(f" {item} does not contain ids")
            return None, None
        except IndexError:
            logger.warning(f" {item} contains empty list of ids")
            return None, None

        try:
            prob0 = item.pop("prob")
        except KeyError:
            logger.warning(f" {item} does not contain prob key")
            prob0 = 1.0

        try:
            ent_type = item.pop("obj")
        except KeyError:
            logger.warning(f" {item} does not contain obj key")
            return None, None

        if prob0 > prob_thr:
            if id0 == "CUI-less":
                doc = item["mention"]
                sub_id = "_".join(doc.split(" ")).lower()
                item_id = f"{ent_type}:{sub_id}"
                ent_db_type = "NA"
            else:
                item_spec = id0.split(":")
                try:
                    ent_db_type, item_id = item_spec
                except:
                    logger.warning(
                        " non standard bern entity (does not look like"
                        f" `<ent_type>:<id>`: {item}. NB: most likely CUI-less"
                        " entity"
                    )
                    return None, None
            return Entity(
                linker_type=EntityLinker.BERN_V2,
                ent_type=ent_type,
                ent_db_type=ent_db_type,
                id=item_id,
            ), (item["span"]["begin"], item["span"]["end"])
        else:
            logger.info(
                f" prob0 > prob_thr did not hold: prob0 {prob0:.3f}, prob_thr {prob_thr:.3f}"
            )
            return None, None

    @staticmethod
    def _normalize_fishing_entity(
        item, prob_thr=0.4
    ) -> tuple[Entity | None, tuple | None]:
        if item["confidence_score"] > prob_thr if "confidence_score" in item else True:
            if "wikidataId" in item:
                db_type = "wikidataId"
                item_id = item["wikidataId"]
            elif "wikipediaExternalRef" in item:
                db_type = "wikipediaExternalRef"
                item_id = item["wikipediaExternalRef"]
            else:
                logger.warning(
                    " non standard fishing entity (does not look like"
                    f" `<ent_type>:<id>` : {item}"
                )
                return None, None
            return Entity(
                linker_type=EntityLinker.FISHING,
                ent_db_type=db_type,
                id=item_id,
            ), (item["offsetStart"], item["offsetEnd"])
        else:
            return None, None


@profile
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

    with ProcessPool() as pool:
        responses = pool.map(
            partial(
                link_simple,
                phrases=phrases,
                elm=entity_linker_manager,
                **kwargs,
            ),
            entity_linker_manager.linker_types,
        )

    entity_pack = []
    for link_mode, r in zip(entity_linker_manager.linker_types, responses):
        epack = entity_linker_manager.normalize(r, link_mode, **kwargs)
        entity_pack.extend(epack)

    for entity, span in entity_pack:
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
        **{e.hash: e for e, _ in entity_pack},
    }

    map_eindex_entity, map_c2e = link_unlinked_entities(
        map_eindex_entity, map_c2e, map_muindex_candidate
    )

    map_c2e += link_candidate_entity(entity_pack_per_phrase, ecl)
    map_eindex_entity = {
        **map_eindex_entity,
        **{e.hash: e for e, _ in entity_pack},
    }

    return map_eindex_entity, map_c2e


@profile(_argnames="link_simple")
def link_simple(
    link_mode: EntityLinker, phrases, elm: EntityLinkerManager, **kwargs
) -> list:
    """

    :param phrases:
    :param link_mode:
    :param elm:
    """

    sep = " "
    text = sep.join(phrases)
    try:
        entity_pack = elm.query(text, link_mode)
    except EntityLinkerFailed as e:
        logging.error(f"EntityLinkerFailed es {e}")
        entity_pack = list()
    return entity_pack


@profile(_argnames="link_mode")
def link_over_phrases(
    link_mode: EntityLinker,
    phrases,
    ecl,
    elm: EntityLinkerManager,
    **kwargs,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param phrases:
    :param ecl:
    :param link_mode:
    :param elm:
    :return:
        s_e -> e ; i_c -> i_se
    """
    map_eindex_entity: dict[str, Entity] = dict()
    map_c2e: list[tuple[MuIndex, str]] = list()

    sep = " "
    text = sep.join(phrases)
    pm = PhraseMapper(phrases, sep)
    try:
        entity_pack = elm.query_and_normalize(text, link_mode, **kwargs)
    except EntityLinkerFailed as e:
        logging.error(f"EntityLinkerFailed es {e}")
        entity_pack = list()

    entity_pack_per_phrase: defaultdict[int, list[tuple[str, tuple[int, int]]]] = (
        defaultdict(list)
    )

    for entity, span in entity_pack:
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
        **{e.hash: e for e, _ in entity_pack},
    }
    return map_eindex_entity, map_c2e


class PhraseMapper:
    def __init__(self, phrases, sep=" "):
        lens = [len(s) for s in phrases]

        self.tri = [0]
        acc = 0
        for _len in lens:
            acc += _len + len(sep)
            self.tri += [acc]

    def __call__(self, n):
        """
        n is positive and < self.tri[-1]
        :param n:
        :return:
        """
        j = 0
        while self.tri[j] <= n:
            j += 1
        return j - 1, n - self.tri[j - 1]

    def span(self, sp):
        a, b = sp
        ip_a, ia = self(a)
        ip_b, ib = self(b)
        if ip_a != ip_b:
            raise ValueError(
                f" in PhraseMapper requested span a, b: {a} {b} covers two"
                f" phrases ip_a, ia: {ip_a} {ia} | ip_b, ib: {ip_b} {ib}"
            )
        return ip_a, (ia, ib)


@dataclasses.dataclass
class APISpec(BaseDataclass):
    route: str | None = None
    host: str | None = None
    url: str | None = None
    port: str = "80"
    text_field: str | None = "text"
    protocol: str | None = "http"
    keyword: str | None = None
    threshold: float = 0.8
    extra_args: dict = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        if self.url is None:
            if self.route is None or self.host is None:
                raise ValueError(
                    "self.route is None or self.port is None or self.host is None"
                )
            self.url = f"{self.protocol}://{self.host}:{self.port}/{self.route}"
        else:
            try:
                self.protocol, rest = self.url.split("://")
            except:
                raise ValueError("protocol could not be identified")

            rest = rest.split("/")

            try:
                prefix = rest[0]
                self.route = "/".join(rest[1:])
            except:
                raise ValueError("host and post could not be identified")

            tmp = prefix.split(":")
            if len(tmp) == 1:
                self.port = "80"
            else:
                self.port = tmp[1]
            self.host = tmp[0]
