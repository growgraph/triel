from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
from enum import Enum

import requests

from lm_service.linking.string import render_gap_mappers, render_index_mapper
from lm_service.onto import BaseDataclass

logger = logging.getLogger(__name__)

EntityHash = str


class EntityCandidateAlignmentError(Exception):
    pass


class EntityLinkerKindNotSpecified(Exception):
    pass


class EntityLinkerTypeNotAvailable(Exception):
    pass


class EntityLinkerFailed(Exception):
    pass


class EntityLinker(str, Enum):
    NA = None
    BERN_V2 = "BERN_V2"
    FISHING = "FISHING"
    SPACY_BASIC = "SPACY_BASIC"
    GG = "GG"
    PELINKER = "PELINKER"
    META = "__META"


ent_db_type_gg_verbatim = "verbatim"


@dataclasses.dataclass
class Entity(BaseDataclass):
    ent_db_type: str
    id: str
    linker_type: EntityLinker = EntityLinker.NA
    hash: EntityHash = ""
    ent_type: str | None = None
    original_form: str | None = None
    description: str | None = None

    def __post_init__(self):
        self.hash = f"{self.linker_type}.{self.ent_db_type}.{self.id}"

    @classmethod
    def from_local_entity(cls, e: LocalEntity):
        return Entity(
            linker_type=e.linker_type,
            ent_db_type=e.ent_db_type,
            id=e.id,
            hash=e.hash,
            ent_type=e.ent_type,
            original_form=e.original_form,
            description=e.description,
        )

    def __hash__(self):
        return int(hashlib.md5(self.hash.encode("utf-8")).hexdigest(), 16)


@dataclasses.dataclass(kw_only=True)
class LocalEntity(Entity):
    """
    Entity that also keeps bounds as well as the score
    """

    a: int
    b: int
    score: float


def interval_overlap_metric(first_item_bnds, second_item_bnds):
    first_item_a, first_item_b = first_item_bnds
    second_item_a, second_item_b = second_item_bnds
    int_a = max([first_item_a, second_item_a])
    int_b = min([first_item_b, second_item_b])
    overlap = max([0, int_b - int_a])
    overlap_norm = min([first_item_b - first_item_a, second_item_b - second_item_a])
    return overlap / overlap_norm if overlap_norm > 0 else 0


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

    def normalize(
        self, response, link_mode, original_text, **kwargs
    ) -> list[LocalEntity]:
        entities = (
            response[self[link_mode].entities_key]
            if self[link_mode].entities_key in response
            else []
        )
        if not entities:
            return []
        normalized_text = response[self[link_mode].normalized_text_key]
        _, mapper_n2o = render_gap_mappers(original_text, normalized_text)
        map_io = render_index_mapper(list(range(len(normalized_text))), mapper_n2o)

        if link_mode == EntityLinker.BERN_V2:
            normalized = [
                EntityLinkerManager._normalize_bern_entity(
                    item, prob_thr=self[link_mode].threshold, mapper_io=map_io, **kwargs
                )
                for item in entities
            ]
        elif link_mode == EntityLinker.FISHING:
            normalized = [
                EntityLinkerManager._normalize_fishing_entity(
                    item, prob_thr=self[link_mode].threshold, mapper_io=map_io
                )
                for item in entities
            ]
        elif link_mode == EntityLinker.PELINKER:
            normalized = [
                EntityLinkerManager._normalize_pelinker_entity(
                    item, prob_thr=self[link_mode].threshold, mapper_io=map_io
                )
                for item in entities
            ]
        else:
            normalized = []

        entity_pack = [x for x in normalized if x is not None]
        logger.info(
            f"for EL {link_mode} {len(normalized) - len(entity_pack)} / {len(normalized)} were filtered "
            f"since they did not pass the confidence threshold or did not contain meaningful data"
        )
        return entity_pack

    @staticmethod
    def _normalize_pelinker_entity(item, prob_thr, **kwargs) -> None | LocalEntity:
        mapper_io = kwargs.pop("mapper_io", dict())

        try:
            prob0 = item.pop("score")
        except KeyError:
            logger.warning(f" {item} does not contain prob key")
            prob0 = 1.0
        if prob0 > prob_thr:
            id0 = item["entity_id_predicted"]
            item_spec = id0.split(".")
            ent_db_type, item_id = item_spec
            ent_type = None

            a = item.pop("a")
            b = item.pop("b")
            a = mapper_io[a] if a in mapper_io else a
            b = mapper_io[b] if b in mapper_io else b

            return LocalEntity(
                linker_type=EntityLinker.PELINKER,
                ent_type=ent_type,
                ent_db_type=ent_db_type,
                id=item_id,
                original_form=item["entity_label"],
                a=a,
                b=b,
                score=prob0,
            )
        else:
            return None

    @staticmethod
    def _normalize_bern_entity(item, prob_thr, **kwargs) -> None | LocalEntity:
        """

        :param item:
        :param prob_thr:
        :return:
        """

        mapper_io = kwargs.pop("mapper_io", dict())
        try:
            _ids = item.pop("id")
            id0 = _ids.pop()
        except KeyError:
            logger.warning(f" {item} does not contain ids")
            return None
        except IndexError:
            logger.warning(f" {item} contains empty list of ids")
            return None

        try:
            prob0 = item.pop("prob")
        except KeyError:
            logger.warning(f" {item} does not contain prob key")
            prob0 = 1.0

        try:
            ent_type = item.pop("obj")
        except KeyError:
            logger.warning(f" {item} does not contain obj key")
            ent_type = "NA"

        if prob0 > prob_thr:
            item_spec = id0.split(":")
            try:
                ent_db_type, item_id = item_spec
            except:
                doc = item["mention"]
                doc2 = re.sub(r"[^a-zA-Z0-9\s]+", " ", doc).lower()
                sub_id = re.sub(r"\s+", "_", doc2)
                item_id = f"{ent_type}:{sub_id}"
                ent_db_type = "NA"
            try:
                span = item.pop("span")
                a = span.pop("begin")
                b = span.pop("end")
            except KeyError:
                logger.warning(f" {item} does not contain span key")
                return None
            a = mapper_io[a] if a in mapper_io else a
            b = mapper_io[b] if b in mapper_io else b
            return LocalEntity(
                linker_type=EntityLinker.BERN_V2,
                ent_type=ent_type,
                ent_db_type=ent_db_type,
                id=item_id,
                a=a,
                b=b,
                score=prob0,
            )
        else:
            logger.info(
                f" prob0 > prob_thr did not hold: prob0 {prob0:.3f}, prob_thr {prob_thr:.3f}"
            )
            return None

    @staticmethod
    def _normalize_fishing_entity(item, prob_thr, **kwargs) -> None | LocalEntity:
        mapper_io = kwargs.pop("mapper_io", dict())
        try:
            prob0 = item.pop("confidence_score")
        except KeyError:
            logger.warning(f" {item} does not contain confidence_score key")
            prob0 = 1.0

        if prob0 > prob_thr:
            if "wikidataId" in item:
                db_type = "wikidataId"
                item_id = str(item["wikidataId"])
            elif "wikipediaExternalRef" in item:
                db_type = "wikipediaExternalRef"
                item_id = str(item["wikipediaExternalRef"])
            else:
                logger.info(f"no wikidataId/ wikipediaExternalRef provided in {item}")
                return None

            a = item["offsetStart"]
            b = item["offsetEnd"]
            a = mapper_io[a] if a in mapper_io else a
            b = mapper_io[b] if b in mapper_io else b

            return LocalEntity(
                linker_type=EntityLinker.FISHING,
                ent_db_type=db_type,
                id=item_id,
                a=a,
                b=b,
                score=prob0,
            )
        else:
            return None


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
    normalized_text_key: str = "text"
    entities_key: str = "entities"
    protocol: str | None = "http"
    keyword: str | None = None
    threshold: float = 0.0
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
