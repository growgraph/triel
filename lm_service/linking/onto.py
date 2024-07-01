from __future__ import annotations

import dataclasses
import logging
from enum import Enum

import requests

from lm_service.onto import BaseDataclass

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


@dataclasses.dataclass(kw_only=True)
class LocalEntity(Entity):
    """
    Entity that also keeps bounds as well as the score
    """

    a: int
    b: int
    score: float


def interval_inclusion_metric(x, y):
    xa, xb = x
    ya, yb = y
    int_a = max([xa, ya])
    int_b = min([xb, yb])
    int_size = max([0, int_b - int_a])
    return (xb - xa) / int_size if int_size > 0 else 0


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

    def normalize(self, response, link_mode, **kwargs) -> list[LocalEntity]:
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
                    EntityLinkerManager._normalize_fishing_entity(
                        item, prob_thr=self[link_mode].threshold
                    )
                    for item in ents
                ]
            else:
                normalized = []
        elif link_mode == EntityLinker.PELINKER:
            if "entities" in response:
                ents = response["entities"]
                normalized = [
                    EntityLinkerManager._normalize_pelinker_entity(
                        item, prob_thr=self[link_mode].threshold
                    )
                    for item in ents
                ]
            else:
                normalized = []
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
        try:
            prob0 = item.pop("score")
        except KeyError:
            logger.warning(f" {item} does not contain prob key")
            prob0 = 1.0
        if prob0 > prob_thr:
            id0 = item["entity"]
            item_spec = id0.split(".")
            ent_db_type, item_id = item_spec
            ent_type = None
            return LocalEntity(
                linker_type=EntityLinker.PELINKER,
                ent_type=ent_type,
                ent_db_type=ent_db_type,
                id=item_id,
                original_form=item["entity_label"],
                a=item["a"],
                b=item["b"],
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
            return None

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
                    return None
            return LocalEntity(
                linker_type=EntityLinker.BERN_V2,
                ent_type=ent_type,
                ent_db_type=ent_db_type,
                id=item_id,
                a=item["span"]["begin"],
                b=item["span"]["end"],
                score=prob0,
            )
        else:
            logger.info(
                f" prob0 > prob_thr did not hold: prob0 {prob0:.3f}, prob_thr {prob_thr:.3f}"
            )
            return None

    @staticmethod
    def _normalize_fishing_entity(item, prob_thr) -> None | LocalEntity:
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
            return LocalEntity(
                linker_type=EntityLinker.FISHING,
                ent_db_type=db_type,
                id=item_id,
                a=item["offsetStart"],
                b=item["offsetEnd"],
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
