from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from enum import Enum

import numpy as np
import requests
from dataclass_wizard import JSONWizard
from spacy.tokens import Token

from lm_service.hash import hashme
from lm_service.onto import Candidate, MuIndex
from lm_service.piles import ExtCandidateList
from lm_service.util import Timer

logger = logging.getLogger(__name__)


class EntityCandidateAlignmentError(Exception):
    pass


class EntityLinkerKindNotSpecified(Exception):
    pass


class EntityLinkerTypeNotAvailable(Exception):
    pass


class EntityLinker(str, Enum):
    BERN_V2 = "BERN_V2"
    FISHING = "FISHING"
    SPACY_BASIC = "SPACY_BASIC"
    LOCAL_NON_EL = "LOCAL_NON_EL"


ent_db_type_local_gg = "ent_db_type_local_gg"


@dataclasses.dataclass(repr=False)
class Entity(JSONWizard):
    """
    represents a token in dep tree
    """

    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"
        skip_defaults = True

    linker_type: EntityLinker
    ent_db_type: str
    id: str
    hash: str = ""
    ent_type: str | None = None
    original_form: str | None = None
    description: str | None = None

    def __post_init__(self):
        self.hash = f"{self.linker_type}/{self.ent_db_type}/{self.id}"

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
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param map_eindex_entity:
    :param map_muindex_candidate:
    :param map_c2e:

    :return:
        i_e -> e ; i_e -> i_mu
    """

    mentions_not_in_entities = set(map_muindex_candidate) - set(
        c for c, e in map_c2e
    )

    for i_mu in mentions_not_in_entities:
        c = map_muindex_candidate[i_mu]
        original_form = " ".join(c.project_to_text()).lower()

        e_id = c.hashme()

        new_entity = Entity(
            linker_type=EntityLinker.LOCAL_NON_EL,
            ent_db_type=ent_db_type_local_gg,
            id=e_id,
            original_form=original_form,
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
    elm: EntityLinkerManager,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    map_eindex_entity: dict[str, Entity] = {}
    map_c2e: list[tuple[MuIndex, str]] = []

    for link_mode in elm.linker_types:
        elm.set_linker_type(link_mode)
        with Timer() as t_linking:
            try:
                map_eindex_entity, map_c2e = link_over_phrases(
                    phrases=phrases,
                    ecl=ecl,
                    elm=elm,
                    map_eindex_entity=map_eindex_entity,
                    map_c2e=map_c2e,
                )
            except Exception as e:
                logger.error(
                    f"in iterate_over_linkers, linker {link_mode} failed."
                )

                logger.error(f" <exception_start>: {e} <exception_end>")

                logger.error(f" ex : {phrases}")

        logger.info(
            f" linking for {link_mode} took {t_linking.elapsed:.2f} sec"
        )

    map_eindex_entity, map_c2e = link_unlinked_entities(
        map_eindex_entity, map_c2e, map_muindex_candidate
    )
    return map_eindex_entity, map_c2e


@dataclasses.dataclass()
class LinkerConfig(JSONWizard):
    """
    represents a token in dep tree
    """

    class _(JSONWizard.Meta):
        key_transform_with_dump = "SNAKE"

    url: str
    text_field: str
    extra_args: dict = dataclasses.field(default_factory=dict)


class EntityLinkerManager:
    def __init__(self, linking_config):
        self.configs: dict[EntityLinker, LinkerConfig] = {
            k: LinkerConfig(**v) for k, v in linking_config.items()
        }
        self.current_kind: EntityLinker | None = None

    @property
    def linker_types(self):
        return self.configs.keys()

    def set_linker_type(self, ltype: EntityLinker):
        if ltype in self.configs:
            self.current_kind = ltype
        else:
            raise EntityLinkerTypeNotAvailable(
                f" linker type {ltype} was not provided with a config"
            )

    def query(self, text):
        return EntityLinkerManager.query0(
            text,
            self.configs[self.current_kind].text_field,
            self.configs[self.current_kind].url,
            self.configs[self.current_kind].extra_args,
        )

    def query_and_normalize(self, text):
        r = self.query(text)
        epack = self.normalize(r)
        return epack

    @staticmethod
    def query0(text, text_field, url, extras):
        # TODO error code processing
        return requests.post(
            url,
            json={text_field: text, **extras},
            verify=False,
        ).json()

    def normalize(self, response):
        if self.current_kind == EntityLinker.BERN_V2:
            ents = response["annotations"]
            return [
                EntityLinkerManager._normalize_bern_entity(item)
                for item in ents
            ]
        elif self.current_kind == EntityLinker.FISHING:
            ents = response["entities"]
            return [
                EntityLinkerManager._normalize_fishing_entity(item)
                for item in ents
            ]
        # elif self.current_kind == EntityLinker.SPACY_BASIC:
        #     return EntityLinkerManager._normalize_spacy_basic(response)
        else:
            return None, None

    @staticmethod
    def _normalize_bern_entity(
        item, prob_thr=0.8
    ) -> tuple[Entity | None, tuple | None]:
        if len(item["id"]) > 0 and (
            item["prob"] > prob_thr if "prob" in item else True
        ):
            item_spec = item["id"][0].split(":")
            try:
                db_type, item_id = item_spec
            except:
                logger.warning(
                    " non standard bern entity (does not look like"
                    f" `<ent_type>:<id>` : {item}"
                )
                item_id = item["id"][0]
                db_type = None
            return Entity(
                linker_type=EntityLinker.BERN_V2,
                ent_type=item["obj"],
                ent_db_type=db_type,
                id=item_id,
            ), (item["span"]["begin"], item["span"]["end"])
        else:
            return None, None

    @staticmethod
    def _normalize_fishing_entity(
        item, prob_thr=0.4
    ) -> tuple[Entity | None, tuple | None]:
        if (
            item["confidence_score"] > prob_thr
            if "confidence_score" in item
            else True
        ):
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

    @staticmethod
    def _normalize_spacy_basic(
        item: list[Token],
    ) -> tuple[Entity | None, tuple | None]:
        """

        :param item:
        :return:
        """
        if item:
            span = item[0].idx, item[-1].idx + len(item[-1].text)
            original_form = " ".join([x.text for x in item]).lower()
            e_id = hashme(original_form)
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


def link_over_phrases(
    phrases,
    ecl,
    elm: EntityLinkerManager,
    map_eindex_entity=None,
    map_c2e=None,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param phrases:
    :param ecl:
    :param elm:
    :param map_eindex_entity:
    :param map_c2e:
    :return:
        s_e -> e ; i_c -> i_se
    """
    if map_eindex_entity is None:
        map_eindex_entity = dict()
    if map_c2e is None:
        map_c2e = list()

    sep = " "
    text = sep.join(phrases)
    pm = PhraseMapper(phrases, sep)

    entity_pack = elm.query_and_normalize(text)
    entity_pack = [
        (x, y) for x, y in entity_pack if x is not None and y is not None
    ]

    entity_pack_per_phrase: defaultdict[
        int, list[tuple[str, tuple[int, int]]]
    ] = defaultdict(list)

    for e, span in entity_pack:
        ip, (ia, ib) = pm.span(span)
        entity_pack_per_phrase[ip] += [(e.hash, (ia, ib))]

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
        for l in lens:
            acc += l + len(sep)
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
