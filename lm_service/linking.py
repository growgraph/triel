from __future__ import annotations

import dataclasses
import logging
import os
from collections import defaultdict
from enum import Enum
from functools import partial

import numpy as np
import requests
from dataclass_wizard import JSONWizard
from pathos.pools import ProcessPool
from spacy.tokens import Token
from suthing import secure_it, time_it
from wordfreq import zipf_frequency

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


class EntityLinkerFailed(Exception):
    pass


class EntityLinker(str, Enum):
    BERN_V2 = "BERN_V2"
    FISHING = "FISHING"
    SPACY_BASIC = "SPACY_BASIC"
    GG = "GG"


ent_db_type_gg_verbatim = "verbatim"


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

    mentions_not_in_entities = set(map_muindex_candidate) - set(
        c for c, e in map_c2e
    )

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
        least_frequent = [
            w for j, w in enumerate(lemmatized) if j in lucky_indices
        ]
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
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    # tuple[dict[str, Entity], list[tuple[MuIndex, str]], dict[str, float]]:

    map_eindex_entity: dict[str, Entity] = {}
    map_c2e: list[tuple[MuIndex, str]] = []

    deco_link_over_phrases = time_it(secure_it(link_over_phrases))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with ProcessPool() as pool:
        rets = pool.map(
            partial(
                deco_link_over_phrases,
                phrases=phrases,
                ecl=ecl,
                elm=entity_linker_manager,
            ),
            list(entity_linker_manager.linker_types),
        )

    for ret in rets:
        map_eindex_entity0, map_c2e0 = ret.ret
        map_c2e += map_c2e0
        map_eindex_entity.update(map_eindex_entity0)

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
    threshold: float = 0.8
    extra_args: dict = dataclasses.field(default_factory=dict)


class EntityLinkerManager:
    def __init__(self, linking_config):
        self.configs: dict[EntityLinker, LinkerConfig] = {
            k: LinkerConfig(**v) for k, v in linking_config.items()
        }

    @property
    def linker_types(self):
        return list(self.configs.keys())

    def set_linker_type(self, ltype: EntityLinker):
        if ltype in self.configs:
            self.current_kind = ltype
        else:
            raise EntityLinkerTypeNotAvailable(
                f" linker type {ltype} was not provided with a config"
            )

    def query(self, text, link_mode):
        return EntityLinkerManager.query0(
            text,
            link_mode,
            self.configs[link_mode].text_field,
            self.configs[link_mode].url,
            self.configs[link_mode].extra_args,
        )

    def query_and_normalize(self, text, link_mode):
        r = self.query(text, link_mode)
        epack = self.normalize(r, link_mode)
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
                f" EntityLinker.{kind} failed, possibly down :"
                f" {q.status_code} {q.json()}"
            )
        return q.json()

    def normalize(self, response, link_mode):
        if link_mode == EntityLinker.BERN_V2:
            ents = response["annotations"]
            return [
                EntityLinkerManager._normalize_bern_entity(
                    item, prob_thr=self.configs[link_mode].threshold
                )
                for item in ents
            ]
        elif link_mode == EntityLinker.FISHING:
            ents = response["entities"]
            return [
                EntityLinkerManager._normalize_fishing_entity(item)
                for item in ents
            ]
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
                    f" `<ent_type>:<id>` : {item}. NB: most likely CUI-less"
                    " entity"
                )
                return None, None
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


def link_over_phrases(
    link_mode: EntityLinker,
    phrases,
    ecl,
    elm: EntityLinkerManager,
    map_eindex_entity=None,
    map_c2e=None,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param phrases:
    :param ecl:
    :param link_mode:
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

    entity_pack = elm.query_and_normalize(text, link_mode)
    entity_pack = [
        (x, y) for x, y in entity_pack if x is not None and y is not None
    ]

    entity_pack_per_phrase: defaultdict[
        int, list[tuple[str, tuple[int, int]]]
    ] = defaultdict(list)

    for entity, span in entity_pack:
        try:
            ip, (ia, ib) = pm.span(span)
            entity_pack_per_phrase[ip] += [(entity.hash, (ia, ib))]
        except ValueError as ex:
            logger.error(
                f"{ex} : span (mapping) for {entity.hash} was not computed"
                " correctly"
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
