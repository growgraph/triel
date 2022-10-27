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

# only v2 is supported by the API
api_spec = {
    "v1": {
        "url": "https://bern.korea.ac.kr/plain",
        "text_field": "sample_text",
    },
    "v2": {"url": "http://bern2.korea.ac.kr/plain", "text_field": "text"},
}


class EntityCandidateAlignmentError(Exception):
    pass


class EntityLinker(str, Enum):
    BERN_V2 = "BERN_V2"
    SPACY_NAIVE_WIKI = "SPACY_NAIVE_WIKI"
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


def normalize_bern_entity(
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


def normalize_naive_wiki_entity_linker(
    item,
) -> tuple[Entity | None, tuple | None]:
    """
    spacy-entity-linker package

    :param item:
    :return:
    """
    if item.get_url():
        span = item.get_span()
        item_id = item.get_url().split("/")[-1]

        try:
            ee = next(iter(item.get_categories()))
            ent_type = ee.get_url().split("/")[-1]
        except:
            ent_type = None
        return Entity(
            linker_type=EntityLinker.SPACY_NAIVE_WIKI,
            ent_db_type="wikidata",
            ent_type=ent_type,
            id=item_id,
            description=item.get_description(),
            original_form=item.get_original_alias(),
        ), (span.start_char, span.end_char)

    else:
        return None, None


def phrase_to_spacy_basic_entities(phrase=None, rdoc=None, nlp=None):
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


def normalize_spacy_basic(
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

    # create entities for unlinked candidates (for some candidates entities were not found)

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


entity_normalized_foo_map = {
    EntityLinker.BERN_V2: normalize_bern_entity,
    EntityLinker.SPACY_NAIVE_WIKI: normalize_naive_wiki_entity_linker,
    EntityLinker.SPACY_BASIC: normalize_spacy_basic,
}


def link_over_phrases(
    phrases,
    ecl,
    link_foo,
    link_foo_kwargs=None,
    map_eindex_entity=None,
    map_c2e=None,
    etype=EntityLinker.BERN_V2,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:
    """

    :param phrases:
    :param ecl:
    :param link_foo: function, maps a phrase to a list of entities
    :param link_foo_kwargs:
    :param map_eindex_entity:
    :param map_c2e:
    :param etype:
    :return:
        s_e -> e ; i_c -> i_se
    """
    if link_foo_kwargs is None:
        link_foo_kwargs = dict()
    if map_eindex_entity is None:
        map_eindex_entity = dict()
    if map_c2e is None:
        map_c2e = list()

    foo_normalize = entity_normalized_foo_map[etype]
    sep = " "
    text = sep.join(phrases)
    pm = PhraseMapper(phrases, sep)

    response = link_foo(text, **link_foo_kwargs)
    entity_pack = [foo_normalize(item) for item in response]  # type: ignore
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


def iterate_over_linkers(
    phrases: list[str],
    ecl: ExtCandidateList,
    map_muindex_candidate: dict[MuIndex, Candidate],
    phrase_entities_foos: dict,
) -> tuple[dict[str, Entity], list[tuple[MuIndex, str]]]:

    map_eindex_entity: dict[str, Entity] = {}
    map_c2e: list[tuple[MuIndex, str]] = []

    for link_mode, link_foo in phrase_entities_foos.items():
        with Timer() as t_linking:

            try:
                map_eindex_entity, map_c2e = link_over_phrases(
                    phrases=phrases,
                    ecl=ecl,
                    map_eindex_entity=map_eindex_entity,
                    map_c2e=map_c2e,
                    link_foo=link_foo,
                    etype=link_mode,
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


def query_bern(text, version="v2"):
    url = api_spec[version]["url"]
    text_field = api_spec[version]["text_field"]
    return requests.post(url, json={text_field: text}, verify=False).json()


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
