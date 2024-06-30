from lm_service.linking import (
    EntityLinker,
    EntityLinkerManager,
    link_over_phrases,
)
from lm_service.text import phrases_to_basis_triples


def test_bern_normalize(nlp_fixture, rules, bern_example):
    bern_ents = bern_example["annotations"]
    for e in bern_ents:
        EntityLinkerManager._normalize_bern_entity(e, nlp=nlp_fixture)
    pass


def test_link_phrases(text, nlp_fixture, rules, el_conf):
    elm = EntityLinkerManager.from_dict(el_conf)
    phrases = [text]
    (
        striples,
        striples_meta,
        candidate_depot,
        relations,
        _,
    ) = phrases_to_basis_triples(nlp_fixture, rules, phrases)

    ecl = candidate_depot.unfold_conjunction()
    map_eindex_entity, map_c2e = link_over_phrases(
        link_mode=EntityLinker.BERN_V2,
        phrases=phrases,
        ecl=ecl,
        elm=elm,
        nlp=nlp_fixture,
    )
    assert True
