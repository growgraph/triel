from lm_service.linking.onto import (
    EntityLinker,
    EntityLinkerManager,
    PhraseMapper,
)
from lm_service.linking.util import link_simple


def test_link_phrases(text, nlp_fixture, rules_v2, el_conf):
    elm = EntityLinkerManager.from_dict(el_conf)
    epack = link_simple(
        link_mode=EntityLinker.BERN_V2,
        text=text,
        elm=elm,
    )
    assert len(epack["annotations"]) == 2
    assert epack["annotations"][0]["id"][0] == "mesh:D017719"


def test_phrasemapper(nlp_fixture):
    phrases = [
        "Diabetic ulcers are related to burns. "
        "Autophagy maintains tumour growth through circulating arginine."
    ]
    text = " ".join(phrases)
    pm = PhraseMapper(phrases, " ")
    i = 39
    ip, m = pm(i)
    assert text[i : i + 9] == phrases[ip][m : m + 9]
