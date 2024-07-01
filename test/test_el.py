from lm_service.linking.onto import (
    EntityLinker,
    EntityLinkerManager,
    PhraseMapper,
)
from lm_service.linking.util import link_simple
from lm_service.text import normalize_text


def test_link_phrases(text, nlp_fixture, rules, el_conf):
    elm = EntityLinkerManager.from_dict(el_conf)
    phrases = [text]
    epack = link_simple(
        link_mode=EntityLinker.BERN_V2,
        phrases=phrases,
        elm=elm,
    )
    assert len(epack["annotations"]) == 2
    assert epack["annotations"][0]["id"][0] == "mesh:D017719"


def test_phrasemapper(nlp_fixture):
    pretext = "Diabetic ulcers are related to burns. Autophagy maintains tumour growth through circulating arginine."
    phrases = normalize_text(pretext, nlp_fixture)
    text = " ".join(phrases)
    pm = PhraseMapper(phrases, " ")
    i = 39
    ip, m = pm(i)
    assert text[i : i + 9] == phrases[ip][m : m + 9]
