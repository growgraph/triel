import pytest
import spacy


@pytest.fixture(scope="module")
def nlp_fixture():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    return nlp
