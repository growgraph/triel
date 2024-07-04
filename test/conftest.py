import numpy as np
import pytest
import spacy
import suthing
from suthing import FileHandle

from lm_service.linking.onto import APISpec


def pytest_addoption(parser):
    parser.addoption("--linker-host", action="store", default="localhost")


@pytest.fixture(scope="session")
def linker_host(pytestconfig):
    return pytestconfig.getoption("linker_host")


@pytest.fixture
def el_conf(linker_host):
    config = FileHandle.load("test.config", "el_config.yaml")
    for c in config["linkers"]:
        c["host"] = linker_host
    return config


@pytest.fixture
def lconf(el_conf):
    lconf = APISpec.from_dict(el_conf["linkers"][0])
    return lconf


@pytest.fixture(scope="module")
def nlp_fixture():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")
    return nlp


@pytest.fixture(scope="module")
def bern_example():
    return suthing.FileHandle.load("test.data", "bern.v2.response.json")


@pytest.fixture(scope="module")
def pelinker_example():
    return suthing.FileHandle.load("test.data", "pelinker.response.json")


@pytest.fixture(scope="module")
def rules():
    return suthing.FileHandle.load("lm_service.config", "prune_noun_compound_v2.yaml")


@pytest.fixture(scope="module")
def text():
    return "Diabetic ulcers are related to burns."


@pytest.fixture(scope="module")
def entities():
    return suthing.FileHandle.load("test.data", "entities.json")


@pytest.fixture(scope="module")
def entities_local():
    return suthing.FileHandle.load("test.data", "local.entities.json")


@pytest.fixture(scope="module")
def bern_score():
    return np.array(suthing.FileHandle.load("test.data", "bern.score.json")["scores"])


@pytest.fixture(scope="module")
def strings():
    return "abbbbc aam123", "abc m123"
