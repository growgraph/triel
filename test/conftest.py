import pathlib

import numpy as np
import pytest
import spacy
import suthing
from suthing import FileHandle

from lm_service.linking.onto import APISpec, EntityLinker, LocalEntity


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
    entities = suthing.FileHandle.load("test.data", "entities.json")
    entity_pack = [LocalEntity.from_dict(item) for item in entities]
    return sorted(entity_pack, key=lambda x: (x.a, x.b))


@pytest.fixture(scope="module")
def entities_local():
    _entities = suthing.FileHandle.load("test.data", "entities.local.sample.json")
    entity_pack = [LocalEntity.from_dict(item) for item in _entities]
    return sorted(entity_pack, key=lambda x: (x.a, x.b))


@pytest.fixture(scope="module")
def bern_score():
    return np.array(suthing.FileHandle.load("test.data", "bern.score.json")["scores"])


@pytest.fixture(scope="module")
def strings():
    return "abbbbc aam123", "abc m123"


@pytest.fixture(scope="module")
def ecl():
    return FileHandle.load("test.data", "ext_candidate_list.pkl")


@pytest.fixture(scope="module")
def phrase_mapper():
    return FileHandle.load("test.data", "phrase_mapper.pkl")


@pytest.fixture(scope="module")
def doc_coref():
    return FileHandle.load("test.data", "doc.coref.pkl")


@pytest.fixture(scope="module")
def muindex_candidate():
    return FileHandle.load("test.data", "muindex_candidate.pkl")


@pytest.fixture(scope="module")
def score_mapper_trivial():
    return lambda _, y: y


@pytest.fixture(scope="module")
def entity_cluster():
    cluster_dict = [
        {
            "linker_type": EntityLinker.FISHING,
            "ent_db_type": "wikidataId",
            "id": "Q376266",
            "hash": "FISHING.wikidataId.Q376266",
            "a": 136,
            "b": 139,
            "score": 0.8945,
        },
        {
            "linker_type": EntityLinker.BERN_V2,
            "ent_db_type": "NCBIGene",
            "id": "925",
            "hash": "BERN_V2.NCBIGene.925",
            "ent_type": "gene",
            "a": 136,
            "b": 141,
            "score": 0.93408203125,
        },
        {
            "linker_type": EntityLinker.BERN_V2,
            "ent_db_type": "NA",
            "id": "cell_type:cd8_+_t_-_cell",
            "hash": "BERN_V2.NA.cell_type:cd8_+_t_-_cell",
            "ent_type": "cell_type",
            "a": 136,
            "b": 150,
            "score": 0.8454045057296753,
        },
    ]
    return [LocalEntity.from_dict(item) for item in cluster_dict]


@pytest.fixture(scope="module")
def phrases_for_coref():
    return (
        "Although he was very busy with his work, Peter Brown had had enough of it.",
        "He and his wife decided they needed a holiday.",
        "They travelled to Spain because they loved the country very much.",
    )


@pytest.fixture(scope="module")
def fig_path():
    fig_path = pathlib.Path("./test/figs")
    fig_path.mkdir(parents=True, exist_ok=True)
    return fig_path


@pytest.fixture(scope="module")
def map_tree_subtree_index():
    return suthing.FileHandle.load("test.data", "map_tree_subtree_index.pkl")


@pytest.fixture(scope="module")
def text_cheops():
    return suthing.FileHandle.load("test.data", "sample.cheops.json")


@pytest.fixture(scope="module")
def documents():
    return [
        "The medium was affected by the near-field radiation",
        (
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass, density,"
            " composition and their formation."
        ),
        "He treated her unfairly.",
    ]


@pytest.fixture(scope="module")
def sample_a():
    return suthing.FileHandle.load("test.data", "sample.a.json")
