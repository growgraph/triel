import pathlib

import numpy as np
import pytest
import spacy
import suthing
from suthing import FileHandle

from triel.linking.onto import APISpec, EntityLinker, LocalEntity


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
def rules_v2():
    return suthing.FileHandle.load("triel.config", "prune_noun_compound_v2.yaml")


@pytest.fixture(scope="module")
def rules_v3():
    return suthing.FileHandle.load("triel.config", "prune_noun_compound_v3.yaml")


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
        "TAMs can also secrete in the TME a number of immunosuppressive cytokines, such as IL-6, TGF-β, and IL-10 that are able to suppress CD8+ T-cell function.",
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


@pytest.fixture(scope="module")
def phrase():
    return (
        "This corresponds to the transit of an Earth-sized planet orbiting a"
        " star of 0.9 R☉ in 60 days detected with a S/Ntransit >10 (100 ppm"
        " transit depth). For example, an Earth-size transit across a G star"
        " creates an 80 ppm depth. The different science objectives require"
        " 500 separate target pointings. Assuming 1 hour per pointing the"
        " mission duration is estimated at 1175 days or 3.2 years."
    )


@pytest.fixture(scope="module")
def phrase_b():
    return (
        "Launched on 18 December 2019, "
        "it is the first Small-class mission in "
        "ESA's Cosmic Vision science programme."
    )


@pytest.fixture(scope="module")
def phrase_c():
    return """All discovered tRFs from TCGA can be downloaded from https://cm.jefferson.edu/tcga-mintmap-profiles or studied interactively through the newly-designed version 2.0 of MINTbase at https://cm.jefferson.edu/MINTbase.\\n\\nNOTE: while the manuscript is under review, the content on the page https://cm.jefferson.edu/tcgamintmap-profiles is password protected and available only to Reviewers.\\n\\nKey PointsO_LIComplexity: tRNAs exhibit a complex fragmentation pattern into a multitude of tRFs that are conserved within the samples of a given cancer but differ across cancers.\\nC_LIO_LIVery extensive mitochondrial contributions: the 22 tRNAs of the mitochondrion (MT) contribute 1/3rd of all tRFs found across cancers, a disproportionately high number compared to the tRFs from the 610 nuclear tRNAs.\\nC_LIO_LIUridylated (not guanylated) 5{acute}-His tRFs: in all human tissues analyzed, tRNAHisGTG produces many abundant modified 5{acute}-tRFs with a U at their \\"-1\\" position (-1U 5{acute}-tRFs), instead of a G.\\nC_LIO_LILikely central roles for tRNAHisGTG: the relative abundances of the -1U 5{acute}-tRFs from tRNAHisGTG remain strikingly conserved across the 32 cancers, a property that makes tRNAHisGTG unique among all tRNAs and isoacceptors.\\nC_LIO_LISelective tRF-mRNA networks: tRFs are negatively correlated with mRNAs that differ characteristically from cancer to cancer.\\nC_LIO_LIMitochondrion-encoded tRFs are associated with nuclear proteins: in nearly all cancers, and in a cancer-specific manner, tRFs produced by the 22 mitochondrial tRNAs are negatively correlated with mRNAs whose protein products localize to the nucleus.\\nC_LIO_LItRFs are associated with membrane proteins: in all cancers, and in a cancer-specific manner, nucleus-encoded and MT-encoded tRFs are negatively correlated with mRNAs whose protein products localize to the cells membrane.\\nC_LIO_LItRFs are associated with secreted proteins: in all cancers, and in a cancer-specific manner, nucleusencoded and MT-encoded tRFs are negatively correlated with mRNAs whose protein products are secreted from the cell.\\nC_LIO_LItRFs are associated with numerous mRNAs through repeat elements: in all cancers, and in a cancerspecific manner, the genomic span of mRNAs that are negatively correlated with tRFs are enriched in specific categories of repeat elements.\\nC_LIO_LIintra-cancer tRF networks can depend on sex and population origin: within a cancer, positive and negative tRF-tRF correlations can be modulated by patient attributes such as sex and population origin.\\nC_LIO_LIweb-enabled exploration of an \\"Atlas for tRFs\\": we released a new version of MINTbase to provide users with the ability to study 26,531 tRFs compiled by mining 11,719 public datasets (TCGA and other sources).\\nC_LI"""


@pytest.fixture(scope="module")
def phrase_split_percent():
    return "VE against infection peaked at 49% (95% Confidence Interval (CI): 35-60%) at 2-4 weeks post-vaccination, with waning to a null effect occurring after 10 weeks (VE: 5% (95% CI: -5-14%)). Similarly, VE against symptomatic infection peaked at 49% (95% CI: 32-63%) after 2-4 weeks, waning after 10 weeks (VE: 5% (95% CI: -7-16%))."
