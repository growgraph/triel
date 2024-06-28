import logging
from pprint import pprint

import pytest
from suthing import FileHandle

from lm_service.coref import graph_component_maps, render_coref_maps_wrapper
from lm_service.graph import phrase_to_deptree, relabel_nodes_and_key
from lm_service.onto import AbsToken, MuIndex, apply_map
from lm_service.phrase import graph_to_triples
from lm_service.relation import (
    compute_distances,
    generate_extra_graphs,
    graph_to_candidate_pile,
)
from lm_service.text import (
    cast_simplified_triples_table,
    normalize_text,
    phrases_to_triples,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def rules():
    return FileHandle.load("lm_service.config", "prune_noun_compound_v2.yaml")


@pytest.fixture
def documents():
    return {
        "near-field": "The medium was affected by the near-field radiation",
        "cheops0_trunc": (
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass"
        ),
        "cheops0": (
            "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
            " telescope to determine the size of known extrasolar planets,"
            " which will allow the estimation of their mass, density,"
            " composition and their formation."
        ),
        "coref": (
            "Although he was very busy with his work, Peter Brown had had"
            " enough of it. He and his wife decided they needed a holiday."
            " They travelled to Spain because they loved the country very"
            " much."
        ),
        "cheops_ext": (
            "Cheops ( CHaracterising ExOPlanets Satellite ) is a European"
            " space telescope to determine the size of known extrasolar"
            " planets , which will allow the estimation of their mass ,"
            " density , composition and their formation. It is the first Small"
            " class mission in ESA 's Cosmic Vision science programme Launched"
            " on 18 December 2019"
        ),
        "photometric": (
            "Cheops measures photometric signals with a precision limited by"
            " stellar photon noise of 150 ppm min for a 9th magnitude star."
            " This corresponds to the transit of an Earth sized planet"
            " orbiting a star of 0 . 9 R in 60 days  detected with a S"
            " Ntransit > 10 ( 100 ppm transit depth )"
        ),
        "thousands": (
            "Thousands of exoplanets have been discovered by the end of the"
            " 2010s; some have minimum mass measurements from the radial"
            " velocity method while others that are seen to transit their"
            " parent stars have measures of their physical size."
        ),
        # "terminals_test": (
        #     "For the planned mission duration of 3.5 years, CHEOPS is to"
        #     " measure the size of known transiting exoplanets orbiting bright"
        #     " and nearby stars  as well as search for transits of exoplanets"
        #     " previously discovered via radial velocity."
        # ),
    }


@pytest.fixture
def reference_projected():
    return {
        "near-field": [("medium", "isAffectedBy", "nearFieldRadiation")],
        "cheops0_trunc": [
            ("CHEOPS", "is", "europeanSpaceTelescope"),
            (
                "europeanSpaceTelescope",
                "determines",
                "sizeOfKnownExtrasolarPlanets",
            ),
        ],
        "cheops_ext": [
            ("Cheops", "is", "europeanSpaceTelescope"),
            (
                "europeanSpaceTelescope",
                "determines",
                "sizeOfKnownExtrasolarPlanets",
            ),
            ("europeanSpaceTelescope", "allows", "estimationOfDensity"),
            (
                "europeanSpaceTelescope",
                "allows",
                "estimationOfComposition",
            ),
            (
                "europeanSpaceTelescope",
                "allows",
                "estimationOfMassOfKnownExtrasolarPlanets",
            ),
            (
                "europeanSpaceTelescope",
                "allows",
                "estimationOfFormationOfKnownExtrasolarPlanets",
            ),
            (
                "europeanSpaceTelescope",
                "is",
                "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
            ),
            (
                "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
                "LaunchedOn",
                "18December2019",
            ),
        ],
        "photometric": [
            ("Cheops", "measures", "photometricSignals"),
            ("Cheops", "measuresWith", "precisionFor9thMagnitudeStar"),
            (
                "precisionFor9thMagnitudeStar",
                "limitedBy",
                "stellarPhotonNoiseOf150PpmMin",
            ),
            ("This", "correspondsTo", "transitOfEarthSizedPlanetIn60Day"),
            ("transitOfEarthSizedPlanetIn60Day", "orbited", "star"),
            ("transitOfEarthSizedPlanetIn60Day", "orbitedOf", "09R"),
        ],
        "thousands": [
            ("thousandOfExoplanets", "isDiscoveredBy", "endOf2010"),
            (
                "some",
                "has",
                "minimumMassMeasurementsFromRadialVelocityMethod",
            ),
            ("others", "isSeen", "toTransitParentStarsOfOthers"),
            ("others", "has", "measuresOfPhysicalSizeOfOthers"),
        ],
    }


@pytest.fixture
def text():
    return (
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space"
        " telescope to determine the size of known extrasolar planets,"
        " which will allow the estimation of their mass, density,"
        " composition and their formation."
        " Launched on 18 December 2019,"
        " it is the first Small-class mission in ESA's Cosmic Vision"
        " science programme.The small satellite features an optical"
        " Ritchey-Chrétien telescope with an aperture of 30 cm, mounted on"
        " a standard small satellite platform. It was placed into a"
        " Sun-synchronous orbit of about 700 km altitude. Thousands of"
        " exoplanets have been discovered by the end of the 2010s; some"
        " have minimum mass measurements from the radial velocity method"
        " while others that are seen to transit their parent stars have"
        " measures of their physical size."
    )


@pytest.mark.skip()
def test_distances(nlp_fixture, rules, documents):
    distance_check = {}
    for key, document in documents.items():
        rdoc, graph0 = phrase_to_deptree(nlp_fixture, document)

        # cast index to compound index
        map_tree_subtree_index = graph_component_maps(graph0)
        map_tree_subtree_index = {
            k: AbsToken.ituple2stuple(v) for k, v in map_tree_subtree_index.items()
        }

        graph_relabeled = relabel_nodes_and_key(graph0, map_tree_subtree_index, "s")

        pile, _, mod_graph = graph_to_candidate_pile(graph_relabeled, rules)

        g_undirected, g_reversed, g_weighted = generate_extra_graphs(graph_relabeled)
        relation_indices = [c.root.s for c in pile.relations]
        (
            distance_undirected,
            distance_directed,
            distance_reversed,
            distance_levels,
        ) = compute_distances(
            mod_graph,
            g_undirected=g_undirected,
            g_weighted=g_weighted,
            g_reversed=g_reversed,
            indices_of_interest=relation_indices,
        )

        distance_check.update(
            {
                key: {
                    "undirected": distance_undirected,
                    "directed": distance_directed,
                    "levels": distance_levels,
                }
            }
        )

    # assert distance_check == reference_distance


def test_relation(nlp_fixture, rules, documents, reference_projected):
    documents = {
        key: documents[key]
        for key in [
            "near-field",
            "cheops0_trunc",
            "cheops_ext",
            "photometric",
            "thousands",
        ]
    }
    acc_triples = []
    triples_projected = {}

    for key, doc in documents.items():
        rdoc, graph = phrase_to_deptree(nlp=nlp_fixture, document=doc)

        # cast index to compound index
        map_tree_subtree_index = graph_component_maps(graph)
        map_tree_subtree_index = {
            k: AbsToken.ituple2stuple(v) for k, v in map_tree_subtree_index.items()
        }
        graph_relabeled = relabel_nodes_and_key(graph, map_tree_subtree_index, "s")

        # coref maps
        (
            map_subbable_to_chain,
            map_chain_to_most_specific,
        ) = render_coref_maps_wrapper(rdoc)

        (
            map_subbable_to_chain_str,
            map_chain_to_most_specific_str,
        ) = apply_map(
            [map_subbable_to_chain, map_chain_to_most_specific],
            map_tree_subtree_index,
        )

        triples = graph_to_triples(
            graph_relabeled,
            map_subbable_to_chain_str,
            map_chain_to_most_specific_str,
            rules,
        )
        triples = [tri.normalize_relation() for tri in triples]
        acc_triples += triples

        triples_projected[key] = [tri.project_to_text() for tri in triples]

    for k in triples_projected:
        if set(triples_projected[k]) != set(reference_projected[k]):
            projected_ = set(triples_projected[k]) - set(reference_projected[k])
            refs_ = set(reference_projected[k]) - set(triples_projected[k])
            print(k)
            print("new")
            pprint(sorted(projected_))
            print("ref")
            pprint(sorted(refs_))

        # assert triples_projected[k] == reference_projected[k]


def test_text_to_relations(nlp_fixture, text, rules):
    phrases = normalize_text(text, nlp_fixture)

    global_triples, map_mu_index_triple, _ = phrases_to_triples(
        phrases, nlp_fixture, rules, window_size=2
    )

    global_triples_txt = cast_simplified_triples_table(
        global_triples, map_mu_index_triple
    )

    reference = {
        MuIndex(meta=True, phrase=0, token="000", running=0): (
            "CHEOPS",
            "is",
            "europeanSpaceTelescope",
        ),
        MuIndex(meta=True, phrase=0, token="000", running=1): (
            "europeanSpaceTelescope",
            "determines",
            "sizeOfKnownExtrasolarPlanets",
        ),
        MuIndex(meta=True, phrase=1, token="000", running=0): (
            "CHEOPS",
            "is",
            "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
        ),
        MuIndex(meta=True, phrase=1, token="000", running=1): (
            "firstSmallClassMissionInEsaCosmicVisionScienceProgramme",
            "launchedOn",
            "18December2019",
        ),
        MuIndex(meta=True, phrase=2, token="000", running=0): (
            "smallSatellite",
            "features",
            "opticalRitcheyChretienTelescopeWithApertureOf30Cm",
        ),
        MuIndex(meta=True, phrase=2, token="000", running=1): (
            "opticalRitcheyChretienTelescopeWithApertureOf30Cm",
            "mountedOn",
            "standardSmallSatellitePlatform",
        ),
        MuIndex(meta=True, phrase=3, token="000", running=0): (
            "opticalRitcheyChretienTelescopeWithApertureOf30Cm",
            "isPlacedInto",
            "SunSynchronousOrbitOf700KmAltitude",
        ),
        MuIndex(meta=True, phrase=4, token="000", running=0): (
            "some",
            "has",
            "minimumMassMeasurementsFromRadialVelocityMethod",
        ),
        MuIndex(meta=True, phrase=4, token="000", running=1): (
            "thousandOfExoplanets",
            "isDiscoveredBy",
            "endOf2010",
        ),
        MuIndex(meta=True, phrase=4, token="000", running=2): (
            "others",
            "has",
            "measuresOfPhysicalSizeOfOthers",
        ),
        MuIndex(meta=True, phrase=4, token="000", running=3): (
            "others",
            "isSeen",
            "toTransitParentStarsOfOthers",
        ),
        MuIndex(meta=True, phrase=0, token="000", running=2): (
            "(meta)determines",
            "allows",
            "estimationOfDensity",
        ),
        MuIndex(meta=True, phrase=0, token="000", running=3): (
            "(meta)determines",
            "allows",
            "estimationOfComposition",
        ),
        MuIndex(meta=True, phrase=0, token="000", running=4): (
            "(meta)determines",
            "allows",
            "estimationOfMassOfKnownExtrasolarPlanets",
        ),
        MuIndex(meta=True, phrase=0, token="000", running=5): (
            "(meta)determines",
            "allows",
            "estimationOfFormationOfKnownExtrasolarPlanets",
        ),
    }

    for k in global_triples_txt:
        if global_triples_txt[k] != reference[k]:
            print(k)
            print("new")
            pprint(global_triples_txt[k])
            print("ref")
            pprint(reference[k])
        # assert global_triples_txt[k] == reference[k]
