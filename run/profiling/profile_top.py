import argparse

import coreferee
import spacy
import yaml

from lm_service.linking import EntityLinker, EntityLinkerManager
from lm_service.text import (
    normalize_text,
    phrases_to_basis_triples,
    phrases_to_triples,
)
from lm_service.top import (
    cast_response_to_unfolded,
    iterate_over_linkers,
    text_to_rel_graph,
)


def main():
    with open("./lm_service/config/prune_noun_compound_v2.yaml") as fp:
        rules = yaml.load(fp, Loader=yaml.FullLoader)

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    conf_bern = {
        EntityLinker.BERN_V2: {
            "url": "http://10.0.0.3:8888/plain",
            "text_field": "text",
            "threshold": 0.75,
        }
    }

    conf_fishing = {
        EntityLinker.FISHING: {
            "url": "http://10.0.0.3:8090/service/disambiguate",
            "text_field": "text",
            "extra_args": {
                "language": {"lang": "en"},
                "mentions": ["ner", "wikipedia"],
            },
        },
    }

    text = "Diabetic ulcers are related to burns."

    phrases = normalize_text(text, nlp)

    global_triples, map_muindex_candidate, ecl = phrases_to_triples(
        phrases, nlp, rules, window_size=2
    )

    elm = EntityLinkerManager(conf_bern)

    map_eindex_entity, map_c2e = iterate_over_linkers(
        phrases=phrases,
        ecl=ecl,
        map_muindex_candidate=map_muindex_candidate,
        entity_linker_manager=elm,
    )

    elm = EntityLinkerManager(conf_fishing)

    map_eindex_entity, map_c2e = iterate_over_linkers(
        phrases=phrases,
        ecl=ecl,
        map_muindex_candidate=map_muindex_candidate,
        entity_linker_manager=elm,
    )

    # response = text_to_rel_graph(text, nlp, rules, elm)
    # response_jsonlike = cast_response_to_unfolded(
    #     response, cast_triple_version="v1"
    # )
    # pprint(response_jsonlike)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="reset test results"
    )
    args = parser.parse_args()
    main()
