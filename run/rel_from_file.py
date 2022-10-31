import argparse
import pkgutil
from pprint import pprint

import coreferee
import spacy
import yaml
from graph_cast.util import ResourceHandler

from lm_service.linking import EntityLinkerManager
from lm_service.top import cast_response_to_unfolded, text_to_rel_graph


def main(fpath, entity_linker_config):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    fp = pkgutil.get_data("lm_service.config", "prune_noun_compound_v2.yaml")
    rules = yaml.load(fp, Loader=yaml.FullLoader)

    elm = EntityLinkerManager(entity_linker_config)

    text = ResourceHandler.load(fpath=fpath)["text"]
    response = text_to_rel_graph(text, nlp, rules, elm)
    response_jsonlike = cast_response_to_unfolded(
        response, cast_triple_version="v1"
    )
    pprint(response_jsonlike)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json-fpath",
        type=str,
    )

    parser.add_argument(
        "--entity-linker-config",
        type=str,
        help="entity linker config as json or yaml",
    )

    args = parser.parse_args()

    main(args.json_fpath, args.entity_linker_config)
