import logging
import pathlib
from pprint import pprint

import click
import spacy
from suthing import FileHandle

from lm_service.linking.onto import EntityLinkerManager
from lm_service.top import (
    cast_response_entity_representation,
    cast_response_redux,
    text_to_graph_mentions_entities,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", type=click.STRING, default="localhost")
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), multiple=True)
@click.option("--conf-el-path", type=click.Path(path_type=pathlib.Path))
@click.option(
    "--output", type=click.Path(path_type=pathlib.Path), required=False, default=None
)
@click.option("--phrase-indexes", "-i", type=click.INT, multiple=True)
@click.option("--localhost-linkers", type=click.STRING, multiple=True)
def run(host, conf_el_path, input_path, output, phrase_indexes, localhost_linkers):
    el_conf = FileHandle.load(fpath=conf_el_path)
    for c in el_conf["linkers"]:
        if "host" not in c:
            c["host"] = host
        if c["keyword"] in localhost_linkers:
            c["host"] = "localhost"

    elm = EntityLinkerManager.from_dict(el_conf)
    rules = FileHandle.load("lm_service.config", "prune_noun_compound_v2.yaml")

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    logger.info("nlp loaded")

    inputs = [
        (
            ".".join(ipath.as_posix().split("/")[-1].split(".")[:-1]),
            FileHandle.load(fpath=ipath),
        )
        for ipath in input_path
    ]

    for name, data in inputs:
        response = text_to_graph_mentions_entities(
            data["text"], nlp, rules, elm, ix_phrases=phrase_indexes
        )

        response_redux = cast_response_redux(response)
        response_jsonlike = response_redux.to_dict()
        pprint(response_jsonlike)
        if output is not None:
            FileHandle.dump(
                response_jsonlike, (output / f"{name}.kg.detailed.json").as_posix()
            )

        response_ent = cast_response_entity_representation(response)

        if output is not None:
            response_jsonlike = response_ent.to_dict()
            pprint(response_jsonlike)

            FileHandle.dump(
                response_jsonlike,
                (output.expanduser() / f"{name}.kg.entities.json").as_posix(),
            )


if __name__ == "__main__":
    run()
