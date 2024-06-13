import pathlib
from pprint import pprint

import click
import spacy
from suthing import FileHandle

from lm_service.linking import EntityLinkerManager
from lm_service.top import cast_response_to_unfolded, text_to_rel_graph


@click.command()
@click.option(
    "--entity-linker-config",
    type=click.Path(path_type=pathlib.Path),
    help="entity linker config as json or yaml",
)
@click.option("--sample-path", type=click.Path(path_type=pathlib.Path), multiple=True)
def main(sample_path: list[pathlib.Path], entity_linker_config):
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    rules = FileHandle.load("lm_service.config", "prune_noun_compound_v2.yaml")

    entity_linker_config_ = FileHandle.load(entity_linker_config)
    elm = EntityLinkerManager(entity_linker_config_)
    samples = [(sp.name, FileHandle.load(fpath=sp)) for sp in sample_path]

    for fname, s in samples:
        text = s["text"]
        response = text_to_rel_graph(text, nlp, rules, elm)
        response_jsonlike = cast_response_to_unfolded(
            response, cast_triple_version="v1"
        )
        pprint(response_jsonlike)


if __name__ == "__main__":
    main()
