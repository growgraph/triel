import logging
import pathlib
from pprint import pprint

import click
import spacy
from suthing import FileHandle

from lm_service.linking.onto import EntityLinkerManager
from lm_service.top import cast_response_to_unfolded, text_to_rel_graph

logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", type=click.STRING, default="localhost")
@click.option("--input-path", type=click.Path(path_type=pathlib.Path))
@click.option("--conf-el-path", type=click.Path(path_type=pathlib.Path))
@click.option("--output", type=click.Path(path_type=pathlib.Path), required=False)
def run(host, conf_el_path, input_path, output):
    el_conf = FileHandle.load(fpath=conf_el_path)
    for c in el_conf["linkers"]:
        c["host"] = host

    elm = EntityLinkerManager.from_dict(el_conf)
    rules = FileHandle.load("lm_service.config", "prune_noun_compound_v2.yaml")

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("coreferee")

    logger.info("nlp loaded")

    inp = FileHandle.load(fpath=input_path)
    if not isinstance(inp, list):
        inp_list = [inp]
    else:
        inp_list = inp

    acc = []
    for s in inp_list:
        response = text_to_rel_graph(s["text"], nlp, rules, elm)
        response_jsonlike = cast_response_to_unfolded(
            response, cast_triple_version="v1"
        )
        pprint(response_jsonlike)
        acc += [response_jsonlike]

    if output:
        FileHandle.dump(acc, output.as_posix())


if __name__ == "__main__":
    run()
