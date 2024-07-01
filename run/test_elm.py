import logging
import logging.config
import pathlib

import click
import pandas as pd
from suthing import FileHandle

from lm_service.linking.onto import EntityLinkerManager
from lm_service.linking.util import map_linkers

logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", type=click.STRING, default="localhost")
@click.option("--input-path", type=click.Path(path_type=pathlib.Path))
@click.option("--conf-el-path", type=click.Path(path_type=pathlib.Path))
@click.option("--output", type=click.Path(path_type=pathlib.Path), required=False)
@click.option("--debug", is_flag=True, default=False, help="logging at debug level")
def run(host, conf_el_path, input_path, output, debug):
    debug_option = ".debug" if debug else ""
    logger_conf = f"logging{debug_option}.conf"
    logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
    logger.debug("debug is on")

    el_conf = FileHandle.load(fpath=conf_el_path)
    for c in el_conf["linkers"]:
        c["host"] = host
        c["threshold"] = 0.0

    elm = EntityLinkerManager.from_dict(el_conf)

    inp = FileHandle.load(fpath=input_path)
    if not isinstance(inp, list):
        inp_list = [inp["text"]]
    else:
        inp_list = inp

    responses = map_linkers(entity_linker_manager=elm, phrases=inp_list)

    entity_pack = []
    for link_mode, r in zip(elm.linker_types, responses):
        epack = elm.normalize(
            r,
            link_mode,
        )
        entity_pack.extend(epack)

    if output:
        df = pd.DataFrame([item.to_dict() for item in entity_pack])
        FileHandle.dump(df, output.as_posix())


if __name__ == "__main__":
    run()
