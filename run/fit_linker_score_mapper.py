import logging
import sys
from importlib.resources import files

import click
import joblib
import numpy as np
import suthing

from lm_service.linking.onto import EntityLinker
from lm_service.linking.score import ScoreMapper

logger = logging.getLogger(__name__)


@click.command()
# @click.option(
#     "--model-dump-path", type=click.Path(path_type=pathlib.Path), default="./data/impact"
# )
def run():
    sample_data = suthing.FileHandle.load("data", "elinker.sample.csv", index_col=0)
    score_dict = sample_data.groupby("linker_type")["score"].apply(np.array).to_dict()
    score_mapper = ScoreMapper(
        score_dict,
        a=0.0,
        b=1.0001,
    )

    file_path = files("lm_service.models.store").joinpath("linker.scaling.model.gz")
    joblib.dump(score_mapper, file_path, compress=3)
    model0 = joblib.load(file_path)
    _ = model0(EntityLinker.PELINKER, score_dict[EntityLinker.PELINKER])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()
