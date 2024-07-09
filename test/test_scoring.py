import pytest

from lm_service.linking.onto import EntityLinker
from lm_service.linking.score import BoundedCubicSpline, ScoreMapper


def test_bcs(bern_score):
    bsc = BoundedCubicSpline(bern_score)
    assert pytest.approx(bsc.predict(0.6), abs=0.05) == 0.01
    assert pytest.approx(bsc.predict(0.9), rel=1e-2) == 0.5


def test_score_mapper(bern_score):
    score_mapper = ScoreMapper(
        {EntityLinker.BERN_V2: bern_score, EntityLinker.PELINKER: 0.3 * bern_score}
    )
    assert pytest.approx(score_mapper(EntityLinker.BERN_V2, 0.6), abs=0.05) == 0.01
    assert pytest.approx(score_mapper(EntityLinker.PELINKER, 0.6), abs=0.05) == 1.0
