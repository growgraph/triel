import pytest

from lm_service.linking.score import BoundedCubicSpline


def test_bcs(bern_score):
    bsc = BoundedCubicSpline(bern_score)
    assert pytest.approx(bsc.predict(0.6), abs=0.05) == 0.01
    assert pytest.approx(bsc.predict(0.9), rel=1e-2) == 0.5
