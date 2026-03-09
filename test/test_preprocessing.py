import logging

from triel.preprocessing import normalize_input_text

logger = logging.getLogger(__name__)


def test_consecutive_candidates(phrase):
    out = normalize_input_text(phrase)
    assert out == [
        (
            "This corresponds to the transit of an Earth-sized planet"
            " orbiting a star of 0.9 R in 60 days detected with a"
            " S/Ntransit >10 (100 ppm transit depth)."
        ),
        ("For example, an Earth-size transit across a G star creates an 80 ppm depth."),
        ("The different science objectives require 500 separate target pointings."),
        (
            "Assuming 1 hour per pointing the mission duration is"
            " estimated at 1175 days or 3.2 years."
        ),
    ]


def test_dash(phrase_b):
    out = normalize_input_text(phrase_b)
    assert out == [
        "Launched on 18 December 2019, it is the first Small-class"
        " mission in ESA's Cosmic Vision science programme."
    ]


def test_newline(phrase_c):
    out = normalize_input_text(phrase_c)
    assert len(out) == 13


def test_percent(phrase_split_percent):
    out = normalize_input_text(phrase_split_percent)
    assert [len(x) for x in out] == [185, 140]
