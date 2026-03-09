import logging

from triel.folding import get_flag
from triel.onto import Token

logger = logging.getLogger(__name__)


def test_flag(rules_v2):
    t = Token(s=7, text="his", dep_="conj", tag_="VBG")
    flag = get_flag(t.__dict__, rules_v2["source_target"]["secondary"])
    assert flag


def test_flag_secondary(rules_v3):
    t = Token(s=7, text="supress", dep_="xcomp", tag_="VB")
    flag = get_flag(t.__dict__, rules_v3["relation"]["secondary"])
    assert flag
