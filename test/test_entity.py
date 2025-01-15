import logging

import pytest

from triel.linking.onto import EntityLinkerManager

logger = logging.getLogger(__name__)


def test_entities(entities):
    ent_a = entities[0]
    ref_a = {
        "linker_type": "BERN_V2",
        "ent_db_type": "NA",
        "id": "cell_type:tams",
        "hash": "BERN_V2.NA.cell_type:tams",
        "ent_type": "cell_type",
        "a": 0,
        "b": 4,
    }
    ent_a_dict = ent_a.to_dict()
    assert pytest.approx(ent_a_dict.pop("score"), abs=0.1) == 0.9
    assert ent_a_dict == ref_a


def test_bern_normalization(bern_example, caplog):
    caplog.set_level(logging.INFO)
    epack = [
        EntityLinkerManager._normalize_bern_entity(item, prob_thr=0.8)
        for item in bern_example["annotations"]
    ]
    assert epack[0].id == "cell_type:tams"
    assert epack[-2] is None
    assert epack[4].id == "cell_type:cd8_t_cell"
    dumped_entity = epack[0].to_dict()
    assert "ent_type" in dumped_entity
    assert dumped_entity["linker_type"] == "BERN_V2"
    assert dumped_entity["a"] == 0


def test_pelinker_normalization(pelinker_example, caplog):
    caplog.set_level(logging.INFO)
    epack = [
        EntityLinkerManager._normalize_pelinker_entity(item, prob_thr=0.8)
        for item in pelinker_example["entities"]
    ]
    assert epack[0].original_form == "suppress"
