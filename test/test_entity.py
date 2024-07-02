import logging

from lm_service.linking.onto import Entity, EntityLinkerManager

logger = logging.getLogger(__name__)


def test_entities(entities):
    ents = [Entity.from_dict(item) for item in entities]
    ent_a = ents[0]
    ent_b = ents[1]
    ent_b.description = "something"
    assert ent_a.to_dict() == {
        "linker_type": "BERN_V2",
        "ent_db_type": "mesh",
        "id": "D017719",
        "hash": "BERN_V2.mesh.D017719",
        "ent_type": "disease",
    }
    assert ent_b.to_dict() == {
        "linker_type": "BERN_V2",
        "ent_db_type": "mesh",
        "id": "D002056",
        "hash": "BERN_V2.mesh.D002056",
        "ent_type": "disease",
        "description": "something",
    }


def test_bern_normalization(bern_example, caplog):
    caplog.set_level(logging.INFO)
    epack = [
        EntityLinkerManager._normalize_bern_entity(item, prob_thr=0.8)
        for item in bern_example["annotations"]
    ]
    assert epack[0].id == "cell_type:tams"
    assert epack[-2] is None
    assert epack[4].id == "cell_type:cd8+_t-cell"
    dumped_entity = epack[0].to_dict()
    assert dumped_entity["a"] == 0


def test_pelinker_normalization(pelinker_example, caplog):
    caplog.set_level(logging.INFO)
    epack = [
        EntityLinkerManager._normalize_pelinker_entity(item, prob_thr=0.8)
        for item in pelinker_example["entities"]
    ]
    assert epack[0].original_form == "suppress"
