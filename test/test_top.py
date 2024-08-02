import pytest
from suthing import FileHandle

from lm_service.linking.onto import EntityLinkerManager
from lm_service.response.onto import (
    REELResponse,
    REELResponseEntity,
    REELResponseRedux,
)
from lm_service.top import (
    cast_response_entity_representation,
    cast_response_redux,
    text_to_graph_mentions_entities,
)


@pytest.fixture
def reel_response():
    return REELResponse.from_dict(FileHandle.load("test.data", "reel_response.json"))


def test_cast_response_redux(reel_response):
    r = cast_response_redux(reel_response)
    assert isinstance(r, REELResponseRedux)
    assert len(r.triples) == 79
    assert len(r.map_mention_entity) == 147
    assert len(r.top_level_mention) == 79


def test_cast_response_er(reel_response):
    r = cast_response_entity_representation(reel_response)
    assert isinstance(r, REELResponseEntity)
    assert len(r.triples) == 171
    assert len(r.entities) == 79


def test_complete(nlp_fixture, rules, el_conf, sample_a):
    elm = EntityLinkerManager.from_dict(el_conf)

    response = text_to_graph_mentions_entities(
        sample_a["text"], nlp_fixture, rules, elm
    )

    _ = cast_response_redux(response)
    response_ent = cast_response_entity_representation(response)
    assert len(response_ent.entities) == 53
    assert len(response_ent.triples) == 68
