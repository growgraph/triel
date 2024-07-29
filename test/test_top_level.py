import pytest
from suthing import FileHandle

from lm_service.response.onto import (  # REELResponseEntity,
    REELResponse,
    REELResponseRedux,
)
from lm_service.top import (  # cast_response_entity_representation,
    cast_response_redux,
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


# def test_cast_response_er(reel_response):
#     r = cast_response_entity_representation(reel_response)
#     assert isinstance(r, REELResponseEntity)
#     # assert len(r.triples) == 79
#     # assert len(r.entities) == 147
