import pytest
from suthing import FileHandle

from triel.coref_adapter import CorefBackend, get_coref_resolver
from triel.linking.onto import EntityLinkerManager
from triel.response.onto import (
    REELResponse,
    REELResponseEntity,
    REELResponseRedux,
)
from triel.top import (
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


def test_complete(nlp_fixture, rules_v2, el_conf, sample_a, coref_resolver_fixture):
    elm = EntityLinkerManager.from_dict(el_conf)

    response = text_to_graph_mentions_entities(
        sample_a["text"],
        nlp_fixture,
        rules_v2,
        elm,
        coref_resolver=coref_resolver_fixture,
    )

    _ = cast_response_redux(response)
    response_ent = cast_response_entity_representation(response)
    assert len(response_ent.entities) == 45
    assert len(response_ent.triples) == 36


def test_dual_run_logs_without_changing_response(monkeypatch, caplog):
    monkeypatch.setattr("triel.top.normalize_text", lambda text, nlp: ["Sample text."])
    monkeypatch.setattr(
        "triel.top.phrases_to_triples", lambda *args, **kwargs: ({}, {})
    )
    monkeypatch.setattr("triel.top.iterate_over_linkers", lambda **kwargs: [])
    monkeypatch.setattr(
        "triel.top.map_mentions_to_entities", lambda *args, **kwargs: ({}, [], [])
    )
    monkeypatch.setattr(
        "triel.top.link_unlinked_entities", lambda *args, **kwargs: ({}, [])
    )

    call_count = {"n": 0}

    def _stitch_mock(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {("c0", ((0, 0),))}, []
        return {("s0", ((0, 0),)), ("s1", ((1, 0),))}, []

    monkeypatch.setattr("triel.top.stitch_coreference", _stitch_mock)

    primary = get_coref_resolver(CorefBackend.NONE)
    shadow = get_coref_resolver(CorefBackend.NONE)
    caplog.set_level("INFO")

    response = text_to_graph_mentions_entities(
        text="Sample text.",
        nlp=object(),
        rules={},
        elm=object(),
        coref_resolver=primary,
        coref_shadow_resolver=shadow,
        coref_dual_run_enabled=True,
    )
    assert isinstance(response, REELResponse)
    assert any("coref_dual_run" in rec.message for rec in caplog.records)
