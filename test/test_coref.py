import pytest
from typing import TYPE_CHECKING, cast

from triel.coref_adapter import (
    CorefBackend,
    CorefSetupError,
    get_coref_resolver,
    get_ready_coref_runtime,
)
from triel.coref import (
    render_coref_maps_wrapper,
    stitch_coreference,
    text_to_coref_classes,
)

if TYPE_CHECKING:
    from spacy import Language


def test_coref_maps(doc_coref):
    edges_chain_token, edges_chain_order = render_coref_maps_wrapper(doc_coref)
    assert len(edges_chain_order) == 1
    assert edges_chain_order[0] == (0, 2)
    assert len(edges_chain_token) == 13


def test_coref_maps_tokenindext(doc_coref, map_tree_subtree_index):
    initial_phrase_index = 0
    # edges_chain_token, edges_chain_order

    edges_chain_token, edges_chain_order = render_coref_maps_wrapper(doc_coref)

    edges_chain_tokenit = [
        (
            (initial_phrase_index, f"c_{k}"),
            tuple([map_tree_subtree_index[vv] for vv in v]),
        )
        for k, v in edges_chain_token
    ]

    edges_chaint_order = [
        ((initial_phrase_index, f"c_{a}"), (initial_phrase_index, f"c_{b}"))
        for a, b in edges_chain_order
    ]
    assert isinstance(edges_chain_tokenit[0][0], tuple)
    assert isinstance(edges_chain_tokenit[0][1], tuple)
    assert isinstance(edges_chain_tokenit[0][1][0], tuple)
    assert isinstance(edges_chaint_order[0][0], tuple)


def test_coref_classes(nlp_fixture, phrases_for_coref):
    initial_phrase_index = 0
    fragment = " ".join(phrases_for_coref)

    edges_chain_tokenit, edges_chaint_order = text_to_coref_classes(
        nlp_fixture, fragment, initial_phrase_index
    )

    assert isinstance(edges_chain_tokenit[0][0], tuple)
    assert isinstance(edges_chain_tokenit[0][1], tuple)
    assert isinstance(edges_chain_tokenit[0][1][0], tuple)
    assert isinstance(edges_chaint_order[0][0], tuple)
    assert max([y[0][0] for _, y in edges_chain_tokenit]) == 2


def test_stitching(nlp_fixture, phrases_for_coref, fig_path):
    edges_chain_token_global, edges_chain_order_global = stitch_coreference(
        nlp=nlp_fixture, phrases_for_coref=phrases_for_coref, window_size=2
    )

    assert len(edges_chain_token_global) == 13
    assert len(edges_chain_order_global) == 1


def test_coref_backend_none_returns_empty(nlp_fixture, phrases_for_coref):
    resolver = get_coref_resolver(CorefBackend.NONE)
    edges_chain_token_global, edges_chain_order_global = stitch_coreference(
        nlp=nlp_fixture,
        phrases_for_coref=phrases_for_coref,
        window_size=2,
        coref_resolver=resolver,
    )
    assert edges_chain_token_global == set()
    assert edges_chain_order_global == []


def test_get_ready_coref_runtime_fallback_to_none():
    class FakeNLP:
        pipe_names = []

        def add_pipe(self, _):
            raise RuntimeError("boom")

    nlp, resolver = get_ready_coref_runtime(
        cast("Language", FakeNLP()),
        CorefBackend.COREFEREE,
        fallback_to_none=True,
    )
    assert nlp is not None
    assert resolver.backend == CorefBackend.NONE


def test_get_ready_coref_runtime_raises_without_fallback():
    class FakeNLP:
        pipe_names = []

        def add_pipe(self, _):
            raise RuntimeError("boom")

    with pytest.raises(CorefSetupError):
        get_ready_coref_runtime(
            cast("Language", FakeNLP()),
            CorefBackend.COREFEREE,
            fallback_to_none=False,
        )
