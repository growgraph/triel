"""Profile TriEL's top-level extraction pipeline on sample text.

This script is designed for quick demos:
- load pruning rules and NLP pipeline
- run mention/entity extraction
- print a compact, readable response
- print profiling stats from `suthing.SProfiler`
"""

from __future__ import annotations

import json
import pkgutil
from typing import Any

import click
import spacy
import yaml
from spacy.language import Language
from suthing import SProfiler

from triel.coref_adapter import (
    CorefBackend,
    configure_nlp_coref_backend,
    get_coref_resolver,
)
from triel.linking.onto import EntityLinkerManager
from triel.top import cast_response_redux, text_to_graph_mentions_entities

DEFAULT_MODEL = "en_core_web_trf"
DEFAULT_TEXT = "Diabetic ulcers are related to burns."
PRUNING_RULES_RESOURCE = "prune_noun_compound_v2.yaml"


def load_pruning_rules() -> dict[str, Any]:
    raw_rules = pkgutil.get_data("triel.config", PRUNING_RULES_RESOURCE)
    if raw_rules is None:
        raise RuntimeError(
            f"Could not load resource: triel.config/{PRUNING_RULES_RESOURCE}"
        )
    return yaml.safe_load(raw_rules) or {}


def build_nlp(
    model_name: str, coref_backend: CorefBackend = CorefBackend.COREFEREE
) -> Language:
    nlp = spacy.load(model_name)
    return configure_nlp_coref_backend(nlp, coref_backend)


def run_profile(
    text: str,
    model_name: str,
    coref_backend: CorefBackend = CorefBackend.COREFEREE,
) -> tuple[dict[str, Any], Any]:
    rules = load_pruning_rules()
    nlp = build_nlp(model_name, coref_backend=coref_backend)
    coref_resolver = get_coref_resolver(coref_backend)
    entity_linker_manager = EntityLinkerManager({})
    profiler = SProfiler()

    response = text_to_graph_mentions_entities(
        text,
        nlp,
        rules,
        entity_linker_manager,
        coref_resolver=coref_resolver,
        _profiler=profiler,
    )
    response_redux = cast_response_redux(response).model_dump()
    return response_redux, profiler.view_stats()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--text",
    default=DEFAULT_TEXT,
    show_default=True,
    help="Input text to analyze.",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="spaCy model to load.",
)
@click.option(
    "--coref-backend",
    default=CorefBackend.COREFEREE.value,
    show_default=True,
    type=click.Choice([item.value for item in CorefBackend]),
    help="Coreference backend to use.",
)
def main(text: str, model: str, coref_backend: str) -> None:
    response_redux, stats = run_profile(
        text, model, coref_backend=CorefBackend(coref_backend)
    )

    click.echo("== TriEL response (redux) ==")
    click.echo(json.dumps(response_redux, indent=2, default=str))
    click.echo("\n== Profiler stats ==")
    click.echo(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
