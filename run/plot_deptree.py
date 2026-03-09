"""Generate dependency and coreference graphs for sample texts.

This script is meant as a small TriEL demo for collaborators:
- it parses one or more texts with spaCy + coreferee
- writes dependency-tree PDFs
- writes matching coreference-chain PDFs
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import click
import spacy
from spacy.language import Language

from triel.coref import render_coref_graph, text_to_compound_index_graph
from triel.util import plot_graph

DEFAULT_MODEL = "en_core_web_trf"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / ".figs"
EXAMPLE_TEXTS = [
    (
        "Although he was very busy with his work, Peter Brown had had enough of it. "
        "He and his wife decided they needed a holiday. They travelled to Spain "
        "because they loved the country very much."
    ),
    (
        "CHEOPS (CHaracterising ExOPlanets Satellite) is a European space telescope "
        "to determine the size of known extrasolar planets, which will allow the "
        "estimation of their mass, density, composition and their formation. "
        "Launched on 18 December 2019, it is the first Small-class mission in ESA's "
        "Cosmic Vision science programme."
    ),
    (
        "TAMs can also secrete in the TME a number of immunosuppressive cytokines, "
        "such as IL-6, TGF-beta, and IL-10 that are able to suppress CD8+ T-cell "
        "function (76)."
    ),
]


def build_nlp(model_name: str) -> Language:
    nlp = spacy.load(model_name)
    if "coreferee" not in nlp.pipe_names:
        nlp.add_pipe("coreferee")
    return nlp


def short_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def plot_text_graphs(text: str, nlp: Language, output_dir: Path) -> tuple[Path, Path]:
    graph_name = short_text_hash(text)
    dep_graph, rdoc, _ = text_to_compound_index_graph(nlp, text, 0)

    plot_graph(dep_graph, str(output_dir), graph_name)
    plot_graph(render_coref_graph(rdoc), str(output_dir), f"{graph_name}_coref")
    return output_dir / f"{graph_name}.pdf", output_dir / f"{graph_name}_coref.pdf"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--text",
    "texts",
    multiple=True,
    help="Text to parse. Can be passed multiple times. Uses built-in examples if omitted.",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help="spaCy model to load.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory for generated figures.",
)
def main(texts: tuple[str, ...], model: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_texts = list(texts) if texts else EXAMPLE_TEXTS
    nlp = build_nlp(model)

    click.echo(
        f"Generating graphs for {len(selected_texts)} text(s) into: {output_dir}"
    )
    for index, text in enumerate(selected_texts, start=1):
        dep_path, coref_path = plot_text_graphs(text, nlp, output_dir)
        click.echo(
            f"[{index}] dependency: {dep_path.name} | coreference: {coref_path.name}"
        )


if __name__ == "__main__":
    main()
