import csv
import json
import pathlib
import time
from enum import StrEnum
from statistics import mean
from typing import Any

import click
import spacy
from pydantic import BaseModel, Field
from suthing import FileHandle

from triel.coref import stitch_coreference
from triel.coref_adapter import (
    CorefBackend,
    configure_nlp_coref_backend,
    get_coref_resolver,
)
from triel.linking.onto import EntityLinkerManager
from triel.text import normalize_text
from triel.top import text_to_graph_mentions_entities


class BenchmarkManifest(BaseModel):
    source_path: str
    selected_indices: list[int] = Field(default_factory=list)
    sample_size: int
    notes: str | None = None


class FailureKind(StrEnum):
    IMPORT = "import"
    SETUP = "setup"
    INFERENCE = "inference"
    OUTPUT_SHAPE = "output_shape"


class TrialSummary(BaseModel):
    avg_latency_s: float | None = None
    p95_latency_s: float | None = None
    coverage_rate: float = 0.0
    avg_triples: float = 0.0
    avg_chain_density: float = 0.0
    avg_mention_span_len: float = 0.0
    failure_counts: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    triples_per_text: list[int] = Field(default_factory=list)


def load_pruning_rules() -> dict[str, Any]:
    return FileHandle.load("triel.config", "prune_noun_compound_v3.yaml")


def load_texts(input_path: pathlib.Path) -> list[str]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        texts: list[str] = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        return texts

    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if all(isinstance(x, str) for x in payload):
                return [x for x in payload if x.strip()]
            return [
                item["text"]
                for item in payload
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and item["text"].strip()
            ]
        if isinstance(payload, dict) and isinstance(payload.get("texts"), list):
            return [x for x in payload["texts"] if isinstance(x, str) and x.strip()]
        raise ValueError("JSON input must be list[str], list[{text}], or {texts:[str]}")

    return [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_stratified_indices(texts: list[str], sample_size: int) -> list[int]:
    if sample_size >= len(texts):
        return list(range(len(texts)))

    short = [i for i, txt in enumerate(texts) if len(txt) <= 160]
    medium = [i for i, txt in enumerate(texts) if 160 < len(txt) <= 420]
    long_ = [i for i, txt in enumerate(texts) if len(txt) > 420]
    strata = [short, medium, long_]

    per_bucket = max(1, sample_size // 3)
    selected: list[int] = []
    for bucket in strata:
        selected.extend(bucket[:per_bucket])
    if len(selected) < sample_size:
        all_remaining = [i for i in range(len(texts)) if i not in set(selected)]
        selected.extend(all_remaining[: sample_size - len(selected)])
    return sorted(selected[:sample_size])


def analyze_coref_quality(
    phrases: list[str], edges_chain_token_global
) -> tuple[float, float]:
    if not edges_chain_token_global:
        return 0.0, 0.0
    mentions_per_chain: dict[Any, list[tuple[int, ...]]] = {}
    mention_lengths: list[int] = []
    for chain, mention in edges_chain_token_global:
        mentions_per_chain.setdefault(chain, []).append(mention)
        mention_lengths.append(len(mention))
    chain_density = (
        (sum(len(v) for v in mentions_per_chain.values()) / len(mentions_per_chain))
        if mentions_per_chain
        else 0.0
    )
    avg_mention_span_len = mean(mention_lengths) if mention_lengths else 0.0
    return float(chain_density), float(avg_mention_span_len)


def classify_failure(exc: Exception) -> FailureKind:
    msg = str(exc).lower()
    if "no module named" in msg or "cannot import" in msg:
        return FailureKind.IMPORT
    if "add_pipe" in msg or "factory" in msg or "not in pipeline" in msg:
        return FailureKind.SETUP
    if "shape" in msg or "tuple" in msg or "span" in msg:
        return FailureKind.OUTPUT_SHAPE
    return FailureKind.INFERENCE


def benchmark_backend_trial(
    texts: list[str], model_name: str, backend: CorefBackend, rules: dict[str, Any]
) -> TrialSummary:
    try:
        nlp = spacy.load(model_name)
        nlp = configure_nlp_coref_backend(nlp, backend)
        resolver = get_coref_resolver(backend)
    except Exception as e:
        return TrialSummary(
            failure_counts={FailureKind.SETUP.value: len(texts)},
            errors=[str(e)],
            triples_per_text=[0 for _ in texts],
        )

    elm = EntityLinkerManager({})
    latencies: list[float] = []
    coverage_flags: list[bool] = []
    triples_per_text: list[int] = []
    chain_densities: list[float] = []
    mention_spans: list[float] = []
    errors: list[str] = []
    failure_counts: dict[str, int] = {kind.value: 0 for kind in FailureKind}

    for text in texts:
        try:
            t0 = time.perf_counter()
            response = text_to_graph_mentions_entities(
                text=text,
                nlp=nlp,
                rules=rules,
                elm=elm,
                coref_resolver=resolver,
            )
            latencies.append(time.perf_counter() - t0)
            triples_per_text.append(len(response.triples))

            phrases = normalize_text(text, nlp)
            edges_chain_token_global, _ = stitch_coreference(
                phrases_for_coref=phrases,
                nlp=nlp,
                window_size=2,
                coref_resolver=resolver,
            )
            coverage_flags.append(bool(edges_chain_token_global))
            chain_density, avg_mention_span_len = analyze_coref_quality(
                phrases, edges_chain_token_global
            )
            chain_densities.append(chain_density)
            mention_spans.append(avg_mention_span_len)
        except Exception as e:
            fk = classify_failure(e).value
            failure_counts[fk] = failure_counts.get(fk, 0) + 1
            errors.append(str(e))
            latencies.append(float("nan"))
            triples_per_text.append(0)
            coverage_flags.append(False)
            chain_densities.append(0.0)
            mention_spans.append(0.0)

    valid_latencies = [x for x in latencies if x == x]
    return TrialSummary(
        avg_latency_s=mean(valid_latencies) if valid_latencies else None,
        p95_latency_s=(
            sorted(valid_latencies)[int(0.95 * (len(valid_latencies) - 1))]
            if valid_latencies
            else None
        ),
        coverage_rate=(
            (sum(1 for x in coverage_flags if x) / len(coverage_flags))
            if coverage_flags
            else 0.0
        ),
        avg_triples=mean(triples_per_text) if triples_per_text else 0.0,
        avg_chain_density=mean(chain_densities) if chain_densities else 0.0,
        avg_mention_span_len=mean(mention_spans) if mention_spans else 0.0,
        failure_counts=failure_counts,
        errors=errors[:10],
        triples_per_text=triples_per_text,
    )


def benchmark_backend(
    texts: list[str],
    model_name: str,
    backend: CorefBackend,
    rules: dict[str, Any],
) -> dict[str, Any]:
    raise NotImplementedError("Use benchmark_backend_with_trials")


def benchmark_backend_with_trials(
    texts: list[str],
    model_name: str,
    backend: CorefBackend,
    rules: dict[str, Any],
    trials: int,
) -> dict[str, Any]:
    trial_summaries = [
        benchmark_backend_trial(texts, model_name, backend, rules)
        for _ in range(trials)
    ]
    avg_latency = [
        x.avg_latency_s for x in trial_summaries if x.avg_latency_s is not None
    ]
    p95_latency = [
        x.p95_latency_s for x in trial_summaries if x.p95_latency_s is not None
    ]
    coverage = [x.coverage_rate for x in trial_summaries]
    avg_triples = [x.avg_triples for x in trial_summaries]
    chain_density = [x.avg_chain_density for x in trial_summaries]
    mention_span = [x.avg_mention_span_len for x in trial_summaries]

    failure_counts: dict[str, int] = {kind.value: 0 for kind in FailureKind}
    errors: list[str] = []
    for summary in trial_summaries:
        for key, value in summary.failure_counts.items():
            failure_counts[key] = failure_counts.get(key, 0) + value
        errors.extend(summary.errors)

    triples_per_text = trial_summaries[-1].triples_per_text if trial_summaries else []
    return {
        "backend": backend.value,
        "sample_size": len(texts),
        "trials": trials,
        "avg_latency_s": mean(avg_latency) if avg_latency else None,
        "p95_latency_s": mean(p95_latency) if p95_latency else None,
        "coverage_rate": mean(coverage) if coverage else 0.0,
        "avg_triples": mean(avg_triples) if avg_triples else 0.0,
        "avg_chain_density": mean(chain_density) if chain_density else 0.0,
        "avg_mention_span_len": mean(mention_span) if mention_span else 0.0,
        "failure_counts": failure_counts,
        "errors": errors[:10],
        "triples_per_text": triples_per_text,
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input-path",
    type=click.Path(path_type=pathlib.Path, exists=True),
    required=True,
    help="Path to input corpus (.txt, .json, or .jsonl).",
)
@click.option(
    "--sample-size",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Number of texts to benchmark (recommendation: 50-100).",
)
@click.option(
    "--trials",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Number of repeated trials per backend.",
)
@click.option(
    "--stratified-sampling/--no-stratified-sampling",
    default=True,
    show_default=True,
    help="Use stratified sampling by text length buckets.",
)
@click.option(
    "--model",
    type=str,
    default="en_core_web_trf",
    show_default=True,
    help="spaCy model name.",
)
@click.option(
    "--backend",
    "backends",
    type=click.Choice([item.value for item in CorefBackend]),
    multiple=True,
    default=(
        # CorefBackend.COREFEREE.value,
        CorefBackend.FASTCOREF.value,
    ),
    show_default=True,
    help="Coreference backend(s) to benchmark.",
)
@click.option(
    "--output-json",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("coref_benchmark_results.json"),
    show_default=True,
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("coref_benchmark_results.csv"),
    show_default=True,
)
@click.option(
    "--manifest-path",
    type=click.Path(path_type=pathlib.Path, exists=True),
    default=None,
    help="Path to fixed benchmark manifest with selected indices.",
)
@click.option(
    "--write-manifest",
    type=click.Path(path_type=pathlib.Path),
    default=None,
    help="Optional path to save the selected subset manifest.",
)
@click.option(
    "--min-coverage-rate",
    type=float,
    default=0.6,
    show_default=True,
    help="Promotion gate: minimum coverage rate.",
)
@click.option(
    "--max-latency-ratio-vs-coreferee",
    type=float,
    default=1.8,
    show_default=True,
    help="Promotion gate: max avg latency ratio vs coreferee.",
)
@click.option(
    "--max-triple-drift-vs-coreferee",
    type=float,
    default=3.0,
    show_default=True,
    help="Promotion gate: max average triple-count drift vs coreferee.",
)
def main(
    input_path: pathlib.Path,
    sample_size: int,
    trials: int,
    stratified_sampling: bool,
    model: str,
    backends: tuple[str, ...],
    output_json: pathlib.Path,
    output_csv: pathlib.Path,
    manifest_path: pathlib.Path | None,
    write_manifest: pathlib.Path | None,
    min_coverage_rate: float,
    max_latency_ratio_vs_coreferee: float,
    max_triple_drift_vs_coreferee: float,
) -> None:
    texts = load_texts(input_path)
    if not texts:
        raise ValueError(f"No texts found in {input_path}")

    if manifest_path is not None:
        manifest = BenchmarkManifest(
            **json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        source_path = pathlib.Path(manifest.source_path)
        source_texts = load_texts(source_path)
        selected_indices = manifest.selected_indices[: manifest.sample_size]
        texts = [source_texts[i] for i in selected_indices if i < len(source_texts)]
        sample_size = len(texts)
    else:
        if sample_size > len(texts):
            sample_size = len(texts)
        selected_indices = (
            build_stratified_indices(texts, sample_size)
            if stratified_sampling
            else list(range(sample_size))
        )
        texts = [texts[i] for i in selected_indices]
        if write_manifest is not None:
            manifest = BenchmarkManifest(
                source_path=input_path.as_posix(),
                selected_indices=selected_indices,
                sample_size=len(selected_indices),
                notes="Generated by benchmark_coref_backends.py",
            )
            write_manifest.write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )

    rules = load_pruning_rules()
    baseline_backend = CorefBackend.COREFEREE
    if baseline_backend.value not in backends:
        bench_backends = (baseline_backend.value, *backends)
    else:
        bench_backends = backends

    results = []
    for backend_name in bench_backends:
        backend = CorefBackend(backend_name)
        click.echo(
            f"Running backend={backend.value} on {len(texts)} texts across {trials} trial(s)..."
        )
        try:
            result = benchmark_backend_with_trials(
                texts=texts,
                model_name=model,
                backend=backend,
                rules=rules,
                trials=trials,
            )
        except Exception as e:
            result = {
                "backend": backend.value,
                "sample_size": len(texts),
                "trials": trials,
                "avg_latency_s": None,
                "p95_latency_s": None,
                "coverage_rate": 0.0,
                "avg_triples": 0.0,
                "avg_chain_density": 0.0,
                "avg_mention_span_len": 0.0,
                "failure_counts": {FailureKind.INFERENCE.value: len(texts) * trials},
                "errors": [str(e)],
                "triples_per_text": [0 for _ in texts],
            }
        results.append(result)

    baseline = next(
        (r for r in results if r["backend"] == baseline_backend.value), None
    )
    if baseline is not None:
        baseline_triples = baseline["triples_per_text"]
        baseline_latency = baseline["avg_latency_s"]
        for result in results:
            current = result["triples_per_text"]
            if len(current) == len(baseline_triples):
                drift = [abs(a - b) for a, b in zip(current, baseline_triples)]
                result["avg_triple_count_drift_vs_coreferee"] = (
                    mean(drift) if drift else 0.0
                )
            else:
                result["avg_triple_count_drift_vs_coreferee"] = None
            if baseline_latency and result["avg_latency_s"]:
                result["latency_ratio_vs_coreferee"] = (
                    result["avg_latency_s"] / baseline_latency
                )
            else:
                result["latency_ratio_vs_coreferee"] = None
    else:
        for result in results:
            result["avg_triple_count_drift_vs_coreferee"] = None
            result["latency_ratio_vs_coreferee"] = None

    gates = []
    for result in results:
        if result["backend"] == CorefBackend.COREFEREE.value:
            continue
        gate_pass = (
            (result["coverage_rate"] >= min_coverage_rate)
            and (
                result["latency_ratio_vs_coreferee"] is None
                or result["latency_ratio_vs_coreferee"]
                <= max_latency_ratio_vs_coreferee
            )
            and (
                result["avg_triple_count_drift_vs_coreferee"] is None
                or result["avg_triple_count_drift_vs_coreferee"]
                <= max_triple_drift_vs_coreferee
            )
        )
        gates.append(
            {
                "backend": result["backend"],
                "pass": gate_pass,
                "coverage_rate": result["coverage_rate"],
                "latency_ratio_vs_coreferee": result["latency_ratio_vs_coreferee"],
                "avg_triple_count_drift_vs_coreferee": result[
                    "avg_triple_count_drift_vs_coreferee"
                ],
            }
        )

    payload = {
        "results": results,
        "promotion_gates": {
            "min_coverage_rate": min_coverage_rate,
            "max_latency_ratio_vs_coreferee": max_latency_ratio_vs_coreferee,
            "max_triple_drift_vs_coreferee": max_triple_drift_vs_coreferee,
            "evaluation": gates,
        },
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "sample_size",
                "trials",
                "avg_latency_s",
                "p95_latency_s",
                "coverage_rate",
                "avg_triples",
                "avg_chain_density",
                "avg_mention_span_len",
                "latency_ratio_vs_coreferee",
                "avg_triple_count_drift_vs_coreferee",
                "failure_counts",
                "errors",
            ],
        )
        writer.writeheader()
        for item in payload["results"]:
            writer.writerow(
                {
                    "backend": item["backend"],
                    "sample_size": item["sample_size"],
                    "trials": item["trials"],
                    "avg_latency_s": item["avg_latency_s"],
                    "p95_latency_s": item["p95_latency_s"],
                    "coverage_rate": item["coverage_rate"],
                    "avg_triples": item["avg_triples"],
                    "avg_chain_density": item["avg_chain_density"],
                    "avg_mention_span_len": item["avg_mention_span_len"],
                    "latency_ratio_vs_coreferee": item["latency_ratio_vs_coreferee"],
                    "avg_triple_count_drift_vs_coreferee": item[
                        "avg_triple_count_drift_vs_coreferee"
                    ],
                    "failure_counts": json.dumps(item["failure_counts"]),
                    "errors": " | ".join(item["errors"]),
                }
            )

    click.echo(f"Wrote JSON results to {output_json}")
    click.echo(f"Wrote CSV summary to {output_csv}")


if __name__ == "__main__":
    main()
