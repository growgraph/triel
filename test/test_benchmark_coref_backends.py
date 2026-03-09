from __future__ import annotations

import json

from click.testing import CliRunner

from run.analysis import benchmark_coref_backends as bench


def test_benchmark_cli_help():
    runner = CliRunner()
    result = runner.invoke(bench.main, ["--help"])
    assert result.exit_code == 0
    assert "sample-size" in result.output
    assert "trials" in result.output
    assert "manifest-path" in result.output


def test_benchmark_output_schema(tmp_path, monkeypatch):
    input_path = tmp_path / "texts.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "One sentence."}),
                json.dumps({"text": "Another sentence."}),
            ]
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "out.json"
    output_csv = tmp_path / "out.csv"

    monkeypatch.setattr(bench, "load_pruning_rules", lambda: {})
    monkeypatch.setattr(
        bench,
        "benchmark_backend_with_trials",
        lambda **kwargs: {
            "backend": kwargs["backend"].value,
            "sample_size": len(kwargs["texts"]),
            "trials": kwargs["trials"],
            "avg_latency_s": 0.1,
            "p95_latency_s": 0.2,
            "coverage_rate": 1.0,
            "avg_triples": 2.0,
            "avg_chain_density": 1.5,
            "avg_mention_span_len": 1.2,
            "failure_counts": {
                "import": 0,
                "setup": 0,
                "inference": 0,
                "output_shape": 0,
            },
            "errors": [],
            "triples_per_text": [2 for _ in kwargs["texts"]],
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        bench.main,
        [
            "--input-path",
            input_path.as_posix(),
            "--sample-size",
            "2",
            "--trials",
            "2",
            "--backend",
            "none",
            "--output-json",
            output_json.as_posix(),
            "--output-csv",
            output_csv.as_posix(),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "results" in payload
    assert "promotion_gates" in payload
    assert payload["results"]
    row = payload["results"][0]
    assert "failure_counts" in row
    assert "avg_chain_density" in row
    assert output_csv.exists()
