"""Tests for benchmark report formatting and command-line output."""

from __future__ import annotations

import json

from expertkit_torch.benchmark import cli
from expertkit_torch.benchmark.runner import BatchBenchmark, BenchmarkReport, RunMetrics


class Loaded:
    model = object()
    tokenizer = object()
    model_type = "deepseek_v2"

    def __enter__(self) -> Loaded:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def sample_report() -> BenchmarkReport:
    return BenchmarkReport(
        model_type="deepseek_v2",
        mode="expertkit",
        input_length=8,
        output_length=2,
        warmup_runs=0,
        measured_runs=1,
        batches=(BatchBenchmark(1, (RunMetrics(1, 8, 2, 0.5, 0.25),)),),
    )


def test_cli_writes_table_and_json(monkeypatch, tmp_path, capsys) -> None:
    load_arguments: dict[str, object] = {}
    benchmark_arguments: dict[str, object] = {}

    def fake_load_model(model_path: str, **arguments: object) -> Loaded:
        load_arguments.update({"model_path": model_path, **arguments})
        return Loaded()

    def fake_run_benchmark(*_: object, **arguments: object) -> BenchmarkReport:
        benchmark_arguments.update(arguments)
        return sample_report()

    monkeypatch.setattr(cli, "load_model", fake_load_model)
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)
    destination = tmp_path / "results" / "benchmark.json"

    result = cli.main(
        [
            "--model-path",
            "/models/deepseek",
            "--batch-sizes",
            "1",
            "32",
            "--input-length",
            "8",
            "--output-length",
            "2",
            "--warmup-runs",
            "0",
            "--runs",
            "1",
            "--device",
            "cpu",
            "--json-output",
            str(destination),
        ]
    )

    assert result == 0
    assert load_arguments["model_path"] == "/models/deepseek"
    assert benchmark_arguments["batch_sizes"] == [1, 32]
    assert "Prefill tok/s" in capsys.readouterr().out
    payload = json.loads(destination.read_text())
    assert payload["model_type"] == "deepseek_v2"
    assert payload["batches"][0]["median"]["output_tps"] == 2 / 0.75
    assert payload["configuration"] == {
        "controller_endpoint": "127.0.0.1:5002",
        "device": "cpu",
        "instance_id": None,
        "model_path": "/models/deepseek",
        "requested_dtype": "auto",
    }
