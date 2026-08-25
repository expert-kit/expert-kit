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
        num_prompts=1,
        output_length=2,
        warmup_runs=0,
        batches=(BatchBenchmark(1, (RunMetrics(1, 8, 8, 2, 0.5, 0.25),)),),
    )


def test_cli_writes_table_and_json(monkeypatch, tmp_path, capsys) -> None:
    load_arguments: dict[str, object] = {}
    benchmark_arguments: dict[str, object] = {}
    benchmark_positional: tuple[object, ...] = ()
    dataset = object()

    def fake_load_model(model_path: str, **arguments: object) -> Loaded:
        load_arguments.update({"model_path": model_path, **arguments})
        return Loaded()

    def fake_run_benchmark(
        *positional: object,
        **arguments: object,
    ) -> BenchmarkReport:
        nonlocal benchmark_positional
        benchmark_positional = positional
        benchmark_arguments.update(arguments)
        return sample_report()

    monkeypatch.setattr(cli, "load_model", fake_load_model)
    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(cli, "ShareGPTDataset", lambda path, *, seed: dataset)
    destination = tmp_path / "results" / "benchmark.json"
    source = tmp_path / "sharegpt.json"

    result = cli.main(
        [
            "--model-path",
            "/models/deepseek",
            "--batch-sizes",
            "1",
            "32",
            "--dataset-path",
            str(source),
            "--seed",
            "7",
            "--num-prompts",
            "32",
            "--output-length",
            "2",
            "--warmup-runs",
            "0",
            "--device",
            "cpu",
            "--json-output",
            str(destination),
        ]
    )

    assert result == 0
    assert load_arguments["model_path"] == "/models/deepseek"
    assert benchmark_positional == (Loaded.model, Loaded.tokenizer, dataset)
    assert benchmark_arguments["batch_sizes"] == [1, 32]
    assert benchmark_arguments["num_prompts"] == 32
    assert "Prefill tok/s" in capsys.readouterr().out
    payload = json.loads(destination.read_text())
    assert payload["model_type"] == "deepseek_v2"
    assert payload["batches"][0]["median"]["output_tps"] == 2 / 0.75
    assert payload["configuration"] == {
        "controller_endpoint": "127.0.0.1:5002",
        "dataset_name": "sharegpt",
        "dataset_path": str(source),
        "device": "cpu",
        "instance_id": None,
        "model_path": "/models/deepseek",
        "requested_dtype": "auto",
        "seed": 7,
    }
