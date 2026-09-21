"""Tests for the Typer benchmark command and JSON output."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from expertkit_torch.benchmark import cli
from expertkit_torch.benchmark.launcher import (
    GlobalBenchmarkReport,
    RankAssignment,
    RankBenchmarkReport,
)
from expertkit_torch.benchmark.runner import BatchBenchmark, BenchmarkReport, RunMetrics


def test_cli_writes_global_table_and_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    benchmark = BenchmarkReport(
        model_type="deepseek_v2",
        mode="local",
        num_prompts=1,
        output_length=2,
        warmup_runs=0,
        batches=(BatchBenchmark(1, (RunMetrics(1, 8, 8, 2, 0.5, 0.25),)),),
    )
    rank = RankBenchmarkReport(
        assignment=RankAssignment(0, 0, 0, 1),
        started_at=0.0,
        ended_at=0.75,
        benchmark=benchmark,
    )
    captured = []

    class FakeLauncher:
        def launch(self, config) -> GlobalBenchmarkReport:
            captured.append(config)
            return GlobalBenchmarkReport("deepseek_v2", "local", (rank,))

    monkeypatch.setattr(cli, "select_launcher", lambda _: FakeLauncher())
    destination = tmp_path / "results" / "benchmark.json"
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--model-path",
            "/models/deepseek",
            "--dataset-path",
            "/datasets/sharegpt.json",
            "--device-platform",
            "cpu",
            "--device-ids",
            "0",
            "--mode",
            "local",
            "--num-prompts",
            "1",
            "--max-concurrency",
            "1",
            "--output-length",
            "2",
            "--warmup-runs",
            "0",
            "--json-output",
            str(destination),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured[0].model_path == Path("/models/deepseek")
    assert "Output throughput (tok/s):" in result.stdout
    payload = json.loads(destination.read_text())
    assert payload["model_type"] == "deepseek_v2"
    assert payload["output_tokens"] == 2
    assert payload["configuration"]["mode"] == "local"
