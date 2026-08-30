"""Tests for benchmark YAML loading and Typer precedence."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from expertkit_torch.benchmark import cli
from expertkit_torch.benchmark.config import BenchmarkConfig
from expertkit_torch.benchmark.launcher import (
    GlobalBenchmarkReport,
    RankAssignment,
    RankBenchmarkReport,
)
from expertkit_torch.benchmark.runner import BatchBenchmark, BenchmarkReport, RunMetrics


def _config_data(tmp_path: Path) -> dict[str, object]:
    return {
        "mode": "local",
        "device-platform": "cpu",
        "device-ids": [0, 1],
        "model-path": "model",
        "dataset-path": "dataset.json",
        "num-prompts": 4,
        "max-concurrency": 4,
        "output-length": 2,
        "warmup-runs": 0,
    }


def test_config_resolves_paths_and_derives_rank_batch(tmp_path: Path) -> None:
    path = tmp_path / "torch-bench.yaml"
    path.write_text(yaml.safe_dump(_config_data(tmp_path)), encoding="utf-8")

    config = BenchmarkConfig.from_yaml(path)

    assert config.model_path == (tmp_path / "model").resolve()
    assert config.dataset_path == (tmp_path / "dataset.json").resolve()
    assert config.batch_size_per_rank == 2


def test_config_rejects_non_divisible_global_concurrency(tmp_path: Path) -> None:
    data = _config_data(tmp_path)
    data["max-concurrency"] = 3

    with pytest.raises(ValueError, match="divisible"):
        BenchmarkConfig.model_validate(data)


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    data = _config_data(tmp_path)
    data["unexpected"] = True

    with pytest.raises(ValueError, match="unexpected"):
        BenchmarkConfig.model_validate(data)


def test_typer_config_defaults_and_cli_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "torch-bench.yaml"
    config_path.write_text(yaml.safe_dump(_config_data(tmp_path)), encoding="utf-8")
    captured: list[BenchmarkConfig] = []

    benchmark = BenchmarkReport(
        model_type="fake",
        mode="local",
        num_prompts=4,
        output_length=2,
        warmup_runs=0,
        batches=(BatchBenchmark(2, (RunMetrics(2, 4, 2, 2, 1.0, 1.0),)),),
    )
    rank = RankBenchmarkReport(
        assignment=RankAssignment(0, 0, 0, 4),
        started_at=0.0,
        ended_at=2.0,
        benchmark=benchmark,
    )

    class FakeLauncher:
        def launch(self, config: BenchmarkConfig) -> GlobalBenchmarkReport:
            captured.append(config)
            return GlobalBenchmarkReport("fake", "local", (rank,))

    monkeypatch.setattr(cli, "select_launcher", lambda _: FakeLauncher())
    result = CliRunner().invoke(
        cli.app,
        [
            "--config",
            str(config_path),
            "run",
            "--device-ids",
            "0",
            "--max-concurrency",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured[0].device_ids == (0,)
    assert captured[0].max_concurrency == 2
    assert captured[0].model_path == (tmp_path / "model").resolve()

    config_only = CliRunner().invoke(
        cli.app,
        ["--config", str(config_path), "run"],
    )

    assert config_only.exit_code == 0, config_only.stdout
    assert captured[1].device_ids == (0, 1)
    assert captured[1].max_concurrency == 4
