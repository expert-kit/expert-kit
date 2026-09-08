"""Structural tests for generated Ascend deployment configuration."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
ASCEND = ROOT / "dev" / "ascend"


def _inputs(
    tmp_path: Path,
    *,
    model_config: str = "qwen3-30b-a3b",
    dataset_type: str = "sharegpt",
) -> tuple[Path, Path]:
    cluster = tmp_path / "cluster.yaml"
    experiment = tmp_path / "experiment.yaml"
    example_dir = ASCEND / "configs" / model_config
    shutil.copy(example_dir / "cluster.example.yaml", cluster)
    shutil.copy(example_dir / "experiment.example.yaml", experiment)

    cluster_data = yaml.safe_load(cluster.read_text(encoding="utf-8"))
    cluster_data["nodes"]["node-a"]["address"] = "192.0.2.10"
    cluster_data["nodes"]["node-b"]["address"] = "192.0.2.11"
    cluster_data["nodes"]["node-c"]["address"] = "192.0.2.12"
    cluster.write_text(yaml.safe_dump(cluster_data), encoding="utf-8")

    data = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    data["dataset"] = {
        "type": dataset_type,
        "name": dataset_type.title(),
    }
    if dataset_type == "sharegpt":
        data["dataset"].update(
            {
                "path_ref": "sharegpt",
                "mounted_path": "/dataset/sharegpt",
                "file": "ShareGPT.json",
            }
        )
        data["run"].pop("input_len", None)
    experiment.write_text(yaml.safe_dump(data), encoding="utf-8")
    return cluster, experiment


def _generate(cluster: Path, experiment: Path, output: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ASCEND)
    completed = subprocess.run(
        [
            sys.executable,
            str(ASCEND / "cli.py"),
            "generate",
            "--cluster",
            str(cluster),
            "--experiment",
            str(experiment),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("model_config", "expected_model_path", "expected_memory_limit"),
    [
        (
            "qwen3-30b-a3b",
            "/home/<USER>/expert-kit/local/models/Qwen3-30B-A3B",
            "20GiB",
        ),
        (
            "deepseek-v3",
            "/home/<USER>/models/DeepSeek-V3-bf16",
            "52GiB",
        ),
    ],
)
def test_sharegpt_generation_emits_host_native_torch_config(
    tmp_path: Path,
    model_config: str,
    expected_model_path: str,
    expected_memory_limit: str,
) -> None:
    cluster, experiment = _inputs(
        tmp_path,
        model_config=model_config,
    )
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)
    config = yaml.safe_load((output / "torch-bench.yaml").read_text())

    assert config["controller-endpoint"] == "192.0.2.10:15002"
    assert config["device-platform"] == "npu"
    assert config["device-ids"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert config["num-prompts"] == 16
    assert config["max-concurrency"] == 1
    assert config["dataset-path"] == (
        "/home/<USER>/expert-kit/local/datasets/sharegpt/ShareGPT.json"
    )
    assert config["model-path"] == expected_model_path

    worker = yaml.safe_load((output / "worker-pool1" / "workers" / "worker-00.yaml").read_text())
    assert worker["worker"]["device_memory_limit"] == expected_memory_limit


def test_generation_splits_control_and_attention_bundles(tmp_path: Path) -> None:
    cluster, experiment = _inputs(tmp_path)
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)
    control = yaml.safe_load((output / "compose.control.yaml").read_text())
    attention = yaml.safe_load((output / "compose.attention.yaml").read_text())

    assert control["name"] == "expert-kit-ascend-control"
    assert set(control["services"]) == {
        "admin",
        "controller",
        "migrations",
        "model-init",
        "postgres",
        "weight-server",
    }
    assert attention["name"] == "expert-kit-ascend-attention"
    assert set(attention["services"]) == {"attention", "benchmark"}
    assert attention["services"]["attention"]["environment"]["EK_ADDR"] == ("192.0.2.10:15002")


@pytest.mark.parametrize(
    "model_config",
    [
        "qwen3-30b-a3b",
        "deepseek-v3",
    ],
)
def test_random_generation_does_not_emit_torch_config(
    tmp_path: Path,
    model_config: str,
) -> None:
    cluster, experiment = _inputs(
        tmp_path,
        model_config=model_config,
        dataset_type="random",
    )
    output = tmp_path / "generated"

    _generate(cluster, experiment, output)

    assert not (output / "torch-bench.yaml").exists()
