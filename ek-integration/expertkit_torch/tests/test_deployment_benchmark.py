"""Environment-gated model benchmarks against real Expert Kit deployments."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("model_name", "model_path_variable", "endpoint_variable", "instance_variable"),
    [
        pytest.param(
            "qwen3_moe",
            "EK_QWEN_MODEL_PATH",
            "EK_QWEN_CONTROLLER_ENDPOINT",
            "EK_QWEN_INSTANCE_ID",
            marks=pytest.mark.qwen,
        ),
        pytest.param(
            "deepseek_v2",
            "EK_DEEPSEEK_V2_MODEL_PATH",
            "EK_DEEPSEEK_V2_CONTROLLER_ENDPOINT",
            "EK_DEEPSEEK_V2_INSTANCE_ID",
            marks=pytest.mark.deepseek_v2,
        ),
    ],
)
def test_model_benchmark_runs_through_expert_kit(
    model_name: str,
    model_path_variable: str,
    endpoint_variable: str,
    instance_variable: str,
) -> None:
    model_path = os.environ.get(model_path_variable)
    if not model_path:
        pytest.skip(f"{model_path_variable} is not configured")

    endpoint = os.environ.get(endpoint_variable, "127.0.0.1:5002")
    command = [
        sys.executable,
        "-m",
        "expertkit_torch.benchmark.cli",
        "--model-path",
        str(Path(model_path).resolve()),
        "--controller-endpoint",
        endpoint,
        "--batch-sizes",
        "1",
        "--input-length",
        "32",
        "--output-length",
        "20",
        "--warmup-runs",
        "0",
        "--runs",
        "1",
    ]
    if instance_id := os.environ.get(instance_variable):
        command.extend(("--instance-id", instance_id))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Model: {model_name}" in completed.stdout
    assert "Output tok/s" in completed.stdout
