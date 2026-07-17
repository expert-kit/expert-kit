"""Environment-gated Qwen generation check against a real Expert Kit deployment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.qwen
def test_qwen_generates_text_through_expert_kit() -> None:
    model_path = os.environ.get("EK_QWEN_MODEL_PATH")
    if not model_path:
        pytest.skip("EK_QWEN_MODEL_PATH is not configured")

    endpoint = os.environ.get("EK_QWEN_CONTROLLER_ENDPOINT", "127.0.0.1:5002")
    instance_id = os.environ.get("EK_QWEN_INSTANCE_ID", "1")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "expertkit_torch.qwen_smoke",
            "--model-path",
            str(Path(model_path).resolve()),
            "--controller-endpoint",
            endpoint,
            "--instance-id",
            instance_id,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Qwen smoke passed:" in completed.stdout
