"""Tests for Backend-specific Worker configuration constraints."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from expertkit_worker.config import WorkerConfig


def _config(cache_path: Path, *, backend: str, device: str) -> dict[str, object]:
    return {
        "model": {
            "instance_id": 1,
            "name": "test",
            "weight_version": "v1",
            "num_layers": 1,
            "experts_per_layer": 4,
            "hidden_dim": 16,
            "expert_intermediate_dim": 32,
            "top_k": 2,
            "activation_dtype": "fp16",
            "weight_dtype": "fp16",
        },
        "worker": {
            "id": "worker-0",
            "backend": backend,
            "device": device,
            "device_memory_limit": "2GiB",
        },
        "transport": {"grpc": {"listen": "127.0.0.1:50051", "advertise": "worker:50051"}},
        "controller": {"endpoint": "controller:50050"},
        "weight_manager": {
            "disk_cache": {"path": cache_path},
            "peer": {
                "listen": "127.0.0.1:50052",
                "advertise": "http://worker:50052",
            },
            "weight_server_endpoint": "http://weights:8080",
        },
    }


def test_ggml_requires_cpu_threads(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="ggml", device="cpu")

    with pytest.raises(ValidationError, match="ggml configuration is required"):
        WorkerConfig.model_validate(raw)

    raw["ggml"] = {"cpu_threads": 8}
    config = WorkerConfig.model_validate(raw)
    assert config.ggml is not None
    assert config.ggml.cpu_threads == 8


def test_fused_rejects_fp32(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="fused", device="cuda:1")
    model = raw["model"]
    assert isinstance(model, dict)
    model["activation_dtype"] = "fp32"

    with pytest.raises(ValidationError, match="supports only FP16 or BF16"):
        WorkerConfig.model_validate(raw)


def test_paths_must_be_absolute(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    weight_manager = raw["weight_manager"]
    assert isinstance(weight_manager, dict)
    weight_manager["disk_cache"] = {"path": "relative/cache"}

    with pytest.raises(ValidationError, match="path must be absolute"):
        WorkerConfig.model_validate(raw)


def test_device_memory_limit_must_be_positive(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["device_memory_limit"] = 0

    with pytest.raises(ValidationError, match="greater than 0"):
        WorkerConfig.model_validate(raw)
