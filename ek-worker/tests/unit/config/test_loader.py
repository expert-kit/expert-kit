"""Tests for loading and validating the Worker YAML configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from expertkit_worker.config import BackendName, ConfigFileError, load_config


def _valid_config(cache_path: Path) -> str:
    return f"""
model:
  instance_id: 7
  name: Qwen-Test
  weight_version: test-revision
  num_layers: 2
  experts_per_layer: 8
  hidden_dim: 64
  expert_intermediate_dim: 128
  top_k: 2
  activation_dtype: bf16
  weight_dtype: bf16
worker:
  id: worker-0
  backend: torch
  device: cuda:0
  max_batch_tokens: 32
  max_active_batches_per_device: 2
  device_memory_limit: 8GiB
transport:
  grpc:
    listen: 127.0.0.1:50051
    advertise: worker-0.internal:50051
controller:
  endpoint: controller.internal:50050
weight_manager:
  disk_cache:
    path: {cache_path}
  peer:
    listen: 127.0.0.1:50052
    advertise: http://worker-0.internal:50052
  weight_server_endpoint: http://weights.internal:8080
"""


def test_loads_valid_config_and_resolves_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(_valid_config(tmp_path / "weights"), encoding="utf-8")

    config = load_config(config_path)

    assert config.worker.backend is BackendName.TORCH
    assert config.transport.max_pending_batches_per_device == 2
    assert config.weight_manager.max_concurrent_loads == 64
    assert config.weight_manager.state_report.max_updates == 64
    assert config.weight_manager.state_report.max_delay_ms == 50
    assert config.weight_manager.disk_cache.writeback is True
    assert config.weight_manager.dram_cache.max_bytes is None
    assert config.logging.level.value == "INFO"
    assert config.logging.format.value == "json"
    assert config.observability.prometheus.enabled is False
    assert config.observability.tracing.enabled is False


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        ("heartbeat_timeout_secs: 2", "heartbeat_timeout_secs"),
        ("top_k: 9", "model.top_k"),
        ("device: cpu", "requires worker.device"),
        ("max_updates: 65", "less than or equal to 64"),
    ],
)
def test_rejects_invalid_values(
    tmp_path: Path,
    replacement: str,
    expected: str,
) -> None:
    contents = _valid_config(tmp_path / "weights")
    if replacement.startswith("heartbeat_timeout_secs"):
        contents = contents.replace(
            "controller:\n  endpoint: controller.internal:50050",
            f"controller:\n  endpoint: controller.internal:50050\n  {replacement}",
        )
    elif replacement.startswith("max_updates"):
        contents += f"  state_report:\n    {replacement}\n"
    elif replacement.startswith("top_k"):
        contents = contents.replace("top_k: 2", replacement)
    else:
        contents = contents.replace("device: cuda:0", replacement)
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValidationError, match=expected):
        load_config(config_path)


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        _valid_config(tmp_path / "weights") + "unexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(config_path)


@pytest.mark.parametrize("contents", ["- not\n- a\n- mapping\n", "model: [\n"])
def test_rejects_invalid_yaml_documents(tmp_path: Path, contents: str) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigFileError):
        load_config(config_path)


def test_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigFileError, match="cannot read Worker configuration"):
        load_config(tmp_path / "missing.yaml")
