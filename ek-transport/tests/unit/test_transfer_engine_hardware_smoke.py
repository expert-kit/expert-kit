"""Argument and wire-protocol tests for the real Transfer Engine smoke script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "smoke-transfer-engine.py"
_SPEC = importlib.util.spec_from_file_location("smoke_transfer_engine", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def _arguments(backend: str = "nvlink_intra") -> list[str]:
    arguments = [
        "--role",
        "listener",
        "--backend",
        backend,
        "--control-endpoint",
        "control.test:45000",
        "--segment-name",
        "listener.test:46000",
        "--device",
        "cuda:1",
        "--run-id",
        "nightly-17",
    ]
    if backend == "rdma":
        arguments.extend(("--rdma-device", "mlx5_0"))
    return arguments


def test_hardware_options_parse_without_importing_torch() -> None:
    config = smoke._parse_config(_arguments())

    assert config.role == "listener"
    assert config.backend == "nvlink_intra"
    assert config.device == "cuda:1"
    assert config.control_endpoint == ("control.test", 45000)
    assert "torch" not in smoke.__dict__


def test_rdma_requires_an_explicit_device() -> None:
    with pytest.raises(SystemExit):
        smoke._parse_config(_arguments("rdma")[:-2])


def test_rdma_initiator_keeps_process_local_engine_and_nic_settings() -> None:
    arguments = _arguments("rdma")
    arguments[arguments.index("listener")] = "initiator"
    arguments[arguments.index("listener.test:46000")] = "initiator.test:46000"

    config = smoke._parse_config(arguments)

    assert config.role == "initiator"
    assert config.backend == "rdma"
    assert config.segment_name == "initiator.test:46000"
    assert config.rdma_device == "mlx5_0"


def test_nvlink_rejects_an_rdma_device() -> None:
    with pytest.raises(SystemExit):
        smoke._parse_config([*_arguments(), "--rdma-device", "mlx5_0"])


@pytest.mark.parametrize("device", ["cuda", "cuda:-1", "cpu", "cuda:any"])
def test_only_indexed_cuda_devices_are_accepted(device: str) -> None:
    arguments = _arguments()
    arguments[arguments.index("cuda:1")] = device

    with pytest.raises(SystemExit):
        smoke._parse_config(arguments)


@pytest.mark.parametrize(
    "endpoint",
    ["", "host", "host:0", "host:65536", "host:not-a-port", "bad host:1234"],
)
def test_invalid_control_endpoints_are_rejected(endpoint: str) -> None:
    arguments = _arguments()
    arguments[arguments.index("control.test:45000")] = endpoint

    with pytest.raises(SystemExit):
        smoke._parse_config(arguments)


def test_segment_wildcard_is_rejected_but_control_wildcard_is_allowed() -> None:
    arguments = _arguments()
    arguments[arguments.index("control.test:45000")] = "[::]:45000"
    config = smoke._parse_config(arguments)
    assert config.control_endpoint == ("::", 45000)

    arguments[arguments.index("listener.test:46000")] = "[::]:46000"
    with pytest.raises(SystemExit):
        smoke._parse_config(arguments)


def test_message_round_trip_and_target_validation() -> None:
    config = smoke._parse_config(_arguments())
    target = smoke._base_message(config, "target")
    target.update(
        {
            "role": "listener",
            "session_id": "listener.test:46001",
            "address": 4096,
            "elements": config.elements,
            "seed": config.seed,
            "bytes": config.elements * smoke._ELEMENT_BYTES,
        }
    )

    decoded = smoke._decode_message(smoke._encode_message(target))

    assert smoke._validate_target(decoded, config) == ("listener.test:46001", 4096)


@pytest.mark.parametrize(
    "encoded",
    [
        b"[]\n",
        b'{"version":1}',
        b'{"version":1,"version":1}\n',
        b"not-json\n",
    ],
)
def test_malformed_control_frames_fail_closed(encoded: bytes) -> None:
    with pytest.raises(smoke.SmokeFailure):
        smoke._decode_message(encoded)


def test_protocol_rejects_bool_as_a_remote_address() -> None:
    config = smoke._parse_config(_arguments())
    target = smoke._base_message(config, "target")
    target.update(
        {
            "role": "listener",
            "session_id": "listener.test:46001",
            "address": True,
            "elements": config.elements,
            "seed": config.seed,
            "bytes": config.elements * smoke._ELEMENT_BYTES,
        }
    )

    with pytest.raises(smoke.SmokeFailure, match="positive integer"):
        smoke._validate_target(target, config)


def test_protocol_rejects_cross_run_and_backend_mixups() -> None:
    config = smoke._parse_config(_arguments())
    hello = smoke._base_message(config, "hello")
    hello.update(
        {
            "role": "initiator",
            "session_id": "initiator.test:46002",
            "elements": config.elements,
            "seed": config.seed,
            "bytes": config.elements * smoke._ELEMENT_BYTES,
        }
    )
    hello["run_id"] = "another-run"

    with pytest.raises(smoke.SmokeFailure, match="run ID mismatch"):
        smoke._validate_hello(hello, config)

    hello["run_id"] = config.run_id
    hello["backend"] = "rdma"
    with pytest.raises(smoke.SmokeFailure, match="backend mismatch"):
        smoke._validate_hello(hello, config)


@pytest.mark.parametrize("variable", smoke._FORBIDDEN_ENVIRONMENT)
def test_forced_or_tent_environment_is_rejected(monkeypatch, variable: str) -> None:
    monkeypatch.setenv(variable, "0")

    with pytest.raises(smoke.SmokeFailure, match=variable):
        smoke._check_environment()
