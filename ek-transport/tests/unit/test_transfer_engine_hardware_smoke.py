"""Argument and wire-protocol tests for the real Transfer Engine smoke script."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
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


def test_phase_marker_is_flushed_and_does_not_expose_endpoints_or_run_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = smoke._parse_config(_arguments("rdma"))

    smoke._report_phase(config, "register_memory:start")

    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "backend": "rdma",
        "event": "phase",
        "phase": "register_memory:start",
        "role": "listener",
    }
    encoded = json.dumps(payload)
    assert config.segment_name not in encoded
    assert config.control_endpoint.host not in encoded
    assert config.run_id not in encoded


class _StopAtRuntimeStart(RuntimeError):
    pass


class _OrderingRuntime:
    backend = "rdma"

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def start(self) -> None:
        self._calls.append("runtime_start")
        raise _StopAtRuntimeStart

    async def close(self) -> None:
        self._calls.append("runtime_close")


class _OrderingCuda:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def synchronize(self, _device: str) -> None:
        self._calls.append("cuda_synchronize")


class _OrderingTorch:
    float32 = object()

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.cuda = _OrderingCuda(calls)

    def full(self, *_args: object, **_kwargs: object) -> object:
        self._calls.append("cuda_allocate")
        return object()


class _OrderingServer:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


def test_listener_closes_accepted_writer_before_waiting_for_server() -> None:
    events: list[str] = []

    class Writer:
        closed = False

        def close(self) -> None:
            events.append("writer_close")
            self.closed = True

        async def wait_closed(self) -> None:
            events.append("writer_wait_closed")

    writer = Writer()

    class Server:
        def close(self) -> None:
            events.append("server_close")

        async def wait_closed(self) -> None:
            assert writer.closed
            events.append("server_wait_closed")

    connections: asyncio.Queue[tuple[object, object]] = asyncio.Queue()
    asyncio.run(smoke._close_listener_control(Server(), writer, connections))

    assert events == [
        "server_close",
        "writer_close",
        "writer_wait_closed",
        "server_wait_closed",
    ]


def test_listener_allocates_cuda_region_before_runtime_start(monkeypatch) -> None:
    calls: list[str] = []
    config = smoke._parse_config(_arguments("rdma"))
    torch = _OrderingTorch(calls)
    runtime = _OrderingRuntime(calls)

    async def start_server(*_args: object, **_kwargs: object) -> _OrderingServer:
        calls.append("control_server_start")
        return _OrderingServer()

    monkeypatch.setattr(smoke, "_load_hardware", lambda _config: (torch, runtime))
    monkeypatch.setattr(smoke.asyncio, "start_server", start_server)

    with pytest.raises(_StopAtRuntimeStart):
        asyncio.run(smoke._run_listener(config, time.monotonic() + 10))

    assert calls == [
        "cuda_allocate",
        "cuda_synchronize",
        "control_server_start",
        "runtime_start",
        "runtime_close",
    ]


def test_initiator_allocates_cuda_region_before_runtime_start(monkeypatch) -> None:
    calls: list[str] = []
    arguments = _arguments("rdma")
    arguments[arguments.index("listener")] = "initiator"
    arguments[arguments.index("listener.test:46000")] = "initiator.test:46000"
    config = smoke._parse_config(arguments)
    torch = _OrderingTorch(calls)
    runtime = _OrderingRuntime(calls)

    def make_pattern(_torch: object, _config: smoke.SmokeConfig) -> object:
        calls.append("cuda_allocate")
        return object()

    monkeypatch.setattr(smoke, "_load_hardware", lambda _config: (torch, runtime))
    monkeypatch.setattr(smoke, "_make_pattern", make_pattern)

    with pytest.raises(_StopAtRuntimeStart):
        asyncio.run(smoke._run_initiator(config, time.monotonic() + 10))

    assert calls == [
        "cuda_allocate",
        "cuda_synchronize",
        "runtime_start",
        "runtime_close",
    ]
