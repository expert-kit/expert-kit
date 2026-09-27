#!/usr/bin/env python3
"""Run a real two-process Mooncake Transfer Engine CUDA write smoke test.

This is deliberately not a fake-engine test. Both roles construct Expert Kit's
``TransferEngineRuntime``, register CUDA Tensor storage, and move the payload
through Mooncake. A small TCP control channel only exchanges the registered
target descriptor and completion acknowledgements.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import os
import re
import signal
import sys
import time
from contextlib import suppress
from typing import Any, NamedTuple

_PROTOCOL_VERSION = 1
_CONTROL_LIMIT = 16 * 1024
_ELEMENT_BYTES = 4
_MAX_ELEMENTS = 16 * 1024 * 1024
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DEVICE_PATTERN = re.compile(r"cuda:(0|[1-9][0-9]*)\Z")
_RDMA_DEVICE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_FORBIDDEN_ENVIRONMENT = ("MC_USE_TENT", "MC_USE_TEV1", "MC_FORCE_TCP")


class SmokeFailure(RuntimeError):
    """A hardware, transfer, validation, or control-protocol failure."""


class Endpoint(NamedTuple):
    host: str
    port: int


class SmokeConfig(NamedTuple):
    role: str
    backend: str
    control_endpoint: Endpoint
    segment_name: str
    device: str
    rdma_device: str
    run_id: str
    elements: int
    seed: int
    timeout_seconds: float


def _parse_endpoint(value: str, *, subject: str) -> Endpoint:
    raw = value.strip()
    if raw.startswith("["):
        closing = raw.find("]")
        if closing <= 1 or raw[closing + 1 : closing + 2] != ":":
            raise ValueError(f"{subject} must be HOST:PORT or [IPv6]:PORT")
        host = raw[1:closing]
        port_text = raw[closing + 2 :]
        if "]" in port_text:
            raise ValueError(f"{subject} must be HOST:PORT or [IPv6]:PORT")
    else:
        if raw.count(":") != 1:
            raise ValueError(f"{subject} must be HOST:PORT or [IPv6]:PORT")
        host, port_text = raw.rsplit(":", 1)
    if not host or any(character.isspace() for character in host):
        raise ValueError(f"{subject} host must not be empty or contain whitespace")
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError(f"{subject} port must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"{subject} port must be between 1 and 65535")
    return Endpoint(host, port)


def _positive_elements(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("elements must be an integer") from error
    if not 1 <= parsed <= _MAX_ELEMENTS:
        raise argparse.ArgumentTypeError(
            f"elements must be between 1 and {_MAX_ELEMENTS}"
        )
    return parsed


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= parsed <= 250:
        raise argparse.ArgumentTypeError("seed must be between 0 and 250")
    return parsed


def _positive_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real CUDA Transfer Engine write between listener and initiator roles. "
            "Start the listener first and use the same backend, run ID, element count, "
            "and seed on both processes."
        ),
        epilog=(
            "The control channel is intentionally unauthenticated and unencrypted; "
            "run this diagnostic only on an isolated, trusted test network. The run ID "
            "correlates peers but is not an authentication credential."
        ),
    )
    parser.add_argument("--role", required=True, choices=("listener", "initiator"))
    parser.add_argument("--backend", required=True, choices=("nvlink_intra", "rdma"))
    parser.add_argument(
        "--control-endpoint",
        required=True,
        metavar="HOST:PORT",
        help="listener bind endpoint or initiator connect endpoint",
    )
    parser.add_argument(
        "--segment-name",
        required=True,
        metavar="HOST:PORT",
        help="this process's unique peer-reachable Mooncake P2P endpoint",
    )
    parser.add_argument("--device", required=True, metavar="cuda:INDEX")
    parser.add_argument(
        "--rdma-device",
        default="",
        metavar="DEVICE",
        help="required only for rdma (for example, an mlx5 device name)",
    )
    parser.add_argument("--run-id", required=True, metavar="ID")
    parser.add_argument("--elements", type=_positive_elements, default=1024 * 1024)
    parser.add_argument("--seed", type=_seed, default=17)
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_timeout,
        default=120.0,
        metavar="SECONDS",
    )
    return parser


def _parse_config(arguments: list[str]) -> SmokeConfig:
    parser = _build_parser()
    namespace = parser.parse_args(arguments)
    try:
        control_endpoint = _parse_endpoint(
            namespace.control_endpoint,
            subject="control endpoint",
        )
        segment_endpoint = _parse_endpoint(
            namespace.segment_name, subject="segment name"
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        segment_ip = ipaddress.ip_address(segment_endpoint.host)
    except ValueError:
        segment_ip = None
    if segment_ip is not None and segment_ip.is_unspecified:
        parser.error(
            "segment name must advertise a peer-reachable address, not a wildcard"
        )
    if not _DEVICE_PATTERN.fullmatch(namespace.device):
        parser.error("device must be an indexed CUDA device such as cuda:0")
    if not _RUN_ID_PATTERN.fullmatch(namespace.run_id):
        parser.error(
            "run ID must use 1-128 ASCII letters, digits, dots, dashes, or underscores"
        )
    if namespace.backend == "rdma":
        if not _RDMA_DEVICE_PATTERN.fullmatch(namespace.rdma_device):
            parser.error("rdma backend requires a valid --rdma-device")
    elif namespace.rdma_device:
        parser.error("--rdma-device is valid only with --backend rdma")
    return SmokeConfig(
        role=namespace.role,
        backend=namespace.backend,
        control_endpoint=control_endpoint,
        segment_name=namespace.segment_name.strip(),
        device=namespace.device,
        rdma_device=namespace.rdma_device,
        run_id=namespace.run_id,
        elements=namespace.elements,
        seed=namespace.seed,
        timeout_seconds=namespace.timeout_seconds,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SmokeFailure(f"control message contains duplicate key {key!r}")
        result[key] = value
    return result


def _encode_message(message: dict[str, Any]) -> bytes:
    encoded = (
        json.dumps(message, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > _CONTROL_LIMIT:
        raise SmokeFailure("control message exceeds the size limit")
    return encoded


def _decode_message(encoded: bytes) -> dict[str, Any]:
    if not encoded.endswith(b"\n"):
        raise SmokeFailure("control message is not newline terminated")
    if len(encoded) > _CONTROL_LIMIT:
        raise SmokeFailure("control message exceeds the size limit")
    try:
        decoded = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeFailure("control message is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise SmokeFailure("control message must be a JSON object")
    return decoded


def _require_exact_keys(message: dict[str, Any], expected: set[str]) -> None:
    actual = set(message)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SmokeFailure(
            f"control message fields mismatch: missing={missing}, unexpected={unexpected}"
        )


def _validate_envelope(
    message: dict[str, Any],
    config: SmokeConfig,
    *,
    kind: str,
    fields: set[str],
) -> None:
    _require_exact_keys(message, {"version", "kind", "run_id", "backend", *fields})
    if type(message["version"]) is not int or message["version"] != _PROTOCOL_VERSION:
        raise SmokeFailure("control protocol version mismatch")
    if message["kind"] != kind:
        raise SmokeFailure(f"expected {kind!r} control message")
    if message["run_id"] != config.run_id:
        raise SmokeFailure("control run ID mismatch")
    if message["backend"] != config.backend:
        raise SmokeFailure("control backend mismatch")


def _require_positive_integer(message: dict[str, Any], field: str) -> int:
    value = message[field]
    if type(value) is not int or value <= 0:
        raise SmokeFailure(f"control field {field!r} must be a positive integer")
    return value


def _require_session(message: dict[str, Any]) -> str:
    session_id = message["session_id"]
    if not isinstance(session_id, str) or not session_id or len(session_id) > 512:
        raise SmokeFailure("control session ID is invalid")
    return session_id


def _base_message(config: SmokeConfig, kind: str) -> dict[str, Any]:
    return {
        "version": _PROTOCOL_VERSION,
        "kind": kind,
        "run_id": config.run_id,
        "backend": config.backend,
    }


def _validate_hello(message: dict[str, Any], config: SmokeConfig) -> str:
    fields = {"role", "session_id", "elements", "seed", "bytes"}
    _validate_envelope(message, config, kind="hello", fields=fields)
    if message["role"] != "initiator":
        raise SmokeFailure("hello must come from the initiator role")
    _validate_geometry(message, config)
    return _require_session(message)


def _validate_target(message: dict[str, Any], config: SmokeConfig) -> tuple[str, int]:
    fields = {"role", "session_id", "address", "elements", "seed", "bytes"}
    _validate_envelope(message, config, kind="target", fields=fields)
    if message["role"] != "listener":
        raise SmokeFailure("target must come from the listener role")
    _validate_geometry(message, config)
    return _require_session(message), _require_positive_integer(message, "address")


def _validate_geometry(message: dict[str, Any], config: SmokeConfig) -> None:
    if _require_positive_integer(message, "elements") != config.elements:
        raise SmokeFailure("control element count mismatch")
    if type(message["seed"]) is not int or message["seed"] != config.seed:
        raise SmokeFailure("control pattern seed mismatch")
    if _require_positive_integer(message, "bytes") != config.elements * _ELEMENT_BYTES:
        raise SmokeFailure("control byte length mismatch")


def _validate_write_complete(message: dict[str, Any], config: SmokeConfig) -> None:
    _validate_envelope(message, config, kind="write_complete", fields={"bytes"})
    if _require_positive_integer(message, "bytes") != config.elements * _ELEMENT_BYTES:
        raise SmokeFailure("completed byte length mismatch")


def _validate_result(message: dict[str, Any], config: SmokeConfig) -> tuple[bool, int]:
    fields = {"ok", "checked_elements", "mismatch_count"}
    _validate_envelope(message, config, kind="result", fields=fields)
    if type(message["ok"]) is not bool:
        raise SmokeFailure("result ok field must be Boolean")
    if _require_positive_integer(message, "checked_elements") != config.elements:
        raise SmokeFailure("result element count mismatch")
    mismatch_count = message["mismatch_count"]
    if type(mismatch_count) is not int or not 0 <= mismatch_count <= config.elements:
        raise SmokeFailure("result mismatch count is invalid")
    if message["ok"] != (mismatch_count == 0):
        raise SmokeFailure("result status contradicts its mismatch count")
    return message["ok"], mismatch_count


def _validate_result_ack(message: dict[str, Any], config: SmokeConfig) -> None:
    _validate_envelope(message, config, kind="result_ack", fields={"invalidated"})
    if message["invalidated"] is not True:
        raise SmokeFailure("initiator did not confirm remote-descriptor invalidation")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SmokeFailure("smoke-test deadline expired")
    return remaining


async def _send_message(
    writer: asyncio.StreamWriter,
    message: dict[str, Any],
    deadline: float,
) -> None:
    writer.write(_encode_message(message))
    try:
        async with asyncio.timeout(_remaining(deadline)):
            await writer.drain()
    except TimeoutError as error:
        raise SmokeFailure("timed out sending a control message") from error


async def _receive_message(
    reader: asyncio.StreamReader,
    deadline: float,
) -> dict[str, Any]:
    try:
        async with asyncio.timeout(_remaining(deadline)):
            encoded = await reader.readline()
    except (TimeoutError, ValueError) as error:
        raise SmokeFailure("timed out or exceeded the control-message limit") from error
    if not encoded:
        raise SmokeFailure("control peer closed the connection")
    return _decode_message(encoded)


async def _close_writer(writer: asyncio.StreamWriter | None) -> None:
    if writer is None:
        return
    writer.close()
    with suppress(ConnectionError, OSError):
        await writer.wait_closed()


async def _close_listener_control(
    server: asyncio.Server | None,
    writer: asyncio.StreamWriter | None,
    pending_connections: asyncio.Queue[
        tuple[asyncio.StreamReader, asyncio.StreamWriter]
    ],
) -> None:
    """Close accepted streams before waiting for the listener to terminate."""

    if server is not None:
        server.close()
    if writer is None:
        with suppress(asyncio.QueueEmpty):
            _reader, writer = pending_connections.get_nowait()
    await _close_writer(writer)
    if server is not None:
        # Python 3.12 waits for active accepted connections here, so their
        # writers must be closed first to avoid a circular wait.
        await server.wait_closed()


async def _connect(config: SmokeConfig, deadline: float) -> tuple[Any, Any]:
    last_error: OSError | None = None
    while True:
        remaining = _remaining(deadline)
        try:
            async with asyncio.timeout(min(remaining, 2.0)):
                return await asyncio.open_connection(
                    config.control_endpoint.host,
                    config.control_endpoint.port,
                    limit=_CONTROL_LIMIT,
                )
        except TimeoutError:
            pass
        except OSError as error:
            last_error = error
        if _remaining(deadline) <= 0.2:
            detail = f": {last_error}" if last_error is not None else ""
            raise SmokeFailure(f"could not connect to the listener{detail}")
        await asyncio.sleep(min(0.2, _remaining(deadline)))


def _check_environment() -> None:
    enabled = [name for name in _FORBIDDEN_ENVIRONMENT if name in os.environ]
    if enabled:
        raise SmokeFailure(
            "explicit Transfer Engine smoke forbids environment variables: "
            + ", ".join(enabled)
        )


def _report_phase(config: SmokeConfig, phase: str) -> None:
    """Emit a secret-free progress marker before or after blocking native work."""

    print(
        json.dumps(
            {
                "event": "phase",
                "role": config.role,
                "backend": config.backend,
                "phase": phase,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _load_hardware(config: SmokeConfig) -> tuple[Any, Any]:
    # Torch must be imported before the native Mooncake module so the smoke
    # observes the same CUDA ABI/load order as Expert Kit.
    import torch

    if not torch.cuda.is_available():
        raise SmokeFailure("CUDA is unavailable")
    device = torch.device(config.device)
    assert device.index is not None
    if device.index >= torch.cuda.device_count():
        raise SmokeFailure(f"CUDA device index {device.index} is unavailable")
    torch.cuda.set_device(device)

    from expertkit_transport.transports.transfer_engine import (
        TransferEngineRuntime,
        TransferEngineRuntimeConfig,
    )

    runtime = TransferEngineRuntime(
        TransferEngineRuntimeConfig(
            segment_name=config.segment_name,
            metadata_server="P2PHANDSHAKE",
            protocol=config.backend,
            device=device,
            device_name=config.rdma_device,
            enable_experimental_rdma=config.backend == "rdma",
        )
    )
    return torch, runtime


def _make_pattern(torch: Any, config: SmokeConfig) -> Any:
    indices = torch.arange(config.elements, dtype=torch.int32, device=config.device)
    return torch.remainder(indices + config.seed, 251).to(dtype=torch.float32)


async def _start_runtime(runtime: Any, config: SmokeConfig) -> None:
    _report_phase(config, "runtime_start:start")
    await runtime.start()
    # start() queries the safety-patched binding's get_configured_backend() and
    # fails on a mismatch. Keep the explicit assertion visible in this smoke.
    if runtime.backend != config.backend:
        raise SmokeFailure(
            f"configured backend mismatch: expected {config.backend}, got {runtime.backend!r}"
        )
    _report_phase(config, "runtime_start:done")


async def _run_listener(config: SmokeConfig, deadline: float) -> dict[str, Any]:
    torch, runtime = _load_hardware(config)
    writer: asyncio.StreamWriter | None = None
    server: asyncio.Server | None = None
    destination: Any | None = None
    registered = False
    descriptor_exposed = False
    peer_invalidated = False
    connections: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
        asyncio.Queue(maxsize=1)
    )

    def accept(
        reader: asyncio.StreamReader, accepted_writer: asyncio.StreamWriter
    ) -> None:
        if connections.full():
            accepted_writer.close()
            return
        connections.put_nowait((reader, accepted_writer))

    try:
        # Allocate the CUDA region before Mooncake initializes, matching the
        # proven production Worker arena lifecycle and allocator ordering.
        destination = torch.full(
            (config.elements,),
            -1.0,
            dtype=torch.float32,
            device=config.device,
        )
        torch.cuda.synchronize(config.device)
        _report_phase(config, "cuda_buffer:ready")
        server = await asyncio.start_server(
            accept,
            config.control_endpoint.host,
            config.control_endpoint.port,
            limit=_CONTROL_LIMIT,
        )
        await _start_runtime(runtime, config)
        _report_phase(config, "register_memory:start")
        await runtime.register_tensor(destination, monotonic_deadline=deadline)
        registered = True
        _report_phase(config, "register_memory:done")
        _report_phase(config, "control_peer:wait")
        try:
            async with asyncio.timeout(_remaining(deadline)):
                reader, writer = await connections.get()
        except TimeoutError as error:
            raise SmokeFailure("timed out waiting for the initiator") from error
        server.close()
        # Do not await server.wait_closed() while the accepted writer is still
        # active. Python 3.12 waits for that connection and would deadlock the
        # control protocol before the target descriptor can be sent.
        _report_phase(config, "control_peer:connected")

        hello = await _receive_message(reader, deadline)
        initiator_session = _validate_hello(hello, config)
        if initiator_session == runtime.session_id:
            raise SmokeFailure("both roles advertised the same Mooncake session")
        target = _base_message(config, "target")
        target.update(
            {
                "role": "listener",
                "session_id": runtime.session_id,
                "address": destination.data_ptr(),
                "elements": config.elements,
                "seed": config.seed,
                "bytes": config.elements * _ELEMENT_BYTES,
            }
        )
        await _send_message(writer, target, deadline)
        descriptor_exposed = True
        _report_phase(config, "target:sent")

        complete = await _receive_message(reader, deadline)
        _validate_write_complete(complete, config)
        _report_phase(config, "remote_write_acquire:start")
        await runtime.acquire_remote_writes(monotonic_deadline=deadline)
        _report_phase(config, "remote_write_acquire:done")
        torch.cuda.synchronize(config.device)
        expected = _make_pattern(torch, config)
        mismatch_count = int(torch.count_nonzero(destination != expected).item())
        result = _base_message(config, "result")
        result.update(
            {
                "ok": mismatch_count == 0,
                "checked_elements": config.elements,
                "mismatch_count": mismatch_count,
            }
        )
        await _send_message(writer, result, deadline)
        acknowledgement = await _receive_message(reader, deadline)
        _validate_result_ack(acknowledgement, config)
        peer_invalidated = True
        if mismatch_count:
            raise SmokeFailure(
                f"elementwise CUDA validation found {mismatch_count} mismatches"
            )
        return _success_report(config)
    finally:
        await _close_listener_control(server, writer, connections)
        if registered and destination is not None:
            if descriptor_exposed and not peer_invalidated:
                runtime.quarantine(
                    "smoke listener lost remote-descriptor retirement acknowledgement"
                )
            else:
                _report_phase(config, "unregister_memory:start")
                await runtime.unregister_tensor(
                    destination, monotonic_deadline=math.inf
                )
                _report_phase(config, "unregister_memory:done")
        _report_phase(config, "runtime_close:start")
        await runtime.close()
        _report_phase(config, "runtime_close:done")


async def _run_initiator(config: SmokeConfig, deadline: float) -> dict[str, Any]:
    torch, runtime = _load_hardware(config)
    writer: asyncio.StreamWriter | None = None
    source: Any | None = None
    registered = False
    target_session: str | None = None
    remote_invalidated = False
    try:
        # Allocate before Mooncake initializes for the same reason as the
        # listener destination and the production TransferArena slabs.
        source = _make_pattern(torch, config)
        torch.cuda.synchronize(config.device)
        _report_phase(config, "cuda_buffer:ready")
        await _start_runtime(runtime, config)
        _report_phase(config, "register_memory:start")
        await runtime.register_tensor(source, monotonic_deadline=deadline)
        registered = True
        _report_phase(config, "register_memory:done")
        reader, writer = await _connect(config, deadline)
        _report_phase(config, "control_peer:connected")
        hello = _base_message(config, "hello")
        hello.update(
            {
                "role": "initiator",
                "session_id": runtime.session_id,
                "elements": config.elements,
                "seed": config.seed,
                "bytes": config.elements * _ELEMENT_BYTES,
            }
        )
        await _send_message(writer, hello, deadline)
        target = await _receive_message(reader, deadline)
        target_session, target_address = _validate_target(target, config)
        if target_session == runtime.session_id:
            raise SmokeFailure("both roles advertised the same Mooncake session")
        _report_phase(config, "target:received")

        _report_phase(config, "batch_write:start")
        await runtime.batch_write(
            target_session,
            [source],
            [target_address],
            [config.elements * _ELEMENT_BYTES],
            monotonic_deadline=deadline,
        )
        _report_phase(config, "batch_write:done")
        complete = _base_message(config, "write_complete")
        complete["bytes"] = config.elements * _ELEMENT_BYTES
        await _send_message(writer, complete, deadline)
        result = await _receive_message(reader, deadline)
        ok, mismatch_count = _validate_result(result, config)

        _report_phase(config, "remote_invalidation:start")
        await runtime.invalidate_remote_session(
            target_session,
            monotonic_deadline=math.inf,
        )
        remote_invalidated = True
        _report_phase(config, "remote_invalidation:done")
        acknowledgement = _base_message(config, "result_ack")
        acknowledgement["invalidated"] = True
        await _send_message(writer, acknowledgement, deadline)
        if not ok:
            raise SmokeFailure(
                f"listener reported {mismatch_count} elementwise CUDA mismatches"
            )
        return _success_report(config)
    finally:
        if target_session is not None and not remote_invalidated:
            try:
                await runtime.invalidate_remote_session(
                    target_session,
                    monotonic_deadline=math.inf,
                )
            except asyncio.CancelledError:
                runtime.quarantine(
                    "smoke cancellation interrupted remote target retirement"
                )
                raise
            except Exception:  # noqa: BLE001 - any failed retirement is fail-stop
                runtime.quarantine(
                    "smoke could not retire the remote target descriptor"
                )
        await _close_writer(writer)
        if registered and source is not None:
            _report_phase(config, "unregister_memory:start")
            await runtime.unregister_tensor(source, monotonic_deadline=math.inf)
            _report_phase(config, "unregister_memory:done")
        _report_phase(config, "runtime_close:start")
        await runtime.close()
        _report_phase(config, "runtime_close:done")


def _success_report(config: SmokeConfig) -> dict[str, Any]:
    return {
        "ok": True,
        "role": config.role,
        "configured_backend": config.backend,
        "exact_backend_match": True,
        "device": config.device,
        "elements_checked": config.elements,
        "bytes_transferred": config.elements * _ELEMENT_BYTES,
        "data_path": "mooncake_transfer_engine_cuda_batch_write",
        "validation": "elementwise_exact",
    }


async def _run(config: SmokeConfig) -> dict[str, Any]:
    _check_environment()
    deadline = time.monotonic() + config.timeout_seconds
    if config.role == "listener":
        return await _run_listener(config, deadline)
    return await _run_initiator(config, deadline)


async def _run_with_signal_shutdown(config: SmokeConfig) -> dict[str, Any]:
    """Route SIGTERM through coroutine cancellation so safety cleanup runs."""

    loop = asyncio.get_running_loop()
    task = asyncio.create_task(_run(config), name=f"te-smoke-{config.role}")
    installed = False
    try:
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        installed = True
    except (NotImplementedError, RuntimeError):
        pass
    try:
        return await task
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGTERM)


def main(arguments: list[str] | None = None) -> int:
    config = _parse_config(sys.argv[1:] if arguments is None else arguments)
    try:
        report = asyncio.run(_run_with_signal_shutdown(config))
    except asyncio.CancelledError:
        print(json.dumps({"ok": False, "error": "terminated"}), file=sys.stderr)
        return 143
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "error": "interrupted"}), file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - CLI emits one structured failure
        print(
            json.dumps(
                {
                    "ok": False,
                    "role": config.role,
                    "backend": config.backend,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
