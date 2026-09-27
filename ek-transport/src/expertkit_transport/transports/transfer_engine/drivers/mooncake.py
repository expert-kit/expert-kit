"""Process-wide Mooncake Transfer Engine adapter.

Mooncake's Python API is synchronous. Native submissions use a bounded executor;
local staging completion is queried in batches on a separate bounded executor.
Registered memory remains owned until device work reaches a terminal state.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar
from urllib.parse import urlsplit

import torch

from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.transfer_engine.drivers.base import MemoryRegion, MemorySlice

_T = TypeVar("_T")
_SUPPORTED_PROTOCOLS = frozenset({"tcp", "rdma", "nvlink", "nvlink_intra", "ascend_direct"})
_QUARANTINED_RUNTIMES: list[object] = []


def _deadline_error(diagnostic: str) -> TransportError:
    return TransportError(
        TransportErrorCode.DEADLINE_EXCEEDED,
        retryable=False,
        diagnostic=diagnostic,
    )


def _p2p_session_id(endpoint: str, rpc_port: int) -> str:
    """Return the peer endpoint using the port Mooncake actually bound."""

    if endpoint.count(":") > 1 and not endpoint.startswith("["):
        return f"[{endpoint}]:{rpc_port}"
    try:
        hostname = urlsplit(f"//{endpoint}").hostname
    except ValueError:
        hostname = None
    if not hostname:
        raise RuntimeError("Mooncake P2P segment name is not a valid host endpoint")
    if ":" in hostname:
        return f"[{hostname}]:{rpc_port}"
    return f"{hostname}:{rpc_port}"


def _require_native_capability(native_api: object, name: str, diagnostic: str) -> None:
    if getattr(native_api, name, False) is not True:
        raise RuntimeError(diagnostic)


def _validate_native_capabilities(
    native_api: object,
    config: MooncakeDriverConfig,
) -> None:
    """Fail before engine construction when a backend lacks EK safety contracts."""

    if config.protocol == "ascend_direct":
        for name in (
            "EK_ASCEND_SYNC_SUCCESS_COMPLETION",
            "EK_HAS_ASCEND_ACQUIRE",
            "EK_FORCE_CONFIGURED_ASCEND_DIRECT_TRANSPORT",
            "EK_DRAINED_ASCEND_REMOTE_DESCRIPTOR_INVALIDATION",
        ):
            _require_native_capability(
                native_api, name, f"the Mooncake wheel lacks required Ascend capability {name}"
            )
    else:
        _require_native_capability(
            native_api,
            "EK_SAFE_TERMINAL_BATCH_SYNC",
            "the installed Mooncake wheel cannot prove terminal DMA state; "
            "install an Expert Kit safety-capable wheel",
        )
    if config.device.type == "cuda":
        _require_native_capability(
            native_api,
            "EK_HAS_GPUDIRECT_ACQUIRE",
            "the installed Mooncake wheel lacks the GPUDirect acquire fence",
        )
    if config.protocol == "nvlink_intra":
        _require_native_capability(
            native_api,
            "EK_INTRA_NVLINK_REGISTRATION_REFCOUNT",
            "the installed Mooncake wheel lacks safe intra-NVLink registration reference counting",
        )
        _require_native_capability(
            native_api,
            "EK_FORCE_CONFIGURED_TRANSPORT",
            "the installed Mooncake wheel cannot force the configured intra-NVLink backend",
        )
        _require_native_capability(
            native_api,
            "EK_DRAINED_NVLINK_INTRA_LOCAL_INVALIDATION",
            "the installed Mooncake wheel lacks drained intra-NVLink segment invalidation",
        )
    elif config.protocol == "rdma":
        _require_native_capability(
            native_api,
            "EK_FORCE_CONFIGURED_RDMA_TRANSPORT",
            "the installed Mooncake wheel cannot force the configured RDMA backend",
        )
        _require_native_capability(
            native_api,
            "EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION",
            "the installed Mooncake wheel lacks drained RDMA remote-descriptor invalidation",
        )


def _configured_backend_query(engine: object, backend: str) -> Callable[[], object] | None:
    if backend not in {"nvlink_intra", "rdma", "ascend_direct"}:
        return None
    query = getattr(engine, "get_configured_backend", None)
    if not callable(query):
        raise RuntimeError("the Mooncake wheel cannot report its configured data backend")
    return query


def _validate_configured_backend(actual: object, expected: str) -> str:
    if not isinstance(actual, str) or actual != expected:
        raise RuntimeError(
            f"Mooncake configured backend mismatch: expected {expected}, got {actual!r}"
        )
    return actual


def _remote_invalidation_api(
    engine: object,
    backend: str,
) -> tuple[Callable[[str], object], str]:
    if backend == "nvlink_intra":
        name = "invalidate_drained_nvlink_intra_segment"
        subject = "Mooncake drained intra-NVLink segment invalidation"
    elif backend == "rdma":
        name = "invalidate_drained_rdma_segment"
        subject = "Mooncake drained RDMA remote-descriptor invalidation"
    elif backend == "ascend_direct":
        name = "invalidate_drained_ascend_segment"
        subject = "Mooncake drained Ascend remote-descriptor invalidation"
    else:
        raise RuntimeError(
            "drained remote-session invalidation requires a forced "
            "intra-NVLink, RDMA, or Ascend Direct backend"
        )
    invalidate = getattr(engine, name, None)
    if not callable(invalidate):
        raise RuntimeError(f"the Mooncake wheel lacks {subject.lower()}")
    return invalidate, subject


@dataclass(frozen=True, slots=True)
class MooncakeDriverConfig:
    """Configure one Mooncake engine shared by every connection in a process."""

    segment_name: str
    metadata_server: str
    protocol: str
    device: torch.device | str
    device_name: str = ""
    max_workers: int = 2
    transport_hint: str = ""
    enable_experimental_rdma: bool = False
    max_registered_bytes: int | None = None
    client_max_in_flight: int | None = None
    ascend_receive_fence: str = "stream"

    def __post_init__(self) -> None:
        if not isinstance(self.ascend_receive_fence, str) or self.ascend_receive_fence not in {
            "stream",
            "device",
        }:
            raise ValueError("ascend_receive_fence must be stream or device")
        for name in ("segment_name", "metadata_server", "protocol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must not be empty")
        for name in ("device_name", "transport_hint"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not isinstance(self.enable_experimental_rdma, bool):
            raise ValueError("enable_experimental_rdma must be a Boolean")
        if self.protocol not in _SUPPORTED_PROTOCOLS:
            raise ValueError("protocol must be tcp, rdma, nvlink, nvlink_intra, or ascend_direct")
        if self.transport_hint:
            raise ValueError(
                "transport_hint must be empty for the single-backend Transfer Engine MVP"
            )
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or self.max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer")
        device = torch.device(self.device)
        if device.type not in {"cpu", "cuda", "npu"}:
            raise ValueError("Transfer Engine runtime device must be CPU, CUDA, or NPU")
        if device.type in {"cuda", "npu"} and device.index is None:
            raise ValueError("Transfer Engine runtime requires an indexed CUDA or NPU device")
        if self.protocol == "ascend_direct":
            if device.type != "npu":
                raise ValueError("Ascend Direct requires an indexed NPU device")
            if self.metadata_server != "P2PHANDSHAKE":
                raise ValueError("Ascend Direct requires metadata_server=P2PHANDSHAKE")
        elif device.type == "npu":
            raise ValueError("NPU Transfer Engine requires protocol=ascend_direct")
        if self.protocol in {"nvlink", "nvlink_intra"} and device.type != "cuda":
            raise ValueError("Transfer Engine NVLink backends require a CUDA device")
        if self.protocol == "rdma":
            if not self.enable_experimental_rdma:
                raise ValueError("Transfer Engine RDMA requires enable_experimental_rdma=True")
            if device.type != "cuda":
                raise ValueError("Transfer Engine RDMA requires an indexed CUDA device")
            if not self.device_name.strip():
                raise ValueError("Transfer Engine RDMA requires a non-empty device_name")
            if self.metadata_server != "P2PHANDSHAKE":
                raise ValueError("Transfer Engine RDMA requires metadata_server=P2PHANDSHAKE")
        elif self.enable_experimental_rdma:
            raise ValueError("enable_experimental_rdma is valid only when protocol is rdma")
        for name in ("max_registered_bytes", "client_max_in_flight"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        object.__setattr__(self, "device", device)


class MooncakeMemoryTransport:
    """Adapt ``mooncake.engine.TransferEngine`` without importing it eagerly."""

    def __init__(
        self,
        config: MooncakeDriverConfig,
        *,
        engine: object | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, MooncakeDriverConfig):
            raise TypeError("config must be a MooncakeDriverConfig")
        self._config = config
        self._engine = engine
        self._clock = clock
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="expertkit-transfer-engine",
        )
        self._start_lock = asyncio.Lock()
        self._registration_lock = asyncio.Lock()
        self._ready_task: asyncio.Task[None] | None = None
        self._session_id: str | None = None
        self._actual_backend: str | None = None
        self._generation = uuid.uuid4().hex
        self._registrations: dict[int, tuple[object, int]] = {}
        self._pending_registrations: dict[int, tuple[object, int]] = {}
        self._operations: set[asyncio.Future[Any]] = set()
        self._staging_events: dict[
            asyncio.Future[Any], tuple[Any, Any, asyncio.AbstractEventLoop]
        ] = {}
        self._staging_condition = threading.Condition()
        self._completion_running = False
        self._staging_failure: BaseException | None = None
        self._completion_executor: ThreadPoolExecutor | None = None
        self._quarantine_reason: str | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def device(self) -> torch.device:
        return self._config.device

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("Transfer Engine runtime has not started")
        return self._session_id

    @property
    def backend(self) -> str:
        return self._actual_backend or self._config.protocol

    @property
    def generation(self) -> str:
        return self._generation

    @property
    def client_max_in_flight(self) -> int | None:
        return self._config.client_max_in_flight

    @property
    def registered_bytes(self) -> int:
        return sum(length for _, length in self._registrations.values())

    async def start(self) -> None:
        async with self._start_lock:
            if self._quarantine_reason is not None:
                raise RuntimeError(
                    f"Transfer Engine runtime is quarantined: {self._quarantine_reason}"
                )
            if self._closing:
                raise RuntimeError("Transfer Engine runtime is closing")
            if self._ready_task is None:
                self._ready_task = asyncio.create_task(
                    self._initialize(),
                    name=f"transfer-engine-{self._config.segment_name}-initialize",
                )
            task = self._ready_task
        await asyncio.shield(task)

    async def register_region(
        self,
        region: MemoryRegion,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        await self.start()
        self._validate_region_device(region)
        address, length = region.address, region.length
        async with self._registration_lock:
            self.ensure_healthy()
            if address in self._registrations or address in self._pending_registrations:
                raise RuntimeError("Transfer Engine memory region is already registered")
            for base, (_, capacity) in (
                *self._registrations.items(),
                *self._pending_registrations.items(),
            ):
                if address < base + capacity and base < address + length:
                    raise ValueError("Transfer Engine registration overlaps an existing region")
            budget = self._config.max_registered_bytes
            pending_bytes = sum(size for _, size in self._pending_registrations.values())
            if budget is not None and self.registered_bytes + pending_bytes + length > budget:
                raise ValueError("Transfer Engine registered memory budget exceeded")
            if self.device.type == "npu" and (address % (2 << 20) or length % (2 << 20)):
                raise ValueError(
                    "Ascend registered region address and length must be 2 MiB aligned"
                )
            self._pending_registrations[address] = (region.owner, length)

            def commit(result: object) -> None:
                self._require_success(result, "Mooncake memory registration")
                self._pending_registrations.pop(address)
                self._registrations[address] = (region.owner, length)

            try:
                engine = self._require_engine()
                await self._run_native(
                    partial(engine.register_memory, address, length),
                    monotonic_deadline=monotonic_deadline,
                    subject="Mooncake memory registration",
                    commit=commit,
                )
            except BaseException:
                if address in self._pending_registrations:
                    self.quarantine(
                        "Mooncake memory registration did not reach a proven successful state"
                    )
                raise

    async def unregister_region(
        self,
        region: MemoryRegion,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        if self._quarantine_reason is not None:
            # Retain both the native registration and Tensor storage for the
            # remainder of the process: a remote peer may still hold a DMA
            # reference after an ambiguous control-plane failure.
            return
        if self._closing:
            # Process-level close owns every remaining registration and retains
            # the Tensor strongly until its native deregistration commits.
            return
        self._validate_region_device(region)
        address = region.address
        try:
            async with self._registration_lock:
                registered = self._registrations.get(address)
                if registered is None or registered[0] is not region.owner:
                    raise RuntimeError("Transfer Engine memory region is not registered")
                engine = self._require_engine()

                def commit(result: object) -> None:
                    self._require_success(result, "Mooncake memory deregistration")
                    self._registrations.pop(address)

                await self._run_native(
                    partial(engine.unregister_memory, address),
                    monotonic_deadline=monotonic_deadline,
                    subject="Mooncake memory deregistration",
                    commit=commit,
                )
        except BaseException:
            registered = self._registrations.get(address)
            if registered is not None and registered[0] is region.owner:
                self.quarantine(
                    "Mooncake memory deregistration did not reach a proven successful state"
                )
            raise

    async def wait_event(
        self,
        event: torch.Event,
        *,
        monotonic_deadline: float,
    ) -> None:
        try:
            await self._run_native(
                event.synchronize,
                # Once the event is recorded, synchronizing it is a memory-lifetime
                # barrier and cannot be skipped merely because the request expired.
                monotonic_deadline=math.inf,
                subject=f"{self.device.type.upper()} staging copy",
            )
        except asyncio.CancelledError:
            # _run_native raises cancellation only after synchronize completed.
            raise
        except BaseException as error:
            self.quarantine(
                f"{self.device.type.upper()} staging-copy completion could not be proven"
            )
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic=(
                    f"{self.device.type.upper()} staging-copy completion could not be proven"
                ),
            ) from error
        if not math.isinf(monotonic_deadline) and self._clock() >= monotonic_deadline:
            raise _deadline_error(
                f"deadline expired during {self.device.type.upper()} staging copy"
            )

    async def acquire_remote_writes(
        self,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Acquire successful native writes before local accelerator consumption."""

        if self.device.type == "cpu":
            return
        try:
            stream = (
                torch.get_device_module(self.device).current_stream(self.device)
                if self.device.type == "npu"
                else None
            )
            await self._run_native(
                partial(self._acquire_native, stream),
                # A recorded remote-write completion still needs an acquire even
                # after the request deadline. Complete and validate the barrier
                # inside this native operation before cancellation can escape.
                monotonic_deadline=math.inf,
                subject=f"{self.device.type.upper()} remote-write acquire fence",
            )
        except asyncio.CancelledError:
            # The native barrier completed before _run_native raised cancellation.
            raise
        except BaseException as error:
            self.quarantine(
                f"{self.device.type.upper()} remote-write visibility could not be proven"
            )
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic=(
                    f"{self.device.type.upper()} remote-write visibility could not be proven"
                ),
            ) from error
        if not math.isinf(monotonic_deadline) and self._clock() >= monotonic_deadline:
            raise _deadline_error(
                f"deadline expired during {self.device.type.upper()} remote-write acquire fence"
            )

    async def read(
        self, target_session: str, slices: Sequence[MemorySlice], *, monotonic_deadline: float
    ) -> None:
        await self._batch_transfer(
            "read", target_session, slices, monotonic_deadline=monotonic_deadline
        )

    async def write(
        self, target_session: str, slices: Sequence[MemorySlice], *, monotonic_deadline: float
    ) -> None:
        await self._batch_transfer(
            "write", target_session, slices, monotonic_deadline=monotonic_deadline
        )

    async def invalidate_remote_session(
        self,
        target_session: str,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        if not target_session:
            raise ValueError("target_session must not be empty")
        self.ensure_healthy()
        engine = self._require_engine()
        invalidate, subject = _remote_invalidation_api(engine, self.backend)
        result = await self._run_native(
            partial(invalidate, target_session),
            monotonic_deadline=monotonic_deadline,
            subject=subject,
        )
        self._require_success(result, subject)

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    def ensure_healthy(self) -> None:
        if self._quarantine_reason is not None:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic=(
                    "Transfer Engine runtime is quarantined after an ambiguous "
                    "remote DMA state; restart the process"
                ),
            )
        if self._closing:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic="Transfer Engine runtime is closing",
            )

    def quarantine(self, diagnostic: str) -> None:
        """Retain every registration after an unprovable remote DMA state."""

        if not isinstance(diagnostic, str) or not diagnostic:
            raise ValueError("quarantine diagnostic must not be empty")
        if self._quarantine_reason is None:
            self._quarantine_reason = diagnostic
            _QUARANTINED_RUNTIMES.append(self)

    async def _initialize(self) -> None:
        if self._config.protocol == "ascend_direct":
            for name, allowed in (
                ("ASCEND_USE_ASYNC_TRANSFER", "0"),
                ("ASCEND_BUFFER_POOL", "0:0"),
                ("ASCEND_USE_SHORT_CONNECTION", "0"),
            ):
                if os.environ.get(name, allowed) != allowed:
                    raise RuntimeError(f"EK Ascend Direct requires {name}={allowed}")
        enabled_tent_variables = tuple(
            name for name in ("MC_USE_TENT", "MC_USE_TEV1") if name in os.environ
        )
        if enabled_tent_variables:
            joined = ", ".join(enabled_tent_variables)
            raise RuntimeError(
                "Expert Kit's validated Transfer Engine path does not support TENT; "
                f"unset {joined} (even values such as '0' enable it)"
            )
        if self._engine is None:
            try:
                from mooncake import engine as mooncake_engine  # type: ignore[import-not-found]
            except (ImportError, OSError) as error:
                raise RuntimeError(
                    "Mooncake Transfer Engine is unavailable; install the "
                    "expertkit-transport[transfer-engine] extra"
                ) from error
            _validate_native_capabilities(mooncake_engine, self._config)
            self._engine = mooncake_engine.TransferEngine()
        engine = self._require_engine()
        backend_query = _configured_backend_query(engine, self._config.protocol)
        if self._config.protocol in {"nvlink_intra", "rdma", "ascend_direct"}:
            _remote_invalidation_api(engine, self._config.protocol)
        result = await self._run_native(
            partial(
                engine.initialize,
                self._config.segment_name,
                self._config.metadata_server,
                self._config.protocol,
                self._config.device_name,
            ),
            monotonic_deadline=math.inf,
            subject="Mooncake initialization",
            allow_during_close=True,
        )
        self._require_success(result, "Mooncake initialization")
        if backend_query is not None:
            actual_backend = await self._run_native(
                backend_query,
                monotonic_deadline=math.inf,
                subject="Mooncake configured backend discovery",
                allow_during_close=True,
            )
            self._actual_backend = _validate_configured_backend(
                actual_backend,
                self._config.protocol,
            )
        rpc_port = await self._run_native(
            engine.get_rpc_port,
            monotonic_deadline=math.inf,
            subject="Mooncake RPC port discovery",
            allow_during_close=True,
        )
        if isinstance(rpc_port, bool) or not isinstance(rpc_port, int) or rpc_port <= 0:
            raise RuntimeError("Mooncake returned an invalid RPC port")
        if self._config.metadata_server.upper() == "P2PHANDSHAKE":
            self._session_id = _p2p_session_id(self._config.segment_name, rpc_port)
        else:
            self._session_id = self._config.segment_name

    def _acquire_native(self, stream: Any = None) -> None:
        if self.device.type == "cpu":
            return
        if self.device.type == "npu" and self._config.ascend_receive_fence == "stream":
            # The forced adapter reports completion only after ADXL TransferSync
            # succeeds. READ data, or peer WRITE followed by its terminal control
            # response, is DMA-terminal before this local consumer runs. Drain
            # its stream without waiting for unrelated execution-slot streams.
            if stream is None:
                stream = torch.get_device_module(self.device).current_stream(self.device)
            stream.synchronize()
            return
        name = "acquire_ascend_writes" if self.device.type == "npu" else "flush_gpudirect_writes"
        flush = getattr(self._require_engine(), name, None)
        if not callable(flush):
            raise RuntimeError(f"the Mooncake wheel lacks required receive fence {name}")
        self._require_success(flush(), f"{self.device.type.upper()} remote-write acquire fence")

    async def run_device_operation(
        self,
        call: Callable[[], _T],
        *,
        monotonic_deadline: float,
        subject: str,
        acquire_writes: bool = False,
        stream: Any = None,
        completion_event: Any = None,
    ) -> _T:
        """Submit local work, then retain its owners until the recorded event completes.

        With a completion event, the callback only enqueues work and records the
        event. Waiting must not occupy a native submission thread. Callers without
        an event retain the synchronous callback contract.
        """
        self.ensure_healthy()

        def operation() -> _T:
            if acquire_writes:
                try:
                    self._acquire_native(stream)
                except BaseException:
                    self.quarantine("remote-write visibility could not be proven")
                    raise
            return call()

        if completion_event is not None and not callable(getattr(completion_event, "query", None)):
            raise TypeError("staging completion event must implement query()")
        return await self._run_native(
            operation,
            monotonic_deadline=monotonic_deadline,
            subject=subject,
            completion_event=completion_event,
        )

    def _staging_submitted(
        self,
        event: Any,
        waiter: asyncio.Future[Any],
        loop: asyncio.AbstractEventLoop,
        submitted: ConcurrentFuture[Any],
    ) -> None:
        try:
            result = submitted.result()
        except BaseException as error:
            self.quarantine("local staging submission failed before completion was proven")
            loop.call_soon_threadsafe(self._finish_staging_batch, ((waiter, None),), error)
            return
        with self._staging_condition:
            if self._staging_failure is not None:
                loop.call_soon_threadsafe(
                    self._finish_staging_batch, ((waiter, None),), self._staging_failure
                )
                return
            self._staging_events[waiter] = (event, result, loop)
            if not self._completion_running:
                self._completion_running = True
                self._completion_executor.submit(self._poll_staging_events)
            self._staging_condition.notify()

    def _query_staging_events(self, events: tuple[Any, ...]) -> tuple[bool, ...]:
        def query() -> tuple[bool, ...]:
            return tuple(event.query() for event in events)

        return self._invoke_native(query)

    def _finish_staging_batch(
        self,
        completed: tuple[tuple[asyncio.Future[Any], Any], ...],
        error: BaseException | None = None,
    ) -> None:
        for waiter, result in completed:
            if waiter.cancelled():
                self.quarantine("staging completion waiter was cancelled before terminal state")
            elif not waiter.done():
                if error is None:
                    waiter.set_result(result)
                else:
                    waiter.set_exception(error)

    def _poll_staging_events(self) -> None:
        # Query and coalesce completions entirely on one native thread. Only
        # terminal batches wake the event loop; pending queries do not round-trip
        # through asyncio or allocate a Task for each staging operation.
        try:
            while True:
                with self._staging_condition:
                    if not self._staging_events:
                        self._completion_running = False
                        return
                    pending = tuple(self._staging_events.items())
                ready = self._query_staging_events(tuple(entry[0] for _, entry in pending))
                if not all(isinstance(complete, bool) for complete in ready):
                    raise TypeError("staging event query() must return a Boolean")
                batches: dict[asyncio.AbstractEventLoop, list[tuple[asyncio.Future[Any], Any]]] = {}
                with self._staging_condition:
                    for (waiter, (_, result, loop)), complete in zip(pending, ready, strict=True):
                        if complete:
                            self._staging_events.pop(waiter)
                            batches.setdefault(loop, []).append((waiter, result))
                for loop, completed in batches.items():
                    loop.call_soon_threadsafe(self._finish_staging_batch, tuple(completed))
                if not batches:
                    # Briefly release the GIL when the device is still busy.
                    # New submissions wake this wait immediately.
                    with self._staging_condition:
                        self._staging_condition.wait(timeout=0.0001)
        except BaseException as error:
            self.quarantine("local staging event query failed before completion was proven")
            with self._staging_condition:
                self._staging_failure = error
                pending = tuple(self._staging_events.items())
                self._staging_events.clear()
                self._completion_running = False
            batches = {}
            for waiter, (_, result, loop) in pending:
                batches.setdefault(loop, []).append((waiter, result))
            for loop, completed in batches.items():
                loop.call_soon_threadsafe(self._finish_staging_batch, tuple(completed), error)

    async def _batch_transfer(
        self,
        operation: str,
        target_session: str,
        slices: Sequence[MemorySlice],
        *,
        monotonic_deadline: float,
        acquire_writes: bool = False,
    ) -> None:
        self.ensure_healthy()
        await self.start()
        if not target_session:
            raise ValueError("target_session must not be empty")
        entries = tuple(slices)
        if not entries:
            raise ValueError("Transfer Engine transfer batch must not be empty")
        local: list[int] = []
        remote: list[int] = []
        sizes: list[int] = []
        for entry in entries:
            self._validate_region_device(entry.local)
            local_address = entry.local.address + entry.local_offset
            self._require_registered(local_address, entry.length)
            local.append(local_address)
            remote.append(entry.remote_address)
            sizes.append(entry.length)
        engine = self._require_engine()
        native = (
            engine.batch_transfer_sync_read
            if operation == "read"
            else engine.batch_transfer_sync_write
        )
        stream = (
            torch.get_device_module(self.device).current_stream(self.device)
            if acquire_writes and self.device.type == "npu"
            else None
        )

        def transfer() -> object:
            result = native(
                target_session, local, list(remote), list(sizes), self._config.transport_hint
            )
            if acquire_writes and result in (None, 0):
                try:
                    self._acquire_native(stream)
                except BaseException:
                    # A status-derived TransportError must quarantine as well;
                    # it is not an ordinary deadline/cancellation from the wait.
                    self.quarantine("remote-write visibility could not be proven")
                    raise
            return result

        try:
            result = await self._run_native(
                transfer,
                monotonic_deadline=monotonic_deadline,
                subject=f"Mooncake batch {operation}" + (" and acquire" if acquire_writes else ""),
            )
            if self._config.protocol == "ascend_direct" and result not in (None, 0):
                raise RuntimeError("Ascend transfer failed without proven DMA termination")
        except (asyncio.CancelledError, TransportError):
            # The safety-patched binding returns cancellation/deadline only
            # after the submitted batch reaches a native terminal state.
            raise
        except BaseException:
            # An unexpected binding exception does not carry the patched
            # terminal-state contract.  Keep every registered slab alive so
            # neither side can reuse storage that native DMA may still touch.
            self.quarantine(
                f"Mooncake batch {operation} raised before terminal DMA state was proven"
            )
            raise
        self._require_success(result, f"Mooncake batch {operation}")

    async def _run_native(
        self,
        call: Callable[[], _T],
        *,
        monotonic_deadline: float,
        subject: str,
        commit: Callable[[_T], None] | None = None,
        allow_during_close: bool = False,
        completion_event: Any = None,
    ) -> _T:
        if self._closing and not allow_during_close:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic="Transfer Engine runtime is closing",
            )
        if not math.isinf(monotonic_deadline) and not math.isfinite(monotonic_deadline):
            raise ValueError("monotonic_deadline must be finite or positive infinity")
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error(f"deadline expired before {subject}")
        loop = asyncio.get_running_loop()
        if completion_event is None:
            future = loop.run_in_executor(self._executor, self._invoke_native, call)
        else:
            # This single terminal future spans submission and device completion.
            # Shielding/draining it retains the caller's callback and Tensor owners
            # without introducing a second coroutine or event-loop wake per phase.
            future = loop.create_future()
            if self._completion_executor is None:
                self._completion_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="expertkit-staging-completion"
                )
            submitted = self._executor.submit(self._invoke_native, call)
            submitted.add_done_callback(
                partial(self._staging_submitted, completion_event, future, loop)
            )
        self._operations.add(future)
        cancelled = False
        expired = False
        try:
            try:
                if math.isinf(remaining):
                    result = await asyncio.shield(future)
                else:
                    async with asyncio.timeout(remaining):
                        result = await asyncio.shield(future)
            except TimeoutError:
                expired = True
                # The safety-patched native call does not return until the DMA
                # reaches an engine terminal state. Keep consuming repeated
                # cancellation while the registered slab is still leased.
                result, later_cancelled = await self._await_native_terminal(future)
                cancelled |= later_cancelled
            except asyncio.CancelledError:
                cancelled = True
                result, later_cancelled = await self._await_native_terminal(future)
                cancelled |= later_cancelled
        finally:
            if future.done():
                self._operations.discard(future)
            else:
                future.add_done_callback(self._operations.discard)
        # Native registration mutations must update Python ownership before a
        # deferred caller cancellation/deadline can escape this coroutine.
        if commit is not None:
            commit(result)
        if cancelled:
            raise asyncio.CancelledError
        if expired or self._clock() >= monotonic_deadline:
            raise _deadline_error(f"deadline expired during {subject}")
        return result

    async def _await_native_terminal(
        self,
        future: asyncio.Future[_T],
    ) -> tuple[_T, bool]:
        cancelled = False
        while True:
            try:
                return await asyncio.shield(future), cancelled
            except asyncio.CancelledError:
                if future.cancelled():
                    # Event-loop shutdown can cancel an internal staging Task.
                    # An already-cancelled future will never reach completion;
                    # retain storage rather than spinning in the drain loop.
                    self.quarantine("native completion waiter was cancelled before terminal state")
                    raise
                cancelled = True

    def _invoke_native(self, call: Callable[[], _T]) -> _T:
        if self.device.type != "cpu":
            torch.get_device_module(self.device).set_device(self.device)
        return call()

    async def _close(self) -> None:
        if self._quarantine_reason is not None:
            return
        ready = self._ready_task
        if ready is not None:
            with suppress(BaseException):
                await asyncio.shield(ready)
        operations = tuple(self._operations)
        if operations:
            await asyncio.gather(*(asyncio.shield(op) for op in operations), return_exceptions=True)
        if self._quarantine_reason is not None:
            return
        async with self._registration_lock:
            engine = self._engine
            if engine is not None:
                for address in tuple(reversed(self._registrations)):
                    try:
                        result = await self._run_native(
                            partial(engine.unregister_memory, address),
                            monotonic_deadline=math.inf,
                            subject="Mooncake shutdown memory deregistration",
                            allow_during_close=True,
                        )
                        self._require_success(result, "Mooncake shutdown memory deregistration")
                        self._registrations.pop(address)
                    except BaseException:
                        self.quarantine(
                            "Mooncake shutdown memory deregistration did not reach a "
                            "proven successful state"
                        )
                        raise
                close = getattr(engine, "close", None)
                if callable(close):
                    await self._run_native(
                        close,
                        monotonic_deadline=math.inf,
                        subject="Mooncake shutdown",
                        allow_during_close=True,
                    )
        self._session_id = None
        self._actual_backend = None
        self._engine = None
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._completion_executor is not None:
            self._completion_executor.shutdown(wait=True, cancel_futures=False)

    def _validate_region_device(self, region: MemoryRegion) -> None:
        if region.device != str(self.device):
            raise ValueError("Transfer Engine region device does not match the runtime")

    def _require_registered(self, address: int, length: int) -> None:
        for base, (_tensor, capacity) in self._registrations.items():
            if base <= address and address + length <= base + capacity:
                return
        raise RuntimeError("Transfer Engine local Tensor is outside registered memory")

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("Transfer Engine native engine is unavailable")
        return self._engine

    @staticmethod
    def _require_success(result: object, subject: str) -> None:
        if result is None or result == 0:
            return
        raise TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=True,
            diagnostic=f"{subject} failed with native status {result}",
        )


__all__ = [
    "MooncakeDriverConfig",
    "MooncakeMemoryTransport",
]
