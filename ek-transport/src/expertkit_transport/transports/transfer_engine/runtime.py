"""Process-wide Mooncake Transfer Engine adapter.

Mooncake's Python API is synchronous.  This module confines every native call
to one bounded executor and does not release a caller's registered memory until
an in-flight native operation reaches a terminal state.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, TypeVar, runtime_checkable
from urllib.parse import urlsplit

import torch

from expertkit_transport.errors import TransportError, TransportErrorCode

_T = TypeVar("_T")
_SUPPORTED_PROTOCOLS = frozenset({"tcp", "rdma", "nvlink", "nvlink_intra"})
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
    config: TransferEngineRuntimeConfig,
) -> None:
    """Fail before engine construction when a backend lacks EK safety contracts."""

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
    if backend not in {"nvlink_intra", "rdma"}:
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
    else:
        raise RuntimeError(
            "drained remote-session invalidation requires a forced intra-NVLink or RDMA backend"
        )
    invalidate = getattr(engine, name, None)
    if not callable(invalidate):
        raise RuntimeError(f"the Mooncake wheel lacks {subject.lower()}")
    return invalidate, subject


@dataclass(frozen=True, slots=True)
class TransferEngineRuntimeConfig:
    """Configure one Mooncake engine shared by every connection in a process."""

    segment_name: str
    metadata_server: str
    protocol: str
    device: torch.device | str
    device_name: str = ""
    max_workers: int = 2
    transport_hint: str = ""
    enable_experimental_rdma: bool = False

    def __post_init__(self) -> None:
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
            raise ValueError("protocol must be tcp, rdma, nvlink, or nvlink_intra")
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
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Transfer Engine runtime device must be CPU or CUDA")
        if device.type == "cuda" and device.index is None:
            raise ValueError("Transfer Engine runtime requires an indexed CUDA device")
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
        object.__setattr__(self, "device", device)


@runtime_checkable
class TransferEngineRuntimeProtocol(Protocol):
    """Injectable process-level contract used by clients and Worker receivers."""

    @property
    def device(self) -> torch.device: ...

    @property
    def session_id(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def generation(self) -> str: ...

    async def start(self) -> None: ...

    async def register_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None: ...

    async def unregister_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None: ...

    async def wait_event(
        self,
        event: torch.cuda.Event,
        *,
        monotonic_deadline: float,
    ) -> None: ...

    async def acquire_remote_writes(
        self,
        *,
        monotonic_deadline: float,
    ) -> None: ...

    async def batch_read(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None: ...

    async def batch_write(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None: ...

    async def invalidate_remote_session(
        self,
        target_session: str,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None: ...

    async def close(self) -> None: ...

    def ensure_healthy(self) -> None: ...

    def quarantine(self, diagnostic: str) -> None: ...


class TransferEngineRuntime(TransferEngineRuntimeProtocol):
    """Adapt ``mooncake.engine.TransferEngine`` without importing it eagerly."""

    def __init__(
        self,
        config: TransferEngineRuntimeConfig,
        *,
        engine: object | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, TransferEngineRuntimeConfig):
            raise TypeError("config must be a TransferEngineRuntimeConfig")
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
        self._registrations: dict[int, tuple[torch.Tensor, int]] = {}
        self._pending_registrations: dict[int, tuple[torch.Tensor, int]] = {}
        self._operations: set[asyncio.Future[Any]] = set()
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

    async def register_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        await self.start()
        address, length = self._tensor_region(tensor)
        async with self._registration_lock:
            self.ensure_healthy()
            if address in self._registrations or address in self._pending_registrations:
                raise RuntimeError("Transfer Engine memory region is already registered")
            self._pending_registrations[address] = (tensor, length)

            def commit(result: object) -> None:
                self._require_success(result, "Mooncake memory registration")
                self._pending_registrations.pop(address)
                self._registrations[address] = (tensor, length)

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

    async def unregister_tensor(
        self,
        tensor: torch.Tensor,
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
        address, _ = self._tensor_region(tensor)
        try:
            async with self._registration_lock:
                registered = self._registrations.get(address)
                if registered is None or registered[0] is not tensor:
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
            if registered is not None and registered[0] is tensor:
                self.quarantine(
                    "Mooncake memory deregistration did not reach a proven successful state"
                )
            raise

    async def wait_event(
        self,
        event: torch.cuda.Event,
        *,
        monotonic_deadline: float,
    ) -> None:
        try:
            await self._run_native(
                event.synchronize,
                # Once the event is recorded, synchronizing it is a memory-lifetime
                # barrier and cannot be skipped merely because the request expired.
                monotonic_deadline=math.inf,
                subject="CUDA staging copy",
            )
        except asyncio.CancelledError:
            # _run_native raises cancellation only after synchronize completed.
            raise
        except BaseException as error:
            self.quarantine("CUDA staging-copy completion could not be proven")
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic="CUDA staging-copy completion could not be proven",
            ) from error
        if not math.isinf(monotonic_deadline) and self._clock() >= monotonic_deadline:
            raise _deadline_error("deadline expired during CUDA staging copy")

    async def acquire_remote_writes(
        self,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Make completed GPUDirect RDMA writes visible to local CUDA work."""

        if self.device.type != "cuda":
            return
        try:
            engine = self._require_engine()
            flush = getattr(engine, "flush_gpudirect_writes", None)
            if not callable(flush):
                raise RuntimeError("the Mooncake wheel lacks the required GPUDirect acquire fence")
            await self._run_native(
                flush,
                # A recorded remote-write completion still needs an acquire even
                # after the request deadline. Validate the native result in the
                # commit hook before deferred cancellation can escape.
                monotonic_deadline=math.inf,
                subject="GPUDirect RDMA acquire fence",
                commit=lambda result: self._require_success(
                    result,
                    "GPUDirect RDMA acquire fence",
                ),
            )
        except asyncio.CancelledError:
            # The commit hook ran before _run_native re-raised cancellation.
            raise
        except BaseException as error:
            self.quarantine("GPUDirect remote-write visibility could not be proven")
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic="GPUDirect remote-write visibility could not be proven",
            ) from error
        if not math.isinf(monotonic_deadline) and self._clock() >= monotonic_deadline:
            raise _deadline_error("deadline expired during GPUDirect RDMA acquire fence")

    async def batch_read(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None:
        await self._batch_transfer(
            "read",
            target_session,
            local_tensors,
            remote_addresses,
            lengths,
            monotonic_deadline=monotonic_deadline,
        )

    async def batch_write(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None:
        await self._batch_transfer(
            "write",
            target_session,
            local_tensors,
            remote_addresses,
            lengths,
            monotonic_deadline=monotonic_deadline,
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
        if self._config.protocol in {"nvlink_intra", "rdma"}:
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

    async def _batch_transfer(
        self,
        operation: str,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None:
        self.ensure_healthy()
        await self.start()
        if not target_session:
            raise ValueError("target_session must not be empty")
        tensors = tuple(local_tensors)
        remote = tuple(remote_addresses)
        sizes = tuple(lengths)
        if not tensors or len(tensors) != len(remote) or len(tensors) != len(sizes):
            raise ValueError("Transfer Engine batch vectors must have the same nonzero length")
        local: list[int] = []
        for tensor, address, length in zip(tensors, remote, sizes, strict=True):
            if isinstance(address, bool) or not isinstance(address, int) or address <= 0:
                raise ValueError("Transfer Engine remote address must be positive")
            if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                raise ValueError("Transfer Engine transfer length must be positive")
            local_address, tensor_bytes = self._tensor_region(tensor)
            if length > tensor_bytes:
                raise ValueError("Transfer Engine transfer exceeds a local Tensor view")
            self._require_registered(local_address, length)
            local.append(local_address)
        engine = self._require_engine()
        native = (
            engine.batch_transfer_sync_read
            if operation == "read"
            else engine.batch_transfer_sync_write
        )
        try:
            result = await self._run_native(
                partial(
                    native,
                    target_session,
                    local,
                    list(remote),
                    list(sizes),
                    self._config.transport_hint,
                ),
                monotonic_deadline=monotonic_deadline,
                subject=f"Mooncake batch {operation}",
            )
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
        future = loop.run_in_executor(self._executor, self._invoke_native, call)
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
        if expired:
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
                cancelled = True

    def _invoke_native(self, call: Callable[[], _T]) -> _T:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
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

    def _tensor_region(self, tensor: torch.Tensor) -> tuple[int, int]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Transfer Engine memory must be a Tensor")
        if tensor.device != self.device:
            raise ValueError("Transfer Engine Tensor device does not match the runtime")
        if not tensor.is_contiguous():
            raise ValueError("Transfer Engine Tensor must be contiguous")
        length = tensor.numel() * tensor.element_size()
        if length <= 0:
            raise ValueError("Transfer Engine Tensor must not be empty")
        return tensor.data_ptr(), length

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
    "TransferEngineRuntime",
    "TransferEngineRuntimeConfig",
    "TransferEngineRuntimeProtocol",
]
