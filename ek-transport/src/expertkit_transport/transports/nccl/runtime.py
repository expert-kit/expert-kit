"""Process-wide ordered NCCL point-to-point runtime.

The runtime, not an individual Worker connection, owns ProcessGroupNCCL.  All
connections in one process must share the same instance so P2P operations are
posted in one deterministic order.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any, Protocol, runtime_checkable

import torch

from expertkit_transport.errors import TransportError, TransportErrorCode


def _deadline_error(diagnostic: str) -> TransportError:
    return TransportError(
        TransportErrorCode.DEADLINE_EXCEEDED,
        retryable=False,
        diagnostic=diagnostic,
    )


@dataclass(frozen=True, slots=True)
class NcclRuntimeConfig:
    """Describe one static NCCL world joined by exactly one CUDA process."""

    rank: int
    world_size: int
    rendezvous_endpoint: str
    group_name: str
    device: torch.device | str
    init_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size < 2
        ):
            raise ValueError("world_size must be an integer of at least two")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 0 <= self.rank < self.world_size
        ):
            raise ValueError("rank must be within the static NCCL world")
        if not self.rendezvous_endpoint:
            raise ValueError("rendezvous_endpoint must not be empty")
        if not self.group_name:
            raise ValueError("group_name must not be empty")
        if not math.isfinite(self.init_timeout_seconds) or self.init_timeout_seconds <= 0:
            raise ValueError("init_timeout_seconds must be a finite positive number")
        device = torch.device(self.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("the production NCCL runtime requires an indexed CUDA device")
        object.__setattr__(self, "device", device)


@runtime_checkable
class NcclWorkerExchangeProtocol(Protocol):
    """Keep one peer lane ordered until its mandatory output has been sent."""

    async def send_output(
        self,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Send the matching output and release the peer lane."""

    async def abort(self, *, monotonic_deadline: float) -> None:
        """Send the preallocated dummy output and release the peer lane."""


@runtime_checkable
class NcclRuntimeProtocol(Protocol):
    """Injectable process-level contract used by NCCL client and receiver tests."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def group_name(self) -> str: ...

    @property
    def rendezvous_endpoint(self) -> str: ...

    @property
    def device(self) -> torch.device: ...

    async def start(self) -> None:
        """Start ProcessGroup initialization in the background without waiting."""

    async def wait_ready(self, *, monotonic_deadline: float) -> None:
        """Wait for every static rank to join or for the caller deadline."""

    async def exchange(
        self,
        peer_rank: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Send three inputs and receive exactly one matching output."""

    async def receive_inputs(
        self,
        peer_rank: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        dummy_output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> NcclWorkerExchangeProtocol:
        """Receive three inputs and return a lease that must send one output."""

    async def close(self) -> None:
        """Release the owned ProcessGroup and progress resources."""


class _NcclWorkerExchange(NcclWorkerExchangeProtocol):
    def __init__(
        self,
        runtime: NcclRuntime,
        peer_rank: int,
        peer_lock: asyncio.Lock,
        dummy_output: torch.Tensor,
    ) -> None:
        self._runtime = runtime
        self._peer_rank = peer_rank
        self._peer_lock: asyncio.Lock | None = peer_lock
        self._dummy_output = dummy_output

    async def send_output(
        self,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        await self._finish(output, monotonic_deadline)

    async def abort(self, *, monotonic_deadline: float) -> None:
        await self._finish(self._dummy_output, monotonic_deadline)

    async def _finish(self, output: torch.Tensor, monotonic_deadline: float) -> None:
        peer_lock = self._peer_lock
        if peer_lock is None:
            raise RuntimeError("NCCL Worker exchange has already finished")
        try:
            await self._runtime._send_output(
                self._peer_rank,
                output,
                monotonic_deadline=monotonic_deadline,
            )
        finally:
            self._peer_lock = None
            peer_lock.release()


class NcclRuntime(NcclRuntimeProtocol):
    """Own one static ProcessGroupNCCL and its ordered progress engine."""

    def __init__(
        self,
        config: NcclRuntimeConfig,
        *,
        process_group: object | None = None,
        distributed: object | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, NcclRuntimeConfig):
            raise TypeError("config must be an NcclRuntimeConfig")
        self._config = config
        self._process_group = process_group
        self._distributed = distributed
        self._clock = clock
        self._owns_process_group = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="expertkit-nccl-progress",
        )
        self._ready_task: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()
        self._post_lock = asyncio.Lock()
        self._peer_locks: dict[int, asyncio.Lock] = {}
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def rank(self) -> int:
        return self._config.rank

    @property
    def world_size(self) -> int:
        return self._config.world_size

    @property
    def group_name(self) -> str:
        return self._config.group_name

    @property
    def rendezvous_endpoint(self) -> str:
        endpoint = self._config.rendezvous_endpoint
        return endpoint if "://" in endpoint else f"tcp://{endpoint}"

    @property
    def device(self) -> torch.device:
        return self._config.device

    async def start(self) -> None:
        """Start the blocking static rendezvous on the private progress thread."""

        async with self._start_lock:
            if self._closing:
                raise RuntimeError("NCCL runtime is closing")
            if self._ready_task is None:
                self._ready_task = asyncio.create_task(
                    self._initialize(),
                    name=f"nccl-runtime-rank-{self.rank}-initialize",
                )

    async def wait_ready(self, *, monotonic_deadline: float) -> None:
        await self.start()
        ready_task = self._ready_task
        assert ready_task is not None
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error("deadline expired while waiting for the NCCL world")
        try:
            if math.isinf(remaining):
                await asyncio.shield(ready_task)
            else:
                async with asyncio.timeout(remaining):
                    await asyncio.shield(ready_task)
        except TimeoutError as error:
            raise _deadline_error("deadline expired while waiting for the NCCL world") from error
        except TransportError:
            raise
        except Exception as error:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic=f"NCCL runtime initialization failed: {error}",
            ) from error

    async def exchange(
        self,
        peer_rank: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Post the fixed input-send and output-receive phases in peer FIFO order."""

        self._validate_peer(peer_rank)
        self._validate_tensor("hidden_states", hidden_states)
        self._validate_tensor("expert_ids", expert_ids)
        self._validate_tensor("routing_weights", routing_weights)
        self._validate_tensor("output", output)
        await self.wait_ready(monotonic_deadline=monotonic_deadline)
        peer_lock = self._peer_locks.setdefault(peer_rank, asyncio.Lock())
        cancelled = await self._acquire_peer(
            peer_lock,
            monotonic_deadline,
            must_match=True,
        )
        try:
            input_ops = self._client_input_ops(
                peer_rank,
                hidden_states,
                expert_ids,
                routing_weights,
            )
            cancelled |= await self._run_ops(
                input_ops,
                monotonic_deadline,
                "NCCL request input send",
                must_match=True,
                propagate_cancel=False,
                propagate_deadline=False,
            )
            output_op = self._client_output_op(peer_rank, output)
            cancelled |= await self._run_ops(
                (output_op,),
                monotonic_deadline,
                "NCCL request output receive",
                must_match=True,
                propagate_cancel=False,
                propagate_deadline=False,
            )
        finally:
            peer_lock.release()
        if cancelled:
            raise asyncio.CancelledError
        if self._clock() >= monotonic_deadline:
            raise _deadline_error("deadline expired during NCCL request exchange")

    async def receive_inputs(
        self,
        peer_rank: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        dummy_output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> NcclWorkerExchangeProtocol:
        """Receive the mandatory inputs while retaining the peer FIFO lane."""

        self._validate_peer(peer_rank)
        for name, tensor in (
            ("hidden_states", hidden_states),
            ("expert_ids", expert_ids),
            ("routing_weights", routing_weights),
            ("dummy_output", dummy_output),
        ):
            self._validate_tensor(name, tensor)
        await self.wait_ready(monotonic_deadline=monotonic_deadline)
        peer_lock = self._peer_locks.setdefault(peer_rank, asyncio.Lock())
        await self._acquire_peer(peer_lock, monotonic_deadline, must_match=True)
        try:
            ops = self._worker_receive_ops(
                peer_rank,
                hidden_states,
                expert_ids,
                routing_weights,
            )
            await self._run_ops(
                ops,
                monotonic_deadline,
                "NCCL input receive",
                must_match=True,
                propagate_cancel=False,
                propagate_deadline=False,
            )
        except BaseException:
            peer_lock.release()
            raise
        return _NcclWorkerExchange(self, peer_rank, peer_lock, dummy_output)

    async def close(self) -> None:
        if self._close_task is None:
            # Publish the gate before cleanup is scheduled so no same-tick
            # caller can start a new untagged operation behind shutdown.
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self._closing = True
        ready_task = self._ready_task
        if ready_task is not None:
            with suppress(BaseException):
                await asyncio.shield(ready_task)
        loop = asyncio.get_running_loop()
        if self._owns_process_group and self._process_group is not None:
            distributed = self._require_distributed()
            process_group = self._process_group
            await loop.run_in_executor(
                self._executor,
                self._abort_and_destroy_process_group,
                distributed,
                process_group,
            )
            self._process_group = None
        await loop.run_in_executor(
            None,
            partial(self._executor.shutdown, wait=True, cancel_futures=True),
        )

    def _abort_and_destroy_process_group(
        self,
        distributed: Any,
        process_group: object,
    ) -> None:
        """Release an owned static world without requiring symmetric peer exit."""

        torch.cuda.set_device(self.device)
        abort = getattr(process_group, "abort", None)
        if not callable(abort):
            raise RuntimeError("the NCCL process group does not support abort")
        # ProcessGroupNCCL.shutdown(), used by destroy_process_group(), is the
        # graceful path and may wait indefinitely when a long-lived Worker rank
        # remains online after its Frontend disconnects.  Abort first so the
        # subsequent destroy only unregisters the local WORLD state.
        abort()
        distributed.destroy_process_group(process_group)

    async def _initialize(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._initialize_blocking)

    def _initialize_blocking(self) -> None:
        distributed = self._require_distributed()
        torch.cuda.set_device(self.device)
        if self._process_group is None:
            if distributed.is_initialized():
                self._process_group = distributed.group.WORLD
            else:
                distributed.init_process_group(
                    backend="nccl",
                    init_method=self._init_method(),
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=timedelta(seconds=self._config.init_timeout_seconds),
                    device_id=self.device,
                )
                self._process_group = distributed.group.WORLD
                self._owns_process_group = True
        actual_rank = int(distributed.get_rank(self._process_group))
        actual_world = int(distributed.get_world_size(self._process_group))
        backend = str(distributed.get_backend(self._process_group)).lower()
        if actual_rank != self.rank or actual_world != self.world_size:
            raise RuntimeError(
                "the injected or existing WORLD process group does not match the "
                "configured EK rank/world_size; an unrelated vLLM or torchrun WORLD "
                "cannot be reused by the NCCL MVP"
            )
        if backend != "nccl":
            raise RuntimeError("the injected process group is not an NCCL backend")
        distributed.barrier(group=self._process_group, device_ids=[self.device.index])

    async def _send_output(
        self,
        peer_rank: int,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        self._validate_tensor("output", output)
        distributed = self._require_distributed()
        operation = distributed.P2POp(
            distributed.isend,
            output,
            peer_rank,
            self._process_group,
        )
        await self._run_ops(
            (operation,),
            monotonic_deadline,
            "NCCL output send",
            must_match=True,
        )

    async def _run_ops(
        self,
        operations: Sequence[object],
        monotonic_deadline: float,
        subject: str,
        *,
        must_match: bool = False,
        propagate_cancel: bool = True,
        propagate_deadline: bool = True,
    ) -> bool:
        if self._closing:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic="NCCL runtime is closing",
            )
        if not must_match and monotonic_deadline <= self._clock():
            raise _deadline_error(f"deadline expired before {subject}")
        loop = asyncio.get_running_loop()
        cancelled = False
        acquire_post_lock = asyncio.create_task(self._post_lock.acquire())
        try:
            await asyncio.shield(acquire_post_lock)
        except asyncio.CancelledError:
            if not must_match:
                acquire_post_lock.cancel()
                raise
            cancelled = True
            await acquire_post_lock
        try:
            posting = loop.run_in_executor(
                self._executor,
                self._post_operations,
                tuple(operations),
            )
            try:
                works = await asyncio.shield(posting)
            except asyncio.CancelledError:
                cancelled = True
                works = await posting
            except Exception as error:
                raise TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=True,
                    diagnostic=f"{subject} could not be posted: {error}",
                ) from error
        finally:
            self._post_lock.release()

        # Once posted, every peer must finish its matching operations.  Shielding
        # here deliberately lets caller cancellation/deadline wait past its SLA
        # rather than corrupt the untagged NCCL operation stream.
        completion = asyncio.create_task(asyncio.to_thread(self._wait_works, tuple(works)))
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError:
            cancelled = True
            await completion
        except Exception as error:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic=f"{subject} failed: {error}",
            ) from error
        if cancelled and propagate_cancel:
            raise asyncio.CancelledError
        if self._clock() >= monotonic_deadline and propagate_deadline:
            raise _deadline_error(f"deadline expired during {subject}")
        return cancelled

    def _post_operations(self, operations: tuple[object, ...]) -> list[object]:
        torch.cuda.set_device(self.device)
        distributed = self._require_distributed()
        return list(distributed.batch_isend_irecv(list(operations)))

    def _wait_works(self, works: tuple[Any, ...]) -> None:
        torch.cuda.set_device(self.device)
        for work in works:
            # ProcessGroupNCCL Work.wait() orders the calling CUDA stream but
            # does not promise a host-blocking completion.  The Worker compute
            # slot and Frontend caller may consume the Tensor on a different
            # stream, so turn that dependency into a precise host-visible
            # completion without synchronizing unrelated device work.
            work.wait()
        completion = torch.cuda.Event()
        completion.record(torch.cuda.current_stream(self.device))
        completion.synchronize()

    def _client_input_ops(
        self,
        peer_rank: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[object, ...]:
        distributed = self._require_distributed()
        group = self._process_group
        return (
            distributed.P2POp(distributed.isend, hidden_states, peer_rank, group),
            distributed.P2POp(distributed.isend, expert_ids, peer_rank, group),
            distributed.P2POp(distributed.isend, routing_weights, peer_rank, group),
        )

    def _client_output_op(self, peer_rank: int, output: torch.Tensor) -> object:
        distributed = self._require_distributed()
        return distributed.P2POp(
            distributed.irecv,
            output,
            peer_rank,
            self._process_group,
        )

    def _worker_receive_ops(
        self,
        peer_rank: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> tuple[object, ...]:
        distributed = self._require_distributed()
        group = self._process_group
        return (
            distributed.P2POp(distributed.irecv, hidden_states, peer_rank, group),
            distributed.P2POp(distributed.irecv, expert_ids, peer_rank, group),
            distributed.P2POp(distributed.irecv, routing_weights, peer_rank, group),
        )

    async def _acquire_peer(
        self,
        lock: asyncio.Lock,
        monotonic_deadline: float,
        *,
        must_match: bool = False,
    ) -> bool:
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0 and not must_match:
            raise _deadline_error("deadline expired before NCCL peer admission")
        acquire = asyncio.create_task(lock.acquire())
        cancelled = False
        try:
            if math.isinf(remaining) or must_match:
                await asyncio.shield(acquire)
            else:
                async with asyncio.timeout(remaining):
                    await asyncio.shield(acquire)
        except asyncio.CancelledError:
            if not must_match:
                acquire.cancel()
                raise
            cancelled = True
            await acquire
        except TimeoutError as error:
            acquire.cancel()
            raise _deadline_error("deadline expired before NCCL peer admission") from error
        return cancelled

    def _validate_peer(self, peer_rank: int) -> None:
        if (
            isinstance(peer_rank, bool)
            or not isinstance(peer_rank, int)
            or not 0 <= peer_rank < self.world_size
            or peer_rank == self.rank
        ):
            raise ValueError("peer_rank must identify another rank in the NCCL world")

    def _validate_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.device != self.device:
            raise ValueError(f"{name} must use the NCCL runtime CUDA device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def _init_method(self) -> str:
        return self.rendezvous_endpoint

    def _require_distributed(self) -> Any:
        if self._distributed is None:
            import torch.distributed as distributed

            self._distributed = distributed
        return self._distributed


__all__ = [
    "NcclRuntime",
    "NcclRuntimeConfig",
    "NcclRuntimeProtocol",
    "NcclWorkerExchangeProtocol",
]
