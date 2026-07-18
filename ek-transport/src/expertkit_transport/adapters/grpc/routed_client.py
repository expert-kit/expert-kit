"""High-level asynchronous and blocking Routed-MoE gRPC clients."""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import time
from collections.abc import Callable

import torch

from expertkit_transport.adapters.grpc.topology import GrpcTopologyProvider
from expertkit_transport.contracts import RoutedLayerBatch, TransportError, TransportErrorCode
from expertkit_transport.orchestration import RoundRobinSelector, execute_routed_layer

_RESULT_GRACE_SECONDS = 0.1


def _validate_timeout(timeout_seconds: float) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")


class GrpcRoutedMoEClient:
    """Execute complete routed layers over a watched direct-Worker topology."""

    def __init__(
        self,
        controller_endpoint: str,
        *,
        instance_id: int,
        num_layers: int,
        experts_per_layer: int,
        hidden_dim: int,
        top_k: int,
        dtype: torch.dtype,
        device: torch.device | str,
        same_worker_retry_delay_seconds: float = 0.001,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not math.isfinite(same_worker_retry_delay_seconds)
            or same_worker_retry_delay_seconds <= 0
        ):
            raise ValueError("same_worker_retry_delay_seconds must be finite and positive")
        self._instance_id = instance_id
        self._hidden_dim = hidden_dim
        self._top_k = top_k
        self._dtype = dtype
        self._device = torch.device(device)
        self._same_worker_retry_delay_seconds = same_worker_retry_delay_seconds
        self._clock = clock
        self._topology = GrpcTopologyProvider(
            controller_endpoint,
            instance_id=instance_id,
            num_layers=num_layers,
            experts_per_layer=experts_per_layer,
            hidden_dim=hidden_dim,
            top_k=top_k,
            dtype=dtype,
            device=self._device,
            clock=clock,
        )
        self._selector = RoundRobinSelector()
        self._started = False
        self._closed = False

    async def start(self, *, monotonic_deadline: float) -> None:
        """Connect to the Controller and install its first complete topology."""

        if self._closed:
            raise RuntimeError("Routed-MoE client is closed")
        await self._topology.start(monotonic_deadline=monotonic_deadline)
        self._started = True

    async def execute(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        distinct_expert_ids: tuple[int, ...],
        monotonic_deadline: float,
    ) -> torch.Tensor:
        """Return the weighted and aggregated result for one routed layer.

        Args:
            layer_id: Zero-based model layer number.
            hidden_states: Activations shaped `[token_count, hidden_dim]`.
            expert_ids: Final int32 assignments shaped `[token_count, top_k]`.
            routing_weights: Final FP32 weights shaped `[token_count, top_k]`.
            distinct_expert_ids: Sorted distinct valid expert numbers.

        Returns:
            Tensor shaped `[token_count, hidden_dim]` on the input device and
            using the input activation dtype.
        """

        if not self._started or self._closed:
            raise RuntimeError("Routed-MoE client is not running")
        if hidden_states.shape[1:] != (self._hidden_dim,):
            raise ValueError("hidden_states does not match the configured hidden dimension")
        if expert_ids.shape[1:] != (self._top_k,):
            raise ValueError("expert_ids does not match the configured top_k")
        if hidden_states.dtype != self._dtype:
            raise ValueError("hidden_states does not match the configured activation dtype")
        if hidden_states.device != self._device:
            raise ValueError("hidden_states does not match the configured Frontend device")
        batch = RoutedLayerBatch(
            instance_id=self._instance_id,
            layer_id=layer_id,
            hidden_states=hidden_states,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
            distinct_expert_ids=distinct_expert_ids,
        )
        return await execute_routed_layer(
            batch,
            self._topology,
            self._selector,
            self._topology.pools,
            monotonic_deadline=monotonic_deadline,
            same_worker_retry_delay_seconds=self._same_worker_retry_delay_seconds,
            clock=self._clock,
        )

    async def close(self) -> None:
        """Stop topology watching and all direct Worker connections."""

        if self._closed:
            return
        self._closed = True
        await self._topology.close()


class BlockingGrpcRoutedMoEClient:
    """Run the asynchronous gRPC client on one private event-loop thread.

    Framework model hooks are commonly synchronous. This wrapper preserves one
    process-lifetime asyncio loop and channel set rather than creating an event
    loop for every layer. Several calling threads may submit work concurrently.
    CUDA events order caller-stream inputs and outputs without a device-wide or
    Host-side CUDA synchronization.
    """

    def __init__(
        self,
        controller_endpoint: str,
        *,
        instance_id: int,
        num_layers: int,
        experts_per_layer: int,
        hidden_dim: int,
        top_k: int,
        dtype: torch.dtype,
        device: torch.device | str,
        same_worker_retry_delay_seconds: float = 0.001,
    ) -> None:
        self._client_args = (controller_endpoint,)
        self._client_kwargs = {
            "instance_id": instance_id,
            "num_layers": num_layers,
            "experts_per_layer": experts_per_layer,
            "hidden_dim": hidden_dim,
            "top_k": top_k,
            "dtype": dtype,
            "device": device,
            "same_worker_retry_delay_seconds": same_worker_retry_delay_seconds,
        }
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: GrpcRoutedMoEClient | None = None
        self._thread_error: BaseException | None = None
        self._thread_ready = threading.Event()
        self._lock = threading.Lock()
        self._active: set[threading.Event] = set()
        self._started = False
        self._closing = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name="expertkit-transport",
            daemon=True,
        )
        self._thread.start()
        self._thread_ready.wait()
        if self._thread_error is not None:
            raise RuntimeError("failed to create the Transport event loop") from self._thread_error

    def start(self, *, timeout_seconds: float) -> None:
        """Connect to the Controller and wait for the first topology snapshot."""

        _validate_timeout(timeout_seconds)
        with self._lock:
            if self._closing:
                raise RuntimeError("Routed-MoE client is closing")
            if self._started:
                return
        loop, client = self._require_loop()
        deadline = time.monotonic() + timeout_seconds
        future = asyncio.run_coroutine_threadsafe(
            client.start(monotonic_deadline=deadline),
            loop,
        )
        future.result(timeout=timeout_seconds + _RESULT_GRACE_SECONDS)
        with self._lock:
            self._started = True

    def execute(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        distinct_expert_ids: tuple[int, ...],
        timeout_seconds: float,
    ) -> torch.Tensor:
        """Block the caller until one asynchronous routed-layer call is submitted."""

        _validate_timeout(timeout_seconds)
        completion = threading.Event()
        with self._lock:
            if not self._started or self._closing:
                raise RuntimeError("Routed-MoE client is not running")
            self._active.add(completion)
        loop, client = self._require_loop()
        deadline = time.monotonic() + timeout_seconds
        input_ready: torch.cuda.Event | None = None
        if hidden_states.device.type == "cuda":
            input_ready = torch.cuda.Event(enable_timing=False, blocking=False)
            input_ready.record(torch.cuda.current_stream(hidden_states.device))
        future = asyncio.run_coroutine_threadsafe(
            self._execute(
                client,
                layer_id=layer_id,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                routing_weights=routing_weights,
                distinct_expert_ids=distinct_expert_ids,
                monotonic_deadline=deadline,
                input_ready=input_ready,
                completion=completion,
            ),
            loop,
        )
        try:
            result, output_ready = future.result(timeout=timeout_seconds + _RESULT_GRACE_SECONDS)
        except concurrent.futures.TimeoutError as error:
            future.cancel()
            completion.wait()
            raise TransportError(
                TransportErrorCode.DEADLINE_EXCEEDED,
                retryable=False,
                diagnostic="the blocking Routed-MoE call exceeded its deadline",
            ) from error
        finally:
            with self._lock:
                self._active.discard(completion)
        if output_ready is not None:
            torch.cuda.current_stream(result.device).wait_event(output_ready)
        return result

    def close(self) -> None:
        """Reject new calls, close asynchronous resources, and join the loop thread."""

        with self._lock:
            if self._closed:
                return
            self._closing = True
        loop, client = self._require_loop()
        future = asyncio.run_coroutine_threadsafe(client.close(), loop)
        future.result()
        with self._lock:
            active = tuple(self._active)
        for completion in active:
            completion.wait()
        loop.call_soon_threadsafe(loop.stop)
        self._thread.join()
        with self._lock:
            self._closed = True

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            client = GrpcRoutedMoEClient(*self._client_args, **self._client_kwargs)
        except BaseException as error:
            self._thread_error = error
            self._thread_ready.set()
            loop.close()
            return
        self._loop = loop
        self._client = client
        loop.call_soon(self._thread_ready.set)
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    def _require_loop(self) -> tuple[asyncio.AbstractEventLoop, GrpcRoutedMoEClient]:
        if self._loop is None or self._client is None:
            raise RuntimeError("Transport event loop is unavailable")
        return self._loop, self._client

    @staticmethod
    async def _execute(
        client: GrpcRoutedMoEClient,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        distinct_expert_ids: tuple[int, ...],
        monotonic_deadline: float,
        input_ready: torch.cuda.Event | None,
        completion: threading.Event,
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        try:
            if input_ready is not None:
                torch.cuda.current_stream(hidden_states.device).wait_event(input_ready)
            result = await client.execute(
                layer_id=layer_id,
                hidden_states=hidden_states,
                expert_ids=expert_ids,
                routing_weights=routing_weights,
                distinct_expert_ids=distinct_expert_ids,
                monotonic_deadline=monotonic_deadline,
            )
            output_ready: torch.cuda.Event | None = None
            if result.device.type == "cuda":
                output_ready = torch.cuda.Event(enable_timing=False, blocking=False)
                output_ready.record(torch.cuda.current_stream(result.device))
            return result, output_ready
        finally:
            completion.set()
