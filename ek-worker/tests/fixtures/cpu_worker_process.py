"""Run the real Python Worker stack on CPU for cross-language lifecycle tests."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import torch
from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.base import BatchBufferConfig
from expertkit_transport.transports.grpc import GrpcWorkerBatchReceiver
from safetensors.torch import save as save_safetensors

from expertkit_worker.app import WorkerApplication
from expertkit_worker.backends import (
    BackendBatch,
    BackendCapabilities,
    BackendCompletion,
    BackendResourceEstimate,
    ComputeBackend,
)
from expertkit_worker.backends.torch import TorchBackend, TorchExpertWeights, TorchWeightAdapter
from expertkit_worker.config.models import LoggingConfig, ObservabilityConfig
from expertkit_worker.control import (
    ControllerConnection,
    ControllerSupervisor,
    ExpertStateReporter,
    HeartbeatSender,
    WeightControlSession,
    WorkerRegistration,
    new_start_id,
)
from expertkit_worker.execution import WorkerExecutor
from expertkit_worker.factory import _create_device_wiring
from expertkit_worker.observability import configure_logging, create_observability
from expertkit_worker.weights import (
    CachedCpuWeight,
    CpuWeightLoader,
    DirectIOWeightDiskCache,
    DiskWriteback,
    ExpertStateChange,
    PeerWeightServer,
    WeightManager,
    max_safetensors_file_bytes,
)
from expertkit_worker.weights.direct_io import (
    AlignedWeightBuffer,
    expert_file_path,
    write_direct_atomic,
)
from expertkit_worker.weights.dram_cache import DramCache
from expertkit_worker.weights.transfer import HttpWeightTransfer

_INSTANCE_ID = 7
_MODEL_NAME = "fixture/model"
_HIDDEN_DIM = 2
_INTERMEDIATE_DIM = 2
_MAX_BATCH_TOKENS = 2


class _GateBackend(ComputeBackend):
    """Delay Backend submission at a filesystem gate without blocking asyncio."""

    def __init__(
        self,
        delegate: TorchBackend,
        *,
        gate: Path | None,
        active_marker: Path,
    ) -> None:
        self._delegate = delegate
        self._gate = gate
        self._active_marker = active_marker
        self._submission_count = 0

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the wrapped Torch Backend capabilities."""

        return self._delegate.capabilities

    def estimate_resources(self, max_batch_tokens: int) -> BackendResourceEstimate:
        """Return the wrapped Torch Backend memory estimate."""

        return self._delegate.estimate_resources(max_batch_tokens)

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        """Mark active execution, wait for the test gate, and run Torch."""

        self._submission_count += 1
        self._active_marker.write_text(str(self._submission_count), encoding="utf-8")
        gate = self._gate
        while gate is not None and not gate.exists():
            time.sleep(0.005)
        if gate is not None:
            gate.unlink(missing_ok=True)
        return self._delegate.submit(batch, prepared_output)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--computation-listen", required=True)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--active-marker", required=True, type=Path)
    parser.add_argument("--pending-marker", required=True, type=Path)
    parser.add_argument("--gate", type=Path)
    return parser.parse_args()


def _install_weight(root: Path) -> None:
    payload = save_safetensors(
        {
            "model.expert.gate_proj.weight": torch.eye(2, dtype=torch.float32),
            "model.expert.up_proj.weight": torch.eye(2, dtype=torch.float32),
            "model.expert.down_proj.weight": torch.eye(2, dtype=torch.float32),
        }
    )
    target = expert_file_path(root, _MODEL_NAME, 0, 0)
    buffer = AlignedWeightBuffer(len(payload))
    view = buffer.view()
    try:
        view[:] = payload
    finally:
        view.release()
    try:
        write_direct_atomic(target, buffer)
    finally:
        buffer.close()


async def _run(args: argparse.Namespace) -> None:
    _install_weight(args.cache_root)
    args.active_marker.unlink(missing_ok=True)
    args.pending_marker.unlink(missing_ok=True)

    device_wiring = _create_device_wiring("cpu")
    runtime = device_wiring.runtime
    adapter = TorchWeightAdapter(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        source_dtype=torch.float32,
        compute_dtype=torch.float32,
        runtime=runtime,
    )
    batch_spec = WorkerEndpointConfig(
        instance_id=_INSTANCE_ID,
        num_layers=1,
        experts_per_layer=1,
        max_batch_tokens=_MAX_BATCH_TOKENS,
        hidden_dim=_HIDDEN_DIM,
        top_k=1,
        dtype=torch.float32,
    )

    def pending_changed(count: int) -> None:
        if count:
            args.pending_marker.touch()
        else:
            args.pending_marker.unlink(missing_ok=True)

    receiver = GrpcWorkerBatchReceiver(
        args.computation_listen,
        batch_spec,
        max_active_batches=1,
        max_pending_batches=1,
        on_pending_changed=pending_changed,
    )
    manager_holder: list[WeightManager[TorchExpertWeights, TorchExpertWeights] | None] = [None]

    def acquire_many(layer_id: int, expert_ids: tuple[int, ...]):
        manager = manager_holder[0]
        if manager is None:
            raise RuntimeError("Weight Manager is not installed")
        return manager.acquire_many(layer_id, expert_ids)

    torch_backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=1,
        dtype=torch.float32,
        runtime=runtime,
        acquire_many=acquire_many,
    )
    backend = _GateBackend(
        torch_backend,
        gate=args.gate,
        active_marker=args.active_marker,
    )
    execution = WorkerExecutor(
        receiver,
        backend,
        create_slot=device_wiring.create_slot,
        instance_id=_INSTANCE_ID,
        buffer_config=BatchBufferConfig(
            max_batch_tokens=_MAX_BATCH_TOKENS,
            hidden_dim=_HIDDEN_DIM,
            top_k=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
        slot_count=1,
    )

    disk_cache = DirectIOWeightDiskCache(
        root=args.cache_root,
        model_name=_MODEL_NAME,
        max_concurrent_operations=1,
    )
    transfer = HttpWeightTransfer(max_connections=1)
    dram_cache: DramCache[CachedCpuWeight[TorchExpertWeights]] = DramCache(
        max_safetensors_file_bytes(adapter.source_tensor_bytes())
    )
    loader = CpuWeightLoader(
        model_name=_MODEL_NAME,
        disk_cache=disk_cache,
        weight_server_endpoint="http://127.0.0.1:1",
        adapter=adapter,
        cache=dram_cache,
        transfer=transfer,
    )
    reporter = ExpertStateReporter(
        num_layers=1,
        experts_per_layer=1,
        max_updates=64,
        max_delay_ms=10,
    )

    def state_changed(change: ExpertStateChange) -> None:
        reporter.record(change)

    manager = WeightManager(
        num_layers=1,
        experts_per_layer=1,
        device="cpu",
        device_weight_capacity_bytes=adapter.ready_weight_bytes(),
        max_concurrent_loads=1,
        adapter=adapter,
        loader=loader,
        writeback=DiskWriteback(
            disk_cache=disk_cache,
            enabled=False,
            max_pending=1,
        ),
        state_changed=state_changed,
    )
    manager_holder[0] = manager
    peer_server = PeerWeightServer(
        model_name=_MODEL_NAME,
        num_layers=1,
        experts_per_layer=1,
        host="127.0.0.1",
        port=0,
        max_concurrent_requests=1,
        loader=loader,
    )

    start_id = new_start_id()
    connection = ControllerConnection(args.controller)
    heartbeat = HeartbeatSender(
        worker_id=args.worker_id,
        start_id=start_id,
        interval_secs=0.05,
    )
    weights = WeightControlSession(
        worker_id=args.worker_id,
        start_id=start_id,
        max_experts=1,
        shutdown_grace_secs=5.0,
        manager=manager,
        reporter=reporter,
        receiver=receiver,
    )
    registration = WorkerRegistration(
        worker_id=args.worker_id,
        start_id=start_id,
        instance_id=_INSTANCE_ID,
        computation_endpoint=args.computation_listen,
        peer_weight_endpoint="http://127.0.0.1:1",
        backend="torch",
        activation_dtype=torch.float32,
        device="cpu",
        max_experts=1,
        max_batch_tokens=_MAX_BATCH_TOKENS,
        max_active_batches=1,
        max_pending_batches=1,
        transport_type="grpc",
    )
    control = ControllerSupervisor(
        connection=connection,
        registration=registration,
        heartbeat=heartbeat,
        weights=weights,
        registration_timeout_secs=2.0,
        retry_initial_secs=0.05,
        retry_max_secs=0.2,
        stable_stream_secs=0.2,
    )
    observability = create_observability(ObservabilityConfig(), worker_id=args.worker_id)
    application = WorkerApplication(
        transfer=transfer,
        manager=manager,
        peer_server=peer_server,
        execution=execution,
        control=control,
        disk_cache=disk_cache,
        observability=observability,
    )
    remove_signal_handlers = application.install_signal_handlers()
    try:
        await application.run()
    finally:
        remove_signal_handlers()


def main() -> None:
    """Parse the fixture arguments and run until a signal or fatal error."""

    configure_logging(LoggingConfig())
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
