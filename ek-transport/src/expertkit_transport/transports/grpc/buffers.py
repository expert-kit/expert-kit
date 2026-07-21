"""Fixed Host staging owned privately by one gRPC Worker connection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from expertkit_transport.transports.grpc.spec import GrpcBatchSpec


@dataclass(slots=True)
class GrpcTransferBuffers:
    """Hold the Host Tensors and Events reused by one in-flight gRPC call."""

    host_hidden_states: torch.Tensor
    host_expert_ids: torch.Tensor
    host_routing_weights: torch.Tensor
    host_partial_output: torch.Tensor | None
    request_copy_event: torch.cuda.Event | None
    receive_event: torch.cuda.Event | None
    request_copy_recorded: bool = False
    receive_recorded: bool = False


class GrpcTransferBufferPool:
    """Preallocate and recycle the private staging for one gRPC connection."""

    def __init__(
        self,
        batch_spec: GrpcBatchSpec,
        *,
        device: torch.device | str,
        capacity: int,
    ) -> None:
        self._device = torch.device(device)
        uses_cuda = self._device.type == "cuda"
        host_options = {"device": "cpu", "pin_memory": uses_cuda}
        self._all = tuple(
            GrpcTransferBuffers(
                host_hidden_states=torch.empty(
                    (batch_spec.max_batch_tokens, batch_spec.hidden_dim),
                    dtype=batch_spec.dtype,
                    **host_options,
                ),
                host_expert_ids=torch.empty(
                    (batch_spec.max_batch_tokens, batch_spec.top_k),
                    dtype=torch.int32,
                    **host_options,
                ),
                host_routing_weights=torch.empty(
                    (batch_spec.max_batch_tokens, batch_spec.top_k),
                    dtype=torch.float32,
                    **host_options,
                ),
                host_partial_output=(
                    torch.empty(
                        (batch_spec.max_batch_tokens, batch_spec.hidden_dim),
                        dtype=batch_spec.dtype,
                        **host_options,
                    )
                    if uses_cuda
                    else None
                ),
                request_copy_event=torch.cuda.Event() if uses_cuda else None,
                receive_event=torch.cuda.Event() if uses_cuda else None,
            )
            for _ in range(capacity)
        )
        self._available = list(self._all)
        self._closed = False

    @property
    def allocated(self) -> tuple[GrpcTransferBuffers, ...]:
        """Return fixed slots for diagnostics and allocation tests."""

        return self._all

    def take(self) -> GrpcTransferBuffers:
        """Take one slot after the caller has acquired connection admission."""

        if self._closed:
            raise RuntimeError("gRPC transfer buffers are closed")
        try:
            return self._available.pop()
        except IndexError as error:
            raise RuntimeError("gRPC admission and transfer buffers diverged") from error

    def put(self, buffers: GrpcTransferBuffers) -> None:
        """Return one slot after its request no longer accesses caller Tensors."""

        if all(candidate is not buffers for candidate in self._all) or any(
            candidate is buffers for candidate in self._available
        ):
            raise RuntimeError("invalid gRPC transfer buffer return")
        self._available.append(buffers)

    def close(self) -> None:
        """Wait for outstanding device copies before releasing fixed staging."""

        if self._closed:
            return
        if len(self._available) != len(self._all):
            raise RuntimeError("cannot close gRPC transfer buffers while calls are active")
        for buffers in self._all:
            for recorded, event in (
                (buffers.request_copy_recorded, buffers.request_copy_event),
                (buffers.receive_recorded, buffers.receive_event),
            ):
                if recorded:
                    assert event is not None
                    event.synchronize()
        self._available.clear()
        self._closed = True
