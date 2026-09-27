"""Fixed CUDA transfer buffers owned by one NCCL Worker connection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from expertkit_transport.transports.base import WorkerEndpointConfig


@dataclass(slots=True)
class NcclTransferBuffers:
    """Hold request and response storage reused by one in-flight submission."""

    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    partial_output: torch.Tensor
    copy_event: torch.cuda.Event | None
    copy_recorded: bool = False


class NcclTransferBufferPool:
    """Preallocate all device storage before the first NCCL request."""

    def __init__(
        self,
        endpoint_config: WorkerEndpointConfig,
        *,
        device: torch.device | str,
        capacity: int,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._device = torch.device(device)
        uses_cuda = self._device.type == "cuda"
        self._all = tuple(
            NcclTransferBuffers(
                hidden_states=torch.empty(
                    (endpoint_config.max_batch_tokens, endpoint_config.hidden_dim),
                    dtype=endpoint_config.dtype,
                    device=self._device,
                ),
                expert_ids=torch.empty(
                    (endpoint_config.max_batch_tokens, endpoint_config.top_k),
                    dtype=torch.int32,
                    device=self._device,
                ),
                routing_weights=torch.empty(
                    (endpoint_config.max_batch_tokens, endpoint_config.top_k),
                    dtype=torch.float32,
                    device=self._device,
                ),
                partial_output=torch.empty(
                    (endpoint_config.max_batch_tokens, endpoint_config.hidden_dim),
                    dtype=endpoint_config.dtype,
                    device=self._device,
                ),
                copy_event=torch.cuda.Event() if uses_cuda else None,
            )
            for _ in range(capacity)
        )
        self._available = list(self._all)
        self._closed = False

    @property
    def allocated(self) -> tuple[NcclTransferBuffers, ...]:
        """Return fixed slots for diagnostics and allocation tests."""

        return self._all

    def take(self) -> NcclTransferBuffers:
        if self._closed:
            raise RuntimeError("NCCL transfer buffers are closed")
        try:
            return self._available.pop()
        except IndexError as error:
            raise RuntimeError("NCCL admission and transfer buffers diverged") from error

    def put(self, buffers: NcclTransferBuffers) -> None:
        if all(candidate is not buffers for candidate in self._all) or any(
            candidate is buffers for candidate in self._available
        ):
            raise RuntimeError("invalid NCCL transfer buffer return")
        self._available.append(buffers)

    def close(self) -> None:
        if self._closed:
            return
        if len(self._available) != len(self._all):
            raise RuntimeError("cannot close NCCL transfer buffers while calls are active")
        for buffers in self._all:
            if buffers.copy_recorded:
                assert buffers.copy_event is not None
                buffers.copy_event.synchronize()
        self._available.clear()
        self._closed = True


__all__ = ["NcclTransferBufferPool", "NcclTransferBuffers"]
