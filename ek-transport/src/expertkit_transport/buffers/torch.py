"""Ordinary Torch output buffers used by the gRPC baseline."""

from __future__ import annotations

import torch

from expertkit_transport.contracts.buffers import (
    OutputBufferProvider,
    OutputSpec,
    PreparedOutput,
)


class _TorchPreparedOutput(PreparedOutput):
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor
        self.reuse_event: torch.cuda.Event | None = None

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor


class TorchOutputBufferProvider(OutputBufferProvider):
    """Prepare unregistered contiguous Torch output tensors."""

    def prepare(self, spec: OutputSpec) -> PreparedOutput:
        output = _TorchPreparedOutput(
            torch.empty(
                (spec.max_batch_tokens, spec.hidden_dim),
                dtype=spec.dtype,
                device=spec.device,
            )
        )
        self.validate(output, spec)
        return output

    def validate(self, output: PreparedOutput, spec: OutputSpec) -> None:
        tensor = output.tensor
        if tensor.shape != (spec.max_batch_tokens, spec.hidden_dim):
            raise ValueError("prepared output has the wrong shape")
        if tensor.dtype != spec.dtype:
            raise ValueError("prepared output has the wrong dtype")
        if tensor.device != spec.device:
            raise ValueError("prepared output is on the wrong device")
        if not tensor.is_contiguous():
            raise ValueError("prepared output must be contiguous")

    def before_receive(self, output: PreparedOutput) -> None:
        if not isinstance(output, _TorchPreparedOutput):
            raise TypeError("output was not prepared by TorchOutputBufferProvider")
        if output.reuse_event is not None:
            torch.cuda.current_stream(output.tensor.device).wait_event(output.reuse_event)

    def after_consume(self, output: PreparedOutput) -> None:
        if not isinstance(output, _TorchPreparedOutput):
            raise TypeError("output was not prepared by TorchOutputBufferProvider")
        if output.tensor.device.type != "cuda":
            return
        if output.reuse_event is None:
            output.reuse_event = torch.cuda.Event(enable_timing=False, blocking=False)
        output.reuse_event.record(torch.cuda.current_stream(output.tensor.device))

    def release(self, output: PreparedOutput) -> None:
        if not isinstance(output, _TorchPreparedOutput):
            raise TypeError("output was not prepared by TorchOutputBufferProvider")
        if output.reuse_event is not None:
            output.reuse_event.synchronize()
