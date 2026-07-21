"""Reusable Frontend Tensor and Host staging buffers for gRPC calls."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from expertkit_transport.buffers.base import OutputBufferProvider, OutputSpec, PreparedOutput
from expertkit_transport.transports.grpc.spec import GrpcBatchSpec


@dataclass(slots=True)
class GrpcPreparedOutput(PreparedOutput):
    """Hold one result Tensor and the Host staging owned by its call slot."""

    _tensor: torch.Tensor
    host_hidden_states: torch.Tensor
    host_expert_ids: torch.Tensor
    host_routing_weights: torch.Tensor
    host_partial_output: torch.Tensor
    request_copy_event: torch.cuda.Event | None
    receive_event: torch.cuda.Event | None
    consume_event: torch.cuda.Event | None
    request_copy_recorded: bool = False
    receive_recorded: bool = False
    consume_recorded: bool = False

    @property
    def tensor(self) -> torch.Tensor:
        """Return the maximum-size Frontend result Tensor."""

        return self._tensor


class GrpcOutputBufferProvider(OutputBufferProvider):
    """Allocate bounded gRPC output and request/response staging Tensors."""

    def __init__(self, batch_spec: GrpcBatchSpec) -> None:
        self._batch_spec = batch_spec

    def prepare(self, spec: OutputSpec) -> PreparedOutput:
        """Allocate one result and its fixed maximum Host staging Tensors."""

        self._validate_spec(spec)
        device = torch.device(spec.device)
        uses_cuda = device.type == "cuda"
        host_options = {"device": "cpu", "pin_memory": uses_cuda}
        output_tensor = torch.empty(
            (spec.max_batch_tokens, spec.hidden_dim),
            dtype=spec.dtype,
            device=device,
        )
        host_partial_output = (
            torch.empty(
                (spec.max_batch_tokens, spec.hidden_dim),
                dtype=spec.dtype,
                **host_options,
            )
            if uses_cuda
            else output_tensor
        )
        return GrpcPreparedOutput(
            _tensor=output_tensor,
            host_hidden_states=torch.empty(
                (spec.max_batch_tokens, spec.hidden_dim),
                dtype=spec.dtype,
                **host_options,
            ),
            host_expert_ids=torch.empty(
                (spec.max_batch_tokens, self._batch_spec.top_k),
                dtype=torch.int32,
                **host_options,
            ),
            host_routing_weights=torch.empty(
                (spec.max_batch_tokens, self._batch_spec.top_k),
                dtype=torch.float32,
                **host_options,
            ),
            host_partial_output=host_partial_output,
            request_copy_event=torch.cuda.Event() if uses_cuda else None,
            receive_event=torch.cuda.Event() if uses_cuda else None,
            consume_event=torch.cuda.Event() if uses_cuda else None,
        )

    def validate(self, output: PreparedOutput, spec: OutputSpec) -> None:
        """Reject buffers not created for this gRPC endpoint and output spec."""

        self._validate_spec(spec)
        prepared = self.require_prepared(output)
        expected = (spec.max_batch_tokens, spec.hidden_dim)
        if prepared.tensor.shape != expected:
            raise ValueError("gRPC output Tensor has the wrong shape")
        if prepared.tensor.dtype != spec.dtype:
            raise ValueError("gRPC output Tensor has the wrong dtype")
        if prepared.tensor.device != torch.device(spec.device):
            raise ValueError("gRPC output Tensor is on the wrong device")

    def before_receive(self, output: PreparedOutput) -> None:
        """Order a response write after prior receive and consumption work."""

        prepared = self.require_prepared(output)
        if prepared.tensor.device.type != "cuda":
            return
        stream = torch.cuda.current_stream(prepared.tensor.device)
        if prepared.receive_recorded:
            assert prepared.receive_event is not None
            stream.wait_event(prepared.receive_event)
        if prepared.consume_recorded:
            assert prepared.consume_event is not None
            stream.wait_event(prepared.consume_event)

    def after_consume(self, output: PreparedOutput) -> None:
        """Record the Frontend stream work that consumed this output."""

        prepared = self.require_prepared(output)
        if prepared.tensor.device.type == "cuda":
            assert prepared.consume_event is not None
            prepared.consume_event.record(torch.cuda.current_stream(prepared.tensor.device))
            prepared.consume_recorded = True

    def release(self, output: PreparedOutput) -> None:
        """Wait for outstanding staging and result use before releasing storage."""

        prepared = self.require_prepared(output)
        for recorded, event in (
            (prepared.request_copy_recorded, prepared.request_copy_event),
            (prepared.receive_recorded, prepared.receive_event),
            (prepared.consume_recorded, prepared.consume_event),
        ):
            if recorded:
                assert event is not None
                event.synchronize()

    def require_prepared(self, output: PreparedOutput) -> GrpcPreparedOutput:
        """Return the concrete gRPC buffers or reject another provider's object."""

        if not isinstance(output, GrpcPreparedOutput):
            raise TypeError("gRPC Transport requires a gRPC prepared output")
        return output

    def _validate_spec(self, spec: OutputSpec) -> None:
        if spec.max_batch_tokens != self._batch_spec.max_batch_tokens:
            raise ValueError("output max_batch_tokens does not match the gRPC endpoint")
        if spec.hidden_dim != self._batch_spec.hidden_dim:
            raise ValueError("output hidden dimension does not match the gRPC endpoint")
        if spec.dtype != self._batch_spec.dtype:
            raise ValueError("output dtype does not match the gRPC endpoint")
