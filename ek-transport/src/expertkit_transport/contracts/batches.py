"""Tensor and metadata contract for one physical Worker batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

ACTIVATION_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


def _require_unsigned(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")


def _validate_routing_tensors(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
) -> None:
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [token_count, hidden_dim]")
    if hidden_states.shape[0] <= 0 or hidden_states.shape[1] <= 0:
        raise ValueError("hidden_states dimensions must be positive")
    if hidden_states.dtype not in ACTIVATION_DTYPES:
        raise ValueError("hidden_states must use FP16, BF16, or FP32")

    if expert_ids.ndim != 2 or expert_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("expert_ids must have shape [token_count, top_k]")
    if expert_ids.shape[1] <= 0:
        raise ValueError("top_k must be positive")
    if expert_ids.dtype != torch.int32:
        raise ValueError("expert_ids must use int32")
    if expert_ids.device != hidden_states.device:
        raise ValueError("expert_ids and hidden_states must be on the same device")

    if routing_weights.shape != expert_ids.shape:
        raise ValueError("routing_weights must have the same shape as expert_ids")
    if routing_weights.dtype != torch.float32:
        raise ValueError("routing_weights must use FP32")
    if routing_weights.device != hidden_states.device:
        raise ValueError("routing_weights and hidden_states must be on the same device")


@dataclass(frozen=True, slots=True)
class RoutedLayerBatch:
    """Hold the final assignments produced by one model router.

    Attributes:
        instance_id: Controller-assigned model instance identifier.
        layer_id: Model layer containing the routed experts.
        hidden_states: Activations shaped `[token_count, hidden_dim]`.
        expert_ids: Stable expert numbers shaped `[token_count, top_k]` using
            int32. A `-1` position must have zero routing weight.
        routing_weights: Final FP32 routing weights shaped `[token_count, top_k]`.

    Note:
        Construction checks Tensor metadata only. Grouping validates routing
        values while resolving them against one Topology snapshot.
    """

    instance_id: int
    layer_id: int
    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor

    def __post_init__(self) -> None:
        _require_unsigned("instance_id", self.instance_id, _UINT64_MAX)
        _require_unsigned("layer_id", self.layer_id, _UINT32_MAX)
        _validate_routing_tensors(
            self.hidden_states,
            self.expert_ids,
            self.routing_weights,
        )

    @property
    def token_count(self) -> int:
        """Return the number of routed activation rows."""

        return self.hidden_states.shape[0]

    @property
    def hidden_dim(self) -> int:
        """Return the activation hidden dimension."""

        return self.hidden_states.shape[1]

    @property
    def top_k(self) -> int:
        """Return the fixed routing width."""

        return self.expert_ids.shape[1]


@dataclass(frozen=True, slots=True)
class WorkerBatch:
    """Describe one asynchronous computation sent to one Worker.

    `hidden_states` remains owned by the caller. When `token_indices` is present,
    the adapter gathers those rows; the common middleware does not compact them.
    The routing tensors are already aligned to the selected rows.

    Attributes:
        instance_id: Controller-assigned model instance identifier.
        layer_id: Model layer containing the routed experts.
        topology_version: Routing snapshot used to form this batch.
        hidden_states: Source activations shaped `[source_tokens, hidden_dim]`.
        token_indices: Optional int64 indices selecting `token_count` source rows.
            `None` selects every source row in order.
        expert_ids: Stable expert numbers shaped `[token_count, top_k]`. Invalid
            positions contain `-1`.
        routing_weights: FP32 weights shaped `[token_count, top_k]`.
        distinct_expert_ids: Sorted distinct valid expert numbers required by
            the batch. The Worker uses an independently validated copy for one
            ready-weight lookup.

    Note:
        Construction validates metadata, shapes, dtypes, and devices without
        reading Tensor values. This avoids a device-to-Host synchronization.
        Orchestration and receiving adapters validate indices and routing values
        while they already process those values.
    """

    instance_id: int
    layer_id: int
    topology_version: int
    hidden_states: torch.Tensor
    token_indices: torch.Tensor | None
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    distinct_expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "distinct_expert_ids", tuple(self.distinct_expert_ids))
        _require_unsigned("instance_id", self.instance_id, _UINT64_MAX)
        _require_unsigned("layer_id", self.layer_id, _UINT32_MAX)
        _require_unsigned("topology_version", self.topology_version, _UINT64_MAX)

        if self.hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [source_tokens, hidden_dim]")
        if self.hidden_states.shape[0] <= 0 or self.hidden_states.shape[1] <= 0:
            raise ValueError("hidden_states dimensions must be positive")
        if self.hidden_states.dtype not in ACTIVATION_DTYPES:
            raise ValueError("hidden_states must use FP16, BF16, or FP32")

        token_count = self.hidden_states.shape[0]
        if self.token_indices is not None:
            if self.token_indices.ndim != 1 or self.token_indices.shape[0] <= 0:
                raise ValueError("token_indices must have shape [token_count]")
            if self.token_indices.dtype != torch.int64:
                raise ValueError("token_indices must use int64")
            if self.token_indices.device != self.hidden_states.device:
                raise ValueError("token_indices and hidden_states must be on the same device")
            token_count = self.token_indices.shape[0]

        if self.expert_ids.ndim != 2 or self.expert_ids.shape[0] != token_count:
            raise ValueError("expert_ids must have shape [token_count, top_k]")
        if self.expert_ids.shape[1] <= 0:
            raise ValueError("top_k must be positive")
        if self.expert_ids.dtype != torch.int32:
            raise ValueError("expert_ids must use int32")
        if self.expert_ids.device != self.hidden_states.device:
            raise ValueError("expert_ids and hidden_states must be on the same device")

        if self.routing_weights.shape != self.expert_ids.shape:
            raise ValueError("routing_weights must have the same shape as expert_ids")
        if self.routing_weights.dtype != torch.float32:
            raise ValueError("routing_weights must use FP32")
        if self.routing_weights.device != self.hidden_states.device:
            raise ValueError("routing_weights and hidden_states must be on the same device")

        previous = -1
        for expert_id in self.distinct_expert_ids:
            _require_unsigned("distinct expert ID", expert_id, _UINT32_MAX)
            if expert_id <= previous:
                raise ValueError("distinct_expert_ids must be sorted with no duplicates")
            previous = expert_id

    @property
    def token_count(self) -> int:
        """Return the number of selected activation rows."""

        if self.token_indices is None:
            return self.hidden_states.shape[0]
        return self.token_indices.shape[0]

    @property
    def hidden_dim(self) -> int:
        """Return the activation hidden dimension."""

        return self.hidden_states.shape[1]

    @property
    def top_k(self) -> int:
        """Return the fixed routing width."""

        return self.expert_ids.shape[1]
