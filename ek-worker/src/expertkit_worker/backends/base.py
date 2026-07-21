"""Runtime contract shared by every Worker Compute backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import torch
from expertkit_transport.batches import ACTIVATION_DTYPES

_UINT32_MAX = (1 << 32) - 1


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_optional_positive_integer(name: str, value: int | None) -> None:
    if value is not None:
        _require_positive_integer(name, value)


@dataclass(frozen=True, slots=True)
class BackendBatch:
    """Hold only the Tensor and expert fields used by computation.

    Attributes:
        layer_id: Model layer that scopes stable expert numbers.
        hidden_states: Contiguous activations shaped ``[token_count, hidden_dim]``.
        expert_ids: Contiguous int32 routes shaped ``[token_count, top_k]``.
        routing_weights: Contiguous FP32 weights with the same routing shape.
        distinct_expert_ids: Sorted unique valid expert numbers. The receiver
            validated this Host metadata against ``expert_ids`` before execution.

    Note:
        Validation intentionally does not inspect Tensor values. In particular, it
        never reads routing IDs back from a CUDA device.
    """

    layer_id: int
    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    distinct_expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "distinct_expert_ids", tuple(self.distinct_expert_ids))
        if (
            isinstance(self.layer_id, bool)
            or not isinstance(self.layer_id, int)
            or not 0 <= self.layer_id <= _UINT32_MAX
        ):
            raise ValueError(f"layer_id must be an integer in [0, {_UINT32_MAX}]")
        if self.hidden_states.ndim != 2 or min(self.hidden_states.shape) <= 0:
            raise ValueError("hidden_states must have positive shape [token_count, hidden_dim]")
        if self.hidden_states.dtype not in ACTIVATION_DTYPES:
            raise ValueError("hidden_states must use FP16, BF16, or FP32")
        if not self.hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous")

        if self.expert_ids.ndim != 2 or self.expert_ids.shape[0] != self.token_count:
            raise ValueError("expert_ids must have shape [token_count, top_k]")
        if self.expert_ids.shape[1] <= 0:
            raise ValueError("top_k must be positive")
        if self.expert_ids.dtype != torch.int32:
            raise ValueError("expert_ids must use int32")
        if self.expert_ids.device != self.hidden_states.device:
            raise ValueError("expert_ids and hidden_states must be on the same device")
        if not self.expert_ids.is_contiguous():
            raise ValueError("expert_ids must be contiguous")

        if self.routing_weights.shape != self.expert_ids.shape:
            raise ValueError("routing_weights must have the same shape as expert_ids")
        if self.routing_weights.dtype != torch.float32:
            raise ValueError("routing_weights must use FP32")
        if self.routing_weights.device != self.hidden_states.device:
            raise ValueError("routing_weights and hidden_states must be on the same device")
        if not self.routing_weights.is_contiguous():
            raise ValueError("routing_weights must be contiguous")

        previous = -1
        for expert_id in self.distinct_expert_ids:
            if (
                isinstance(expert_id, bool)
                or not isinstance(expert_id, int)
                or not 0 <= expert_id <= _UINT32_MAX
            ):
                raise ValueError(f"distinct expert ID must be an integer in [0, {_UINT32_MAX}]")
            if expert_id <= previous:
                raise ValueError("distinct_expert_ids must be sorted with no duplicates")
            previous = expert_id

    @property
    def token_count(self) -> int:
        """Return the number of activation rows in this physical batch."""

        return self.hidden_states.shape[0]

    @property
    def hidden_dim(self) -> int:
        """Return the model hidden dimension."""

        return self.hidden_states.shape[1]

    @property
    def top_k(self) -> int:
        """Return the fixed routing width."""

        return self.expert_ids.shape[1]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declare semantic capacity without exposing private kernel tuning."""

    supports_dynamic_tokens: bool
    supports_concurrent_batches: bool
    required_assignment_alignment: int = 1
    max_batch_tokens: int | None = None
    max_assignments_per_batch: int | None = None
    supported_token_profiles: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_positive_integer(
            "required_assignment_alignment",
            self.required_assignment_alignment,
        )
        _require_optional_positive_integer("max_batch_tokens", self.max_batch_tokens)
        _require_optional_positive_integer(
            "max_assignments_per_batch",
            self.max_assignments_per_batch,
        )
        profiles = tuple(self.supported_token_profiles)
        object.__setattr__(self, "supported_token_profiles", profiles)
        previous = 0
        for profile in profiles:
            _require_positive_integer("supported token profile", profile)
            if profile <= previous:
                raise ValueError("supported_token_profiles must be sorted with no duplicates")
            previous = profile
        if not self.supports_dynamic_tokens and not profiles:
            raise ValueError("a fixed-token Backend must declare supported_token_profiles")

    def validate_runtime(self, *, max_batch_tokens: int, active_batches: int) -> None:
        """Reject Worker limits that this Backend cannot execute safely."""

        _require_positive_integer("max_batch_tokens", max_batch_tokens)
        _require_positive_integer("active_batches", active_batches)
        if self.max_batch_tokens is not None and max_batch_tokens > self.max_batch_tokens:
            raise ValueError("worker.max_batch_tokens exceeds the Backend capability")
        if active_batches > 1 and not self.supports_concurrent_batches:
            raise ValueError("the Backend does not support concurrent active batches")
        if (
            not self.supports_dynamic_tokens
            and max_batch_tokens > self.supported_token_profiles[-1]
        ):
            raise ValueError("worker.max_batch_tokens exceeds the largest Backend token profile")


@dataclass(frozen=True, slots=True)
class BackendResourceEstimate:
    """Report conservative computation memory used for startup budgeting."""

    temporary_bytes_per_active_batch: int
    shared_temporary_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("temporary_bytes_per_active_batch", self.temporary_bytes_per_active_batch),
            ("shared_temporary_bytes", self.shared_temporary_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    def total_bytes(self, active_batches: int) -> int:
        """Return the worst-case temporary bytes for the execution-slot count."""

        _require_positive_integer("active_batches", active_batches)
        return self.shared_temporary_bytes + self.temporary_bytes_per_active_batch * active_batches


class BackendRequestError(RuntimeError):
    """Base class for failures scoped safely to one computation request."""


class InvalidBackendInput(BackendRequestError):
    """Reject malformed computation input without retrying it."""


class UnsupportedBackendBatch(BackendRequestError):
    """Reject a valid batch outside this Backend's declared capability."""


class BackendWeightUnavailable(BackendRequestError):
    """Reject a batch whose ready weight references cannot all be retained."""

    def __init__(self, unavailable_expert_ids: Iterable[int]) -> None:
        expert_ids = tuple(unavailable_expert_ids)
        super().__init__("required expert weights are not ready")
        self.unavailable_expert_ids = expert_ids


class BackendFatalReason(StrEnum):
    """Classify Backend failures that make continued service unsafe."""

    DEVICE_OOM = "device_oom"
    DEVICE_FAILURE = "device_failure"
    ASYNC_EXECUTION = "async_execution"
    UNEXPECTED = "unexpected"


class BackendFatalError(RuntimeError):
    """Report a Backend or device failure that must terminate the Worker."""

    def __init__(self, reason: BackendFatalReason, diagnostic: str = "") -> None:
        super().__init__(diagnostic or reason.value)
        self.reason = reason
        self.diagnostic = diagnostic


class BackendCompletion(ABC):
    """Retain submission resources and report asynchronous execution errors."""

    @abstractmethod
    def wait_host(self) -> None:
        """Block the current execution thread until Backend work is complete."""

    @abstractmethod
    def close(self) -> None:
        """Release retained Backend resources after all consumers are finished."""


class CompletedSubmission(BackendCompletion):
    """Represent synchronous work while retaining its resources until release."""

    def __init__(self, retained_resources: Iterable[object] = ()) -> None:
        self._retained_resources = list(retained_resources)
        self._closed = False

    def wait_host(self) -> None:
        """Return immediately because the Backend finished before submission returned."""

    def close(self) -> None:
        """Drop retained resources exactly once."""

        if self._closed:
            return
        self._closed = True
        self._retained_resources.clear()


class ComputeBackend(ABC):
    """Execute one Worker-batch computation into caller-owned output storage."""

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return stable semantic limits for this Backend instance."""

    @abstractmethod
    def estimate_resources(self, max_batch_tokens: int) -> BackendResourceEstimate:
        """Return a conservative temporary-memory estimate for startup planning."""

    @abstractmethod
    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        """Write one Weighted partial output and return its completion state.

        Args:
            batch: Valid computation fields whose Tensor storage remains live until
                the returned completion is closed.
            prepared_output: Contiguous Tensor shaped ``[token_count, hidden_dim]``
                on the batch device and in the activation dtype. The Backend writes
                the complete local weighted reduction into this storage.

        Returns:
            A completion that retains every weight and temporary resource needed by
            the submission. The caller closes it only after output communication no
            longer uses those resources.

        Raises:
            BackendRequestError: The batch can be rejected without terminating the
                Worker.
            BackendFatalError: Backend or device state is unsafe to continue using.
        """
