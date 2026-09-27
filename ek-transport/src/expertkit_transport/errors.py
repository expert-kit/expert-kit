"""Transport failures and malformed-message errors."""

from __future__ import annotations

from enum import StrEnum


class TransportErrorCode(StrEnum):
    """Stable failure classes consumed by retry and routing policy."""

    BUSY = "busy"
    DRAINING = "draining"
    STALE_TOPOLOGY = "stale_topology"
    EXPERT_NOT_READY = "expert_not_ready"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"
    PROTOCOL = "protocol"


class TransportProtocolError(ValueError):
    """Report malformed or inconsistent data at a Transport boundary."""


class TransportError(RuntimeError):
    """Report one failed Worker contribution without parsing message text."""

    def __init__(
        self,
        code: TransportErrorCode,
        *,
        retryable: bool,
        unsafe_tensor_ownership: bool = False,
        unsafe_output: bool = False,
        observed_topology_version: int | None = None,
        min_topology_version: int | None = None,
        unavailable_expert_ids: tuple[int, ...] = (),
        diagnostic: str = "",
    ) -> None:
        for name, value in (
            ("unsafe_tensor_ownership", unsafe_tensor_ownership),
            ("unsafe_output", unsafe_output),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if (unsafe_tensor_ownership or unsafe_output) and retryable:
            raise ValueError("an unsafe Tensor ownership failure cannot be retryable")
        super().__init__(diagnostic or code.value)
        self.code = code
        self.retryable = retryable
        self.unsafe_tensor_ownership = unsafe_tensor_ownership or unsafe_output
        self.unsafe_output = unsafe_output
        self.observed_topology_version = observed_topology_version
        self.min_topology_version = min_topology_version
        self.unavailable_expert_ids = unavailable_expert_ids
        self.diagnostic = diagnostic
