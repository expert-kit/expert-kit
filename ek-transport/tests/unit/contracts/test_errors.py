"""Tests for structured Transport failure data."""

from expertkit_transport.contracts import TransportError, TransportErrorCode


def test_transport_error_preserves_recovery_fields() -> None:
    error = TransportError(
        TransportErrorCode.EXPERT_NOT_READY,
        retryable=True,
        observed_topology_version=12,
        unavailable_expert_ids=(3, 5),
        diagnostic="two experts unavailable",
    )

    assert error.code is TransportErrorCode.EXPERT_NOT_READY
    assert error.retryable is True
    assert error.observed_topology_version == 12
    assert error.unavailable_expert_ids == (3, 5)
    assert str(error) == "two experts unavailable"
