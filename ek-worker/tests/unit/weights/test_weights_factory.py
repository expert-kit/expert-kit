"""Tests for constructing Weight Manager support services."""

from expertkit_worker.weights.factory import _split_address


def test_split_address_handles_ipv4_hostnames_and_ipv6() -> None:
    assert _split_address("127.0.0.1:5000") == ("127.0.0.1", 5000)
    assert _split_address("worker.local:5001") == ("worker.local", 5001)
    assert _split_address("[2001:db8::1]:5002") == ("2001:db8::1", 5002)
