"""Tests for the Worker Transport abstract interface."""

import pytest

from expertkit_transport.contracts import WorkerTransport


class IncompleteTransport(WorkerTransport):
    pass


def test_transport_cannot_omit_lifecycle_or_submission_methods() -> None:
    with pytest.raises(TypeError, match="abstract"):
        IncompleteTransport()
