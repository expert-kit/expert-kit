"""Tests for Worker-side shared-memory slot ownership and validation."""

import pytest
import torch

from expertkit_transport.errors import TransportProtocolError
from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.shm.codec import ExecuteSlot, OpenSession
from expertkit_transport.transports.shm.memory import SharedMemoryLayout, SharedMemoryRegion
from expertkit_transport.transports.shm.session import (
    SharedMemorySlotBusy,
    WorkerSharedMemorySession,
)


def spec() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(7, 4, 8, 4, 3, 2, torch.float32)


def test_session_claims_valid_rows_and_releases_each_generation() -> None:
    layout = SharedMemoryLayout(2, 4, 3, 2, torch.float32)
    creator = SharedMemoryRegion.create(layout, device="cpu")
    slot = creator.slot(0)
    slot.hidden_states[:2].copy_(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    slot.expert_ids[:2].copy_(torch.tensor([[1, -1], [0, 3]], dtype=torch.int32))
    slot.routing_weights[:2].copy_(torch.tensor([[1.0, 0.0], [0.25, 0.75]]))
    session = WorkerSharedMemorySession(
        OpenSession("a" * 32, creator.name, layout),
        spec=spec(),
        device="cpu",
    )
    request = ExecuteSlot("a" * 32, 0, 1, 2, 9, 2, 5_000_000)
    try:
        claimed = session.claim(request)
        assert claimed.batch.distinct_expert_ids == (0, 1, 3)
        torch.testing.assert_close(claimed.batch.hidden_states, slot.hidden_states[:2])
        claimed.output_destination.fill_(7)
        torch.testing.assert_close(slot.partial_output[:2], torch.full((2, 3), 7.0))

        with pytest.raises(SharedMemorySlotBusy):
            session.claim(ExecuteSlot("a" * 32, 0, 2, 2, 9, 2, 5_000_000))
        session.release(0, 1)
        with pytest.raises(TransportProtocolError, match="stale"):
            session.claim(request)
        second = session.claim(ExecuteSlot("a" * 32, 0, 2, 2, 9, 2, 5_000_000))
        session.release(second.slot_index, second.generation)
    finally:
        session.close()
        del slot
        creator.close()


def test_session_rejects_invalid_network_routing_and_releases_slot() -> None:
    layout = SharedMemoryLayout(1, 4, 3, 2, torch.float32)
    creator = SharedMemoryRegion.create(layout, device="cpu")
    slot = creator.slot(0)
    slot.expert_ids[0].copy_(torch.tensor([-1, 2], dtype=torch.int32))
    slot.routing_weights[0].copy_(torch.tensor([0.5, 0.5]))
    session = WorkerSharedMemorySession(
        OpenSession("a" * 32, creator.name, layout),
        spec=spec(),
        device="cpu",
    )
    try:
        with pytest.raises(TransportProtocolError, match="zero routing weight"):
            session.claim(ExecuteSlot("a" * 32, 0, 1, 0, 1, 1, 5_000_000))
        assert session.active_count == 0
    finally:
        session.close()
        del slot
        creator.close()
