"""Tests for fixed shared-memory layout and region lifetime."""

import os

import pytest
import torch

from expertkit_transport.transports.shm.memory import SharedMemoryLayout, SharedMemoryRegion


def layout() -> SharedMemoryLayout:
    return SharedMemoryLayout(
        slot_count=2,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float16,
    )


def test_layout_fields_do_not_overlap_and_slots_are_page_aligned() -> None:
    current = layout()

    assert current.hidden_offset == 0
    assert current.expert_ids_offset >= current.hidden_bytes
    assert current.routing_weights_offset >= current.expert_ids_offset + current.routing_bytes
    assert current.output_offset >= current.routing_weights_offset + current.routing_bytes
    assert current.slot_stride >= current.output_offset + current.hidden_bytes
    assert current.slot_stride % os.sysconf("SC_PAGE_SIZE") == 0
    assert current.segment_size == 2 * current.slot_stride


def test_two_regions_share_tensor_storage_and_creator_unlinks(tmp_path) -> None:
    current = layout()
    creator = SharedMemoryRegion.create(current, device="cpu", directory=tmp_path)
    attached = SharedMemoryRegion.open(
        creator.name,
        current,
        device="cpu",
        directory=tmp_path,
    )
    creator_slot = creator.slot(1)
    attached_slot = attached.slot(1)
    creator_slot.hidden_states.fill_(3)

    torch.testing.assert_close(attached_slot.hidden_states, creator_slot.hidden_states)
    path = creator.path
    del creator_slot, attached_slot
    attached.close()
    assert path.exists()
    creator.close()
    assert not path.exists()


def test_attacher_rejects_inconsistent_size(tmp_path) -> None:
    current = layout()
    creator = SharedMemoryRegion.create(current, device="cpu", directory=tmp_path)
    wrong = SharedMemoryLayout(
        slot_count=1,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float16,
    )
    try:
        with pytest.raises(ValueError, match="size"):
            SharedMemoryRegion.open(
                creator.name,
                wrong,
                device="cpu",
                directory=tmp_path,
            )
    finally:
        creator.close()


@pytest.mark.parametrize("name", ("../escape", "other-123", "expertkit-not-hex"))
def test_attacher_rejects_untrusted_names(tmp_path, name: str) -> None:
    with pytest.raises(ValueError, match="segment_name"):
        SharedMemoryRegion.open(name, layout(), device="cpu", directory=tmp_path)
