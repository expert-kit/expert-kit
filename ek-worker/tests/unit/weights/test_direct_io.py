"""Linux direct-I/O tests for persistent expert-cache files."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from expertkit_worker.weights.direct_io import (
    AlignedWeightBuffer,
    DirectIOError,
    cleanup_temporary_files,
    expert_file_path,
    read_direct,
    write_direct_atomic,
)


def make_buffer(payload: bytes) -> AlignedWeightBuffer:
    """Return one aligned buffer filled with the test payload."""

    result = AlignedWeightBuffer(len(payload))
    view = result.view()
    try:
        view[:] = payload
    finally:
        view.release()
    return result


def require_direct_io(path: Path, payload: bytes) -> None:
    """Write one probe or skip when the test filesystem lacks O_DIRECT."""

    probe = make_buffer(payload)
    try:
        write_direct_atomic(path, probe)
    except OSError as error:
        if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            pytest.skip("test filesystem does not support strict direct I/O")
        raise
    finally:
        probe.close()


@pytest.mark.direct_io
def test_direct_read_and_atomic_write_preserve_unaligned_logical_length(tmp_path: Path) -> None:
    payload = bytes(range(251)) * 19
    path = tmp_path / "model" / "l2-e7"
    require_direct_io(path, payload)

    loaded = read_direct(path)
    view = loaded.view()
    try:
        assert bytes(view) == payload
        assert loaded.logical_size == len(payload)
        assert loaded.allocation_size % loaded.alignment == 0
    finally:
        view.release()
        loaded.close()


@pytest.mark.direct_io
def test_direct_read_enforces_maximum_file_size(tmp_path: Path) -> None:
    payload = b"weight" * 1_000
    path = tmp_path / "weight"
    require_direct_io(path, payload)

    with pytest.raises(DirectIOError, match="byte limit"):
        read_direct(path, max_bytes=len(payload) - 1)


def test_aligned_buffer_does_not_close_while_a_view_is_alive() -> None:
    buffer = make_buffer(b"weight")
    view = buffer.view()

    with pytest.raises(BufferError):
        buffer.close()

    view.release()
    buffer.close()
    with pytest.raises(RuntimeError, match="closed"):
        buffer.view()


def test_aligned_buffer_trims_only_within_its_current_logical_length() -> None:
    buffer = make_buffer(b"weight-data")
    try:
        buffer.trim(6)
        assert buffer.logical_size == 6
        view = buffer.view()
        try:
            assert bytes(view) == b"weight"
        finally:
            view.release()

        with pytest.raises(ValueError, match="within the buffer"):
            buffer.trim(7)
        with pytest.raises(ValueError, match="within the buffer"):
            buffer.trim(0)
    finally:
        buffer.close()


def test_cleanup_removes_only_weight_temporary_files(tmp_path: Path) -> None:
    nested = tmp_path / "model"
    nested.mkdir()
    abandoned = nested / ".ek-weight-tmp-dead"
    other = nested / "keep"
    abandoned.write_bytes(b"partial")
    other.write_bytes(b"complete")

    assert cleanup_temporary_files(tmp_path) == 1
    assert abandoned.exists() is False
    assert other.read_bytes() == b"complete"


def test_expert_file_path_keeps_existing_layout(tmp_path: Path) -> None:
    assert expert_file_path(tmp_path, "Qwen", 3, 9) == tmp_path / "Qwen" / "l3-e9"
