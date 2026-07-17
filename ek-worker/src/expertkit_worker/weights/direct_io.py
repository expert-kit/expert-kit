"""Linux direct-I/O buffers and atomic persistent-cache file operations."""

from __future__ import annotations

import mmap
import os
import stat
import sys
import uuid
from contextlib import suppress
from pathlib import Path

_TEMPORARY_PREFIX = ".ek-weight-tmp-"
_PROBE_PREFIX = ".ek-weight-probe-"


class DirectIOError(OSError):
    """Report that strict application-managed direct I/O could not complete."""


class InvalidWeightFile(DirectIOError):
    """Report a local file that cannot be a complete accepted expert weight."""


def _require_direct_io() -> int:
    if sys.platform != "linux" or not hasattr(os, "O_DIRECT"):
        raise DirectIOError("direct weight I/O requires Linux os.O_DIRECT support")
    return os.O_DIRECT


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class AlignedWeightBuffer:
    """Own page-aligned anonymous memory with logical and I/O-sized views."""

    def __init__(self, logical_size: int, *, alignment: int = mmap.PAGESIZE) -> None:
        if isinstance(logical_size, bool) or not isinstance(logical_size, int) or logical_size <= 0:
            raise ValueError("logical_size must be a positive integer")
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            raise ValueError("alignment must be a positive power of two")
        self.logical_size = logical_size
        self.alignment = alignment
        self.allocation_size = _round_up(logical_size, alignment)
        self._mapping: mmap.mmap | None = mmap.mmap(-1, self.allocation_size)

    def view(self) -> memoryview:
        """Return the logical bytes retained by the DRAM cache."""

        return self._require_mapping_view()[: self.logical_size]

    def io_view(self) -> memoryview:
        """Return the aligned allocation including zero padding for direct I/O."""

        return self._require_mapping_view()

    def trim(self, logical_size: int) -> None:
        """Reduce the logical length after a bounded streaming download."""

        if (
            isinstance(logical_size, bool)
            or not isinstance(logical_size, int)
            or not 0 < logical_size <= self.logical_size
        ):
            raise ValueError("trimmed logical_size must be positive and within the buffer")
        self.logical_size = logical_size

    def close(self) -> None:
        """Release the anonymous mapping after all derived views are gone."""

        mapping = self._mapping
        if mapping is None:
            return
        mapping.close()
        self._mapping = None

    def _require_mapping_view(self) -> memoryview:
        mapping = self._mapping
        if mapping is None:
            raise RuntimeError("aligned weight buffer is closed")
        return memoryview(mapping)

    def __del__(self) -> None:
        with suppress(BufferError):
            self.close()


def read_direct(path: Path, *, max_bytes: int | None = None) -> AlignedWeightBuffer:
    """Read one complete regular file into aligned anonymous memory with O_DIRECT."""

    direct_flag = _require_direct_io()
    file_descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | direct_flag)
    result: AlignedWeightBuffer | None = None
    try:
        file_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise InvalidWeightFile("direct weight source must be a regular file")
        logical_size = file_stat.st_size
        if logical_size <= 0:
            raise InvalidWeightFile("direct weight source must not be empty")
        if max_bytes is not None:
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
                raise ValueError("max_bytes must be a positive integer")
            if logical_size > max_bytes:
                raise InvalidWeightFile("direct weight source exceeds its configured byte limit")

        result = AlignedWeightBuffer(logical_size)
        target = result.io_view()
        try:
            total = 0
            while total < logical_size:
                if total % result.alignment:
                    raise InvalidWeightFile("direct weight read returned an unaligned short result")
                count = os.preadv(file_descriptor, [target[total:]], total)
                if count <= 0:
                    raise InvalidWeightFile("direct weight read ended before the file was complete")
                total += count
            if total != logical_size:
                raise InvalidWeightFile("direct weight read exceeded the logical file length")
        finally:
            target.release()
        return result
    except BaseException:
        if result is not None:
            result.close()
        raise
    finally:
        os.close(file_descriptor)


def write_direct_atomic(path: Path, source: AlignedWeightBuffer) -> None:
    """Durably publish one aligned buffer without using buffered file writes."""

    direct_flag = _require_direct_io()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f"{_TEMPORARY_PREFIX}{os.getpid()}-{uuid.uuid4().hex}"
    file_descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | direct_flag,
        0o600,
    )
    published = False
    try:
        content = source.io_view()
        try:
            total = 0
            while total < source.allocation_size:
                if total % source.alignment:
                    raise DirectIOError("direct weight write returned an unaligned short result")
                count = os.pwritev(file_descriptor, [content[total:]], total)
                if count <= 0:
                    raise DirectIOError("direct weight write made no progress")
                total += count
        finally:
            content.release()
        os.ftruncate(file_descriptor, source.logical_size)
        os.fsync(file_descriptor)
        os.close(file_descriptor)
        file_descriptor = -1
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        published = True
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if not published:
            with suppress(FileNotFoundError):
                temporary.unlink()


def cleanup_temporary_files(directory: Path) -> int:
    """Remove abandoned writeback and startup-probe files.

    Temporary files owned by a live process are left in place because Workers
    serving the same model may share a disk-cache directory.
    """

    if not directory.exists():
        return 0
    removed = 0
    for prefix in (_TEMPORARY_PREFIX, _PROBE_PREFIX):
        for path in directory.rglob(f"{prefix}*"):
            if not path.is_file() or _temporary_file_has_live_owner(path, prefix):
                continue
            with suppress(FileNotFoundError):
                path.unlink()
                removed += 1
    return removed


def initialize_direct_io_directory(directory: Path) -> None:
    """Create, clean, and validate one strict direct-I/O cache directory."""

    directory.mkdir(parents=True, exist_ok=True)
    cleanup_temporary_files(directory)
    probe_path = directory / f"{_PROBE_PREFIX}{os.getpid()}-{uuid.uuid4().hex}"
    source = AlignedWeightBuffer(mmap.PAGESIZE)
    loaded: AlignedWeightBuffer | None = None
    expected = bytes(index % 251 for index in range(source.logical_size))
    source_view = source.view()
    try:
        source_view[:] = expected
    finally:
        source_view.release()
    try:
        write_direct_atomic(probe_path, source)
        loaded = read_direct(probe_path, max_bytes=source.logical_size)
        loaded_view = loaded.view()
        try:
            if bytes(loaded_view) != expected:
                raise DirectIOError("direct-I/O startup probe returned different bytes")
        finally:
            loaded_view.release()
    finally:
        if loaded is not None:
            loaded.close()
        source.close()
        with suppress(FileNotFoundError):
            probe_path.unlink()


def _temporary_file_has_live_owner(path: Path, prefix: str) -> bool:
    owner_text = path.name.removeprefix(prefix).partition("-")[0]
    try:
        owner_pid = int(owner_text)
    except ValueError:
        return False
    if owner_pid <= 0:
        return False
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def expert_file_path(root: Path, model_name: str, layer_id: int, expert_id: int) -> Path:
    """Return the existing per-model, per-expert cache path."""

    if not model_name:
        raise ValueError("model_name must not be empty")
    for name, value in (("layer_id", layer_id), ("expert_id", expert_id)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    return root / model_name / f"l{layer_id}-e{expert_id}"
