"""Fixed Tensor layout and CUDA registration for same-host shared memory."""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

_CACHE_LINE_BYTES = 64
_MAX_SEGMENT_BYTES = 8 * 1024**3
_SHM_DIRECTORY = Path("/dev/shm")
_SEGMENT_NAME = re.compile(r"^expertkit-[0-9a-f]{32}$")
_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def new_session_id() -> str:
    """Return a random protocol-safe shared-memory session identifier."""

    return uuid.uuid4().hex


def validate_session_id(value: str) -> str:
    """Return one valid session identifier or raise a protocol error."""

    if not _SESSION_ID.fullmatch(value):
        raise ValueError("shared-memory session_id must contain 32 lowercase hex digits")
    return value


@dataclass(frozen=True, slots=True)
class SharedMemoryLayout:
    """Describe fixed input and output regions for one connection.

    Every slot holds maximum-size hidden states, expert IDs, routing weights,
    and partial output in that order. Field starts are cache-line aligned and
    slot starts are page aligned.
    """

    slot_count: int
    max_batch_tokens: int
    hidden_dim: int
    top_k: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        for name in ("slot_count", "max_batch_tokens", "hidden_dim", "top_k"):
            _positive(name, getattr(self, name))
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("shared-memory dtype must be FP16, BF16, or FP32")
        if self.segment_size > _MAX_SEGMENT_BYTES:
            raise ValueError("shared-memory segment exceeds the 8 GiB safety bound")

    @property
    def activation_element_bytes(self) -> int:
        """Return bytes per hidden-state and output element."""

        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def hidden_bytes(self) -> int:
        """Return bytes reserved for one maximum-size activation Tensor."""

        return self.max_batch_tokens * self.hidden_dim * self.activation_element_bytes

    @property
    def routing_bytes(self) -> int:
        """Return bytes reserved for one IDs or routing-weights Tensor."""

        return self.max_batch_tokens * self.top_k * 4

    @property
    def hidden_offset(self) -> int:
        """Return the slot-relative hidden-state offset."""

        return 0

    @property
    def expert_ids_offset(self) -> int:
        """Return the slot-relative expert-ID offset."""

        return _align(self.hidden_bytes, _CACHE_LINE_BYTES)

    @property
    def routing_weights_offset(self) -> int:
        """Return the slot-relative routing-weight offset."""

        return _align(self.expert_ids_offset + self.routing_bytes, _CACHE_LINE_BYTES)

    @property
    def output_offset(self) -> int:
        """Return the slot-relative partial-output offset."""

        return _align(self.routing_weights_offset + self.routing_bytes, _CACHE_LINE_BYTES)

    @property
    def slot_stride(self) -> int:
        """Return page-aligned bytes reserved for each reusable slot."""

        return _align(self.output_offset + self.hidden_bytes, mmap.PAGESIZE)

    @property
    def segment_size(self) -> int:
        """Return the exact shared-memory file size."""

        return self.slot_count * self.slot_stride

    def slot_offset(self, slot_index: int) -> int:
        """Return one validated slot's segment-relative start."""

        if (
            isinstance(slot_index, bool)
            or not isinstance(slot_index, int)
            or not 0 <= slot_index < self.slot_count
        ):
            raise ValueError("shared-memory slot index is outside the layout")
        return slot_index * self.slot_stride


@dataclass(slots=True)
class SharedMemorySlot:
    """Hold maximum-size Tensor views into one shared-memory slot."""

    index: int
    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    partial_output: torch.Tensor


class _CudaHostRegistration:
    """Register one existing mapping with the CUDA runtime exactly once."""

    def __init__(self, pointer: int, size: int, device: torch.device) -> None:
        self._pointer = pointer
        self._device = device
        self._library = self._load_runtime()
        register = self._library.cudaHostRegister
        register.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
        register.restype = ctypes.c_int
        unregister = self._library.cudaHostUnregister
        unregister.argtypes = (ctypes.c_void_p,)
        unregister.restype = ctypes.c_int

        with torch.cuda.device(device):
            torch.cuda.init()
            result = register(ctypes.c_void_p(pointer), size, 1)
        if result != 0:
            raise RuntimeError(
                f"cudaHostRegister failed for shared memory: {self._error_text(result)}"
            )
        self._registered = True

    def close(self) -> None:
        """Unregister the mapping; repeated calls are safe."""

        if not self._registered:
            return
        with torch.cuda.device(self._device):
            result = self._library.cudaHostUnregister(ctypes.c_void_p(self._pointer))
        self._registered = False
        if result != 0:
            raise RuntimeError(
                f"cudaHostUnregister failed for shared memory: {self._error_text(result)}"
            )

    @staticmethod
    def _load_runtime() -> ctypes.CDLL:
        torch_root = Path(torch.__file__).resolve().parent.parent
        cuda_major = (torch.version.cuda or "").split(".", 1)[0]
        candidates: list[Path | str] = []
        if cuda_major:
            candidates.extend(
                sorted(
                    torch_root.glob(f"nvidia/cu{cuda_major}/lib/libcudart.so*"),
                    reverse=True,
                )
            )
        candidates.extend(
            sorted(
                torch_root.glob("nvidia/cuda_runtime/lib/libcudart.so*"),
                reverse=True,
            )
        )
        discovered = ctypes.util.find_library("cudart")
        if discovered is not None:
            candidates.append(discovered)
        for candidate in candidates:
            try:
                return ctypes.CDLL(str(candidate))
            except OSError:
                continue
        raise RuntimeError("cannot load the CUDA runtime required for shared memory")

    def _error_text(self, result: int) -> str:
        get_error = self._library.cudaGetErrorString
        get_error.argtypes = (ctypes.c_int,)
        get_error.restype = ctypes.c_char_p
        message = get_error(result)
        return message.decode("utf-8", errors="replace") if message else f"error {result}"


class SharedMemoryRegion:
    """Own or attach to one fixed Expert Kit shared-memory file."""

    def __init__(
        self,
        *,
        name: str,
        path: Path,
        file_descriptor: int,
        mapping: mmap.mmap,
        layout: SharedMemoryLayout,
        owner: bool,
        device: torch.device,
    ) -> None:
        self.name = name
        self.path = path
        self.layout = layout
        self._file_descriptor = file_descriptor
        self._mapping: mmap.mmap | None = mapping
        self._owner = owner
        self._registration: _CudaHostRegistration | None = None
        self._closed = False
        if device.type == "cuda":
            pointer = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            self._registration = _CudaHostRegistration(pointer, layout.segment_size, device)

    @classmethod
    def create(
        cls,
        layout: SharedMemoryLayout,
        *,
        device: torch.device | str,
        directory: Path = _SHM_DIRECTORY,
    ) -> SharedMemoryRegion:
        """Create, size, map, and optionally pin one private shared-memory file."""

        resolved_device = torch.device(device)
        name = f"expertkit-{uuid.uuid4().hex}"
        path = directory / name
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        file_descriptor = os.open(path, flags, 0o600)
        mapping: mmap.mmap | None = None
        try:
            os.fchmod(file_descriptor, 0o600)
            os.ftruncate(file_descriptor, layout.segment_size)
            mapping = mmap.mmap(file_descriptor, layout.segment_size, access=mmap.ACCESS_WRITE)
            return cls(
                name=name,
                path=path,
                file_descriptor=file_descriptor,
                mapping=mapping,
                layout=layout,
                owner=True,
                device=resolved_device,
            )
        except BaseException:
            if mapping is not None:
                mapping.close()
            os.close(file_descriptor)
            path.unlink(missing_ok=True)
            raise

    @classmethod
    def open(
        cls,
        name: str,
        layout: SharedMemoryLayout,
        *,
        device: torch.device | str,
        directory: Path = _SHM_DIRECTORY,
    ) -> SharedMemoryRegion:
        """Open one creator-owned shared-memory file after strict validation."""

        if not _SEGMENT_NAME.fullmatch(name):
            raise ValueError("shared-memory segment_name is invalid")
        path = directory / name
        flags = os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        file_descriptor = os.open(path, flags)
        mapping: mmap.mmap | None = None
        try:
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("shared-memory segment must be a regular tmpfs file")
            if metadata.st_uid != os.geteuid():
                raise ValueError("shared-memory segment must be owned by the Worker user")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise ValueError("shared-memory segment permissions must not allow group access")
            if metadata.st_size != layout.segment_size:
                raise ValueError("shared-memory segment has an inconsistent size")
            mapping = mmap.mmap(file_descriptor, layout.segment_size, access=mmap.ACCESS_WRITE)
            return cls(
                name=name,
                path=path,
                file_descriptor=file_descriptor,
                mapping=mapping,
                layout=layout,
                owner=False,
                device=torch.device(device),
            )
        except BaseException:
            if mapping is not None:
                mapping.close()
            os.close(file_descriptor)
            raise

    def slot(self, slot_index: int) -> SharedMemorySlot:
        """Return fixed maximum-size Tensor views for one slot."""

        mapping = self._require_mapping()
        base = self.layout.slot_offset(slot_index)
        hidden_shape = (self.layout.max_batch_tokens, self.layout.hidden_dim)
        routing_shape = (self.layout.max_batch_tokens, self.layout.top_k)
        return SharedMemorySlot(
            index=slot_index,
            hidden_states=torch.frombuffer(
                mapping,
                dtype=self.layout.dtype,
                count=self.layout.max_batch_tokens * self.layout.hidden_dim,
                offset=base + self.layout.hidden_offset,
            ).reshape(hidden_shape),
            expert_ids=torch.frombuffer(
                mapping,
                dtype=torch.int32,
                count=self.layout.max_batch_tokens * self.layout.top_k,
                offset=base + self.layout.expert_ids_offset,
            ).reshape(routing_shape),
            routing_weights=torch.frombuffer(
                mapping,
                dtype=torch.float32,
                count=self.layout.max_batch_tokens * self.layout.top_k,
                offset=base + self.layout.routing_weights_offset,
            ).reshape(routing_shape),
            partial_output=torch.frombuffer(
                mapping,
                dtype=self.layout.dtype,
                count=self.layout.max_batch_tokens * self.layout.hidden_dim,
                offset=base + self.layout.output_offset,
            ).reshape(hidden_shape),
        )

    def unlink(self) -> None:
        """Remove the name while keeping existing mappings valid."""

        self.path.unlink(missing_ok=True)

    def close(self) -> None:
        """Unpin, unmap, close, and optionally unlink the region."""

        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        if self._registration is not None:
            try:
                self._registration.close()
            except BaseException as cause:
                error = cause
            self._registration = None
        mapping = self._mapping
        self._mapping = None
        if mapping is not None:
            try:
                mapping.close()
            except BaseException as cause:
                if error is None:
                    error = cause
        try:
            os.close(self._file_descriptor)
        except OSError as cause:
            if error is None:
                error = cause
        if self._owner:
            try:
                self.unlink()
            except OSError as cause:
                if error is None:
                    error = cause
        if error is not None:
            raise error

    def _require_mapping(self) -> mmap.mmap:
        if self._mapping is None:
            raise RuntimeError("shared-memory region is closed")
        return self._mapping
