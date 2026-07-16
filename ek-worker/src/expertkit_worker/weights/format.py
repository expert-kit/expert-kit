"""Bounded SafeTensors parsing over Weight Manager-owned memory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_MAX_TENSOR_BYTES = (1 << 63) - 1


class SafeTensorFormatError(ValueError):
    """Reject malformed or unsupported SafeTensors input."""


def max_safetensors_file_bytes(tensor_bytes: int) -> int:
    """Return the parser's complete-file bound for known Tensor data bytes."""

    if isinstance(tensor_bytes, bool) or not isinstance(tensor_bytes, int) or tensor_bytes <= 0:
        raise ValueError("tensor_bytes must be a positive integer")
    return 8 + MAX_SAFETENSORS_HEADER_BYTES + tensor_bytes


class SafeTensorDType(StrEnum):
    """Unquantized MVP weight dtypes and their SafeTensors names."""

    FP16 = "F16"
    BF16 = "BF16"
    FP32 = "F32"

    @property
    def element_bytes(self) -> int:
        """Return the fixed encoded element width."""

        if self is SafeTensorDType.FP32:
            return 4
        return 2


@dataclass(frozen=True, slots=True)
class SafeTensorRegion:
    """Describe one validated Tensor view into an owned SafeTensors buffer."""

    name: str
    dtype: SafeTensorDType
    shape: tuple[int, ...]
    data: memoryview

    @property
    def byte_count(self) -> int:
        """Return this Tensor's validated encoded byte length."""

        return self.data.nbytes


@dataclass(frozen=True, slots=True)
class SafeTensorData:
    """Retain one complete SafeTensors buffer and its validated Tensor regions."""

    buffer: memoryview
    tensors: Mapping[str, SafeTensorRegion]
    metadata: Mapping[str, str]

    def find_unique_suffix(self, suffixes: tuple[str, ...]) -> SafeTensorRegion:
        """Return the only Tensor whose name ends in one accepted suffix."""

        matches = [
            tensor
            for name, tensor in self.tensors.items()
            if any(name.endswith(suffix) for suffix in suffixes)
        ]
        if not matches:
            raise SafeTensorFormatError(
                f"required weight Tensor with suffix {suffixes!r} is missing"
            )
        if len(matches) != 1:
            raise SafeTensorFormatError(f"weight Tensor suffix {suffixes!r} is ambiguous")
        return matches[0]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SafeTensorFormatError(f"duplicate SafeTensors header key: {key}")
        result[key] = value
    return result


def _require_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SafeTensorFormatError(f"{description} must be an integer")
    return value


def _checked_element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        if dimension < 0:
            raise SafeTensorFormatError("SafeTensors shape dimensions must be nonnegative")
        if dimension != 0 and count > _MAX_TENSOR_BYTES // dimension:
            raise SafeTensorFormatError("SafeTensors shape product overflows the supported range")
        count *= dimension
    return count


def _parse_tensor(
    name: str,
    value: object,
    data: memoryview,
) -> tuple[SafeTensorRegion, int, int]:
    if not name:
        raise SafeTensorFormatError("SafeTensors Tensor names must not be empty")
    if not isinstance(value, dict):
        raise SafeTensorFormatError(f"SafeTensors entry {name!r} must be an object")
    if set(value) != {"dtype", "shape", "data_offsets"}:
        raise SafeTensorFormatError(f"SafeTensors entry {name!r} has invalid fields")

    raw_dtype = value["dtype"]
    if not isinstance(raw_dtype, str):
        raise SafeTensorFormatError(f"SafeTensors entry {name!r} dtype must be a string")
    try:
        dtype = SafeTensorDType(raw_dtype)
    except ValueError as error:
        raise SafeTensorFormatError(
            f"SafeTensors entry {name!r} uses unsupported dtype {raw_dtype!r}"
        ) from error

    raw_shape = value["shape"]
    if not isinstance(raw_shape, list):
        raise SafeTensorFormatError(f"SafeTensors entry {name!r} shape must be an array")
    shape = tuple(
        _require_integer(dimension, f"SafeTensors entry {name!r} shape dimension")
        for dimension in raw_shape
    )
    element_count = _checked_element_count(shape)
    if element_count > _MAX_TENSOR_BYTES // dtype.element_bytes:
        raise SafeTensorFormatError(f"SafeTensors entry {name!r} byte length overflows")
    expected_bytes = element_count * dtype.element_bytes

    offsets = value["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise SafeTensorFormatError(
            f"SafeTensors entry {name!r} data_offsets must contain two integers"
        )
    start = _require_integer(offsets[0], f"SafeTensors entry {name!r} start offset")
    end = _require_integer(offsets[1], f"SafeTensors entry {name!r} end offset")
    if start < 0 or end < start or end > data.nbytes:
        raise SafeTensorFormatError(f"SafeTensors entry {name!r} offsets are out of bounds")
    if end - start != expected_bytes:
        raise SafeTensorFormatError(f"SafeTensors entry {name!r} byte length is inconsistent")
    return SafeTensorRegion(name, dtype, shape, data[start:end]), start, end


def _parse_metadata(value: object) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise SafeTensorFormatError("SafeTensors __metadata__ must map strings to strings")
    return MappingProxyType(dict(value))


def parse_safetensors(buffer: object) -> SafeTensorData:
    """Validate SafeTensors structure without copying its Tensor data.

    Args:
        buffer: Contiguous object supporting Python's buffer protocol. The caller
            keeps ownership through the returned memory views.

    Returns:
        Immutable metadata and zero-copy Tensor regions into the original buffer.

    Raises:
        SafeTensorFormatError: The header, metadata, ranges, dtype, or lengths are
            malformed or outside the unquantized MVP format.
    """

    try:
        view = memoryview(buffer)
        if not view.c_contiguous:
            raise SafeTensorFormatError("SafeTensors input buffer must be contiguous")
        view = view.cast("B")
    except (TypeError, ValueError) as error:
        if isinstance(error, SafeTensorFormatError):
            raise
        raise SafeTensorFormatError("SafeTensors input must support a byte buffer") from error
    if view.nbytes < 8:
        raise SafeTensorFormatError("SafeTensors input is shorter than its length prefix")

    header_bytes = int.from_bytes(view[:8], byteorder="little", signed=False)
    if not 2 <= header_bytes <= MAX_SAFETENSORS_HEADER_BYTES:
        raise SafeTensorFormatError("SafeTensors header length is outside the supported bound")
    data_start = 8 + header_bytes
    if data_start > view.nbytes:
        raise SafeTensorFormatError("SafeTensors header length exceeds the input buffer")
    try:
        header_text = bytes(view[8:data_start]).decode("utf-8")
    except UnicodeDecodeError as error:
        raise SafeTensorFormatError("SafeTensors header is not valid UTF-8") from error
    if not header_text.startswith("{"):
        raise SafeTensorFormatError("SafeTensors header must begin with a JSON object")
    try:
        header = json.loads(header_text, object_pairs_hook=_unique_object)
    except SafeTensorFormatError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise SafeTensorFormatError("SafeTensors header is not valid JSON") from error
    if not isinstance(header, dict):
        raise SafeTensorFormatError("SafeTensors header must be a JSON object")

    metadata = _parse_metadata(header.pop("__metadata__", None))
    data = view[data_start:]
    tensors: dict[str, SafeTensorRegion] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, value in header.items():
        if not isinstance(name, str):
            raise SafeTensorFormatError("SafeTensors Tensor names must be strings")
        tensor, start, end = _parse_tensor(name, value, data)
        tensors[name] = tensor
        ranges.append((start, end, name))
    if not tensors:
        raise SafeTensorFormatError("SafeTensors input contains no Tensors")

    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            relation = "overlaps another Tensor" if start < cursor else "leaves an unexplained hole"
            raise SafeTensorFormatError(f"SafeTensors entry {name!r} {relation}")
        cursor = end
    if cursor != data.nbytes:
        raise SafeTensorFormatError("SafeTensors data leaves unexplained trailing bytes")

    return SafeTensorData(
        buffer=view,
        tensors=MappingProxyType(tensors),
        metadata=metadata,
    )
