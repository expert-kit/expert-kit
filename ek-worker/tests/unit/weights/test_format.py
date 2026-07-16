"""Tests for bounded zero-copy SafeTensors parsing."""

from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import load as official_load
from safetensors.torch import save as official_save

from expertkit_worker.weights import (
    MAX_SAFETENSORS_HEADER_BYTES,
    SafeTensorDType,
    SafeTensorFormatError,
    max_safetensors_file_bytes,
    parse_safetensors,
)


def raw_file(header: dict[str, object], data: bytes = b"") -> bytes:
    """Encode a compact test file without applying SafeTensors validation."""

    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(encoded)) + encoded + data


def test_complete_file_bound_adds_prefix_header_and_tensor_bytes() -> None:
    assert max_safetensors_file_bytes(100) == 8 + MAX_SAFETENSORS_HEADER_BYTES + 100
    with pytest.raises(ValueError, match="positive integer"):
        max_safetensors_file_bytes(0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_parser_matches_official_library_without_copying_tensor_data(
    dtype: torch.dtype,
) -> None:
    expected = {
        "model.experts.2.gate_proj.weight": torch.arange(12, dtype=dtype).reshape(3, 4),
        "model.experts.2.up_proj.weight": torch.arange(12, 24, dtype=dtype).reshape(3, 4),
        "model.experts.2.down_proj.weight": torch.arange(12, dtype=dtype).reshape(4, 3),
    }
    owned = bytearray(official_save(expected, metadata={"model": "fixture"}))

    parsed = parse_safetensors(owned)

    assert parsed.buffer.obj is owned
    assert parsed.metadata == {"model": "fixture"}
    assert set(parsed.tensors) == set(expected)
    oracle = official_load(bytes(owned))
    for name, region in parsed.tensors.items():
        assert region.dtype is SafeTensorDType(dtype_name(dtype))
        assert region.shape == tuple(expected[name].shape)
        actual = torch.frombuffer(region.data, dtype=dtype).reshape(region.shape)
        torch.testing.assert_close(actual, oracle[name])


def dtype_name(dtype: torch.dtype) -> str:
    """Return the SafeTensors spelling used by the parser fixture."""

    return {
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
    }[dtype]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"short", "shorter than"),
        (struct.pack("<Q", 17) + b"{}", "exceeds the input"),
        (struct.pack("<Q", 1), "supported bound"),
        (struct.pack("<Q", 2) + b"\xff}", "UTF-8"),
        (struct.pack("<Q", 2) + b"[]", "JSON object"),
        (raw_file({}), "contains no Tensors"),
    ],
)
def test_parser_rejects_invalid_file_envelopes(payload: bytes, match: str) -> None:
    with pytest.raises(SafeTensorFormatError, match=match):
        parse_safetensors(payload)


def test_parser_rejects_duplicate_json_keys() -> None:
    tensor = '{"dtype":"F32","shape":[0],"data_offsets":[0,0]}'
    header = f'{{"weight":{tensor},"weight":{tensor}}}'.encode()
    payload = struct.pack("<Q", len(header)) + header

    with pytest.raises(SafeTensorFormatError, match="duplicate"):
        parse_safetensors(payload)


@pytest.mark.parametrize(
    ("entry", "data", "match"),
    [
        (
            {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]},
            b"1234",
            "unsupported dtype",
        ),
        (
            {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]},
            b"1234",
            "byte length is inconsistent",
        ),
        (
            {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
            b"123456",
            "unexplained hole",
        ),
        (
            {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            b"12345678",
            "trailing bytes",
        ),
    ],
)
def test_parser_rejects_invalid_tensor_ranges(
    entry: dict[str, object],
    data: bytes,
    match: str,
) -> None:
    with pytest.raises(SafeTensorFormatError, match=match):
        parse_safetensors(raw_file({"weight": entry}, data))


def test_parser_rejects_overlap_and_invalid_metadata() -> None:
    header = {
        "first": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "second": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
    }
    with pytest.raises(SafeTensorFormatError, match="overlaps"):
        parse_safetensors(raw_file(header, b"123456"))

    header["second"] = {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}
    header["__metadata__"] = {"number": 7}
    with pytest.raises(SafeTensorFormatError, match="map strings to strings"):
        parse_safetensors(raw_file(header, b"12345678"))


def test_suffix_lookup_rejects_missing_and_ambiguous_weights() -> None:
    one = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    parsed = parse_safetensors(raw_file({"a.gate_proj.weight": one}, b"1234"))

    with pytest.raises(SafeTensorFormatError, match="missing"):
        parsed.find_unique_suffix(("up_proj.weight",))

    header = {
        "a.gate_proj.weight": one,
        "b.gate_proj.weight": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
    }
    parsed = parse_safetensors(raw_file(header, b"12345678"))
    with pytest.raises(SafeTensorFormatError, match="ambiguous"):
        parsed.find_unique_suffix(("gate_proj.weight",))
