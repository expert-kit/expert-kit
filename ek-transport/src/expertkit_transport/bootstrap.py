"""Deployment configuration for internal process-level transport initialization."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import fields
from pathlib import Path

import torch
from expertkit_proto.ek.control.v2 import lifecycle_pb2

from expertkit_transport.transports.base import WorkerTransportRuntime


def runtimes_from_environment(device: torch.device) -> dict[int, WorkerTransportRuntime] | None:
    """Read an optional JSON/TOML file without changing framework constructors.

    Called in the client's own loop/thread after framework process creation.
    Native initialization remains lazy. Explicit runtime injection takes precedence.
    Device comes from the existing client; a shared deployment file cannot assign
    all DP processes to one fixed NPU. Mooncake publishes its actual bound port.
    """

    path_text = os.environ.get("EK_TRANSPORT_CONFIG")
    if not path_text:
        return None
    path = Path(path_text)
    if path.suffix == ".json":
        document = json.loads(path.read_text())
    elif path.suffix == ".toml":
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    else:
        raise ValueError("EK_TRANSPORT_CONFIG must name a .json or .toml file")
    if not isinstance(document, dict) or set(document) != {"transfer_engine"}:
        raise ValueError("transport config must contain only the transfer_engine table")
    options = document["transfer_engine"]
    if not isinstance(options, dict):
        raise ValueError("transfer_engine config must be an object/table")
    from expertkit_transport.transports.transfer_engine.runtime import (
        TransferEngineRuntime,
        TransferEngineRuntimeConfig,
    )

    allowed = {item.name for item in fields(TransferEngineRuntimeConfig)} - {"device"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unsupported transfer_engine configuration: {', '.join(sorted(unknown))}")
    options = dict(options)
    options.setdefault("metadata_server", "P2PHANDSHAKE")
    options.setdefault("client_max_in_flight", 1)
    runtime = TransferEngineRuntime(TransferEngineRuntimeConfig(device=device, **options))
    return {lifecycle_pb2.WORKER_TRANSPORT_TRANSFER_ENGINE: runtime}
