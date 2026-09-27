#!/usr/bin/env python3
"""Statically verify Expert Kit transport backend prerequisites.

The probe deliberately does not construct transports, initialize CUDA/NCCL,
open sockets, or create Mooncake engines.  It only imports Python bindings and
inspects build-time capabilities, standard host facilities, and public APIs.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

_BACKENDS = ("grpc", "shm", "nccl", "te-nvlink", "te-rdma")
_EXPECTED_MOONCAKE_VERSION = "0.3.12.dev20260817+ek.731c4521.rdma2"
_EXPECTED_TORCH_VERSION = "2.10.0+cu128"
_EXPECTED_TORCH_CUDA = "12.8"
_FORBIDDEN_TE_ENVIRONMENT = ("MC_USE_TENT", "MC_USE_TEV1", "MC_FORCE_TCP")
_COMMON_TE_CAPABILITIES = (
    "EK_SAFE_TERMINAL_BATCH_SYNC",
    "EK_HAS_GPUDIRECT_ACQUIRE",
)
_COMMON_TE_METHODS = (
    "initialize",
    "get_rpc_port",
    "register_memory",
    "unregister_memory",
    "batch_transfer_sync_read",
    "batch_transfer_sync_write",
    "flush_gpudirect_writes",
    "get_configured_backend",
)
_TE_BACKEND_REQUIREMENTS = {
    "te-nvlink": {
        "capabilities": (
            "SUPPORT_CUDA",
            "SUPPORT_INTRA_NVLINK",
            "EK_INTRA_NVLINK_REGISTRATION_REFCOUNT",
            "EK_FORCE_CONFIGURED_TRANSPORT",
            "EK_DRAINED_NVLINK_INTRA_LOCAL_INVALIDATION",
        ),
        "methods": ("invalidate_drained_nvlink_intra_segment",),
    },
    "te-rdma": {
        "capabilities": (
            "SUPPORT_CUDA",
            "EK_FORCE_CONFIGURED_RDMA_TRANSPORT",
            "EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION",
        ),
        "methods": ("invalidate_drained_rdma_segment",),
    },
}

Importer = Callable[[str], object]
VersionReader = Callable[[str], str]


def _check(name: str, ok: bool, observed: str) -> dict[str, object]:
    return {"name": name, "ok": bool(ok), "observed": observed}


def _result(backend: str, checks: Sequence[dict[str, object]]) -> dict[str, object]:
    supported = all(check["ok"] is True for check in checks)
    failed_checks = [str(check["name"]) for check in checks if check["ok"] is not True]
    return {
        "backend": backend,
        "status": "supported" if supported else "unsupported",
        "supported": supported,
        "failed_checks": failed_checks,
        "checks": list(checks),
    }


def _safe_import(importer: Importer, name: str) -> tuple[object | None, str]:
    try:
        return importer(name), "installed"
    except Exception as error:  # noqa: BLE001 - optional native modules can fail at load time
        return None, f"unavailable:{type(error).__name__}"


def _module_version(module: object) -> str:
    value = getattr(module, "__version__", None)
    return str(value) if isinstance(value, (str, int, float)) else "unknown"


def _probe_grpc(importer: Importer) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for module_name in ("grpc", "google.protobuf"):
        module, observed = _safe_import(importer, module_name)
        if module is not None:
            observed = _module_version(module)
        checks.append(
            _check(f"python_package:{module_name}", module is not None, observed)
        )
    return _result("grpc", checks)


def _probe_shm(
    *,
    system_name: str,
    shm_directory: Path,
    access: Callable[[Path, int], bool],
) -> dict[str, object]:
    linux = system_name == "Linux"
    try:
        is_directory = shm_directory.is_dir()
    except OSError:
        is_directory = False
    try:
        readable = is_directory and access(shm_directory, os.R_OK)
        writable = is_directory and access(shm_directory, os.W_OK)
    except OSError:
        readable = False
        writable = False
    return _result(
        "shm",
        (
            _check("operating_system:linux", linux, "linux" if linux else "non-linux"),
            _check(
                "standard_shm_directory",
                is_directory,
                "present" if is_directory else "missing",
            ),
            _check("standard_shm_readable", readable, "yes" if readable else "no"),
            _check("standard_shm_writable", writable, "yes" if writable else "no"),
        ),
    )


def _probe_nccl(importer: Importer) -> dict[str, object]:
    torch, torch_observed = _safe_import(importer, "torch")
    distributed, distributed_observed = _safe_import(importer, "torch.distributed")
    checks = [
        _check(
            "python_package:torch",
            torch is not None,
            _module_version(torch) if torch is not None else torch_observed,
        ),
        _check(
            "python_package:torch.distributed",
            distributed is not None,
            distributed_observed,
        ),
    ]
    if torch is None or distributed is None:
        checks.extend(
            (
                _check("torch_cuda_build", False, "not_checked"),
                _check("torch_distributed_build", False, "not_checked"),
                _check("torch_nccl_build", False, "not_checked"),
            )
        )
        return _result("nccl", checks)

    torch_version = getattr(torch, "version", None)
    cuda_version = getattr(torch_version, "cuda", None)
    distributed_available = getattr(distributed, "is_available", None)
    nccl_available = getattr(distributed, "is_nccl_available", None)
    has_distributed = (
        callable(distributed_available) and distributed_available() is True
    )
    has_nccl = callable(nccl_available) and nccl_available() is True
    checks.extend(
        (
            _check(
                "torch_cuda_build",
                isinstance(cuda_version, str) and bool(cuda_version),
                str(cuda_version) if cuda_version else "not_built",
            ),
            _check(
                "torch_distributed_build",
                has_distributed,
                "available" if has_distributed else "unavailable",
            ),
            _check(
                "torch_nccl_build",
                has_nccl,
                "available" if has_nccl else "unavailable",
            ),
        )
    )
    return _result("nccl", checks)


def _probe_transfer_engine(
    backend: str,
    *,
    importer: Importer,
    version_reader: VersionReader,
    environment: Mapping[str, str],
) -> dict[str, object]:
    enabled_overrides = [
        name for name in _FORBIDDEN_TE_ENVIRONMENT if name in environment
    ]
    torch, torch_observed = _safe_import(importer, "torch")
    engine, import_observed = _safe_import(importer, "mooncake.engine")
    torch_build = getattr(torch, "version", None) if torch is not None else None
    torch_cuda = getattr(torch_build, "cuda", None)
    checks: list[dict[str, object]] = [
        _check(
            "environment:unsupported_transport_override_absent",
            not enabled_overrides,
            "unset" if not enabled_overrides else f"set:{','.join(enabled_overrides)}",
        ),
        _check(
            "python_package:torch",
            torch is not None,
            _module_version(torch) if torch is not None else torch_observed,
        ),
        _check(
            "torch_version",
            torch is not None and _module_version(torch) == _EXPECTED_TORCH_VERSION,
            _module_version(torch) if torch is not None else "not_installed",
        ),
        _check(
            "torch_cuda_abi",
            torch_cuda == _EXPECTED_TORCH_CUDA,
            str(torch_cuda) if torch_cuda is not None else "not_built",
        ),
        _check("python_package:mooncake.engine", engine is not None, import_observed),
    ]

    try:
        installed_version = version_reader("mooncake-transfer-engine")
    except Exception as error:  # noqa: BLE001 - metadata providers have varied failures
        installed_version = f"unavailable:{type(error).__name__}"
    checks.append(
        _check(
            "mooncake_version",
            installed_version == _EXPECTED_MOONCAKE_VERSION,
            installed_version,
        )
    )

    required_capabilities = (
        *_COMMON_TE_CAPABILITIES,
        *_TE_BACKEND_REQUIREMENTS[backend]["capabilities"],
    )
    transfer_engine_type = getattr(engine, "TransferEngine", None) if engine else None
    checks.append(
        _check(
            "mooncake_api:TransferEngine",
            transfer_engine_type is not None,
            "present" if transfer_engine_type is not None else "missing",
        )
    )
    for capability in required_capabilities:
        enabled = engine is not None and getattr(engine, capability, False) is True
        checks.append(
            _check(
                f"mooncake_capability:{capability}",
                enabled,
                "true" if enabled else "missing_or_false",
            )
        )
    required_methods = (
        *_COMMON_TE_METHODS,
        *_TE_BACKEND_REQUIREMENTS[backend]["methods"],
    )
    for method in required_methods:
        present = transfer_engine_type is not None and callable(
            getattr(transfer_engine_type, method, None)
        )
        checks.append(
            _check(
                f"mooncake_api:TransferEngine.{method}",
                present,
                "present" if present else "missing",
            )
        )
    return _result(backend, checks)


def probe_backends(
    backends: Sequence[str] = _BACKENDS,
    *,
    importer: Importer = importlib.import_module,
    version_reader: VersionReader = importlib.metadata.version,
    environment: Mapping[str, str] = os.environ,
    system_name: str | None = None,
    shm_directory: Path = Path("/dev/shm"),
    access: Callable[[Path, int], bool] = os.access,
) -> dict[str, object]:
    """Return a JSON-serializable, side-effect-free backend capability report."""

    unknown = sorted(set(backends).difference(_BACKENDS))
    if unknown:
        raise ValueError(f"unknown transport backend(s): {', '.join(unknown)}")
    requested = list(dict.fromkeys(backends))
    results: dict[str, object] = {}
    for backend in requested:
        if backend == "grpc":
            result = _probe_grpc(importer)
        elif backend == "shm":
            result = _probe_shm(
                system_name=system_name or platform.system(),
                shm_directory=shm_directory,
                access=access,
            )
        elif backend == "nccl":
            result = _probe_nccl(importer)
        else:
            result = _probe_transfer_engine(
                backend,
                importer=importer,
                version_reader=version_reader,
                environment=environment,
            )
        results[backend] = result

    supported = [name for name, result in results.items() if result["supported"]]
    unsupported = [name for name, result in results.items() if not result["supported"]]
    return {
        "schema_version": 1,
        "probe_mode": "static_no_initialization",
        "requested_backends": requested,
        "all_supported": not unsupported,
        "supported_backends": supported,
        "unsupported_backends": unsupported,
        "backends": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Statically verify Expert Kit transport prerequisites without initializing "
            "GPU, NCCL, Mooncake, or network state."
        )
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=(*_BACKENDS, "all"),
        help="backend to verify; repeat for multiple backends (default: all)",
    )
    parser.add_argument("--pretty", action="store_true", help="indent the JSON output")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="return exit status 0 even when a requested backend is unsupported",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = args.backend or list(_BACKENDS)
    if "all" in selected:
        selected = list(_BACKENDS)
    report = probe_backends(selected)
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
    if report["all_supported"] or args.allow_unsupported:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
