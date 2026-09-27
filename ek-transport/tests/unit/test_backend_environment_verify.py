"""Static transport environment preflight checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "verify-transport-backends.py"
_SPEC = importlib.util.spec_from_file_location("verify_transport_backends", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)


class _FakePath:
    def __init__(self, *, is_directory: bool) -> None:
        self._is_directory = is_directory

    def is_dir(self) -> bool:
        return self._is_directory


def _importer(modules: dict[str, object]):
    def load(name: str) -> object:
        try:
            return modules[name]
        except KeyError as error:
            raise ModuleNotFoundError(name) from error

    return load


def _method() -> None:
    return None


def _torch(
    *,
    cuda: str | None = verify._EXPECTED_TORCH_CUDA,
    version: str = verify._EXPECTED_TORCH_VERSION,
) -> object:
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
    )


def _complete_mooncake_engine() -> object:
    methods = {
        name: _method
        for name in (
            *verify._COMMON_TE_METHODS,
            "invalidate_drained_nvlink_intra_segment",
            "invalidate_drained_rdma_segment",
        )
    }
    engine_type = type("TransferEngine", (), methods)
    capabilities = {
        name: True
        for requirements in verify._TE_BACKEND_REQUIREMENTS.values()
        for name in requirements["capabilities"]
    }
    capabilities.update({name: True for name in verify._COMMON_TE_CAPABILITIES})
    return SimpleNamespace(TransferEngine=engine_type, **capabilities)


def test_grpc_probe_only_imports_python_packages() -> None:
    imports: list[str] = []
    modules = {
        "grpc": SimpleNamespace(__version__="1.71.0"),
        "google.protobuf": SimpleNamespace(__version__="5.29.6"),
    }

    def importer(name: str) -> object:
        imports.append(name)
        return modules[name]

    report = verify.probe_backends(["grpc"], importer=importer)

    assert report["all_supported"] is True
    assert report["supported_backends"] == ["grpc"]
    assert imports == ["grpc", "google.protobuf"]


@pytest.mark.parametrize(
    ("system_name", "is_directory", "can_access", "supported"),
    [
        ("Linux", True, True, True),
        ("Darwin", True, True, False),
        ("Linux", False, False, False),
        ("Linux", True, False, False),
    ],
)
def test_shm_probe_requires_linux_and_accessible_dev_shm(
    system_name: str,
    is_directory: bool,
    can_access: bool,
    supported: bool,
) -> None:
    report = verify.probe_backends(
        ["shm"],
        system_name=system_name,
        shm_directory=_FakePath(is_directory=is_directory),
        access=lambda _path, _mode: can_access,
    )

    assert report["backends"]["shm"]["supported"] is supported


def test_nccl_probe_checks_build_flags_without_touching_cuda_runtime() -> None:
    class ExplodingCuda:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"CUDA runtime must not be probed: {name}")

    torch = SimpleNamespace(
        __version__="2.11.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=ExplodingCuda(),
    )
    distributed = SimpleNamespace(
        is_available=lambda: True,
        is_nccl_available=lambda: True,
    )
    report = verify.probe_backends(
        ["nccl"],
        importer=_importer({"torch": torch, "torch.distributed": distributed}),
    )

    assert report["backends"]["nccl"]["supported"] is True
    assert report["probe_mode"] == "static_no_initialization"


def test_missing_torch_is_structured_unsupported() -> None:
    report = verify.probe_backends(["nccl"], importer=_importer({}))

    nccl = report["backends"]["nccl"]
    assert nccl["status"] == "unsupported"
    assert "python_package:torch" in nccl["failed_checks"]
    json.dumps(report)


def test_missing_mooncake_is_structured_unsupported() -> None:
    report = verify.probe_backends(
        ["te-nvlink", "te-rdma"],
        importer=_importer({"torch": _torch()}),
        version_reader=lambda _name: (_ for _ in ()).throw(LookupError()),
        environment={},
    )

    assert report["unsupported_backends"] == ["te-nvlink", "te-rdma"]
    for backend in report["unsupported_backends"]:
        result = report["backends"][backend]
        assert result["status"] == "unsupported"
        assert "python_package:mooncake.engine" in result["failed_checks"]
    json.dumps(report)


def test_current_patched_mooncake_supports_both_te_backends_without_construction() -> None:
    engine = _complete_mooncake_engine()
    report = verify.probe_backends(
        ["te-nvlink", "te-rdma"],
        importer=_importer(
            {
                "torch": _torch(),
                "mooncake.engine": engine,
            }
        ),
        version_reader=lambda _name: verify._EXPECTED_MOONCAKE_VERSION,
        environment={},
    )

    assert report["all_supported"] is True
    assert report["supported_backends"] == ["te-nvlink", "te-rdma"]


def test_te_probe_imports_torch_before_native_mooncake() -> None:
    imports: list[str] = []
    modules = {
        "torch": _torch(),
        "mooncake.engine": _complete_mooncake_engine(),
    }

    def importer(name: str) -> object:
        imports.append(name)
        return modules[name]

    report = verify.probe_backends(
        ["te-rdma"],
        importer=importer,
        version_reader=lambda _name: verify._EXPECTED_MOONCAKE_VERSION,
        environment={},
    )

    assert report["all_supported"] is True
    assert imports == ["torch", "mooncake.engine"]


@pytest.mark.parametrize("variable", ("MC_USE_TENT", "MC_USE_TEV1", "MC_FORCE_TCP"))
def test_any_transport_override_presence_rejects_te_even_when_set_to_zero(
    variable: str,
) -> None:
    engine = _complete_mooncake_engine()
    report = verify.probe_backends(
        ["te-rdma"],
        importer=_importer(
            {
                "torch": _torch(),
                "mooncake.engine": engine,
            }
        ),
        version_reader=lambda _name: verify._EXPECTED_MOONCAKE_VERSION,
        environment={variable: "0"},
    )

    result = report["backends"]["te-rdma"]
    assert result["status"] == "unsupported"
    assert result["failed_checks"] == ["environment:unsupported_transport_override_absent"]


@pytest.mark.parametrize("cuda", [None, "12.6", "12.8.0"])
def test_te_probe_requires_exact_torch_cuda_abi(cuda: str | None) -> None:
    report = verify.probe_backends(
        ["te-rdma"],
        importer=_importer(
            {
                "torch": _torch(cuda=cuda),
                "mooncake.engine": _complete_mooncake_engine(),
            }
        ),
        version_reader=lambda _name: verify._EXPECTED_MOONCAKE_VERSION,
        environment={},
    )

    result = report["backends"]["te-rdma"]
    assert result["status"] == "unsupported"
    assert result["failed_checks"] == ["torch_cuda_abi"]


def test_te_probe_requires_the_model_benchmark_torch_build() -> None:
    report = verify.probe_backends(
        ["te-rdma"],
        importer=_importer(
            {
                "torch": _torch(version="2.11.0+cu130"),
                "mooncake.engine": _complete_mooncake_engine(),
            }
        ),
        version_reader=lambda _name: verify._EXPECTED_MOONCAKE_VERSION,
        environment={},
    )

    result = report["backends"]["te-rdma"]
    assert result["status"] == "unsupported"
    assert result["failed_checks"] == ["torch_version"]


def test_stock_or_wrong_mooncake_version_is_rejected() -> None:
    engine = _complete_mooncake_engine()
    report = verify.probe_backends(
        ["te-rdma"],
        importer=_importer(
            {
                "torch": _torch(),
                "mooncake.engine": engine,
            }
        ),
        version_reader=lambda _name: "0.3.11.post1",
        environment={},
    )

    result = report["backends"]["te-rdma"]
    assert result["status"] == "unsupported"
    assert "mooncake_version" in result["failed_checks"]


def test_unknown_backend_is_rejected_before_probing() -> None:
    with pytest.raises(ValueError, match="unknown transport backend"):
        verify.probe_backends(["tcp"])
