"""Enforce the package boundaries required for adding another Transport."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterable
from pathlib import Path

from expertkit_transport.transports.base import WorkerBatchReceiver
from expertkit_transport.transports.grpc import GrpcWorkerBatchReceiver
from expertkit_transport.transports.shm import ShmWorkerBatchReceiver

_REPOSITORY = Path(__file__).resolve().parents[3]
_TRANSPORT_PACKAGE = _REPOSITORY / "ek-transport/src/expertkit_transport"
_WORKER_PACKAGE = _REPOSITORY / "ek-worker/src/expertkit_worker"


def _python_files(*roots: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for root in roots for path in root.rglob("*.py")))


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


def _defined_or_imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _assert_no_import_prefix(files: Iterable[Path], prefixes: tuple[str, ...]) -> None:
    violations = {
        str(path.relative_to(_REPOSITORY)): tuple(
            module for module in _imports(path) if module.startswith(prefixes)
        )
        for path in files
    }
    assert not {path: modules for path, modules in violations.items() if modules}


def test_removed_generic_package_buckets_do_not_exist() -> None:
    for name in ("contracts", "orchestration", "adapters"):
        assert not (_TRANSPORT_PACKAGE / name).exists()


def test_old_production_names_are_not_defined_or_reexported() -> None:
    old_names = {
        "ActivePosition",
        "BlockingGrpcRoutedMoEClient",
        "GrpcBatchSpec",
        "GrpcRoutedMoEClient",
        "GrpcTopologyProvider",
        "GrpcWorkerPositionBuffers",
        "GrpcWorkerServer",
        "PositionResult",
        "ReceivedWorkerBatch",
        "WorkerExecution",
        "WorkerPositionBuffers",
        "WorkerPositionSpec",
        "WorkerTarget",
    }
    violations = {
        str(path.relative_to(_REPOSITORY)): sorted(old_names & _defined_or_imported_names(path))
        for path in _python_files(_TRANSPORT_PACKAGE, _WORKER_PACKAGE)
    }
    assert not {path: names for path, names in violations.items() if names}


def test_grpc_and_shm_do_not_import_each_others_implementation() -> None:
    _assert_no_import_prefix(
        _python_files(_TRANSPORT_PACKAGE / "transports/grpc"),
        ("expertkit_transport.transports.shm",),
    )
    _assert_no_import_prefix(
        _python_files(_TRANSPORT_PACKAGE / "transports/shm"),
        ("expertkit_transport.transports.grpc",),
    )


def test_generic_frontend_code_does_not_import_concrete_transports() -> None:
    _assert_no_import_prefix(
        (
            *_python_files(
                _TRANSPORT_PACKAGE / "routing",
                _TRANSPORT_PACKAGE / "controller",
            ),
            _TRANSPORT_PACKAGE / "client.py",
        ),
        (
            "expertkit_transport.transports.grpc",
            "expertkit_transport.transports.shm",
        ),
    )


def test_worker_subsystems_do_not_import_concrete_transports() -> None:
    _assert_no_import_prefix(
        _python_files(
            _WORKER_PACKAGE / "control",
            _WORKER_PACKAGE / "execution",
            _WORKER_PACKAGE / "backends",
        ),
        (
            "expertkit_transport.transports.grpc",
            "expertkit_transport.transports.shm",
        ),
    )


def test_worker_does_not_import_transport_private_protocol_code() -> None:
    _assert_no_import_prefix(
        _python_files(_WORKER_PACKAGE),
        ("expertkit_transport._proto",),
    )


def test_frontend_integrations_use_only_public_transport_imports() -> None:
    integration_files = _python_files(
        _REPOSITORY / "ek-integration/expertkit_torch/expertkit_torch",
        _REPOSITORY / "ek-integration/expertkit_vllm/expertkit_vllm",
    )
    _assert_no_import_prefix(
        integration_files,
        (
            "expertkit_transport.controller",
            "expertkit_transport.routing",
            "expertkit_transport.transports",
        ),
    )


def test_both_production_receivers_implement_the_complete_interface() -> None:
    required = WorkerBatchReceiver.__abstractmethods__
    for receiver in (GrpcWorkerBatchReceiver, ShmWorkerBatchReceiver):
        assert issubclass(receiver, WorkerBatchReceiver)
        assert not inspect.isabstract(receiver)
        assert receiver.__module__.startswith("expertkit_transport.transports.")
        assert required <= receiver.__dict__.keys()


def test_worker_factory_contains_one_explicit_branch_per_receiver() -> None:
    path = _WORKER_PACKAGE / "factory.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("GrpcWorkerBatchReceiver") == 1
    assert calls.count("ShmWorkerBatchReceiver") == 1


def test_routing_does_not_import_protocol_transfer_buffers() -> None:
    routing_files = (
        *_python_files(_TRANSPORT_PACKAGE / "routing"),
        _TRANSPORT_PACKAGE / "client.py",
        _TRANSPORT_PACKAGE / "buffers.py",
    )
    forbidden_names = {"OutputBufferProvider", "OutputSpec", "PreparedOutput"}
    violations = {
        str(path.relative_to(_REPOSITORY)): sorted(
            forbidden_names & _defined_or_imported_names(path)
        )
        for path in routing_files
    }
    assert not {path: names for path, names in violations.items() if names}
    _assert_no_import_prefix(
        routing_files,
        (
            "expertkit_transport.transports.grpc.buffers",
            "expertkit_transport.transports.shm.buffers",
        ),
    )
