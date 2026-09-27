"""Unit checks for static NCCL runtime configuration."""

import asyncio

import pytest

from expertkit_transport.transports.nccl import NcclRuntime, NcclRuntimeConfig
from expertkit_transport.transports.nccl import runtime as runtime_module


def test_worker_host_port_is_canonicalized_for_torch_rendezvous() -> None:
    runtime = NcclRuntime(
        NcclRuntimeConfig(
            rank=1,
            world_size=2,
            rendezvous_endpoint="controller:29500",
            group_name="expertkit-test",
            device="cuda:0",
        ),
        process_group=object(),
        distributed=object(),
    )

    assert runtime.rendezvous_endpoint == "tcp://controller:29500"
    asyncio.run(runtime.close())


def test_production_runtime_rejects_cpu_and_unindexed_cuda_devices() -> None:
    with pytest.raises(ValueError, match="indexed CUDA"):
        NcclRuntimeConfig(0, 2, "controller:29500", "expertkit-test", "cpu")
    with pytest.raises(ValueError, match="indexed CUDA"):
        NcclRuntimeConfig(0, 2, "controller:29500", "expertkit-test", "cuda")


def test_work_completion_is_made_host_visible_after_stream_waits(monkeypatch) -> None:
    operations: list[object] = []

    class FakeWork:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait(self) -> None:
            operations.append(("wait", self.name))

    class FakeEvent:
        def record(self, stream: object) -> None:
            operations.append(("record", stream))

        def synchronize(self) -> None:
            operations.append("synchronize")

    monkeypatch.setattr(
        runtime_module.torch.cuda,
        "set_device",
        lambda device: operations.append(("set_device", device)),
    )
    monkeypatch.setattr(runtime_module.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        runtime_module.torch.cuda,
        "current_stream",
        lambda device: ("stream", device),
    )
    runtime = NcclRuntime(
        NcclRuntimeConfig(0, 2, "controller:29500", "expertkit-test", "cuda:0"),
        process_group=object(),
        distributed=object(),
    )

    runtime._wait_works((FakeWork("first"), FakeWork("second")))

    assert operations == [
        ("set_device", runtime.device),
        ("wait", "first"),
        ("wait", "second"),
        ("record", ("stream", runtime.device)),
        "synchronize",
    ]
    asyncio.run(runtime.close())


def test_owned_runtime_aborts_before_unregistering_process_group(monkeypatch) -> None:
    operations: list[object] = []

    class FakeProcessGroup:
        def abort(self) -> None:
            operations.append("abort")

    class FakeDistributed:
        def destroy_process_group(self, process_group: object) -> None:
            operations.append(("destroy", process_group))

    process_group = FakeProcessGroup()
    distributed = FakeDistributed()
    monkeypatch.setattr(
        runtime_module.torch.cuda,
        "set_device",
        lambda device: operations.append(("set_device", device)),
    )
    runtime = NcclRuntime(
        NcclRuntimeConfig(0, 2, "controller:29500", "expertkit-test", "cuda:0"),
        process_group=process_group,
        distributed=distributed,
    )
    runtime._owns_process_group = True

    asyncio.run(runtime.close())

    assert operations == [
        ("set_device", runtime.device),
        "abort",
        ("destroy", process_group),
    ]
