"""Tests for high-level routed-layer client startup."""

import asyncio
import time
from typing import ClassVar

import pytest
import torch
from expertkit_proto.ek.control.v2 import lifecycle_pb2

import expertkit_transport.client as client_module
from expertkit_transport.client import BlockingRoutedMoEClient, RoutedMoEClient
from expertkit_transport.controller import ResolvedDefaultInstance
from expertkit_transport.errors import TransportError, TransportErrorCode


class FakeTopology:
    instances: ClassVar[list["FakeTopology"]] = []

    def __init__(self, endpoint: str, **configuration: object) -> None:
        self.endpoint = endpoint
        self.configuration = configuration
        self.pools: dict[object, object] = {}
        self.started = False
        self.closed = False
        self.__class__.instances.append(self)

    async def start(self, *, monotonic_deadline: float) -> None:
        assert monotonic_deadline > time.monotonic()
        self.started = True

    async def close(self) -> None:
        self.closed = True


class FakeTransportRuntime:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True


class _FakeCudaDevice:
    type = "cuda"


class _FakeCudaTensor:
    def __init__(self, *, fail_record_stream: bool = False) -> None:
        self.device = _FakeCudaDevice()
        self.fail_record_stream = fail_record_stream
        self.recorded_streams: list[object] = []

    def record_stream(self, stream: object) -> None:
        self.recorded_streams.append(stream)
        if self.fail_record_stream:
            raise RuntimeError("injected tensor record_stream failure")


class _FakeCudaEvent:
    def __init__(self, *, fail_record: bool = False) -> None:
        self.fail_record = fail_record
        self.recorded_streams: list[object] = []

    def record(self, stream: object) -> None:
        self.recorded_streams.append(stream)
        if self.fail_record:
            raise RuntimeError("injected event record failure")


class _FakeCudaStream:
    def __init__(self, *, fail_wait: bool = False) -> None:
        self.fail_wait = fail_wait
        self.waited_events: list[object] = []

    def wait_event(self, event: object) -> None:
        self.waited_events.append(event)
        if self.fail_wait:
            raise RuntimeError("injected stream wait failure")


def _blocking_client(monkeypatch, async_client_type: type) -> BlockingRoutedMoEClient:
    monkeypatch.setattr(client_module, "RoutedMoEClient", async_client_type)
    client = BlockingRoutedMoEClient(
        "controller:5002",
        num_layers=1,
        experts_per_layer=2,
        hidden_dim=3,
        top_k=1,
        dtype=torch.float32,
        device="cpu",
    )
    client.start(timeout_seconds=1)
    return client


def _execute_fake_cuda(client: BlockingRoutedMoEClient) -> object:
    return client.execute(
        layer_id=0,
        hidden_states=_FakeCudaTensor(),  # type: ignore[arg-type]
        expert_ids=_FakeCudaTensor(),  # type: ignore[arg-type]
        routing_weights=_FakeCudaTensor(),  # type: ignore[arg-type]
        distinct_expert_ids=(0,),
        timeout_seconds=1,
    )


def test_client_resolves_default_before_constructing_instance_bound_topology(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        requests: list[int | None] = []
        submitted: list[object] = []

        async def resolve(endpoint, *, requested_instance_id, timeout_seconds):
            assert endpoint == "controller:5002"
            assert timeout_seconds > 0
            requests.append(requested_instance_id)
            return ResolvedDefaultInstance(7, "model", "default")

        async def execute(batch, *args, **kwargs):
            submitted.append(batch)
            return batch.hidden_states

        FakeTopology.instances.clear()
        monkeypatch.setattr(client_module, "resolve_default_instance", resolve)
        monkeypatch.setattr(client_module, "ControllerTopologyWatcher", FakeTopology)
        monkeypatch.setattr(client_module, "execute_routed_layer", execute)
        runtime = FakeTransportRuntime()
        client = RoutedMoEClient(
            "controller:5002",
            num_layers=2,
            experts_per_layer=4,
            hidden_dim=3,
            top_k=2,
            dtype=torch.float32,
            device="cpu",
            transport_runtime=runtime,
        )

        await client.start(monotonic_deadline=time.monotonic() + 1)
        hidden = torch.ones((1, 3))
        result = await client.execute(
            layer_id=1,
            hidden_states=hidden,
            expert_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            distinct_expert_ids=(0, 1),
            monotonic_deadline=time.monotonic() + 1,
        )

        assert result is hidden
        assert requests == [None]
        assert runtime.started is False
        assert FakeTopology.instances[0].configuration["instance_id"] == 7
        registry = FakeTopology.instances[0].configuration["runtime_registry"]
        assert registry.runtime_for(lifecycle_pb2.WORKER_TRANSPORT_NCCL) is runtime
        assert registry.runtime_for(lifecycle_pb2.WORKER_TRANSPORT_TRANSFER_ENGINE) is runtime
        assert submitted[0].instance_id == 7
        await client.close()
        assert FakeTopology.instances[0].closed is True
        assert runtime.closed is True

    asyncio.run(scenario())


def test_blocking_timeout_surfaces_a_late_unsafe_staging_fatal(monkeypatch) -> None:
    class LateUnsafeClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self, *, monotonic_deadline: float) -> None:
            assert monotonic_deadline > time.monotonic()

        async def execute(self, **kwargs) -> torch.Tensor:
            del kwargs
            try:
                await asyncio.sleep(0.15)
            except asyncio.CancelledError:
                # Model a transport that defers cancellation until its native
                # ownership fence discovers a fatal completion failure.
                await asyncio.sleep(0.01)
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                unsafe_tensor_ownership=True,
                diagnostic="late CUDA input-copy completion could not be proven",
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(client_module, "RoutedMoEClient", LateUnsafeClient)
    client = BlockingRoutedMoEClient(
        "controller:5002",
        num_layers=1,
        experts_per_layer=2,
        hidden_dim=3,
        top_k=1,
        dtype=torch.float32,
        device="cpu",
    )
    try:
        client.start(timeout_seconds=1)
        with pytest.raises(TransportError, match="could not be proven") as caught:
            client.execute(
                layer_id=0,
                hidden_states=torch.ones((1, 3)),
                expert_ids=torch.zeros((1, 1), dtype=torch.int32),
                routing_weights=torch.ones((1, 1), dtype=torch.float32),
                distinct_expert_ids=(0,),
                timeout_seconds=0.01,
            )
        assert not caught.value.retryable
        assert caught.value.unsafe_tensor_ownership
        assert caught.value.code is TransportErrorCode.UNAVAILABLE
        with pytest.raises(TransportError) as start_rejected:
            client.start(timeout_seconds=1)
        assert start_rejected.value is caught.value
        with pytest.raises(TransportError) as execute_rejected:
            client.execute(
                layer_id=0,
                hidden_states=torch.ones((1, 3)),
                expert_ids=torch.zeros((1, 1), dtype=torch.int32),
                routing_weights=torch.ones((1, 1), dtype=torch.float32),
                distinct_expert_ids=(0,),
                timeout_seconds=1,
            )
        assert execute_rejected.value is caught.value
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["create", "record"])
def test_blocking_input_fence_failure_cleans_active_and_retains_inputs(
    failure: str,
    monkeypatch,
) -> None:
    class NeverExecutedClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self, *, monotonic_deadline: float) -> None:
            del monotonic_deadline

        async def execute(self, **kwargs) -> torch.Tensor:
            del kwargs
            raise AssertionError("input fence failure must not submit work")

        async def close(self) -> None:
            return None

    stream = _FakeCudaStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    if failure == "create":

        def fail_event_create(**kwargs) -> _FakeCudaEvent:
            del kwargs
            raise RuntimeError("injected event creation failure")

        monkeypatch.setattr(torch.cuda, "Event", fail_event_create)
    else:
        monkeypatch.setattr(
            torch.cuda,
            "Event",
            lambda **kwargs: _FakeCudaEvent(fail_record=True),
        )
    client = _blocking_client(monkeypatch, NeverExecutedClient)
    hidden = _FakeCudaTensor()
    expert_ids = _FakeCudaTensor()
    routing_weights = _FakeCudaTensor()
    try:
        with pytest.raises(TransportError, match="input-ready fence") as caught:
            client.execute(
                layer_id=0,
                hidden_states=hidden,  # type: ignore[arg-type]
                expert_ids=expert_ids,  # type: ignore[arg-type]
                routing_weights=routing_weights,  # type: ignore[arg-type]
                distinct_expert_ids=(0,),
                timeout_seconds=1,
            )
        assert not caught.value.retryable
        assert caught.value.unsafe_tensor_ownership
        assert not client._active  # type: ignore[attr-defined]
        retained = client_module._QUARANTINED_BLOCKING_GRAPHS[-1]  # type: ignore[attr-defined]
        for owner in (client, hidden, expert_ids, routing_weights):
            assert any(candidate is owner for candidate in retained)
    finally:
        client.close()


def test_blocking_output_event_failure_retains_result_and_inputs(monkeypatch) -> None:
    result = _FakeCudaTensor()

    class ImmediateClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self, *, monotonic_deadline: float) -> None:
            del monotonic_deadline

        async def execute(self, **kwargs) -> object:
            del kwargs
            return result

        async def close(self) -> None:
            return None

    event_count = 0

    def event_factory(**kwargs) -> _FakeCudaEvent:
        nonlocal event_count
        del kwargs
        event_count += 1
        return _FakeCudaEvent(fail_record=event_count == 2)

    monkeypatch.setattr(torch.cuda, "Event", event_factory)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: _FakeCudaStream(),
    )
    client = _blocking_client(monkeypatch, ImmediateClient)
    hidden = _FakeCudaTensor()
    expert_ids = _FakeCudaTensor()
    routing_weights = _FakeCudaTensor()
    try:
        with pytest.raises(TransportError, match="output-ready fence") as caught:
            client.execute(
                layer_id=0,
                hidden_states=hidden,  # type: ignore[arg-type]
                expert_ids=expert_ids,  # type: ignore[arg-type]
                routing_weights=routing_weights,  # type: ignore[arg-type]
                distinct_expert_ids=(0,),
                timeout_seconds=1,
            )
        assert caught.value.unsafe_tensor_ownership
        retained = client_module._QUARANTINED_BLOCKING_GRAPHS[-1]  # type: ignore[attr-defined]
        for owner in (result, hidden, expert_ids, routing_weights):
            assert any(candidate is owner for candidate in retained)
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["wait", "record_stream"])
def test_blocking_caller_fence_failure_retains_and_poison_closes(
    failure: str,
    monkeypatch,
) -> None:
    result = _FakeCudaTensor(fail_record_stream=failure == "record_stream")

    class ImmediateClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self, *, monotonic_deadline: float) -> None:
            del monotonic_deadline

        async def execute(self, **kwargs) -> object:
            del kwargs
            return result

        async def close(self) -> None:
            return None

    streams = [
        _FakeCudaStream(),
        _FakeCudaStream(),
        _FakeCudaStream(),
        _FakeCudaStream(fail_wait=failure == "wait"),
    ]
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: _FakeCudaEvent())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: streams.pop(0))
    client = _blocking_client(monkeypatch, ImmediateClient)
    hidden = _FakeCudaTensor()
    expert_ids = _FakeCudaTensor()
    routing_weights = _FakeCudaTensor()
    try:
        with pytest.raises(TransportError, match="caller-stream output fence") as caught:
            client.execute(
                layer_id=0,
                hidden_states=hidden,  # type: ignore[arg-type]
                expert_ids=expert_ids,  # type: ignore[arg-type]
                routing_weights=routing_weights,  # type: ignore[arg-type]
                distinct_expert_ids=(0,),
                timeout_seconds=1,
            )
        assert caught.value.unsafe_tensor_ownership
        retained = client_module._QUARANTINED_BLOCKING_GRAPHS[-1]  # type: ignore[attr-defined]
        for owner in (client, result, hidden, expert_ids, routing_weights):
            assert any(candidate is owner for candidate in retained)
        with pytest.raises(TransportError) as rejected:
            _execute_fake_cuda(client)
        assert rejected.value is caught.value
    finally:
        client.close()


def test_blocking_result_records_the_actual_caller_stream(monkeypatch) -> None:
    result = _FakeCudaTensor()

    class ImmediateClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        async def start(self, *, monotonic_deadline: float) -> None:
            del monotonic_deadline

        async def execute(self, **kwargs) -> object:
            del kwargs
            return result

        async def close(self) -> None:
            return None

    caller_output_stream = _FakeCudaStream()
    streams = [
        _FakeCudaStream(),
        _FakeCudaStream(),
        _FakeCudaStream(),
        caller_output_stream,
    ]
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: _FakeCudaEvent())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: streams.pop(0))
    client = _blocking_client(monkeypatch, ImmediateClient)
    try:
        assert _execute_fake_cuda(client) is result
        assert result.recorded_streams == [caller_output_stream]
        assert len(caller_output_stream.waited_events) == 1
    finally:
        client.close()
