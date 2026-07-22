"""Tests for Controller default-instance resolution."""

import asyncio

import grpc
import pytest
from expertkit_proto.ek.control.v2 import lifecycle_pb2, lifecycle_pb2_grpc

from expertkit_transport.controller import resolve_default_instance
from expertkit_transport.errors import TransportError, TransportErrorCode


class InstanceService(lifecycle_pb2_grpc.InstanceServiceServicer):
    def __init__(self, instance_id: int = 7) -> None:
        self.instance_id = instance_id
        self.requests: list[int] = []

    async def ResolveDefaultInstance(self, request, context):
        self.requests.append(request.requested_instance_id)
        if request.requested_instance_id not in (0, self.instance_id):
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "stale instance ID")
        return lifecycle_pb2.ResolveDefaultInstanceResponse(
            instance_id=self.instance_id,
            model_name="DeepSeek-V2-Lite-Chat",
            instance_name="deepseek-v2-lite-demo",
        )


def test_omitted_and_explicit_ids_resolve_to_the_controller_default() -> None:
    async def scenario() -> None:
        service = InstanceService()
        server = grpc.aio.server()
        lifecycle_pb2_grpc.add_InstanceServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        assert port > 0
        await server.start()
        try:
            omitted = await resolve_default_instance(
                f"127.0.0.1:{port}",
                requested_instance_id=None,
                timeout_seconds=1,
            )
            explicit = await resolve_default_instance(
                f"127.0.0.1:{port}",
                requested_instance_id=7,
                timeout_seconds=1,
            )
        finally:
            await server.stop(None)

        assert omitted == explicit
        assert omitted.instance_id == 7
        assert service.requests == [0, 7]

    asyncio.run(scenario())


def test_stale_explicit_id_is_a_nonretryable_startup_error() -> None:
    async def scenario() -> None:
        service = InstanceService()
        server = grpc.aio.server()
        lifecycle_pb2_grpc.add_InstanceServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        assert port > 0
        await server.start()
        try:
            with pytest.raises(TransportError) as captured:
                await resolve_default_instance(
                    f"127.0.0.1:{port}",
                    requested_instance_id=8,
                    timeout_seconds=1,
                )
        finally:
            await server.stop(None)

        assert captured.value.code is TransportErrorCode.INVALID_REQUEST
        assert captured.value.retryable is False
        assert "stale instance ID" in captured.value.diagnostic

    asyncio.run(scenario())
