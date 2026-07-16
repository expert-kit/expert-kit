"""Localhost integration tests for streamed HTTP expert transfer."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

import pytest
from aiohttp import web

from expertkit_worker.weights.transfer import (
    HttpWeightTransfer,
    WeightNotFound,
    WeightTransferError,
)


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Run one isolated HTTP integration scenario."""

    return asyncio.run(coroutine)


async def with_server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    scenario: Callable[[str], Awaitable[None]],
) -> None:
    """Run one scenario against an ephemeral localhost aiohttp server."""

    app = web.Application()
    app.router.add_get("/expert", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        await scenario(f"http://127.0.0.1:{port}/expert")
    finally:
        await runner.cleanup()


def test_http_transfer_streams_declared_length_into_aligned_buffer() -> None:
    payload = bytes(range(251)) * 41

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(body=payload)

    async def scenario(url: str) -> None:
        transfer = HttpWeightTransfer(max_connections=2, chunk_bytes=127)
        await transfer.start()
        try:
            result = await transfer.download(url, max_bytes=len(payload) + 1)
            view = result.view()
            try:
                assert bytes(view) == payload
                assert result.logical_size == len(payload)
            finally:
                view.release()
                result.close()
        finally:
            await transfer.close()

    run(with_server(handler, scenario))


def test_http_transfer_accepts_chunked_response_and_trims_capacity() -> None:
    payload = b"chunked-weight-data"

    async def handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(payload[:7])
        await response.write(payload[7:])
        await response.write_eof()
        return response

    async def scenario(url: str) -> None:
        transfer = HttpWeightTransfer(max_connections=1, chunk_bytes=4)
        await transfer.start()
        try:
            result = await transfer.download(url, max_bytes=128)
            view = result.view()
            try:
                assert bytes(view) == payload
                assert result.logical_size == len(payload)
            finally:
                view.release()
                result.close()
        finally:
            await transfer.close()

    run(with_server(handler, scenario))


@pytest.mark.parametrize(
    ("status", "error_type", "retryable"),
    [
        (404, WeightNotFound, False),
        (400, WeightTransferError, False),
        (503, WeightTransferError, True),
    ],
)
def test_http_transfer_maps_source_status(
    status: int,
    error_type: type[WeightTransferError],
    retryable: bool,
) -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=status)

    async def scenario(url: str) -> None:
        transfer = HttpWeightTransfer(max_connections=1)
        await transfer.start()
        try:
            with pytest.raises(error_type) as caught:
                await transfer.download(url, max_bytes=128)
            assert caught.value.retryable is retryable
        finally:
            await transfer.close()

    run(with_server(handler, scenario))


def test_http_transfer_rejects_oversized_and_encoded_responses() -> None:
    async def oversized(_request: web.Request) -> web.Response:
        return web.Response(body=b"x" * 129)

    async def encoded(_request: web.Request) -> web.Response:
        return web.Response(body=b"weight", headers={"Content-Encoding": "gzip"})

    async def reject(url: str, match: str) -> None:
        transfer = HttpWeightTransfer(max_connections=1)
        await transfer.start()
        try:
            with pytest.raises(WeightTransferError, match=match) as caught:
                await transfer.download(url, max_bytes=128)
            assert caught.value.retryable is False
        finally:
            await transfer.close()

    run(with_server(oversized, lambda url: reject(url, "byte limit")))
    run(with_server(encoded, lambda url: reject(url, "content encoding")))


def test_http_transfer_requires_start_and_closes_idempotently() -> None:
    async def scenario() -> None:
        transfer = HttpWeightTransfer(max_connections=1)
        with pytest.raises(RuntimeError, match="not been started"):
            await transfer.download("http://127.0.0.1/expert", max_bytes=10)
        await transfer.start()
        await transfer.start()
        await transfer.close()
        await transfer.close()

    run(scenario())
