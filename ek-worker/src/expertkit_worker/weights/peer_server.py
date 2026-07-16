"""Bounded local-only HTTP serving for cached expert SafeTensors files."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import structlog
from aiohttp import web

from expertkit_worker.weights.dram_cache import WeightKey
from expertkit_worker.weights.loader import (
    CpuWeightLoader,
    WeightLoadErrorCode,
    WeightLoadFailed,
)

logger = structlog.get_logger(__name__)


class PeerWeightServer[CpuWeightT, ReadyWeightT]:
    """Serve only DRAM and disk-cache weights without recursive fetching."""

    def __init__(
        self,
        *,
        model_name: str,
        num_layers: int,
        experts_per_layer: int,
        host: str,
        port: int,
        max_concurrent_requests: int,
        loader: CpuWeightLoader[CpuWeightT, ReadyWeightT],
        request_timeout_secs: float = 30.0,
        chunk_bytes: int = 1024 * 1024,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not host:
            raise ValueError("peer server host must not be empty")
        for name, value in (
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
            ("max_concurrent_requests", max_concurrent_requests),
            ("chunk_bytes", chunk_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("peer server port must be between 0 and 65535")
        if (
            isinstance(request_timeout_secs, bool)
            or not isinstance(request_timeout_secs, int | float)
            or request_timeout_secs <= 0
        ):
            raise ValueError("request_timeout_secs must be positive")

        self._model_name = model_name
        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._host = host
        self._port = port
        self._max_concurrent_requests = max_concurrent_requests
        self._loader = loader
        self._request_timeout_secs = float(request_timeout_secs)
        self._chunk_bytes = chunk_bytes
        self._active_requests = 0
        self._runner: web.AppRunner | None = None

    @property
    def addresses(self) -> tuple[tuple[str, int], ...]:
        """Return bound addresses after startup, including an ephemeral test port."""

        runner = self._runner
        if runner is None:
            return ()
        return tuple((str(host), int(port)) for host, port, *_ in runner.addresses)

    async def start(self) -> None:
        """Bind the configured plaintext HTTP listener exactly once."""

        if self._runner is not None:
            return
        app = web.Application()
        app.router.add_get(r"/expert/{tail:.*}", self._handle)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, self._host, self._port)
            await site.start()
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner

    async def close(self) -> None:
        """Stop admission and close the HTTP listener and active connections."""

        runner = self._runner
        if runner is None:
            return
        self._runner = None
        await runner.cleanup()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if self._active_requests >= self._max_concurrent_requests:
            logger.debug("peer_weight_server_busy")
            return web.Response(status=503)
        self._active_requests += 1
        try:
            parsed = self._parse_tail(request.match_info["tail"])
            if parsed is None:
                return web.Response(status=404)
            key = parsed
            try:
                async with asyncio.timeout(self._request_timeout_secs):
                    lease = await self._loader.acquire_local(key)
            except WeightLoadFailed as error:
                last = error.failures[-1]
                if last.code is not WeightLoadErrorCode.NOT_FOUND:
                    logger.warning(
                        "peer_weight_local_load_failed",
                        layer_id=key.layer_id,
                        expert_id=key.expert_id,
                        error_code=last.code.value,
                        diagnostic=last.diagnostic,
                    )
                return web.Response(status=404)
            except TimeoutError:
                logger.warning(
                    "peer_weight_local_load_timed_out",
                    layer_id=key.layer_id,
                    expert_id=key.expert_id,
                )
                return web.Response(status=503)

            try:
                return await self._send(request, lease.cached.buffer.view())
            finally:
                await lease.close()
        finally:
            self._active_requests -= 1

    async def _send(self, request: web.Request, view: memoryview) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(view.nbytes),
            },
        )
        try:
            await response.prepare(request)
            for start in range(0, view.nbytes, self._chunk_bytes):
                chunk = view[start : start + self._chunk_bytes]
                try:
                    await response.write(chunk)
                finally:
                    chunk.release()
            await response.write_eof()
            return response
        finally:
            view.release()

    def _parse_tail(self, tail: str) -> WeightKey | None:
        with suppress(ValueError):
            model_name, layer_text, expert_text = tail.rsplit("/", 2)
            if model_name != self._model_name:
                return None
            layer_id = int(layer_text)
            expert_id = int(expert_text)
            if not 0 <= layer_id < self._num_layers:
                return None
            if not 0 <= expert_id < self._experts_per_layer:
                return None
            return WeightKey(layer_id, expert_id)
        return None
