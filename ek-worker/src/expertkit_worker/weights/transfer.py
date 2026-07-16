"""Pluggable bulk weight-transfer interface and HTTP implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import aiohttp

from expertkit_worker.weights.direct_io import AlignedWeightBuffer


class WeightTransferError(RuntimeError):
    """Report one failed source transfer with Controller retry meaning."""

    def __init__(self, diagnostic: str, *, retryable: bool) -> None:
        super().__init__(diagnostic)
        self.diagnostic = diagnostic
        self.retryable = retryable


class WeightNotFound(WeightTransferError):
    """Report that one concrete source definitively lacks the requested expert."""

    def __init__(self, diagnostic: str = "weight source returned not found") -> None:
        super().__init__(diagnostic, retryable=False)


class WeightTransfer(ABC):
    """Move one complete expert file into application-owned aligned memory."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize process-lifetime connections before placement handling."""

    @abstractmethod
    async def download(self, url: str, *, max_bytes: int) -> AlignedWeightBuffer:
        """Download one complete expert file within a strict byte bound."""

    @abstractmethod
    async def close(self) -> None:
        """Close process-lifetime connections after loading has stopped."""


class HttpWeightTransfer(WeightTransfer):
    """Stream HTTP response chunks into one final aligned Host buffer."""

    def __init__(
        self,
        *,
        max_connections: int,
        connect_timeout_secs: float = 10.0,
        read_timeout_secs: float = 300.0,
        chunk_bytes: int = 1024 * 1024,
    ) -> None:
        if (
            isinstance(max_connections, bool)
            or not isinstance(max_connections, int)
            or max_connections <= 0
        ):
            raise ValueError("max_connections must be a positive integer")
        for name, value in (
            ("connect_timeout_secs", connect_timeout_secs),
            ("read_timeout_secs", read_timeout_secs),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        self._max_connections = max_connections
        self._timeout = aiohttp.ClientTimeout(
            total=None,
            connect=float(connect_timeout_secs),
            sock_connect=float(connect_timeout_secs),
            sock_read=float(read_timeout_secs),
        )
        self._chunk_bytes = chunk_bytes
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Create the single process-lifetime aiohttp session."""

        if self._session is not None:
            return
        connector = aiohttp.TCPConnector(limit=self._max_connections)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=self._timeout,
            auto_decompress=False,
            headers={"Accept-Encoding": "identity"},
        )

    async def download(self, url: str, *, max_bytes: int) -> AlignedWeightBuffer:
        """Copy bounded response chunks without assembling a second complete value."""

        if not url:
            raise ValueError("weight download URL must not be empty")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        session = self._session
        if session is None:
            raise RuntimeError("HTTP weight transfer has not been started")

        result: AlignedWeightBuffer | None = None
        try:
            async with session.get(url) as response:
                if response.status == 404:
                    raise WeightNotFound()
                if response.status >= 400:
                    diagnostic = f"weight source returned HTTP {response.status}"
                    raise WeightTransferError(diagnostic, retryable=response.status >= 500)
                encoding = response.headers.get("Content-Encoding", "identity").lower()
                if encoding != "identity":
                    raise WeightTransferError(
                        "weight source used unsupported content encoding",
                        retryable=False,
                    )
                declared = self._parse_content_length(response.headers.get("Content-Length"))
                if declared is not None and declared > max_bytes:
                    raise WeightTransferError(
                        "weight response exceeds its configured byte limit",
                        retryable=False,
                    )
                result = AlignedWeightBuffer(declared or max_bytes)
                target = result.view()
                try:
                    received = 0
                    async for chunk in response.content.iter_chunked(self._chunk_bytes):
                        end = received + len(chunk)
                        if end > target.nbytes:
                            raise WeightTransferError(
                                "weight response exceeds its configured byte limit",
                                retryable=False,
                            )
                        target[received:end] = chunk
                        received = end
                finally:
                    target.release()
                if received <= 0:
                    raise WeightTransferError("weight response is empty", retryable=False)
                if declared is not None and received != declared:
                    raise WeightTransferError(
                        "weight response length does not match Content-Length",
                        retryable=True,
                    )
                if declared is None:
                    result.trim(received)
                return result
        except WeightTransferError:
            if result is not None:
                result.close()
            raise
        except (aiohttp.ClientError, TimeoutError) as error:
            if result is not None:
                result.close()
            raise WeightTransferError(str(error), retryable=True) from error
        except BaseException:
            if result is not None:
                result.close()
            raise

    async def close(self) -> None:
        """Close the shared aiohttp connection pool exactly once."""

        session = self._session
        if session is None:
            return
        self._session = None
        await session.close()

    @staticmethod
    def _parse_content_length(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            result = int(value)
        except ValueError as error:
            raise WeightTransferError(
                "weight response has invalid Content-Length",
                retryable=False,
            ) from error
        if result <= 0:
            raise WeightTransferError(
                "weight response has invalid Content-Length",
                retryable=False,
            )
        return result
