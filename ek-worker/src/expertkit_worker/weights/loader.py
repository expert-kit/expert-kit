"""CPU weight lookup across DRAM, disk, peers, and the central server."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import quote

from expertkit_worker.weights.adapter import WeightAdapter
from expertkit_worker.weights.direct_io import AlignedWeightBuffer, InvalidWeightFile
from expertkit_worker.weights.disk_cache import WeightDiskCache
from expertkit_worker.weights.dram_cache import (
    DramCache,
    DramCacheLease,
    WeightKey,
)
from expertkit_worker.weights.format import (
    SafeTensorData,
    SafeTensorFormatError,
    max_safetensors_file_bytes,
    parse_safetensors,
)
from expertkit_worker.weights.transfer import (
    WeightNotFound,
    WeightTransfer,
    WeightTransferError,
)


class WeightSource(StrEnum):
    """Identify the source that supplied or failed one expert load."""

    DRAM = "dram"
    DISK = "disk"
    PEER = "peer"
    WEIGHT_SERVER = "weight_server"


class WeightLoadStage(StrEnum):
    """Identify the loading step that failed before device placement."""

    FETCH = "fetch"
    READ = "read"
    PARSE = "parse"
    VALIDATE = "validate"


class WeightLoadErrorCode(StrEnum):
    """Classify a loading failure without parsing its diagnostic text."""

    NOT_FOUND = "not_found"
    IO = "io"
    NETWORK = "network"
    INVALID_FORMAT = "invalid_format"
    UNEXPECTED_METADATA = "unexpected_metadata"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class WeightLoadFailure:
    """Describe one concrete source attempt that did not produce a valid weight."""

    source: WeightSource
    stage: WeightLoadStage
    code: WeightLoadErrorCode
    retryable: bool
    diagnostic: str


class WeightLoadFailed(RuntimeError):
    """Report that every configured source failed once for one load attempt."""

    def __init__(self, key: WeightKey, failures: tuple[WeightLoadFailure, ...]) -> None:
        if not failures:
            raise ValueError("WeightLoadFailed requires at least one source failure")
        self.key = key
        self.failures = failures
        self.retryable = any(failure.retryable for failure in failures)
        last = failures[-1]
        self.stage = last.stage
        self.code = last.code
        diagnostic = "; ".join(
            f"{failure.source.value}: {failure.diagnostic}" for failure in failures
        )
        super().__init__(diagnostic)


@dataclass(frozen=True, slots=True)
class CachedCpuWeight[CpuWeightT]:
    """Keep parsed CPU data and the aligned SafeTensors owner together."""

    buffer: AlignedWeightBuffer
    parsed: SafeTensorData
    value: CpuWeightT
    byte_count: int


class CpuWeightLease[CpuWeightT]:
    """Retain one cached CPU weight through device conversion or peer serving."""

    def __init__(
        self,
        cache_lease: DramCacheLease[CachedCpuWeight[CpuWeightT]],
        source: WeightSource,
    ) -> None:
        self._cache_lease = cache_lease
        self.cached = cache_lease.value
        self.source = source

    async def close(self) -> None:
        """Release the DRAM-cache reference exactly once."""

        await self._cache_lease.close()

    async def __aenter__(self) -> CpuWeightLease[CpuWeightT]:
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.close()


@dataclass(frozen=True, slots=True)
class _PreparedWeight[CpuWeightT]:
    entry: CachedCpuWeight[CpuWeightT]


@dataclass(frozen=True, slots=True)
class _PreparationFailure:
    stage: WeightLoadStage
    code: WeightLoadErrorCode
    diagnostic: str


class CpuWeightLoader[CpuWeightT, ReadyWeightT]:
    """Load and cache validated CPU weights without performing device placement."""

    def __init__(
        self,
        *,
        model_name: str,
        disk_cache: WeightDiskCache,
        weight_server_endpoint: str,
        adapter: WeightAdapter[CpuWeightT, ReadyWeightT],
        cache: DramCache[CachedCpuWeight[CpuWeightT]],
        transfer: WeightTransfer,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not weight_server_endpoint:
            raise ValueError("weight_server_endpoint must not be empty")
        self._model_name = model_name
        self._disk_cache = disk_cache
        self._weight_server_endpoint = weight_server_endpoint
        self._adapter = adapter
        self._cache = cache
        self._transfer = transfer
        self._max_file_bytes = max_safetensors_file_bytes(adapter.source_tensor_bytes())
        self._reservation_bytes = self._max_file_bytes + adapter.cpu_extra_bytes()

    @property
    def max_file_bytes(self) -> int:
        """Return the strict maximum accepted SafeTensors file size."""

        return self._max_file_bytes

    @property
    def max_cache_entry_bytes(self) -> int:
        """Return the conservative DRAM admission size for one expert."""

        return self._reservation_bytes

    async def acquire(
        self,
        key: WeightKey,
        *,
        peer_endpoints: tuple[str, ...] = (),
    ) -> CpuWeightLease[CpuWeightT]:
        """Try DRAM, disk, each peer, and the central server once in that order."""

        cached = await self._cache.acquire(key)
        if cached is not None:
            return CpuWeightLease(cached, WeightSource.DRAM)

        reservation = await self._cache.reserve(self._reservation_bytes)
        try:
            cached = await self._cache.acquire(key)
            if cached is not None:
                return CpuWeightLease(cached, WeightSource.DRAM)

            failures: list[WeightLoadFailure] = []
            prepared = await self._load_disk(key, failures)
            source = WeightSource.DISK
            if prepared is None:
                attempted_endpoints: set[str] = set()
                for endpoint in peer_endpoints:
                    normalized = endpoint.rstrip("/")
                    if not normalized or normalized in attempted_endpoints:
                        continue
                    attempted_endpoints.add(normalized)
                    prepared = await self._load_http(
                        key,
                        endpoint=normalized,
                        source=WeightSource.PEER,
                        failures=failures,
                    )
                    if prepared is not None:
                        source = WeightSource.PEER
                        break

                central = self._weight_server_endpoint.rstrip("/")
                if prepared is None and central not in attempted_endpoints:
                    prepared = await self._load_http(
                        key,
                        endpoint=central,
                        source=WeightSource.WEIGHT_SERVER,
                        failures=failures,
                    )
                    source = WeightSource.WEIGHT_SERVER

            if prepared is None:
                raise WeightLoadFailed(key, tuple(failures))
            cache_lease = await reservation.commit(
                key,
                prepared.entry,
                actual_bytes=prepared.entry.byte_count,
            )
            return CpuWeightLease(cache_lease, source)
        finally:
            await reservation.cancel()

    async def _load_disk(
        self,
        key: WeightKey,
        failures: list[WeightLoadFailure],
    ) -> _PreparedWeight[CpuWeightT] | None:
        try:
            buffer = await self._disk_cache.read(key, max_bytes=self._max_file_bytes)
        except FileNotFoundError:
            failures.append(
                WeightLoadFailure(
                    WeightSource.DISK,
                    WeightLoadStage.READ,
                    WeightLoadErrorCode.NOT_FOUND,
                    False,
                    "expert file does not exist",
                )
            )
            return None
        except InvalidWeightFile as error:
            await self._disk_cache.remove(key)
            failures.append(
                WeightLoadFailure(
                    WeightSource.DISK,
                    WeightLoadStage.READ,
                    WeightLoadErrorCode.INVALID_FORMAT,
                    False,
                    str(error),
                )
            )
            return None
        except OSError as error:
            failures.append(
                WeightLoadFailure(
                    WeightSource.DISK,
                    WeightLoadStage.READ,
                    WeightLoadErrorCode.IO,
                    True,
                    str(error),
                )
            )
            return None

        prepared = self._prepare(buffer)
        if isinstance(prepared, _PreparedWeight):
            return prepared
        buffer.close()
        await self._disk_cache.remove(key)
        failures.append(
            WeightLoadFailure(
                WeightSource.DISK,
                prepared.stage,
                prepared.code,
                False,
                prepared.diagnostic,
            )
        )
        return None

    async def _load_http(
        self,
        key: WeightKey,
        *,
        endpoint: str,
        source: WeightSource,
        failures: list[WeightLoadFailure],
    ) -> _PreparedWeight[CpuWeightT] | None:
        url = self._expert_url(endpoint, key)
        try:
            buffer = await self._transfer.download(url, max_bytes=self._max_file_bytes)
        except WeightNotFound as error:
            failures.append(
                WeightLoadFailure(
                    source,
                    WeightLoadStage.FETCH,
                    WeightLoadErrorCode.NOT_FOUND,
                    False,
                    error.diagnostic,
                )
            )
            return None
        except WeightTransferError as error:
            failures.append(
                WeightLoadFailure(
                    source,
                    WeightLoadStage.FETCH,
                    WeightLoadErrorCode.NETWORK,
                    error.retryable,
                    error.diagnostic,
                )
            )
            return None

        prepared = self._prepare(buffer)
        if isinstance(prepared, _PreparedWeight):
            return prepared
        buffer.close()
        failures.append(
            WeightLoadFailure(
                source,
                prepared.stage,
                prepared.code,
                False,
                prepared.diagnostic,
            )
        )
        return None

    def _prepare(
        self,
        buffer: AlignedWeightBuffer,
    ) -> _PreparedWeight[CpuWeightT] | _PreparationFailure:
        view = buffer.view()
        try:
            try:
                parsed = parse_safetensors(view)
            except SafeTensorFormatError as error:
                return _PreparationFailure(
                    WeightLoadStage.PARSE,
                    WeightLoadErrorCode.INVALID_FORMAT,
                    str(error),
                )
        finally:
            view.release()

        try:
            cpu_weight = self._adapter.make_cpu_weight(parsed)
        except (SafeTensorFormatError, ValueError, TypeError) as error:
            return _PreparationFailure(
                WeightLoadStage.VALIDATE,
                WeightLoadErrorCode.UNEXPECTED_METADATA,
                str(error),
            )
        except Exception as error:
            return _PreparationFailure(
                WeightLoadStage.VALIDATE,
                WeightLoadErrorCode.INTERNAL,
                str(error),
            )
        byte_count = buffer.logical_size + self._adapter.cpu_extra_bytes()
        return _PreparedWeight(CachedCpuWeight(buffer, parsed, cpu_weight, byte_count))

    def _expert_url(self, endpoint: str, key: WeightKey) -> str:
        model = quote(self._model_name, safe="")
        return f"{endpoint}/expert/{model}/{key.layer_id}/{key.expert_id}"
