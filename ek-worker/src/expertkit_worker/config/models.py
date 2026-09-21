"""Pydantic models for the Python Worker startup configuration."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ByteSize,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class BackendName(StrEnum):
    """Built-in Compute backend selected for this Worker process."""

    TORCH = "torch"
    GGML = "ggml"
    FUSED = "fused"


class ActivationDType(StrEnum):
    """Activation dtypes accepted by the v2 computation contract."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


class LogLevel(StrEnum):
    """Supported process log thresholds."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    """Supported process log renderers."""

    JSON = "json"
    CONSOLE = "console"


def _validate_network_address(value: str) -> str:
    if "://" in value:
        raise ValueError("network address must use host:port without a URL scheme")

    if value.startswith("["):
        closing = value.find("]")
        if closing <= 1 or closing + 1 >= len(value) or value[closing + 1] != ":":
            raise ValueError("IPv6 network address must use [host]:port")
        host = value[1:closing]
        port_text = value[closing + 2 :]
    else:
        try:
            host, port_text = value.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError("network address must use host:port") from exc

    if not host or any(character.isspace() for character in host):
        raise ValueError("network address host must be non-empty and contain no whitespace")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("network address port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("network address port must be between 1 and 65535")
    return value


NetworkAddress = Annotated[str, AfterValidator(_validate_network_address)]


def _validate_absolute_path(value: Path) -> Path:
    if not value.is_absolute():
        raise ValueError("path must be absolute")
    return value


AbsolutePath = Annotated[Path, AfterValidator(_validate_absolute_path)]
PositiveByteSize = Annotated[
    ByteSize,
    Field(gt=0),
    PlainSerializer(
        lambda size: size.human_readable(decimal=False),
        return_type=str,
        when_used="json",
    ),
]


class ModelConfig(_StrictModel):
    """Model metadata that fixes Worker-batch validation and expert layout."""

    instance_id: int | None = Field(default=None, gt=0)
    name: str = Field(min_length=1)
    weight_version: str = Field(min_length=1)
    num_layers: int = Field(gt=0)
    experts_per_layer: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    expert_intermediate_dim: int = Field(gt=0)
    top_k: int = Field(gt=0)
    activation_dtype: ActivationDType
    weight_dtype: ActivationDType
    activation: Literal["silu"] = "silu"

    @model_validator(mode="after")
    def validate_routing_shape(self) -> ModelConfig:
        """Ensure the fixed top-k is representable by the configured expert set."""

        if self.top_k > self.experts_per_layer:
            raise ValueError("model.top_k cannot exceed model.experts_per_layer")
        return self


class GgmlConfig(_StrictModel):
    """Experimental CPU-only GGML backend settings."""

    cpu_threads: int = Field(gt=0)


class WorkerProcessConfig(_StrictModel):
    """Identity, Backend, device, admission, and shutdown settings."""

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    backend: BackendName = BackendName.TORCH
    device: str
    max_batch_tokens: int = Field(default=4096, gt=0)
    max_active_batches_per_device: int = Field(default=1, gt=0)
    device_memory_limit: PositiveByteSize
    shutdown_grace_secs: float = Field(default=30.0, gt=0)
    ggml: GgmlConfig | None = None

    @model_validator(mode="after")
    def validate_backend(self) -> WorkerProcessConfig:
        """Enforce device support and Backend-specific configuration."""

        is_cpu = self.device == "cpu"
        is_cuda = re.fullmatch(r"cuda:\d+", self.device) is not None
        is_npu = re.fullmatch(r"npu:\d+", self.device) is not None

        if self.backend is BackendName.GGML and self.device != "cpu":
            raise ValueError("the MVP GGML backend requires worker.device: cpu")

        # TODO: NPU support on fused backend, and then merge these two if below
        if self.backend is BackendName.TORCH and not (is_cpu or is_cuda or is_npu):
            raise ValueError(
                f"the MVP {self.backend.value} backend requires cpu, cuda:<id> or npu:<id>"
            )
        # NOTE: Currently only CUDA is supported in fused backend
        if self.backend is BackendName.FUSED and not is_cuda:
            raise ValueError(f"the MVP {self.backend.value} backend requires cuda:<id>")

        if self.backend is BackendName.GGML and self.ggml is None:
            raise ValueError("worker.ggml configuration is required when worker.backend is ggml")
        if self.backend is not BackendName.GGML and self.ggml is not None:
            raise ValueError("worker.ggml is valid only when worker.backend is ggml")
        return self


class GrpcTransportConfig(_StrictModel):
    """gRPC Tensor receiver settings for one Worker process."""

    type: Literal["grpc"]
    max_pending_batches_per_device: int = Field(gt=0)
    listen: NetworkAddress
    advertise: NetworkAddress


class ShmTransportConfig(_StrictModel):
    """Same-host shared-memory receiver and notification RPC settings."""

    type: Literal["shm"]
    max_pending_batches_per_device: int = Field(gt=0)
    rpc_listen: NetworkAddress
    rpc_advertise: NetworkAddress
    shared_memory_dir: Literal["/dev/shm"] = "/dev/shm"


TransportConfig = Annotated[
    GrpcTransportConfig | ShmTransportConfig,
    Field(discriminator="type"),
]


class ControllerConfig(_StrictModel):
    """Controller connection and Worker heartbeat timing."""

    endpoint: NetworkAddress
    heartbeat_interval_secs: float = Field(default=3.0, gt=0)
    heartbeat_timeout_secs: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def validate_heartbeat_timing(self) -> ControllerConfig:
        """Reject a timeout that can expire before the next expected heartbeat."""

        if self.heartbeat_timeout_secs <= self.heartbeat_interval_secs:
            raise ValueError(
                "controller.heartbeat_timeout_secs must exceed heartbeat_interval_secs"
            )
        return self


class DramCacheConfig(_StrictModel):
    """Application-managed Host weight-cache byte budget."""

    max_bytes: PositiveByteSize | None = None


class DiskCacheConfig(_StrictModel):
    """Persistent per-expert SafeTensors cache settings."""

    path: AbsolutePath
    writeback: bool = True


class StateReportConfig(_StrictModel):
    """Bounded coalescing thresholds for expert state changes."""

    max_updates: int = Field(default=64, gt=0, le=64)
    max_delay_ms: int = Field(default=50, gt=0)


class PeerWeightConfig(_StrictModel):
    """Local peer-weight HTTP server addresses."""

    listen: NetworkAddress
    advertise: AnyHttpUrl


class WeightManagerConfig(_StrictModel):
    """Weight loading, source, cache, peer, and state-report settings."""

    max_concurrent_loads: int = Field(default=64, gt=0)
    dram_cache: DramCacheConfig = Field(default_factory=DramCacheConfig)
    disk_cache: DiskCacheConfig
    peer: PeerWeightConfig
    weight_server_endpoint: AnyHttpUrl
    state_report: StateReportConfig = Field(default_factory=StateReportConfig)


class PrometheusConfig(_StrictModel):
    """Optional Prometheus exporter settings."""

    enabled: bool = False
    listen: NetworkAddress = "0.0.0.0:9091"


class TracingConfig(_StrictModel):
    """Optional OpenTelemetry exporter settings."""

    enabled: bool = False
    endpoint: AnyHttpUrl | None = None
    sample_ratio: float = Field(default=0.01, gt=0, le=1)

    @model_validator(mode="after")
    def validate_export_endpoint(self) -> TracingConfig:
        """Require an explicit collector only when tracing is enabled."""

        if self.enabled and self.endpoint is None:
            raise ValueError("observability.tracing.endpoint is required when tracing is enabled")
        if self.enabled and self.endpoint is not None and self.endpoint.scheme != "http":
            raise ValueError("the MVP OpenTelemetry exporter requires a plaintext HTTP endpoint")
        return self


class ObservabilityConfig(_StrictModel):
    """Optional metrics and tracing settings, both disabled by default."""

    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)


class LoggingConfig(_StrictModel):
    """Process-wide logging threshold and output format."""

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.CONSOLE


class WorkerConfig(_StrictModel):
    """Complete immutable configuration for one Python Worker process."""

    model: ModelConfig
    worker: WorkerProcessConfig
    transport: TransportConfig
    controller: ControllerConfig
    weight_manager: WeightManagerConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @model_validator(mode="before")
    @classmethod
    def resolve_cross_section_defaults(cls, value: Any) -> Any:
        """Resolve pending Transport capacity from active computation capacity."""

        if not isinstance(value, dict):
            return value
        resolved = dict(value)
        worker = resolved.get("worker")
        transport = resolved.get("transport")
        if isinstance(worker, dict) and isinstance(transport, dict):
            transport = dict(transport)
            transport.setdefault(
                "max_pending_batches_per_device",
                worker.get("max_active_batches_per_device", 1),
            )
            resolved["transport"] = transport
        return resolved

    @model_validator(mode="after")
    def validate_backend_combination(self) -> WorkerConfig:
        """Reject unused or incomplete Backend-specific configuration."""
        # TODO: NPU support on SHM?
        if self.worker.device.startswith("npu:") and isinstance(self.transport, ShmTransportConfig):
            raise ValueError("NPU workers currently require the gRPC transport")

        if self.worker.backend is BackendName.FUSED:
            supported = {ActivationDType.FP16, ActivationDType.BF16}
            if (
                self.model.activation_dtype not in supported
                or self.model.weight_dtype not in supported
            ):
                raise ValueError(
                    "the fused backend supports only FP16 or BF16 activations and weights"
                )
        return self
