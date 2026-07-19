# Expert Kit Python Worker

`expertkit-worker` runs routed expert FFNs for one model instance on one compute
device. It receives bounded layer batches through gRPC or the experimental
same-host shared-memory data path, executes only experts that the Controller has
made ready, and returns one weighted partial output per batch.

## Install

Create the default locked Torch environment from the repository root:

```bash
uv sync --project ek-worker --locked
```

Activate `ek-worker/.venv` before using the unified CLI so `ek-cli worker` can
find the `ek-worker` executable:

```bash
source ek-worker/.venv/bin/activate
```

The optional CPU-only GGML environment is installed with:

```bash
uv sync --project ek-worker --locked --extra ggml
```

The experimental NVIDIA fused environment is installed with:

```bash
uv sync --project ek-worker --locked --extra fused
```

GGML remains experimental and CPU-only. The fused Backend supports the current
unquantized FP16 and BF16 SiLU path. Torch is the default Backend.

## Configuration

Start from
[`examples/qwen3-30b-a3b.torch.yaml`](./examples/qwen3-30b-a3b.torch.yaml).
The file is loaded once at startup and unknown fields are rejected. Worker
fields do not have environment-variable overrides.

Important settings:

- `model.instance_id` is the numeric instance stored by the Controller.
- `model.name` must match the model name served by the Weight Server.
- `worker.id` must match the node name assigned by the Controller.
- `worker.device` is one explicit `cuda:<id>` for Torch. Start another process
  for another device.
- `worker.device_memory_limit` is the complete device budget for this process.
  Startup rejects a budget larger than the device or current available memory.
- `worker.max_batch_tokens` defaults to `4096`.
- `worker.max_active_batches_per_device` defaults to `1`.
- `transport.max_pending_batches_per_device` defaults to the active-batch count.
- `weight_manager.max_concurrent_loads` defaults to `64`.
- The DRAM cache limit defaults to enough bytes for the model's complete expert
  set. Set `weight_manager.dram_cache.max_bytes` to impose a smaller LRU cache.
- Disk-cache writeback defaults to enabled. A remote weight is validated before
  it is published in the cache.
- Heartbeats default to every 3 seconds with a 10-second Controller timeout.
- Expert state changes are sent in groups of at most 64 or after 50 ms. Heartbeat
  and expert state reporting use separate control streams.

`transport.grpc.advertise` and `weight_manager.peer.advertise` must be reachable
from other processes. Their `listen` counterparts select local bind addresses.
The same computation gRPC endpoint also handles shared-memory setup and
notifications. Shared memory is selected by the Frontend and requires both
processes to see the same `/dev/shm` namespace.

The Weight Manager looks for a requested assigned expert in this order:

```text
DRAM cache -> disk cache -> Controller-provided peers -> Weight Server
```

A computation request never starts a weight load. The Controller sends placement
commands, and the Worker reports an expert as ready only after its final Backend
weight is usable on the configured device.

## Start

Run the Worker directly:

```bash
ek-worker --config /absolute/path/to/worker.yaml
```

The configuration path may instead be selected with `EK_CONFIG`. An explicit
`--config` takes precedence:

```bash
EK_CONFIG=/absolute/path/to/worker.yaml ek-worker
```

Or use the unified launcher, which replaces itself with the same Python process:

```bash
target/release/ek-cli --config /absolute/path/to/worker.yaml worker
```

The unified launcher accepts the same environment-based selection:

```bash
EK_CONFIG=/absolute/path/to/worker.yaml target/release/ek-cli worker
```

Sending `SIGTERM` starts the Controller-coordinated shutdown. The Worker keeps
serving the published topology until replacements are ready, then stops new
admission, finishes already accepted work, flushes state, and exits within
`worker.shutdown_grace_secs`.

## Logging and observability

The default service output is structured JSON through `structlog`. Use
`logging.format: console` only for interactive development.

Prometheus and OpenTelemetry require the optional dependencies:

```bash
uv sync --project ek-worker --locked --extra observability
```

Both are disabled by default. Enable them in the Worker YAML:

```yaml
observability:
  prometheus:
    enabled: true
    listen: 127.0.0.1:9091
  tracing:
    enabled: true
    endpoint: http://127.0.0.1:4317
    sample_ratio: 0.01
```

Prometheus serves `/metrics`. The OpenTelemetry exporter uses asynchronous,
sampled plaintext OTLP over gRPC.

For each sampled computation call, the automatic gRPC server span contains
Worker child spans for request decoding, waiting for an active position, active
batch execution, input preparation, Backend submission and completion, output
preparation, device completion waiting, and response encoding. CUDA execution
adds the following attributes to `worker.batch.execute`:

- `expertkit.cuda.input_stage_ms`
- `expertkit.cuda.backend_stage_ms`
- `expertkit.cuda.output_stage_ms`
- `expertkit.cuda.total_stage_ms`

These values use CUDA Events on the Worker's existing stream and are read only
after the response path's existing completion wait. Tracing does not add a CUDA
synchronization. The stage values include any stream idle time between their
recorded boundaries, so they describe Worker stream stages rather than pure
copy-engine or kernel-only time. Unsampled requests skip custom spans and CUDA
timing. An incoming standard `traceparent` remains on the Host and parents the
Worker spans; it is not part of the computation payload or any Tensor.

## Transport and security limits

The current computation path uses only plaintext gRPC. It serializes Tensor
bytes through Host memory and is not a GPU-direct path. SHM, RDMA, NCCL,
NVSHMEM, Arrow Flight, and Mooncake are not implemented by this Worker.

There is no TLS, mTLS, authentication, or authorization. Run all Controller,
Worker, Weight Manager, Weight Server, metrics, and tracing endpoints only on a
trusted isolated network protected by network-level rules.

## Tests

```bash
uv run --project ek-worker ruff check ek-worker/src ek-worker/tests
uv run --project ek-worker ruff format --check ek-worker/src ek-worker/tests
uv run --project ek-worker pytest ek-worker/tests
```

CUDA, direct-I/O, and real multi-process checks require the matching hardware or
filesystem and are marked separately.
