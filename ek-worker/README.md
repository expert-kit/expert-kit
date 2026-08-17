# Expert Kit Python Worker

`expertkit-worker` runs routed expert FFNs for one model instance on one compute
device. It receives bounded layer batches through gRPC, the experimental
same-host shared-memory path, a static CUDA NCCL world, or Mooncake Transfer
Engine, executes only experts that the Controller has made ready, and returns
one weighted partial output per batch.

## Install

Create the default locked Torch environment from the repository root:

```bash
uv sync --locked
```

This creates the repository root `.venv` and installs Proto, Transport, Worker,
and the Torch integration. Run all commands below from the repository root.

Transfer Engine deployments must additionally install the EK safety-patched
`mooncake-transfer-engine` Linux wheel. It is intentionally not resolved from
the public package index; the Transport runtime verifies its EK capability
flags before registering any Tensor memory.

The optional CPU-only GGML environment is installed with:

```bash
uv sync --locked --extra ggml
```

The experimental NVIDIA fused environment is installed with:

```bash
uv sync --locked --extra fused
```

GGML remains experimental and CPU-only. The fused Backend supports the current
unquantized FP16 and BF16 SiLU path. Torch is the default Backend.

## Configuration

Start from
[`examples/qwen3-30b-a3b.torch.yaml`](./examples/qwen3-30b-a3b.torch.yaml).
The file is loaded once at startup and unknown fields are rejected. Worker
fields do not have environment-variable overrides.

Important settings:

- `model.instance_id` is optional. When omitted, startup resolves the instance
  named by the Controller configuration. An explicit ID must match that default.
- `model.name` must match the model name served by the Weight Server.
- `worker.id` must match the node name assigned by the Controller.
- `worker.device` is one explicit `cuda:<id>` for Torch. Start another process
  for another device.
- `worker.device_memory_limit` is the complete device budget for this process.
  Startup rejects a budget larger than the device or current available memory.
- `worker.max_batch_tokens` defaults to `4096`.
- `worker.max_active_batches_per_device` defaults to `1`.
- `worker.ggml.cpu_threads` is required when `worker.backend` is `ggml`.
- `transport.max_pending_batches_per_device` is the number of decoded requests
  allowed to wait outside the fixed execution slots. It defaults to
  `worker.max_active_batches_per_device` when omitted.
- `weight_manager.max_concurrent_loads` defaults to `64`.
- The Host weight budget defaults to enough bytes for the model's complete expert
  cache plus concurrent conversion capacity. An explicit
  `weight_manager.dram_cache.max_bytes` includes both; startup subtracts
  `max_concurrent_loads * adapter.host_conversion_temporary_bytes()` before sizing
  the LRU cache. See [quantization implementation](./QUANTIZATION.md) for details.
- Disk-cache writeback defaults to enabled. A remote weight is validated before
  it is published in the cache.
- Heartbeats default to every 3 seconds with a 10-second Controller timeout.
- Expert state changes are sent in groups of at most 64 or after 50 ms. Heartbeat
  and expert state reporting use separate control streams.

For a gRPC Tensor Worker, configure:

```yaml
transport:
  type: grpc
  max_pending_batches_per_device: 1
  listen: 0.0.0.0:51051
  advertise: worker-a100:51051
```

For a same-host shared-memory Worker, configure:

```yaml
transport:
  type: shm
  max_pending_batches_per_device: 1
  rpc_listen: 0.0.0.0:51051
  rpc_advertise: worker-a100:51051
  shared_memory_dir: /dev/shm
```

`advertise` or `rpc_advertise`, together with
`weight_manager.peer.advertise`, must be reachable from the relevant
processes. Their listen counterparts select local bind addresses. In SHM mode,
the RPC endpoint handles only session setup and small notifications; Tensor
payloads use `/dev/shm`. The Frontend and Worker must see the same shared-memory
namespace and run as the same Unix user.

For an NCCL Worker, configure a unique rank in the process group:

```yaml
transport:
  type: nccl
  max_pending_batches_per_device: 1
  control_listen: 0.0.0.0:51051
  control_advertise: worker-h200-0:51051
  rank: 1
  world_size: 2
  rendezvous_endpoint: frontend-h200:29500
  group_name: qwen3-production
```

The static world includes the Frontend and every NCCL Worker. Give every
process one unique rank and one indexed CUDA device; for a single Frontend plus
one Worker, the usual ranks are `0` and `1`. All participants must use the same
`world_size`, rendezvous endpoint, and group name. The rank-zero process hosts
the PyTorch TCP rendezvous at `rendezvous_endpoint`. The Frontend creates one
`NcclRuntime` with its own rank and passes it as the `transport_runtime` of its
`RoutedMoEClient` or `BlockingRoutedMoEClient`; that client owns and closes the
runtime. Each process must use a distinct rendezvous port for a distinct group.

For a same-host Transfer Engine Worker using P2P handshake and intra-node
NVLink, use a distinct Mooncake endpoint for the Worker process:

```yaml
transport:
  type: transfer_engine
  max_pending_batches_per_device: 1
  control_listen: 0.0.0.0:52051
  control_advertise: 127.0.0.1:52051
  segment_advertise: 127.0.0.1:12011
  metadata_server: P2PHANDSHAKE
  protocol: nvlink_intra
  device_name: ""
  max_workers: 2
  transport_hint: ""
```

The Frontend must create one process-shared `TransferEngineRuntime` with a
different `segment_name`, such as `127.0.0.1:12012`, select the same
`nvlink_intra` backend, and pass it to its routed client. One runtime is one data
backend; this MVP does not mix intra-node NVLink and cross-host RDMA peers. The
Worker registers the same generated `start_id` with both the Controller and its
private session service. The private protocol also binds a Worker-issued nonce
to a Frontend runtime generation, so stale epochs and endpoint reuse cannot
silently reach old arena addresses. A Frontend crash without a successful
`CloseSession` requires the corresponding Worker to restart before that
Frontend endpoint is reused.

Cross-host RDMA uses the same Worker type but is deliberately gated while its
native lifecycle API is being validated:

```yaml
transport:
  type: transfer_engine
  max_pending_batches_per_device: 1
  control_listen: 0.0.0.0:52051
  control_advertise: 192.0.2.11:52051
  segment_advertise: 192.0.2.11:12011
  metadata_server: P2PHANDSHAKE
  protocol: rdma
  device_name: mlx5_0
  max_workers: 2
  transport_hint: ""
  enable_experimental_rdma: true
```

The Frontend must select `rdma`, an indexed CUDA device, a peer-reachable
segment endpoint, an explicit RDMA device, and the same opt-in. Startup requires
native `EK_FORCE_CONFIGURED_RDMA_TRANSPORT` and an exact `rdma` result from
`get_configured_backend()`, plus
`EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION` and
`invalidate_drained_rdma_segment(target_session)`. RDMA requires
`metadata_server: P2PHANDSHAKE`. A wheel missing any one of those contracts
fails before registering memory. TENT must remain disabled; merely defining
`MC_USE_TENT` or `MC_USE_TEV1`, including with value `0`, is rejected.

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
uv run --package expertkit-worker ek-worker --config /absolute/path/to/worker.yaml
```

The configuration path may instead be selected with `EK_CONFIG`. An explicit
`--config` takes precedence:

```bash
EK_CONFIG=/absolute/path/to/worker.yaml \
  uv run --package expertkit-worker ek-worker
```

Or use the unified launcher, which replaces itself with the same Python process:

```bash
uv run --package expertkit-worker \
  target/release/ek-cli --config /absolute/path/to/worker.yaml worker
```

The unified launcher accepts the same environment-based selection:

```bash
EK_CONFIG=/absolute/path/to/worker.yaml \
  uv run --package expertkit-worker target/release/ek-cli worker
```

Sending `SIGTERM` starts the Controller-coordinated shutdown. The Worker keeps
serving the published topology until replacements are ready, then stops new
admission, finishes already accepted work, flushes state, and exits within
`worker.shutdown_grace_secs`.

## Logging and observability

The default console output is human-readable and matches the Rust services'
`<LEVEL>(timestamp) message` layout. Set `logging.format: json` when a log
collector requires structured JSON.

Prometheus and OpenTelemetry require the optional dependencies:

```bash
uv sync --locked --extra observability
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
Worker child spans for request decoding, waiting for an execution slot, active
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

The gRPC path serializes Tensor bytes through Host memory. The SHM path avoids
protobuf Tensor payloads and loopback Tensor copies, but CUDA inputs and outputs
still pass through pinned Host memory. The NCCL path keeps admission, metadata,
completion, and structured errors on a private gRPC control plane; only the
three input Tensors and one output Tensor use NCCL point-to-point operations.
NCCL membership is static, and communicator failure or rank replacement
currently requires recreating the complete group. Transfer Engine uses a
registered GPU arena and one-sided READ/WRITE operations; it accepts dynamic
Worker generations without creating a global communicator. The validated safe
lifecycle is same-host `nvlink_intra`; experimental RDMA uses the same
generation binding and two-phase close only when the native binding proves
forced RDMA selection, reports the actual backend exactly, and supports drained
RDMA descriptor invalidation. Graceful retire gates sibling sessions, drains
DMA, synchronously deregisters the owning arena, and then evicts the old remote
address/rkey descriptor before commit completes. Shared QPs remain installed.
Any ambiguous step retains the arena and requires process restart.

The Worker retains an RDMA endpoint-to-runtime generation tombstone after the
last session closes. Route removal and re-add from the same live runtime are
supported, but a different process generation cannot reuse that endpoint until
the Worker restarts; alternatively, advertise a new endpoint. Network faults,
crash windows, and deregistration failures still need real hardware fault
injection before this opt-in can be promoted to production. The first Transfer
Engine version still performs one local device copy at each endpoint. NVSHMEM
and Arrow Flight are not implemented by this Worker.

There is no TLS, mTLS, authentication, or authorization. Run all Controller,
Worker, Weight Manager, Weight Server, metrics, and tracing endpoints only on a
trusted isolated network protected by network-level rules.

## Tests

```bash
uv run --package expertkit-worker ruff check ek-worker/src ek-worker/tests
uv run --package expertkit-worker ruff format --check ek-worker/src ek-worker/tests
uv run --package expertkit-worker pytest ek-worker/tests
```

CUDA, direct-I/O, and real multi-process checks require the matching hardware or
filesystem and are marked separately.

## Quantized routed experts

Weight Server reads checkpoint `config.json` and publishes the **routed expert**
recipe through `/meta/model/{model}`. The Worker selects an adapter at startup and
validates the tensors against that recipe. Original V4 declares global FP8 but
stores routed experts in FP4; selecting from the global `quant_method` alone is
incorrect. Quantization also needs its activation scheme and scale granularity.

Use [`examples/deepseek-v4-flash.torch.yaml`](./examples/deepseek-v4-flash.torch.yaml)
for either V4 checkpoint, changing `model.name` to the Weight Server directory
name. Keep `auto_model_metadata: true` and `metadata_required: true`; no manual
quantization entry is needed. Dimensions and routing settings still belong in
Worker configuration and are checked against the server. Change `weight_version`
and the disk-cache directory when replacing a checkpoint.

| Checkpoint format | Torch execution | Ready GPU weights |
| --- | --- | --- |
| Original DeepSeek-V4-Flash, packed E2M1 FP4 + E8M0 scales | A100 compatibility path: decode at placement, emulate block FP8 activation rounding, use floating GEMMs | BF16 with the example configuration |
| `sgl-npu/DeepSeek-V4-Flash-W8A8`, compressed-tensors `int-quantized` | Dynamic per-token symmetric INT8 activation quantization, INT8 GEMM with INT32 accumulation, per-channel rescaling | INT8 matrices + FP32 scales |
| Symmetric canonical AutoGPTQ v1 INT4 | Decode at placement, floating GEMMs | FP16/BF16 |

The W8A8 adapter accepts per-channel `weight_scale` shaped `[output, 1]` in
FP16/BF16/FP32, and optional INT8 zero-valued `weight_zero_point`. It rejects
static or asymmetric activations, grouped weights, transforms, regex targets,
partially ignored routed experts, and unexpected auxiliary tensors. The name
“W8A8” alone does not establish checkpoint compatibility. Older `blockwise_int8`
exports, QuaRot transforms, and Ascend execution require their own validated
format or backend support.

Both V4 paths apply the configured SwiGLU clipping, calculate the intermediate
activation in FP32, and apply routing weights **before** the down projection.
The FP4 compatibility path uses different GEMM accumulation from the native V4
kernels; it is not a claim of bitwise equality or native FP4 acceleration. Its
expanded weights do not save GPU weight memory. W8A8 keeps weights compressed,
but the eager quantization and scaling operations can cost more time for small
batches; no end-to-end speedup has been established.

These changes support **routed expert computation**. The current Frontend
loader does not implement the complete V4 model, including its attention,
routing, shared experts, and hyper-connections. Full-model generation requires
that integration and separate accuracy and performance validation. A100 tests
use synthetic expert tensors with the official 4096/2048 dimensions; no V4
checkpoint weights are downloaded for these tests.

Run CPU tests and the CUDA qualification from the repository root:

```bash
uv run pytest ek-worker/tests/unit
uv run pytest ek-worker/tests/integration/weights/test_v4_quantized_cuda.py -v
cargo test -p ek-db --lib
```

CUDA qualification must report two passed tests, not skips. The metadata and
layout fixtures were taken from the [original V4 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
and [the selected W8A8 export](https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8).
Expert arithmetic follows the [official inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py).
