# Expert Kit Torch integration

This package connects supported Hugging Face MoE models to Expert Kit. Attention,
routing, dense feed-forward layers, and shared experts remain in the Frontend
process. Each routed MoE layer sends its final expert assignments and FP32
routing weights through `expertkit-transport`, then receives the weighted result
aggregated across Workers.

The integration targets Transformers 5.5.3 and supports:

- Qwen3-MoE
- DeepSeek-V2
- DeepSeek-V3
- Mixtral

The model loader detects the family from `config.json`. One Frontend process
uses one device. The default computation path is plaintext gRPC, so deployments
must run on a trusted and isolated cluster network.

## Install

Create the shared locked environment from the repository root:

```bash
uv sync --locked
```

The root workspace installs Proto, Transport, Worker, and this Torch integration
into the root `.venv`. It uses official PyPI by default.

## Required deployment

Start the Controller, Weight Server, and one Python Worker process per compute
device before using `expertkit` mode. The Controller must publish every expert
needed by the model as ready. Worker and Frontend startup both resolve the
Controller's configured default instance.

The Frontend sends original model layer IDs. DeepSeek therefore skips its early
dense-only layers, while Qwen follows its configured sparse-layer positions.
The Worker and Weight Server must use the same checkpoint configuration.

For an NCCL deployment, create one process-shared runtime for the Frontend and
pass it to `load_model`. The Frontend and all NCCL Workers must use unique ranks
in the same static world, matching `world_size`, rendezvous endpoint, and group
name:

```python
from expertkit_torch import load_model
from expertkit_transport.transports.nccl import NcclRuntime, NcclRuntimeConfig

runtime = NcclRuntime(
    NcclRuntimeConfig(
        rank=0,
        world_size=2,
        rendezvous_endpoint="frontend-h200:29500",
        group_name="qwen3-production",
        device="cuda:0",
    )
)
with load_model(
    "/models/DeepSeek-V2-Lite-Chat",
    mode="expertkit",
    controller_endpoint="192.0.2.10:5002",
    device="cuda:0",
    timeout_seconds=120.0,
    transport_runtime=runtime,
) as loaded:
    output = loaded.model.generate(...)
```

The first routed call lazily establishes the static NCCL world, so set
`timeout_seconds` high enough to cover rendezvous and communicator setup on the
deployment. Later routed calls reuse that world. The loaded model's Transport
client owns and closes the runtime. NCCL carries
CUDA Tensor payloads; a private gRPC endpoint still handles admission,
metadata, completion, and errors. If `torch.distributed` is already initialized,
the current implementation can reuse only an exact matching EK `WORLD`; it does
not yet create a separate NCCL group beside an unrelated vLLM or torchrun world.

For Transfer Engine, pass one process-wide Mooncake runtime in the same way:

```python
from expertkit_torch import load_model
from expertkit_transport.transports.transfer_engine import (
    TransferEngineRuntime,
    TransferEngineRuntimeConfig,
)

runtime = TransferEngineRuntime(
    TransferEngineRuntimeConfig(
        segment_name="127.0.0.1:12012",
        metadata_server="P2PHANDSHAKE",
        protocol="nvlink_intra",
        device="cuda:0",
    )
)
with load_model(
    "/models/DeepSeek-V2-Lite-Chat",
    mode="expertkit",
    controller_endpoint="127.0.0.1:5002",
    device="cuda:0",
    timeout_seconds=120.0,
    transport_runtime=runtime,
) as loaded:
    output = loaded.model.generate(...)
```

Use distinct `segment_name` endpoints for the Frontend and Worker. Both sides
must select the same single backend. Install an EK safety-capable Linux wheel
explicitly; the validated `nvlink_intra` profile checks terminal DMA semantics,
GPUDirect visibility, registration reference counts, forced selection, and
drained IPC-cache invalidation before memory registration.

An experimental cross-host runtime can instead request RDMA explicitly:

```python
runtime = TransferEngineRuntime(
    TransferEngineRuntimeConfig(
        segment_name="192.0.2.10:12012",
        metadata_server="P2PHANDSHAKE",
        protocol="rdma",
        device="cuda:0",
        device_name="mlx5_0",
        enable_experimental_rdma=True,
    )
)
```

The Worker needs a peer-reachable endpoint, the same `rdma` selection and
opt-in, and its own RDMA device. This profile additionally requires native
`EK_FORCE_CONFIGURED_RDMA_TRANSPORT` and an exact `rdma` result from
`get_configured_backend()`, plus
`EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION` and
`invalidate_drained_rdma_segment(target_session)`. RDMA requires
`metadata_server="P2PHANDSHAKE"`. A wheel missing any one of those contracts
fails closed before memory registration. Defining `MC_USE_TENT` or
`MC_USE_TEV1`, even as `0`, is rejected. Graceful route re-add is limited to the
same runtime generation; a new process must use a new endpoint or restart the
Worker that retains the old endpoint-generation tombstone.

## Load a model

`load_model` returns a context-manageable object containing the model and
tokenizer. Closing it also closes the shared Transport client:

```python
from expertkit_torch import load_model

with load_model(
    "/models/DeepSeek-V2-Lite-Chat",
    mode="expertkit",
    controller_endpoint="192.0.2.10:5002",
    device="cuda:0",
) as loaded:
    output = loaded.model.generate(...)
```

Use `mode="local"` to load the unmodified Hugging Face experts. Local mode is
useful as a baseline, but the complete model must fit on the Frontend device.

## Benchmark

The shared benchmark reads native ShareGPT JSON, applies the same prompt-length
limits as vLLM's ShareGPT benchmark, and selects prompts with a deterministic
shuffle. Each static Torch batch is left-padded, performs one prefill call, and
then runs greedy one-token decode calls that reuse the attention key/value
cache. EOS does not stop the run, so every prompt generates the configured
number of tokens.
The CLI also displays one prompt-level progress bar; each completed forward
batch advances it by the batch's actual size, including a smaller final batch.

```bash
uv run --package expertkit-torch ek-torch-benchmark run \
  --model-path /models/DeepSeek-V2-Lite-Chat \
  --mode expertkit \
  --controller-endpoint 192.0.2.10:5002 \
  --dataset-path /datasets/ShareGPT.json \
  --device-platform cuda \
  --device-ids 0 \
  --seed 0 \
  --num-prompts 16 \
  --max-concurrency 1 \
  --output-length 128 \
  --warmup-runs 1 \
  --dtype auto \
  --json-output /tmp/deepseek-v2-benchmark.json
```

For generated Ascend inputs, pass the flat YAML as optional defaults. Explicit
`run` options override the corresponding file values:

```bash
uv run --package expertkit-torch ek-torch-benchmark \
  --config dev/ascend/generated/torch-bench.yaml run \
  --device-ids 0 --max-concurrency 4
```

The Frontend does not select a Transport on the command line. Each Worker
registers `grpc`, `shm`, `nccl`, or `transfer_engine`, the Controller publishes
that value in topology, and the Transport package creates the matching
connection. NCCL and Transfer Engine require the corresponding process-shared
runtime shown above. An SHM Worker and the Frontend must share the same OS
shared-memory namespace and Unix user. SHM does not work across machines and
does not remove GPU-to-Host or Host-to-GPU transfers.

The command reports one measured static batch per rank. Global maximum
concurrency must be divisible by the selected rank count; each rank receives
`max_concurrency / rank_count` requests. Multi-device runs spawn one process
per device without DDP or HCCL:

- `Prefill ms` is the first full-input model call.
- `Prefill tok/s` is aggregate input tokens divided by prefill time.
- `Decode ms` covers the remaining `output_length - 1` cached model calls.
- `Decode tok/s` is aggregate tokens from those decode calls.
- `Decode ms/step` is the average wall time of one batched decode step.
- `Total ms` is prefill plus decode.
- `Output tok/s` counts all generated tokens over total time.

Model loading and tokenization are excluded. CUDA and NPU devices are
synchronized only at phase boundaries. The timings include the complete
Frontend, Transport, and Worker path in Expert Kit mode. Use the existing
tracing configuration when a Frontend, Transport, and Worker breakdown is
needed.

When `--json-output` is set, every measured batch also records its generated
token IDs and decoded text. Token transfer and decoding happen after timing, so
they do not inflate the latency values. Compare those fields between otherwise
identical local and Expert Kit runs when qualifying a dependency upgrade.

Run the same workload locally by changing only the mode:

```bash
uv run --package expertkit-torch ek-torch-benchmark run \
  --model-path /models/DeepSeek-V2-Lite-Chat \
  --mode local \
  --dataset-path /datasets/ShareGPT.json \
  --num-prompts 16 \
  --device-platform cuda \
  --device-ids 0 \
  --max-concurrency 1 \
  --output-length 20
```

## Tests

CPU tests cover model routing semantics, real layer IDs, bounded Transformers
class replacement, benchmark token counts, metrics, and command output:

```bash
uv run --package expertkit-torch ruff check \
  ek-integration/expertkit_torch/expertkit_torch \
  ek-integration/expertkit_torch/tests
uv run --package expertkit-torch ruff format --check \
  ek-integration/expertkit_torch/expertkit_torch \
  ek-integration/expertkit_torch/tests
uv run --package expertkit-torch pytest ek-integration/expertkit_torch/tests
```

Real Qwen3 and DeepSeek-V2 checks are enabled only when their model paths are
configured:

```bash
EK_QWEN_MODEL_PATH=/models/Qwen3-30B-A3B \
EK_QWEN_CONTROLLER_ENDPOINT=192.0.2.10:5002 \
EK_BENCHMARK_DATASET_PATH=/datasets/ShareGPT.json \
uv run --package expertkit-torch pytest \
  ek-integration/expertkit_torch/tests/test_deployment_benchmark.py -m qwen

EK_DEEPSEEK_V2_MODEL_PATH=/models/DeepSeek-V2-Lite-Chat \
EK_DEEPSEEK_V2_CONTROLLER_ENDPOINT=192.0.2.10:5002 \
EK_BENCHMARK_DATASET_PATH=/datasets/ShareGPT.json \
uv run --package expertkit-torch pytest \
  ek-integration/expertkit_torch/tests/test_deployment_benchmark.py -m deepseek_v2
```

## Current limits

- The adapters match the model classes shipped in Transformers 5.5.3.
- DeepSeek-V3 works only with weight dtypes supported by the current Worker:
  FP16, BF16, or FP32. FP8 and W8A8 DeepSeek-R1 checkpoints are not supported.
- Full Mixtral and DeepSeek-V3 deployment tests require more device memory than
  the current development host can provide after accounting for Frontend and
  Worker copies.
- The gRPC path copies Tensor bytes through protobuf and Host memory.
- The experimental shared-memory path avoids protobuf and loopback-socket
  Tensor copies, but still stages data through pinned Host memory.
