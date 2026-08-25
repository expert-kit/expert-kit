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

## Load a model

`load_model` returns a context-manageable object containing the model and
tokenizer. Closing it also closes the shared Transport client:

```python
from expertkit_torch import load_model

with load_model(
    "/models/DeepSeek-V2-Lite-Chat",
    mode="expertkit",
    controller_endpoint="10.0.0.10:5002",
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

```bash
uv run --package expertkit-torch ek-torch-benchmark \
  --model-path /models/DeepSeek-V2-Lite-Chat \
  --mode expertkit \
  --controller-endpoint 10.0.0.10:5002 \
  --dataset-path /datasets/ShareGPT.json \
  --seed 0 \
  --num-prompts 16 \
  --batch-sizes 1 \
  --output-length 128 \
  --warmup-runs 1 \
  --device cuda:0 \
  --dtype auto \
  --json-output /tmp/deepseek-v2-benchmark.json
```

The Frontend does not select a Transport on the command line. Each Worker
registers `grpc` or `shm`, the Controller publishes that value in topology, and
the Transport package creates the matching connection. An SHM Worker and the
Frontend must share the same OS shared-memory namespace and Unix user. SHM does
not work across machines and does not remove GPU-to-Host or Host-to-GPU
transfers.

The command reports medians across the selected static input batches. The
prompt count must be divisible by every requested batch size:

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
uv run --package expertkit-torch ek-torch-benchmark \
  --model-path /models/DeepSeek-V2-Lite-Chat \
  --mode local \
  --dataset-path /datasets/ShareGPT.json \
  --num-prompts 16 \
  --batch-sizes 1 \
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
EK_QWEN_CONTROLLER_ENDPOINT=10.0.0.10:5002 \
EK_BENCHMARK_DATASET_PATH=/datasets/ShareGPT.json \
uv run --package expertkit-torch pytest \
  ek-integration/expertkit_torch/tests/test_deployment_benchmark.py -m qwen

EK_DEEPSEEK_V2_MODEL_PATH=/models/DeepSeek-V2-Lite-Chat \
EK_DEEPSEEK_V2_CONTROLLER_ENDPOINT=10.0.0.10:5002 \
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
