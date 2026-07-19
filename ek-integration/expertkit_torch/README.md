# Expert Kit Torch integration

This package connects supported Hugging Face MoE models to Expert Kit. Attention,
routing, dense feed-forward layers, and shared experts remain in the Frontend
process. Each routed MoE layer sends its final expert assignments and FP32
routing weights through `expertkit-transport`, then receives the weighted result
aggregated across Workers.

The integration targets Transformers 4.57.3 and supports:

- Qwen3-MoE
- DeepSeek-V2
- DeepSeek-V3
- Mixtral

The model loader detects the family from `config.json`. One Frontend process
uses one device. The normal computation path is plaintext gRPC, so deployments
must run on a trusted and isolated cluster network.

## Install

Create the locked environment from this directory:

```bash
uv sync --locked
source .venv/bin/activate
```

The local uv configuration installs `expertkit-transport` from
`../../ek-transport`. A packaged deployment must make the same
`expertkit-transport==0.1.0` release available.

## Required deployment

Start the Controller, Weight Server, and one Python Worker process per compute
device before using `expertkit` mode. The Controller must publish every expert
needed by the model as ready. The numeric instance ID must match the instance
registered by those Workers.

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
    instance_id=1,
    device="cuda:0",
) as loaded:
    output = loaded.model.generate(...)
```

Use `mode="local"` to load the unmodified Hugging Face experts. Local mode is
useful as a baseline, but the complete model must fit on the Frontend device.

## Benchmark

The shared benchmark uses deterministic token IDs with an exact input length.
It performs one prefill call and then greedy one-token decode calls that reuse
the attention key/value cache. EOS does not stop the measured run, so every
configuration executes the same number of token steps.

```bash
ek-torch-benchmark \
  --model-path /models/DeepSeek-V2-Lite-Chat \
  --mode expertkit \
  --controller-endpoint 10.0.0.10:5002 \
  --instance-id 1 \
  --batch-sizes 1 32 \
  --input-length 128 \
  --output-length 20 \
  --warmup-runs 1 \
  --runs 5 \
  --device cuda:0 \
  --dtype auto \
  --json-output /tmp/deepseek-v2-benchmark.json
```

The command reports medians across measured runs:

- `Prefill ms` is the first full-input model call.
- `Prefill tok/s` is aggregate input tokens divided by prefill time.
- `Decode ms` covers the remaining `output_length - 1` cached model calls.
- `Decode tok/s` is aggregate tokens from those decode calls.
- `Decode ms/step` is the average wall time of one batched decode step.
- `Total ms` is prefill plus decode.
- `Output tok/s` counts all generated tokens over total time.

Model loading and tokenization are excluded. CUDA is synchronized only at phase
boundaries. The timings include the complete Frontend, Transport, and Worker
path in Expert Kit mode. Use the existing tracing configuration when a
Frontend, Transport, and Worker breakdown is needed.

Run the same workload locally by changing only the mode:

```bash
ek-torch-benchmark \
  --model-path /models/DeepSeek-V2-Lite-Chat \
  --mode local \
  --batch-sizes 1 \
  --input-length 128 \
  --output-length 20
```

## Tests

CPU tests cover model routing semantics, real layer IDs, bounded Transformers
class replacement, benchmark token counts, metrics, and command output:

```bash
uv run pytest tests
```

Real Qwen3 and DeepSeek-V2 checks are enabled only when their model paths are
configured:

```bash
EK_QWEN_MODEL_PATH=/models/Qwen3-30B-A3B \
EK_QWEN_CONTROLLER_ENDPOINT=10.0.0.10:5002 \
EK_QWEN_INSTANCE_ID=1 \
uv run pytest tests/test_deployment_benchmark.py -m qwen

EK_DEEPSEEK_V2_MODEL_PATH=/models/DeepSeek-V2-Lite-Chat \
EK_DEEPSEEK_V2_CONTROLLER_ENDPOINT=10.0.0.10:5002 \
EK_DEEPSEEK_V2_INSTANCE_ID=1 \
uv run pytest tests/test_deployment_benchmark.py -m deepseek_v2
```

## Current limits

- The adapters match the model classes shipped in Transformers 4.57.3.
- DeepSeek-V3 works only with weight dtypes supported by the current Worker:
  FP16, BF16, or FP32. FP8 and W8A8 DeepSeek-R1 checkpoints are not supported.
- Full Mixtral and DeepSeek-V3 deployment tests require more device memory than
  the current development host can provide after accounting for Frontend and
  Worker copies.
- The current gRPC path copies Tensor bytes through Host memory.
