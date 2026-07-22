# Deploying DeepSeek-V2-Lite with Expert Kit

## Overview

This guide deploys `deepseek-ai/DeepSeek-V2-Lite-Chat` with the Python Worker
and either the Transformers 5.5.3 or vLLM 0.25.1 Frontend:

```text
Transformers wrapper --\
                       +-> process-local BlockingRoutedMoEClient -> Transport
vLLM wrapper ---------/                                      |
                                                        Python Worker
                                                              |
                                                        Torch backend
```

The framework wrappers do not share a Python client object across processes.
Each creates its own process-local
`expertkit_transport.BlockingRoutedMoEClient`. Transformers or vLLM retains its
native router and local shared experts; `expertkit-transport` owns topology,
Worker grouping, gRPC/SHM selection, sending, retry, and weighted result
aggregation.

The base example uses one Worker on one NVIDIA A100 40 GB. The Worker loads all
1,664 routed experts. The Transformers Frontend and Worker use the same GPU and
communicate over gRPC. Later sections show the same-host SHM configuration and
an opt-in vLLM smoke test.

## Requirements

- Linux, Rust, Cargo, Python 3.12,
  [uv 0.11.30](https://docs.astral.sh/uv/), and Docker Compose.
- The official BF16 `DeepSeek-V2-Lite-Chat` checkpoint.
- One NVIDIA A100 40 GB or another GPU with enough memory for the Worker and
  Frontend.
- At least 32 GB of available Host memory for the Worker weight cache.
- A local filesystem that supports direct I/O for the Worker disk cache.

All commands below run from the repository root. Set these paths first:

```bash
export DEEPSEEK_ROOT=/path/to/DeepSeek-V2-Lite-Chat
# EK_RUN is the temporary directory for this deployment.
export EK_RUN=/tmp/expert-kit/deepseek-v2-lite
# EK_WORKER_CACHE stores the Worker's expert cache.
export EK_WORKER_CACHE=/absolute/path/to/deepseek-v2-lite-worker-cache

mkdir -p "$EK_RUN" "$EK_WORKER_CACHE" /tmp/expert-kit/cache
```

The last component of `DEEPSEEK_ROOT` must be `DeepSeek-V2-Lite-Chat`. Expert
Kit uses that directory name as the model name.

## Installation

Build the Rust services and install the Python Worker and Frontend:

```bash
cargo build --release --bin ek-cli
uv sync --locked
```

This consumes the repository's only committed `uv.lock` and creates the root
`.venv`. The committed workspace resolves against official PyPI. Configure a
regional mirror only in developer-local uv configuration; do not change the
committed `pyproject.toml` or lockfile for a mirror.

The default profile installs Proto, Transport, Worker, and the Transformers
5.5.3 Frontend. Root optional profiles are:

```bash
uv sync --locked --extra fused
uv sync --locked --extra ggml
uv sync --locked --extra observability
uv sync --locked --extra vllm
```

`fused` and `ggml` add dependencies for experimental Worker backends,
`observability` adds Prometheus and OpenTelemetry dependencies, and `vllm`
installs the vLLM 0.25.1 Frontend. Run the last command before following the
vLLM section below.

## Deployment

### 1. Create the Controller configuration

Create `$EK_RUN/controller.yaml`:

```bash
cat > "$EK_RUN/controller.yaml" <<'EOF'
inference:
  hidden_dim: 2048
  intermediate_dim: 1408
  instance_name: deepseek-v2-lite-demo
  model_name: DeepSeek-V2-Lite-Chat

db:
  db_dsn: postgres://dev:dev@127.0.0.1:5432/dev
  max_conn_size: 32

weight:
  server:
    addr: http://127.0.0.1:6543
  cache:
    Fs:
      path: /tmp/expert-kit/cache

controller:
  listen: 0.0.0.0
  broadcast: 127.0.0.1
  ports:
    intra: 5001
    inter: 5002
  fault_detection:
    heartbeat_timeout_secs: 10
    node_active_threshold_secs: 60
    poller_interval_secs: 5
EOF
```

Workers connect to port 5001. The Frontend receives topology from port 5002.

### 2. Start PostgreSQL and initialize the database

```bash
docker compose -f dev/meta-db.docker-compose.yaml up -d
target/release/ek-cli --config "$EK_RUN/controller.yaml" db migrate
```

### 3. Start the Weight Server and register the model

Start the Weight Server in terminal 1 and leave it running:

```bash
target/release/ek-cli --config "$EK_RUN/controller.yaml" \
  weight-server --model "$DEEPSEEK_ROOT"
```

In terminal 2, register the model:

```bash
target/release/ek-cli --config "$EK_RUN/controller.yaml" \
  model upsert --name DeepSeek-V2-Lite-Chat
```

### 4. Assign the experts

Create `$EK_RUN/workers.yaml`:

```bash
cat > "$EK_RUN/workers.yaml" <<'EOF'
nodes:
  - id: deepseek-worker-a100
    address: http://127.0.0.1:52151
    channel: grpc
    device: cuda:0
EOF
```

Create the model instance and assign all experts to the Worker:

```bash
target/release/ek-cli --config "$EK_RUN/controller.yaml" \
  schedule static --inventory "$EK_RUN/workers.yaml"
```

The scheduler creates `deepseek-v2-lite-demo`. Worker and Frontend startup
resolve its database-generated numeric ID through the Controller.

### 5. Create the Worker configuration

Create `$EK_RUN/worker.yaml`:

```bash
cat > "$EK_RUN/worker.yaml" <<EOF
model:
  name: DeepSeek-V2-Lite-Chat
  weight_version: main
  num_layers: 27
  experts_per_layer: 64
  hidden_dim: 2048
  expert_intermediate_dim: 1408
  top_k: 6
  activation_dtype: bf16
  weight_dtype: bf16

worker:
  id: deepseek-worker-a100
  backend: torch
  device: cuda:0
  max_batch_tokens: 1024
  max_active_batches_per_device: 1
  device_memory_limit: 34GiB
  shutdown_grace_secs: 30

transport:
  type: grpc
  max_pending_batches_per_device: 1
  listen: 127.0.0.1:52151
  advertise: 127.0.0.1:52151

controller:
  endpoint: 127.0.0.1:5001
  heartbeat_interval_secs: 3
  heartbeat_timeout_secs: 10

weight_manager:
  max_concurrent_loads: 64
  disk_cache:
    path: $EK_WORKER_CACHE
    writeback: true
  peer:
    listen: 127.0.0.1:52152
    advertise: http://127.0.0.1:52152
  weight_server_endpoint: http://127.0.0.1:6543

logging:
  level: INFO
  format: console

observability:
  prometheus:
    enabled: false
  tracing:
    enabled: false
EOF
```

`device_memory_limit` is the total Worker GPU budget. Reduce it when other
processes use the same GPU. If the resulting expert capacity is below 1,664,
use more Worker GPUs instead.

One unquantized BF16 A10 is not sufficient for this placement. The 1,664 routed
experts require 28,789,702,656 bytes (26.81 GiB) before runtime reserve, while
an A10 exposes 22.06 GiB of physical memory. With a 20 GiB Worker budget, the
measured Worker had 17.84 GiB available for weights and reported capacity for
only 1,107 experts. An A100 Frontend plus one A10 Worker is therefore rejected
during placement admission before either gRPC or SHM executes.

NVIDIA MPS multiplexes CUDA processes but does not increase physical GPU
memory, so enabling it cannot make that topology fit. Use a Worker GPU with
enough capacity, multiple Workers with sharded assignments, or a future
supported quantized-expert backend. Do not increase `device_memory_limit`
beyond actual free memory.

### 6. Select gRPC or same-host SHM

The configuration above uses gRPC. To use the experimental same-host SHM data
path, replace only its `transport` block:

```yaml
transport:
  type: shm
  max_pending_batches_per_device: 1
  rpc_listen: 127.0.0.1:52151
  rpc_advertise: 127.0.0.1:52151
  shared_memory_dir: /dev/shm
```

Keep the inventory's legacy `channel: grpc` field as shown; it is not the
Frontend data-path selector. The running Worker registers the authoritative
Transport type through the v2 lifecycle API, and both Frontends select gRPC or
SHM from published topology. There is no Frontend Transport command-line flag
or environment variable.

SHM requires the Frontend and Worker to run on the same Host, in the same
`/dev/shm` mount namespace, and under the same Unix user. It removes protobuf
Tensor payloads and loopback Tensor copies, but CUDA data still stages through
pinned Host memory. It is not CUDA IPC or GPU Direct.

### 7. Start the Controller and Worker

Start the Controller in terminal 2 and leave it running:

```bash
target/release/ek-cli --config "$EK_RUN/controller.yaml" controller
```

Start the Worker in terminal 3:

```bash
CUDA_VISIBLE_DEVICES=0 EK_CONFIG="$EK_RUN/worker.yaml" \
  uv run --package expertkit-worker target/release/ek-cli worker
```

The Worker logs expert-loading progress. Wait until it reports that all 1,664
experts are ready before starting inference.

## Test with the Transformers 5.5.3 Frontend

In terminal 4, run a short benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run --package expertkit-torch ek-torch-benchmark \
  --model-path "$DEEPSEEK_ROOT" \
  --mode expertkit \
  --controller-endpoint 127.0.0.1:5002 \
  --batch-sizes 1 \
  --input-length 16 \
  --output-length 16 \
  --warmup-runs 1 \
  --runs 3 \
  --device cuda:0 \
  --dtype bfloat16 \
  --json-output "$EK_RUN/expertkit.json"
```

The command prints Prefill, Decode, end-to-end latency, and output throughput.
Its JSON output also records generated token IDs and decoded text.

For a native Transformers reference on the same GPU, stop the Worker to release
its expert weights, then run the same model, dtype, batch size, lengths, warmup,
and measured-run count with local experts:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run --package expertkit-torch ek-torch-benchmark \
  --model-path "$DEEPSEEK_ROOT" \
  --mode local \
  --batch-sizes 1 \
  --input-length 16 \
  --output-length 16 \
  --warmup-runs 1 \
  --runs 3 \
  --device cuda:0 \
  --dtype bfloat16 \
  --json-output "$EK_RUN/native.json"
```

Compare the `batches[].runs[].generated_token_ids` and `generated_text` fields
between the two files. Report native and Expert Kit latency and aggregate
throughput side by side; do not infer native prefill or decode values if the
native tool does not provide them.

## Test with the vLLM 0.25.1 Frontend

First install the root vLLM profile if it is not already active:

```bash
uv sync --locked --extra vllm
```

The following is a functional 16-input/16-output-token smoke test, not a
performance benchmark. A separate Frontend GPU is the simplest configuration
because vLLM reserves memory for model state, compilation, and its KV cache.
Set `EK_FRONTEND_GPU` to the Host GPU index that should run vLLM:

```bash
export EK_FRONTEND_GPU=1

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="$EK_FRONTEND_GPU" \
EK_ENABLE=1 \
EK_ADDR=127.0.0.1:5002 \
EK_CLIENT_TIMEOUT=6 \
uv run --package expertkit-vllm python - "$DEEPSEEK_ROOT" <<'PY'
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

model_path = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_path)
seed_ids = tokenizer.encode(
    "Expert Kit routed mixture of experts benchmark input",
    add_special_tokens=False,
)
input_ids = (seed_ids * ((16 + len(seed_ids) - 1) // len(seed_ids)))[:16]

llm = LLM(
    model=model_path,
    dtype="bfloat16",
    max_model_len=32,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=False,
    seed=0,
)
sampling = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
result = llm.generate(
    [TokensPrompt(prompt_token_ids=input_ids)],
    sampling,
    use_tqdm=False,
)[0].outputs[0]
assert len(input_ids) == 16
assert len(result.token_ids) == 16
print(list(result.token_ids))
print(result.text)
PY
```

`EK_ENABLE=1` enables the installed `vllm.general_plugins` entry point before
model construction. `EK_ADDR` selects Controller port 5002, and
`EK_CLIENT_TIMEOUT` is the positive per-call timeout in seconds. Transport
resolves the same Controller default instance used by the Worker. The plugin
does not have a gRPC/SHM setting; changing the Worker configuration and
published topology is sufficient.

The verified scope is deliberately narrower than the code paths:

- Transformers 5.5.3 has completed real DeepSeek-V2-Lite end-to-end runs over
  both gRPC and SHM.
- vLLM 0.25.1 has loaded the official checkpoint and completed a real
  16-input/16-output-token Expert Kit generation over gRPC.
- vLLM uses the same Transport client class, and SHM has shared Transport
  conformance coverage, but a real DeepSeek-vLLM SHM end-to-end run has not
  yet completed. Do not treat it as performance-qualified.

On a fresh database, the Controller can replay the initial expert topology in
several updates. If a Frontend reports `one or more experts have no ready
route`, wait for the complete ready topology before retrying. If replay remains
stuck, restart only the Controller, keep the Worker running, and rerun the
Frontend.

## Stop the deployment

Stop the Worker, Controller, and Weight Server with `Ctrl-C`, then stop
PostgreSQL:

```bash
docker compose -f dev/meta-db.docker-compose.yaml down
```
