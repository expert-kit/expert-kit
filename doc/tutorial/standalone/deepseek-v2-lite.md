# Deploying DeepSeek-V2-Lite with Expert Kit

## Overview

This guide deploys `deepseek-ai/DeepSeek-V2-Lite-Chat` with the current Python
Worker and Torch Frontend:

```text
Torch Frontend -> Transport -> Python Worker -> Torch backend
                         Controller
                             |
                         PostgreSQL
```

The example uses one Worker on one NVIDIA A100 40 GB. The Worker loads all
1,664 routed experts. The Frontend and Worker use the same GPU and communicate
over gRPC.

## Requirements

- Linux, Rust, Cargo, Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker
  Compose.
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

Read the instance ID created by the scheduler:

```bash
docker compose -f dev/meta-db.docker-compose.yaml exec -T pg \
  psql -U dev -d dev -Atc \
  "SELECT id FROM instance WHERE name = 'deepseek-v2-lite-demo';"
```

The rest of this guide calls this value `<INSTANCE_ID>`.

### 5. Create the Worker configuration

Create `$EK_RUN/worker.yaml`. Replace `<INSTANCE_ID>` with the value from the
previous command:

```bash
cat > "$EK_RUN/worker.yaml" <<EOF
model:
  instance_id: <INSTANCE_ID>
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

### 6. Start the Controller and Worker

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

## Testing the deployment

In terminal 4, run a short benchmark. Replace `<INSTANCE_ID>` with the same
instance ID used in the Worker configuration:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run --package expertkit-torch ek-torch-benchmark \
  --model-path "$DEEPSEEK_ROOT" \
  --mode expertkit \
  --controller-endpoint 127.0.0.1:5002 \
  --instance-id <INSTANCE_ID> \
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

On a fresh database, the current branch can replay the initial expert topology
in several updates. If the first benchmark reports `one or more experts have no
ready route`, restart only the Controller, keep the Worker running, and rerun
the benchmark.

## Stop the deployment

Stop the Worker, Controller, and Weight Server with `Ctrl-C`, then stop
PostgreSQL:

```bash
docker compose -f dev/meta-db.docker-compose.yaml down
```
