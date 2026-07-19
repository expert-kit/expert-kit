# Run Qwen3-30B-A3B with the Python Worker

This guide exercises the current Expert Kit path with the BF16
`Qwen/Qwen3-30B-A3B` checkpoint. The model configuration has 48 routed layers,
128 experts per layer, hidden size 2048, expert intermediate size 768, and top-k
8. Those values come from the model's
[`config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json).

The final check sends one prompt and generates at most 20 new tokens. It is the
small Torch Frontend check used by the Worker migration.

## Before starting

You need:

- Linux hosts with Python 3.12, uv, Rust, Docker, and enough local storage for
  the checkpoint and Worker disk caches.
- One Frontend GPU for attention, routing, and non-expert model weights.
- Enough Worker GPUs to hold every routed expert. The three BF16 FFN matrices
  contain 54 GiB of expert data before allocator and runtime overhead. Each
  Worker reports its actual expert capacity to the Controller.
- A filesystem supporting Linux direct I/O for each Worker disk-cache path.
- A trusted isolated network. The MVP uses plaintext gRPC and HTTP without TLS
  or application authentication.

One Worker process serves one device. The example below uses four Worker
processes on one host only to make the addresses concrete. Adjust the number of
Workers, device budgets, hosts, and ports for the available hardware. A
Frontend sharing one of those GPUs must leave enough memory outside the
Worker's configured budget.

## 1. Build the services and Python environments

Run from the repository root:

```bash
cargo build --release --bin ek-cli
uv sync --project ek-worker --locked
uv sync --project ek-integration/expertkit_torch --locked
```

Download the official BF16 checkpoint and set its absolute path:

```bash
export QWEN_ROOT=/models/Qwen3-30B-A3B
```

The last path component, `Qwen3-30B-A3B`, is the model name used by the Weight
Server, Controller config, and every Worker config.

## 2. Create the Controller config

Save the following as `/tmp/qwen-controller.yaml`:

```yaml
inference:
  hidden_dim: 2048
  intermediate_dim: 768
  instance_name: qwen3-demo
  model_name: Qwen3-30B-A3B

db:
  db_dsn: postgres://dev:dev@127.0.0.1:5432/dev
  max_conn_size: 32

weight:
  server:
    addr: http://127.0.0.1:6543
  cache:
    Fs:
      path: /var/cache/expert-kit/weight-server

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
```

Port 5001 carries Worker registration, heartbeat, and weight control. Port 5002
publishes Frontend topology. They are separate endpoints.

## 3. Start metadata and the Weight Server

```bash
docker compose -f dev/meta-db.docker-compose.yaml up -d
target/release/ek-cli --config /tmp/qwen-controller.yaml db migrate
target/release/ek-cli --config /tmp/qwen-controller.yaml \
  weight-server --model "$QWEN_ROOT"
```

Keep the Weight Server running. In another terminal, register the model:

```bash
target/release/ek-cli --config /tmp/qwen-controller.yaml \
  model upsert --name Qwen3-30B-A3B
```

## 4. Assign experts

Create `/tmp/qwen-workers.yaml` with one entry for every Worker process. The IDs
must match `worker.id` in the Python Worker files:

```yaml
nodes:
  - id: qwen-worker-0
    address: http://127.0.0.1:51051
    channel: grpc
    device: cuda:0
  - id: qwen-worker-1
    address: http://127.0.0.1:51151
    channel: grpc
    device: cuda:1
  - id: qwen-worker-2
    address: http://127.0.0.1:51251
    channel: grpc
    device: cuda:2
  - id: qwen-worker-3
    address: http://127.0.0.1:51351
    channel: grpc
    device: cuda:3
```

Create the model instance and static assignments:

```bash
target/release/ek-cli --config /tmp/qwen-controller.yaml \
  schedule static --inventory /tmp/qwen-workers.yaml
```

Read the numeric instance ID created by that command:

```bash
docker compose -f dev/meta-db.docker-compose.yaml exec -T pg \
  psql -U dev -d dev -Atc "SELECT id FROM instance WHERE name = 'qwen3-demo';"
```

Use that value for `model.instance_id` in every Worker file and for the
Frontend check.

## 5. Configure and start the Python Workers

Copy the Worker example once per device:

```bash
cp ek-worker/examples/qwen3-30b-a3b.torch.yaml /tmp/qwen-worker-0.yaml
```

For every copy, set:

- the same numeric `model.instance_id`;
- a unique `worker.id` matching the inventory;
- the process's `worker.device` and realistic `device_memory_limit`;
- unique gRPC and peer listen ports;
- advertised addresses reachable by the Frontend and other Workers;
- a unique, writable, direct-I/O-compatible absolute disk-cache path;
- Controller port 5001 and the Weight Server address.

The sample's 20 GiB device budget is only an example. Worker startup subtracts
fixed input/output buffers, Backend temporary memory, conversion memory, and
allocator headroom before calculating `max_experts`. The sum of available
Worker slots must cover all 6144 routed experts, and no Worker's static
assignment may exceed its reported slots.

Start the Controller:

```bash
target/release/ek-cli --config /tmp/qwen-controller.yaml controller
```

Activate the Worker environment, then start one process per config in separate
terminals:

```bash
source ek-worker/.venv/bin/activate
target/release/ek-cli --config /tmp/qwen-worker-0.yaml worker
```

Each Worker registers, receives its complete target list, and loads from its
DRAM cache, disk cache, eligible peers, or the Weight Server in that order. A
remote fetch is written to the disk cache by default. The Controller publishes
an expert only after the Worker reports it ready.

Do not start inference until every routed expert has at least one ready route.
If a Worker rejects its assignment because `max_experts` is too small, add
capacity and run the assignment again instead of increasing the configured
budget past actual free device memory.

## 6. Run the Qwen benchmark

In the Torch Frontend environment, use Controller port 5002:

```bash
source ek-integration/expertkit_torch/.venv/bin/activate
ek-torch-benchmark \
  --model-path "$QWEN_ROOT" \
  --mode expertkit \
  --controller-endpoint 127.0.0.1:5002 \
  --instance-id 1 \
  --batch-sizes 1 \
  --input-length 128 \
  --output-length 20 \
  --warmup-runs 1 \
  --runs 5
```

Replace `1` with the instance ID queried above. The command performs exact
fixed-length prefill and decode work, then prints median prefill, decode, and
complete output throughput. The benchmark ignores EOS so every measured run
executes 20 output-token steps.

The same path is exposed as an environment-gated test:

```bash
cd ek-integration/expertkit_torch
EK_QWEN_MODEL_PATH="$QWEN_ROOT" \
EK_QWEN_CONTROLLER_ENDPOINT=127.0.0.1:5002 \
EK_QWEN_INSTANCE_ID=1 \
uv run pytest tests/test_deployment_benchmark.py -m qwen
```

## Current limits

- Computation Transport is gRPC-only and copies Tensor bytes through Host
  memory. There is no current SHM, RDMA, NCCL, or NVSHMEM path.
- Torch is the only Backend targeted for full migration qualification.
- GGML is experimental and CPU-only. The fused Backend is experimental and
  requires a compatible CUDA and Triton environment.
- The Controller does not relay computation as a fallback during topology
  changes. Frontend requests can receive retryable failures until a replacement
  topology is installed.
- vLLM runtime qualification is deferred. This guide uses the Torch Frontend.
- All internal traffic is plaintext and unauthenticated. Public or untrusted
  network deployment is unsupported.
