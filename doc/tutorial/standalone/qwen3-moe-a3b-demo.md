# Run Qwen3-30B-A3B with the Python Worker

This guide exercises the Expert Kit Transformers 5.5.3 path with the BF16
`Qwen/Qwen3-30B-A3B` checkpoint. The model configuration has 48 routed layers,
128 experts per layer, hidden size 2048, expert intermediate size 768, and top-k
8. Those values come from the model's
[`config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json).

The final check sends one prompt and generates at most 20 new tokens. It is the
small Torch Frontend check used by the Worker migration.

## Before starting

You need:

- Linux hosts with Python 3.12, uv 0.11.30, Rust, Docker, and enough local
  storage for the checkpoint and Worker disk caches.
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
uv sync --locked
```

The command consumes the only committed `uv.lock` and creates the repository
root `.venv`. The committed workspace uses official PyPI. Configure a regional
mirror only in developer-local uv configuration; do not change the committed
`pyproject.toml` or lockfile for a mirror.

The default profile already installs Proto, Transport, Worker, and the
Transformers Frontend used by this guide. Optional root profiles are:

```bash
uv sync --locked --extra fused
uv sync --locked --extra ggml
uv sync --locked --extra observability
uv sync --locked --extra vllm
```

`fused` and `ggml` add dependencies for experimental Worker backends,
`observability` adds Prometheus and OpenTelemetry dependencies, and `vllm`
installs the opt-in vLLM 0.25.1 Frontend. None is required for the default
Torch path below.

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

## 4. Configure and start the Python Workers

Copy the Worker example once per device:

```bash
cp ek-worker/examples/qwen3-30b-a3b.torch.yaml /tmp/qwen-worker-0.yaml
```

For every copy, set:

- a unique `worker.id`;
- the process's `worker.device` and realistic `device_memory_limit`;
- unique gRPC and peer listen ports;
- advertised addresses reachable by the Frontend and other Workers;
- a unique, writable, direct-I/O-compatible absolute disk-cache path;
- Controller port 5001 and the Weight Server address.

The sample's 20 GiB device budget is only an example. Worker startup subtracts
fixed input/output buffers, Backend temporary memory, conversion memory, and
allocator headroom before calculating `max_experts`. The sum of available
Worker slots must cover all 6144 routed experts.

Start the Controller:

```bash
target/release/ek-cli --config /tmp/qwen-controller.yaml controller
```

Start one process per config in separate terminals. `uv run` exposes the root
environment to the Rust launcher, so it can find `ek-worker`:

```bash
uv run --package expertkit-worker \
  target/release/ek-cli --config /tmp/qwen-worker-0.yaml worker
```

Each Worker initially receives an empty target list. Wait until every process
logs `worker_registered` so the Controller can use its measured `max_experts`
instead of predeclared Worker capacity.

## 5. Rebalance experts onto active Workers

After all intended Workers have registered, run:

```bash
target/release/ek-cli --config /tmp/qwen-controller.yaml schedule rebalance
```

The Controller selects Workers with recent heartbeats for `qwen3-demo`, reads
the `max_experts` reported by each process, and rejects the operation unless
their summed capacity covers all 6144 routed experts. It then builds a stable
capacity-proportional assignment, commits it as one database transaction, and
immediately sends each running Worker its complete target list. Workers load
from their DRAM cache, disk cache, eligible peers, or the Weight Server in that
order. The Controller publishes an expert only after its Worker reports it
ready.

Do not start inference until every routed expert has at least one ready route.
If rebalance reports insufficient aggregate capacity, add Workers or increase a
budget only when the device truly has that free memory, wait for registration,
and rerun the command. Running it again with the same active Worker set produces
the same assignment. Run it again after intentionally adding or removing a
Worker.

## 6. Run the Qwen benchmark

Use Controller port 5002:

```bash
uv run --package expertkit-torch ek-torch-benchmark \
  --model-path "$QWEN_ROOT" \
  --mode expertkit \
  --controller-endpoint 127.0.0.1:5002 \
  --dataset-path /datasets/ShareGPT.json \
  --num-prompts 16 \
  --batch-sizes 1 \
  --output-length 20 \
  --warmup-runs 1
```

The command resolves the configured `qwen3-demo` instance, selects ShareGPT
prompts deterministically, then prints median prefill, decode, and complete
output throughput. The benchmark ignores EOS so every prompt executes 20
output-token steps.

The same path is exposed as an environment-gated test:

```bash
EK_QWEN_MODEL_PATH="$QWEN_ROOT" \
EK_QWEN_CONTROLLER_ENDPOINT=127.0.0.1:5002 \
EK_BENCHMARK_DATASET_PATH=/datasets/ShareGPT.json \
uv run --package expertkit-torch pytest \
  ek-integration/expertkit_torch/tests/test_deployment_benchmark.py -m qwen
```

## Current limits

- Cross-host computation uses gRPC and copies Tensor bytes through Host memory.
  The experimental same-host SHM path removes protobuf Tensor payloads but
  still stages CUDA transfers through pinned Host memory. RDMA, NCCL, and
  NVSHMEM are not available.
- Torch is the only Backend targeted for full migration qualification.
- GGML is experimental and CPU-only. The fused Backend is experimental and
  requires a compatible CUDA and Triton environment.
- The Controller does not relay computation as a fallback during topology
  changes. Frontend requests can receive retryable failures until a replacement
  topology is installed.
- This guide uses the Transformers Frontend; a full Qwen-vLLM deployment has
  not been qualified. Separately, vLLM 0.25.1 has completed a real
  DeepSeek-V2-Lite 16-input/16-output-token run over gRPC. vLLM uses the shared
  `expertkit-transport` client, and SHM has Transport conformance coverage, but
  a real vLLM DeepSeek model has not yet completed an SHM end-to-end run.
- All internal traffic is plaintext and unauthenticated. Public or untrusted
  network deployment is unsupported.
