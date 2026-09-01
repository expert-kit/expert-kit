# Ascend deployment tutorials

These tutorials use the Host-local configuration generator in
[`dev/ascend`](../../../dev/ascend/README.md). The deployment lifecycle is the
same for each model; checkpoint format, model geometry, memory budget, and
qualification status differ.

| Model | Status | Guide |
| --- | --- | --- |
| Qwen3-30B-A3B | Qualified with the current BF16 Worker and vLLM-Ascend path | [Deploy Qwen3-30B-A3B](qwen3-30b-a3b.md) |
| DeepSeek-V3 | BF16 qualification path; vLLM integration still needs an end-to-end hardware run | [Qualify DeepSeek-V3](deepseek-v3.md) |

Both guides keep authored configuration, generated YAML, caches, and results
inside the EK checkout under the operator's home directory. Container paths
such as `/etc/expert-kit` remain container-local mount targets.

## Common deployment workflow

Complete the selected model tutorial's checkpoint and configuration steps,
then use this workflow to generate and run the deployment.

### Before you start

Use the same EK revision and absolute `<PROJECT_ROOT>` on every Host. Every
Host needs Docker Compose v2 with Buildx, access to its assigned NPU devices,
and permission to use Docker. Keep Host-local configuration, generated YAML,
caches, and results inside `<PROJECT_ROOT>`.

The shared commands use these placeholders:

| Placeholder | Meaning |
| --- | --- |
| `<PROJECT_ROOT>` | EK checkout at the same absolute path on every Host |
| `<RESULTS_DIR>` | Writable result directory inside `<PROJECT_ROOT>` |
| `<ATTENTION_HOST_IP>` | Routable address of the attention Host |
| `<SERVED_MODEL_NAME>` | Exact `model.name` value from `experiment.yaml` |

The examples use this placement:

| Logical node | Services |
| --- | --- |
| `node-a` | PostgreSQL, migrations, Controller, Weight Server, attention frontend, and benchmark |
| `node-b` | `worker-pool1` |
| `node-c` | `worker-pool2` |

Replace the example node names and pool IDs when `cluster.yaml` uses different
values. The model checkpoint is mounted only on the attention Host. Expert
Hosts receive assigned expert blobs from the Weight Server.

If a Host requires a Python mirror, prefix commands that resolve packages:

```bash
UV_DEFAULT_INDEX=https://mirror.nju.edu.cn/pypi/web/simple \
  uv sync
```

### 1. Create the Host-local inputs

Create the ignored Host-local inputs once:

```bash
cp dev/ascend/configs/cluster.example.yaml dev/ascend/configs/cluster.yaml
cp dev/ascend/configs/experiment.example.yaml dev/ascend/configs/experiment.yaml
```

Replace the RFC 5737 documentation addresses and `/home/<USER>` paths, then
set the model, dataset, and runtime fields described by the selected tutorial.
Create the result directory as the Host user before a benchmark container
writes to it:

```bash
mkdir -p <RESULTS_DIR>
```

Copy the same Host-local inputs to every participating Host. Generate the same
output on every Host, or generate it once and distribute the complete
`dev/ascend/generated/` directory.

### 2. Generate and validate

Generate from the repository root:

```bash
uv run python dev/ascend/cli.py generate
```

Explicit paths are available for automation:

```bash
uv run python dev/ascend/cli.py generate \
  --cluster dev/ascend/configs/cluster.yaml \
  --experiment dev/ascend/configs/experiment.yaml \
  --output dev/ascend/generated
```

The default output directory is replaced on every generation. A custom output
directory must be nonexistent, empty, or contain the generated
`.expert-kit-generated` marker.

Validate the image definition, attention bundle, and every expert pool before
deployment:

```bash
dev/ascend/run-compose.sh image config -q
dev/ascend/run-compose.sh attention config -q
dev/ascend/run-compose.sh expert worker-pool1 config -q
dev/ascend/run-compose.sh expert worker-pool2 config -q
```

### 3. Build the images

The attention image installs local EK wheels. Build them before building that
image:

```bash
mkdir -p dist/attention-wheels

uv build --wheel --package expertkit-proto \
  --out-dir dist/attention-wheels
uv build --wheel --package expertkit-transport \
  --out-dir dist/attention-wheels
uv build --wheel --package expertkit-vllm \
  --out-dir dist/attention-wheels
```

Build only the targets required by the current Host:

```bash
# node-a
dev/ascend/run-compose.sh image build control-image attention-image

# node-b and node-c
dev/ascend/run-compose.sh image build worker-image
```

`run-compose.sh image` reads `generated/compose.build.yaml`; the image and base
image references therefore come from `cluster.yaml`.

### 4. Start the deployment

On `node-a`, start the control stack. The `controller` dependency chain also
starts PostgreSQL, migrations, model initialization, and the Weight Server:

```bash
dev/ascend/run-compose.sh attention up -d controller
dev/ascend/run-compose.sh attention ps
```

Start one Worker on each expert Host and inspect its logs before scaling out:

```bash
# node-b
dev/ascend/run-compose.sh expert worker-pool1 up -d worker-00
dev/ascend/run-compose.sh expert worker-pool1 logs --tail=100 worker-00

# node-c
dev/ascend/run-compose.sh expert worker-pool2 up -d worker-00
dev/ascend/run-compose.sh expert worker-pool2 logs --tail=100 worker-00
```

If both Workers pass their model-specific checks, start the complete pools:

```bash
# node-b
dev/ascend/run-compose.sh expert worker-pool1 up -d

# node-c
dev/ascend/run-compose.sh expert worker-pool2 up -d
```

After every Worker has registered, rebalance placement from `node-a` and start
the attention frontend:

```bash
dev/ascend/run-compose.sh attention run --rm admin \
  ek-cli schedule rebalance

dev/ascend/run-compose.sh attention up -d attention
dev/ascend/run-compose.sh attention ps
```

Compose `depends_on` supplies startup ordering inside one bundle. It does not
restart an already-running dependency unless its configuration or container
state requires recreation.

### 5. Verify one request

Use the model name from `experiment.yaml` and the attention address and port
from `cluster.yaml`:

```bash
curl -fsS http://<ATTENTION_HOST_IP>:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<SERVED_MODEL_NAME>",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "temperature": 0,
    "max_tokens": 1
  }'
```

A healthy HTTP response is not sufficient for first-time model qualification.
Confirm that every Worker registered, the Weight Server served routed-expert
blobs, and Worker logs show expert loading and computation.

### 6. Run the benchmark

Run the one-shot vLLM benchmark container with the Host UID/GID so result
files remain writable outside Docker:

```bash
dev/ascend/run-compose.sh attention run --rm \
  --user "$(id -u):$(id -g)" \
  benchmark
```

Inspect logs without changing the deployment:

```bash
dev/ascend/run-compose.sh attention logs --tail=100 attention controller
dev/ascend/run-compose.sh expert worker-pool1 logs --tail=100
dev/ascend/run-compose.sh expert worker-pool2 logs --tail=100
```

For a ShareGPT experiment, generation also writes `torch-bench.yaml`. The
optional Torch frontend runs directly on the attention Host and connects to
the same Controller and Workers:

```bash
uv sync --package expertkit-torch --no-dev --extra npu

uv run --no-sync --package expertkit-torch \
  ek-torch-benchmark \
  --config dev/ascend/generated/torch-bench.yaml \
  run
```

The vLLM benchmark measures online HTTP serving. The Torch benchmark uses
fixed offline batches; report the two execution modes separately.

### 7. Stop the deployment

Stop containers without deleting named volumes:

```bash
# node-a
dev/ascend/run-compose.sh attention down

# node-b
dev/ascend/run-compose.sh expert worker-pool1 down

# node-c
dev/ascend/run-compose.sh expert worker-pool2 down
```

### Runtime version boundaries

| Role | Python | PyTorch | torch-npu | CANN | vLLM / vLLM-Ascend |
| --- | ---: | ---: | ---: | ---: | --- |
| Worker image | `3.12` | `2.11.0` | `2.11.0rc1` | `9.0.1` base image | Not installed |
| Torch ablation | `>=3.12` | `2.11.0` | `2.11.0rc1` | Host installation | Not installed |
| Attention image | Image-owned | Image-owned Torch 2.10 profile | Image-owned | `9.0.1` profile | vLLM `0.25.1`; selected vLLM-Ascend `main-a3` snapshot |

Do not synchronize the root uv environment over the attention image's
accelerator stack.
