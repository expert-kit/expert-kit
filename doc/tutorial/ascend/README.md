# Run Expert Kit on two Ascend 910C Hosts

This tutorial runs Qwen3-30B-A3B with the control and attention services on one
Host and 16 expert Workers on another. It covers the vLLM online benchmark and
the optional Host-native Torch ablation.

The configuration model and generated-file reference live in
[`dev/ascend/README.md`](../../../dev/ascend/README.md).

## Before you start

Use these placeholders throughout the tutorial:

| Placeholder | Meaning |
| --- | --- |
| `<PROJECT_ROOT>` | EK checkout on each Host |
| `<ATTENTION_HOST_IP>` | Routable address of the attention Host |
| `<EXPERT_HOST_IP>` | Routable address of the expert Host |
| `<MODEL_DIR>` | Host directory containing Qwen3-30B-A3B |
| `<DATASET_DIR>` | Host directory containing the ShareGPT file |
| `<RESULTS_DIR>` | Writable Host result directory inside the checkout |

Both Hosts need the same EK revision, Docker Compose v2 with Buildx, access to
the NPU devices, and permission to use Docker. Keep local configuration,
generated YAML, caches, and results inside `<PROJECT_ROOT>`.

If the Hosts require a Python mirror, prefix each package-resolving command.
The inline variable is inherited by that command and its child processes without
changing later commands in the shell:

```bash
UV_DEFAULT_INDEX=https://mirror.nju.edu.cn/pypi/web/simple \
  uv sync --locked
```

## 1. Prepare the checkpoint and dataset

On the attention Host, install the ModelScope CLI if it is unavailable, then
download the model:

```bash
python3 -m pip install \
  --index-url https://mirror.nju.edu.cn/pypi/web/simple \
  modelscope

modelscope download \
  --model Qwen/Qwen3-30B-A3B \
  --local_dir <MODEL_DIR>
```

Download ShareGPT on the attention Host and convert the JSONL file into the
JSON array expected by both benchmark frontends:

```bash
mkdir -p <DATASET_DIR>

modelscope download \
  --dataset AI-ModelScope/sharegpt_gpt4 \
  sharegpt_gpt4.jsonl \
  --local_dir <DATASET_DIR>

jq -s '.' \
  <DATASET_DIR>/sharegpt_gpt4.jsonl \
  > <DATASET_DIR>/sharegpt_gpt4.json
```

## 2. Create the Host-local configuration

From `<PROJECT_ROOT>`:

```bash
cp dev/ascend/configs/cluster.example.yaml dev/ascend/configs/cluster.yaml
cp dev/ascend/configs/experiment.example.yaml dev/ascend/configs/experiment.yaml
```

In `cluster.yaml`, set:

- both node addresses and their physical NPU lists;
- the control, attention, and expert image references;
- `paths.models.qwen3-30b-a3b` to `<MODEL_DIR>`;
- `paths.datasets.sharegpt` to `<DATASET_DIR>`;
- `paths.results` to `<RESULTS_DIR>`.

Use an absolute, project-local result path and create it before running a
container as the Host user:

```bash
mkdir -p <RESULTS_DIR>
```

Select ShareGPT in `experiment.yaml`:

```yaml
dataset:
  type: sharegpt
  name: ShareGPT
  path_ref: sharegpt
  mounted_path: /dataset/sharegpt
  file: sharegpt_gpt4.json
```

Remove `run.input_len`; file-backed datasets determine their prompt lengths
from the file. Keep the same `project_name`, inputs, and generated files on both
Hosts.

## 3. Build the images

The attention image installs local EK wheels. Build them before building that
image:

```bash
mkdir -p dist/attention-wheels

UV_DEFAULT_INDEX=https://mirror.nju.edu.cn/pypi/web/simple \
  uv build --wheel --package expertkit-proto \
    --out-dir dist/attention-wheels
UV_DEFAULT_INDEX=https://mirror.nju.edu.cn/pypi/web/simple \
  uv build --wheel --package expertkit-transport \
    --out-dir dist/attention-wheels
UV_DEFAULT_INDEX=https://mirror.nju.edu.cn/pypi/web/simple \
  uv build --wheel --package expertkit-vllm \
    --out-dir dist/attention-wheels
```

On the attention Host:

```bash
docker compose \
  -f dev/ascend/compose/compose.build.yaml \
  build control-image attention-image
```

On the expert Host:

```bash
docker compose \
  -f dev/ascend/compose/compose.build.yaml \
  build worker-image
```

## 4. Generate and validate the runtime files

Generate once and copy the complete `dev/ascend/generated/` directory to the
other Host together with the matching checkout:

```bash
UV_DEFAULT_INDEX=https://mirror.nju.edu.cn/pypi/web/simple \
  uv run dev/ascend/main.py generate
```

Validate the role bundle on each Host:

```bash
# Attention Host
dev/ascend/run-compose.sh attention config -q

# Expert Host
dev/ascend/run-compose.sh expert config -q
```

## 5. Start EK

Start the Controller on the attention Host. Compose also starts PostgreSQL,
migrations, model initialization, and the Weight Server:

```bash
dev/ascend/run-compose.sh attention up -d controller
dev/ascend/run-compose.sh attention ps
```

Start one Worker on the expert Host and inspect it before starting the rest:

```bash
dev/ascend/run-compose.sh expert up -d worker-00
dev/ascend/run-compose.sh expert logs --tail=100 worker-00

dev/ascend/run-compose.sh expert up -d
dev/ascend/run-compose.sh expert ps
```

Once the Workers have registered and loaded their assigned experts, rebalance
placement from the attention Host:

```bash
dev/ascend/run-compose.sh attention run --rm admin \
  ek-cli schedule rebalance
```

Start vLLM after the Controller and Workers are ready:

```bash
dev/ascend/run-compose.sh attention up -d attention
dev/ascend/run-compose.sh attention ps
```

Check the OpenAI-compatible endpoint:

```bash
curl -fsS http://<ATTENTION_HOST_IP>:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-30B-A3B",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

## 6. Run the vLLM online benchmark

The generated benchmark config targets the attention service over the Compose
network. Run the one-shot benchmark container with the Host UID/GID so result
files remain writable outside Docker:

```bash
dev/ascend/run-compose.sh attention run --rm \
  --user "$(id -u):$(id -g)" \
  benchmark
```

Inspect status or logs without changing the deployment:

```bash
dev/ascend/run-compose.sh attention ps
dev/ascend/run-compose.sh attention logs --tail=100 attention controller
dev/ascend/run-compose.sh expert logs --tail=100
```

## 7. Optional: run the Torch offline ablation

For ShareGPT, generation also writes `generated/torch-bench.yaml`. The Torch
frontend runs directly on the attention Host and connects to the same
Controller and Workers; it does not use the vLLM or benchmark containers.

Prepare the Host environment once:

```bash
UV_DEFAULT_INDEX=https://mirror.nju.edu.cn/pypi/web/simple \
  uv sync --package expertkit-torch --no-dev --extra npu
```

Run without another dependency sync:

```bash
uv run --no-sync --package expertkit-torch \
  ek-torch-benchmark \
  --config dev/ascend/generated/torch-bench.yaml \
  run
```

The vLLM benchmark measures online HTTP serving, while the Torch benchmark uses
fixed offline batches. Match the dataset, prompt count, output length, and
concurrency, but report the two execution modes explicitly.

## Version matrix

| Role | Python | PyTorch | torch-npu | CANN | vLLM / vLLM-Ascend |
| --- | ---: | ---: | ---: | ---: | --- |
| Worker image | `3.12` | `2.11.0` | `2.11.0rc1` | `9.0.1` base image | Not installed |
| Torch ablation | `>=3.12` | `2.11.0` | `2.11.0rc1` | Host installation | Not installed |
| Attention image | Image-owned | Image-owned Torch 2.10 profile | Image-owned | `9.0.1` profile | vLLM `0.25.1`; selected vLLM-Ascend `main-a3` snapshot |

The exact attention image is pinned by digest in
[`compose.build.yaml`](../../../dev/ascend/compose/compose.build.yaml). Do not
synchronize the root uv environment over that image's accelerator stack.

The Worker and Torch ablation currently pair Torch 2.11 with torch-npu
2.11.0rc1 against CANN 9.0.1. Treat that combination as an EK qualification
profile rather than a general compatibility claim.

## Stop the deployment

Stop containers without deleting named volumes:

```bash
dev/ascend/run-compose.sh attention down
dev/ascend/run-compose.sh expert down
```
