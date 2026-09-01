# Deploy Qwen3-30B-A3B on Ascend 910C

This tutorial runs Qwen3-30B-A3B as an 8A32E deployment: eight vLLM data
parallel ranks on one attention/control Host and two pools of sixteen expert
Workers on two expert Hosts. It covers the vLLM online benchmark and the
optional Host-native Torch ablation.

## Before the start

Read the [common deployment workflow](README.md#common-deployment-workflow)
before using this tutorial. It owns the shared Host preparation, generation,
image build, startup, benchmark, and shutdown steps. This tutorial adds these
model-specific placeholders:

| Placeholder | Meaning |
| --- | --- |
| `<MODEL_DIR>` | Qwen3-30B-A3B checkpoint directory on the attention Host |
| `<DATASET_DIR>` | ShareGPT directory on the attention Host |

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

Download ShareGPT and convert the JSONL file into the JSON array expected by
both benchmark frontends:

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

Create the inputs through the
[shared Host-local input step](README.md#1-create-the-host-local-inputs).
Keep the example's 8A32E placement and make these model-specific changes.

In `cluster.yaml`:

- set `paths.models.qwen3-30b-a3b` to `<MODEL_DIR>`;
- set `paths.datasets.sharegpt` to `<DATASET_DIR>`;
- set `paths.results` to `<RESULTS_DIR>`.

Keep the Qwen model block in `experiment.yaml` and select ShareGPT:

```yaml
model:
  name: Qwen3-30B-A3B
  path_ref: qwen3-30b-a3b
  weight_version: main
  num_layers: 48
  experts_per_layer: 128
  hidden_dim: 2048
  intermediate_dim: 768
  topk: 8
  weight_dtype: bf16
  activation_dtype: bf16

dataset:
  type: sharegpt
  name: ShareGPT
  path_ref: sharegpt
  mounted_path: /dataset/sharegpt
  file: sharegpt_gpt4.json
```

Remove `run.input_len`; file-backed datasets determine prompt lengths from the
file.

## 3. Deploy and benchmark

Continue with [generate and validate](README.md#2-generate-and-validate),
then complete the remaining shared build, startup, request, benchmark, and
shutdown steps. Use `Qwen3-30B-A3B` as `<SERVED_MODEL_NAME>` in the verification
request.

This BF16 8A32E path has completed both the vLLM online benchmark and the
Host-native Torch ablation. Keep the dataset, prompt count, output length, and
concurrency aligned when comparing them, but report the online and offline
execution modes separately.
