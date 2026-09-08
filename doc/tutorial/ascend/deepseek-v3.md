# Qualify DeepSeek-V3 on Ascend 910C

This tutorial applies the same generated 8A32E deployment used by the
[Qwen3-30B-A3B guide](qwen3-30b-a3b.md) to DeepSeek-V3: eight attention ranks
on `node-a` and two sixteen-Worker expert pools on `node-b` and `node-c`.

## Before the start

Read the [common deployment workflow](README.md#common-deployment-workflow)
before using this tutorial. It owns the shared Host preparation, generation,
image build, startup, benchmark, and shutdown steps.

DeepSeek publishes the official V3 checkpoint in FP8 and provides an official
FP8-to-BF16 conversion script. The current Ascend Worker path supports
unquantized FP16/BF16 expert weights, so this guide uses a converted BF16
checkpoint. Do not point `weight_dtype: bf16` at the original FP8 files; YAML
does not convert weights.

DeepSeek-V3.2 is outside this qualification. Its `model_type` is
`deepseek_v32`, and its checkpoint and routed-MoE integration require separate
qualification. DeepSeek Sparse Attention remains inside the vLLM-Ascend
frontend; EK does not need to implement the attention algorithm.

Sources:

- [Official DeepSeek-V3 repository and BF16 conversion](https://github.com/deepseek-ai/DeepSeek-V3)
- [Official DeepSeek-V3 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [DeepSeek-V3 on ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3)
- [ModelScope BF16 mirror](https://www.modelscope.cn/models/unsloth/DeepSeek-V3-bf16)
- [vLLM-Ascend supported-model matrix](https://docs.vllm.ai/projects/ascend/en/main/user_guide/support_matrix/supported_models.html)

vLLM-Ascend's DeepSeek support does not by itself qualify EK. EK replaces the
routed-expert execution path, so the attention plugin and remote Worker must
both support the selected checkpoint format.

## 1. Prepare a BF16 checkpoint

For servers without Hugging Face access, ModelScope provides two paths.

Download the official FP8 checkpoint and convert it with DeepSeek's
`inference/fp8_cast_bf16.py` on a compatible conversion machine:

```bash
modelscope download \
  --model deepseek-ai/DeepSeek-V3 \
  --local_dir <DEEPSEEK_V3_FP8_DIR>

python fp8_cast_bf16.py \
  --input-fp8-hf-path <DEEPSEEK_V3_FP8_DIR> \
  --output-bf16-hf-path <DEEPSEEK_V3_BF16_DIR>
```

The official conversion script uses CUDA/Triton, so it is not directly usable
on an Ascend-only Host without being ported. It converts the tensors and writes
a new weight index, but it does not populate the complete tokenizer/model
metadata directory. Copy the non-weight files from the official checkpoint and
remove the stale `quantization_config` from the converted `config.json`.

Alternatively, download the third-party BF16 mirror from ModelScope after
verifying its provenance and file checksums:

```bash
modelscope download \
  --model unsloth/DeepSeek-V3-bf16 \
  --local_dir ~/models/DeepSeek-V3-bf16
```

For the remaining configuration, `<DEEPSEEK_V3_BF16_DIR>` means the absolute
expansion of `~/models/DeepSeek-V3-bf16`; do not put an unexpanded `~` in
`cluster.yaml`.

Allow roughly 1.37 TB for the BF16 checkpoint. Confirm that the converted
checkpoint no longer declares FP8 quantization and that its expert tensors are
BF16:

```bash
jq '{
  model_type,
  num_hidden_layers,
  first_k_dense_replace,
  n_routed_experts,
  hidden_size,
  moe_intermediate_size,
  num_experts_per_tok,
  quantization_config,
  torch_dtype
}' <DEEPSEEK_V3_BF16_DIR>/config.json
```

The expected model geometry is:

| Field | Value |
| --- | ---: |
| `model_type` | `deepseek_v3` |
| `num_hidden_layers` | `61` |
| `first_k_dense_replace` | `3` |
| `n_routed_experts` | `256` |
| `hidden_size` | `7168` |
| `moe_intermediate_size` | `2048` |
| `num_experts_per_tok` | `8` |

The first three layers are dense, so routed expert IDs occur at the original
layer IDs 3 through 60. EK keeps `num_layers: 61`; it must not renumber the 58
MoE layers to 0 through 57.

## 2. Configure the cluster

Create the inputs through the
[shared Host-local input step](README.md#1-create-the-host-local-inputs)
using `deepseek-v3` as `<MODEL_CONFIG>`, then replace its Host-local addresses
and paths. This pair already carries the 8A32E placement and `52GiB` Worker
admission budget required by the BF16 checkpoint.

In `cluster.yaml`, replace the example BF16 checkpoint path with the absolute
path produced by the ModelScope download:

```yaml
paths:
  models:
    deepseek-v3-bf16: <DEEPSEEK_V3_BF16_DIR>
```

The BF16 routed experts require about 1.2 TiB before runtime overhead. Evenly
distributed over 32 Workers, the raw expert matrices use about 38 GiB per NPU.
Start with a Worker admission budget such as `52GiB` on 64 GiB devices, then
verify actual free HBM and runtime overhead on every Host:

```yaml
expert:
  runtime:
    device_memory_limit: 52GiB
```

## 3. Configure the experiment

Use the original model layer IDs and routed-expert geometry from `config.json`:

```yaml
model:
  name: DeepSeek-V3-BF16
  path_ref: deepseek-v3-bf16
  weight_version: main
  num_layers: 61
  experts_per_layer: 256
  hidden_dim: 7168
  intermediate_dim: 2048
  topk: 8
  weight_dtype: bf16
  activation_dtype: bf16

dataset:
  type: sharegpt
  name: ShareGPT
  path_ref: sharegpt
  mounted_path: /dataset/sharegpt
  file: sharegpt_gpt4.json

serve:
  gpu_memory_utilization: 0.8
  max_model_len: 4096

run:
  num_prompts: 16
  max_concurrency: 1
  output_len: 128
  num_warmups: 1
  ignore_eos: true
  temperature: 0
  save_result: true
```

The first qualification deliberately uses a short context and concurrency one.
Increase concurrency, prompt count, and context length only after one routed
forward completes on all eight attention ranks.

## 4. Deploy and qualify

Continue with [generate and validate](README.md#2-generate-and-validate).
At its one-Worker gate, do not continue if either Worker rejects the weight
dtype, cannot parse the expert blob, or exceeds its memory budget.

Use `DeepSeek-V3-BF16` as `<SERVED_MODEL_NAME>` in the shared verification
request. Success requires more than a healthy HTTP response: confirm that the
Controller sees all Workers, the Weight Server serves original layer IDs 3
through 60, and Worker logs show a real expert load and computation.

## 5. Benchmark gates

After one routed request passes, run the benchmark from the shared workflow.
Scale the workload in these gates:

1. 16 prompts at concurrency 1;
2. 200 prompts at concurrency 8;
3. 200 prompts at concurrency 32;
4. the target input and output lengths.

Record failures, throughput, TTFT, TPOT, Worker load time, and peak HBM at each
gate. DeepSeek-V3 has not completed this EK vLLM-Ascend path yet, so the first
successful routed forward is a qualification result rather than a performance
baseline.

## Current limits

The Weight Server already recognizes `deepseek_v3`, discovers routed layers 3
through 60, and maps standard DeepSeek expert tensor names. The remaining risks
for the BF16 path are attention-plugin compatibility, the per-rank attention
memory footprint, and full 8A32E hardware qualification.

The original FP8 checkpoint additionally requires Worker-side quantization
support:

- preserve and validate FP8 expert matrices and scales when loading a blob;
- quantize BF16 activations for an FP8 expert kernel;
- execute gate/up/down projections with a quantization-aware backend;
- return BF16 activations through the existing transport contract.

Adding `--quantization ascend` only to vLLM does not solve this boundary because
the routed experts execute in EK Workers, outside vLLM.
