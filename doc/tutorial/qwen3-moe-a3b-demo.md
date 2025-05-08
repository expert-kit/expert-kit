# Run Deepseek-Tiny model in Expert-Kit

[`Qwen3-30B-A3B`](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B/files) is a MoE model published by the Qwen team. This tutorial will show how to run the `qwen3-moe` model in Expert-Kit.

## Step by Step Guide

1. Download the `qwen3-moe` model weights from [hf](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B/files) and adding ek address to the config file.

```bash

git clone https://huggingface.co/Qwen/Qwen3-30B-A3B

export QWEN3_ROOT=$(realpath ./Qwen3-30B-A3B)
cd  "$QWEN3_ROOT"
sed -i 's/{/{\n"ek_addr":"localhost:5002",\n/' config.json
```

2. Convert the model weights to expert-kit format and prepare the cluster meta.

```bash
export CONVERTED_ROOT=$(realpath ./data/ek-qwen3)
mkdir -p $CONVERTED_ROOT

python py/expert_split.py \
    --model_dir  "$QWEN3_ROOT"\
    --fs_path "$CONVERTED_ROOT" \
    --model_idx_file model.safetensors.index.json \
    --check_remote  \
    --upload_missing

# register the model weights to the meta server
python py/create_model.py \
    --model_name ds_qwen3 \
    --fs_path "$CONVERTED_ROOT"

# create the expert distribution config based on the static configs.
python py/schedule_static.py \
  --fs_path "$CONVERTED_ROOT"  \
  --instance_name qwen3_moe_30b_local_test \
  --inventory ./dev/local.inventory.yaml \
  --model_name ds_qwen3
```

3. Create config file

```bash
cat > /etc/expert-kit/config.yaml <<EOF
storage_provider: fs
storage_fs_path: $CONVERTED_ROOT
db_dsn: postgres://dev:dev@localhost:5432/dev
hidden_dim:  2048
intermediate_dim: 768
instance_name: qwen3_moe_30b_local_test
EOF

```

4. run the frontend controller and backend worker

```bash
# lib torch will be dynamically loaded by the expert-kit, so you need to set the environment variable to point to the libtorch.so
# for mac 
export DYLD_FALLBACK_LIBRARY_PATH=<LIB_TORCH_PATH>
# for linux: TODO

# run the frontend controller
cargo run  --release --bin controller

# create a new terminal session, then run worker
EK_HOSTNAME=local-dev cargo run  --release --bin worker
```

5. run a simple test

```bash

cd ek-integration/expertkit_torch/

python3 -m expertkit_torch.models.qwen3_moe --model_path=/Users/liuhancheng/Project/qwen3 --enable_ek
```


expected output:

```plain

thinking content: 
content: <think>
Okay, the user is asking about what an MoE model is. I need to explain this clearly. Let me start by recalling that MoE stands for Mixture of Experts. I remember it's a type of neural network architecture where different...
```
