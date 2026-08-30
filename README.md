# Expert Kit: A Distributed, Expert-Centric Framework for MoE LLM Inference

> [!CAUTION]
> Early Work-in-Progress. This project is currently a proof-of-concept demo and is under active development. It is not intended for production use and may contain significant bugs, security vulnerabilities, and unexpected behavior. We appreciate community feedback and contributions as we continue to build and refine this project.

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/expert-kit/expert-kit)
[![project chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://expert-kit.zulipchat.com/)

## About

![](./doc/assets/logo-lr-bg.svg)

**Expert Kit (EK)** is a high-performance framework for scalable MoE (Mixture of Experts) LLM inference. The vision of EK is to provide an efficient foundation of Expert Parallelism (EP) on heterogeneous hardware (e.g., CPU and GPU) over commodity networks (e.g. PCIe, TCP, RDMA), thereby enabling easy deployment and fine-grained expert-level scaling.

EK features Expert-Attention (E/A) separation architecture, enabling MoE LLMs to be deployed efficiently in a distributed environment composed of _x_ CPUs and _y_ GPUs.
The motivation behind the E/A separation lies in our observation that, in modern MoE LLMs, expert parameters account for the vast majority of the model size (e.g., over 90% in DeepSeek-V3).
By decoupling expert modules and deploying them across distributed GPUs and CPUs, EK leverages the high bandwidth and large capacity of distributed memory and storage systems.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./doc/assets/arch-illustration-dark.svg">
  <img alt="arch-illustration-light" src="./doc/assets/arch-illustration.svg">
</picture>

https://github.com/user-attachments/assets/9f1f5b23-28fe-44cf-b592-2f6ad0ad4dad

## Quick Start

Here are some tutorials to help you quickly start with Expert Kit.

1. [Qwen3-30B-A3B](./doc/tutorial/standalone/qwen3-moe-a3b-demo.md): A demo for running the Qwen3-30B-A3B model with Expert Kit, showcasing the framework's capabilities in handling real-world MoE models.
2. [DeepSeek-V2-Lite](./doc/tutorial/standalone/deepseek-v2-lite.md): A complete Python Worker and Torch Frontend deployment using the official DeepSeek-V2-Lite checkpoint.
3. [Two-Host Ascend 910C](./doc/tutorial/ascend/README.md): Deploy the Controller, vLLM attention frontend, and expert Workers, then run online and offline benchmarks.

## Python Workspace

Expert Kit uses one uv workspace, one root `uv.lock`, and one root `.venv` for
Proto, Transport, Worker, and Frontend development. Install the default
environment from the repository root:

```bash
uv sync --locked
```

Optional profiles are explicit:

```bash
uv sync --locked --extra fused
uv sync --locked --extra ggml
uv sync --locked --extra observability
uv sync --locked --extra vllm
uv sync --locked --extra npu
```

The root `npu` extra enables the Worker and Host-native `expertkit_torch`
frontend with Torch 2.11 and torch-npu 2.11.0rc1. The separately pinned
vLLM-Ascend attention image keeps its own accelerator stack; see the
[Ascend tutorial](./doc/tutorial/ascend/README.md).

Run package checks through the shared environment, for example:

```bash
uv run --package expertkit-worker pytest ek-worker/tests
uv run --package expertkit-transport pytest ek-transport/tests
uv run --package expertkit-torch pytest ek-integration/expertkit_torch/tests
```

The committed workspace resolves against official PyPI. A regional mirror is a
developer-local uv setting and must not be committed with changes to
`pyproject.toml` or `uv.lock`. CI and release validation use uv 0.11.30 and the
committed lockfile.

## Key Features

- **Low-Cost Deployment**: supports distributed and mixed GPUs and CPUs.
- **Fine-Grained Expert-Level Scalability**: provides independent scaling of attention and experts, with dynamic scaling of hot experts on demand

## Performance

| Model                  | Throughput (tokens/s) | Environment                                         |
| ---------------------- | --------------------- | --------------------------------------------------- |
| DeepSeek-V3 671B W8A16 | 14.26                 | 1xNvidia 4090(24G) + 5xAMD EPYC 7302                |
| Qwen3-MoE-30B FP16     | 36.38                 | 1xNvidia A10(24G) + 1xAMD EPYC 7302 + 1xKunpeng 920 |

## Repository Map

- [ek-worker](./ek-worker): Python Worker, Compute backends, Weight Manager, and observability.
- [ek-transport](./ek-transport): shared routed-MoE dispatch, retry, and aggregation.
- [ek-proto](./ek-proto): protobuf contracts for computation and control paths.
- [ek-integration](./ek-integration): Transformers and vLLM Frontend integrations.
- [ek-computation](./ek-computation): Controller placement, topology, and recovery state.
- [ek-db](./ek-db): metadata and fine-grained expert-weight storage.
- [ek-cli](./ek-cli): service and administrative command entry point.
- [ek-solution](./ek-solution): deployment recipes.

## Roadmap

### Core Features

- [x] **Frontend** for request schedule
  - [x] Simple Executor
  - [ ] Extensible Executor
  - [ ] Schedule Interface
- [x] **Backend** compute engine for expert computation
  - [x] pytorch
  - [ ] onnxruntime
  - [ ] candle
- [x] **Integration** with existing framework for attention computation
  - [x] pytorch
  - [x] vLLM
- [x] **Transport** channel between frontend and backend
  - [x] gRPC
  - [x] Shared memory for same-host payloads
  - [ ] RDMA
  - [ ] DSM

## Contact Us

If you have any questions, please join our discussion at https://expert-kit.zulipchat.com/ or post new [issues](https://github.com/expert-kit/expert-kit/issues).

## License Agreement

- **Primary License**: This project as a whole is licensed under the [GNU GPL 3.0](LICENSE).

- **Third-Party Components**:

  - Licenses and copyright notices for third-party components are located alongside the component code directory.
  - The following components are included:
    - **Qwen3-MoE**: Located in `ek-integration/expertkit_torch/expertkit_torch/models/`. This code is licensed under [Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

- **Compliance**: All third-party components are used in compliance with their original license terms.
