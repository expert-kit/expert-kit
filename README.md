# Expert Kit: A Distributed, Expert-Centric Framework for MoE LLM Inference

> [!CAUTION]
> Early Work-in-Progress. This project is currently a proof-of-concept demo and is under active development. It is not intended for production use and may contain significant bugs, security vulnerabilities, and unexpected behavior. We appreciate community feedback and contributions as we continue to build and refine this project.

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

## Quick Start

Here are some tutorials to help you quickly start with Expert Kit.

1. [DeepSeek-tiny](./doc/tutorial/deepseek-tiny.md): A tailored MoE model with DeepSeek-V3 architecture and small parameter count, designed for quick evaluation and testing of the Expert Kit framework.
2. [Qwen3-30B-A3B](./doc//tutorial/qwen3-moe-a3b-demo.md): A demo for running the Qwen3-30B-A3B model with Expert Kit, showcasing the framework's capabilities in handling real-world MoE models.

## Key Features

- **Low-Cost Deployment**: supports distributed and mixtured GPU and CPUs.
- **Fine-Grained Expert-Level Scalability**: provides independent scaling of attention and experts, with dynamic scaling of hot experts on demand

## Performance

| Model                  | Throughput (tokens/s) | Environment                                 |
| ---------------------- | --------------------- | ------------------------------------------- |
| DeepSeek-V3 671B W8A16 | 14.26                 | 1x4090(24G) + 5xAMD Epyc 7302               |
| Qwen3-MoE-30B FP16     | 36.38                 | 1xA10(24G) + 1xAMD Epyc 7302 +1xKunPeng 920 |

## Repository Map

- [ek-computation](./ek-agent): performs schedule(frontend) and computation(backend) task.
- [ek-db](./ek-edb): supports registering and loading experts' weight in fine-grained granularity.
- [ek-benchmark](./ek-benchmark): contains several micro-benchmarks help you know the performance.
- [ek-solution](./ek-solution): contains several recipes to quickly setup a running cluster.

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
  - [ ] RDMA
  - [ ] DSM

## Contact Us

If you have any questions, please join our discussion at https://expert-kit.zulipchat.com/

## License Agreement

- **Primary License**: This project as a whole is licensed under the [GNU GPL 3.0](LICENSE).

- **Third-Party Components**:

  - Licenses and copyright notices for third-party components are located alongside the component code directory.
  - The following components are included:
    - **DeepSeek-V3 (Code/Complementary Material)**: Located in `ek-integration/expertkit-torch/expertkit-torch/models/deepseek_v3/`. This code is licensed under the [DeepSeek License Agreement v1.0](ek-integration/expertkit_torch/expertkit_torch/models/deepseek_v3/LICENSE-DEEPSEEK) and the [MIT License](ek-integration/expertkit_torch/expertkit_torch/models/deepseek_v3/LICENSE-MIT). Please be aware that use of the associated DeepSeek Model is subject to the **use restrictions** detailed in **Attachment A** of the DeepSeek License Agreement v1.0.
    - **Qwen3-MoE**: Located in `ek-integration/expertkit-torch/expertkit-torch/models/`. This code is licensed under [Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

- **Compliance**: All third-party components are used in compliance with their original license terms.
