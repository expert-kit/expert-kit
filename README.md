# Expert Kit: A Distributed, Expert-Centric Framework for MoE LLM Inference

> [!CAUTION]
> Early Work-in-Progress. This project is currently a proof-of-concept demo and is under active development. It is not intended for production use and may contain significant bugs, security vulnerabilities, and unexpected behavior. We appreciate community feedback and contributions as we continue to build and refine this project.

<!--[![project chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://expert-kit.zulipchat.com/) -->

## About 
![](./doc/assets/logo-lr-bg.svg) 

**Expert Kit (EK)** is a high-performance framework for scalable MoE (Mixture of Experts) LLM inference. The vision of EK is to provide an efficient foundation of Expert Parallelism (EP) on heterogeneous hardware (e.g., CPU and GPU) over commodity networks (e.g. PCIe, TCP, RDMA), thereby enabling easy deployment and fine-grained expert-level scaling. 

EK features Expert-Attention (E/A) separation architecture, enabling MoE LLMs to be deployed efficiently in a distributed environment composed of *x* CPUs and *y* GPUs.
The motivation behind the E/A separation lies in our observation that, in modern MoE LLMs, expert parameters account for the vast majority of the model size (e.g., over 90% in DeepSeek-V3). 
By decoupling expert modules and deploying them across distributed GPUs and CPUs, EK leverages the high bandwidth and large capacity of distributed memory and storage systems.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./doc/assets/arch-illustration-dark.svg">
  <img alt="arch-illustration-light" src="./doc/assets/arch-illustration.svg">
</picture>

<!--
## Why Expert Kit

The Challenge: Modern MoE models like DeepSeek-V3 contain up to 671B parameters, with nearly 98% dedicated to experts. Traditional inference approaches face:

- Extreme memory pressure on single devices
- Inefficient resource allocation with fixed model deployment
- Limited scaling options that require replicating entire models
- Poor utilization of heterogeneous hardware capabilities

Our Solution: **Expert Kit** decouples attention computation from expert computation, creating a distributed architecture that enables:

- Fine-grained resource management at individual expert level
- Dynamic scaling based on actual expert usage patterns
- Seamless integration of heterogeneous hardware
- Zero-downtime cluster expansion as needs grow
- Running massive MoE models on every day hardware
-->

## Quick Start

Here are some tutorials to help you quickly start with Expert Kit.

1. [DeepSeek-tiny](./doc/tutorial/deepseek-tiny.md): A tailored MoE model with DeepSeek-V3 architecture and small parameter count, designed for quick evaluation and testing of the Expert Kit framework.
2. [Qwen3-30B-A3B](./doc//tutorial/qwen3-moe-a3b-demo.md): A demo for running the Qwen3-30B-A3B model with Expert Kit, showcasing the framework's capabilities in handling real-world MoE models.

## Key Features
- **Low-Cost Deployment**: supports distributed GPU and CPUs, a single GPU is all you need
- **Fine-Grained Expert-Level Scalability**: provides independent scaling of attentioin and experts, with dynamic scaling of hot experts on demand 

<!--
Expert-Level Parallelism

- Fine-Grained Scaling: Schedule and allocate resources at individual expert level
- Dynamic Expert Management: Load experts based on usage patterns and demand
- Seamless Expansion: Add new compute nodes with automatic workload redistribution

Heterogeneous Hardware Support

- Mixed Hardware Pipelines: Combine different GPU generations, CPUs, and accelerators
- Intelligent Placement: Assign experts to the most appropriate compute resources
- Optimized Communication: Efficient cross-device tensor transfer with minimal overhead

Memory & Framework Optimization

- Dramatic Memory Reduction: Offload up to 98% of parameters through expert disaggregation
- Ecosystem Integration: Works with vLLM and Transformers, leveraging PagedAttention and optimized kernels
- Simple Adoption: Drop-in replacement for standard MoE layers with minimal code changes

Universal Accessibility

- Consumer Hardware: Run 600B+ parameter models across everyday devices
- Enterprise Efficiency: Maximize resource utilization in production environments
- Flexible Deployment: Scale from personal setups to data centers with the same architecture
-->

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
If you have any questions, please join in our discussion at https://expert-kit.zulipchat.com/

## License Agreement
- **Primary License**: This project as a whole is licensed under the [GNU GPL 3.0](LICENSE).

- **Third-Party Components**:
  - Licenses and copyright notices for third-party components are located alongside the component code  directory.
  - The following components are included:
    - **DeepSeek-V3 (Code/Complementary Material)**: Located in `ek-integration/expertkit-torch/expertkit-torch/models/deepseek_v3/`. This code is licensed under the [DeepSeek License Agreement v1.0](ek-integration/expertkit_torch/expertkit_torch/models/deepseek_v3/LICENSE-DEEPSEEK) and the [MIT License](ek-integration/expertkit_torch/expertkit_torch/models/deepseek_v3/LICENSE-MIT). Please be aware that use of the associated DeepSeek Model is subject to the **use restrictions** detailed in **Attachment A** of the DeepSeek License Agreement v1.0.
    - **Qwen3-MoE**: Located in `ek-integration/expertkit-torch/expertkit-torch/models/`. This code is licensed under [Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

- **Compliance**: All third-party components are used in compliance with their original license terms.