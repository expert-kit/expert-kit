# EK Issue Label Taxonomy and GitHub Configuration Guidelines

This document defines the GitHub Issue taxonomy, label naming rules, Issue Form configuration approach, and recommended Rulesets for branch and tag protection in the EK repository.

## Design Principles

Issue classification is organized into two layers:

1. **Issue Type**: Marks the primary nature of the issue. Each issue should usually select only one Issue Type.
2. **Area Label**: Marks the technical area, module, or concern related to the issue. One issue may select multiple Area Labels.

GitHub Labels are flat by design and do not provide true hierarchical namespaces. To improve readability and extensibility, this document uses a `/`-style naming convention:

- `type/*`: Primary issue type
- `area/*`: Technical area or module

Examples:

```text
type/performance + area/gpu
type/bug + area/scheduling
type/feature + area/observability
```

This naming style is clearer than bare labels such as `bug` or `gpu`, and it can be extended later with categories such as `priority/*`, `status/*`, or `component/*`.

## Issue Types

**Issue Type labels mark the primary nature of an issue. Each issue should usually select only one Issue Type.**

| GitHub Label       | Name        | Scope |
| ------------------ | ----------- | ----- |
| `type/bug`         | Bug Fix     | Confirmed program errors, abnormal behavior, inference hangs, incorrect results, crashes, and similar defects |
| `type/performance` | Performance | Latency, throughput, p95/p99, GPU/CPU compute optimization, communication optimization, scheduling optimization, and related work |
| `type/feature`     | Feature     | New features, new capabilities, new scheduling strategies, new APIs, and new toolchain capabilities |
| `type/docs`        | Docs        | README updates, deployment docs, usage guides, examples, architecture documentation, FAQs, and similar documentation work |
| `type/usage`       | Usage       | Installation, compilation, dependency setup, configuration, runtime, deployment, and usage questions |
| `type/security`    | Security    | Security vulnerabilities, permission risks, sensitive information leakage, supply-chain security, and related concerns |
| `type/adaptation`  | Adaptation  | Adaptation and validation for new hardware environments, new models, or new software stacks |

## Area Labels

**Area Labels mark the technical area, module, or concern related to an issue. One issue may select multiple Area Labels.**

| GitHub Label         | Name                     | Scope |
| -------------------- | ------------------------ | ----- |
| `area/scheduling`    | Scheduling               | Controller scheduling, LPT strategy, heterogeneous scheduling, tail-latency optimization, priority strategy, expert placement, and related work |
| `area/transport`     | Transport                | gRPC, connection pools, SHM, RDMA, serialization, network communication, transport protocol optimization, and related work |
| `area/gpu`           | GPU Compute              | CUDA, Torch GPU backend, kernel launch, CUDA Graph, streams, GPU expert compute optimization, and related work |
| `area/cpu`           | CPU Compute              | CPU backend, AMX, AVX, NEON/SVE, NUMA, CPU expert compute optimization, and related work |
| `area/cache`         | Cache                    | Expert weight cache, hierarchical cache, memory/disk/peer/central cache, peer fetch, cache hit rate, cache eviction, cache statistics, and related work |
| `area/observability` | Observability / Sensing  | Tracing, metrics, Prometheus, environment sensing, health checks, and system metrics such as CPU/GPU/network usage, queues, and lock waits |

## Issue Labeling Examples

When creating a new issue, select all labels from the **Labels** field.

**Each issue should use one Issue Type and add one or more Area Labels as needed.**

**Example**：

Issue: There is no failure handling after expert selection fails, which may cause `remaining_experts` to wait indefinitely.

Issue Type: `type/bug`

Area Labels: `area/scheduling`
