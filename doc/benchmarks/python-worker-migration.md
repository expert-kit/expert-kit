# Python Worker migration benchmark

This benchmark was recorded before removing the legacy Rust Worker. It compares
the public computation paths at commit
`ebe3f716a7bef484e75ecff45c3ec1e4f12ef6bf` on
`refactor/python-worker`.

The migration has no numerical performance gate. The purpose of this record is
to detect material regressions, confirm that the target workloads fit, and
leave measured follow-up work rather than unverified performance claims.

## Test environment

- Date: 2026-07-17
- CPU: 2 x AMD EPYC 7302, 32 physical cores and 64 hardware threads
- Host memory: 629 GiB
- GPU 0: NVIDIA A100-PCIE-40GB
- GPU 1: NVIDIA A10 23GB
- NVIDIA driver: 595.58.03
- Linux: 6.2.0-33-generic
- Python: 3.12.3
- PyTorch: 2.11.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- grpcio: 1.71.0
- safetensors: 0.8.0
- Model: Qwen3-30B-A3B BF16 from local storage

Unrelated processes occupied about 7.6 GiB on the A100 throughout the tests.
The device-memory results therefore compare complete test configurations on
the same host; they are not isolated per-process allocation measurements.

Aggregate Host CPU is the sum of user and system CPU time from `/proc` for the
Frontend driver, Controller, Weight Server, and active Workers, divided by wall
time. Values above 100% mean that more than one CPU core was active.

## Torch routed-layer comparison

This is the required Torch-to-Torch comparison. Both cases used real layer-0
Qwen expert weights on the A100:

- 64 BF16 token rows with hidden size 2048;
- top-k 8 and uniform routing weights;
- the same eight experts: 1, 15, 24, 25, 28, 30, 35, and 37;
- 10 warm-up requests followed by 100 sequential measured requests;
- one A100 Torch Worker performed the selected computation, while the
  corresponding A10 Torch Worker remained loaded but idle.

The Python path was Frontend integration → v2 Transport middleware → Python
Torch Worker. The Controller supplied topology but did not proxy computation.
The Rust path was the legacy Frontend → Controller v1 computation proxy → Rust
Torch Worker path. Its model-facing timing includes the legacy Frontend
weighted aggregation.

| Metric | Rust v1 | Python v2 | Python change |
|---|---:|---:|---:|
| Routed tokens/s | 2,604.23 | 3,784.38 | +45.3% |
| Assignments/s | 20,833.83 | 30,275.06 | +45.3% |
| Request P50 | 24.34 ms | 16.74 ms | -31.2% |
| Request P95 | 27.77 ms | 22.43 ms | -19.2% |
| Aggregate Host CPU | 246.6% | 3,466.3% | 14.1x |
| A100 peak device memory | 29,330 MiB | 29,178 MiB | -152 MiB |
| A10 peak device memory | 20,677 MiB | 20,699 MiB | +22 MiB |
| Frontend peak allocated memory | 4.75 MiB | 2.51 MiB | -2.24 MiB |

The v2 path improved latency and throughput for this shape, but its aggregate
Host CPU use is materially higher. The benchmark does not identify one cause.
Possible contributors include protobuf and Tensor staging, Python-side
assignment grouping, and repeated single-expert Torch dispatch. Those are
hypotheses, not findings, until component profiling separates them.

## Full Qwen generation supplement

The full model's approximately 54 GiB of expert weights did not fit in the two
available GPUs after runtime and unrelated-process memory were reserved. The
same exact three-Worker placement was therefore used for both implementations:

- A100 Torch Worker: 2,047 experts, assignment hash
  `35a3778830b4c4e638fe0ce9dfd6d7a4`;
- A10 Torch Worker: 2,044 experts, assignment hash
  `46828319a2f795ed3c17f25889901dba`;
- CPU GGML Worker with 32 threads: 2,053 experts, assignment hash
  `42b422bdf2cd33c06bff52aede2d6a14`.

This supplement validates the full distributed model but is not a pure-Torch
comparison. Each run used batch size 1, the prompt
`What is a mixture-of-experts model?`, one warm-up token, and 20 measured
generated tokens. Both runs produced nonempty text with exactly 20 output
tokens.

| Metric | Rust v1 | Python v2 | Python change |
|---|---:|---:|---:|
| Output tokens/s | 2.406 | 1.247 | -48.2% |
| Routed tokens/s | 207.85 | 107.73 | -48.2% |
| Layer P50 | 5.69 ms | 13.09 ms | +130.3% |
| Layer P95 | 12.73 ms | 24.69 ms | +94.0% |
| Aggregate Host CPU | 684.3% | 2,430.0% | 3.55x |
| A100 peak device memory | 32,588 MiB | 32,160 MiB | -428 MiB |
| A10 peak device memory | 21,369 MiB | 20,775 MiB | -594 MiB |
| Frontend peak allocated memory | 2,959.24 MiB | 2,960.74 MiB | +1.50 MiB |

The end-to-end Python path is materially slower in this mixed-backend case.
The CPU GGML Worker is a plausible contributor, but a single aggregate run
cannot distinguish GGML computation from serialization, middleware, or
Frontend costs. The result does not block the migration because the configured
request completed without memory exhaustion and the design defines no fixed
performance threshold.

## Follow-up measurements

The following performance work remains tracked here:

1. Profile v2 protobuf encoding, Tensor staging, grouping, aggregation, and
   single-expert Torch dispatch separately.
2. Compare the Rust and Python GGML Workers directly with the same Worker batch
   to isolate the full-generation regression.
3. Record per-process CPU time and repeat each case on an otherwise idle host
   before attributing the aggregate CPU difference.
4. Repeat representative prefill and decode shapes before changing execution
   concurrency, batching, or the protocol.
