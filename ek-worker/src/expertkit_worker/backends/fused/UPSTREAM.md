# Fused MoE upstream source manifest

The experimental fused Backend is adapted from vLLM:

- release: `v0.25.1`
- commit: `752a3a504485790a2e8491cacbb35c137339ad34`
- repository: <https://github.com/vllm-project/vllm>
- upstream license: Apache License 2.0
- bundled license: `LICENSE-APACHE-2.0`

The adapted execution boundary uses these upstream sources:

| Upstream path | Adapted behavior |
| --- | --- |
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | Unquantized `fused_moe_kernel`, its one-assignment naive block path, and the two-GEMM gate/up/down execution order. |
| `vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py` | FP32 routing-weight multiplication and top-k reduction. |
| `vllm/model_executor/layers/fused_moe/activation.py` | `SiLU(gate) * up` activation semantics. |

Local changes deliberately remove vLLM registries, scheduler and executor code,
custom C++ assignment sorting, quantization, ROCm paths, model loading, and
platform-wide configuration. Expert weights use preallocated Worker slots and a
persistent `(layer, expert) -> slot` CUDA Tensor. Each assignment uses the
upstream naive block strategy, so no request-time weight stacking or Host token
loop is required. The final reduction writes into the caller-prepared output.

The adapted kernel file retains the upstream Apache-2.0 SPDX and copyright
headers. Updating the upstream commit is an explicit code and numerical-test
change.
