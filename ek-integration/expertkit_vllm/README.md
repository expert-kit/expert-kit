# Expert Kit integration for vLLM

This package replaces vLLM's routed MoE factory with an implementation that
keeps vLLM's router and shared experts local while sending one complete routed
layer to Expert Kit Transport. It targets vLLM `0.25.1` exactly.

Runtime qualification is deferred during the Python Worker migration. The
current code is version-pinned, and its configuration and registration have
unit tests. The supported Qwen text-generation smoke test uses the Torch
integration.

## Install

Install `expertkit-transport` from this repository first, then install this
package:

```bash
pip install -e ../../ek-transport
pip install -e .
```

## Configure

The plugin is disabled unless `EK_ENABLE=1` is set. When enabled, it reads:

- `EK_ADDR`: Controller gRPC endpoint. The default is `localhost:5002`.
- `EK_INSTANCE_ID`: required positive numeric model instance ID.
- `EK_CLIENT_TIMEOUT`: positive timeout in seconds. The default is `6`.

For example:

```bash
export EK_ENABLE=1
export EK_ADDR=controller.internal:5002
export EK_INSTANCE_ID=1
export EK_CLIENT_TIMEOUT=6
```

The Controller and Workers must already be running, and the instance's expert
weights must be ready before inference starts. The MVP assumes a trusted,
isolated cluster network and does not provide TLS or application authentication.

## Current limits

- Routed experts must use the unquantized SiLU FFN path.
- Tensor, expert, sequence, prefill-context parallelism and EPLB are rejected.
- Pipeline and ordinary data-parallel model processes are allowed.
- Full CUDA Graph capture is changed to piecewise capture because remote MoE
  calls perform network I/O between graph segments.
- The integration depends on vLLM internal interfaces and therefore stays
  pinned until a later version is reviewed and adapted.

The plugin preserves vLLM's model router, including grouped or custom routing,
and sends final `int32` expert assignments and FP32 routing weights through
`expertkit-transport`. Transport returns the weighted, aggregated activation
Tensor to the vLLM layer.
