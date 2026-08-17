# Expert Kit Mooncake build

Expert Kit's Transfer Engine transport requires a safety-patched Mooncake
binding. Do not replace it with the public `mooncake-transfer-engine` wheel:
the runtime checks EK-specific capabilities before registering any Tensor.

The EK builds are based on Mooncake commit `731c4521`. The current `rdma2`
artifact supports two explicitly forced profiles: the legacy same-host
`nvlink_intra` backend and the experimental cross-host `rdma` backend. A
runtime selects exactly one backend; an RDMA initialization must report exactly
`rdma` through `get_configured_backend()` and refuses silent TCP fallback.

The wheels were built with TENT available, but EK does not support TENT
selection. Always leave `MC_USE_TENT` and `MC_USE_TEV1` unset. The 2026-08-11
consistency run of TENT's NVLink selector exposed a missing producer-stream
fence, so the mere presence of TENT in a binary is not a supported runtime
configuration.

The source changes are an ordered two-patch series applied to the exact
upstream base below. The RDMA supplemental patch must be applied after the
common safety patch:

```text
base: 731c4521dae71a78bc6fe4e218ad52cfc6ed3554

1. patches/ek-731c4521.patch
   sha256: c6168451f17443bc00e01455a55eff411a2787d8c88139aed43adf9048d78e26
2. patches/ek-rdma-731c4521.patch
   sha256: 67cfe82e5191c8be2582f989f346a8d21527b87a2601314b36269cec0370b3f2
```

Reproduce the patched source tree from a clean Mooncake checkout. Set the
following paths for your build environment:

```bash
MOONCAKE_SRC=/path/to/mooncake
EK_REPO=/path/to/expert-kit

git -C "$MOONCAKE_SRC" switch --detach \
  731c4521dae71a78bc6fe4e218ad52cfc6ed3554
git -C "$MOONCAKE_SRC" apply --check \
  "$EK_REPO/third_party/mooncake/patches/ek-731c4521.patch"
git -C "$MOONCAKE_SRC" apply \
  "$EK_REPO/third_party/mooncake/patches/ek-731c4521.patch"
git -C "$MOONCAKE_SRC" apply --check \
  "$EK_REPO/third_party/mooncake/patches/ek-rdma-731c4521.patch"
git -C "$MOONCAKE_SRC" apply \
  "$EK_REPO/third_party/mooncake/patches/ek-rdma-731c4521.patch"
```

The original, NVLink-only validated CPython 3.12 artifact is retained for
reproducibility:

```text
mooncake_transfer_engine-0.3.12.dev20260811+ek.731c4521-cp312-cp312-manylinux_2_35_x86_64.whl
size:   53249663 bytes
sha256: 2b7e66b9c9b79fd11ad1e5b1d7acc14646fccee05965d070d36df4b44fb450fd
```

The current dual-backend CPython 3.12 artifact is:

```text
mooncake_transfer_engine-0.3.12.dev20260817+ek.731c4521.rdma2-cp312-cp312-manylinux_2_35_x86_64.whl
size:   53256502 bytes
sha256: e1492eb1fff71d2a9b2a8155fc6c57d6ea25d72cf6433f6486bde2c96d809b13
```

Install the current wheel explicitly after enabling the `transfer-engine`
extra:

```bash
uv sync --locked --extra transfer-engine
WHEEL_DIR=/path/to/private-wheel-directory
uv pip install --python .venv/bin/python \
  "$WHEEL_DIR/mooncake_transfer_engine-0.3.12.dev20260817+ek.731c4521.rdma2-cp312-cp312-manylinux_2_35_x86_64.whl"
```

Run the resulting environment through `.venv/bin/python` (or `uv run
--no-sync`). A later ordinary `uv sync` cannot restore this private wheel from
the public lock file and may remove or bypass it; production packaging must
publish the exact artifact to an internal index and pin the version and hash.

This artifact was linked against CUDA 12.8 and its compiled NVLink code refers
to `cudaMemcpyBatchAsync`. If a CUDA 12.6 PyTorch build loads its bundled
`libcudart.so.12` first, importing Mooncake fails with an undefined-symbol
error even when the selected runtime backend is RDMA. Use a PyTorch/CUDA
runtime matched to CUDA 12.8, or preload the host CUDA 12.8 runtime before the
Python process starts. The validated compatibility invocation was:

```bash
LD_PRELOAD=/usr/local/cuda-12.8/lib64/libcudart.so.12 .venv/bin/python ...
```

Do not change `LD_PRELOAD` globally; set it only for the EK process and verify
the exact host CUDA path in deployment. A future RDMA-only artifact built
without the CUDA 12.8 NVLink object would remove this dual-backend ABI coupling.

Every EK binding must expose these common Boolean capabilities:

```text
EK_SAFE_TERMINAL_BATCH_SYNC
EK_HAS_GPUDIRECT_ACQUIRE
```

The forced `nvlink_intra` profile additionally requires:

```text
EK_INTRA_NVLINK_REGISTRATION_REFCOUNT
EK_FORCE_CONFIGURED_TRANSPORT
EK_DRAINED_NVLINK_INTRA_LOCAL_INVALIDATION
```

It must also expose `flush_gpudirect_writes()`, report `nvlink_intra` from
`get_configured_backend()`, and provide
`invalidate_drained_nvlink_intra_segment(target_session)`.

The experimental forced `rdma` profile is supported only with
`P2PHANDSHAKE` metadata and additionally requires:

```text
EK_FORCE_CONFIGURED_RDMA_TRANSPORT
EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION
```

It must report exactly `rdma` from `get_configured_backend()` and expose
`invalidate_drained_rdma_segment(target_session)`. After EK gates and drains
all sessions for a target, that API evicts only the process-local remote
descriptor. It deliberately does not tear down RDMA endpoints or QPs because
those resources can be shared by multiple remote targets.

Descriptor eviction is not endpoint generation replacement. Reusing the same
RDMA endpoint for a new process generation requires restarting the Worker so no
old endpoint/QP state can reach the new generation. Ambiguous transfer,
deregistration, or close outcomes remain fail-stop: quarantine the affected
arena and restart the affected EK processes instead of reusing its address or
rkey. RDMA remains experimental pending hardware fault-injection coverage for
these crash boundaries.
