# Expert Kit Mooncake build

Expert Kit's Transfer Engine transport requires a safety-patched Mooncake
binding. Do not replace it with the public `mooncake-transfer-engine` wheel:
the runtime checks EK-specific capabilities before registering any Tensor.

The validated server build is based on Mooncake commit
`731c4521` and has distribution version
`0.3.12.dev20260811+ek.731c4521`. It was configured with CUDA,
intra-node NVLink, TENT, TCP, and HTTP enabled. Expert Kit currently selects a
single forced legacy `nvlink_intra` backend and does not use TENT selection.
Always unset `MC_USE_TENT` and `MC_USE_TEV1` for the validated EK path. The
2026-08-11 consistency run of TENT's NVLink selector exposed a missing
producer-stream fence, so the mere presence of TENT in the binary is not a
supported runtime configuration.

The complete source patch is vendored at
[`patches/ek-731c4521.patch`](patches/ek-731c4521.patch). It applies to the
exact upstream base below:

```text
base:   731c4521dae71a78bc6fe4e218ad52cfc6ed3554
sha256: c6168451f17443bc00e01455a55eff411a2787d8c88139aed43adf9048d78e26
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
```

The validated CPython 3.12 artifact is:

```text
mooncake_transfer_engine-0.3.12.dev20260811+ek.731c4521-cp312-cp312-manylinux_2_35_x86_64.whl
size:   53249663 bytes
sha256: 2b7e66b9c9b79fd11ad1e5b1d7acc14646fccee05965d070d36df4b44fb450fd
```

Install that wheel explicitly before enabling the `transfer-engine` extra:

```bash
uv sync --locked --extra transfer-engine
WHEEL_DIR=/path/to/private-wheel-directory
uv pip install --python .venv/bin/python \
  "$WHEEL_DIR/mooncake_transfer_engine-0.3.12.dev20260811+ek.731c4521-cp312-cp312-manylinux_2_35_x86_64.whl"
```

Run the resulting environment through `.venv/bin/python` (or `uv run
--no-sync`). A later ordinary `uv sync` cannot restore this private wheel from
the public lock file and may remove or bypass it; production packaging must
publish the exact artifact to an internal index and pin the version and hash.

The binding must expose all of these Boolean capabilities:

```text
EK_SAFE_TERMINAL_BATCH_SYNC
EK_HAS_GPUDIRECT_ACQUIRE
EK_INTRA_NVLINK_REGISTRATION_REFCOUNT
EK_FORCE_CONFIGURED_TRANSPORT
EK_DRAINED_NVLINK_INTRA_LOCAL_INVALIDATION
```

It must also expose `flush_gpudirect_writes()` and
`invalidate_drained_nvlink_intra_segment(target_session)`. The invalidation call
is valid only after Expert Kit has drained every session referring to that
target segment.
