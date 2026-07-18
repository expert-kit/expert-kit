# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Expert Kit for the restricted fixed-slot unquantized execution path.
"""Minimal unquantized Triton MoE path adapted from vLLM 0.25.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_BLOCK_M = 16
_BLOCK_N = 32
_BLOCK_K = 32
_ACTIVATION_BLOCK = 128
_REDUCTION_BLOCK = 128


@triton.jit
def _expert_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    route_ids_ptr,
    slot_mapping_ptr,
    num_assignments,
    num_experts,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Multiply every routed assignment by its fixed-slot expert matrix."""

    program_id = tl.program_id(axis=0)
    num_n_programs = tl.cdiv(N, BLOCK_N)
    assignment_id = program_id // num_n_programs
    n_program_id = program_id % num_n_programs
    if assignment_id >= num_assignments:
        return

    route_id = tl.load(route_ids_ptr + assignment_id)
    route_is_valid = (route_id >= 0) & (route_id < num_experts)
    slot = tl.load(
        slot_mapping_ptr + route_id,
        mask=route_is_valid,
        other=-1,
    ).to(tl.int64)

    row_offsets = tl.arange(0, BLOCK_M)
    assignment_offsets = tl.where(row_offsets == 0, assignment_id, num_assignments)
    row_mask = assignment_offsets < num_assignments
    n_offsets = n_program_id * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + assignment_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = row_mask[:, None] & (n_offsets[None, :] < N)
    if slot < 0:
        tl.store(c_ptrs, 0.0, mask=c_mask)
        return

    k_offsets = tl.arange(0, BLOCK_K)
    input_rows = assignment_offsets // TOP_K
    a_ptrs = a_ptr + input_rows[:, None] * stride_am + k_offsets[None, :] * stride_ak
    b_ptrs = (
        b_ptr + slot * stride_be + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - k_block * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & (k_offsets[None, :] < remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < remaining) & (n_offsets[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _silu_and_mul_kernel(
    gate_up_ptr,
    activated_ptr,
    intermediate_dim,
    gate_up_stride,
    activated_stride,
    BLOCK: tl.constexpr,
):
    """Apply the fused SiLU gate and up multiplication."""

    assignment_id = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < intermediate_dim
    gate = tl.load(
        gate_up_ptr + assignment_id * gate_up_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up_ptr + assignment_id * gate_up_stride + intermediate_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up
    tl.store(
        activated_ptr + assignment_id * activated_stride + offsets,
        activated,
        mask=mask,
    )


@triton.jit
def _weighted_reduce_kernel(
    expert_output_ptr,
    routing_weights_ptr,
    prepared_output_ptr,
    hidden_dim,
    expert_output_stride,
    output_stride,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Apply FP32 router weights and reduce top-k into caller-owned output."""

    token_id = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < hidden_dim
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for route_index in tl.static_range(TOP_K):
        assignment_id = token_id * TOP_K + route_index
        expert_output = tl.load(
            expert_output_ptr + assignment_id * expert_output_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        routing_weight = tl.load(
            routing_weights_ptr + assignment_id,
        ).to(tl.float32)
        accumulator += expert_output * routing_weight
    tl.store(
        prepared_output_ptr + token_id * output_stride + offsets,
        accumulator,
        mask=mask,
    )


@dataclass(frozen=True, slots=True)
class FusedWorkspace:
    """Retain all request-local buffers until CUDA execution completes."""

    gate_up_output: torch.Tensor
    activated: torch.Tensor
    expert_output: torch.Tensor


def launch_fused_moe(
    *,
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    slot_mapping: torch.Tensor,
    prepared_output: torch.Tensor,
) -> FusedWorkspace:
    """Launch the restricted two-GEMM SiLU MoE path on the current CUDA stream."""

    token_count, hidden_dim = hidden_states.shape
    top_k = expert_ids.shape[1]
    num_assignments = token_count * top_k
    intermediate_dim = down.shape[2]
    gate_up_output = torch.empty(
        (num_assignments, 2 * intermediate_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    activated = torch.empty(
        (num_assignments, intermediate_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_output = torch.empty(
        (num_assignments, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    flat_expert_ids = expert_ids.view(-1)

    first_grid = (num_assignments * triton.cdiv(2 * intermediate_dim, _BLOCK_N),)
    _expert_gemm_kernel[first_grid](
        hidden_states,
        gate_up,
        gate_up_output,
        flat_expert_ids,
        slot_mapping,
        num_assignments,
        slot_mapping.numel(),
        2 * intermediate_dim,
        hidden_dim,
        hidden_states.stride(0),
        hidden_states.stride(1),
        gate_up.stride(0),
        gate_up.stride(1),
        gate_up.stride(2),
        gate_up_output.stride(0),
        gate_up_output.stride(1),
        TOP_K=top_k,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    activation_grid = (
        num_assignments,
        triton.cdiv(intermediate_dim, _ACTIVATION_BLOCK),
    )
    _silu_and_mul_kernel[activation_grid](
        gate_up_output,
        activated,
        intermediate_dim,
        gate_up_output.stride(0),
        activated.stride(0),
        BLOCK=_ACTIVATION_BLOCK,
        num_warps=4,
    )

    second_grid = (num_assignments * triton.cdiv(hidden_dim, _BLOCK_N),)
    _expert_gemm_kernel[second_grid](
        activated,
        down,
        expert_output,
        flat_expert_ids,
        slot_mapping,
        num_assignments,
        slot_mapping.numel(),
        hidden_dim,
        intermediate_dim,
        activated.stride(0),
        activated.stride(1),
        down.stride(0),
        down.stride(1),
        down.stride(2),
        expert_output.stride(0),
        expert_output.stride(1),
        TOP_K=1,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    reduction_grid = (token_count, triton.cdiv(hidden_dim, _REDUCTION_BLOCK))
    _weighted_reduce_kernel[reduction_grid](
        expert_output,
        routing_weights,
        prepared_output,
        hidden_dim,
        expert_output.stride(0),
        prepared_output.stride(0),
        TOP_K=top_k,
        BLOCK=_REDUCTION_BLOCK,
        num_warps=4,
    )
    return FusedWorkspace(gate_up_output, activated, expert_output)
