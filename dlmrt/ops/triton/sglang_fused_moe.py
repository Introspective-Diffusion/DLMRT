"""SGLang's Triton fused-MoE kernel on its BF16 path, with the launcher pieces the SGLang-numerics LLaDA2 mode uses:
the kernel launch, the default tile configs and the moe_align_block_size grouping.

Adapted from SGLang (python/sglang/srt/layers/moe/fused_moe_triton/: fused_moe_kernel and invoke_fused_moe_kernel in
fused_moe_triton_kernels.py, get_default_config in fused_moe_triton_config.py, moe_align_block_size.py), Apache
License 2.0, itself adapted from vLLM's fused_moe.py (Apache License 2.0); see THIRD_PARTY_NOTICES.md. Changes: the
BF16 path only (no quantized, TMA, bias, expert-filter or swapped-operand variants; the statements on that path are
unchanged) and dlmrt's programmatic dependent launch (common.py).
"""

import torch
import triton
import triton.language as tl

from .common import LAUNCH, _release, _wait


@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
                     num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am, stride_ak, stride_be,
                     stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                     MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr, compute_type: tl.constexpr,
                     even_Ks: tl.constexpr):
    """C[entry] = A[entry // top_k] @ B[expert].T (times the entry's routing weight with MUL_ROUTED_WEIGHT) for the
    (token, slot) entries of each BLOCK_SIZE_M block of sorted_token_ids, all routed to that block's expert
    (expert_ids). Each output is one FP32 accumulation over K in order, rounded once to compute_type."""
    _wait()
    # Grouped ordering of the (m, n) tiles for L2 reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k_start), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    _release()


def invoke_fused_moe_kernel(a, b, c, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
                            mul_routed_weight, top_k, config):
    """BF16 a [tokens, K] and b [experts, N, K] into c ([tokens, top_k, N] or [entries, N])."""
    grid = (triton.cdiv(sorted_token_ids.shape[0], config["BLOCK_SIZE_M"]) * triton.cdiv(b.shape[1],
                                                                                      config["BLOCK_SIZE_N"]),)
    fused_moe_kernel[grid](a, b, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, b.shape[1],
                           b.shape[2], sorted_token_ids.shape[0], topk_ids.numel(), a.stride(0), a.stride(1),
                           b.stride(0), b.stride(2), b.stride(1), c.stride(-2), c.stride(-1),
                           MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=tl.bfloat16,
                           even_Ks=b.shape[2] % config["BLOCK_SIZE_K"] == 0, **config, **LAUNCH)


def default_config(m, experts):
    """get_default_config's BF16 tiles for m tokens (SGLang has no tuned table for LLaDA2's experts on H100)."""
    if m <= experts:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}


def moe_align_block_size(topk_ids, block_size, num_experts):
    """The (token, slot) entries grouped by expert, each group padded to block_size entries: (sorted_token_ids,
    expert_ids of each block, padded total), by sgl_kernel's moe_align_block_size."""
    from sgl_kernel import moe_align_block_size as align

    t, dev = topk_ids.numel(), topk_ids.device
    padded = t * block_size if t < num_experts + 1 else t + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty(padded, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(triton.cdiv(padded, block_size), dtype=torch.int32, device=dev)
    total = torch.empty(1, dtype=torch.int32, device=dev)
    cumsum = torch.empty(num_experts + 2, dtype=torch.int32, device=dev)
    align(topk_ids, num_experts + 1, block_size, sorted_ids, expert_ids, total, cumsum, True)
    return sorted_ids, expert_ids, total
