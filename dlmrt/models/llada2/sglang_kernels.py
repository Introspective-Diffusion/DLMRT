"""Kernels that reproduce SGLang's LLaDA2 numerics bitwise in fewer launches (sglang_numerics.py): the FP32 router
GEMM in cuBLAS's split-K order, the MoE grouping and expert combines, and the decisions' step statistics (the QK norm
+ RoPE is shared with SDAR: ops/triton/sglang_rope.py). Each docstring names the arithmetic it reproduces."""

import torch
import triton
import triton.language as tl

from ...ops.triton.common import LAUNCH, _release, _wait

STAT_CHUNK = 4096  # row_stats' vocabulary chunk


@triton.jit
def _moe_sum(X, OUT, S, scale, H: tl.constexpr, BLOCK: tl.constexpr, SEQ: tl.constexpr):
    """Token t's expert combine: its 8 top-k outputs (X [tokens, 8, H]) summed in FP32, times scale, rounded to BF16,
    plus the shared expert's output S as SGLang's BF16 add of the rounded combine. SEQ (more than 32 tokens):
    sgl_kernel's moe_sum_reduce, in order; otherwise the inductor kernel of the paper's runs, in adjacent pairs."""
    _wait()
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    base = X + t.to(tl.int64) * 8 * H + cols
    v0 = tl.load(base, mask=mask).to(tl.float32)
    v1 = tl.load(base + H, mask=mask).to(tl.float32)
    v2 = tl.load(base + 2 * H, mask=mask).to(tl.float32)
    v3 = tl.load(base + 3 * H, mask=mask).to(tl.float32)
    v4 = tl.load(base + 4 * H, mask=mask).to(tl.float32)
    v5 = tl.load(base + 5 * H, mask=mask).to(tl.float32)
    v6 = tl.load(base + 6 * H, mask=mask).to(tl.float32)
    v7 = tl.load(base + 7 * H, mask=mask).to(tl.float32)
    if SEQ:
        s = (((((((v0 + v1) + v2) + v3) + v4) + v5) + v6) + v7) * scale
    else:
        s = (((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7))) * scale
    s = s.to(tl.bfloat16).to(tl.float32) + tl.load(S + t.to(tl.int64) * H + cols, mask=mask).to(tl.float32)
    tl.store(OUT + t.to(tl.int64) * H + cols, s.to(tl.bfloat16), mask=mask)
    _release()


def moe_sum(x, shared, out, scale):
    """[tokens, 8, hidden] expert outputs combined into out, with the shared expert's output added."""
    tokens, topk, hidden = x.shape
    if topk != 8 or not x.is_contiguous():
        raise NotImplementedError("expert combine: contiguous [tokens, 8, hidden]")
    _moe_sum[(tokens, triton.cdiv(hidden, 1024))](x, out, shared, float(scale), hidden, 1024, tokens > 32,
                                                  num_warps=4, **LAUNCH)


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@triton.jit
def _moe_align(IDS, SORTED, EXPERT_IDS, NTP, CURSOR, T: tl.constexpr, TP: tl.constexpr, E: tl.constexpr,
               EP: tl.constexpr, BM: tl.constexpr, EM: tl.constexpr, EMP: tl.constexpr, NB: tl.constexpr,
               NBP: tl.constexpr):
    """moe_align_block_size's outputs in one program: the T (token, slot) entries of IDS grouped by expert, each
    expert's group padded to BM entries with T (SORTED, EM long), the expert of each BM block (EXPERT_IDS) and the
    padded total (NTP). Entries within an expert's group come in any order: the expert kernels compute and store
    every entry's row on its own."""
    _wait()
    t = tl.arange(0, TP)
    valid = t < T
    ids = tl.load(IDS + t, mask=valid, other=E)
    e = tl.arange(0, EP)
    tl.store(CURSOR + e, tl.zeros((EP,), tl.int32))
    s = tl.arange(0, EMP)
    tl.store(SORTED + s, tl.full((EMP,), T, tl.int32), mask=s < EM)
    b = tl.arange(0, NBP)
    tl.store(EXPERT_IDS + b, tl.zeros((NBP,), tl.int32), mask=b < NB)
    tl.debug_barrier()
    rank = tl.atomic_add(CURSOR + ids, 1, mask=valid)  # the entry's place in its expert's group; CURSOR: counts
    tl.debug_barrier()
    counts = tl.where(e < E, tl.load(CURSOR + e), 0)
    padded = (counts + BM - 1) // BM * BM
    starts = tl.cumsum(padded, 0) - padded
    tl.store(NTP, tl.sum(padded, 0))
    tl.debug_barrier()
    tl.store(CURSOR + e, starts)
    tl.store(EXPERT_IDS + starts // BM, e, mask=counts > 0)  # each nonempty expert's first block
    tl.debug_barrier()
    tl.store(SORTED + tl.load(CURSOR + ids, mask=valid, other=0) + rank, t, mask=valid)
    first = tl.load(EXPERT_IDS + b, mask=b < NB, other=0)
    tl.store(EXPERT_IDS + b, tl.associative_scan(first, 0, _max), mask=b < NB)
    _release()


def moe_align(ids, block_m, experts, cursor):
    """moe_align_block_size's (sorted_token_ids, expert_ids, padded total) for up to 1024 entries; `cursor`: an int32
    scratch of next_power_of_2(experts + 1)."""
    t = ids.numel()
    em = t * block_m if t < experts + 1 else t + (experts + 1) * (block_m - 1)
    nb = triton.cdiv(em, block_m)
    sorted_ids = torch.empty(em, dtype=torch.int32, device=ids.device)
    expert_ids = torch.empty(nb, dtype=torch.int32, device=ids.device)
    total = torch.empty(1, dtype=torch.int32, device=ids.device)
    p2 = triton.next_power_of_2
    _moe_align[(1,)](ids, sorted_ids, expert_ids, total, cursor, t, p2(t), experts, p2(experts + 1), block_m, em,
                     p2(em), nb, p2(nb), num_warps=8, **LAUNCH)
    return sorted_ids, expert_ids, total


@triton.jit
def _router_logits(X, W, OUT, PART, COUNT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, CHUNK: tl.constexpr,
                   BN: tl.constexpr):
    """SGLang's router logits of M BF16 rows, bitwise in one launch: cuBLAS's FP32 GEMM x.float() @ W.T (for these
    shapes split-K: each K chunk of CHUNK one in-order FMA chain, the chunks' partials summed in chunk order), rounded
    to BF16, kept as FP32. Program (n tile, chunk) runs its chunk's chain (Triton's FP32 FMA dot); the last of an n
    tile's programs to finish sums its partials. COUNT: one zeroed counter per n tile, left zeroed."""
    _wait()
    NC: tl.constexpr = K // CHUNK
    pn = tl.program_id(0)
    c = tl.program_id(1)
    m = tl.arange(0, M)
    n = pn * BN + tl.arange(0, BN)
    k = c * CHUNK + tl.arange(0, CHUNK)
    a = tl.load(X + m[:, None] * K + k[None, :]).to(tl.float32)
    b = tl.load(W + n[None, :] * K + k[:, None])
    tile = m[:, None] * N + n[None, :]
    tl.store(PART + c * M * N + tile, tl.dot(a, b, input_precision="ieee"))
    tl.debug_barrier()
    if tl.atomic_add(COUNT + pn, 1, sem="acq_rel") == NC - 1:
        total = tl.load(PART + tile, cache_modifier=".cg")
        for ci in range(1, NC):
            total = total + tl.load(PART + ci * M * N + tile, cache_modifier=".cg")
        tl.store(OUT + tile, total.to(tl.bfloat16).to(tl.float32))
        tl.store(COUNT + pn, 0)
    _release()


def router_logits(x, w, out, part, count):
    """For 32 or 64 rows (the shapes whose cuBLAS split-K order this reproduces): x [m, hidden] BF16, w [experts,
    hidden] FP32; `part` FP32 scratch of hidden // 64 * m * experts, `count` zeroed int32 of experts // 16."""
    m, (e, hidden) = x.shape[0], w.shape
    _router_logits[(e // 16, hidden // 64)](x, w, out, part, count, m, e, hidden, 64, 16, num_warps=m // 32, **LAUNCH)


@triton.jit
def _round_bf16(X, N: tl.constexpr, BLOCK: tl.constexpr):
    """FP32 values rounded to BF16 and widened back, in place (SGLang's router logits .to(bf16) then .float())."""
    _wait()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    tl.store(X + i, tl.load(X + i, mask=mask).to(tl.bfloat16).to(tl.float32), mask=mask)
    _release()


def round_bf16(x):
    _round_bf16[(triton.cdiv(x.numel(), 1024),)](x, x.numel(), 1024, num_warps=4, **LAUNCH)


@triton.jit
def _row_max(X, stride_x, V, AMAX, XMAX, PART_M, PART_I, COUNT, NC: tl.constexpr, NCP: tl.constexpr,
             BLOCK: tl.constexpr):
    """row_stats' argmax (first index of the maximum) and maximum per row, one program per (row, BLOCK chunk), the
    last of a row's programs combining the chunks' (both are exact in any order). COUNT: zeroed, left zeroed."""
    _wait()
    row = tl.program_id(0)
    c = tl.program_id(1)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row.to(tl.int64) * stride_x + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
    cm = tl.max(x, 0)
    tl.store(PART_M + row * NC + c, cm)
    tl.store(PART_I + row * NC + c, tl.min(tl.where(x == cm, offs, 2 ** 31 - 1), 0))
    tl.debug_barrier()
    if tl.atomic_add(COUNT + row, 1, sem="acq_rel") == NC - 1:
        ci = tl.arange(0, NCP)
        pm = tl.load(PART_M + row * NC + ci, mask=ci < NC, other=float("-inf"), cache_modifier=".cg")
        pi = tl.load(PART_I + row * NC + ci, mask=ci < NC, other=2 ** 31 - 1, cache_modifier=".cg")
        m = tl.max(pm, 0)
        tl.store(AMAX + row, tl.min(tl.where(pm == m, pi, 2 ** 31 - 1), 0).to(tl.int64))
        tl.store(XMAX + row, m)
        tl.store(COUNT + row, 0)
    _release()


@triton.jit
def _row_conf(X, stride_x, V, T, t_scalar, XMAX, OUT, stride_out, PART, COUNT, T_FROM_PTR: tl.constexpr,
              NC: tl.constexpr, BLOCK: tl.constexpr):
    """row_conf_prob, 1 / sum over the row of exp((x - xmax) / T), with its arithmetic: each BLOCK chunk summed as
    its loop sums one (same tile, warps and reduction), the chunk sums added in chunk order from zero, here by the
    last of a row's (row, chunk) programs. T: a pointer to the 0-dim temperature or the clamped python float."""
    _wait()
    row = tl.program_id(0)
    c = tl.program_id(1)
    if T_FROM_PTR:
        temp = tl.maximum(tl.load(T).to(tl.float32), 1e-6)
    else:
        temp = t_scalar
    xm = tl.load(XMAX + row)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row.to(tl.int64) * stride_x + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
    tl.store(PART + row * NC + c, tl.sum(tl.exp((x - xm) / temp), 0))
    tl.debug_barrier()
    if tl.atomic_add(COUNT + row, 1, sem="acq_rel") == NC - 1:
        s = tl.zeros((), tl.float32)
        for ci in range(NC):
            s += tl.load(PART + row * NC + ci, cache_modifier=".cg")
        tl.store(OUT + row * stride_out, 1.0 / s)
        tl.store(COUNT + row, 0)
    _release()


def step_stats(logits, t, stats, proposal, part, part_i, count, xmax):
    """The decisions' inputs from SGLang's row_stats / row_conf_prob: argmax (first index) into `proposal`, the
    top-token probability at confidence temperature `t` (a python float, or a 0-dim device tensor read with the same
    arithmetic) into stats[:, 1]. The log-normalizer and entropy, which the decisions do not read, are not computed.
    Scratch for up to len(count) rows: part / part_i of cdiv(vocab, STAT_CHUNK) per row, count zeroed, xmax."""
    n, v = logits.shape
    nc, xmax = triton.cdiv(v, STAT_CHUNK), xmax[:n]
    _row_max[(n, nc)](logits, logits.stride(0), v, proposal, xmax, part, part_i, count, nc, triton.next_power_of_2(nc),
                      STAT_CHUNK, num_warps=8, **LAUNCH)
    ptr = isinstance(t, torch.Tensor)
    _row_conf[(n, nc)](logits, logits.stride(0), v, t if ptr else xmax, 0.0 if ptr else max(float(t), 1e-6), xmax,
                       stats[:, 1], stats.stride(0), part, count, ptr, nc, STAT_CHUNK, num_warps=8, **LAUNCH)
