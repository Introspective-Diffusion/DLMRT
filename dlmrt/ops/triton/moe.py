"""Mixture-of-experts routing (softmax top-k, renormalized, per-expert scale) and the masked expert layout's
gather / GeGLU / combine kernels, shared by DiffusionGemma and LLaDA2.
"""

import torch
import triton
import triton.language as tl

from .common import LAUNCH, _geglu_row, _groups, _release, _row_tile, _wait


@triton.jit
def _route_rows(L, PES, IDS, W, POS, E: tl.constexpr, K: tl.constexpr, PACK: tl.constexpr):
    """One token: top-k of the FP32 router softmax, renormalized and scaled per expert, listed in ascending expert
    order; POS[token, e] is the list position of expert e, or -1. With PACK, IDS gets expert << 16 | BF16 weight bits
    instead (FlashInfer's packed top-k)."""
    _wait()
    row = tl.program_id(0)
    cols = tl.arange(0, E)
    x = tl.load(L + row * E + cols).to(tl.float32)
    p = tl.exp(x - tl.max(x, 0))
    p = p / tl.sum(p, 0)
    taken = cols < 0
    for _ in tl.static_range(K):
        avail = tl.where(taken, -1.0, p)
        pick = tl.min(tl.where(avail == tl.max(avail, 0), cols, E), 0)
        taken = taken | (cols == pick)
    top = tl.where(taken, p, 0.0)
    w = top / tl.sum(top, 0) * tl.load(PES + cols).to(tl.float32)
    t = taken.to(tl.int32)
    rank = tl.cumsum(t, 0) - t
    if PACK:
        bits = w.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        tl.store(IDS + row * K + rank, (cols << 16) | bits, taken)
    else:
        tl.store(IDS + row * K + rank, cols, taken)
        tl.store(W + row * K + rank, w, taken)
        tl.store(POS + row * E + cols, tl.where(taken, rank, -1))
    _release()


@triton.jit
def _route_slots(POS, SLOTS, COUNTS, T: tl.constexpr, E: tl.constexpr, K: tl.constexpr):
    """One expert: its tokens numbered in row order (deterministic, no atomics)."""
    _wait()
    e = tl.program_id(0)
    rows = tl.arange(0, T)
    pos = tl.load(POS + rows * E + e)
    taken = pos >= 0
    t = taken.to(tl.int32)
    tl.store(SLOTS + rows * K + pos, tl.cumsum(t, 0) - t, taken)
    tl.store(COUNTS + e, tl.sum(t, 0))
    _release()


def route(logits, per_expert_scale, ids, weights, slots, counts, top_k):
    t, e = logits.shape
    pos = torch.empty(t, e, device=logits.device, dtype=torch.int32)
    _route_rows[(t,)](logits, per_expert_scale, ids, weights, pos, e, top_k, False, num_warps=1, **LAUNCH)
    _route_slots[(e,)](pos, slots, counts, t, e, top_k, num_warps=4, **LAUNCH)


def route_topk(logits, per_expert_scale, ids, weights, pos, top_k):
    """route's experts (int32) and FP32 weights only, [tokens, top_k] each; `pos` is [tokens, experts] int32
    scratch."""
    t, e = logits.shape
    _route_rows[(t,)](logits, per_expert_scale, ids, weights, pos, e, top_k, False, num_warps=1, **LAUNCH)


def route_packed(logits, per_expert_scale, packed, top_k):
    """route's experts and weights packed as expert << 16 | BF16 weight bits, [tokens, top_k] int32."""
    t, e = logits.shape
    _route_rows[(t,)](logits, per_expert_scale, packed, packed, packed, e, top_k, True, num_warps=1, **LAUNCH)


@triton.jit
def _moe_gather(X, IDS, SLOTS, A, lo, LOCAL: tl.constexpr, M_MAX: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                G: tl.constexpr):
    _wait()
    row, k = tl.program_id(0), tl.program_id(1)
    cols, mask = _row_tile(N, G)
    e = tl.load(IDS + row * K + k) - lo
    if (e >= 0) & (e < LOCAL):
        dst = e * M_MAX + tl.load(SLOTS + row * K + k)
        tl.store(A + dst.to(tl.int64) * N + cols, tl.load(X + row * N + cols, mask, 0.0), mask)
    _release()


def moe_gather(x, ids, slots, a, lo=0):
    """Copy every token row to the masked-layout slots of its experts in [lo, lo + a.shape[0]): a[e - lo, slot] = x."""
    t, n = x.shape
    k = ids.shape[1]
    _moe_gather[(t, k)](x, ids, slots, a, lo, a.shape[0], a.shape[1], k, n, _groups(n), num_warps=4, **LAUNCH)


@triton.jit
def _moe_geglu(X, Y, IDS, SLOTS, lo, LOCAL: tl.constexpr, M_MAX: tl.constexpr, F: tl.constexpr, G: tl.constexpr):
    _wait()
    a, part = tl.program_id(0), tl.program_id(1)
    e = tl.load(IDS + a) - lo
    if (e >= 0) & (e < LOCAL):
        row = e.to(tl.int64) * M_MAX + tl.load(SLOTS + a)
        _geglu_row(X, Y, row, row, part, F, G)
    _release()


def moe_geglu(x, ids, slots, out, lo=0):
    """GeGLU of the assigned rows only, in the masked layout [E, M_max, 2F] -> [E, M_max, F]."""
    f = out.shape[2]
    _moe_geglu[(ids.numel(), triton.cdiv(f, 1024))](x, out, ids, slots, lo, x.shape[0], x.shape[1], f, 8,
                                                     num_warps=4, **LAUNCH)


@triton.jit
def _moe_combine(D, IDS, SLOTS, W, Y, lo, LOCAL: tl.constexpr, M_MAX: tl.constexpr, K: tl.constexpr,
                 N: tl.constexpr, G: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    cols, mask = _row_tile(N, G)
    acc = tl.zeros((G, 128), dtype=tl.float32)
    for k in tl.static_range(K):
        e = tl.load(IDS + row * K + k) - lo
        local = (e >= 0) & (e < LOCAL)
        src = e.to(tl.int64) * M_MAX + tl.load(SLOTS + row * K + k)
        acc += tl.where(local, tl.load(W + row * K + k), 0.0) * tl.load(D + src * N + cols, mask & local, 0.0).to(
            tl.float32)
    tl.store(Y + row * N + cols, acc.to(tl.bfloat16), mask)
    _release()


def moe_combine(d, ids, slots, weights, out, lo=0):
    """out[token] = sum over its experts in [lo, lo + d.shape[0]) of weight * d[expert - lo, slot], FP32, ascending
    expert order (this rank's partial under expert parallelism)."""
    t, k = ids.shape
    n = d.shape[2]
    _moe_combine[(t,)](d, ids, slots, weights, out, lo, d.shape[0], d.shape[1], k, n, _groups(n), num_warps=8,
                       **LAUNCH)
