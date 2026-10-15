"""BF16 GEMMs with cuBLAS's B200 (SM 10.0) summation order, for SM 12.x.

Blackwell's BF16 tensor cores round identically on SM 10.x and SM 12.x; what tells two GEMMs apart is the order in
which K is summed. cuBLAS on B200 accumulates each output in increasing K in FP32, except where its heuristics split
K: then K is cut into `split` ranges of ceil(K / 64 / split) 64-wide tiles (the last range shorter), each summed so,
and the FP32 partials are added in order before the BF16 rounding. `mm` does the same with any tiling, so its result
is bitwise cuBLAS's on B200 for the split given.
"""

import torch
import triton
import triton.language as tl

from .common import LAUNCH, _release, _wait


@triton.jit
def _mm(A, W, Y, P, M, N: tl.constexpr, K: tl.constexpr, KS: tl.constexpr, SWK: tl.constexpr, SWN: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    _wait()
    pm, pn, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    rmask, cmask = rows < M, cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(s * KS, tl.minimum((s + 1) * KS, K), BK):
        a = tl.load(A + rows[:, None].to(tl.int64) * K + k0 + k[None, :], rmask[:, None], 0.0)
        w = tl.load(W + cols[None, :].to(tl.int64) * SWN + (k0 + k[:, None]).to(tl.int64) * SWK, cmask[None, :], 0.0)
        acc = tl.dot(a, w, acc)
    mask = rmask[:, None] & cmask[None, :]
    if SPLIT == 1:
        tl.store(Y + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16), mask)
    else:
        tl.store(P + (s * M + rows[:, None]).to(tl.int64) * N + cols[None, :], acc, mask)
    _release()


@triton.jit
def _reduce(P, Y, MN, SPLIT: tl.constexpr, B: tl.constexpr):
    _wait()
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    mask = i < MN
    acc = tl.load(P + i, mask, 0.0)
    for s in tl.static_range(1, SPLIT):
        acc = acc + tl.load(P + s * MN + i, mask, 0.0)
    tl.store(Y + i, acc.to(tl.bfloat16), mask)
    _release()


def _tiles(m, n):
    """(BM, BN, warps): weight-streaming tiles for decode-sized M, wider ones for prefill chunks."""
    if m > 64:
        return (128, 64, 8) if n >= 3072 else (64, 64, 4)
    bm = 16 if m <= 16 else 32
    return bm, 128 if n >= 32768 else 32 if n <= 4096 else 64, 4


def mm(a, w, out=None, split=1, nn=False):
    """out = a @ w^T for w [N, K] (or a @ w for w [K, N] with nn), BF16 in and out, K summed as cuBLAS on B200
    with `split` K ranges (FP32 partials when split > 1)."""
    m, k = a.shape
    n = w.shape[1] if nn else w.shape[0]
    swk, swn = (w.stride(0), w.stride(1)) if nn else (w.stride(1), w.stride(0))
    out = torch.empty(m, n, device=a.device, dtype=torch.bfloat16) if out is None else out
    bm, bn, warps = _tiles(m, n)
    if k % 64:
        raise ValueError(f"K={k} is not a multiple of 64")
    ks = triton.cdiv(k // 64, split) * 64
    bk = 128 if ks % 128 == 0 and k % 128 == 0 and m <= 64 else 64
    stages = 4 if (bm + bn) * bk * 2 * 4 <= 96 * 1024 else 3
    p = out
    if split > 1:
        p = torch.empty(split * m * n, device=a.device, dtype=torch.float32)
    _mm[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](a, w, out, p, m, n, k, ks, swk, swn, bm, bn, bk, split,
                                                         num_warps=warps, num_stages=stages, **LAUNCH)
    if split > 1:
        _reduce[(triton.cdiv(m * n, 1024),)](p, out, m * n, split, 1024, num_warps=4, **LAUNCH)
    return out
