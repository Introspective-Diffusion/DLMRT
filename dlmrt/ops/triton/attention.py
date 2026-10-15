"""Flash attention in Triton over a contiguous KV arena: the attention of DiffusionGemma's full layers (and of its
sliding layers off Hopper) and of LLaDA2 off Hopper, where FA3 does not run. Query heads of one KV head are packed
along M; causal, sliding-window, left-bounded and block-causal key ranges; split-KV with a combine pass."""

import torch
import triton
import triton.language as tl

from .common import LAUNCH, _release, _wait


@triton.jit
def _attention(Q, K, V, O, PO, PL, START, END, LEFT, scale, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
               DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
               WINDOW: tl.constexpr, HAS_LEFT: tl.constexpr, SPLITS: tl.constexpr, ROWS: tl.constexpr,
               GRID: tl.constexpr):
    """Flash attention, FP32 scores times `scale`. Query heads of one KV head are packed along M; each program owns
    one DV-wide slice of the output (the QK product is recomputed per slice) and, with SPLITS > 1, one of SPLITS key
    ranges, written as a normalized FP32 partial and its log-sum-exp. Keys: [LEFT, END) (HAS_LEFT), causal up to
    the query position, and within WINDOW positions of it (causal sliding layers), or block-causal on a GRID-token
    grid (GRID > 0: a row sees every key before the end of its grid block). K/V may be FP8 (read as BF16)."""
    _wait()
    pid_m, kv, program = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    part, split = program % (D // DV), program // (D // DV)
    group: tl.constexpr = HQ // HK
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row = m // group
    head = kv * group + m % group
    d = tl.arange(0, D)
    dv = part * DV + tl.arange(0, DV)
    q = tl.load(Q + (row * HQ + head)[:, None].to(tl.int64) * D + d[None, :])
    start = tl.load(START)
    end = tl.load(END)
    qpos = start + row
    stop = tl.minimum(end, start + (pid_m * BLOCK_M + BLOCK_M - 1) // group + 1) if CAUSAL else end
    if GRID > 0:
        stop = tl.minimum(end, ((start + (pid_m * BLOCK_M + BLOCK_M - 1) // group) // GRID + 1) * GRID)
    first = 0
    left = 0
    if HAS_LEFT:
        left = tl.load(LEFT)
        first = left
    if WINDOW > 0 and CAUSAL:
        first = tl.maximum(first, start + (pid_m * BLOCK_M) // group - (WINDOW - 1))
    first = first // BLOCK_N * BLOCK_N
    lo, hi = first, stop
    if SPLITS > 1:
        chunk = tl.cdiv(tl.cdiv(stop - first, SPLITS), BLOCK_N) * BLOCK_N
        lo = first + split * chunk
        hi = tl.minimum(stop, lo + chunk)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, DV), tl.float32)
    for n0 in range(lo, hi, BLOCK_N):
        kpos = n0 + tl.arange(0, BLOCK_N)
        live = kpos < end
        if HAS_LEFT:
            live = live & (kpos >= left)
        base = (kpos.to(tl.int64) * HK + kv) * D
        k = tl.load(K + base[:, None] + d[None, :], live[:, None], 0.0).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k)) * scale
        keep = live[None, :]
        if CAUSAL:
            keep = keep & (kpos[None, :] <= qpos[:, None])
            if WINDOW > 0:
                keep = keep & (kpos[None, :] > qpos[:, None] - WINDOW)
        if GRID > 0:
            keep = keep & (kpos[None, :] < (qpos[:, None] // GRID + 1) * GRID)
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        if SPLITS > 1:  # a split's keys can all be masked for some causal rows
            m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + base[:, None] + dv[None, :], live[:, None], 0.0).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    index = (row * HQ + head).to(tl.int64)
    if SPLITS == 1:
        tl.store(O + index[:, None] * D + dv[None, :], (acc / l_i[:, None]).to(tl.bfloat16))
    else:
        tl.store(PO + (split * ROWS + index)[:, None] * D + dv[None, :],
                 tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0))
        if part == 0:
            tl.store(PL + split * ROWS + index, tl.where(l_i > 0, m_i + tl.log(l_i), float("-inf")))
    _release()


@triton.jit
def _attention_combine(PO, PL, O, ROWS: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    _wait()
    r = tl.program_id(0)
    s = tl.arange(0, SPLITS)
    lse = tl.load(PL + s * ROWS + r)
    w = tl.exp(lse - tl.max(lse, 0))
    d = tl.arange(0, D)
    o = tl.load(PO + (s * ROWS + r)[:, None].to(tl.int64) * D + d[None, :])
    tl.store(O + r.to(tl.int64) * D + d, (tl.sum(o * w[:, None], 0) / tl.sum(w, 0)).to(tl.bfloat16))
    _release()


def attention(q, k, v, start, end, causal, out=None, *, window=0, left=None, programs=128, block_m=64, block_n=32,
              dv=256, num_warps=4, num_stages=2, scale=1.0, grid=0):
    """q [1, rows, HQ, D] over the arena k/v [1, S, HK, D] (BF16 or FP8) at [0, end); rows sit at
    [start, start + rows). `left` (a device scalar) drops keys below it; `window` > 0 limits causal rows to their
    last `window` keys; `grid` > 0 makes rows block-causal on that grid (LLaDA's prefill). `scale`: softmax scale.

    Key ranges are split so that ~`programs` programs run (one range when the query tiles already fill the GPU)."""
    _, rows, hq, d = q.shape
    hk = k.shape[2]
    out = torch.empty_like(q) if out is None else out
    dv = min(dv, d)
    if (rows * hq // hk) % block_m or d % dv:
        raise ValueError(f"attention: rows x query group must tile by {block_m} and head_dim by {dv}")
    tiles = rows * hq // hk // block_m
    splits = triton.next_power_of_2(max(1, programs // (tiles * hk * (d // dv))))
    po = pl = out
    if splits > 1:
        po = torch.empty(splits, rows * hq, d, device=q.device, dtype=torch.float32)
        pl = torch.empty(splits, rows * hq, device=q.device, dtype=torch.float32)
    _attention[(tiles, hk, d // dv * splits)](q, k, v, out, po, pl, start, end, start if left is None else left,
                                              scale, hq, hk, d, dv, block_m, block_n, causal, window,
                                              left is not None, splits, rows * hq, grid, num_warps=num_warps,
                                              num_stages=num_stages, **LAUNCH)
    if splits > 1:
        _attention_combine[(rows * hq,)](po, pl, out, rows * hq, d, splits, num_warps=4, **LAUNCH)
    return out
