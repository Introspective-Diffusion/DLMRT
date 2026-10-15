"""LLaDA2's Triton attention (`attention.attention` with 64 x 64 tiles, 4 warps; full or block-causal keys) on
SM 12.x with the numerics it has on SM 10.x, in Gluon.

The two GPUs' tensor cores give the same QK and PV products, and exp / max are elementwise or order-free; what
differs is the row sum of the probabilities, which Triton reduces in the register layout of the scores. On SM 10.x
the scores come out of tensor memory with each thread holding 32 consecutive keys of one row (two threads per row),
so a row sums its two 32-key halves in key order and adds them; on SM 12.x the MMA layout spreads a row over four
warps. Here the probabilities are moved into SM 10.x's layout for that sum, so the output is bitwise the B200's.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from .attention import _attention_combine


@gluon.jit
def _attention(Q, K, V, O, PO, PL, START, END, scale, HQ: gl.constexpr, HK: gl.constexpr, D: gl.constexpr,
               BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, SPLITS: gl.constexpr, ROWS: gl.constexpr,
               GRID: gl.constexpr):
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8])
    a_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=2)
    b_op: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=2)
    # The scores' register layout on SM 10.x (tensor-memory load of a 64 x 64 FP32 tile by 4 warps).
    tmem: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]],
        warp_bases=[[16, 0], [32, 0]], block_bases=[], shape=[BLOCK_M, BLOCK_N])
    ld: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    rows_m: gl.constexpr = gl.SliceLayout(1, mma)
    cols_m: gl.constexpr = gl.SliceLayout(0, mma)
    pid_m, kv, split = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    group: gl.constexpr = HQ // HK
    m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, ld))
    d = gl.arange(0, D, layout=gl.SliceLayout(0, ld))
    q = gl.load(Q + ((m // group) * HQ + kv * group + m % group)[:, None].to(gl.int64) * D + d[None, :])
    q = gl.convert_layout(q, a_op)
    start = gl.load(START)
    end = gl.load(END)
    stop = end
    if GRID > 0:
        stop = gl.minimum(end, ((start + (pid_m * BLOCK_M + BLOCK_M - 1) // group) // GRID + 1) * GRID)
    lo = 0
    hi = stop
    if SPLITS > 1:
        chunk = ((stop + SPLITS - 1) // SPLITS + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        lo = split * chunk
        hi = gl.minimum(stop, lo + chunk)
    mm_ = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=rows_m)
    qpos = start + mm_ // group
    m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=rows_m)
    l_i = gl.zeros([BLOCK_M], gl.float32, layout=rows_m)
    acc = gl.zeros([BLOCK_M, D], gl.float32, layout=mma)
    for n0 in range(lo, hi, BLOCK_N):
        kpos = n0 + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, ld))
        live = kpos < end
        base = (kpos.to(gl.int64) * HK + kv) * D
        k = gl.load(K + base[:, None] + d[None, :], mask=live[:, None], other=0.0).to(gl.bfloat16)
        s = mma_v2(q, gl.convert_layout(gl.permute(k, [1, 0]), b_op),
                   gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=mma)) * scale
        kpos_s = n0 + gl.arange(0, BLOCK_N, layout=cols_m)
        keep = (kpos_s < end)[None, :]
        if GRID > 0:
            keep = keep & (kpos_s[None, :] < (qpos[:, None] // GRID + 1) * GRID)
        s = gl.where(keep, s, float("-inf"))
        m_new = gl.maximum(m_i, gl.max(s, axis=1))
        if SPLITS > 1:
            m_new = gl.where(m_new == float("-inf"), 0.0, m_new)
        p = gl.exp(s - m_new[:, None])
        alpha = gl.exp(m_i - m_new)
        l_i = l_i * alpha + gl.convert_layout(gl.sum(gl.convert_layout(p, tmem), axis=1), rows_m)
        v = gl.load(V + base[:, None] + d[None, :], mask=live[:, None], other=0.0).to(gl.bfloat16)
        acc = mma_v2(gl.convert_layout(p.to(gl.bfloat16), a_op), gl.convert_layout(v, b_op), acc * alpha[:, None])
        m_i = m_new
    index = ((mm_ // group) * HQ + kv * group + mm_ % group).to(gl.int64)
    dm = gl.arange(0, D, layout=cols_m)
    if SPLITS == 1:
        gl.store(O + index[:, None] * D + dm[None, :], (acc / l_i[:, None]).to(gl.bfloat16))
    else:
        gl.store(PO + (split * ROWS + index)[:, None] * D + dm[None, :],
                 gl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0))
        gl.store(PL + split * ROWS + index, gl.where(l_i > 0, m_i + gl.log(l_i), float("-inf")))


def attention(q, k, v, start, end, out=None, *, scale=1.0, grid=0, programs=128, block_m=64, block_n=64):
    """`attention.attention(q, k, v, start, end, False, scale=scale, grid=grid, dv=D, block_m=64, block_n=64,
    num_warps=4)` with the B200's summation order. q [1, rows, HQ, D] over k/v [1, S, HK, D] at [0, end)."""
    _, rows, hq, d = q.shape
    hk = k.shape[2]
    out = torch.empty_like(q) if out is None else out
    if (block_m, block_n) != (64, 64) or (rows * hq // hk) % block_m:
        raise ValueError("b200_attention: 64 x 64 tiles, rows x query group a multiple of 64")
    tiles = rows * hq // hk // block_m
    splits = triton.next_power_of_2(max(1, programs // (tiles * hk)))
    po = pl = out
    if splits > 1:
        po = torch.empty(splits, rows * hq, d, device=q.device, dtype=torch.float32)
        pl = torch.empty(splits, rows * hq, device=q.device, dtype=torch.float32)
    _attention[(tiles, hk, splits)](q, k, v, out, po, pl, start, end, scale, hq, hk, d, block_m, block_n, splits,
                                    rows * hq, grid, num_warps=4)
    if splits > 1:
        _attention_combine[(rows * hq,)](po, pl, out, rows * hq, d, splits, num_warps=4)
    return out
