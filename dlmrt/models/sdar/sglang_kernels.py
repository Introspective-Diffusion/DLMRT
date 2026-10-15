"""Kernels for SDAR's SGLang numerics (sglang_numerics.py): SGLang's row statistics at the confidence temperature, the
decisions of its two decoder paths as one program each, and the MLP's SiLU-and-multiply. Each docstring names the
arithmetic it reproduces.

_row_stats_temp is _row_stats_temp_ptr_kernel of the paper runtime's SGLang fork (added to its
sglang/srt/dllm/algorithm/fused_step_kernels.py by the ORL code's SDAR engine patch; upstream SGLang has no such file;
Apache License 2.0, see THIRD_PARTY_NOTICES.md) with dlmrt's programmatic dependent launch and without the
log-normalizer output the decisions do not read.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ...ops.triton.common import LAUNCH, _release, _wait

STAT_BLOCK = 4096  # SGLang's vocabulary chunk for the row statistics


@triton.jit
def _row_stats_temp(X, stride_x, V, INV_T, AMAX, PMAX, ENT, BLOCK: tl.constexpr):
    """One pass over a logits row at confidence temperature T (INV_T: the FP32 1/T): argmax (first index),
    softmax(x/T)[argmax] = 1 / sum_j exp((x_j - m)/T) and the entropy of softmax(x/T) in nats = log s - t/s with
    s = sum exp(u_j), t = sum u_j exp(u_j), u_j = (x_j - m)/T, online over BLOCK chunks."""
    _wait()
    row = tl.program_id(0)
    inv_t = tl.load(INV_T).to(tl.float32)
    xp = X + row.to(tl.int64) * stride_x
    m = tl.full((), -1.0e30, tl.float32)
    s = tl.zeros((), tl.float32)
    t = tl.zeros((), tl.float32)
    best_idx = tl.full((), 2**31 - 1, tl.int32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(xp + offs, mask=mask, other=float("-inf")).to(tl.float32)
        cm = tl.max(x, 0)
        m_new = tl.maximum(m, cm)
        alpha = tl.exp((m - m_new) * inv_t)
        u = (x - m_new) * inv_t
        e = tl.exp(u)
        s_chunk = tl.sum(e, 0)
        t_chunk = tl.sum(tl.where(mask, u * e, 0.0), 0)
        t = alpha * (t + (m - m_new) * inv_t * s) + t_chunk
        s = alpha * s + s_chunk
        idx_c = tl.min(tl.where(x == cm, offs, 2**31 - 1), 0)
        best_idx = tl.where(cm > m, idx_c, best_idx)
        m = m_new
    log_s = tl.log(s)
    tl.store(AMAX + row, best_idx.to(tl.int64))
    tl.store(PMAX + row, 1.0 / s)
    tl.store(ENT + row, log_s - t / s)
    _release()


def row_stats_temp(logits, inv_t, proposal, pmax, entropy):
    """Per row of logits [rows, vocab] (BF16 or FP32): argmax into proposal, top-token probability and entropy at
    the temperature 1 / inv_t (a 1-element FP32 device tensor) into pmax / entropy."""
    rows, vocab = logits.shape
    _row_stats_temp[(rows,)](logits, logits.stride(0), vocab, inv_t, proposal, pmax, entropy, STAT_BLOCK,
                             num_warps=8, **LAUNCH)


@triton.jit
def _decide(ORDER, SORTED, EXCL, ACC, TOP, PROPOSAL, PP, TOKENS, ACCEPTED, PROMPT, STEP, REMAINING, NOISE_BLOCK,
            threshold, edit_threshold, seed, high, B: tl.constexpr, ENTROPY: tl.constexpr):
    """One denoise step's decisions on a block of B positions, as SGLang's SDARPrefixKV takes them (all integer,
    boolean and comparison arithmetic, so equal to its torch chains). ENTROPY (ORL, its fused fixed-shape path,
    post_step_fused): accept the positions whose exclusive cumulative entropy in ascending order (SORTED / EXCL /
    ORDER, from torch.sort and torch.cumsum) stays within `threshold`, or else the leftmost eligible one. Otherwise
    (Base, its generic path): accept the positions above the probability `threshold`, or else TOP, the
    torch.topk(k=1) of the eligible probabilities. Then edit accepted positions whose proposal changed with
    probability above `edit_threshold`, advance the step and refill the unresolved positions with the step's
    deterministic noise; REMAINING counts them."""
    _wait()
    i = tl.arange(0, B)
    accepted = tl.load(ACCEPTED + i) != 0
    prompt = tl.load(PROMPT + i) != 0
    eligible = ~accepted & ~prompt
    any_eligible = tl.sum(eligible.to(tl.int32), 0) > 0
    pp = tl.load(PP + i)
    if ENTROPY:
        order = tl.load(ORDER + i)
        selected = (tl.load(EXCL + i) <= threshold) & (tl.load(SORTED + i) < float("inf"))
        tl.store(ACC + order, selected.to(tl.int8))
        tl.debug_barrier()
        accept = tl.load(ACC + i) != 0
        first = tl.min(tl.where(eligible, i, B), 0)
        accept = accept | ((tl.sum(accept.to(tl.int32), 0) == 0) & eligible & (i == first))
    else:
        high_conf = (pp > threshold) & eligible
        accept = tl.where(tl.sum(high_conf.to(tl.int32), 0) > 0, high_conf, (i == tl.load(TOP)) & any_eligible)
    token = tl.load(TOKENS + i)
    proposal = tl.load(PROPOSAL + i)
    edit = accepted & ~prompt & (pp > edit_threshold) & (proposal != token)
    accepted = accepted | accept
    token = tl.where(accept | edit, proposal, token)
    step = tl.load(STEP) + 1
    position = (i + tl.load(NOISE_BLOCK) * B).to(tl.int64)
    noise = ((position + 1) * 1103515245 + (step + 1).to(tl.int64) * 12345 + seed) % high
    tl.store(TOKENS + i, tl.where(accepted, token, noise))
    tl.store(ACCEPTED + i, accepted)
    tl.store(STEP, step)
    tl.store(REMAINING, tl.sum((~accepted).to(tl.int32), 0))
    _release()


def decide(state, proposal, pp, decoder, *, order=None, sorted_entropy=None, excl=None, scratch=None, top=None):
    """`state`: tokens, accepted, prompt, step, remaining, noise_block (the engine's block buffers)."""
    tokens, accepted, prompt, step, remaining, noise_block = state
    entropy = decoder.accept_rule == "entropy_budget"
    dummy = tokens
    _decide[(1,)](order if entropy else dummy, sorted_entropy if entropy else pp, excl if entropy else pp,
                  scratch if entropy else dummy, dummy if entropy else top, proposal, pp, tokens, accepted, prompt,
                  step, remaining, noise_block, decoder.entropy_bound if entropy else decoder.prob_threshold,
                  decoder.edit_threshold, decoder.randtok_seed, decoder.random_token_high, tokens.numel(), entropy,
                  num_warps=4, **LAUNCH)


@triton.jit
def _silu_mul(X, Y, H: tl.constexpr, TOTAL: tl.constexpr, N: tl.constexpr):
    """SGLang's SiluAndMul on [rows, 2H] (gate, up): gate * sigmoid(gate) * up in FP32, to BF16."""
    _wait()
    i = tl.program_id(0) * N + tl.arange(0, N)
    offset = (i // H) * (2 * H) + i % H
    a = tl.load(X + offset, i < TOTAL, 0).to(tl.float32)
    b = tl.load(X + offset + H, i < TOTAL, 0).to(tl.float32)
    tl.store(Y + i, a * tl.sigmoid(a) * b, i < TOTAL)
    _release()


def silu_mul(x, out):
    _silu_mul[(triton.cdiv(out.numel(), 1024),)](x, out, out.shape[1], out.numel(), 1024, **LAUNCH)


@triton.jit
def _scaled_add(X, D, scale, TOTAL: tl.constexpr, N: tl.constexpr):
    """x + scale * d in BF16 as torch's two ops round it: bf16(d * fp32(scale)), then bf16(x + that); into X."""
    _wait()
    i = tl.program_id(0) * N + tl.arange(0, N)
    y = (tl.load(D + i, i < TOTAL, 0).to(tl.float32) * scale).to(tl.bfloat16).to(tl.float32)
    tl.store(X + i, (tl.load(X + i, i < TOTAL, 0).to(tl.float32) + y).to(tl.bfloat16), i < TOTAL)
    _release()


def scaled_add_(x, d, scale):
    """x += scale * d (BF16, contiguous, python float scale) with torch's roundings."""
    _scaled_add[(triton.cdiv(x.numel(), 1024),)](x, d, scale, x.numel(), 1024, **LAUNCH)
    return x


@triton.jit
def _block_sum(acc, WARPS: tl.constexpr, LOG2_WARPS: tl.constexpr):
    """BlockReduce's sum of per-thread values acc [WARPS * 32] (thread = warp * 32 + lane): a shfl_down tree within
    each warp (lane l with l + 16, then + 8, ...), then the same tree over the warps' sums, as pairwise adds."""
    acc = tl.reshape(acc, (WARPS, 32))
    for level in tl.static_range(4, -1, -1):
        acc = tl.sum(tl.reshape(acc, (WARPS, 2, 1 << level)), 1)
    acc = tl.reshape(acc, (WARPS,))
    for level in tl.static_range(LOG2_WARPS - 1, -1, -1):
        acc = tl.sum(tl.reshape(acc, (2, 1 << level)), 0)
    return tl.reshape(acc, ())


@triton.jit
def _soft_probs(X, stride_x, OUT, stride_o, V: tl.constexpr, THREADS: tl.constexpr, LOG2_WARPS: tl.constexpr):
    """torch.softmax(x.float(), -1).to(bfloat16) of a BF16 row, bitwise: cunn_SoftMaxForward<ILP 4, float> with
    THREADS threads per row (PyTorch 2.11, aten/src/ATen/native/cuda/SoftMax.cu): thread t sums exp(x - max) over
    its float4 lanes [4t, 4t + 4) of every 4 * THREADS chunk in order, then the tail element t; a shfl_down tree
    within each warp, then over the warps; precise expf; exp(x - max) / sum."""
    _wait()
    row = tl.program_id(0)
    x = X + row.to(tl.int64) * stride_x
    o = OUT + row.to(tl.int64) * stride_o
    t = tl.arange(0, THREADS)
    CHUNK: tl.constexpr = 4 * THREADS
    MAIN: tl.constexpr = V - V % CHUNK
    m = tl.full((THREADS,), float("-inf"), tl.float32)
    for k in range(0, MAIN, CHUNK):
        lanes = tl.load(x + k + 4 * t[:, None] + tl.arange(0, 4)[None, :]).to(tl.float32)
        m = tl.maximum(m, tl.max(lanes, 1))
    tail = MAIN + t
    m = tl.maximum(m, tl.load(x + tail, mask=tail < V, other=float("-inf")).to(tl.float32))
    mx = tl.max(m, 0)
    acc = tl.zeros((THREADS,), tl.float32)
    for k in range(0, MAIN, CHUNK):
        lanes = tl.load(x + k + 4 * t[:, None] + tl.arange(0, 4)[None, :]).to(tl.float32)
        even, odd = tl.split(tl.reshape(lanes, (THREADS, 2, 2)))
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        acc = acc + libdevice.exp(x0 - mx)
        acc = acc + libdevice.exp(x1 - mx)
        acc = acc + libdevice.exp(x2 - mx)
        acc = acc + libdevice.exp(x3 - mx)
    acc = acc + tl.where(tail < V, libdevice.exp(tl.load(x + tail, mask=tail < V, other=0.0).to(tl.float32) - mx), 0.0)
    total = _block_sum(acc, THREADS // 32, LOG2_WARPS)
    for k in range(0, V, CHUNK):
        i = k + tl.arange(0, CHUNK)
        v = tl.load(x + i, mask=i < V, other=0.0).to(tl.float32)
        tl.store(o + i, tl.math.div_rn(libdevice.exp(v - mx), total).to(tl.bfloat16), mask=i < V)
    _release()


def soft_probs(logits, out):
    """The self-conditioning probabilities of BF16 logits rows [rows, vocab] into BF16 out, as SGLang computes them
    (torch.softmax of the rows widened to FP32, then BF16); PyTorch launches 1024 threads per row at this vocabulary."""
    rows, vocab = logits.shape
    threads = min(1024, triton.cdiv(vocab, 32) * 32)
    assert threads == 1024, "the reproduction covers rows of 1024 threads or more"
    _soft_probs[(rows,)](logits, logits.stride(0), out, out.stride(0), vocab, threads, (threads // 32).bit_length() - 1,
                         num_warps=32, **LAUNCH)


SPLIT_TILE = (128, 256, 64)  # BM, BN, BK of the split-K GEMM


@triton.jit
def _split_gemm(A, B, C, PART, COUNT, K, CHUNK, sam, sbk, scm, SPLITS: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """cuBLAS's split-K product, bitwise: program (m tile, n tile, split s) accumulates its K chunk in order (FP32
    MMA), stores the partial, and the last of a tile's programs to finish sums the partials in split order and rounds
    once to BF16. COUNT: one zeroed counter per tile, left zeroed."""
    _wait()
    pm, pn, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(s * CHUNK, tl.minimum(s * CHUNK + CHUNK, K), BK):
        k = k0 + tl.arange(0, BK)
        acc += tl.dot(tl.load(A + m[:, None] * sam + k[None, :]), tl.load(B + k[:, None] * sbk + n[None, :]))
    tile = pm * tl.num_programs(1) + pn
    cell = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(PART + (tile * SPLITS + s) * BM * BN + cell, acc)
    tl.debug_barrier()
    if tl.atomic_add(COUNT + tile, 1, sem="acq_rel") == SPLITS - 1:
        total = tl.load(PART + tile * SPLITS * BM * BN + cell, cache_modifier=".cg")
        for i in range(1, SPLITS):
            total = total + tl.load(PART + (tile * SPLITS + i) * BM * BN + cell, cache_modifier=".cg")
        tl.store(C + m[:, None] * scm + n[None, :], total.to(tl.bfloat16))
        tl.store(COUNT + tile, 0)
    _release()


def split_gemm(a, b, out, split, part, count):
    """out [M, N] = a [M, K] @ b [K, N] (row-major BF16) as cuBLAS computes each M_c-row problem with the split-K
    `split` = (splits, chunk) (M a multiple of M_c and of the tile); part / count: scratch from split_scratch."""
    bm, bn, bk = SPLIT_TILE
    splits, chunk = split
    grid = (a.shape[0] // bm, b.shape[1] // bn, splits)
    _split_gemm[grid](a, b, out, part, count, a.shape[1], chunk, a.stride(0), b.stride(0), out.stride(0), splits, bm,
                      bn, bk, num_warps=8, num_stages=3, **LAUNCH)


def split_scratch(rows, n, splits, device):
    bm, bn, _ = SPLIT_TILE
    tiles = rows // bm * (n // bn)
    return (torch.empty(tiles * splits * bm * bn, device=device, dtype=torch.float32),
            torch.zeros(tiles, device=device, dtype=torch.int32))


def cublas_split(rows, b, max_splits=32):
    """The split-K cuBLAS uses for [rows, K] @ b [K, N] here, as (splits, chunk), found by reproducing its result on
    random probabilities; None if no in-order split matches (or the shapes do not tile)."""
    bm, bn, bk = SPLIT_TILE
    k, n = b.shape
    if rows % bm or n % bn or k % bk:
        return None
    a = torch.softmax(torch.randn(rows, k, device=b.device) * 4, -1).to(torch.bfloat16)
    ref = torch.mm(a, b)
    out = torch.empty_like(ref)
    for splits in range(1, max_splits + 1):
        chunk = triton.cdiv(triton.cdiv(k, splits), bk) * bk
        if triton.cdiv(k, chunk) != splits:
            continue
        split_gemm(a, b, out, (splits, chunk), *split_scratch(rows, n, splits, b.device))
        if torch.equal(out, ref):
            return splits, chunk
    return None
