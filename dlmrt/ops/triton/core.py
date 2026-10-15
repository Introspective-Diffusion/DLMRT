"""Triton kernels between the GEMMs and attention: RMSNorm, SwiGLU, FP8 row quantization, Q/K norm + RoPE,
embedding lookup, and the diffusion-mode frame trace.

Row kernels own one block of rows. FP8 outputs use 1x128 groups along K with fp32 scales stored K-group-major
([K/128, M]), the layout DeepGEMM consumes.
"""

import torch
import triton
import triton.language as tl

from .common import LAUNCH, _release, _wait


GROUP = 128


@triton.jit
def _store_row(Y, S, y, row, g, cols, mask, M, N: tl.constexpr, FP8: tl.constexpr):
    """Store a [groups, 128] tile of one row as BF16, or as FP8 with one scale per 128 columns."""
    if FP8:
        s = tl.maximum(tl.max(tl.abs(y), 1), 1e-12) / 448.0
        tl.store(Y + row * N + cols, (y / s[:, None]).to(tl.float8e4nv), mask)
        tl.store(S + g * M + row, s, g < N // 128)
    else:
        tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask)


@triton.jit
def _rmsnorm(H, D, W, Y, S, M, eps, scale, N: tl.constexpr, G: tl.constexpr,
             HAS_D: tl.constexpr, HAS_W: tl.constexpr, FP8: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    g = tl.arange(0, G)
    cols = g[:, None] * 128 + tl.arange(0, 128)[None, :]
    mask = cols < N
    h = tl.load(H + row * N + cols, mask, 0.0).to(tl.float32)
    if HAS_D:
        h += tl.load(D + row * N + cols, mask, 0.0).to(tl.float32)
        tl.store(H + row * N + cols, h.to(tl.bfloat16), mask)
    y = h * (tl.rsqrt(tl.sum(tl.sum(h * h, 1), 0) / N + eps) * scale)
    if HAS_W:
        y *= tl.load(W + cols, mask, 0.0).to(tl.float32)
    _store_row(Y, S, y, row, g, cols, mask, M, N, FP8)
    _release()


def rmsnorm(h, weight, eps, *, delta=None, scale=1.0, out=None, scales=None):
    """y = rmsnorm(h [+= delta]) * weight * scale; FP8 when `out` is float8 (then `scales` is required)."""
    m, n = h.shape
    fp8 = out is not None and out.dtype == torch.float8_e4m3fn
    out = torch.empty_like(h) if out is None else out
    _rmsnorm[(m,)](h, delta if delta is not None else h, weight if weight is not None else h, out,
                   scales if fp8 else h, m, eps, scale, n, triton.next_power_of_2(triton.cdiv(n, 128)),
                   delta is not None, weight is not None, fp8, num_warps=8, **LAUNCH)
    return out


@triton.jit
def _swiglu(X, Y, S, M, F: tl.constexpr, G: tl.constexpr, FP8: tl.constexpr, ROUND: tl.constexpr):
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    g = part * G + tl.arange(0, G)
    cols = g[:, None] * 128 + tl.arange(0, 128)[None, :]
    mask = cols < F
    gate = tl.load(X + row * 2 * F + cols, mask, 0.0).to(tl.float32)
    up = tl.load(X + row * 2 * F + F + cols, mask, 0.0).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    if ROUND:
        silu = silu.to(tl.bfloat16).to(tl.float32)
    _store_row(Y, S, silu * up, row, g, cols, mask, M, F, FP8)
    _release()


def swiglu(x, *, out=None, scales=None, round_silu=False):
    """silu(gate) * up for x = [gate | up] along columns; `round_silu` rounds silu to BF16 first."""
    m, f = x.shape[0], x.shape[1] // 2
    fp8 = out is not None and out.dtype == torch.float8_e4m3fn
    out = torch.empty((m, f), device=x.device, dtype=torch.bfloat16) if out is None else out
    _swiglu[(m, triton.cdiv(f, 1024))](x, out, scales if fp8 else x, m, f, 8, fp8, round_silu,
                                       num_warps=4, **LAUNCH)
    return out


@triton.jit
def _quant(X, Y, S, M, K: tl.constexpr, G: tl.constexpr):
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    g = part * G + tl.arange(0, G)
    cols = g[:, None] * 128 + tl.arange(0, 128)[None, :]
    mask = cols < K
    _store_row(Y, S, tl.load(X + row * K + cols, mask, 0.0).to(tl.float32), row, g, cols, mask, M, K, True)
    _release()


def quant(x, out, scales):
    m, k = x.shape
    _quant[(m, triton.cdiv(k, 1024))](x, out, scales, m, k, 8, num_warps=4, **LAUNCH)
    return out


@triton.jit
def _rope_heads(X, W, heads, lo, hi, cos, sin, OUT, eps, D: tl.constexpr):
    """Per-head RMSNorm (rounded to BF16) then RoPE for a [heads, D] tile, split into its two halves."""
    a = tl.load(X + heads + lo).to(tl.float32)
    b = tl.load(X + heads + hi).to(tl.float32)
    r = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1))[:, None] / D + eps)
    na = (a * r * tl.load(W + lo).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    nb = (b * r * tl.load(W + hi).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    # The sine product is rounded before the cosine FMA, as in the standalone rotary kernel.
    tl.store(OUT + heads + lo, tl.fma(na, cos, -(nb * sin)).to(tl.bfloat16))
    tl.store(OUT + heads + hi, tl.fma(nb, cos, na * sin).to(tl.bfloat16))


@triton.jit
def _qk_rope(QKV, QW, KW, ROPE, START, Q, K, V, eps, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    pos = tl.load(START) + row
    half = tl.arange(0, D // 2)
    lo, hi = half[None, :], half[None, :] + D // 2
    cos = tl.load(ROPE + pos * D + half)[None, :]
    sin = tl.load(ROPE + pos * D + D // 2 + half)[None, :]
    src = QKV + row * (HQ + 2 * HK) * D
    hq = tl.arange(0, HQ)[:, None] * D
    hk = tl.arange(0, HK)[:, None] * D
    _rope_heads(src, QW, hq, lo, hi, cos, sin, Q + row * HQ * D, eps, D)
    _rope_heads(src + HQ * D, KW, hk, lo, hi, cos, sin, K + pos * HK * D, eps, D)
    hv = hk + tl.arange(0, D)[None, :]
    tl.store(V + pos * HK * D + hv, tl.load(src + (HQ + HK) * D + hv))
    _release()


def qk_rope(qkv, q_weight, k_weight, rope, start, q, k, v, heads, kv_heads, eps):
    """Q/K RMSNorm + RoPE for one block; Q to `q`, K/V into the arena at [start, start + rows)."""
    _qk_rope[(qkv.shape[0],)](qkv, q_weight, k_weight, rope, start, q, k, v, eps, heads, kv_heads, 128,
                              num_warps=4, **LAUNCH)
    return q


@triton.jit
def _embed_norm(IDS, E, D, Y, eps, delta_scale, out_scale, N: tl.constexpr, BLOCK: tl.constexpr,
                HAS_D: tl.constexpr, NORM: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    token = tl.load(IDS + row)
    x = tl.load(E + token * N + cols, mask, 0.0).to(tl.float32)
    # BF16 rounding after the SC add and after the unweighted norm follows the trained adapter.
    if HAS_D:
        x = (x + delta_scale * tl.load(D + row * N + cols, mask, 0.0).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    if NORM:
        x = (x * tl.rsqrt(tl.sum(x * x, 0) / N + eps)).to(tl.bfloat16).to(tl.float32) * out_scale
    tl.store(Y + row * N + cols, x.to(tl.bfloat16), mask)
    _release()


def embed_norm(ids, embedding, out, eps, *, delta=None, delta_scale=1.0, norm=True, out_scale=1.0):
    """Block input: E[ids] (+ delta_scale * SC delta), then the SC post-norm."""
    n = embedding.shape[1]
    _embed_norm[(ids.numel(),)](ids, embedding, delta if delta is not None else out, out, eps,
                                delta_scale, out_scale, n, triton.next_power_of_2(n),
                                delta is not None, norm, num_warps=8, **LAUNCH)
    return out


@triton.jit
def _trace(TOKENS, STEP, FRAMES, LIMIT, B: tl.constexpr, N: tl.constexpr):
    _wait()
    i = tl.arange(0, N)
    s = tl.minimum(tl.load(STEP), LIMIT - 1)
    tl.store(FRAMES + s * B + i, tl.load(TOKENS + i, i < B, 0), i < B)
    _release()


def trace(tokens, step, frames):
    """frames[step] = tokens: the block a denoise round starts from, random tokens included (diffusion mode)."""
    b = tokens.numel()
    _trace[(1,)](tokens, step, frames, frames.shape[0], b, triton.next_power_of_2(b), num_warps=4, **LAUNCH)
