"""Triton kernels of the DiffusionGemma path (see model.py).

BF16 rounding follows the HF reference wherever it rounds between ops (norm outputs, residual adds,
activation products, RoPE products). Every kernel waits for its predecessor before touching memory and
releases its successor after its last store (PDL), see dlmrt.ops.triton.common.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ...ops.triton.common import LAUNCH, _bf, _geglu_row, _groups, _release, _rms, _row_tile, _store_bf16, _wait


# Philox stream offsets per use, above the 256 x 262144 sampling range.
INIT_STREAM = tl.constexpr(1 << 28)

RENOISE_STREAM = tl.constexpr(1 << 29)


@triton.jit
def _key(seed, block, step):
    """Philox key of one (request seed, block, round)."""
    return (seed * 1000003 + block * 7919 + step * 104729 + 1) & 0x7FFFFFFF


@triton.jit
def _embed(IDS, E, D, Y, scale, eps, N: tl.constexpr, G: tl.constexpr, HAS_D: tl.constexpr, DECODER: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    cols, mask = _row_tile(N, G)
    token = tl.load(IDS + row)
    x = _bf(tl.load(E + token * N + cols, mask, 0.0).to(tl.float32) * scale)
    if DECODER:
        if HAS_D:
            x = _bf(x + tl.load(D + row * N + cols, mask, 0.0).to(tl.float32))
        x = _rms(x, N, eps)
    tl.store(Y + row * N + cols, x.to(tl.bfloat16), mask)
    _release()


def embed(ids, embedding, scale, out, eps, *, decoder, delta=None):
    """E[ids] * scale; the decoder adds the SC delta and applies the unweighted post-norm."""
    n = embedding.shape[1]
    _embed[(ids.numel(),)](ids, embedding, delta if delta is not None else out, out, scale, eps, n, _groups(n),
                           delta is not None, decoder, num_warps=8, **LAUNCH)
    return out


@triton.jit
def _norm(X, W, Y, pre, eps, N: tl.constexpr, G: tl.constexpr, HAS_W: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    cols, mask = _row_tile(N, G)
    x = _bf(tl.load(X + row * N + cols, mask, 0.0).to(tl.float32) * pre)
    y = _rms(x, N, eps)
    if HAS_W:
        y *= tl.load(W + cols, mask, 0.0).to(tl.float32)
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask)
    _release()


def norm(x, weight, eps, *, pre=1.0, out=None):
    """rmsnorm(bf16(x * pre)) [* weight]."""
    m, n = x.shape
    out = torch.empty_like(x) if out is None else out
    _norm[(m,)](x, weight if weight is not None else x, out, pre, eps, n, _groups(n), weight is not None,
                num_warps=8, **LAUNCH)
    return out


@triton.jit
def _rcp(x):
    """rcp.approx.ftz.f32, as TensorRT-LLM's NVFP4 quantization computes its reciprocals."""
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _e2m1x2(lo, hi):
    """One byte of two E2M1 codes (cvt.rn.satfinite, `lo` in the low nibble). Blackwell only."""
    return tl.inline_asm_elementwise("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }",
                                     "=r,f,f", [lo, hi], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _nvfp4_row(x, GS, Q, S, row, N: tl.constexpr, G: tl.constexpr):
    """FlashInfer's fp4_quantize of one row x [G, 128] (BF16 values in FP32), bit for bit: TensorRT-LLM's fast-math
    NVFP4 recipe with global scale *GS, E2M1 codes to Q [rows, N / 2] (low nibble first) and E4M3 scales per 16 to
    S [rows, N / 16] (linear layout)."""
    blocks = tl.reshape(x, [G, 8, 16])
    vmax = tl.max(tl.abs(blocks), 2)
    gs = tl.load(GS)
    sf = (gs * (vmax * _rcp(tl.full([G, 8], 6.0, tl.float32)))).to(tl.float8e4nv)
    scale = tl.where(vmax != 0, _rcp(sf.to(tl.float32) * _rcp(tl.full([G, 8], gs, tl.float32))), 0.0)
    lo, hi = tl.split(tl.reshape(blocks * scale[:, :, None], [G, 64, 2]))
    g = tl.arange(0, G)[:, None]
    q = g * 64 + tl.arange(0, 64)[None, :]
    tl.store(Q + row * (N // 2) + q, _e2m1x2(lo, hi).to(tl.uint8), q < N // 2)
    s = g * 8 + tl.arange(0, 8)[None, :]
    tl.store(S + row * (N // 16) + s, sf.to(tl.uint8, bitcast=True), s < N // 16)


@triton.jit
def _post_attn(H, A, WA, W1, W2, RS, X1, X2, R, GS, Q, S, eps, rscale, N: tl.constexpr, G: tl.constexpr,
               FP4: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    cols, mask = _row_tile(N, G)
    # Every load is issued before the first row reduction: each reduction ends in a barrier that later loads
    # would otherwise queue behind (one L2 round trip per phase).
    x = tl.load(A + row * N + cols, mask, 0.0).to(tl.float32)
    wa = tl.load(WA + cols, mask, 0.0).to(tl.float32)
    h0 = tl.load(H + row * N + cols, mask, 0.0).to(tl.float32)
    w1 = tl.load(W1 + cols, mask, 0.0).to(tl.float32)
    w2 = tl.load(W2 + cols, mask, 0.0).to(tl.float32)
    rs = tl.load(RS + cols, mask, 0.0).to(tl.float32)
    h = _bf(h0 + _bf(_rms(x, N, eps) * wa))
    tl.store(H + row * N + cols, h.to(tl.bfloat16), mask)
    n = _rms(h, N, eps)
    tl.store(X1 + row * N + cols, (n * w1).to(tl.bfloat16), mask)
    if FP4:
        _nvfp4_row((n * w2).to(tl.bfloat16).to(tl.float32), GS, Q, S, row, N, G)
    else:
        tl.store(X2 + row * N + cols, (n * w2).to(tl.bfloat16), mask)
    tl.store(R + row * N + cols, (_bf(_bf(n) * rs) * rscale).to(tl.bfloat16), mask)
    _release()


def post_attn(h, attn, w_post, w_mlp, w_moe, router_scale, x_mlp, x_moe, router_in, eps, fp4=None):
    """h += norm(attn); dense-MLP input, expert input and router input (unweighted norm * scale / sqrt(H)).
    `fp4`: (global scale, codes, scales) to write the expert input NVFP4-quantized instead (see _nvfp4_row)."""
    m, n = h.shape
    gs, q, s = fp4 if fp4 is not None else (h, h, h)
    _post_attn[(m,)](h, attn, w_post, w_mlp, w_moe, router_scale, x_mlp, x_moe, router_in, gs, q, s, eps, n ** -0.5,
                     n, _groups(n), fp4 is not None, num_warps=8, **LAUNCH)


@triton.jit
def _post_ffn(H, Y1, Y2, W1, W2, W, WN, X, scalar, eps, N: tl.constexpr, G: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    cols, mask = _row_tile(N, G)
    y1 = tl.load(Y1 + row * N + cols, mask, 0.0).to(tl.float32)
    y2 = tl.load(Y2 + row * N + cols, mask, 0.0).to(tl.float32)
    w1 = tl.load(W1 + cols, mask, 0.0).to(tl.float32)
    w2 = tl.load(W2 + cols, mask, 0.0).to(tl.float32)
    w = tl.load(W + cols, mask, 0.0).to(tl.float32)
    h0 = tl.load(H + row * N + cols, mask, 0.0).to(tl.float32)
    wn = tl.load(WN + cols, mask, 0.0).to(tl.float32)
    n1 = _bf(_rms(y1, N, eps) * w1)
    n2 = _bf(_rms(y2, N, eps) * w2)
    t = _bf(_rms(_bf(n1 + n2), N, eps) * w)
    h = _bf(_bf(h0 + t) * scalar)
    tl.store(H + row * N + cols, h.to(tl.bfloat16), mask)
    tl.store(X + row * N + cols, (_rms(h, N, eps) * wn).to(tl.bfloat16), mask)
    _release()


def post_ffn(h, y_mlp, y_moe, w_mlp, w_moe, w_post, w_next, out, scalar, eps):
    """h = (h + norm(norm(y_mlp) + norm(y_moe))) * layer scalar; out = the next norm of h."""
    m, n = h.shape
    _post_ffn[(m,)](h, y_mlp, y_moe, w_mlp, w_moe, w_post, w_next, out, scalar, eps, n, _groups(n),
                    num_warps=8, **LAUNCH)


@triton.jit
def _geglu(X, Y, F: tl.constexpr, G: tl.constexpr):
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    _geglu_row(X, Y, row, row, part, F, G)
    _release()


def geglu(x, out=None):
    """gelu_tanh(gate) * up for x = [gate | up] along columns."""
    m, f = x.shape[0], x.shape[1] // 2
    out = torch.empty((m, f), device=x.device, dtype=torch.bfloat16) if out is None else out
    _geglu[(m, triton.cdiv(f, 1024))](x, out, f, 8, num_warps=4, **LAUNCH)
    return out


@triton.jit
def _norm_rope(a, b, W, cos, sin, lo, hi, heads, OUT, eps, D: tl.constexpr, ROPE: tl.constexpr,
               FP8: tl.constexpr = False):
    """Per-head RMSNorm [* weight] then NeoX RoPE with BF16 products, as in the reference; `a` / `b` are the heads'
    two halves in FP32."""
    r = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1))[:, None] / D + eps)
    if ROPE:
        na = _bf(a * r * tl.load(W + lo).to(tl.float32))
        nb = _bf(b * r * tl.load(W + hi).to(tl.float32))
        _store_bf16(OUT + heads + lo, _bf(na * cos) + _bf(-nb * sin), FP8)
        _store_bf16(OUT + heads + hi, _bf(nb * cos) + _bf(na * sin), FP8)
    else:
        _store_bf16(OUT + heads + lo, a * r, FP8)
        _store_bf16(OUT + heads + hi, b * r, FP8)


@triton.jit
def _qkv_rope(QKV, QW, KW, COS, SIN, START, Q, K, V, eps, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
              HAS_V: tl.constexpr, KV_FP8: tl.constexpr, QH: tl.constexpr):
    """One row: all its heads (QH == HQ), or, per program along axis 1, QH of its query heads / its K and V."""
    _wait()
    row = tl.program_id(0)
    part = tl.program_id(1)
    pos = tl.load(START) + row
    half = tl.arange(0, D // 2)
    lo, hi = half[None, :], half[None, :] + D // 2
    width = HQ + (2 if HAS_V else 1) * HK
    src = QKV + row * width * D
    hk = tl.arange(0, HK)[:, None] * D
    # In each part the row's own values are loaded before the RoPE tables, whose addresses wait on START: loads
    # issue in order, so the other way round every load would queue behind the START round trip.
    if part * QH < HQ:
        hq = (part * QH + tl.arange(0, QH))[:, None] * D
        a = tl.load(src + hq + lo).to(tl.float32)
        b = tl.load(src + hq + hi).to(tl.float32)
        cos = tl.load(COS + pos * (D // 2) + half).to(tl.float32)[None, :]
        sin = tl.load(SIN + pos * (D // 2) + half).to(tl.float32)[None, :]
        _norm_rope(a, b, QW, cos, sin, lo, hi, hq, Q + row * HQ * D, eps, D, True)
    if QH == HQ or part * QH == HQ:
        # Full-attention layers have no value projection: the value is the unweighted norm of the raw key.
        v_src = src + (HQ + HK) * D if HAS_V else src + HQ * D
        ka = tl.load(src + HQ * D + hk + lo).to(tl.float32)
        kb = tl.load(src + HQ * D + hk + hi).to(tl.float32)
        va = tl.load(v_src + hk + lo).to(tl.float32)
        vb = tl.load(v_src + hk + hi).to(tl.float32)
        cos = tl.load(COS + pos * (D // 2) + half).to(tl.float32)[None, :]
        sin = tl.load(SIN + pos * (D // 2) + half).to(tl.float32)[None, :]
        _norm_rope(ka, kb, KW, cos, sin, lo, hi, hk, K + pos.to(tl.int64) * HK * D, eps, D, True, KV_FP8)
        _norm_rope(va, vb, QW, cos, sin, lo, hi, hk, V + pos.to(tl.int64) * HK * D, eps, D, False, KV_FP8)
    _release()


def qkv_rope(qkv, q_weight, k_weight, cos, sin, start, q, k, v, heads, kv_heads, head_dim, has_v, eps,
             kv_fp8=False, split=False):
    """Q/K norm + RoPE, V norm; Q to `q`, K/V into the arena at [start, start + rows) (FP8-rounded with kv_fp8).
    `split`: 4 query heads per program and one program for K/V, rather than one program per row."""
    qh = min(4, heads) if split else heads
    grid = (qkv.shape[0], heads // qh + 1 if split else 1)
    _qkv_rope[grid](qkv, q_weight, k_weight, cos, sin, start, q, k, v, eps, heads, kv_heads, head_dim, has_v, kv_fp8,
                    qh, num_warps=4 if split or head_dim <= 256 else 8, **LAUNCH)
    return q


@triton.jit
def _logit_parts(X, STEP, SEED, BLOCK, TEMP, PART, cap, t_min, t_max, steps, VOCAB: tl.constexpr,
                 OFFSET: tl.constexpr, TOTAL: tl.constexpr, PARTS: tl.constexpr, N: tl.constexpr,
                 ADAPTIVE: tl.constexpr, RAND4: tl.constexpr):
    """Per (row, vocabulary part), at z = softcap(logits) / T: max, sum exp, centered entropy term, argmax,
    Gumbel-max sample; and max / sum exp of bf16(z), the normalizer of the next round's SC softmax.
    X holds this rank's VOCAB columns starting at global column OFFSET of TOTAL; indices and noise are global.
    T follows the linear schedule, or is this round's adaptive temperature in TEMP. RAND4: one Philox4x32 call
    draws the noise of four consecutive columns (a different stream than one call per column)."""
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * N + tl.arange(0, N)
    live = col < VOCAB
    step = tl.load(STEP)
    if ADAPTIVE:
        t = tl.load(TEMP)
    else:
        t = t_min + (t_max - t_min) * (steps - step).to(tl.float32) / steps
    x = tl.load(X + row * VOCAB + col, live, 0.0).to(tl.float32)
    z = tl.where(live, cap * libdevice.tanh(x / cap) / t, float("-inf"))
    m = tl.max(z, 0)
    e = tl.exp(z - m)
    s = tl.sum(e, 0)
    c = tl.sum(tl.where(live, e * (z - m), 0.0), 0)
    arg = tl.min(tl.where(z == m, col + OFFSET, 2147483647), 0)
    zb = _bf(z)
    mb = tl.max(zb, 0)
    sb = tl.sum(tl.exp(zb - mb), 0)
    key = _key(tl.load(SEED), tl.load(BLOCK), step)
    if RAND4:
        q = (row * TOTAL + OFFSET + part * N) // 4 + tl.arange(0, N // 4)
        r0, r1, r2, r3 = tl.rand4x(key, q)
        u = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), [N])
    else:
        u = tl.rand(key, row * TOTAL + col + OFFSET)
    g = tl.where(live, z - tl.log(-tl.log(tl.maximum(u, 1e-20))), float("-inf"))
    gm = tl.max(g, 0)
    gi = tl.min(tl.where(g == gm, col + OFFSET, 2147483647), 0)
    base = PART + (row * PARTS + part) * 8
    tl.store(base, m)
    tl.store(base + 1, s)
    tl.store(base + 2, c)
    tl.store(base + 3, arg.to(tl.float32, bitcast=True))
    tl.store(base + 4, mb)
    tl.store(base + 5, sb)
    tl.store(base + 6, gm)
    tl.store(base + 7, gi.to(tl.float32, bitcast=True))
    _release()


@triton.jit
def _logit_finish(PART, STATS, ARGMAX, SAMPLE, PARTS: tl.constexpr, RANKS: tl.constexpr, STRIDE: tl.constexpr,
                  N: tl.constexpr):
    """Combine the per-part records of every vocabulary shard (`RANKS` blocks of `STRIDE` floats)."""
    _wait()
    row = tl.program_id(0)
    j = tl.arange(0, N)
    live = j < RANKS * PARTS
    base = PART + (j // PARTS) * STRIDE + (row * PARTS + j % PARTS) * 8
    m = tl.load(base, live, float("-inf"))
    s = tl.load(base + 1, live, 0.0)
    c = tl.load(base + 2, live, 0.0)
    arg = tl.load(base + 3, live, 0.0).to(tl.int32, bitcast=True)
    mb = tl.load(base + 4, live, float("-inf"))
    sb = tl.load(base + 5, live, 0.0)
    gm = tl.load(base + 6, live, float("-inf"))
    gi = tl.load(base + 7, live, 0.0).to(tl.int32, bitcast=True)
    top = tl.max(m, 0)
    f = tl.where(live, tl.exp(m - top), 0.0)
    total = tl.sum(s * f, 0)
    centered = tl.sum(f * (c + s * tl.where(live, m - top, 0.0)), 0)
    top_b = tl.max(mb, 0)
    tl.store(STATS + row * 2, top_b + tl.log(tl.sum(tl.where(live, sb * tl.exp(mb - top_b), 0.0), 0)))
    tl.store(STATS + row * 2 + 1, tl.maximum(0.0, tl.log(total) - centered / total))
    tl.store(ARGMAX + row, tl.min(tl.where(live & (m == top), arg, 2147483647), 0).to(tl.int64))
    g_top = tl.max(gm, 0)
    tl.store(SAMPLE + row, tl.min(tl.where(live & (gm == g_top), gi, 2147483647), 0).to(tl.int64))
    _release()


def logit_stats(logits, step, seed, block, config, stats, argmax, sample, cap, *, offset=0, group=None, temp=None,
                rand4=False):
    """Per canvas row: SC log-normalizer and entropy [rows, 2], argmax and a sample of softmax(z).

    With a `group`, `logits` holds this rank's vocabulary columns starting at `offset`; the per-part records
    of all ranks are gathered in rank order so every rank reaches the same result. With an adaptive decoder
    the temperature is read from `temp` (see `adaptive_temperature`). `rand4`: four columns per Philox call.
    """
    rows, vocab = logits.shape
    ranks = group.world if group is not None else 1
    n = 2048
    parts = triton.cdiv(vocab, n)
    stride = triton.cdiv(rows * parts * 8, 4) * 4  # 16-byte multiple for the multicast gather
    scratch = torch.empty(stride, device=logits.device, dtype=torch.float32)
    _logit_parts[(rows, parts)](logits, step, seed, block, temp if config.adaptive else step, scratch, cap,
                                config.t_min, config.t_max, config.max_denoising_steps, vocab, offset, vocab * ranks,
                                parts, n, config.adaptive, rand4, num_warps=8, **LAUNCH)
    if group is not None:
        scratch = group.all_gather(scratch, "logit_stats")
    _logit_finish[(rows,)](scratch, stats, argmax, sample, parts, ranks, stride,
                           triton.next_power_of_2(parts * ranks), **LAUNCH)


@triton.jit
def _sc_probs(X, STATS, STEP, TEMP, Y, cap, t_min, t_max, steps, V: tl.constexpr, BLOCK: tl.constexpr,
              ADAPTIVE: tl.constexpr):
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    mask = cols < V
    step = tl.load(STEP)
    # The previous round's temperature (TEMP is updated later in this round); round 0 has no SC signal.
    if ADAPTIVE:
        t = tl.load(TEMP)
    else:
        t = t_min + (t_max - t_min) * (steps - step + 1).to(tl.float32) / steps
    x = tl.load(X + row * V + cols, mask & (step > 0), 0.0).to(tl.float32)
    z = _bf(cap * libdevice.tanh(x / cap) / t)
    p = tl.where(step > 0, tl.exp(z - tl.load(STATS + row * 2)), 0.0)
    tl.store(Y + row * V + cols, p.to(tl.bfloat16), mask)
    _release()


def sc_probs(logits, stats, step, config, out, cap, temp=None):
    """softmax(bf16(softcap(previous logits) / T_previous)) in BF16; zeros in round 0."""
    rows, v = logits.shape
    _sc_probs[(rows, triton.cdiv(v, 4096))](logits, stats, step, temp if config.adaptive else step, out, cap,
                                            config.t_min, config.t_max, config.max_denoising_steps, v, 4096,
                                            config.adaptive, num_warps=8, **LAUNCH)
    return out


@triton.jit
def _raw_parts(X, PART, cap, VOCAB: tl.constexpr, PARTS: tl.constexpr, N: tl.constexpr):
    """Per (row, vocabulary part) of softcap(logits) at temperature one: max, sum exp, centered entropy term."""
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * N + tl.arange(0, N)
    live = col < VOCAB
    x = tl.load(X + row * VOCAB + col, live, 0.0).to(tl.float32)
    z = tl.where(live, cap * libdevice.tanh(x / cap), float("-inf"))
    m = tl.max(z, 0)
    e = tl.exp(z - m)
    base = PART + (row * PARTS + part) * 4
    tl.store(base, m)
    tl.store(base + 1, tl.sum(e, 0))
    tl.store(base + 2, tl.sum(tl.where(live, e * (z - m), 0.0), 0))
    _release()


@triton.jit
def _raw_finish(PART, H, log_v, PARTS: tl.constexpr, RANKS: tl.constexpr, STRIDE: tl.constexpr, N: tl.constexpr):
    """Normalized entropy of one row from the part records of every vocabulary shard."""
    _wait()
    row = tl.program_id(0)
    j = tl.arange(0, N)
    live = j < RANKS * PARTS
    base = PART + (j // PARTS) * STRIDE + (row * PARTS + j % PARTS) * 4
    m = tl.load(base, live, float("-inf"))
    s = tl.load(base + 1, live, 0.0)
    c = tl.load(base + 2, live, 0.0)
    top = tl.max(m, 0)
    f = tl.where(live, tl.exp(m - top), 0.0)
    total = tl.sum(s * f, 0)
    centered = tl.sum(f * (c + s * tl.where(live, m - top, 0.0)), 0)
    tl.store(H + row, tl.maximum(0.0, tl.log(total) - centered / total) / log_v)
    _release()


@triton.jit
def _temperature(H, ACCEPTED, STEP, TEMP, HREF, TPREV, t_min, t_max, power, B: tl.constexpr):
    """This round's adaptive temperature from the mean entropy of the positions the previous round did not accept
    (all positions if it accepted every one); round 0 runs at t_max and sets the reference entropy."""
    _wait()
    i = tl.arange(0, B)
    h = tl.load(H + i)
    scope = tl.load(ACCEPTED + i) == 0
    scope = scope | (tl.sum(scope.to(tl.int32), 0) == 0)
    h_scope = tl.sum(tl.where(scope, h, 0.0), 0) / tl.maximum(tl.sum(scope.to(tl.int32), 0), 1)
    first = tl.load(STEP) == 0
    h_ref = tl.where(first, tl.maximum(h_scope, 1e-8), tl.load(HREF))
    ratio = tl.minimum(tl.maximum(h_scope / tl.maximum(h_ref, 1e-8), 0.0), 1.0)
    proposed = t_min + (t_max - t_min) * tl.exp(power * tl.log(ratio))
    t = tl.where(first, t_max, tl.minimum(tl.load(TPREV), proposed))
    t = tl.minimum(tl.maximum(t, t_min), t_max)
    tl.store(TEMP, t)
    tl.store(HREF, h_ref)
    tl.store(TPREV, t)
    _release()


def adaptive_temperature(logits, step, accepted, temp, h_ref, t_prev, config, cap, *, group=None):
    """Adaptive-scalar temperature of this round into `temp` (state in `h_ref`, `t_prev`; scope from `accepted`)."""
    rows, vocab = logits.shape
    ranks = group.world if group is not None else 1
    n = 2048
    parts = triton.cdiv(vocab, n)
    stride = rows * parts * 4
    scratch = torch.empty(stride, device=logits.device, dtype=torch.float32)
    _raw_parts[(rows, parts)](logits, scratch, cap, vocab, parts, n, num_warps=8, **LAUNCH)
    if group is not None:
        scratch = group.all_gather(scratch, "raw_entropy")
    entropy = torch.empty(rows, device=logits.device, dtype=torch.float32)
    _raw_finish[(rows,)](scratch, entropy, math.log(max(vocab * ranks, 2)), parts, ranks, stride,
                         triton.next_power_of_2(parts * ranks), **LAUNCH)
    _temperature[(1,)](entropy, accepted, step, temp, h_ref, t_prev, config.t_min, config.t_max,
                       config.adaptive_power, rows, num_warps=4, **LAUNCH)


@triton.jit
def _begin(P, TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, window, VOCAB: tl.constexpr,
           B: tl.constexpr):
    _wait()
    i = tl.arange(0, B)
    offset, ordinal, seed = tl.load(P), tl.load(P + 1), tl.load(P + 2)
    noise = tl.randint(_key(seed, ordinal, 0), INIT_STREAM + i) % VOCAB
    tl.store(TOKENS + i, noise.to(tl.int64))
    tl.store(OUT + i, tl.full((B,), -1, tl.int64))
    tl.store(ACCEPTED + i, tl.zeros((B,), tl.int8))
    tl.store(START, offset)
    tl.store(END, offset + B)
    tl.store(LEFTPAD, tl.maximum(offset - (window - 1), 0))
    tl.store(STEP, 0)
    tl.store(REMAINING, 1)
    tl.store(SEED, seed)
    tl.store(BLOCK, ordinal)
    _release()


def begin(params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block, window, vocab):
    """Canvas of uniform random tokens at [offset, offset + 256); argmax history and accepted mask cleared."""
    _begin[(1,)](params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block, window, vocab,
                 tokens.numel(), **LAUNCH)


@triton.jit
def _decide(STATS, ARGMAX, SAMPLE, TOKENS, OUT, ACCEPTED, STEP, REMAINING, SEED, BLOCK, bound, confidence,
            VOCAB: tl.constexpr, B: tl.constexpr, ADAPTIVE: tl.constexpr):
    _wait()
    i = tl.arange(0, B)
    step = tl.load(STEP)
    entropy = tl.load(STATS + i * 2 + 1)
    # Nonnegative float bits sort numerically; the index breaks ties deterministically.
    key = (entropy.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | i.to(tl.uint64)
    order = (tl.sort(key, descending=False) & 0xFFFFFFFF).to(tl.int32)
    h = tl.gather(entropy, order, 0)
    take = (tl.cumsum(h, 0) - h) <= bound
    noise = tl.randint(_key(tl.load(SEED), tl.load(BLOCK), step), RENOISE_STREAM + order) % VOCAB
    tl.store(TOKENS + order, tl.where(take, tl.load(SAMPLE + order), noise.to(tl.int64)))
    if ADAPTIVE:  # the next round's temperature averages over the positions not accepted here
        tl.store(ACCEPTED + order, take.to(tl.int8))
    argmax = tl.load(ARGMAX + i)
    stable = (step > 0) & (tl.sum((argmax != tl.load(OUT + i)).to(tl.int32), 0) == 0)
    confident = tl.sum(entropy, 0) / B < confidence
    tl.store(OUT + i, argmax)
    tl.store(REMAINING, tl.where(stable & confident, 0, 1))
    tl.store(STEP, step + 1)
    _release()


def decide(stats, argmax, sample, tokens, out, accepted, step, remaining, seed, block, config, vocab):
    """Keep samples at the lowest-entropy positions within the entropy bound, re-noise the rest; stop once the
    argmax canvas is unchanged and the mean entropy is below the confidence threshold."""
    _decide[(1,)](stats, argmax, sample, tokens, out, accepted, step, remaining, seed, block, config.entropy_bound,
                  config.confidence_threshold, vocab, tokens.numel(), config.adaptive, num_warps=4, **LAUNCH)
