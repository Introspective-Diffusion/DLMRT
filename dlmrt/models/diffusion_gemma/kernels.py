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
from .config import STOP_RULES

# Philox stream offsets per use, above the 256 x 262144 sampling range.
INIT_STREAM = tl.constexpr(1 << 28)

RENOISE_STREAM = tl.constexpr(1 << 29)

# A batched request's stage (per KV slot), see batched.py.
DENOISE, COMMIT, PREFILL, DONE, IDLE, ADMIT, SWITCH = (tl.constexpr(i) for i in range(7))

# `decide`'s STOP code of each GemmaDecode.stop_rule.
STOPS = {name: code for code, name in enumerate(STOP_RULES)}


def _classes(classes, stops):
    """Canvas lengths and their stop rules as four constexpr pairs (unused classes 0)."""
    if not 0 < len(classes) <= 4 or len(stops) != len(classes):
        raise ValueError("one to four canvas lengths, one stop rule each")
    pad = 4 - len(classes)
    return tuple(classes) + (0,) * pad, tuple(stops) + (0,) * pad


@triton.jit
def _key(seed, block, step):
    """Philox key of one (request seed, block, round)."""
    return (seed * 1000003 + block * 7919 + step * 104729 + 1) & 0x7FFFFFFF


@triton.jit
def _embed(IDS, E, D, Y, MODES, SLOTS, STEP, INDEX, START, END, LEFT, STATE, scale, eps, N: tl.constexpr,
           G: tl.constexpr, HAS_D: tl.constexpr, DECODER: tl.constexpr, MIXED: tl.constexpr = False,
           LANES: tl.constexpr = False, HAS_STATE: tl.constexpr = False):
    _wait()
    row = tl.program_id(0)
    cols, mask = _row_tile(N, G)
    token = tl.load(IDS + row)
    x = _bf(tl.load(E + token * N + cols, mask, 0.0).to(tl.float32) * scale)
    if MIXED:
        # Row r of the request at slot SLOTS[r]: a canvas (mode DENOISE) takes the decoder path, other modes the plain
        # one. Round 0's SC delta is zero (no previous logits), so it is not read (the rows carry a stale one).
        # LANES: rows at a negative INDEX are a commit lane (causal rows before the canvas), plain too.
        slot = tl.load(SLOTS + row)
        mode = tl.load(MODES + slot)
        if HAS_STATE:
            # The slot's (start, end, left, mode) at the row, read by the forward's RoPE and attention in one load.
            tl.store(STATE + row * 4, tl.load(START + slot))
            tl.store(STATE + row * 4 + 1, tl.load(END + slot))
            tl.store(STATE + row * 4 + 2, tl.load(LEFT + slot))
            tl.store(STATE + row * 4 + 3, mode)
        canvas = mode == DENOISE
        if LANES:
            canvas = canvas & (tl.load(INDEX + row) >= 0)
        if canvas:
            if tl.load(STEP + slot) > 0:
                x = _bf(x + tl.load(D + row * N + cols, mask, 0.0).to(tl.float32))
            x = _rms(x, N, eps)
    elif DECODER:
        if HAS_D:
            x = _bf(x + tl.load(D + row * N + cols, mask, 0.0).to(tl.float32))
        x = _rms(x, N, eps)
    tl.store(Y + row * N + cols, x.to(tl.bfloat16), mask)
    _release()


def embed(ids, embedding, scale, out, eps, *, decoder, delta=None, modes=None, slots=None, step=None, index=None,
          state=None):
    """E[ids] * scale; the decoder adds the SC delta and applies the unweighted post-norm. `modes` and `step` (per KV
    slot, with `slots` per row): the decoder path for the rows of requests in mode DENOISE only, the delta from
    round 1 on. `index` (per row): rows at a negative index (commit lanes) take the plain path. `state` (with
    `modes`): (start, end, left per slot, out [rows, 4] int32) to gather each row's (start, end, left, mode) into."""
    n = embedding.shape[1]
    start, end, left, rows = state if state is not None else (ids,) * 4
    _embed[(ids.numel(),)](ids, embedding, delta if delta is not None else out, out,
                           modes if modes is not None else ids, slots if slots is not None else ids,
                           step if step is not None else ids, index if index is not None else ids, start, end, left,
                           rows, scale, eps, n, _groups(n), delta is not None, decoder, modes is not None,
                           index is not None, state is not None, num_warps=8, **LAUNCH)
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
def _qkv_rope(QKV, QW, KW, COS, SIN, START, Q, K, V, SLOTS, INDEX, STATE, S, eps, HQ: tl.constexpr,
              HK: tl.constexpr, D: tl.constexpr, HAS_V: tl.constexpr, KV_FP8: tl.constexpr, QH: tl.constexpr,
              HAS_SLOTS: tl.constexpr = False, HAS_STATE: tl.constexpr = False):
    """One row: all its heads (QH == HQ), or, per program along axis 1, QH of its query heads / its K and V.
    HAS_SLOTS: row r belongs to the request at slot SLOTS[r], at position START[slot] + INDEX[r] of the slot's
    arena K/V[slot] (S positions per slot); HAS_STATE: START[slot] read at STATE[r, 0] instead (see embed)."""
    _wait()
    row = tl.program_id(0)
    part = tl.program_id(1)
    if HAS_SLOTS:
        slot = tl.load(SLOTS + row)
        if HAS_STATE:
            pos = tl.load(STATE + row * 4) + tl.load(INDEX + row)
        else:
            pos = tl.load(START + slot) + tl.load(INDEX + row)
        K += slot.to(tl.int64) * S * HK * D
        V += slot.to(tl.int64) * S * HK * D
    else:
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
             kv_fp8=False, split=False, slots=None, index=None, state=None):
    """Q/K norm + RoPE, V norm; Q to `q`, K/V into the arena at [start, start + rows) (FP8-rounded with kv_fp8).
    `split`: 4 query heads per program and one program for K/V, rather than one program per row. `slots` / `index`
    (per row): row r at position start[slots[r]] + index[r] of arena slot slots[r]; `state` (per row, see embed):
    start[slots[r]] at state[r, 0]."""
    qh = min(4, heads) if split else heads
    grid = (qkv.shape[0], heads // qh + 1 if split else 1)
    _qkv_rope[grid](qkv, q_weight, k_weight, cos, sin, start, q, k, v, slots if slots is not None else start,
                    index if index is not None else start, state if state is not None else start, k.shape[1], eps,
                    heads, kv_heads, head_dim, has_v, kv_fp8, qh, slots is not None, state is not None,
                    num_warps=4 if split or head_dim <= 256 else 8, **LAUNCH)
    return q


@triton.jit
def _logit_parts(X, STEP, SEED, BLOCK, TEMP, PART, SLOTS, INDEX, ROWS, cap, t_min, t_max, steps, VOCAB: tl.constexpr,
                 OFFSET: tl.constexpr, TOTAL: tl.constexpr, PARTS: tl.constexpr, N: tl.constexpr,
                 ADAPTIVE: tl.constexpr, RAND4: tl.constexpr, HAS_SLOTS: tl.constexpr = False,
                 HAS_ROWS: tl.constexpr = False, ARGMAX_ONLY: tl.constexpr = 0):
    """Per (row, vocabulary part), at z = softcap(logits) / T: max, sum exp, centered entropy term, argmax,
    Gumbel-max sample; and max / sum exp of bf16(z), the normalizer of the next round's SC softmax.
    X holds this rank's VOCAB columns starting at global column OFFSET of TOTAL; indices and noise are global.
    T follows the linear schedule, or is this round's adaptive temperature in TEMP. RAND4: one Philox4x32 call
    draws the noise of four consecutive columns (a different stream than one call per column). HAS_ROWS: only rows
    whose ROWS[row, 3] (the forward's mode, see embed) is DENOISE; ARGMAX_ONLY (with HAS_ROWS): the canvas lengths
    (a sum of powers of two) whose stop rule reads only the argmax, whose rows draw no sample."""
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    if HAS_ROWS:
        if tl.load(ROWS + row * 4 + 3) != DENOISE:
            return
    # HAS_SLOTS: row r is row INDEX[r] of the canvas of the request at slot SLOTS[r]; its noise is keyed by that index
    # (so a request draws the same noise in any batch).
    if HAS_SLOTS:
        slot = tl.load(SLOTS + row)
        local = tl.load(INDEX + row)
    else:
        slot = 0
        local = row
    col = part * N + tl.arange(0, N)
    live = col < VOCAB
    step = tl.load(STEP + slot)
    if ADAPTIVE:
        t = tl.load(TEMP + slot)
    else:
        t = t_min + (t_max - t_min) * (steps - step).to(tl.float32) / steps
    x = tl.load(X + row.to(tl.int64) * VOCAB + col, live, 0.0).to(tl.float32)
    z = tl.where(live, cap * libdevice.tanh(x / cap) / t, float("-inf"))
    m = tl.max(z, 0)
    e = tl.exp(z - m)
    s = tl.sum(e, 0)
    c = tl.sum(tl.where(live, e * (z - m), 0.0), 0)
    arg = tl.min(tl.where(z == m, col + OFFSET, 2147483647), 0)
    zb = _bf(z)
    mb = tl.max(zb, 0)
    sb = tl.sum(tl.exp(zb - mb), 0)
    sampled = True
    if ARGMAX_ONLY > 0:
        sampled = ((tl.load(ROWS + row * 4 + 1) - tl.load(ROWS + row * 4)) & ARGMAX_ONLY) == 0
    gm = tl.full([], float("-inf"), tl.float32)
    gi = tl.full([], 0, tl.int32)
    if sampled:
        key = _key(tl.load(SEED + slot), tl.load(BLOCK + slot), step)
        if RAND4:
            q = (local * TOTAL + OFFSET + part * N) // 4 + tl.arange(0, N // 4)
            r0, r1, r2, r3 = tl.rand4x(key, q)
            u = tl.reshape(tl.join(tl.join(r0, r1), tl.join(r2, r3)), [N])
        else:
            u = tl.rand(key, local * TOTAL + col + OFFSET)
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
def _logit_finish(PART, STATS, ARGMAX, SAMPLE, ROWS, PARTS: tl.constexpr, RANKS: tl.constexpr, STRIDE: tl.constexpr,
                  N: tl.constexpr, HAS_ROWS: tl.constexpr = False):
    """Combine the per-part records of every vocabulary shard (`RANKS` blocks of `STRIDE` floats)."""
    _wait()
    row = tl.program_id(0)
    if HAS_ROWS:
        if tl.load(ROWS + row * 4 + 3) != DENOISE:
            return
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
                rand4=False, slots=None, index=None, rows=None, argmax_only=0):
    """Per canvas row: SC log-normalizer and entropy [rows, 2], argmax and a sample of softmax(z).

    With a `group`, `logits` holds this rank's vocabulary columns starting at `offset`; the per-part records
    of all ranks are gathered in rank order so every rank reaches the same result. With an adaptive decoder
    the temperature is read from `temp` (see `adaptive_temperature`). `rand4`: four columns per Philox call.
    `slots` / `index` (per row): row r is row index[r] of the canvas of the request whose state is at slots[r].
    `rows`: the forward's row state (see embed); rows not in mode DENOISE are skipped (their outputs left as they were),
    and with `argmax_only` (a sum of canvas lengths) the rows of canvases of those lengths draw no sample.
    """
    state = rows
    rows, vocab = logits.shape
    ranks = group.world if group is not None else 1
    n = 2048
    parts = triton.cdiv(vocab, n)
    stride = triton.cdiv(rows * parts * 8, 4) * 4  # 16-byte multiple for the multicast gather
    scratch = torch.empty(stride, device=logits.device, dtype=torch.float32)
    _logit_parts[(rows, parts)](logits, step, seed, block, temp if config.adaptive else step, scratch,
                                slots if slots is not None else step, index if index is not None else step,
                                state if state is not None else step, cap, config.t_min, config.t_max,
                                config.max_denoising_steps, vocab, offset, vocab * ranks, parts, n, config.adaptive,
                                rand4, slots is not None, state is not None, argmax_only if state is not None else 0,
                                num_warps=8, **LAUNCH)
    if group is not None:
        scratch = group.all_gather(scratch, "logit_stats")
    _logit_finish[(rows,)](scratch, stats, argmax, sample, state if state is not None else stats, parts, ranks, stride,
                           triton.next_power_of_2(parts * ranks), state is not None, **LAUNCH)


@triton.jit
def _sc_probs(X, STATS, STEP, TEMP, Y, SLOTS, MODES, cap, t_min, t_max, steps, V: tl.constexpr, BLOCK: tl.constexpr,
              ADAPTIVE: tl.constexpr, HAS_SLOTS: tl.constexpr = False, HAS_MODES: tl.constexpr = False):
    """HAS_MODES (with HAS_SLOTS): rows whose request is not in DENOISE or at round 0 are skipped (their SC delta is
    not read)."""
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    if HAS_SLOTS:
        slot = tl.load(SLOTS + row)
        if HAS_MODES:
            if (tl.load(MODES + slot) != DENOISE) | (tl.load(STEP + slot) == 0):
                return
    else:
        slot = 0
    cols = part * BLOCK + tl.arange(0, BLOCK)
    mask = cols < V
    step = tl.load(STEP + slot)
    # The previous round's temperature (TEMP is updated later in this round); round 0 has no SC signal.
    if ADAPTIVE:
        t = tl.load(TEMP + slot)
    else:
        t = t_min + (t_max - t_min) * (steps - step + 1).to(tl.float32) / steps
    base = row.to(tl.int64) * V
    x = tl.load(X + base + cols, mask & (step > 0), 0.0).to(tl.float32)
    z = _bf(cap * libdevice.tanh(x / cap) / t)
    p = tl.where(step > 0, tl.exp(z - tl.load(STATS + row * 2)), 0.0)
    tl.store(Y + base + cols, p.to(tl.bfloat16), mask)
    _release()


def sc_probs(logits, stats, step, config, out, cap, temp=None, slots=None, modes=None):
    """softmax(bf16(softcap(previous logits) / T_previous)) in BF16; zeros in round 0. `slots`: per row, the slot of
    its request's state; `modes` (per slot): only the rows of requests in DENOISE past round 0."""
    rows, v = logits.shape
    _sc_probs[(rows, triton.cdiv(v, 4096))](logits, stats, step, temp if config.adaptive else step, out,
                                            slots if slots is not None else step, modes if modes is not None else step,
                                            cap, config.t_min, config.t_max, config.max_denoising_steps, v, 4096,
                                            config.adaptive, slots is not None, modes is not None, num_warps=8,
                                            **LAUNCH)
    return out


@triton.jit
def _raw_parts(X, PART, ROWS, cap, VOCAB: tl.constexpr, PARTS: tl.constexpr, N: tl.constexpr,
               HAS_ROWS: tl.constexpr = False):
    """Per (row, vocabulary part) of softcap(logits) at temperature one: max, sum exp, centered entropy term."""
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    if HAS_ROWS:
        if tl.load(ROWS + row * 4 + 3) != DENOISE:
            return
    col = part * N + tl.arange(0, N)
    live = col < VOCAB
    x = tl.load(X + row.to(tl.int64) * VOCAB + col, live, 0.0).to(tl.float32)
    z = tl.where(live, cap * libdevice.tanh(x / cap), float("-inf"))
    m = tl.max(z, 0)
    e = tl.exp(z - m)
    base = PART + (row * PARTS + part) * 4
    tl.store(base, m)
    tl.store(base + 1, tl.sum(e, 0))
    tl.store(base + 2, tl.sum(tl.where(live, e * (z - m), 0.0), 0))
    _release()


@triton.jit
def _raw_finish(PART, H, ROWS, log_v, PARTS: tl.constexpr, RANKS: tl.constexpr, STRIDE: tl.constexpr, N: tl.constexpr,
                HAS_ROWS: tl.constexpr = False):
    """Normalized entropy of one row from the part records of every vocabulary shard."""
    _wait()
    row = tl.program_id(0)
    if HAS_ROWS:
        if tl.load(ROWS + row * 4 + 3) != DENOISE:
            return
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
def _temperature(H, ACCEPTED, STEP, TEMP, HREF, TPREV, SLOTS, OFFSETS, BLOCKS, t_min, t_max, power, B: tl.constexpr,
                 HAS_SLOTS: tl.constexpr = False, B1: tl.constexpr = 0, B2: tl.constexpr = 0, B3: tl.constexpr = 0):
    """This round's adaptive temperature from the mean entropy of the positions the previous round did not accept
    (all positions if it accepted every one); round 0 runs at t_max and sets the reference entropy. HAS_SLOTS: one
    program per request, its state at slot SLOTS[program] and its BLOCKS[slot] rows from OFFSETS[slot], one of the
    canvas lengths B, B1, B2, B3 (0: unused)."""
    _wait()
    if HAS_SLOTS:
        slot = tl.load(SLOTS + tl.program_id(0))
        rows = tl.load(BLOCKS + slot)
        o = tl.load(OFFSETS + slot)
        if rows == B:
            _temperature_one(H + o, ACCEPTED + o, STEP, TEMP, HREF, TPREV, slot, t_min, t_max, power, B)
        if B1 > 0:
            if rows == B1:
                _temperature_one(H + o, ACCEPTED + o, STEP, TEMP, HREF, TPREV, slot, t_min, t_max, power, B1)
        if B2 > 0:
            if rows == B2:
                _temperature_one(H + o, ACCEPTED + o, STEP, TEMP, HREF, TPREV, slot, t_min, t_max, power, B2)
        if B3 > 0:
            if rows == B3:
                _temperature_one(H + o, ACCEPTED + o, STEP, TEMP, HREF, TPREV, slot, t_min, t_max, power, B3)
    else:
        _temperature_one(H, ACCEPTED, STEP, TEMP, HREF, TPREV, 0, t_min, t_max, power, B)
    _release()


@triton.jit
def _temperature_one(H, ACCEPTED, STEP, TEMP, HREF, TPREV, slot, t_min, t_max, power, B: tl.constexpr):
    i = tl.arange(0, B)
    h = tl.load(H + i)
    scope = tl.load(ACCEPTED + i) == 0
    scope = scope | (tl.sum(scope.to(tl.int32), 0) == 0)
    h_scope = tl.sum(tl.where(scope, h, 0.0), 0) / tl.maximum(tl.sum(scope.to(tl.int32), 0), 1)
    first = tl.load(STEP + slot) == 0
    h_ref = tl.where(first, tl.maximum(h_scope, 1e-8), tl.load(HREF + slot))
    ratio = tl.minimum(tl.maximum(h_scope / tl.maximum(h_ref, 1e-8), 0.0), 1.0)
    proposed = t_min + (t_max - t_min) * tl.exp(power * tl.log(ratio))
    t = tl.where(first, t_max, tl.minimum(tl.load(TPREV + slot), proposed))
    t = tl.minimum(tl.maximum(t, t_min), t_max)
    tl.store(TEMP + slot, t)
    tl.store(HREF + slot, h_ref)
    tl.store(TPREV + slot, t)


def adaptive_temperature(logits, step, accepted, temp, h_ref, t_prev, config, cap, *, group=None, slots=None,
                         offsets=None, blocks=None, classes=(), rows=None):
    """Adaptive-scalar temperature of this round into `temp` (state in `h_ref`, `t_prev`; scope from `accepted`).
    `slots` (per request) with `offsets` / `blocks` (per slot): request p's canvas is rows [offsets[s], + blocks[s])
    for s = slots[p], its length one of `classes`. `rows`: the forward's row state (see embed); the entropy of rows not
    in DENOISE is not computed (a request that is not denoising gets a temperature nobody reads: round 0 resets it)."""
    state = rows
    rows, vocab = logits.shape
    ranks = group.world if group is not None else 1
    n = 2048
    parts = triton.cdiv(vocab, n)
    stride = rows * parts * 4
    scratch = torch.empty(stride, device=logits.device, dtype=torch.float32)
    _raw_parts[(rows, parts)](logits, scratch, state if state is not None else scratch, cap, vocab, parts, n,
                              state is not None, num_warps=8, **LAUNCH)
    if group is not None:
        scratch = group.all_gather(scratch, "raw_entropy")
    entropy = torch.empty(rows, device=logits.device, dtype=torch.float32)
    _raw_finish[(rows,)](scratch, entropy, state if state is not None else entropy, math.log(max(vocab * ranks, 2)),
                         parts, ranks, stride, triton.next_power_of_2(parts * ranks), state is not None, **LAUNCH)
    if slots is None:
        _temperature[(1,)](entropy, accepted, step, temp, h_ref, t_prev, step, step, step, config.t_min, config.t_max,
                           config.adaptive_power, rows, num_warps=4, **LAUNCH)
    else:
        b, _ = _classes(classes, classes)
        _temperature[(slots.numel(),)](entropy, accepted, step, temp, h_ref, t_prev, slots, offsets, blocks,
                                       config.t_min, config.t_max, config.adaptive_power, b[0], True, b[1], b[2], b[3],
                                       num_warps=4, **LAUNCH)


@triton.jit
def _begin(P, TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, window, VOCAB: tl.constexpr,
           B: tl.constexpr):
    _wait()
    _start_canvas(TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, 0, tl.load(P),
                  tl.load(P + 1), tl.load(P + 2), window, VOCAB, B)
    _release()


@triton.jit
def _start_canvas(TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, slot, offset, ordinal,
                  seed, window, VOCAB: tl.constexpr, B: tl.constexpr):
    """A canvas of uniform random tokens at [offset, offset + B); state at `slot`."""
    i = tl.arange(0, B)
    noise = tl.randint(_key(seed, ordinal, 0), INIT_STREAM + i) % VOCAB
    tl.store(TOKENS + i, noise.to(tl.int64))
    tl.store(OUT + i, tl.full((B,), -1, tl.int64))
    tl.store(ACCEPTED + i, tl.zeros((B,), tl.int8))
    tl.store(START + slot, offset)
    tl.store(END + slot, offset + B)
    tl.store(LEFTPAD + slot, tl.maximum(offset - (window - 1), 0))
    tl.store(STEP + slot, 0)
    tl.store(REMAINING + slot, 1)
    tl.store(SEED + slot, seed)
    tl.store(BLOCK + slot, ordinal)


def begin(params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block, window, vocab):
    """Canvas of uniform random tokens at [offset, offset + B); argmax history and accepted mask cleared."""
    _begin[(1,)](params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block, window, vocab,
                 tokens.numel(), **LAUNCH)


@triton.jit
def _decide(STATS, ARGMAX, SAMPLE, TOKENS, OUT, ACCEPTED, STEP, REMAINING, SEED, BLOCK, SLOTS, OFFSETS, BLOCKS, BAD,
            bound, confidence, VOCAB: tl.constexpr, B: tl.constexpr, ADAPTIVE: tl.constexpr, STOP: tl.constexpr = 0,
            HAS_SLOTS: tl.constexpr = False, B1: tl.constexpr = 0, B2: tl.constexpr = 0, B3: tl.constexpr = 0,
            STOP1: tl.constexpr = 0, STOP2: tl.constexpr = 0, STOP3: tl.constexpr = 0, HAS_BAD: tl.constexpr = False):
    """HAS_SLOTS: one program per request, its state at slot SLOTS[program] and its canvas the BLOCKS[slot] rows from
    OFFSETS[slot]: canvas lengths B, B1, B2, B3 (0: unused) with stop rules STOP, STOP1, STOP2, STOP3. A request with
    no running canvas (stopped, or in another stage of a batched forward) leaves everything unchanged. HAS_BAD: *BAD
    += the positions without a token (see _decide_one)."""
    _wait()
    if HAS_SLOTS:
        slot = tl.load(SLOTS + tl.program_id(0))
        rows = tl.load(BLOCKS + slot)
        o = tl.load(OFFSETS + slot)
        alive = tl.load(REMAINING + slot) != 0
        if rows == B:
            _decide_one(STATS + 2 * o, ARGMAX + o, SAMPLE + o, TOKENS + o, OUT + o, ACCEPTED + o, STEP, REMAINING,
                        SEED, BLOCK, BAD, slot, alive, bound, confidence, VOCAB, B, ADAPTIVE, STOP, HAS_BAD)
        if B1 > 0:
            if rows == B1:
                _decide_one(STATS + 2 * o, ARGMAX + o, SAMPLE + o, TOKENS + o, OUT + o, ACCEPTED + o, STEP, REMAINING,
                            SEED, BLOCK, BAD, slot, alive, bound, confidence, VOCAB, B1, ADAPTIVE, STOP1, HAS_BAD)
        if B2 > 0:
            if rows == B2:
                _decide_one(STATS + 2 * o, ARGMAX + o, SAMPLE + o, TOKENS + o, OUT + o, ACCEPTED + o, STEP, REMAINING,
                            SEED, BLOCK, BAD, slot, alive, bound, confidence, VOCAB, B2, ADAPTIVE, STOP2, HAS_BAD)
        if B3 > 0:
            if rows == B3:
                _decide_one(STATS + 2 * o, ARGMAX + o, SAMPLE + o, TOKENS + o, OUT + o, ACCEPTED + o, STEP, REMAINING,
                            SEED, BLOCK, BAD, slot, alive, bound, confidence, VOCAB, B3, ADAPTIVE, STOP3, HAS_BAD)
    else:
        _decide_one(STATS, ARGMAX, SAMPLE, TOKENS, OUT, ACCEPTED, STEP, REMAINING, SEED, BLOCK, BAD, 0,
                    tl.full([], 1, tl.int1), bound, confidence, VOCAB, B, ADAPTIVE, STOP, HAS_BAD)
    _release()


@triton.jit
def _decide_one(STATS, ARGMAX, SAMPLE, TOKENS, OUT, ACCEPTED, STEP, REMAINING, SEED, BLOCK, BAD, slot, alive, bound,
                confidence, VOCAB: tl.constexpr, B: tl.constexpr, ADAPTIVE: tl.constexpr, STOP: tl.constexpr,
                HAS_BAD: tl.constexpr):
    """A position whose logits row had no argmax / sample (logit_stats' 2147483647 for a row of NaN) is never taken:
    it is re-noised, and its argmax-canvas entry is that noise (no token outside the vocabulary leaves the decoder)."""
    i = tl.arange(0, B)
    rows = (i < B) & alive
    step = tl.load(STEP + slot)
    entropy = tl.load(STATS + i * 2 + 1)
    rkey = _key(tl.load(SEED + slot), tl.load(BLOCK + slot), step)
    if STOP > 0:
        # Freeze: an accepted position keeps its token for the rest of the canvas (ACCEPTED is cumulative; begin
        # clears it); each round accepts the lowest-entropy unfrozen positions within the bound (at least one), and
        # the canvas stops once every position is frozen. STOP 1 freezes the sample, STOP 2 the argmax.
        frozen = tl.load(ACCEPTED + i) != 0
        hkey = tl.where(frozen, 1.0e9, entropy)
        key = (hkey.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | i.to(tl.uint64)
        order = (tl.sort(key, descending=False) & 0xFFFFFFFF).to(tl.int32)
        h = tl.gather(hkey, order, 0)
        fz = tl.gather(frozen.to(tl.int32), order, 0) != 0
        noise = tl.randint(rkey, RENOISE_STREAM + order) % VOCAB
        current = tl.load(TOKENS + order)
        argmax = tl.load(ARGMAX + order)
        chosen = argmax if STOP == 2 else tl.load(SAMPLE + order)
        bad = (argmax < 0) | (argmax >= VOCAB) | (chosen < 0) | (chosen >= VOCAB)
        take = ((tl.cumsum(h, 0) - h) <= bound) & (fz == 0) & (bad == 0)
        new = tl.where(fz, current, tl.where(take, chosen, noise.to(tl.int64)))
        tl.store(TOKENS + order, new, rows)
        done = fz | take
        tl.store(ACCEPTED + order, done.to(tl.int8), rows)
        tl.store(OUT + order, tl.where(done | bad, new, argmax), rows)
        remaining = tl.where(tl.sum(done.to(tl.int32), 0) == B, 0, 1)
    else:
        # Nonnegative float bits sort numerically; the index breaks ties deterministically.
        key = (entropy.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | i.to(tl.uint64)
        order = (tl.sort(key, descending=False) & 0xFFFFFFFF).to(tl.int32)
        h = tl.gather(entropy, order, 0)
        sample = tl.load(SAMPLE + order)
        take = ((tl.cumsum(h, 0) - h) <= bound) & (sample >= 0) & (sample < VOCAB)
        noise = tl.randint(rkey, RENOISE_STREAM + order) % VOCAB
        tl.store(TOKENS + order, tl.where(take, sample, noise.to(tl.int64)), rows)
        if ADAPTIVE:  # the next round's temperature averages over the positions not accepted here
            tl.store(ACCEPTED + order, take.to(tl.int8), rows)
        argmax = tl.load(ARGMAX + i)
        bad = (argmax < 0) | (argmax >= VOCAB)
        argmax = tl.where(bad, (tl.randint(rkey, RENOISE_STREAM + i) % VOCAB).to(tl.int64), argmax)
        stable = (step > 0) & (tl.sum((argmax != tl.load(OUT + i)).to(tl.int32), 0) == 0)
        confident = tl.sum(entropy, 0) / B < confidence
        tl.store(OUT + i, argmax, rows)
        remaining = tl.where(stable & confident, 0, 1)
    tl.store(REMAINING + slot, tl.where(alive, remaining, 0))
    tl.store(STEP + slot, step + alive.to(tl.int32))
    if HAS_BAD:
        missing = tl.sum((bad & rows).to(tl.int32), 0)
        if missing > 0:
            tl.atomic_add(BAD, missing)


def decide(stats, argmax, sample, tokens, out, accepted, step, remaining, seed, block, config, vocab, slots=None,
           offsets=None, blocks=None, classes=(), stops=(), bad=None):
    """Keep samples at the lowest-entropy positions within the entropy bound, re-noise the rest, under the decoder's
    stop rule (GemmaDecode.stop_rule): "renoise" stops once the argmax canvas is unchanged and the mean entropy is
    below the confidence threshold; "freeze" / "freeze_argmax" keep accepted samples / argmax tokens and stop once
    every position is accepted. `slots` (per request) with `offsets` / `blocks` (per slot): request p's canvas is
    rows [offsets[s], + blocks[s]) for s = slots[p], its length one of `classes`, decided by the stop rule of its
    class in `stops` (default the decoder's for all). A position whose logits are NaN is re-noised; `bad` (int32, one
    element) counts them."""
    count = bad if bad is not None else step
    if slots is None:
        _decide[(1,)](stats, argmax, sample, tokens, out, accepted, step, remaining, seed, block, step, step, step,
                      count, config.entropy_bound, config.confidence_threshold, vocab, tokens.numel(), config.adaptive,
                      STOPS[config.stop_rule], HAS_BAD=bad is not None, num_warps=4, **LAUNCH)
    else:
        b, s = _classes(classes, [STOPS[x] for x in stops or (config.stop_rule,) * len(classes)])
        _decide[(slots.numel(),)](stats, argmax, sample, tokens, out, accepted, step, remaining, seed, block, slots,
                                  offsets, blocks, count, config.entropy_bound, config.confidence_threshold, vocab,
                                  b[0], config.adaptive, s[0], True, b[1], b[2], b[3], s[1], s[2], s[3],
                                  bad is not None, num_warps=4, **LAUNCH)


@triton.jit
def _advance(MODE, P, TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, SLOTS, OFFSETS, BLOCKS,
             WANT, PROMPT, PLEN, MAX_NEW, TOKEN_OUT, STEP_OUT, BLOCK_OUT, CLOCK_OUT, CANVASES, EOS, CTRL, COUNTS, LANE,
             WANTS, window, prompt_stride, out_stride, steps_stride, limit, VOCAB: tl.constexpr, NEOS: tl.constexpr,
             B: tl.constexpr, B1: tl.constexpr = 0, B2: tl.constexpr = 0, B3: tl.constexpr = 0,
             FUSED: tl.constexpr = False):
    """After a batched forward, one program per request: its state at slot SLOTS[program], its BLOCKS[slot] rows from
    OFFSETS[slot] (canvas lengths B, B1, B2, B3; 0: unused), see _advance_one. FUSED: WANTS[program] = 1 if its canvas
    stopped and waits for a commit lane (see _claim), else 0."""
    _wait()
    p = tl.program_id(0)
    slot = tl.load(SLOTS + p)
    rows = tl.load(BLOCKS + slot)
    o = tl.load(OFFSETS + slot)
    if FUSED:
        tl.store(WANTS + p, 0)
    if rows == B:
        _advance_one(MODE, P, TOKENS, o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, WANT,
                     PROMPT, PLEN, MAX_NEW, TOKEN_OUT, STEP_OUT, BLOCK_OUT, CLOCK_OUT, CANVASES, EOS, CTRL, COUNTS,
                     LANE, WANTS + p, slot, window, prompt_stride, out_stride, steps_stride, limit, VOCAB, B, NEOS,
                     FUSED)
    if B1 > 0:
        if rows == B1:
            _advance_one(MODE, P, TOKENS, o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                         WANT, PROMPT, PLEN, MAX_NEW, TOKEN_OUT, STEP_OUT, BLOCK_OUT, CLOCK_OUT, CANVASES, EOS, CTRL,
                         COUNTS, LANE, WANTS + p, slot, window, prompt_stride, out_stride, steps_stride, limit, VOCAB,
                         B1, NEOS, FUSED)
    if B2 > 0:
        if rows == B2:
            _advance_one(MODE, P, TOKENS, o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                         WANT, PROMPT, PLEN, MAX_NEW, TOKEN_OUT, STEP_OUT, BLOCK_OUT, CLOCK_OUT, CANVASES, EOS, CTRL,
                         COUNTS, LANE, WANTS + p, slot, window, prompt_stride, out_stride, steps_stride, limit, VOCAB,
                         B2, NEOS, FUSED)
    if B3 > 0:
        if rows == B3:
            _advance_one(MODE, P, TOKENS, o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                         WANT, PROMPT, PLEN, MAX_NEW, TOKEN_OUT, STEP_OUT, BLOCK_OUT, CLOCK_OUT, CANVASES, EOS, CTRL,
                         COUNTS, LANE, WANTS + p, slot, window, prompt_stride, out_stride, steps_stride, limit, VOCAB,
                         B3, NEOS, FUSED)
    _release()


@triton.jit
def _advance_one(MODE, P, ROWS, o, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, WANT, PROMPT, PLEN,
                 MAX_NEW, TOKEN_OUT, STEP_OUT, BLOCK_OUT, CLOCK_OUT, CANVASES, EOS, CTRL, COUNTS, LANE, WANTS, slot,
                 window, prompt_stride, out_stride, steps_stride, limit, VOCAB: tl.constexpr, B: tl.constexpr,
                 NEOS: tl.constexpr, FUSED: tl.constexpr):
    """One request of B rows from row o (ROWS: the token rows of the batch; OUT / ACCEPTED at its rows):
    DENOISE, canvas stopped (by the decoder, or after `limit` rounds as GemmaEngine's loop bound): record it
        (TOKEN_OUT after the earlier canvases' tokens, STEP_OUT / BLOCK_OUT per canvas, CLOCK_OUT the run's forwards
        before this one, COUNTS[16]); it ends the request (DONE, CTRL[2] += 1) with an EOS within the token budget or
        as the last canvas, uncommitted; otherwise it is committed next (COMMIT), or, FUSED, when this launch has
        lanes (LANE[0] > 0) of at least B rows and the next canvas has the same length, it waits for a commit lane
        (*WANTS = 1, still DENOISE with REMAINING 0; _claim).
    COMMIT (just committed): the next canvas.
    PREFILL (a prompt chunk of B tokens just committed): the next chunk, or the first canvas after the prompt.
    The next canvas begins if the request's wanted canvas length WANT[slot] is B, else see _next_canvas."""
    i = tl.arange(0, B)
    TOKENS = ROWS + o
    P += slot * 3
    mode = tl.load(MODE + slot)
    if mode == DENOISE:
        if (tl.load(REMAINING + slot) == 0) | (tl.load(STEP + slot) >= limit):
            tl.store(REMAINING + slot, 0)
            ordinal = tl.load(P + 1)
            emitted = tl.load(P) - tl.load(PLEN + slot)
            canvas = tl.load(OUT + i)
            tl.store(TOKEN_OUT + slot.to(tl.int64) * out_stride + emitted + i, canvas)
            tl.store(STEP_OUT + slot * steps_stride + ordinal, tl.load(STEP + slot))
            tl.store(BLOCK_OUT + slot * steps_stride + ordinal, B)
            tl.store(CLOCK_OUT + slot * steps_stride + ordinal, tl.load(COUNTS + 16))
            budget = tl.load(MAX_NEW + slot) - emitted
            hit = canvas == tl.load(EOS)
            for e in tl.static_range(1, NEOS):
                hit = hit | (canvas == tl.load(EOS + e))
            if (tl.sum((hit & (i < budget)).to(tl.int32), 0) > 0) | (budget <= B):
                tl.store(MODE + slot, DONE)
                tl.store(CANVASES + slot, ordinal + 1)
                tl.atomic_add(CTRL + 2, 1)
            else:
                if FUSED:
                    if (tl.load(WANT + slot) == B) & (tl.load(LANE) > 0) & (tl.load(LANE + 1) >= B):
                        tl.store(WANTS, 1)
                    else:
                        tl.store(TOKENS + i, canvas)
                        tl.store(MODE + slot, COMMIT)
                else:
                    tl.store(TOKENS + i, canvas)
                    tl.store(MODE + slot, COMMIT)
    elif mode == COMMIT:
        offset = tl.load(P) + B
        ordinal = tl.load(P + 1) + 1
        tl.store(P, offset)
        tl.store(P + 1, ordinal)
        _next_canvas(MODE, P, TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, WANT, CTRL, slot,
                     offset, ordinal, window, VOCAB, B)
    elif mode == PREFILL:
        pos = tl.load(START + slot) + B
        plen = tl.load(PLEN + slot)
        if pos >= plen:
            tl.store(P, plen)
            tl.store(P + 1, 0)
            _next_canvas(MODE, P, TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, WANT, CTRL,
                         slot, plen, 0, window, VOCAB, B)
        else:
            src = pos + i
            tl.store(TOKENS + i, tl.load(PROMPT + slot.to(tl.int64) * prompt_stride + src, src < plen, 0))
            tl.store(START + slot, pos)
            tl.store(END + slot, pos + B)


@triton.jit
def _next_canvas(MODE, P, TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, WANT, CTRL, slot,
                 offset, ordinal, window, VOCAB: tl.constexpr, B: tl.constexpr):
    """Begin canvas `ordinal` at `offset` (P already holds both) if the request wants B rows; else park the request
    (SWITCH, CTRL[2] += 1, which ends the launch) until the host gives it rows of the length it wants."""
    if tl.load(WANT + slot) == B:
        _start_canvas(TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, slot, offset, ordinal,
                      tl.load(P + 2), window, VOCAB, B)
        tl.store(MODE + slot, DENOISE)
    else:
        tl.store(MODE + slot, SWITCH)
        tl.atomic_add(CTRL + 2, 1)


def advance(mode, params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block, slots, offsets,
            blocks, want, prompt, prompt_len, max_new, token_out, step_out, block_out, canvases, eos, ctrl, window,
            vocab, classes, limit, *, lanes=None, clock=None):
    """`lanes`: (lane state, waiting flag per request position) for fused commits (see _advance_one, _claim).
    `clock`: (counts, clock_out like step_out): each stopped canvas records counts[16] (forwards before this one);
    none: not recorded."""
    b, _ = _classes(classes, classes)
    lane, wants = lanes if lanes is not None else (ctrl, ctrl)
    counts, clock_out = clock if clock is not None else (torch.zeros(17, device=ctrl.device, dtype=torch.int32),
                                                         torch.empty_like(step_out))
    _advance[(slots.numel(),)](mode, params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block,
                               slots, offsets, blocks, want, prompt, prompt_len, max_new, token_out, step_out,
                               block_out, clock_out, canvases, eos, ctrl, counts, lane, wants, window, prompt.shape[1],
                               token_out.shape[1], step_out.shape[1], limit, vocab, eos.numel(), *b, lanes is not None,
                               **LAUNCH)


@triton.jit
def _claim(MODE, P, ROWS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, SLOTS, OFFSETS, BLOCKS,
           WANTS, LANE, ROW_SLOT, ROW_INDEX, window, N: tl.constexpr, NP: tl.constexpr, VOCAB: tl.constexpr,
           B: tl.constexpr, B1: tl.constexpr = 0, B2: tl.constexpr = 0, B3: tl.constexpr = 0):
    """After `advance` (fused), one program per request position: the canvases waiting for a commit lane (WANTS[p]
    = 1) take the launch's LANE[0] lanes in position order, the rest commit in their own rows next (see _claim_one).
    Program 0 keeps the lane counters (LANE layout in batched.py): lanes claimed, lane commits and their rows, and
    the waiting canvases per canvas length (B, B1, B2, B3)."""
    _wait()
    p = tl.program_id(0)
    j = tl.arange(0, NP)
    live = j < N
    wants = tl.load(WANTS + j, live, 0)
    lanes = tl.load(LANE)
    if tl.load(WANTS + p) != 0:
        rank = tl.sum(tl.where(j < p, wants, 0), 0)
        slot = tl.load(SLOTS + p)
        rows = tl.load(BLOCKS + slot)
        o = tl.load(OFFSETS + slot)
        if rows == B:
            _claim_one(MODE, P, ROWS, o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                       LANE, ROW_SLOT, ROW_INDEX, slot, rank, lanes, window, VOCAB, B)
        if B1 > 0:
            if rows == B1:
                _claim_one(MODE, P, ROWS, o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                           LANE, ROW_SLOT, ROW_INDEX, slot, rank, lanes, window, VOCAB, B1)
        if B2 > 0:
            if rows == B2:
                _claim_one(MODE, P, ROWS, o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                           LANE, ROW_SLOT, ROW_INDEX, slot, rank, lanes, window, VOCAB, B2)
        if B3 > 0:
            if rows == B3:
                _claim_one(MODE, P, ROWS, o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                           LANE, ROW_SLOT, ROW_INDEX, slot, rank, lanes, window, VOCAB, B3)
    if p == 0:
        rows = tl.load(BLOCKS + tl.load(SLOTS + j, live, 0), live, 0)
        got = (wants != 0) & (tl.cumsum(wants, 0) - wants < lanes)
        claimed = tl.sum(got.to(tl.int32), 0)
        tl.store(LANE + 3, claimed)
        tl.store(LANE + 4, tl.load(LANE + 4) + claimed)
        tl.store(LANE + 5, tl.load(LANE + 5) + tl.sum(tl.where(got, rows, 0), 0))
        _count_class(LANE + 8, wants != 0, rows, B)
        _count_class(LANE + 9, wants != 0, rows, B1)
        _count_class(LANE + 10, wants != 0, rows, B2)
        _count_class(LANE + 11, wants != 0, rows, B3)
    _release()


@triton.jit
def _count_class(COUNT, sel, rows, B: tl.constexpr):
    """*COUNT += the selected requests of canvas length B (B = 0: unused class)."""
    if B > 0:
        tl.store(COUNT, tl.load(COUNT) + tl.sum((sel & (rows == B)).to(tl.int32), 0))


@triton.jit
def _claim_one(MODE, P, ROWS, o, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, LANE, ROW_SLOT,
               ROW_INDEX, slot, rank, lanes, window, VOCAB: tl.constexpr, B: tl.constexpr):
    """A stopped canvas of B rows from row o, the rank-th waiting one in position order. Within the launch's `lanes`
    (LANE = [lanes, rows per lane, first lane row, ...]): copy its tokens to lane `rank`, map the lane's rows to its
    slot at indices -B..-1 (positions start - B.., causal, start being the next canvas's), record (slot, B) at
    LANE[16 + 2 rank] and begin the next canvas, whose first round runs in the same forward as the commit. Past
    them: commit in its own rows next (COMMIT), as without lanes."""
    i = tl.arange(0, B)
    TOKENS = ROWS + o
    P += slot * 3
    canvas = tl.load(OUT + i)
    if rank < lanes:
        first = tl.load(LANE + 2) + rank * tl.load(LANE + 1)
        tl.store(ROWS + first + i, canvas)
        tl.store(ROW_SLOT + first + i, slot)
        tl.store(ROW_INDEX + first + i, i - B)
        tl.store(LANE + 16 + 2 * rank, slot)
        tl.store(LANE + 17 + 2 * rank, B)
        offset = tl.load(P) + B
        ordinal = tl.load(P + 1) + 1
        tl.store(P, offset)
        tl.store(P + 1, ordinal)
        _start_canvas(TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, slot, offset, ordinal,
                      tl.load(P + 2), window, VOCAB, B)
    else:
        tl.store(TOKENS + i, canvas)
        tl.store(MODE + slot, COMMIT)


def claim(mode, params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block, slots, offsets,
          blocks, wants, lane, row_slot, row_index, window, vocab, classes):
    """Commit lanes for the canvases `advance` left waiting, in request-position order (see _claim)."""
    b, _ = _classes(classes, classes)
    n = slots.numel()
    _claim[(n,)](mode, params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block, slots, offsets,
                 blocks, wants, lane, row_slot, row_index, window, n, triton.next_power_of_2(n), vocab, *b, **LAUNCH)


@triton.jit
def _release_lanes(LANE, ROW_SLOT, ROW_INDEX, idle, B0: tl.constexpr):
    """Before the lanes are claimed again: their rows (B0 per program) back to the idle slot, no lane claimed."""
    _wait()
    p = tl.program_id(0)
    i = tl.arange(0, B0)
    row = tl.load(LANE + 2) + p * B0 + i
    live = p * B0 + i < tl.load(LANE) * tl.load(LANE + 1)
    tl.store(ROW_SLOT + row, idle, live)
    tl.store(ROW_INDEX + row, i, live)
    if p == 0:
        tl.store(LANE + 3, 0)
    _release()


def release_lanes(lane, row_slot, row_index, idle, b0, rows):
    """`rows`: the most lane rows there can be."""
    _release_lanes[(max(1, rows // b0),)](lane, row_slot, row_index, idle, b0, **LAUNCH)


@triton.jit
def _admit(MODE, P, TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, SLOTS, OFFSETS, BLOCKS,
           PROMPT, PLEN, window, prompt_stride, VOCAB: tl.constexpr, B: tl.constexpr, B1: tl.constexpr = 0,
           B2: tl.constexpr = 0, B3: tl.constexpr = 0):
    """At a launch's start, per request (state at slot SLOTS[program], BLOCKS[slot] rows from OFFSETS[slot]): one
    admitted by the host (ADMIT) starts its prefill, one parked for new rows (SWITCH) begins its next canvas."""
    _wait()
    slot = tl.load(SLOTS + tl.program_id(0))
    rows = tl.load(BLOCKS + slot)
    o = tl.load(OFFSETS + slot)
    if rows == B:
        _admit_one(MODE, P, TOKENS + o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, PROMPT,
                   PLEN, slot, window, prompt_stride, VOCAB, B)
    if B1 > 0:
        if rows == B1:
            _admit_one(MODE, P, TOKENS + o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                       PROMPT, PLEN, slot, window, prompt_stride, VOCAB, B1)
    if B2 > 0:
        if rows == B2:
            _admit_one(MODE, P, TOKENS + o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                       PROMPT, PLEN, slot, window, prompt_stride, VOCAB, B2)
    if B3 > 0:
        if rows == B3:
            _admit_one(MODE, P, TOKENS + o, OUT + o, ACCEPTED + o, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK,
                       PROMPT, PLEN, slot, window, prompt_stride, VOCAB, B3)
    _release()


@triton.jit
def _admit_one(MODE, P, TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, PROMPT, PLEN, slot,
               window, prompt_stride, VOCAB: tl.constexpr, B: tl.constexpr):
    """ADMIT: the first B prompt tokens (0 past the prompt); SWITCH: the canvas P holds, B rows."""
    mode = tl.load(MODE + slot)
    if mode == ADMIT:
        i = tl.arange(0, B)
        plen = tl.load(PLEN + slot)
        tl.store(TOKENS + i, tl.load(PROMPT + slot.to(tl.int64) * prompt_stride + i, i < plen, 0))
        tl.store(START + slot, 0)
        tl.store(END + slot, B)
        tl.store(REMAINING + slot, 0)
        tl.store(MODE + slot, PREFILL)
    elif mode == SWITCH:
        P += slot * 3
        _start_canvas(TOKENS, OUT, ACCEPTED, START, END, LEFTPAD, STEP, REMAINING, SEED, BLOCK, slot, tl.load(P),
                      tl.load(P + 1), tl.load(P + 2), window, VOCAB, B)
        tl.store(MODE + slot, DENOISE)


def admit(mode, params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block, slots, offsets,
          blocks, prompt, prompt_len, window, vocab, classes):
    b, _ = _classes(classes, classes)
    _admit[(slots.numel(),)](mode, params, tokens, out, accepted, start, end, leftpad, step, remaining, seed, block,
                             slots, offsets, blocks, prompt, prompt_len, window, prompt.shape[1], vocab, *b, **LAUNCH)


@triton.jit
def _control(MODE, STEP, SLOTS, BLOCKS, CTRL, COUNTS, LANE, N: tl.constexpr, NP: tl.constexpr, ROWS: tl.constexpr,
             HEAD: tl.constexpr, FUSED: tl.constexpr = False, B: tl.constexpr = 0, B1: tl.constexpr = 0,
             B2: tl.constexpr = 0, B3: tl.constexpr = 0):
    """The batched loop's WHILE state CTRL = [run, forwards, finished, finish limit, denoising, conditioned, clock]: run
    while some request is in PREFILL / DENOISE / COMMIT and fewer than CTRL[3] have finished (or parked, SWITCH) since
    the launch (HEAD: the launch's start); CTRL[4] = 1 if the next forward has a canvas round, CTRL[5] = 1 if one of
    them is past round 0 (reads an SC delta). Over every forward that runs, COUNTS[mode] accumulates its requests per
    mode (positions past the batch, at the idle slot, are not requests) and COUNTS[8 + mode] their rows (of ROWS, the
    rest IDLE); COUNTS[16] counts the forwards (CTRL[6] a copy). FUSED (LANE layout in batched.py): LANE[6]
    accumulates the lane rows held, LANE[12 + c] the active requests of canvas length class c (B, B1, B2, B3) in
    forwards whose lanes they may claim."""
    _wait()
    j = tl.arange(0, NP)
    slot = tl.load(SLOTS + j, j < N, 0)
    mode = tl.load(MODE + slot, j < N, -1)
    rows = tl.load(BLOCKS + slot, j < N, 0)
    step = tl.load(STEP + slot, j < N, 0)
    active = tl.sum(((mode == DENOISE) | (mode == COMMIT) | (mode == PREFILL)).to(tl.int32), 0)
    tl.store(CTRL + 4, tl.where(tl.sum((mode == DENOISE).to(tl.int32), 0) > 0, 1, 0))
    tl.store(CTRL + 5, tl.where(tl.sum(((mode == DENOISE) & (step > 0)).to(tl.int32), 0) > 0, 1, 0))
    if HEAD:
        tl.store(CTRL + 1, 0)
        tl.store(CTRL + 2, 0)
        tl.store(CTRL + 6, tl.load(COUNTS + 16))
        finished = 0
    else:
        tl.store(CTRL + 1, tl.load(CTRL + 1) + 1)
        clock = tl.load(COUNTS + 16) + 1
        tl.store(COUNTS + 16, clock)
        tl.store(CTRL + 6, clock)
        finished = tl.load(CTRL + 2)
    run = (active > 0) & (finished < tl.load(CTRL + 3))
    tl.store(CTRL, tl.where(run, 1, 0))
    k = tl.arange(0, 8)
    sel = (mode[None, :] == k[:, None]).to(tl.int32)
    tl.store(COUNTS + k, tl.load(COUNTS + k) + tl.where(run & (k != IDLE), tl.sum(sel, 1), 0))
    used = tl.where(k == IDLE, ROWS - tl.sum(tl.where(mode == IDLE, 0, rows), 0), tl.sum(sel * rows[None, :], 1))
    tl.store(COUNTS + 8 + k, tl.load(COUNTS + 8 + k) + tl.where(run, used, 0))
    if FUSED:
        lanes, size = tl.load(LANE), tl.load(LANE + 1)
        tl.store(LANE + 6, tl.load(LANE + 6) + tl.where(run, lanes * size, 0))
        live = run & (lanes > 0) & ((mode == DENOISE) | (mode == COMMIT) | (mode == PREFILL))
        _count_class(LANE + 12, live & (size >= B), rows, B)
        _count_class(LANE + 13, live & (size >= B1), rows, B1)
        _count_class(LANE + 14, live & (size >= B2), rows, B2)
        _count_class(LANE + 15, live & (size >= B3), rows, B3)
    _release()


def control(mode, step, slots, blocks, ctrl, counts, rows, head, *, lane=None, classes=()):
    """`lane`: the lane state of a fused engine (with its canvas lengths `classes`), whose counters it keeps."""
    n = slots.numel()
    b, _ = _classes(classes, classes) if lane is not None else ((0,) * 4, ())
    _control[(1,)](mode, step, slots, blocks, ctrl, counts, lane if lane is not None else ctrl, n,
                   triton.next_power_of_2(n), rows, head, lane is not None, *b, **LAUNCH)


