"""W4A4 NVFP4 routed experts on FP4 tensor cores (`tl.dot_scaled`, SM 12.x), with the numerics of FlashInfer's
TRT-LLM-gen MoE on SM 10.x:

    x (BF16) -> NVFP4 against the GEMM1 input scale -> GEMM1 (FP32) x alpha -> SwiGLU (FP32) -> NVFP4 against the
    GEMM2 input scale -> GEMM2 (FP32) x alpha -> BF16 per (token, expert) -> sum over the token's experts (FP32,
    BF16 routing weights) -> BF16

NVFP4 rounding is FlashInfer's cvt_warp_fp16_to_fp4 (fast math): per 16 values, scale = E4M3(g * amax * rcp(6)),
values times rcp(scale * rcp(g)) rounded to E2M1 (nearest, ties to even, saturating), with the approximate
reciprocal of PTX. Experts see their rows through `amap` (expert-major slots holding token * top_k + k), the layout
`dlmrt.models.llada2.kernels.route` writes.
"""

import torch
import triton
import triton.language as tl

from .common import LAUNCH, _release, _row_tile, _wait


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _e2m1(v):
    """E2M1 code of FP32 v (sign in bit 3): nearest, ties to the even code, saturating at 6."""
    a = tl.abs(v)
    c = tl.where(a <= 0.25, 0, tl.where(a < 0.75, 1, tl.where(a <= 1.25, 2, tl.where(a < 1.75, 3, tl.where(
        a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(a <= 5.0, 6, 7)))))))
    return c | tl.where(v < 0, 8, 0)


@triton.jit
def _nvfp4(v, g, BM: tl.constexpr, BN: tl.constexpr):
    """NVFP4 of an FP32 [BM, BN] tile against global scale g: codes [BM, BN / 2] (even element in the low nibble),
    E4M3 scales [BM, BN / 16]."""
    b = tl.reshape(v, (BM, BN // 16, 16))
    amax = tl.max(tl.abs(b), 2)
    gs = amax * 0.0 + g
    sf = (gs * (amax * _rcp(amax * 0.0 + 6.0))).to(tl.float8e4nv)
    scale = tl.where(amax != 0, _rcp(sf.to(tl.float32) * _rcp(gs)), 0.0)
    c = tl.reshape(_e2m1(b * scale[:, :, None]), (BM, BN // 2, 2))
    lo, hi = tl.split(c)
    return (lo | (hi << 4)).to(tl.uint8), sf


@triton.jit
def _quantize(X, G, Q, S, N: tl.constexpr, BN: tl.constexpr):
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    cols = part * BN + tl.arange(0, BN)
    v = tl.load(X + row * N + cols[None, :]).to(tl.float32)
    codes, sf = _nvfp4(v, tl.load(G), 1, BN)
    tl.store(Q + row * (N // 2) + part * (BN // 2) + tl.arange(0, BN // 2)[None, :], codes)
    tl.store(S + row * (N // 16) + part * (BN // 16) + tl.arange(0, BN // 16)[None, :], sf)
    _release()


def quantize(x, global_scale, codes, scales):
    """x BF16 [rows, N] -> NVFP4 codes [rows, N / 2] uint8 and E4M3 scales [rows, N / 16] (linear layout)."""
    rows, n = x.shape
    bn = 512 if n % 512 == 0 else 128
    _quantize[(rows, n // bn)](x, global_scale, codes, scales, n, bn, num_warps=4, **LAUNCH)


@triton.jit
def _gate_up(XQ, XS, AMAP, COUNTS, W, WS, G1, A2I, AQ, AS, lo, TOP_K: tl.constexpr, M_MAX: tl.constexpr,
             F: tl.constexpr, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """One (expert, BN intermediate columns, BM of the expert's rows): GEMM1 (gate and up), SwiGLU in FP32,
    NVFP4."""
    _wait()
    e, n, m0 = tl.program_id(0), tl.program_id(1), tl.program_id(2) * BM
    count = tl.load(COUNTS + lo + e)
    m = tl.arange(0, BM)
    cols = n * BN + tl.arange(0, BN)
    kb, ks = tl.arange(0, BK // 2), tl.arange(0, BK // 16)
    gate = e.to(tl.int64) * 2 * F + cols
    up = gate + F
    g1, a2i = tl.load(G1 + e), tl.load(A2I + e)
    if m0 < count:
        valid = m0 + m < count
        slot = e.to(tl.int64) * M_MAX + m0 + m
        row = (tl.load(AMAP + (lo + e).to(tl.int64) * M_MAX + m0 + m, valid, 0) // TOP_K).to(tl.int64)
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, H, BK):
            a = tl.load(XQ + row[:, None] * (H // 2) + k0 // 2 + kb[None, :])
            sa = tl.load(XS + row[:, None] * (H // 16) + k0 // 16 + ks[None, :])
            wg = tl.load(W + gate[None, :] * (H // 2) + k0 // 2 + kb[:, None])
            sg = tl.load(WS + gate[:, None] * (H // 16) + k0 // 16 + ks[None, :])
            acc_g = tl.dot_scaled(a, sa, "e2m1", wg, sg, "e2m1", acc_g)
            wu = tl.load(W + up[None, :] * (H // 2) + k0 // 2 + kb[:, None])
            su = tl.load(WS + up[:, None] * (H // 16) + k0 // 16 + ks[None, :])
            acc_u = tl.dot_scaled(a, sa, "e2m1", wu, su, "e2m1", acc_u)
        g = acc_g * g1
        codes, sf = _nvfp4(g * tl.sigmoid(g) * (acc_u * g1), a2i, BM, BN)
        tl.store(AQ + slot[:, None] * (F // 2) + n * (BN // 2) + tl.arange(0, BN // 2)[None, :], codes,
                 valid[:, None])
        tl.store(AS + slot[:, None] * (F // 16) + n * (BN // 16) + tl.arange(0, BN // 16)[None, :], sf,
                 valid[:, None])
    _release()


@triton.jit
def _down(AQ, AS, AMAP, COUNTS, W, WS, G2, Y, lo, M_MAX: tl.constexpr, N: tl.constexpr, F: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """One (expert, BN output columns, BM of the expert's rows): GEMM2, BF16 into Y[token * top_k + k]."""
    _wait()
    e, n, m0 = tl.program_id(0), tl.program_id(1), tl.program_id(2) * BM
    count = tl.load(COUNTS + lo + e)
    m = tl.arange(0, BM)
    cols = n * BN + tl.arange(0, BN)
    kb, ks = tl.arange(0, BK // 2), tl.arange(0, BK // 16)
    w = e.to(tl.int64) * N + cols
    g2 = tl.load(G2 + e)
    if m0 < count:
        valid = m0 + m < count
        slot = e.to(tl.int64) * M_MAX + m0 + m
        dst = tl.load(AMAP + (lo + e).to(tl.int64) * M_MAX + m0 + m, valid, 0).to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, F, BK):
            a = tl.load(AQ + slot[:, None] * (F // 2) + k0 // 2 + kb[None, :])
            sa = tl.load(AS + slot[:, None] * (F // 16) + k0 // 16 + ks[None, :])
            b = tl.load(W + w[None, :] * (F // 2) + k0 // 2 + kb[:, None])
            sb = tl.load(WS + w[:, None] * (F // 16) + k0 // 16 + ks[None, :])
            acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
        tl.store(Y + dst[:, None] * N + cols[None, :], (acc * g2).to(tl.bfloat16), valid[:, None])
    _release()


@triton.jit
def _combine(Y, IDS, W, S, OUT, lo, LOCAL: tl.constexpr, K: tl.constexpr, N: tl.constexpr, G: tl.constexpr,
             HAS_S: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    cols, mask = _row_tile(N, G)
    acc = tl.zeros((G, 128), dtype=tl.float32)
    for k in tl.static_range(K):
        e = tl.load(IDS + row * K + k) - lo
        local = (e >= 0) & (e < LOCAL)
        y = tl.load(Y + (row * K + k).to(tl.int64) * N + cols, mask & local, 0.0).to(tl.float32)
        acc += tl.where(local, tl.load(W + row * K + k), 0.0) * y
    out = acc.to(tl.bfloat16)
    if HAS_S:
        out = (out.to(tl.float32) + tl.load(S + row * N + cols, mask, 0.0).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * N + cols, out, mask)
    _release()


class Buffers:
    """Scratch of one MoE layer call for up to `rows` tokens, shared by every layer."""

    def __init__(self, rows, hidden, intermediate, local_experts, top_k, device):
        u8 = {"device": device, "dtype": torch.uint8}
        self.m_max = rows
        self.xq = torch.zeros(rows, hidden // 2, **u8)
        self.xs = torch.zeros(rows, hidden // 16, device=device, dtype=torch.float8_e4m3fn)
        self.aq = torch.zeros(local_experts * rows, intermediate // 2, **u8)
        self.as_ = torch.zeros(local_experts * rows, intermediate // 16, device=device, dtype=torch.float8_e4m3fn)
        self.y = torch.zeros(rows * top_k, hidden, device=device, dtype=torch.bfloat16)


def moe(x, ids, weights, amap, counts, experts, buffers, out, shared=None):
    """out = sum over each token's experts in this rank's [lo, lo + count) of weight * expert(x) [+ shared], BF16.
    `ids` int32 / `weights` FP32 [rows, top_k] (the BF16-rounded routing weights), `amap` / `counts` the
    expert-major slots and per-expert row counts of `route`. `experts`: see `dlmrt.quantization.nvfp4.triton_experts`."""
    rows, h = x.shape
    k = ids.shape[1]
    n_local, two_f = experts["w1"].shape[0], experts["w1"].shape[1]
    f, lo, b = two_f // 2, experts["lo"], buffers
    # (BM, BN, BK, warps, stages): decode rounds stream a hundred-odd experts' weights for a few rows each;
    # prefill chunks touch nearly every expert with tens of rows.
    gu, dn = ((16, 64, 256, 8, 2), (32, 128, 128, 4, 3)) if rows <= 64 else \
        ((32, 128, 256, 4, 2), (128, 128, 128, 4, 3))
    quantize(x, experts["a1_inv"], b.xq, b.xs)
    _gate_up[(n_local, f // gu[1], triton.cdiv(rows, gu[0]))](
        b.xq, b.xs, amap, counts, experts["w1"], experts["s1"], experts["g1"], experts["a2_inv"], b.aq, b.as_, lo, k,
        b.m_max, f, h, *gu[:3], num_warps=gu[3], num_stages=gu[4], **LAUNCH)
    _down[(n_local, h // dn[1], triton.cdiv(rows, dn[0]))](
        b.aq, b.as_, amap, counts, experts["w2"], experts["s2"], experts["g2"], b.y, lo, b.m_max, h, f, *dn[:3],
        num_warps=dn[3], num_stages=dn[4], **LAUNCH)
    _combine[(rows,)](b.y, ids, weights, shared if shared is not None else out, out, lo, n_local, k, h,
                      triton.next_power_of_2(triton.cdiv(h, 128)), shared is not None, num_warps=8, **LAUNCH)
    return out
