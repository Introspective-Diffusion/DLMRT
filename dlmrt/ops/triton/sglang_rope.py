"""SGLang's QK RMSNorm + neox RoPE, bitwise, fused with the KV-cache placement (the SGLang-numerics modes of LLaDA2 and
SDAR)."""

import triton
import triton.language as tl

from .common import LAUNCH, _release, _wait


@triton.jit
def _norm_rope(src, w, cs, eps, H: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr):
    """The [H, D] heads at src through SGLang's in-place QK RMSNorm (fused_qknorm_warp: lane l's elements 4l .. 4l + 3
    squared and summed in order, a 32-lane butterfly, rsqrt(sum / D + eps), (x * factor) * weight to BF16), then its
    neox RoPE over the first ROT dims (fused_rope_kernel as nvcc contracts it: x' = fma(x, cos, -(y * sin)),
    y' = fma(y, cos, x * sin), to BF16). Returns BF16 [H, D]."""
    h = tl.arange(0, H)[:, None] * D
    d = tl.arange(0, D)[None, :]
    x = tl.load(src + h + d).to(tl.float32)
    a, b = tl.split(tl.reshape(x, (H, D // 4, 2, 2)))
    x0, x2 = tl.split(a)
    x1, x3 = tl.split(b)
    s = tl.fma(x3, x3, tl.fma(x2, x2, tl.fma(x1, x1, x0 * x0)))
    # The warp's xor butterfly (lanes l and l ^ 16, then ^ 8, ...) as explicit pairwise adds, independent of how
    # Triton lays the lanes out over threads.
    for width in tl.static_range(4, -1, -1):
        s = tl.sum(tl.reshape(s, (H, 2, 1 << width)), 1)
    factor = tl.math.rsqrt(s / D + eps)
    n = (x * factor * tl.load(w + d).to(tl.float32)).to(tl.bfloat16)
    HALF: tl.constexpr = ROT // 2
    rotated = d < ROT
    partner = tl.where(rotated, (d + HALF) % ROT, d)  # the other element of d's rotation pair
    other = (tl.load(src + h + partner).to(tl.float32) * factor * tl.load(w + partner).to(tl.float32)).to(
        tl.bfloat16).to(tl.float32)
    j = d % HALF
    cos = tl.load(cs + j, mask=rotated, other=0.0)
    sin = tl.load(cs + HALF + j, mask=rotated, other=0.0)
    sign = tl.where(d < HALF, -1.0, 1.0)
    return tl.where(rotated, tl.fma(n.to(tl.float32), cos, sign * (other * sin)).to(tl.bfloat16), n)


@triton.jit
def _qk_norm_rope_place(QKV, Q, KC, VC, POS, COS_SIN, W, eps, QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
                        ROT: tl.constexpr, WRITE_BACK: tl.constexpr):
    """Program (row r, part): part 0 the q heads into Q[r]; part 1 the k heads into the K cache at slot POS[r] (and
    with WRITE_BACK back into qkv) and the v heads into the V cache. W: [q_norm, k_norm]; COS_SIN: [positions, ROT],
    the cosines then the sines."""
    _wait()
    r = tl.program_id(0)
    slot = tl.load(POS + r).to(tl.int64)
    row = QKV + r.to(tl.int64) * ((QH + 2 * KH) * D)
    cs = COS_SIN + slot * ROT
    d = tl.arange(0, D)[None, :]
    if tl.program_id(1) == 0:
        q = _norm_rope(row, W, cs, eps, QH, D, ROT)
        tl.store(Q + r.to(tl.int64) * (QH * D) + tl.arange(0, QH)[:, None] * D + d, q)
    else:
        hk = tl.arange(0, KH)[:, None] * D
        k = _norm_rope(row + QH * D, W + D, cs, eps, KH, D, ROT)
        tl.store(KC + slot * (KH * D) + hk + d, k)
        if WRITE_BACK:
            tl.store(row + QH * D + hk + d, k)
        tl.store(VC + slot * (KH * D) + hk + d, tl.load(row + (QH + KH) * D + hk + d))
    _release()


def qk_norm_rope_place(qkv, q, k_cache, v_cache, positions, cos_sin, qk_norm, eps, q_heads, kv_heads, head_dim,
                       rotary_dim, write_back=True):
    """qkv [rows, (q_heads + 2 kv_heads) head_dim] BF16 at `positions` (int64, = KV slots): q into q [rows, q_heads,
    head_dim], K / V into the caches [slots, 1, kv_heads, head_dim]; `write_back`: the normalized K into qkv too.
    Head counts are powers of two."""
    _qk_norm_rope_place[(qkv.shape[0], 2)](
        qkv, q, k_cache, v_cache, positions, cos_sin, qk_norm, eps, q_heads, kv_heads, head_dim, rotary_dim, write_back,
        num_warps=8, **LAUNCH)
