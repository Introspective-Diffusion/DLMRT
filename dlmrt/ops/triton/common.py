"""Shared Triton plumbing: programmatic dependent launch (PDL) and the row-tile / BF16-rounding helpers the model
kernels build on.

Every kernel is a programmatic dependent launch: it waits for its predecessor before touching memory and releases
its successor after its last store, so the next launch overlaps this kernel's tail. Without the launch attribute
(DLMRT_PDL=0) both are no-ops.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from ... import envs


PDL = envs.PDL()

LAUNCH = {"launch_pdl": True} if PDL else {}


@triton.jit
def _wait():
    tl.extra.cuda.gdc_wait()


@triton.jit
def _release():
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _bf(x):
    """Round to BF16 and back. In PTX because FMA contraction otherwise fuses a rounded product into the next add."""
    return tl.inline_asm_elementwise("{ .reg .b16 r; cvt.rn.bf16.f32 r, $1; cvt.f32.bf16 $0, r; }", "=f,f", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _store_bf16(ptr, x, FP8: tl.constexpr):
    """Store as BF16; with FP8, the BF16 value goes through FP8 E4M3 (unit scale, saturating): stored as such into an
    FP8 arena, or rounded and kept as BF16 in a BF16 arena."""
    y = x.to(tl.bfloat16)
    if FP8:
        y = y.to(tl.float32).to(tl.float8e4nv)
        if ptr.dtype.element_ty != tl.float8e4nv:
            y = y.to(tl.float32).to(tl.bfloat16)
    tl.store(ptr, y)


@triton.jit
def _rms(x, N: tl.constexpr, eps):
    """x / sqrt(mean(x^2) + eps) over a [groups, 128] fp32 tile of one row."""
    return x * tl.rsqrt(tl.sum(tl.sum(x * x, 1), 0) / N + eps)


@triton.jit
def _row_tile(N: tl.constexpr, G: tl.constexpr):
    g = tl.arange(0, G)
    cols = g[:, None] * 128 + tl.arange(0, 128)[None, :]
    return cols, cols < N


def _groups(n):
    return triton.next_power_of_2(triton.cdiv(n, 128))


@triton.jit
def _gelu_tanh(x):
    return 0.5 * x * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


@triton.jit
def _geglu_row(X, Y, src, dst, part, F: tl.constexpr, G: tl.constexpr):
    g = part * G + tl.arange(0, G)
    cols = g[:, None] * 128 + tl.arange(0, 128)[None, :]
    mask = cols < F
    gate = tl.load(X + src * 2 * F + cols, mask, 0.0).to(tl.float32)
    up = tl.load(X + src * 2 * F + F + cols, mask, 0.0).to(tl.float32)
    tl.store(Y + dst * F + cols, (_bf(_gelu_tanh(gate)) * up).to(tl.bfloat16), mask)
