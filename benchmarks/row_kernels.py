"""Per-call latency of DiffusionGemma's per-layer row kernels at one TP=8 rank's shapes, each as a PDL chain of
20 calls in one CUDA graph (as in the engine's round graph), next to an empty PDL kernel (the chain's floor).

usage: python3 benchmarks/row_kernels.py
"""

import torch
import triton

from dlmrt.models.diffusion_gemma import kernels as gk
from dlmrt.ops.triton import moe
from dlmrt.ops.triton.common import LAUNCH, _release, _wait

dev, T, H, EPS = "cuda", 256, 2816, 1e-6


def graph_us(fn, n=20, reps=20):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n / reps * 1000


@triton.jit
def _empty(X):
    _wait()
    _release()


def bf(*shape, scale=1.0):
    return (torch.randn(*shape, device=dev) * scale).to(torch.bfloat16)


def main():
    x = torch.empty(1, device=dev)
    for grid in (1, 128, 256):
        print(f"empty PDL kernel, grid {grid}: {graph_us(lambda: _empty[(grid,)](x, **LAUNCH)):.2f} us")

    h, a, x1, x2, r = (bf(T, H) for _ in range(5))
    w = [bf(H, scale=0.1) + 1 for _ in range(5)]
    for nw in (4, 8, 16):
        # GS, Q, S are unused without FP4: h stands in for them, as post_attn passes it
        us = graph_us(lambda: gk._post_attn[(T,)](h, a, w[0], w[1], w[2], w[3], x1, x2, r, h, h, h, EPS, H ** -0.5,
                                                  H, gk._groups(H), False, num_warps=nw, **LAUNCH))
        print(f"post_attn num_warps {nw}: {us:.2f} us")
    for nw in (4, 8, 16):
        us = graph_us(lambda: gk._post_ffn[(T,)](h, x1, x2, w[0], w[1], w[2], w[3], r, 0.5, EPS, H, gk._groups(H),
                                                 num_warps=nw, **LAUNCH))
        print(f"post_ffn  num_warps {nw}: {us:.2f} us")

    S = 4096
    start = torch.tensor([1024], device=dev, dtype=torch.int32)
    for d, has_v in ((256, True), (512, False)):
        width = 2 + (2 if has_v else 1)
        qkv = bf(T, width * d)
        q = torch.empty(1, T, 2, d, device=dev, dtype=torch.bfloat16)
        k = torch.zeros(1, S, 1, d, device=dev, dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        cos, sin = bf(S, d // 2), bf(S, d // 2)
        qw, kw = bf(d) + 1, bf(d) + 1
        for split in (False, True):
            us = graph_us(lambda: gk.qkv_rope(qkv, qw, kw, cos, sin, start, q, k, v, 2, 1, d, has_v, EPS,
                                              kv_fp8=True, split=split))
            print(f"qkv_rope d {d} split {split}: {us:.2f} us")

    logits = bf(T, 128)
    pes = torch.rand(128, device=dev) + 0.5
    packed = torch.empty(T, 8, device=dev, dtype=torch.int32)
    print(f"route_packed: {graph_us(lambda: moe.route_packed(logits, pes, packed, 8)):.2f} us")

    from flashinfer import fp4_quantize
    xs = bf(T, H)
    inv = torch.tensor([0.5], device=dev)
    print(f"fp4_quantize: {graph_us(lambda: fp4_quantize(xs, inv, 16, False, False)):.2f} us")

    gu = bf(T, 640)
    out = torch.empty(T, 320, device=dev, dtype=torch.bfloat16)
    print(f"geglu: {graph_us(lambda: gk.geglu(gu, out)):.2f} us")


if __name__ == "__main__":
    main()
