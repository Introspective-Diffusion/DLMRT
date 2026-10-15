"""Tile sweep of DiffusionGemma's Triton attention (dlmrt.ops.triton.attention) at its shapes (TP=1:
full layers 16 q / 2 kv heads x 512, sliding 16 / 8 x 256), canvas (bidirectional) and commit (causal) passes at
several contexts; CUDA-graph replay times. Every config is checked against the smallest tiles.

usage: python3 benchmarks/attention_sweep.py [hq_full hk_full hq_sliding hk_sliding [full|sliding]]
       e.g. 2 1 2 1 for one rank at TP=8. CHECK=1: correctness only, one call per config (run it under
       compute-sanitizer before timing configs that never ran on this GPU).
"""

import itertools
import os
import sys

import torch
import triton

from dlmrt.models.diffusion_gemma.attention import tiles
from dlmrt.ops.triton.attention import attention

T, W, dev = 256, 1024, "cuda"
# Fits every GPU's shared memory (SM 12.x has about 100 KB per block).
SAFE = dict(block_m=16, block_n=16, num_warps=4, num_stages=1, dv=128)


def graph_us(fn, n=20, reps=20):
    """Per-call time of fn with `reps` calls captured in one CUDA graph (graph launch cost amortized)."""
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


def main():
    hqf, hkf, hqs, hks = (int(x) for x in sys.argv[1:5]) if len(sys.argv) > 4 else (16, 2, 16, 8)
    # On B200 only tiles within TMEM (DV + BLOCK_N <= 512 columns): an overflow faults and can take the GPU down with
    # it. Shared-memory overflows are caught at compile time.
    def configs(d):
        return [dict(block_m=bm, block_n=bn, num_warps=nw, num_stages=ns, dv=dv, programs=p)
                for bm, bn, nw, ns, dv, p in itertools.product((16, 32, 64, 128), (16, 32, 64), (4, 8), (1, 2),
                                                               (128, 256, 512), (8, 16, 32, 64, 128))
                if dv <= d and (major != 10 or dv + bn <= 512)]
    major = torch.cuda.get_device_capability()[0]
    kinds = (("full", hqf, hkf, 512, False), ("sliding", hqs, hks, 256, True))
    if len(sys.argv) > 5:
        kinds = [k for k in kinds if k[0] == sys.argv[5]]
    # CHECK=1: one call per config at one context, correctness only (run it under compute-sanitizer before timing
    # configs that never ran on this GPU).
    check = os.environ.get("CHECK") == "1"
    for name, hq, hk, d, sliding in kinds:
        for ctx in (2048,) if check else (0, 2048, 8192):
            cap = ctx + T
            k = torch.randn(1, cap, hk, d, device=dev).to(torch.bfloat16)
            v = torch.randn(1, cap, hk, d, device=dev).to(torch.bfloat16)
            q = (torch.randn(1, T, hq, d, device=dev) * 0.05).to(torch.bfloat16)
            s = torch.tensor([ctx], device=dev, dtype=torch.int32)
            e = s + T
            left = torch.tensor([max(0, ctx - (W - 1))], device=dev, dtype=torch.int32)
            out = torch.empty_like(q)
            for causal in (False, True):
                kw = dict(window=W if causal and sliding else 0, left=left if sliding and not causal else None)
                ref = attention(q, k, v, s, e, causal, **kw, **SAFE)
                results = []
                for cfg in configs(d):
                    try:
                        o = attention(q, k, v, s, e, causal, out=out, **kw, **cfg)
                    except triton.runtime.errors.OutOfResources:
                        continue
                    torch.cuda.synchronize()
                    err = ((o.float() - ref.float()).norm() / ref.float().norm()).item()
                    if err > 1e-2:
                        print("wrong result", cfg, err)
                        continue
                    if check:
                        continue
                    results.append((graph_us(lambda: attention(q, k, v, s, e, causal, out=out, **kw, **cfg)), cfg))
                if check:
                    print(f"{name} checked {len(configs(d))} configs", flush=True)
                    continue
                # The engine's tiles on this GPU.
                engine = tiles(major, sliding, hk)
                used = graph_us(lambda: attention(q, k, v, s, e, causal, out=out, **kw, **engine))
                results.sort(key=lambda r: r[0])
                best = ", ".join(f"{us:.1f} us {c}" for us, c in results[:3])
                print(f"{name:7s} ctx {ctx:5d} {'causal' if causal else 'canvas'}: engine {used:.1f} us; best {best}",
                      flush=True)


if __name__ == "__main__":
    main()
