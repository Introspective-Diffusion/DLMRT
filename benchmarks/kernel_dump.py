"""Dump the outputs of DiffusionGemma's row kernels on seeded inputs, to compare two dlmrt trees bit for bit.

usage: PYTHONPATH=<tree A> python3 benchmarks/kernel_dump.py a.pt; PYTHONPATH=<tree B> python3 benchmarks/kernel_dump.py b.pt
       python3 benchmarks/kernel_dump.py --compare a.pt b.pt
"""

import sys

import torch

T, H, EPS, S = 256, 2816, 1e-6, 4096


def dump(path):
    from dlmrt.models.diffusion_gemma import kernels as gk

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)

    def bf(*shape, scale=1.0):
        return (torch.randn(*shape, device=dev, generator=g) * scale).to(torch.bfloat16)

    out = {}
    h, a, y1, y2 = bf(T, H), bf(T, H), bf(T, H), bf(T, H)
    w = [bf(H, scale=0.1) + 1 for _ in range(5)]
    hp = h.clone()
    x1, x2, r = (torch.empty(T, H, device=dev, dtype=torch.bfloat16) for _ in range(3))
    gk.post_attn(hp, a, w[0], w[1], w[2], w[3], x1, x2, r, EPS)
    out.update(post_attn_h=hp, post_attn_x1=x1, post_attn_x2=x2, post_attn_r=r)
    hf, xn = h.clone(), torch.empty(T, H, device=dev, dtype=torch.bfloat16)
    gk.post_ffn(hf, y1, y2, w[0], w[1], w[2], w[3], xn, 0.75, EPS)
    out.update(post_ffn_h=hf, post_ffn_x=xn)
    start = torch.tensor([1000], device=dev, dtype=torch.int32)
    for hq, hk in ((2, 1), (16, 8)):
        for d, has_v in ((256, True), (512, False)):
            qkv = bf(T, (hq + (2 if has_v else 1) * hk) * d)
            cos, sin = bf(S, d // 2), bf(S, d // 2)
            qw, kw = bf(d) + 1, bf(d) + 1
            for split in (False, True):
                for fp8 in (False, True):
                    q = torch.empty(1, T, hq, d, device=dev, dtype=torch.bfloat16)
                    k = torch.zeros(1, S, hk, d, device=dev, dtype=torch.bfloat16)
                    v = torch.zeros_like(k)
                    gk.qkv_rope(qkv, qw, kw, cos, sin, start, q, k, v, hq, hk, d, has_v, EPS, kv_fp8=fp8, split=split)
                    key = f"qkv_{hq}x{hk}_d{d}_split{int(split)}_fp8{int(fp8)}"
                    out[key + "_q"], out[key + "_k"], out[key + "_v"] = q, k, v
    torch.cuda.synchronize()
    torch.save({k: v.cpu() for k, v in out.items()}, path)
    print(f"dumped {len(out)} tensors to {path}")


def compare(a, b):
    x, y = torch.load(a), torch.load(b)
    bad = [k for k in x if not torch.equal(x[k], y[k])]
    for k in bad:
        d = (x[k].float() - y[k].float()).abs()
        print(f"DIFF {k}: {int((d > 0).sum())} elements, max {d.max().item():.3g}")
    print(f"{len(x) - len(bad)} / {len(x)} tensors bit-identical")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        dump(sys.argv[1])
