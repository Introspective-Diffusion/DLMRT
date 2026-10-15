"""LLaDA2 pieces of the NVFP4 / Blackwell path: FP8 KV rounding of qk_rope, the block-causal Triton attention that
replaces FA3 off Hopper, the SwiGLU NVFP4 MoE against the W4A4 math in torch (TRT-LLM-gen on SM 10.x, dlmrt's Triton
MoE on SM 12.x), and the B200-order GEMM and attention of SM 12.x."""

import pytest
import torch

from test_nvfp4 import fake_quantize, modelopt_quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
dev = "cuda"
HQ, HK, D, R, B = 16, 4, 128, 64, 32


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_qk_rope_fp8_kv():
    from dlmrt.models.llada2 import kernels as lk

    rows = 64
    qkv = torch.randn(rows, (HQ + 2 * HK) * D, device=dev).to(torch.bfloat16)
    qw, kw = (torch.rand(D, device=dev) + 0.5).to(torch.bfloat16), (torch.rand(D, device=dev) + 0.5).to(torch.bfloat16)
    angle = torch.arange(1024, device=dev)[:, None].float() * (1e-4 ** (torch.arange(0, R, 2, device=dev) / R))[None]
    rope = torch.cat((angle.cos(), angle.sin()), -1).contiguous()
    start = torch.tensor([96], device=dev, dtype=torch.int32)
    out = {}
    for fp8 in (False, True):
        q = torch.empty(1, rows, HQ, D, device=dev, dtype=torch.bfloat16)
        kc, vc = (torch.zeros(1024, HK, D, device=dev, dtype=torch.bfloat16) for _ in range(2))
        lk.qk_rope(qkv, qw, kw, rope, start, q, kc, vc, HQ, HK, D, R, 1e-6, kv_fp8=fp8)
        out[fp8] = q, kc, vc
    assert torch.equal(out[True][0], out[False][0])
    for plain, rounded in zip(out[False][1:], out[True][1:]):
        assert torch.equal(rounded, plain.to(torch.float8_e4m3fn).to(torch.bfloat16))
        assert not torch.equal(rounded, plain)


@pytest.mark.parametrize("rows, grid", [(32, 0), (96, B)])
def test_block_causal_attention(rows, grid):
    """One block over [0, end) (grid 0), and three blocks block-causal on the 32-token grid, with softmax scale
    D^-1/2, against FP32 attention under the same mask."""
    from dlmrt.ops.triton.attention import attention

    begin = 160
    end = begin + rows
    q = torch.randn(1, rows, HQ, D, device=dev).to(torch.bfloat16)
    k, v = (torch.randn(1, 1024, HK, D, device=dev).to(torch.bfloat16) for _ in range(2))
    start, stop = (torch.tensor([x], device=dev, dtype=torch.int32) for x in (begin, end))
    out = attention(q, k, v, start, stop, False, scale=D ** -0.5, grid=grid, dv=D, block_m=64, block_n=64)
    qpos = begin + torch.arange(rows, device=dev)
    kpos = torch.arange(end, device=dev)
    visible = kpos[None] < ((qpos[:, None] // B + 1) * B if grid else end)
    kk, vv = (t[0, :end].float().repeat_interleave(HQ // HK, 1).transpose(0, 1) for t in (k, v))
    s = (q[0].float().transpose(0, 1) @ kk.transpose(1, 2)) * D ** -0.5
    ref = (s.masked_fill(~visible, float("-inf")).softmax(-1) @ vv).transpose(0, 1)
    assert ((out[0].float() - ref).norm() / ref.norm()).item() < 1e-2


def swiglu_case():
    """LLaDA2's experts (SwiGLU, 256 x [512, 2048], top-8, scaled BF16 routing weights) as NVFP4 exports, routed inputs,
    W4A4 in torch with the same NVFP4 roundings and the unquantized result: (parts, x, ids, weights, ref, truth)."""
    from dlmrt.quantization import nvfp4

    h, f, e, k, t = 2048, 512, 256, 8, 64
    a13, a2 = torch.tensor(3.0 / 2688, device=dev), torch.tensor(0.4 / 2688, device=dev)
    parts = {}
    for x in range(e):
        gate, up, down = (torch.randn(*s, device=dev) * 0.02 for s in ((f, h), (f, h), (h, f)))
        s13 = torch.maximum(gate.abs().max(), up.abs().max()) / 2688
        for proj, w, s2, a in (("gate", gate, s13, a13), ("up", up, s13, a13), ("down", down, down.abs().max() / 2688,
                                                                                   a2)):
            packed, scale = modelopt_quantize(w, s2)
            parts[(0, x, proj, "weight")], parts[(0, x, proj, "weight_scale")] = packed, scale
            parts[(0, x, proj, "weight_scale_2")], parts[(0, x, proj, "input_scale")] = s2.float(), a.float()
    w13, w2 = nvfp4.expert_weights(parts, 0, 0, e, dev)
    x = (torch.randn(t, h, device=dev) * 0.5).to(torch.bfloat16)
    ids = torch.rand(t, e, device=dev).topk(k, -1).indices.sort(-1).values.int()
    weights = torch.rand(t, k, device=dev)
    weights = (weights / weights.sum(-1, keepdim=True) * 2.5).to(torch.bfloat16).float()
    ref, truth = torch.zeros(t, h, device=dev), torch.zeros(t, h, device=dev)
    xq = fake_quantize(x, a13).float()
    for x_ in range(e):
        rows, slot = torch.where(ids == x_)
        if rows.numel():
            mid = (xq[rows] @ w13[x_].float().t()).to(torch.bfloat16).float()
            act = (torch.nn.functional.silu(mid[:, :f]) * mid[:, f:]).to(torch.bfloat16)
            ref[rows] += (fake_quantize(act, a2).float() @ w2[x_].float().t()) * weights[rows, slot, None]
            mid = x[rows].float() @ w13[x_].float().t()
            truth[rows] += ((torch.nn.functional.silu(mid[:, :f]) * mid[:, f:]) @ w2[x_].float().t()) * \
                weights[rows, slot, None]
    return parts, x, ids, weights, ref, truth


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                    reason="needs SM 10.x FP4 tensor cores")
def test_trtllm_swiglu_moe_matches_w4a4_reference():
    """FlashInfer's TRT-LLM-gen MoE against the W4A4 math in torch. As test_nvfp4's GeGLU case, the activation's FP4
    rounding flips on BF16-level differences, so they agree to a few percent and sit equally far from unquantized
    math."""
    from dlmrt.quantization import nvfp4

    parts, x, ids, weights, ref, truth = swiglu_case()
    out = torch.empty_like(x)
    nvfp4.trtllm_moe(x, nvfp4.pack(ids, weights), nvfp4.trtllm_experts(parts, 0, 0, 256, dev), out, 256, 0, 512,
                     activation="swiglu")
    assert rel(out, ref) < 0.08
    assert abs(rel(out, truth) - rel(ref, truth)) < 0.02


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
                    reason="needs SM 12.x FP4 tensor cores")
def test_triton_swiglu_moe_matches_w4a4_reference():
    """dlmrt's Triton NVFP4 MoE (SM 12.x) against the same W4A4 math, and deterministic."""
    from dlmrt.ops.triton import nvfp4_moe
    from dlmrt.quantization import nvfp4

    parts, x, ids, weights, ref, truth = swiglu_case()
    t, k = ids.shape
    amap = torch.zeros(256, t, device=dev, dtype=torch.int32)
    count = [0] * 256
    for i, e in enumerate(ids.flatten().tolist()):
        amap[e, count[e]] = i
        count[e] += 1
    counts = torch.tensor(count, device=dev, dtype=torch.int32)
    experts = nvfp4.triton_experts(parts, 0, 0, 256, dev)
    buffers = nvfp4_moe.Buffers(t, 2048, 512, 256, k, dev)
    outs = [nvfp4_moe.moe(x, ids, weights, amap, counts, experts, buffers, torch.empty_like(x)) for _ in range(2)]
    assert torch.equal(outs[0], outs[1])
    assert rel(outs[0], ref) < 0.08
    assert abs(rel(outs[0], truth) - rel(ref, truth)) < 0.02


@pytest.mark.parametrize("m, n, k, split", [(32, 2048, 2048, 1), (32, 256, 2048, 2), (64, 2048, 5120, 2),
                                            (32, 2048, 157184 // 8, 9)])
def test_b200_order_gemm_depends_only_on_the_k_order(m, n, k, split):
    """dlmrt.ops.triton.gemm: the result is fixed by the K ranges alone (any tiles give the same bits), the split
    result is the in-order sum of the per-range products, and both are BF16 GEMMs."""
    import triton

    from dlmrt.ops.triton import gemm

    a = torch.randn(m, k, device=dev).to(torch.bfloat16)
    w = torch.randn(n, k, device=dev).to(torch.bfloat16)
    out = gemm.mm(a, w, split=split)
    ks = triton.cdiv(k // 64, split) * 64
    p = torch.empty(split * m * n, device=dev)
    other = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    gemm._mm[(triton.cdiv(m, 16), triton.cdiv(n, 16), split)](a, w, other, p if split > 1 else other, m, n, k, ks,
                                                              w.stride(1), w.stride(0), 16, 16, 64, split,
                                                              num_warps=4, num_stages=2)
    if split > 1:
        parts = p.view(split, m, n)
        other = parts[0].clone()
        for s in range(1, split):
            other = other + parts[s]
        other = other.to(torch.bfloat16)
    assert torch.equal(out, other)
    ref = a.double() @ w.double().t()
    assert ((out.double() - ref).norm() / ref.norm()).item() < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
                    reason="Gluon kernel for SM 12.x")
@pytest.mark.parametrize("rows, grid", [(32, 0), (96, B)])
def test_b200_order_attention_matches_triton_attention(rows, grid):
    """The Gluon attention with the B200's summation order against the Triton kernel it mirrors."""
    from dlmrt.ops.triton import b200_attention
    from dlmrt.ops.triton.attention import attention

    q = torch.randn(1, rows, HQ, D, device=dev).to(torch.bfloat16)
    k, v = (torch.randn(1, 4096, HK, D, device=dev).to(torch.bfloat16) for _ in range(2))
    start, stop = (torch.tensor([x], device=dev, dtype=torch.int32) for x in (1024, 1024 + rows))
    out = b200_attention.attention(q, k, v, start, stop, scale=D ** -0.5, grid=grid)
    ref = attention(q, k, v, start, stop, False, scale=D ** -0.5, grid=grid, dv=D, block_m=64, block_n=64)
    assert rel(out, ref) < 1e-3
