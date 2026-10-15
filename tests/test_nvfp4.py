"""NVFP4 (nvfp4.py) against a torch transcription of ModelOpt's NVFP4QTensor.quantize: dequantization, the fused
expert-input quantization, the TRT-LLM-gen MoE (SM 10.x) and the CUTLASS MoE (SM 12.x); and the FP8 KV rounding of
qkv_rope."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, T, E, K, F = 2816, 256, 128, 8, 704
dev = "cuda"
BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def modelopt_quantize(x, scale2):
    """NVFP4QTensor.quantize(x, 16, weights_scaling_factor_2=scale2): packed codes and FP8 block scales."""
    blocks = x.float().unflatten(-1, (-1, 16))
    scale = blocks.abs().amax(-1) / (6 * scale2)
    scale[scale == 0] = 1.0
    scale = scale.clamp(2 ** -9, 448).to(torch.float8_e4m3fn)
    y = (blocks / (scale.float() * scale2)[..., None]).flatten(-2)
    a = y.abs()
    bounds = torch.tensor(BOUNDS, device=x.device)
    code = torch.searchsorted(bounds, a.contiguous()).to(torch.uint8)
    code += (a[..., None] == bounds[[1, 3, 5]]).any(-1).to(torch.uint8)
    code += (y < 0).to(torch.uint8) << 3
    return (code[..., 1::2] << 4) | code[..., 0::2], scale


def fake_quantize(x, scale2):
    """The NVFP4 round trip, decoded independently of the packing: sign and magnitude per element."""
    blocks = x.float().unflatten(-1, (-1, 16))
    scale = blocks.abs().amax(-1) / (6 * scale2)
    scale[scale == 0] = 1.0
    s = (scale.clamp(2 ** -9, 448).to(torch.float8_e4m3fn).float() * scale2)[..., None]
    y = blocks / s
    values, bounds = torch.tensor(VALUES, device=x.device), torch.tensor(BOUNDS, device=x.device)
    a = y.abs()
    mag = values[torch.searchsorted(bounds, a.contiguous()) + (a[..., None] == bounds[[1, 3, 5]]).any(-1)]
    return (torch.where(y < 0, -mag, mag) * s).flatten(-2).to(torch.bfloat16)


def test_dequantize_matches_round_trip():
    from dlmrt.quantization import nvfp4

    w = torch.randn(2 * F, H, device=dev) * 0.02
    w[:, :16] = 0  # an all-zero block
    scale2 = w.abs().max() / (6 * 448)
    packed, scale = modelopt_quantize(w, scale2)
    out = nvfp4.dequantize(packed, scale, scale2)
    assert torch.equal(out, fake_quantize(w, scale2))
    assert ((out.float() - w).norm() / w.norm()).item() < 0.12


def test_route_packed_matches_route():
    from dlmrt.ops.triton import moe
    from dlmrt.quantization import nvfp4

    logits = torch.randn(T, E, device=dev).to(torch.bfloat16)
    pes = (torch.rand(E, device=dev) + 0.5).to(torch.bfloat16)
    ids = torch.zeros(T, K, device=dev, dtype=torch.int32)
    slots, counts, weights = torch.zeros_like(ids), torch.zeros(E, device=dev, dtype=torch.int32), torch.zeros(T, K, device=dev)
    moe.route(logits, pes, ids, weights, slots, counts, K)
    packed = torch.zeros_like(ids)
    moe.route_packed(logits, pes, packed, K)
    assert torch.equal(packed, nvfp4.pack(ids, weights))
    top_ids, top_weights = torch.zeros_like(ids), torch.zeros_like(weights)
    moe.route_topk(logits, pes, top_ids, top_weights, torch.empty(T, E, device=dev, dtype=torch.int32), K)
    assert torch.equal(top_ids, ids) and torch.equal(top_weights, weights)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                    reason="needs Blackwell FP4 conversions")
def test_post_attn_quantizes_like_fp4_quantize():
    """post_attn's fused NVFP4 expert input equals FlashInfer's fp4_quantize of its BF16 expert input, byte for byte;
    the other outputs are unchanged."""
    from flashinfer import fp4_quantize

    from dlmrt.models.diffusion_gemma import kernels as gk

    h, attn = torch.randn(T, H, device=dev).to(torch.bfloat16), torch.randn(T, H, device=dev).to(torch.bfloat16)
    h[0], attn[0] = 0, 0  # an all-zero row: zero block maxima
    w = [(torch.randn(H, device=dev) * 0.1 + 1).to(torch.bfloat16) for _ in range(4)]
    outs = [torch.empty(T, H, device=dev, dtype=torch.bfloat16) for _ in range(6)]
    gk.post_attn(h.clone(), attn, *w, outs[0], outs[1], outs[2], 1e-6)
    gs = (2688 / outs[1].float().abs().amax() / 1.7).reshape(1)  # ModelOpt-style 1 / input_scale, some clipping
    codes = torch.empty(T, H // 2, device=dev, dtype=torch.uint8)
    scales = torch.empty(T, H // 16, device=dev, dtype=torch.uint8)
    gk.post_attn(h.clone(), attn, *w, outs[3], outs[4], outs[5], 1e-6, fp4=(gs, codes, scales))
    xq, xs = fp4_quantize(outs[1], gs, 16, False, False)
    assert torch.equal(codes, xq.view(torch.uint8).reshape(T, -1))
    assert torch.equal(scales, xs.view(torch.uint8).reshape(T, -1))
    assert torch.equal(outs[0], outs[3]) and torch.equal(outs[2], outs[5])


def w4a4_case():
    """Random NVFP4 experts and routed inputs, with the W4A4 math in torch (NVFP4 inputs and GeGLU outputs,
    dequantized weights) and the unquantized result: (parts, x, ids, weights, reference, truth)."""
    from dlmrt.ops.triton import moe
    from dlmrt.quantization import nvfp4

    a13, a2 = torch.tensor(3.0 / 2688, device=dev), torch.tensor(0.4 / 2688, device=dev)
    parts = {}
    for e in range(E):
        gate, up, down = (torch.randn(*s, device=dev) * 0.02 for s in ((F, H), (F, H), (H, F)))
        s13 = torch.maximum(gate.abs().max(), up.abs().max()) / 2688
        for proj, w, s2, a in (("gate", gate, s13, a13), ("up", up, s13, a13), ("down", down, down.abs().max() / 2688, a2)):
            packed, scale = modelopt_quantize(w, s2)
            parts[(0, e, proj, "weight")], parts[(0, e, proj, "weight_scale")] = packed, scale
            parts[(0, e, proj, "weight_scale_2")], parts[(0, e, proj, "input_scale")] = s2.float(), a.float()
    w13, w2 = nvfp4.expert_weights(parts, 0, 0, E, dev)
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    logits = torch.randn(T, E, device=dev).to(torch.bfloat16)
    ids = torch.zeros(T, K, device=dev, dtype=torch.int32)
    slots, counts, weights = torch.zeros_like(ids), torch.zeros(E, device=dev, dtype=torch.int32), torch.zeros(T, K, device=dev)
    moe.route(logits, torch.ones(E, device=dev, dtype=torch.bfloat16), ids, weights, slots, counts, K)
    ref, truth = torch.zeros(T, H, device=dev), torch.zeros(T, H, device=dev)
    xq = fake_quantize(x, a13).float()
    for e in range(E):
        rows, k = torch.where(ids == e)
        if rows.numel():
            mid = (xq[rows] @ w13[e].float().t()).to(torch.bfloat16).float()
            act = (torch.nn.functional.gelu(mid[:, :F], approximate="tanh") * mid[:, F:]).to(torch.bfloat16)
            ref[rows] += (fake_quantize(act, a2).float() @ w2[e].float().t()) * weights[rows, k, None]
            mid = x[rows].float() @ w13[e].float().t()
            act = torch.nn.functional.gelu(mid[:, :F], approximate="tanh") * mid[:, F:]
            truth[rows] += (act @ w2[e].float().t()) * weights[rows, k, None]
    return parts, x, ids, weights, ref, truth


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
                    reason="needs SM 10.x FP4 tensor cores")
def test_trtllm_moe_matches_w4a4_reference():
    """The TRT-LLM-gen NVFP4 MoE against the W4A4 math in torch. The GeGLU output's FP4 rounding flips on BF16-level
    differences, so the two agree to a few percent, and both are as far from the unquantized activations."""
    from dlmrt.quantization import nvfp4

    parts, x, ids, weights, ref, truth = w4a4_case()
    out = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
    nvfp4.trtllm_moe(x, nvfp4.pack(ids, weights), nvfp4.trtllm_experts(parts, 0, 0, E, dev), out, E, 0, F)
    assert rel(out, ref) < 0.08
    assert abs(rel(out, truth) - rel(ref, truth)) < 0.02


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
                    reason="needs SM 12.x FP4 tensor cores")
def test_cutlass_moe_matches_w4a4_reference():
    """The CUTLASS NVFP4 MoE (SM 12.x) against the same W4A4 math (its GeGLU's erf GELU differs from the tanh GELU far
    below the FP4 rounding)."""
    from dlmrt.quantization import nvfp4

    parts, x, ids, weights, ref, truth = w4a4_case()
    out = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
    nvfp4.cutlass_moe(x, ids, weights, nvfp4.cutlass_experts(parts, 0, 0, E, dev), out, E, T)
    assert rel(out, ref) < 0.08
    assert abs(rel(out, truth) - rel(ref, truth)) < 0.02


@pytest.mark.parametrize("sliding", [True, False])
def test_qkv_rope_fp8_kv(sliding):
    from dlmrt.models.diffusion_gemma import kernels as gk
    from dlmrt.models.diffusion_gemma.model import _rope_table

    hq, hk, d = (16, 8, 256) if sliding else (16, 2, 512)
    inv = 1.0 / 1e4 ** (torch.arange(0, d, 2, device=dev).float() / d)
    cos, sin = _rope_table(inv, 1024)
    qkv = torch.randn(T, (hq + (2 if sliding else 1) * hk) * d, device=dev).to(torch.bfloat16)
    qw, kw = (torch.rand(d, device=dev) + 0.5).to(torch.bfloat16), (torch.rand(d, device=dev) + 0.5).to(torch.bfloat16)
    start = torch.tensor([300], device=dev, dtype=torch.int32)
    out = {}
    for fp8, arena in ((False, torch.bfloat16), (True, torch.bfloat16), (True, torch.float8_e4m3fn)):
        q = torch.empty(1, T, hq, d, device=dev, dtype=torch.bfloat16)
        kc, vc = (torch.zeros(1, 1024, hk, d, device=dev, dtype=arena) for _ in range(2))
        gk.qkv_rope(qkv, qw, kw, cos, sin, start, q, kc, vc, hq, hk, d, sliding, 1e-6, kv_fp8=fp8)
        out[fp8, arena] = q, kc, vc
    plain, rounded, native = out[False, torch.bfloat16], out[True, torch.bfloat16], out[True, torch.float8_e4m3fn]
    assert torch.equal(rounded[0], plain[0]) and torch.equal(native[0], plain[0])
    for bf16, fp8, stored in zip(plain[1:], rounded[1:], native[1:]):
        assert torch.equal(fp8, bf16.to(torch.float8_e4m3fn).to(torch.bfloat16))
        assert not torch.equal(fp8, bf16)
        assert torch.equal(stored.to(torch.bfloat16), fp8)
