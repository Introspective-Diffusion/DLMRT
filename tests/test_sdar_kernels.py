"""QwenDLM's kernels against torch references: the shared row kernels at its shapes, the ThunderKittens gate_up +
SwiGLU and the post-attention megakernel (bit for bit, H100), its decoder kernels."""

import pytest
import torch

from dlmrt.ops.triton import core as kernels, decoding
from dlmrt.models.sdar import DecodeConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def dequant(q, s, m):
    return (q.float().view(m, -1, 128) * s.t().reshape(m, -1, 1)).view(m, -1)


@pytest.mark.parametrize("fp8", [False, True])
def test_add_rmsnorm(fp8):
    torch.manual_seed(0)
    h = torch.randn(256, 2560, device="cuda", dtype=torch.bfloat16)
    d = torch.randn_like(h)
    w = torch.rand(2560, device="cuda", dtype=torch.bfloat16) + 0.5
    ref_h = (h.float() + d.float())
    ref = ref_h * torch.rsqrt(ref_h.square().mean(-1, keepdim=True) + 1e-6) * w.float() * 0.5
    out = torch.empty(256, 2560, device="cuda", dtype=torch.float8_e4m3fn) if fp8 else None
    scales = torch.empty(20, 256, device="cuda", dtype=torch.float32) if fp8 else None
    y = kernels.rmsnorm(h, w, 1e-6, delta=d, scale=0.5, out=out, scales=scales)
    torch.testing.assert_close(h.float(), ref_h.to(torch.bfloat16).float())
    got = dequant(y, scales, 256) if fp8 else y.float()
    assert ((got - ref).norm() / ref.norm()).item() < (0.04 if fp8 else 4e-3)


@pytest.mark.parametrize("fp8", [False, True])
def test_swiglu(fp8):
    torch.manual_seed(1)
    x = torch.randn(256, 2 * 9728, device="cuda", dtype=torch.bfloat16)
    g, u = x.float().chunk(2, -1)
    ref = torch.nn.functional.silu(g) * u
    out = torch.empty(256, 9728, device="cuda", dtype=torch.float8_e4m3fn) if fp8 else None
    scales = torch.empty(76, 256, device="cuda", dtype=torch.float32) if fp8 else None
    y = kernels.swiglu(x, out=out, scales=scales)
    got = dequant(y, scales, 256) if fp8 else y.float()
    assert ((got - ref).norm() / ref.norm()).item() < (0.04 if fp8 else 4e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                    reason="the ThunderKittens kernel is built for Hopper (WGMMA)")
@pytest.mark.parametrize("f,k", [(9728, 2560), (512, 256)])
@pytest.mark.parametrize("round_silu", [False, True])
def test_fused_gate_up_swiglu_is_bit_exact(f, k, round_silu):
    deep_gemm = pytest.importorskip("deep_gemm")
    from dlmrt.ops.extensions import tk

    torch.manual_seed(4)
    x = torch.randn(256, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(2 * f, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    gate_up = torch.empty(256, 2 * f, device="cuda", dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(x, w, gate_up)
    ref = kernels.swiglu(gate_up, round_silu=round_silu)
    out = torch.empty_like(ref)
    tk().gate_up_swiglu(x, w, out, False, 1, 2, round_silu)
    assert torch.equal(out, ref)


def test_post_attention_megakernel_is_bit_exact():
    deep_gemm = pytest.importorskip("deep_gemm")
    from dlmrt.ops.extensions import ogud, tk

    if torch.cuda.get_device_properties(0).multi_processor_count != 132:
        pytest.skip("needs a 132-SM H100")
    torch.manual_seed(7)
    a0 = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
    h0 = torch.randn(256, 2560, device="cuda", dtype=torch.bfloat16)
    w_o = torch.randn(2560, 4096, device="cuda", dtype=torch.bfloat16) / 64
    norm = torch.rand(2560, device="cuda", dtype=torch.bfloat16) + 0.5
    w_gate_up = torch.randn(2 * 9728, 2560, device="cuda", dtype=torch.bfloat16) / 2560**0.5
    w_down = torch.randn(2560, 9728, device="cuda", dtype=torch.bfloat16) / 9728**0.5
    # Model.forward's separate kernels
    h_ref = h0.clone()
    o_ref = torch.empty_like(h0)
    deep_gemm.bf16_gemm_nt(a0, w_o, o_ref)
    x_ref = kernels.rmsnorm(h_ref, norm, 1e-6, delta=o_ref)
    y_ref = torch.empty(256, 9728, device="cuda", dtype=torch.bfloat16)
    tk().gate_up_swiglu(x_ref, w_gate_up, y_ref, False, 1, 2, False)
    ref = torch.empty_like(h0)
    deep_gemm.bf16_gemm_nt(y_ref, w_down, ref)
    counters = torch.zeros(3, 512, device="cuda", dtype=torch.int32)
    for i in range(7):  # each launch re-arms the next slot
        a, h = a0.clone(), h0.clone()  # the kernel discards `a` from L2 once read
        o, x, y, out = torch.empty_like(h), torch.empty_like(h), torch.empty_like(y_ref), torch.empty_like(h)
        ogud().post_attention(a, h, w_o, norm, 1e-6, 1.0, w_gate_up, w_down, o, x, y, out, counters, i % 3, False)
        assert torch.equal(h, h_ref) and torch.equal(x, x_ref) and torch.equal(y, y_ref) and torch.equal(out, ref)


def test_sc_probs_masks_prompt_and_first_round():
    torch.manual_seed(2)
    logits = torch.randn(256, 151936, device="cuda", dtype=torch.bfloat16)
    stats = torch.zeros(256, 3, device="cuda")
    stats[:, 0] = torch.logsumexp(logits.float(), -1)
    prompt = torch.arange(256, device="cuda") < 10
    out = torch.empty_like(logits)
    step = torch.ones(1, device="cuda", dtype=torch.int32)
    decoding.sc_probs(logits, stats, prompt, step, out)
    ref = logits.float().softmax(-1)
    ref[:10] = 0
    torch.testing.assert_close(out.float(), ref, atol=1e-3, rtol=1e-2)
    step.zero_()
    decoding.sc_probs(logits, stats, prompt, step, out)
    assert not out.any()


def test_begin_block_matches_reference_noise():
    b, cfg = 256, DecodeConfig()
    params = torch.tensor([512, 7, 768, 255], device="cuda", dtype=torch.int32)
    prompt_ids = torch.arange(b, device="cuda", dtype=torch.int64) + 1000
    tokens = torch.zeros(b, device="cuda", dtype=torch.int64)
    prompt = torch.zeros(b, device="cuda", dtype=torch.bool)
    accepted = torch.zeros_like(prompt)
    scalars = [torch.zeros(1, device="cuda", dtype=torch.int32) for _ in range(5)]
    last = torch.zeros(1, device="cuda", dtype=torch.int64)
    decoding.begin_block(params, prompt_ids, tokens, prompt, accepted, *scalars, last, cfg)
    pos = torch.arange(b, device="cuda", dtype=torch.int64)
    noise = ((pos + 768 + 1) * 1103515245 + 12345 + cfg.randtok_seed).remainder(cfg.random_token_high)
    expected = torch.where(pos < 7, prompt_ids, noise)
    assert torch.equal(tokens, expected)
    assert torch.equal(prompt, pos < 7) and torch.equal(accepted, pos < 7)
    start, end, noise_start, step, remaining = (int(s) for s in scalars)
    assert (start, end, noise_start, step, remaining, int(last)) == (512, 768, 768, 0, 249, 255)
