"""SDAR's SGLang-numerics kernels (models/sdar/sglang_kernels.py) against torch transcriptions of SDARPrefixKV's steps.
End-to-end agreement with the paper runtime's records: docs/performance.md (4816 of 4820 answers)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

B, V, SEED, HIGH = 256, 151936, 12345, 151669
dev = "cuda"


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def block_state(prompt_rows, accepted_rows):
    tokens = torch.randint(0, HIGH, (B,), device=dev)
    prompt = torch.arange(B, device=dev) < prompt_rows
    accepted = prompt | (torch.rand(B, device=dev) < accepted_rows / B)
    step = torch.tensor([3], device=dev, dtype=torch.int32)
    return tokens, accepted, prompt, step, torch.zeros(1, device=dev, dtype=torch.int32), torch.tensor([2], device=dev)


def reference(tokens, accepted, prompt, step, proposal, pp, accept, edit_threshold):
    """SDARPrefixKV after the accept mask: edits, updates and the next step's noise for the unresolved rows."""
    edit = accepted & ~prompt & (pp > edit_threshold) & (proposal != tokens)
    tokens = torch.where(accept | edit, proposal, tokens)
    accepted = accepted | accept
    position = torch.arange(B, device=dev) + 2 * B
    noise = ((position + 1) * 1103515245 + (int(step) + 2) * 12345 + SEED).remainder(HIGH)
    return torch.where(accepted, tokens, noise), accepted


@pytest.mark.parametrize("accepted_rows", [0, 100, 250])
def test_entropy_decisions(accepted_rows):
    from dlmrt.models.sdar import DecodeConfig
    from dlmrt.models.sdar import sglang_kernels as sk

    decoder = DecodeConfig(entropy_bound=0.6)
    state = block_state(40, accepted_rows)
    tokens, accepted, prompt, step = (x.clone() for x in state[:4])
    proposal = torch.randint(0, HIGH, (B,), device=dev)
    pp, entropy = torch.rand(B, device=dev), torch.rand(B, device=dev) * 0.3
    unresolved = ~accepted & ~prompt
    sorted_entropy, order = torch.sort(torch.where(unresolved, entropy, float("inf")))
    excl = torch.cumsum(sorted_entropy, -1) - sorted_entropy
    # _build_accept_mask_fixed
    accept = torch.zeros_like(unresolved)
    accept[order] = (excl <= decoder.entropy_bound) & torch.isfinite(sorted_entropy)
    first = torch.argmax(unresolved.to(torch.int8))
    accept = accept | ((~accept.any()) & unresolved & (torch.arange(B, device=dev) == first))
    want_tokens, want_accepted = reference(tokens, accepted, prompt, step, proposal, pp, accept,
                                           decoder.edit_threshold)
    sk.decide(state, proposal, pp, decoder, order=order, sorted_entropy=sorted_entropy, excl=excl,
              scratch=torch.zeros(B, device=dev, dtype=torch.int8))
    assert torch.equal(state[0], want_tokens) and torch.equal(state[1], want_accepted)
    assert int(state[3]) == int(step) + 1 and int(state[4]) == int((~want_accepted).sum())


@pytest.mark.parametrize("threshold", [0.85, 0.999])  # positions above it, or the top-1 floor
def test_threshold_decisions(threshold):
    from dlmrt.models.sdar import DecodeConfig
    from dlmrt.models.sdar import sglang_kernels as sk

    decoder = DecodeConfig(accept_rule="prob_threshold", prob_threshold=threshold, edit_threshold=threshold,
                           confidence_temperature_min=1.0, confidence_temperature_step_decrement=0.0)
    state = block_state(40, 100)
    tokens, accepted, prompt, step = (x.clone() for x in state[:4])
    proposal = torch.randint(0, HIGH, (B,), device=dev)
    pp = torch.rand(B, device=dev) * 0.999
    eligible = ~accepted & ~prompt
    # _build_accept_mask, prob_threshold with the floor of one
    high = (pp > threshold) & eligible
    if bool(high.any()):
        accept = high
    else:
        accept = torch.zeros_like(eligible)
        accept[torch.topk(torch.where(eligible, pp, torch.full_like(pp, -float("inf"))), k=1).indices] = True
    want_tokens, want_accepted = reference(tokens, accepted, prompt, step, proposal, pp, accept,
                                           decoder.edit_threshold)
    top = torch.topk(torch.where(eligible, pp, torch.full_like(pp, -float("inf"))), k=1).indices
    sk.decide(state, proposal, pp, decoder, top=top)
    assert torch.equal(state[0], want_tokens) and torch.equal(state[1], want_accepted)


def test_row_stats_temp_match_softmax():
    from dlmrt.models.sdar import sglang_kernels as sk

    logits = (torch.randn(B, V, device=dev) * 3).to(torch.bfloat16)
    proposal = torch.zeros(B, device=dev, dtype=torch.int64)
    pmax, entropy = torch.zeros(B, device=dev), torch.zeros(B, device=dev)
    sk.row_stats_temp(logits, torch.tensor([1 / 0.8], device=dev), proposal, pmax, entropy)
    p = torch.softmax(logits.float() / 0.8, -1)
    assert torch.equal(proposal, logits.float().argmax(-1))
    assert torch.allclose(pmax, p.amax(-1), rtol=1e-4)
    assert torch.allclose(entropy, -(p * p.clamp_min(1e-30).log()).sum(-1), rtol=1e-3, atol=1e-4)


def test_silu_mul_is_sgl_kernels():
    from sgl_kernel import silu_and_mul

    from dlmrt.models.sdar import sglang_kernels as sk

    x = torch.randn(64, 2 * 9728, device=dev, dtype=torch.bfloat16) * 4
    out = torch.empty(64, 9728, device=dev, dtype=torch.bfloat16)
    sk.silu_mul(x, out)
    assert torch.equal(out, silu_and_mul(x))


def test_scaled_add_is_torch():
    """The adapter's x + scale * delta, with torch's two roundings."""
    from dlmrt.models.sdar import sglang_kernels as sk

    x, delta = (torch.randn(B, 2560, device=dev) * 3).to(torch.bfloat16), torch.randn(B, 2560, device=dev).to(
        torch.bfloat16)
    want = x + 0.37 * 1.3 * delta
    assert torch.equal(sk.scaled_add_(x, delta, 0.37 * 1.3), want)


def test_soft_embedding_is_torch():
    """The self-conditioning soft embedding: torch's softmax, then cuBLAS's split-K product per 128-row chunk."""
    from dlmrt.models.sdar import sglang_kernels as sk

    logits = (torch.randn(B, V, device=dev) * 4).to(torch.bfloat16)
    probs = torch.empty_like(logits)
    sk.soft_probs(logits, probs)
    assert torch.equal(probs, torch.softmax(logits.float(), -1).to(torch.bfloat16))
    embedding = (torch.randn(V, 2560, device=dev) * 0.02).to(torch.bfloat16)
    split = sk.cublas_split(128, embedding)
    assert split is not None
    out = torch.empty(B, 2560, device=dev, dtype=torch.bfloat16)
    sk.split_gemm(probs, embedding, out, split, *sk.split_scratch(B, 2560, split[0], dev))
    assert torch.equal(out, torch.cat([torch.mm(probs[:128], embedding), torch.mm(probs[128:], embedding)]))


@pytest.mark.parametrize("rows", [256, 1024])
def test_paged_numerics_attention_is_paged_fa3(rows):
    """The ragged TMA kernel with the paged kernel's tiles: bitwise FlashInfer's paged FA3 (page size 1), LSE too."""
    import flashinfer

    from dlmrt.ops.fa3_tma import PagedNumericsWrapper

    hq, hk, d, slots = 32, 8, 128, 12288
    k = torch.randn(slots, 1, hk, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    types = {"q_data_type": torch.bfloat16, "kv_data_type": torch.bfloat16}
    paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 << 20, dtype=torch.uint8, device=dev),
                                                           "NHD", backend="fa3")
    ours = PagedNumericsWrapper(torch.empty(128 << 20, dtype=torch.uint8, device=dev))
    for end in (1, 97, 256, 1000, 8549):
        q = torch.randn(rows, hq, d, device=dev, dtype=torch.bfloat16) * 2
        qo, kv = torch.tensor([0, rows], dtype=torch.int32), torch.tensor([0, end], dtype=torch.int32)
        paged.plan(qo, kv, torch.arange(end, dtype=torch.int32, device=dev), torch.ones(1, dtype=torch.int32), hq, hk,
                   d, 1, causal=False, **types)
        ours.plan(qo, kv, hq, hk, d, **types)
        want, want_lse = paged.forward_return_lse(q, (k, v), causal=False)
        got, got_lse = ours.forward_return_lse(q, k.view(slots, hk, d), v.view(slots, hk, d), causal=False)
        assert torch.equal(got, want) and torch.equal(got_lse, want_lse), end


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_properties(0).multi_processor_count != 132,
                    reason="ogud needs a 132-SM H100")
def test_ogud_sglang_is_separate_chain():
    """ogud's SGLang variant: bitwise the separate chain (cuBLAS O, sgl_kernel fused_add_rmsnorm, gate_up + SiLU-and-
    multiply, cuBLAS split-K down), residual and intermediates included."""
    import torch.nn.functional as F
    from sgl_kernel import fused_add_rmsnorm, silu_and_mul

    from dlmrt.ops.extensions import ogud
    from dlmrt.ops.triton.common import PDL

    def weight(n, k):
        return (torch.randn(n, k, device=dev) * k ** -0.5).to(torch.bfloat16)

    w_o, w_gate_up, w_down = weight(2560, 4096), weight(2 * 9728, 2560), weight(2560, 9728)
    norm = (1 + 0.1 * torch.randn(2560, device=dev)).to(torch.bfloat16)
    counters = torch.zeros(2, 512, device=dev, dtype=torch.int32)
    for slot, scale in enumerate((0.3, 3.0)):
        attn = (torch.randn(256, 4096, device=dev) * scale).to(torch.bfloat16)
        h = (torch.randn(256, 2560, device=dev) * 4 * scale).to(torch.bfloat16)
        want_h = h.clone()
        want_x = F.linear(attn, w_o)
        fused_add_rmsnorm(want_x, want_h, norm, 1e-6)
        want_y = silu_and_mul(F.linear(want_x, w_gate_up))
        o, x, out = torch.empty_like(h), torch.empty_like(h), torch.empty_like(h)
        y = torch.empty(256, 9728, device=dev, dtype=torch.bfloat16)
        ogud().post_attention_sglang(attn.clone(), h, w_o, norm, 1e-6, w_gate_up, w_down, o, x, y, out, counters, slot,
                                     PDL)
        assert torch.equal(h, want_h) and torch.equal(x, want_x) and torch.equal(y, want_y)
        assert torch.equal(out, F.linear(want_y, w_down))
