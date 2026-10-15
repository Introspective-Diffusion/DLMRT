"""LLaDA2 kernels against torch transcriptions of the reference ops (random weights, real sizes)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, T, E, K, F, G, TOPG = 2048, 32, 256, 8, 512, 8, 4
dev = "cuda"
MASK, EOS = 156895, 156892


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def biased_grouped_topk(logits, bias):
    """SGLang's biased_grouped_topk_impl (renormalized, unscaled), on BF16-rounded logits."""
    scores = logits.to(torch.bfloat16).float().sigmoid()
    choice = scores + bias
    group = choice.view(-1, G, E // G).topk(2, -1)[0].sum(-1)
    keep = torch.zeros_like(group).scatter_(1, group.topk(TOPG, -1)[1], 1)
    masked = choice.masked_fill(~keep.repeat_interleave(E // G, 1).bool(), float("-inf"))
    ids = masked.topk(K, -1)[1]
    w = scores.gather(1, ids)
    return ids, w / w.sum(-1, keepdim=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])  # FP32 router GEMM, BF16 router GEMM
@pytest.mark.parametrize("rows", [T, 96])  # one block, and a three-block prefill chunk
def test_route_matches_grouped_topk(rows, dtype):
    from dlmrt.models.llada2 import kernels as lk

    logits = (torch.randn(rows, E, device=dev) * 2).to(dtype)
    bias = torch.randn(E, device=dev) * 0.1
    ids = torch.zeros(rows, K, device=dev, dtype=torch.int32)
    slots, counts = torch.zeros_like(ids), torch.zeros(E, device=dev, dtype=torch.int32)
    w = torch.zeros(rows, K, device=dev)
    lk.route(logits, bias, ids, w, slots, counts, K, G, TOPG)
    ref_ids, ref_w = biased_grouped_topk(logits, bias)
    order = ref_ids.argsort(-1)
    assert torch.equal(ids.long(), ref_ids.gather(1, order))
    assert torch.allclose(w, ref_w.gather(1, order), rtol=1e-5, atol=1e-6)
    assert torch.equal(counts.long(), torch.bincount(ref_ids.flatten(), minlength=E))
    pairs = ids.long() * rows + slots.long()
    assert pairs.unique().numel() == rows * K and bool((slots < counts[ids.long()]).all())


@pytest.mark.parametrize("gemm", ["triton", "deepgemm"])
@pytest.mark.parametrize("bf16_router", [False, True])
def test_moe_matches_expert_loop(bf16_router, gemm, monkeypatch):
    from dlmrt.models.llada2 import model as llada
    from dlmrt.models.llada2 import LladaModel

    monkeypatch.setattr(llada, "MOE_GEMM", gemm)

    model = object.__new__(LladaModel)
    model.config = llada.LladaConfig(hidden_size=H, intermediate_size=4 * F, moe_intermediate_size=F, num_experts=E,
                                     num_experts_per_tok=K, n_group=G, topk_group=TOPG, routed_scaling_factor=2.5,
                                     first_k_dense_replace=1, num_hidden_layers=2, num_attention_heads=16,
                                     num_key_value_heads=4, head_dim=128, rotary_dim=64, rope_theta=6e5,
                                     vocab_size=1024)
    model.dtype, model.device = torch.bfloat16, torch.device(dev)
    model.local_experts, model.expert_lo = E, 0
    model.shared_out, model.side = torch.zeros(64, H, device=dev, dtype=torch.bfloat16), None
    m = 128  # as LladaModel: DeepGEMM's SM100 masked grouped GEMM returns garbage below 128 rows per expert
    model.moe_in, model.moe_out = (torch.zeros(E, m, H, device=dev, dtype=torch.bfloat16) for _ in range(2))
    model.moe_mid = torch.zeros(E, m, 2 * F, device=dev, dtype=torch.bfloat16)
    model.moe_act = torch.zeros(E, m, F, device=dev, dtype=torch.bfloat16)
    model.expert_ids, model.expert_slots = (torch.zeros(m, K, device=dev, dtype=torch.int32) for _ in range(2))
    model.expert_weights = torch.zeros(m, K, device=dev)
    model.expert_counts = torch.zeros(E, device=dev, dtype=torch.int32)
    bf = {"device": dev, "dtype": torch.bfloat16}
    shared = {"gate_up": llada.Linear(torch.randn(2 * F, H, **bf) * 0.02, False),
              "down": llada.Linear(torch.randn(H, F, **bf) * 0.02, False)}
    layer = {"router": torch.randn(E, H, device=dev) * 0.02, "bias": torch.randn(E, device=dev) * 0.01,
             "w13": torch.randn(E, 2 * F, H, **bf) * 0.02, "w2": torch.randn(E, H, F, **bf) * 0.02, "shared": shared}
    if bf16_router:  # a BF16-exact router runs as one BF16 GEMM
        layer["router"] = llada.Linear(layer["router"].to(torch.bfloat16), False)
    x = torch.randn(T, H, **bf)
    out = model.moe(x, layer, torch.empty_like(x))
    router = layer["router"].weight.float() if bf16_router else layer["router"]
    ids, w = biased_grouped_topk(x.float() @ router.t(), layer["bias"])

    def swiglu_mlp(v, w13, w2):
        gu = (v @ w13.t()).float()
        return (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]).to(torch.bfloat16) @ w2.t()

    routed = torch.zeros(T, H, device=dev)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            routed[t] += (w[t, k] * swiglu_mlp(x[t:t + 1], layer["w13"][e], layer["w2"][e])[0].float()).to(
                torch.bfloat16).float()
    gu = (x @ shared["gate_up"].weight.t()).float()
    s = (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]).to(torch.bfloat16) @ shared["down"].weight.t()
    ref = ((routed * 2.5).to(torch.bfloat16).float() + s.float()).to(torch.bfloat16)
    assert rel(out, ref) < 5e-3, rel(out, ref)


def test_qk_rope_matches_hf():
    from dlmrt.models.llada2 import kernels as lk
    from dlmrt.models.llada2 import LladaConfig
    from dlmrt.models.llada2.model import _rope_table

    hq, hk, d, r, start = 16, 4, 128, 64, 1000
    c = LladaConfig(hidden_size=H, intermediate_size=1, moe_intermediate_size=1, num_experts=1, num_experts_per_tok=1,
                    n_group=1, topk_group=1, routed_scaling_factor=1, first_k_dense_replace=1, num_hidden_layers=1,
                    num_attention_heads=hq, num_key_value_heads=hk, head_dim=d, rotary_dim=r, rope_theta=6e5,
                    vocab_size=1)
    qkv = torch.randn(T, (hq + 2 * hk) * d, device=dev, dtype=torch.bfloat16)
    qw, kw = (torch.rand(d, device=dev) + 0.5).to(torch.bfloat16), (torch.rand(d, device=dev) + 0.5).to(torch.bfloat16)
    rope = _rope_table(c, 2048, dev)
    q = torch.empty(1, T, hq, d, device=dev, dtype=torch.bfloat16)
    kc, vc = (torch.zeros(1, 2048, hk, d, device=dev, dtype=torch.bfloat16) for _ in range(2))
    lk.qk_rope(qkv, qw, kw, rope, torch.tensor([start], device=dev, dtype=torch.int32), q, kc, vc, hq, hk, d, r, 1e-6)

    def ref(x, w):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        y = (y * w.float()).to(torch.bfloat16).float()
        cos, sin = rope[start:start + T, :r // 2], rope[start:start + T, r // 2:]
        cos, sin = torch.cat((cos, cos), -1)[:, None], torch.cat((sin, sin), -1)[:, None]
        rot, rest = y[..., :r], y[..., r:]
        rotated = rot * cos + torch.cat((-rot[..., r // 2:], rot[..., :r // 2]), -1) * sin
        return torch.cat((rotated, rest), -1).to(torch.bfloat16)

    view = qkv.view(T, hq + 2 * hk, d)
    assert rel(q[0], ref(view[:, :hq], qw)) < 4e-3
    assert rel(kc[0, start:start + T], ref(view[:, hq:hq + hk], kw)) < 4e-3
    assert torch.equal(vc[0, start:start + T], view[:, hq + hk:])


def reference_block(confidences, proposals, count, thresh, edit, refines):
    """SGLang SDARPrefixKV fused_next on one block (prob_threshold, mask noise, greedy), the T2T refinement of a
    finished block counted as in the deferred path; returns tokens, denoise and refine rounds per block."""
    b = proposals.shape[1]
    tokens = torch.full((b,), MASK, dtype=torch.long)
    prompt = torch.arange(b) < count
    tokens[prompt] = 7
    accepted = prompt.clone()
    denoise = refine = 0
    r = 0
    while not bool(accepted.all()):
        conf, prop = confidences[r], proposals[r]
        before = accepted.clone()
        eligible = ~accepted & ~prompt
        high = (conf > thresh) & eligible
        if int(high.sum()) >= 1:
            accept = high
        else:
            accept = torch.zeros(b, dtype=torch.bool)
            accept[torch.where(eligible, conf, torch.tensor(-float("inf"))).argmax()] = True
        tokens = torch.where(accept, prop, tokens)
        accepted |= accept
        e = before & ~prompt & (conf > edit) & (prop != tokens)
        tokens = torch.where(e, prop, tokens)
        denoise += 1
        r += 1
    terminal = bool(((tokens == EOS) & ~prompt).any())
    edits = 0
    while True:
        conf, prop = confidences[r], proposals[r]
        r += 1
        refine += 1
        e = ~prompt & (conf > edit) & (prop != tokens)
        if not bool(e.any()):
            break
        tokens = torch.where(e, prop, tokens)
        edits += 1
        if edits >= refines:
            if not terminal:
                r += 1  # the confirm forward: K/V only
            break
    return tokens, denoise, refine, r, terminal


@pytest.mark.parametrize("count", [0, 5])
def test_decide_matches_reference_loop(count):
    from dlmrt.models.llada2 import kernels as lk
    from dlmrt.models.llada2 import LladaDecode

    b, rounds, vocab = 32, 64, 8
    config = LladaDecode(prob_threshold=0.7, edit_threshold=0.5, refine_steps=4)
    for trial in range(20):
        logits = torch.randn(rounds, b, vocab, device=dev) * (1 + trial % 4)
        if trial % 3 == 0:  # nearly settled blocks reach the refine cap or settle quickly
            logits[:, :, 3] += 4
        if trial % 5 == 0:
            logits[:, :, 2] += 3  # some EOS proposals (vocabulary index 2 maps to EOS below)
        probs = logits.softmax(-1)
        conf, prop = probs.max(-1)
        prop = torch.where(prop == 2, torch.tensor(EOS, device=dev), prop)
        tokens, denoise, refine, total, terminal = reference_block(conf.cpu(), prop.cpu(), count, 0.7, 0.5, 4)
        params = torch.tensor([32, count, 1], device=dev, dtype=torch.int32)
        prompt_ids = torch.full((b,), 7, device=dev, dtype=torch.long)
        tok, carry_tok = (torch.zeros(b, device=dev, dtype=torch.long) for _ in range(2))
        prompt, accepted, carry_prompt = (torch.zeros(b, device=dev, dtype=torch.bool) for _ in range(3))
        start, end, pair_start, pair_ends, ctr = (torch.zeros(n, device=dev, dtype=torch.int32) for n in (1, 1, 1, 2, 16))
        lk.begin(params, prompt_ids, tok, prompt, accepted, start, end, pair_start, pair_ends, ctr, MASK)
        stats = torch.zeros(b, 3, device=dev)
        slot = {name: getattr(lk, name).value for name in ("STEP", "REMAINING", "DENOISE", "REFINE", "TERMINAL")}
        for r in range(rounds):
            if not int(ctr[slot["REMAINING"]]):
                break
            stats[:, 1] = conf[r]
            lk.decide(stats, prop[r].contiguous(), tok, prompt, accepted, carry_tok, carry_prompt, ctr, config, EOS,
                      False)
        assert torch.equal(tok.cpu(), tokens), trial
        assert int(ctr[slot["DENOISE"]]) == denoise and int(ctr[slot["REFINE"]]) == refine, (trial, ctr.tolist())
        assert int(ctr[slot["STEP"]]) == total and bool(ctr[slot["TERMINAL"]]) == terminal, (trial, ctr.tolist())


def test_confidence_schedule():
    from dlmrt.models.llada2 import LladaDecode

    assert LladaDecode().confidence.confidence_temperature_step_decrement == 0.0
    slow = LladaDecode(confidence_temperature_min=0.1, confidence_temperature_step_decrement=0.01).confidence
    assert slow.confidence_temperature_min == pytest.approx(0.69)  # the reference caps the round index at 31
    fast = LladaDecode(confidence_temperature_min=0.6, confidence_temperature_step_decrement=0.08).confidence
    assert fast.confidence_temperature_min == 0.6
