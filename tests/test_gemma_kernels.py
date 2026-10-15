"""DiffusionGemma kernels against direct torch transcriptions of the HF reference ops (random weights, real sizes)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, T, E, K, F = 2816, 256, 128, 8, 704
dev = "cuda"


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def ulp_close(a, b):
    """BF16 rounding noise only: fp32 reduction order before rsqrt differs from torch, flipping some roundings."""
    a, b = a.float(), b.float()
    err = rel(a, b)
    worst = ((a - b).abs() - b.abs() * 2 ** -6).max().item()
    print(f"rel {err:.2e}, mismatched {(a != b).float().mean().item():.3f}, worst beyond 2 ulp {worst:.2e}")
    return err < 2e-3 and worst <= 1e-6


def rms(x, w=None, eps=1e-6):
    y = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (y * w.float() if w is not None else y).to(torch.bfloat16)


def gelu_glu(gate, up):
    return torch.nn.functional.gelu(gate, approximate="tanh") * up


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_route_matches_topk():
    from dlmrt.ops.triton import moe

    logits = torch.randn(T, E, device=dev).to(torch.bfloat16)
    pes = (torch.rand(E, device=dev) + 0.5).to(torch.bfloat16)
    ids = torch.zeros(T, K, device=dev, dtype=torch.int32)
    slots, counts = torch.zeros_like(ids), torch.zeros(E, device=dev, dtype=torch.int32)
    w = torch.zeros(T, K, device=dev)
    moe.route(logits, pes, ids, w, slots, counts, K)
    p = logits.float().softmax(-1)
    top_w, top_i = p.topk(K, -1)
    top_w = top_w / top_w.sum(-1, keepdim=True) * pes[top_i].float()
    order = top_i.argsort(-1)
    assert torch.equal(ids.long(), top_i.gather(1, order))
    assert torch.allclose(w, top_w.gather(1, order), rtol=1e-5, atol=1e-6)
    assert torch.equal(counts.long(), torch.bincount(top_i.flatten(), minlength=E))
    pairs = ids.long() * T + slots.long()
    assert pairs.unique().numel() == T * K and bool((slots < counts[ids.long()]).all())


def config():
    """The expert shapes of GemmaConfig (only what the expert backends read)."""
    return type("C", (), {"top_k_experts": K, "num_experts": E, "hidden_size": H, "moe_intermediate_size": F})()


def test_moe_matches_expert_loop():
    from dlmrt.models.diffusion_gemma.experts import DeepGemmExperts

    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    router_in = torch.randn(T, H, device=dev).to(torch.bfloat16)
    w13 = (torch.randn(E, 2 * F, H, device=dev) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, H, F, device=dev) * 0.02).to(torch.bfloat16)
    proj = (torch.randn(E, H, device=dev) * 0.02).to(torch.bfloat16)
    pes = torch.ones(E, device=dev, dtype=torch.bfloat16)
    experts = DeepGemmExperts(config(), 0, E, T, dev)
    out = experts(x, router_in @ proj.t(), {"expert_scale": pes, "w13": w13, "w2": w2}, torch.empty_like(x))
    p = (router_in @ proj.t()).float().softmax(-1)
    top_w, top_i = p.topk(K, -1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    ref = torch.zeros(T, H, device=dev)
    for e in range(E):
        rows, k = torch.where(top_i == e)
        if rows.numel():
            gate, up = (x[rows] @ w13[e].t()).chunk(2, -1)
            y = gelu_glu(gate, up) @ w2[e].t()
            ref[rows] += y.float() * top_w[rows, k, None]
    assert rel(out, ref) < 5e-3


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("start", [0, 700])
@pytest.mark.parametrize("hq", [16, 4, 2])  # one GPU (one key range), and the per-rank heads at TP=4 / TP=8 (split)
def test_attention_head_dim_512(causal, start, hq):
    from dlmrt.ops.triton.attention import attention

    hk, d, cap = 2 if hq == 16 else 1, 512, 2048
    k = torch.randn(1, cap, hk, d, device=dev).to(torch.bfloat16)
    v = torch.randn(1, cap, hk, d, device=dev).to(torch.bfloat16)
    q = (torch.randn(1, T, hq, d, device=dev) * 0.05).to(torch.bfloat16)
    s = torch.tensor([start], device=dev, dtype=torch.int32)
    out = attention(q, k, v, s, s + T, causal)
    end = start + T
    kk = k[0, :end].float().repeat_interleave(hq // hk, 1)
    vv = v[0, :end].float().repeat_interleave(hq // hk, 1)
    scores = torch.einsum("qhd,khd->hqk", q[0].float(), kk)
    if causal:
        keep = torch.arange(end, device=dev)[None, :] <= torch.arange(start, end, device=dev)[:, None]
        scores = scores.masked_fill(~keep[None], float("-inf"))
    ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv)
    assert rel(out[0], ref) < 5e-3


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("start", [0, 1500])
@pytest.mark.parametrize("fp8", [False, True])
def test_attention_sliding(causal, start, fp8):
    """The sliding layers' attention without FA3: causal rows see their last `window` keys, canvas rows [left, end)."""
    from dlmrt.ops.triton.attention import attention

    hq, hk, d, cap, window = 16, 8, 256, 2048, 1024
    k = torch.randn(1, cap, hk, d, device=dev).to(torch.bfloat16)
    v = torch.randn(1, cap, hk, d, device=dev).to(torch.bfloat16)
    if fp8:
        k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    q = (torch.randn(1, T, hq, d, device=dev) * 0.05).to(torch.bfloat16)
    s = torch.tensor([start], device=dev, dtype=torch.int32)
    end, left = start + T, max(0, start - (window - 1))
    out = attention(q, k, v, s, s + T, causal, window=window if causal else 0,
                       left=None if causal else torch.tensor([left], device=dev, dtype=torch.int32))
    kk = k[0, :end].float().repeat_interleave(hq // hk, 1)
    vv = v[0, :end].float().repeat_interleave(hq // hk, 1)
    scores = torch.einsum("qhd,khd->hqk", q[0].float(), kk)
    kpos, qpos = torch.arange(end, device=dev)[None, :], torch.arange(start, end, device=dev)[:, None]
    keep = (kpos <= qpos) & (kpos > qpos - window) if causal else (kpos >= left).expand(T, end)
    ref = torch.einsum("hqk,khd->qhd", scores.masked_fill(~keep[None], float("-inf")).softmax(-1), vv)
    assert rel(out[0], ref) < 5e-3


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("sliding", [True, False])
def test_qkv_norm_rope(sliding, split):
    from dlmrt.models.diffusion_gemma import kernels as gk
    from dlmrt.models.diffusion_gemma.model import _rope_table

    hq, hk, d = (16, 8, 256) if sliding else (16, 2, 512)
    if sliding:
        inv = 1.0 / 1e4 ** (torch.arange(0, d, 2, device=dev).float() / d)
    else:
        inv = torch.zeros(d // 2, device=dev)
        inv[:64] = 1.0 / 1e6 ** (torch.arange(0, 128, 2, device=dev).float() / d)
    cos, sin = _rope_table(inv, 1024)
    width = hq + (2 if sliding else 1) * hk
    qkv = torch.randn(T, width * d, device=dev).to(torch.bfloat16)
    qw, kw = (torch.rand(d, device=dev) + 0.5).to(torch.bfloat16), (torch.rand(d, device=dev) + 0.5).to(torch.bfloat16)
    q = torch.empty(1, T, hq, d, device=dev, dtype=torch.bfloat16)
    kc, vc = torch.zeros(1, 1024, hk, d, device=dev, dtype=torch.bfloat16), torch.zeros(1, 1024, hk, d, device=dev,
                                                                                         dtype=torch.bfloat16)
    start = 300
    gk.qkv_rope(qkv, qw, kw, cos, sin, torch.tensor([start], device=dev, dtype=torch.int32), q, kc, vc, hq, hk, d,
                sliding, 1e-6, split=split)
    c = torch.cat([cos[start:start + T]] * 2, -1)[:, None, :]
    s = torch.cat([sin[start:start + T]] * 2, -1)[:, None, :]

    def rope(x):
        rot = torch.cat([-x[..., d // 2:], x[..., :d // 2]], -1)
        return x * c + rot * s

    raw = qkv.view(T, width, d)
    rq, rk = raw[:, :hq], raw[:, hq:hq + hk]
    rv = raw[:, hq + hk:] if sliding else rk
    assert ulp_close(q[0], rope(rms(rq, qw)))
    assert ulp_close(kc[0, start:start + T], rope(rms(rk, kw)))
    assert ulp_close(vc[0, start:start + T], rms(rv))


def test_sandwich_norms():
    from dlmrt.models.diffusion_gemma import kernels as gk

    def w():
        return (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)

    h = torch.randn(T, H, device=dev).to(torch.bfloat16)
    attn, y1, y2 = (torch.randn(T, H, device=dev).to(torch.bfloat16) for _ in range(3))
    wa, w1, w2, rs, wm1, wm2, wp, wn = (w() for _ in range(8))
    x1, x2, r, x = (torch.empty_like(h) for _ in range(4))
    hh = h.clone()
    gk.post_attn(hh, attn, wa, w1, w2, rs, x1, x2, r, 1e-6)
    ref_h = h + rms(attn, wa)
    assert ulp_close(hh, ref_h)
    assert ulp_close(x1, rms(hh, w1)) and ulp_close(x2, rms(hh, w2))
    assert ulp_close(r, rms(hh) * rs * H ** -0.5)
    scalar = 0.6875
    h2 = hh.clone()
    gk.post_ffn(h2, y1, y2, wm1, wm2, wp, wn, x, scalar, 1e-6)
    ref = (hh + rms(rms(y1, wm1) + rms(y2, wm2), wp)) * torch.tensor(scalar, dtype=torch.bfloat16, device=dev)
    assert ulp_close(h2, ref)
    assert ulp_close(x, rms(h2, wn))


@pytest.mark.parametrize("rand4", [False, True])
def test_logit_stats_and_decide(rand4):
    from dlmrt.models.diffusion_gemma import kernels as gk
    from dlmrt.models.diffusion_gemma import GemmaDecode

    v, cap = 262144, 30.0
    d = GemmaDecode()
    logits = (torch.randn(T, v, device=dev) * 3).to(torch.bfloat16)
    logits[:T // 2, 7] = 60.0  # half the rows nearly deterministic
    i32 = {"device": dev, "dtype": torch.int32}
    step, seed, block = torch.tensor([5], **i32), torch.tensor([1234], **i32), torch.tensor([2], **i32)
    stats = torch.zeros(T, 2, device=dev)
    argmax, sample = torch.zeros(T, device=dev, dtype=torch.int64), torch.zeros(T, device=dev, dtype=torch.int64)
    gk.logit_stats(logits, step, seed, block, d, stats, argmax, sample, cap, rand4=rand4)
    t = d.t_min + (d.t_max - d.t_min) * (d.max_denoising_steps - 5) / d.max_denoising_steps
    z = cap * torch.tanh(logits.float() / cap) / t
    ref_h = torch.distributions.Categorical(logits=z).entropy()
    assert torch.allclose(stats[:, 1], ref_h, rtol=1e-3, atol=1e-4)
    assert torch.equal(argmax, z.argmax(-1))
    assert torch.allclose(stats[:, 0], z.to(torch.bfloat16).float().logsumexp(-1), rtol=1e-5, atol=1e-4)
    assert bool((sample[:T // 2] == 7).all()) and sample[T // 2:].unique().numel() > T // 2 - 8

    tokens, canvas = torch.zeros(T, device=dev, dtype=torch.int64), torch.full((T,), -1, device=dev, dtype=torch.int64)
    remaining = torch.tensor([1], **i32)
    accepted = torch.zeros(T, device=dev, dtype=torch.int8)
    gk.decide(stats, argmax, sample, tokens, canvas, accepted, step, remaining, seed, block, d, v)
    order = ref_h.argsort()
    hs = ref_h[order]
    take = torch.zeros(T, dtype=torch.bool, device=dev)
    take[order] = (hs.cumsum(0) - hs) <= d.entropy_bound
    assert torch.equal(tokens[take], sample[take]) and torch.equal(canvas, argmax)
    assert int(step) == 6 and int(remaining) == 1  # mean entropy is high: not confident
    confident = stats.clone()
    confident[:, 1] = 1e-4
    gk.decide(confident, argmax, sample, tokens, canvas, accepted, step, remaining, seed, block, d, v)
    assert int(remaining) == 0  # argmax unchanged and confident


def test_adaptive_temperature():
    """Three rounds against a transcription of the paper's vLLM adaptive-scalar controller."""
    from dlmrt.models.diffusion_gemma import kernels as gk
    from dlmrt.models.diffusion_gemma import ROLES

    d, v, cap = ROLES["orl"], 262144, 30.0
    f32 = {"device": dev, "dtype": torch.float32}
    temp, h_ref, t_prev = torch.zeros(1, **f32), torch.zeros(1, **f32), torch.zeros(1, **f32)
    accepted = torch.zeros(T, device=dev, dtype=torch.int8)
    ref_h, ref_t = None, None
    for step in range(3):
        logits = (torch.randn(T, v, device=dev) * (2 + 2 * step)).to(torch.bfloat16)  # sharper each round
        if step:
            accepted = (torch.rand(T, device=dev) < 0.3).to(torch.int8)
        gk.adaptive_temperature(logits, torch.tensor([step], device=dev, dtype=torch.int32), accepted, temp, h_ref,
                                t_prev, d, cap)
        z = cap * torch.tanh(logits.float() / cap)
        entropy = torch.distributions.Categorical(logits=z).entropy() / torch.log(torch.tensor(float(v)))
        scope = accepted == 0
        h = entropy[scope].mean()
        if step == 0:
            ref_h, ref_t = h.clamp_min(1e-8), d.t_max
        else:
            proposed = d.t_min + (d.t_max - d.t_min) * (h / ref_h).clamp(0, 1) ** d.adaptive_power
            ref_t = min(ref_t, float(proposed))
        assert abs(float(temp) - ref_t) < 1e-4, (step, float(temp), ref_t)
    assert d.t_min <= float(temp) < d.t_max
