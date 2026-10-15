"""The SGLang-numerics LLaDA2 kernels (models/llada2/sglang_kernels.py, ops/triton/sglang_fused_moe.py) against torch
transcriptions of the ops they reproduce (random weights, real sizes). End-to-end agreement with the paper runtime's
records: docs/performance.md (ORL Speed HumanEval, 164 of 164 answers)."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, E, K, FF, QH, KH, D, ROT, V = 2048, 256, 8, 512, 16, 4, 128, 64, 157184
dev = "cuda"


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def routed_ids(tokens):
    return torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(tokens)]).int()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                    reason="cuBLAS's split-K order for these shapes, as on H100")
@pytest.mark.parametrize("m", [32, 64])
def test_router_logits_match_cublas(m):
    from dlmrt.models.llada2 import sglang_kernels as sk

    x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
    w = torch.randn(E, H, device=dev) * 0.02
    out = torch.empty(m, E, device=dev)
    count = torch.zeros(E // 16, device=dev, dtype=torch.int32)
    sk.router_logits(x, w, out, torch.empty(H // 64 * 64 * E, device=dev), count)
    assert torch.equal(out, F.linear(x.float(), w).to(torch.bfloat16).float())
    assert not count.any()


@pytest.mark.parametrize("tokens", [32, 64])  # adjacent pairs, then sgl_kernel's in-order sum
def test_moe_sum_orders(tokens):
    from dlmrt.models.llada2 import sglang_kernels as sk

    x = torch.randn(tokens, K, H, device=dev, dtype=torch.bfloat16)
    shared = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16)
    out = torch.empty(tokens, H, device=dev, dtype=torch.bfloat16)
    sk.moe_sum(x, shared, out, 2.5)
    v = x.float().unbind(1)
    if tokens > 32:
        s = v[0]
        for t in v[1:]:
            s = s + t
    else:
        s = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))
    assert torch.equal(out, ((s * 2.5).to(torch.bfloat16).float() + shared.float()).to(torch.bfloat16))


@pytest.mark.parametrize("tokens", [32, 64, 128])
def test_moe_align_groups_as_sgl_kernel(tokens):
    import triton

    from dlmrt.models.llada2 import sglang_kernels as sk
    from dlmrt.ops.triton import sglang_fused_moe as fm

    ids = routed_ids(tokens)
    cursor = torch.empty(triton.next_power_of_2(E + 1), device=dev, dtype=torch.int32)
    a = sk.moe_align(ids, 16, E, cursor)
    b = fm.moe_align_block_size(ids, 16, E)
    total = int(b[2])
    assert int(a[2]) == total
    assert torch.equal(a[1][:total // 16], b[1][:total // 16])
    assert torch.equal(a[0][:total].view(-1, 16).sort(-1).values, b[0][:total].view(-1, 16).sort(-1).values)


def test_fused_moe_matches_expert_loop_and_tiles_agree():
    from dlmrt.models.llada2.sglang_numerics import MOE_TILES
    from dlmrt.ops.triton import sglang_fused_moe as fm

    m = 32
    x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
    w1 = (torch.randn(E, 2 * FF, H, device=dev) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, H, FF, device=dev) * 0.02).to(torch.bfloat16)
    ids = routed_ids(m)
    weights = torch.rand(m, K, device=dev)
    outs = []
    for config in (fm.default_config(m, E), MOE_TILES[m]):
        align = fm.moe_align_block_size(ids, config["BLOCK_SIZE_M"], E)
        up = torch.empty(m * K, 2 * FF, device=dev, dtype=torch.bfloat16)
        fm.invoke_fused_moe_kernel(x, w1, up, weights, ids, *align, False, K, config)
        act = (F.silu(up[:, :FF].float()) * up[:, FF:].float()).to(torch.bfloat16)
        down = torch.empty(m, K, H, device=dev, dtype=torch.bfloat16)
        fm.invoke_fused_moe_kernel(act, w2, down, weights, ids, *align, True, 1, config)
        outs.append((up, down))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    e = ids.long().flatten()
    ref_up = torch.einsum("th,tnh->tn", x.float().repeat_interleave(K, 0), w1[e].float())
    assert rel(outs[0][0], ref_up) < 1e-2
    act = (F.silu(outs[0][0][:, :FF].float()) * outs[0][0][:, FF:].float()).to(torch.bfloat16)
    ref_down = torch.einsum("tf,thf->th", act.float(), w2[e].float()) * weights.flatten()[:, None]
    assert rel(outs[0][1].view(-1, H), ref_down) < 1e-2


def test_step_stats_match_softmax():
    import triton

    from dlmrt.models.llada2 import sglang_kernels as sk

    n = 32
    logits = (torch.randn(n, V, device=dev) * 3).to(torch.bfloat16)
    chunks = triton.cdiv(V, sk.STAT_CHUNK) * 64
    scratch = (torch.empty(chunks, device=dev), torch.empty(chunks, device=dev, dtype=torch.int32),
               torch.zeros(64, device=dev, dtype=torch.int32), torch.empty(64, device=dev))
    stats, proposal = torch.zeros(n, 3, device=dev), torch.zeros(n, device=dev, dtype=torch.int64)
    sk.step_stats(logits, 0.8, stats, proposal, *scratch)
    assert torch.equal(proposal, logits.float().argmax(-1))
    assert torch.allclose(stats[:, 1], torch.softmax(logits.float() / 0.8, -1).amax(-1), rtol=1e-5)
    from_tensor = torch.zeros_like(stats)
    sk.step_stats(logits, torch.tensor(0.8, device=dev), from_tensor, proposal, *scratch)
    assert torch.equal(from_tensor[:, 1], stats[:, 1])
    assert not scratch[2].any()


@pytest.mark.parametrize("rot", [ROT, D])  # LLaDA2's partial rotary, SDAR's full
def test_qk_norm_rope_place(rot):
    from dlmrt.ops.triton.sglang_rope import qk_norm_rope_place

    n, eps = 32, 1e-6
    qkv = torch.randn(n, (QH + 2 * KH) * D, device=dev, dtype=torch.bfloat16)
    weights = (1 + 0.1 * torch.randn(2, D, device=dev)).to(torch.bfloat16)
    positions = torch.arange(100, 100 + n, device=dev)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rot, 2, device=dev).float() / rot))
    freqs = torch.arange(4096, device=dev).float()[:, None] * inv
    cos_sin = torch.cat((freqs.cos(), freqs.sin()), -1)
    q = torch.empty(n, QH, D, device=dev, dtype=torch.bfloat16)
    kc, vc = (torch.zeros(4096, 1, KH, D, device=dev, dtype=torch.bfloat16) for _ in range(2))
    src = qkv.clone()
    qk_norm_rope_place(qkv, q, kc, vc, positions, cos_sin, weights, eps, QH, KH, D, rot)

    def norm_rope(x, w):
        x = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w.float()
        a, b = x[..., :rot // 2], x[..., rot // 2:rot]
        cos, sin = cos_sin[positions, None, :rot // 2], cos_sin[positions, None, rot // 2:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin, x[..., rot:]), -1)

    heads = src.view(n, QH + 2 * KH, D)
    assert rel(q, norm_rope(heads[:, :QH], weights[0])) < 1e-2
    assert rel(kc[positions, 0], norm_rope(heads[:, QH:QH + KH], weights[1])) < 1e-2
    assert torch.equal(kc[positions, 0], qkv.view(n, QH + 2 * KH, D)[:, QH:QH + KH])
    assert torch.equal(vc[positions, 0], heads[:, QH + KH:])
