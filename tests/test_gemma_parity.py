"""DiffusionGemma forward parity against HF transformers (DLMRT_DG_CHECKPOINT, two GPUs, transformers>=5.11).

The reference (cuda:1, DynamicCache: sliding layers keep the last window - 1 keys) runs with SDPA and, as the
control, with eager attention. DiffusionGemma amplifies BF16 noise through its top-8 routing (the two reference
runs themselves diverge to ~20% relative residual error by the last layers), so dlmrt must
stay within the reference's own spread: every error against HF-SDPA at most 1.25x the eager-vs-SDPA error.
Checked: prefix K/V after a 1500-token causal prefill (crosses the 1024 window), canvas logits without and with
self-conditioning, and the K/V written by the commit (causal re-encode of the canvas).
"""

import os

import pytest
import torch

CKPT = os.environ.get("DLMRT_DG_CHECKPOINT")
pytestmark = pytest.mark.skipif(not (CKPT and torch.cuda.device_count() >= 2), reason="needs DLMRT_DG_CHECKPOINT and 2 GPUs")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DiffusionGemmaForBlockDiffusion"):
    pytest.skip("transformers without DiffusionGemma", allow_module_level=True)

PROMPT = 1500


def rel(a, b):
    return ((a.float().cpu() - b.float().cpu()).norm() / b.float().cpu().norm()).item()


def agree(new, ref, margin=1.0):
    """Top-1 agreement on rows whose reference top-2 logit gap exceeds `margin` (near-ties are noise)."""
    new, ref = new.float().cpu(), ref.float().cpu()
    top = ref.topk(2, -1).values
    rows = (top[:, 0] - top[:, 1]) > margin
    return (new.argmax(-1) == ref.argmax(-1))[rows].float().mean().item()


def hf_run(ref, prompt, noisy, canvas, sc_logits=None):
    """Prefix K/V, canvas logits (plain and self-conditioned), commit K/V of the reference."""
    from transformers import DynamicCache

    dev = "cuda:1"
    cache = DynamicCache(config=ref.config.get_text_config(decoder=True))
    ref.model.encoder(input_ids=prompt[None].to(dev), past_key_values=cache,
                      position_ids=torch.arange(PROMPT, device=dev)[None])
    prefix = [(layer.keys[0].transpose(0, 1).cpu(), layer.values[0].transpose(0, 1).cpu()) for layer in cache.layers]
    pos = torch.arange(PROMPT, PROMPT + 256, device=dev)[None]
    plain = ref(decoder_input_ids=noisy[None].to(dev), past_key_values=cache, decoder_position_ids=pos).logits[0].cpu()
    sc_logits = (plain / 0.8).to(torch.bfloat16) if sc_logits is None else sc_logits
    sc = ref(decoder_input_ids=noisy[None].to(dev), past_key_values=cache, self_conditioning_logits=sc_logits[None].to(dev),
             decoder_position_ids=pos).logits[0].cpu()
    ref.model.encoder(input_ids=canvas[None].to(dev), past_key_values=cache, position_ids=pos)
    commit = [(layer.keys[0].transpose(0, 1)[-256:].cpu(), layer.values[0].transpose(0, 1)[-256:].cpu())
              for layer in cache.layers]
    return {"prefix": prefix, "plain": plain, "sc": sc, "commit": commit, "sc_logits": sc_logits}


def new_forward(model, ids, start, *, causal, delta=None):
    from dlmrt.models.diffusion_gemma import kernels as gk

    c = model.config
    dev = model.device
    ids = ids.to(dev)
    h = torch.empty(ids.numel(), c.hidden_size, device=dev, dtype=torch.bfloat16)
    gk.embed(ids, model.embedding, model.embed_scale, h, c.rms_norm_eps, decoder=not causal, delta=delta)
    s = torch.tensor([start], device=dev, dtype=torch.int32)
    leftpad = torch.tensor([max(0, start - (c.sliding_window - 1))], device=dev, dtype=torch.int32)
    return model.forward(h, s, s + ids.numel(), leftpad, causal=causal, splits=1)


def new_run(model, prompt, noisy, canvas, sc_logits):
    from dlmrt.models.diffusion_gemma import softcap

    cap = model.config.final_logit_softcapping
    for offset in range(0, PROMPT, 256):  # the engine's chunked prefill; rows past the prompt write scratch K/V
        chunk = torch.zeros(256, dtype=torch.long)
        chunk[:min(256, PROMPT - offset)] = prompt[offset:offset + 256]
        new_forward(model, chunk, offset, causal=True)
    prefix = [(k[0, :PROMPT].cpu(), v[0, :PROMPT].cpu()) for k, v in zip(model.k, model.v)]
    plain = softcap(model.logits(new_forward(model, noisy, PROMPT, causal=False)), cap).cpu()
    probs = sc_logits.float().softmax(-1).to(torch.bfloat16).to(model.device)
    sc = softcap(model.logits(new_forward(model, noisy, PROMPT, causal=False, delta=model.self_condition(probs))),
                 cap).cpu()
    new_forward(model, canvas, PROMPT, causal=True)
    commit = [(k[0, PROMPT:PROMPT + 256].cpu(), v[0, PROMPT:PROMPT + 256].cpu()) for k, v in zip(model.k, model.v)]
    return {"prefix": prefix, "plain": plain, "sc": sc, "commit": commit}


def kv_error(run, ref):
    """Max over layers and K/V; sliding layers compare the keys the reference cache retains."""
    worst = 0.0
    for (k, v), (rk, rv) in zip(run, ref):
        n = rk.shape[0]
        worst = max(worst, rel(k[-n:], rk), rel(v[-n:], rv))
    return worst


@pytest.fixture(scope="module")
def runs():
    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

    from dlmrt.models.diffusion_gemma import GemmaModel

    text = open(__import__("inspect").getsourcefile(torch.nn.functional)).read()[20000:80000]
    ids = torch.tensor(AutoTokenizer.from_pretrained(CKPT).encode(text, add_special_tokens=False)[:PROMPT + 256])
    prompt, canvas = ids[:PROMPT], ids[PROMPT:PROMPT + 256].clone()
    torch.manual_seed(0)  # a partially re-noised canvas, as in denoising
    noisy = canvas.clone()
    noisy[::3] = torch.randint(0, 262144, noisy[::3].shape)
    ref = DiffusionGemmaForBlockDiffusion.from_pretrained(CKPT, dtype=torch.bfloat16, device_map={"": "cuda:1"},
                                                          attn_implementation="sdpa").eval()
    with torch.inference_mode():
        sdpa = hf_run(ref, prompt, noisy, canvas)
        ref.set_attn_implementation("eager")
        eager = hf_run(ref, prompt, noisy, canvas, sdpa["sc_logits"])
        del ref
        torch.cuda.empty_cache()
        new = new_run(GemmaModel.load(CKPT, max_seq_len=4096, device="cuda:0"), prompt, noisy, canvas,
                      sdpa["sc_logits"])
    return sdpa, eager, new


def test_forward_within_reference_spread(runs):
    sdpa, eager, new = runs
    report = {}
    for name in ("prefix", "commit"):
        report[name] = (kv_error(new[name], sdpa[name]), kv_error(eager[name], sdpa[name]))
    for name in ("plain", "sc"):
        report[name] = (rel(new[name], sdpa[name]), rel(eager[name], sdpa[name]))
        report[name + "_top1"] = (agree(new[name], sdpa[name]), agree(eager[name], sdpa[name]))
    layer0 = kv_error(new["prefix"][:1], sdpa["prefix"][:1])
    print({k: tuple(round(x, 4) for x in v) for k, v in report.items()}, "layer-0 K/V", round(layer0, 5))
    assert layer0 < 3e-3  # layer 0 isolates the operator path
    for name in ("prefix", "commit", "plain", "sc"):
        ours, control = report[name]
        assert ours <= 1.25 * control + 1e-3, (name, ours, control)
    for name in ("plain_top1", "sc_top1"):
        ours, control = report[name]
        assert ours >= control - 0.01, (name, ours, control)
