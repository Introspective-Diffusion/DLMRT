"""LLaDA2 forward parity against the checkpoint's HF implementation (DLMRT_LLADA_CHECKPOINT, two GPUs).

The reference (cuda:1) runs the HF decoder layers on [prompt blocks | one block] with the block-causal mask of
LLaDA2's `generate`, with SDPA and, as the control, with eager attention. Top-8 routing amplifies BF16 noise, so
dlmrt must stay within the reference's own spread: every error against HF-SDPA at most 1.25x the eager-vs-SDPA
error. Checked: the prefix K/V of a block-causal prefill (several blocks per forward in dlmrt), the block's K/V and
logits, and, for a checkpoint with the self-conditioning adapter, the block logits with an SC signal.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

CKPT = os.environ.get("DLMRT_LLADA_CHECKPOINT")
pytestmark = pytest.mark.skipif(not (CKPT and torch.cuda.device_count() >= 2),
                                reason="needs DLMRT_LLADA_CHECKPOINT and 2 GPUs")
transformers = pytest.importorskip("transformers")

PROMPT, B = 320, 32  # ten prompt blocks: two prefill forwards of up to 256 rows


def rel(a, b):
    return ((a.float().cpu() - b.float().cpu()).norm() / b.float().cpu().norm()).item()


def agree(new, ref, margin=1.0):
    """Top-1 agreement on rows whose reference top-2 logit gap exceeds `margin` (near-ties are noise)."""
    new, ref = new.float().cpu(), ref.float().cpu()
    top = ref.topk(2, -1).values
    rows = (top[:, 0] - top[:, 1]) > margin
    return (new.argmax(-1) == ref.argmax(-1))[rows].float().mean().item()


def hf_checkpoint():
    """The checkpoint with its modeling file patched for transformers 5 (`_tied_weights_keys` must be a dict)."""
    root = Path(tempfile.mkdtemp(prefix="llada-hf-"))
    for f in Path(CKPT).iterdir():
        if f.name == "modeling_llada2_moe.py":
            text = f.read_text().replace('_tied_weights_keys = ["lm_head.weight"]', "_tied_weights_keys = {}")
            (root / f.name).write_text(text)
        else:
            (root / f.name).symlink_to(f.resolve())
    return root


def sc_delta(weights, embedding, probs, eps):
    """The reference adapter (SGLang LLaDA2SelfConditioning, all scales one, no post-norm), BF16 as it rounds."""
    signal = probs.matmul(embedding)
    x = signal.float() * torch.rsqrt(signal.float().pow(2).mean(-1, keepdim=True) + eps)
    x = (x * weights["pre_norm.weight"].float()).to(torch.bfloat16)
    gated = torch.nn.functional.silu(x @ weights["gate_proj.weight"].t()) * (x @ weights["up_proj.weight"].t())
    return gated @ weights["down_proj.weight"].t()


def hf_run(model, ids, embeds=None):
    """Prefix and block K/V per layer and the block logits, block-causal over [prompt | block]."""
    from transformers import DynamicCache

    inner = model.model
    dev = inner.word_embeddings.weight.device
    n = ids.numel()
    block = torch.arange(n, device=dev) // B
    allowed = block[None, :] <= block[:, None]
    if model.config._attn_implementation == "eager":
        mask = torch.zeros(n, n, device=dev, dtype=torch.bfloat16).masked_fill(~allowed, torch.finfo(torch.bfloat16).min)
    else:
        mask = allowed
    h = inner.word_embeddings(ids[None].to(dev)) if embeds is None else embeds[None].to(dev)
    pos = torch.arange(n, device=dev)[None]
    rope = inner.rotary_emb(h, pos)
    cache = DynamicCache()
    for layer in inner.layers:
        h = layer(h, attention_mask=mask[None, None], position_ids=pos, past_key_value=cache, use_cache=True,
                  position_embeddings=rope)[0]
    logits = model.lm_head(inner.norm(h))[0, -B:].cpu()
    kv = [(lay.keys[0].transpose(0, 1).cpu(), lay.values[0].transpose(0, 1).cpu()) for lay in cache.layers]
    return {"prefix": [(k[:PROMPT], v[:PROMPT]) for k, v in kv], "block": [(k[PROMPT:], v[PROMPT:]) for k, v in kv],
            "logits": logits}


def new_run(model, ids, delta=None):
    from dlmrt.ops.triton import core as kernels
    from dlmrt.models.llada2.engine import LladaEngine

    engine = LladaEngine(model)
    dev, c = model.device, model.config
    engine._prefill(ids.to(dev), PROMPT)
    prefix = [(model.k[i, 0, :PROMPT].cpu(), model.v[i, 0, :PROMPT].cpu()) for i in range(c.num_hidden_layers)]
    h = torch.empty(B, c.hidden_size, device=dev, dtype=torch.bfloat16)
    kernels.embed_norm(ids[PROMPT:].to(dev), model.embedding, h, c.rms_norm_eps, delta=delta, norm=False)
    start = torch.tensor([PROMPT], device=dev, dtype=torch.int32)
    logits = model.logits(model.forward(h, start, start + B)).cpu()
    block = [(model.k[i, 0, PROMPT:PROMPT + B].cpu(), model.v[i, 0, PROMPT:PROMPT + B].cpu())
             for i in range(c.num_hidden_layers)]
    return {"prefix": prefix, "block": block, "logits": logits}


def kv_error(run, ref):
    return max(max(rel(k, rk), rel(v, rv)) for (k, v), (rk, rv) in zip(run, ref))


@pytest.fixture(scope="module")
def runs():
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dlmrt.models.llada2 import LladaModel

    text = open(__import__("inspect").getsourcefile(torch.nn.functional)).read()[20000:80000]
    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False)[:PROMPT + B])
    torch.manual_seed(0)  # a partially masked block, as in denoising
    ids[PROMPT:][torch.rand(B) < 0.5] = 156895
    root = hf_checkpoint()
    out = {}

    def reference(impl):  # the remote code does not support switching the attention implementation in place
        return AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, dtype=torch.bfloat16,
                                                    device_map={"": "cuda:1"}, attn_implementation=impl).eval()

    try:
        with torch.inference_mode():
            eager = reference("eager")
            out["eager"] = hf_run(eager, ids)
            ref = reference("sdpa")
            out["sdpa"] = hf_run(ref, ids)
            model = LladaModel.load(CKPT, max_seq_len=1024, device="cuda:0")
            out["new"] = new_run(model, ids)
            if model.sc is not None:
                # SC signal: the softmax of the plain block logits on the masked rows, as in the second round.
                probs = out["sdpa"]["logits"].float().softmax(-1).to(torch.bfloat16)
                probs[ids[PROMPT:] != 156895] = 0
                files = sorted(Path(CKPT).glob("*.safetensors"))
                sc = {}
                for f in files:
                    with safe_open(f, framework="pt", device="cpu") as source:
                        for key in source.keys():  # noqa: SIM118
                            if "self_conditioning." in key:
                                sc[key.split("self_conditioning.")[1]] = source.get_tensor(key).to("cuda:1",
                                                                                                torch.bfloat16)
                emb = ref.model.word_embeddings.weight
                delta_ref = sc_delta(sc, emb, probs.to("cuda:1"), model.config.rms_norm_eps)
                embeds = ref.model.word_embeddings(ids.to("cuda:1"))
                embeds[PROMPT:] = embeds[PROMPT:] + delta_ref
                out["sdpa_sc"] = hf_run(ref, ids, embeds)
                out["eager_sc"] = hf_run(eager, ids, embeds)
                delta = model.self_condition(probs.to(model.device))
                out["delta"] = (delta.cpu(), delta_ref.cpu())
                out["new_sc"] = new_run(model, ids, delta)
    finally:
        shutil.rmtree(root)
    return out


def test_forward_within_reference_spread(runs):
    sdpa, eager, new = runs["sdpa"], runs["eager"], runs["new"]
    report = {name: (kv_error(new[name], sdpa[name]), kv_error(eager[name], sdpa[name])) for name in ("prefix", "block")}
    report["logits"] = (rel(new["logits"], sdpa["logits"]), rel(eager["logits"], sdpa["logits"]))
    report["top1"] = (agree(new["logits"], sdpa["logits"]), agree(eager["logits"], sdpa["logits"]))
    layer0 = kv_error(new["prefix"][:1], sdpa["prefix"][:1])
    print({k: tuple(round(x, 4) for x in v) for k, v in report.items()}, "layer-0 K/V", round(layer0, 5))
    for name in ("prefix", "block"):
        print(name, "per-layer K error (ours, control):",
              [(round(rel(k, rk), 3), round(rel(ek, rk), 3))
               for (k, _), (ek, _), (rk, _) in zip(new[name], eager[name], sdpa[name])])
    assert layer0 < 3e-3  # layer 0 isolates the operator path
    for name in ("prefix", "block", "logits"):
        ours, control = report[name]
        assert ours <= 1.25 * control + 1e-3, (name, ours, control)
    assert report["top1"][0] >= report["top1"][1] - 0.01, report["top1"]


def test_self_conditioning_within_reference_spread(runs):
    if "new_sc" not in runs:
        pytest.skip("checkpoint without the self-conditioning adapter")
    delta, delta_ref = runs["delta"]
    sdpa, eager, new = runs["sdpa_sc"], runs["eager_sc"], runs["new_sc"]
    ours, control = rel(new["logits"], sdpa["logits"]), rel(eager["logits"], sdpa["logits"])
    print("delta", round(rel(delta, delta_ref), 5), "sc logits", round(ours, 4), round(control, 4),
          "sc vs plain logits", round(rel(sdpa["logits"], runs["sdpa"]["logits"]), 4))
    assert rel(delta, delta_ref) < 1e-2
    assert ours <= 1.25 * control + 1e-3, (ours, control)
