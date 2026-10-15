"""QwenDLM (`sdar`) engine on a tiny random model: determinism, the token budget and trace mode (one GPU, no
checkpoint)."""

import pytest
import torch

from dlmrt.models.sdar import DecodeConfig, ModelConfig
from dlmrt.models.sdar import Engine
from dlmrt.models.sdar import Model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def tiny_engine(fp8=False, entropy_bound=0.4, trace=False):
    torch.manual_seed(7)
    c = ModelConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=128, vocab_size=1024, eos_token_id=1001,
                    self_conditioning_prob=1.0, self_conditioning_post_norm=True,
                    self_conditioning_post_norm_scale=0.02, self_conditioning_prenorm_scale=0.125,
                    self_conditioning_reparam_out_scale=0.125)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.05

    def ones(n):
        return torch.ones(n, device="cuda", dtype=torch.bfloat16)

    w = {"model.embed_tokens.weight": rand(1024, 256), "model.norm.weight": ones(256),
         "self_conditioning.pre_norm.weight": ones(256), "self_conditioning.gate_proj.weight": rand(512, 256),
         "self_conditioning.up_proj.weight": rand(512, 256), "self_conditioning.down_proj.weight": rand(256, 512)}
    for i in range(2):
        p = f"model.layers.{i}."
        w.update({p + "input_layernorm.weight": ones(256), p + "post_attention_layernorm.weight": ones(256),
                  p + "self_attn.q_norm.weight": ones(128), p + "self_attn.k_norm.weight": ones(128),
                  p + "self_attn.q_proj.weight": rand(512, 256), p + "self_attn.k_proj.weight": rand(256, 256),
                  p + "self_attn.v_proj.weight": rand(256, 256), p + "self_attn.o_proj.weight": rand(256, 512),
                  p + "mlp.gate_proj.weight": rand(512, 256), p + "mlp.up_proj.weight": rand(512, 256),
                  p + "mlp.down_proj.weight": rand(256, 512)})
    model = Model(c, w, 4096, fp8=fp8)
    return Engine(model, DecodeConfig(entropy_bound=entropy_bound, random_token_high=1000), trace=trace)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("prompt_len", [10, 300])
def test_generation_is_deterministic_and_respects_budget(fp8, prompt_len):
    engine = tiny_engine(fp8)
    prompt = [(i * 37) % 1000 for i in range(prompt_len)]
    first = engine.generate(prompt, 600, stop_on_eos=False)
    second = engine.generate(prompt, 600, stop_on_eos=False)
    assert first.token_ids == second.token_ids and first.denoise_steps == second.denoise_steps
    assert len(first.token_ids) == 600
    # A short prompt shares the first block (246 + 256 + 98); a long one is prefilled (256 + 256 + 88).
    assert len(first.denoise_steps) == 3
    assert all(1 <= s <= 256 for s in first.denoise_steps)
    assert all(0 <= t < 1024 for t in first.token_ids)


@pytest.mark.parametrize("prompt_len", [10, 300])
def test_trace_records_every_round_and_changes_nothing(prompt_len):
    engine = tiny_engine(trace=True)
    prompt = [(i * 37) % 1000 for i in range(prompt_len)]
    plain = engine.generate(prompt, 600, stop_on_eos=False)
    blocks = []
    traced = engine.generate(prompt, 600, stop_on_eos=False, trace=True,
                             on_block=lambda tokens, steps, elapsed, frames: blocks.append((tokens, steps, frames)))
    assert traced.token_ids == plain.token_ids and traced.denoise_steps == plain.denoise_steps
    for tokens, steps, frames in blocks:
        assert len(frames) == steps and all(len(f) == len(tokens) for f in frames)
        # Round 0 starts from pure noise (random_token_high = 1000); the emitted block differs from it.
        assert all(0 <= t < 1000 for t in frames[0]) and frames[0] != tokens


def test_short_budget_truncates_the_block():
    engine = tiny_engine()
    result = engine.generate(list(range(20)), 17, stop_on_eos=False)
    assert len(result.token_ids) == 17 and len(result.denoise_steps) == 1
