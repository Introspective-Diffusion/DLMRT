"""LLaDA2 engine: fused rounds (the reference's fused_next) against refining every block in place
(DLMRT_LLADA_CHECKPOINT, one GPU)."""

import os

import pytest
import torch

CKPT = os.environ.get("DLMRT_LLADA_CHECKPOINT")
pytestmark = pytest.mark.skipif(not (CKPT and torch.cuda.is_available()), reason="needs DLMRT_LLADA_CHECKPOINT")

PROMPTS = [
    "Write a Python function that returns the n-th prime number, with a few tests.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips "
    "did Natalia sell altogether in April and May? Let's think step by step and output the final answer after "
    "\"####\".",
    "Explain in two paragraphs why the sky is blue. " * 6,  # a prompt longer than one block: prefill + tail
]


@pytest.fixture(scope="module")
def setup():
    from transformers import AutoTokenizer

    from dlmrt.models.llada2 import LladaModel

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = LladaModel.load(CKPT, max_seq_len=4096)
    ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True)) for p in PROMPTS]
    return model, ids


DECODERS = {
    "default": {},
    # The paper's ORL Speed mode: the step_linear confidence temperature; fused rounds must give each half of a pair
    # forward its own round index (this block's first round at temperature one).
    "speed_anneal": {"prob_threshold": 0.6, "edit_threshold": 0.0, "confidence_temperature_min": 0.6,
                     "confidence_temperature_step_decrement": 0.08},
    # Annealing with a positive edit threshold: the carried block's edits depend on its own (continued) round index.
    "edit_anneal": {"prob_threshold": 0.85, "edit_threshold": 0.5, "confidence_temperature_min": 0.6,
                    "confidence_temperature_step_decrement": 0.08},
}


@pytest.mark.parametrize("name", sorted(DECODERS))
def test_fused_matches_in_place(setup, name):
    from dlmrt.models.llada2 import LladaDecode
    from dlmrt.models.llada2.engine import LladaEngine

    model, prompts = setup
    decoder = LladaDecode(**DECODERS[name])
    plain, fused = LladaEngine(model, decoder, fused=False), LladaEngine(model, decoder, fused=True, trace=True)
    same = 0
    for ids in prompts:
        a, b = plain.generate(ids, 512), fused.generate(ids, 512)
        prefix = next((i for i, (x, y) in enumerate(zip(a.token_ids, b.token_ids)) if x != y),
                      min(len(a.token_ids), len(b.token_ids)))
        print(len(ids), "prompt tokens:", len(a.token_ids), len(b.token_ids), "tokens, common prefix", prefix,
              "forwards", sum(a.forwards), sum(b.forwards), "denoise", sum(a.denoise_steps), sum(b.denoise_steps))
        same += a.token_ids == b.token_ids
        assert ids is not prompts[0] or len(b.token_ids) / sum(b.denoise_steps) > 5  # a healthy model on code
        n = min(len(a.denoise_steps), len(b.denoise_steps))
        # The pair round settles the carried block and doubles as the next block's first denoise round.
        assert sum(b.forwards[:n]) < sum(a.forwards[:n])
        frames = []
        traced = fused.generate(ids, 512, trace=True, on_block=lambda t, s, e, f: frames.append((t, f)))
        assert traced.token_ids == b.token_ids
        assert sum(len(t) for t, _ in frames) == len(b.token_ids)
        assert [x for t, _ in frames for x in t] == b.token_ids
    assert same >= len(prompts) - 1  # a 64-row forward may differ from a 32-row one by BF16 rounding
