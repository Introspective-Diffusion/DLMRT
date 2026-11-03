"""BatchedGemmaEngine on a checkpoint (DLMRT_DG_CHECKPOINT, one GPU, ~40 GB). One request at a time it emits
GemmaEngine's tokens and denoise steps: 256-row canvases under the reference stop rule, 32-row canvases under
freeze_argmax, and the load-adaptive 256 / 128 / 32 engine (256 for one request). A small batched run with commit lanes
finishes every request with tokens in the vocabulary and no NaN logits, and repeats token for token.
"""

import os

import pytest
import torch

CKPT = os.environ.get("DLMRT_DG_CHECKPOINT")
pytestmark = pytest.mark.skipif(not (CKPT and torch.cuda.is_available()), reason="needs DLMRT_DG_CHECKPOINT and a GPU")

QUESTIONS = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many "
             "clips did Natalia sell altogether in April and May?",
             "Write a Python function that checks whether a number is prime.",
             "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"]
MAX_NEW, SEQ = 768, 2048


@pytest.fixture(scope="module")
def prompts():
    from transformers import AutoTokenizer

    from dlmrt.entrypoints.bench.diffusion_gemma import chat_ids

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    return [chat_ids(tokenizer, [{"role": "user", "content": q}]) for q in QUESTIONS]


def batched_options():
    """The small batched run: 256-row canvases for up to two requests, then 32-row ones with commit lanes."""
    from dlmrt.models.diffusion_gemma.batched import LIMIT, LoadPolicy

    return dict(blocks=(256, 32), stops={32: "freeze_argmax"}, policy=LoadPolicy(((2, 256), (LIMIT, 32))),
                fused=True, lane_min=2)


def model(canvas_length, max_batch, **options):
    from dlmrt.models.diffusion_gemma import GemmaModel
    from dlmrt.models.diffusion_gemma.batched import sizes

    _, _, buckets = sizes(options.pop("blocks", (canvas_length,)), max_batch, **options)
    return GemmaModel.load(CKPT, SEQ, canvas_length=canvas_length, slots=max_batch + 1, max_rows=buckets[-1],
                           row_buckets=buckets)


@pytest.fixture(scope="module")
def m256():
    """Also built for the adaptive engine at batch 1 and the batched run (their KV slots, rows and tactics)."""
    return model(256, 4, **batched_options())


@pytest.fixture(scope="module")
def m32():
    return model(32, 1)


def identical(solo, batched, prompts):
    for ids in prompts:
        a = solo.generate(ids, MAX_NEW)
        b = batched.run([ids], MAX_NEW)[0]
        assert a.token_ids == b.token_ids and a.denoise_steps == b.denoise_steps, (len(a.token_ids), len(b.token_ids))
        assert b.blocks == [solo.block] * len(b.denoise_steps)


def test_identity_256(m256, prompts):
    from dlmrt.models.diffusion_gemma import ROLES
    from dlmrt.models.diffusion_gemma.batched import BatchedGemmaEngine
    from dlmrt.models.diffusion_gemma.engine import GemmaEngine

    d = ROLES["orl"]
    identical(GemmaEngine(m256, d), BatchedGemmaEngine(m256, d, 1), prompts)


def test_identity_adaptive(m256, prompts):
    from dlmrt.models.diffusion_gemma import ROLES
    from dlmrt.models.diffusion_gemma.batched import BatchedGemmaEngine, LoadPolicy
    from dlmrt.models.diffusion_gemma.engine import GemmaEngine

    d = ROLES["orl"]
    engine = BatchedGemmaEngine(m256, d, 1, blocks=(256, 128, 32), stops={32: "freeze_argmax"}, policy=LoadPolicy())
    identical(GemmaEngine(m256, d), engine, prompts)


def test_identity_32(m32, prompts):
    import dataclasses

    from dlmrt.models.diffusion_gemma import ROLES
    from dlmrt.models.diffusion_gemma.batched import BatchedGemmaEngine
    from dlmrt.models.diffusion_gemma.engine import GemmaEngine

    d = dataclasses.replace(ROLES["orl"], stop_rule="freeze_argmax")
    identical(GemmaEngine(m32, d), BatchedGemmaEngine(m32, d, 1), prompts)


def test_batched_run(m256, prompts):
    from dlmrt.models.diffusion_gemma import ROLES
    from dlmrt.models.diffusion_gemma.batched import BatchedGemmaEngine

    engine = BatchedGemmaEngine(m256, ROLES["orl"], 4, **batched_options())
    many = (prompts * 3)[:8]
    first = engine.run(many, MAX_NEW, concurrency=4)
    vocab = m256.config.vocab_size
    assert all(r is not None and 0 < len(r.token_ids) <= MAX_NEW for r in first)
    assert all(0 <= t < vocab for r in first for t in r.token_ids)
    assert engine.nan_positions() == 0 and engine.lane_commits() > 0
    assert 32 in {b for r in first for b in r.blocks}
    again = engine.run(many, MAX_NEW, concurrency=4)
    assert [(r.token_ids, r.denoise_steps, r.blocks) for r in again] == \
        [(r.token_ids, r.denoise_steps, r.blocks) for r in first]
