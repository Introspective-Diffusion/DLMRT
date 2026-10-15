"""Block decoder kernels of QwenDLM (`sdar`; the row statistics and the self-conditioning softmax also serve LLaDA2):
begin a block, per-row statistics over the (sharded) vocabulary, accept / edit / re-noise, and the self-conditioning
softmax.
"""

import torch
import triton
import triton.language as tl

from .common import LAUNCH, _release, _wait


@triton.jit
def _sc_probs(X, STATS, PROMPT, STEP, Y, V: tl.constexpr, BLOCK: tl.constexpr):
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    mask = cols < V
    off = tl.load(PROMPT + row) | (tl.load(STEP) == 0)
    x = tl.load(X + row * V + cols, mask & ~off, 0.0).to(tl.float32)
    lse = tl.load(STATS + row * 3)
    tl.store(Y + row * V + cols, tl.where(off, 0.0, tl.exp(x - lse)).to(tl.bfloat16), mask)
    _release()


def sc_probs(logits, stats, prompt, step, out):
    """Raw (temperature-one) softmax of the previous round; prompt rows and round 0 give zeros."""
    rows, v = logits.shape
    _sc_probs[(rows, triton.cdiv(v, 4096))](logits, stats, prompt, step, out, v, 4096, num_warps=8, **LAUNCH)
    return out


@triton.jit
def _begin_block(P, PROMPT_IDS, TOKENS, PROMPT, ACCEPTED, START, END, NOISE_START, STEP, REMAINING,
                 LAST, B: tl.constexpr, N: tl.constexpr, HIGH: tl.constexpr, SEED: tl.constexpr):
    _wait()
    i = tl.arange(0, N)
    offset, count, noise_start, last = tl.load(P), tl.load(P + 1), tl.load(P + 2), tl.load(P + 3)
    noise = ((i + noise_start + 1).to(tl.int64) * 1103515245 + 12345 + SEED) % HIGH
    is_prompt = i < count
    token = tl.where(is_prompt, tl.load(PROMPT_IDS + i, (i < B) & is_prompt, 0), noise)
    tl.store(TOKENS + i, token, i < B)
    tl.store(PROMPT + i, is_prompt, i < B)
    tl.store(ACCEPTED + i, is_prompt, i < B)
    tl.store(START, offset)
    tl.store(END, offset + B)
    tl.store(NOISE_START, noise_start)
    tl.store(STEP, 0)
    tl.store(REMAINING, B - count)
    tl.store(LAST, last.to(tl.int64))
    _release()


def begin_block(params, prompt_ids, tokens, prompt, accepted, start, end, noise_start, step, remaining,
                last, config):
    b = tokens.numel()
    _begin_block[(1,)](params, prompt_ids, tokens, prompt, accepted, start, end, noise_start, step,
                       remaining, last, b, triton.next_power_of_2(b), config.random_token_high,
                       config.randtok_seed, **LAUNCH)


@triton.jit
def _row_parts(X, STEP, PART, VOCAB: tl.constexpr, OFFSET: tl.constexpr, PARTS: tl.constexpr, TMAX: tl.constexpr,
               TMIN: tl.constexpr, DEC: tl.constexpr, N: tl.constexpr):
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * N + tl.arange(0, N)
    x = tl.load(X + row * VOCAB + col, col < VOCAB, -float("inf")).to(tl.float32)
    t = tl.maximum(TMIN, TMAX - tl.load(STEP) * DEC)
    m = tl.max(x, 0)
    a = x - m
    z = tl.sum(tl.exp(a), 0)
    b = a / t
    pt = tl.exp(b)
    base = PART + (row * PARTS + part) * 5
    tl.store(base, m)
    tl.store(base + 1, z)
    tl.store(base + 2, tl.sum(pt, 0))
    tl.store(base + 3, tl.sum(pt * tl.where(col < VOCAB, b, 0.0), 0))
    tl.store(base + 4, tl.min(tl.where(x == m, col + OFFSET, 2147483647), 0).to(tl.float32))
    _release()


@triton.jit
def _row_finish(PART, STEP, STATS, IDS, ROWS: tl.constexpr, PARTS: tl.constexpr, RANKS: tl.constexpr,
                STRIDE: tl.constexpr, TMAX: tl.constexpr, TMIN: tl.constexpr, DEC: tl.constexpr, N: tl.constexpr):
    """Combine the per-part statistics of every vocabulary shard (`RANKS` blocks of `STRIDE` floats)."""
    _wait()
    row = tl.program_id(0)
    j = tl.arange(0, N)
    live = j < RANKS * PARTS
    base = PART + (j // PARTS) * STRIDE + (row * PARTS + j % PARTS) * 5
    m = tl.load(base, live, -float("inf"))
    z = tl.load(base + 1, live, 0)
    zt = tl.load(base + 2, live, 0)
    st = tl.load(base + 3, live, 0)
    idx = tl.load(base + 4, live, 2147483647).to(tl.int32)
    maximum = tl.max(m, 0)
    delta = m - maximum
    t = tl.maximum(TMIN, TMAX - tl.load(STEP) * DEC)
    factor = tl.exp(delta / t)
    scaled_total = tl.sum(zt * factor, 0)
    centered = tl.sum(factor * (st + zt * tl.where(live, delta / t, 0.0)), 0)
    tl.store(STATS + row * 3, maximum + tl.log(tl.sum(z * tl.exp(delta), 0)))
    tl.store(STATS + row * 3 + 1, 1.0 / scaled_total)
    tl.store(STATS + row * 3 + 2, tl.maximum(0.0, tl.log(scaled_total) - centered / scaled_total))
    tl.store(IDS + row, tl.min(tl.where(m == maximum, idx, 2147483647), 0))
    _release()


def row_stats(logits, step, config, stats, ids, *, offset=0, group=None):
    """Per row: log-normalizer, top-token probability and entropy at the step's temperature, argmax.

    With a `group`, `logits` holds this rank's vocabulary columns starting at `offset`; the per-part
    statistics of all ranks are gathered so every rank reaches the same result.
    """
    rows, vocab = logits.shape
    parts = triton.cdiv(vocab, 2048)
    stride = triton.cdiv(rows * parts * 5, 4) * 4  # 16-byte multiple for the multicast gather
    scratch = torch.empty(stride, device=logits.device, dtype=torch.float32)
    t = (config.confidence_temperature, config.confidence_temperature_min,
         config.confidence_temperature_step_decrement)
    _row_parts[(rows, parts)](logits, step, scratch, vocab, offset, parts, *t, 2048, **LAUNCH)
    ranks = 1
    if group is not None:
        scratch, ranks = group.all_gather(scratch, f"row_stats{rows}"), group.world
    _row_finish[(rows,)](scratch, step, stats, ids, rows, parts, ranks, stride, *t,
                         triton.next_power_of_2(parts * ranks), **LAUNCH)


@triton.jit
def _decide(STATS, PROPOSAL, TOKENS, ACCEPTED, PROMPT, PREFIX, NOISE_START, STEP, REMAINING,
            B: tl.constexpr, N: tl.constexpr, SHIFT: tl.constexpr, ENTROPY: tl.constexpr,
            BOUND: tl.constexpr, THRESH: tl.constexpr, EDIT: tl.constexpr, CHANGE: tl.constexpr,
            HIGH: tl.constexpr, SEED: tl.constexpr):
    _wait()
    i = tl.arange(0, N)
    prefix = tl.load(PREFIX)
    step = tl.load(STEP)
    if SHIFT:
        row = tl.where(i > 0, i - 1, tl.where(prefix > 0, B, 0))
    else:
        row = i
    accepted = tl.load(ACCEPTED + i, i < B, 1)
    prompt = tl.load(PROMPT + i, i < B, 1)
    confidence = tl.load(STATS + row * 3 + 1, i < B, 0)
    entropy = tl.load(STATS + row * 3 + 2, i < B, float("inf"))
    eligible = (i < B) & ~accepted & ~prompt
    score = tl.where(eligible, entropy if ENTROPY else (1.0 - confidence), float("inf"))
    # Nonnegative float bits sort numerically; the index breaks ties deterministically.
    key = (score.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) | i.to(tl.uint64)
    order = (tl.sort(key, descending=False) & 0xFFFFFFFF).to(tl.int32)
    valid = tl.gather(eligible, order, 0)
    if ENTROPY:
        h = tl.where(valid, tl.gather(entropy, order, 0), 0.0)
        take = valid & ((tl.cumsum(h, 0) - h) <= BOUND)
    else:
        take = valid & ((tl.gather(confidence, order, 0) > THRESH) | (i == 0))
    old = tl.load(ACCEPTED + order, order < B, 1)
    is_prompt = tl.load(PROMPT + order, order < B, 1)
    current = tl.load(TOKENS + order, order < B, 0)
    if SHIFT:
        source = tl.where(order > 0, order - 1, tl.where(prefix > 0, B, 0))
    else:
        source = order
    candidate = tl.load(PROPOSAL + source, order < B, 0)
    edit = CHANGE & old & ~is_prompt & (tl.load(STATS + source * 3 + 1, order < B, 0) > EDIT)
    final = old | take
    value = tl.where(take | edit, candidate, current)
    absolute = (tl.load(NOISE_START) + order).to(tl.int64)
    noise = ((absolute + 1) * 1103515245 + (step + 2).to(tl.int64) * 12345 + SEED) % HIGH
    tl.store(TOKENS + order, tl.where(final, value, noise), order < B)
    tl.store(ACCEPTED + order, final, order < B)
    tl.store(REMAINING, tl.sum(((order < B) & ~final).to(tl.int32), 0))
    tl.store(STEP, step + 1)
    _release()


def decide(stats, proposal, tokens, accepted, prompt, prefix, noise_start, step, remaining, config):
    """Accept by entropy budget or threshold, edit accepted tokens, refresh noise, count remaining."""
    b = tokens.numel()
    _decide[(1,)](stats, proposal, tokens, accepted, prompt, prefix, noise_start, step, remaining, b,
                  triton.next_power_of_2(b), config.logits_shift, config.accept_rule == "entropy_budget",
                  config.entropy_bound, config.prob_threshold, config.edit_threshold,
                  config.allow_accepted_change, config.random_token_high, config.randtok_seed,
                  num_warps=4, **LAUNCH)
