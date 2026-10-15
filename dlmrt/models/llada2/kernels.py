"""Triton kernels of the LLaDA2 path (see model.py and engine.py).

Rounding follows the paper runtime (the SGLang fork's LLaDA2 model): Q/K norm output rounded to BF16 before RoPE,
router logits rounded to BF16 before the sigmoid, each expert's weighted output rounded to BF16 before the top-k
sum, and the routed sum scaled before the shared expert is added. Every kernel waits for its predecessor before
touching memory and releases its successor after its last store (PDL; see dlmrt.ops.triton.common).
"""

import torch
import triton
import triton.language as tl

from ...ops.triton.common import LAUNCH, _bf, _release, _row_tile, _store_bf16, _wait

# Slots of the decoder's int32 counter array (see decide and decide_pair). CARRY_T: the carried block's own round
# index, for its confidence temperature in the fused rounds.
(STEP, REMAINING, STATE, DENOISE, REFINE, TERMINAL, CARRY, CARRY_LOOP, CARRY_STEP, CARRY_REFINE, CARRY_NEW, RESOLVED,
 FORWARDS, CARRY_T) = (tl.constexpr(i) for i in range(14))


@triton.jit
def _norm_rope_heads(src, W, heads, cos, sin, OUT, eps, D: tl.constexpr, R: tl.constexpr, FP8: tl.constexpr):
    """Per-head RMSNorm over all D dims (rounded to BF16), then NeoX RoPE on the first R dims only. FP8: the BF16
    result is rounded through FP8 E4M3 (the FP8 KV cache of an NVFP4 export)."""
    half = tl.arange(0, R // 2)[None, :]
    rest = R + tl.arange(0, D - R)[None, :]
    a = tl.load(src + heads + half).to(tl.float32)
    b = tl.load(src + heads + half + R // 2).to(tl.float32)
    c = tl.load(src + heads + rest).to(tl.float32)
    r = tl.rsqrt((tl.sum(a * a, 1) + tl.sum(b * b, 1) + tl.sum(c * c, 1))[:, None] / D + eps)
    na = _bf(a * r * tl.load(W + half).to(tl.float32))
    nb = _bf(b * r * tl.load(W + half + R // 2).to(tl.float32))
    _store_bf16(OUT + heads + half, tl.fma(na, cos, -(nb * sin)), FP8)
    _store_bf16(OUT + heads + half + R // 2, tl.fma(nb, cos, na * sin), FP8)
    _store_bf16(OUT + heads + rest, c * r * tl.load(W + rest).to(tl.float32), FP8)


@triton.jit
def _qk_rope(QKV, QW, KW, ROPE, START, Q, K, V, eps, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
             R: tl.constexpr, KV_FP8: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    pos = tl.load(START) + row
    half = tl.arange(0, R // 2)
    cos = tl.load(ROPE + pos * R + half)[None, :]
    sin = tl.load(ROPE + pos * R + R // 2 + half)[None, :]
    src = QKV + row * (HQ + 2 * HK) * D
    hq = tl.arange(0, HQ)[:, None] * D
    hk = tl.arange(0, HK)[:, None] * D
    _norm_rope_heads(src, QW, hq, cos, sin, Q + row * HQ * D, eps, D, R, False)
    _norm_rope_heads(src + HQ * D, KW, hk, cos, sin, K + pos.to(tl.int64) * HK * D, eps, D, R, KV_FP8)
    hv = hk + tl.arange(0, D)[None, :]
    _store_bf16(V + pos.to(tl.int64) * HK * D + hv, tl.load(src + (HQ + HK) * D + hv), KV_FP8)
    _release()


def qk_rope(qkv, q_weight, k_weight, rope, start, q, k, v, heads, kv_heads, head_dim, rotary_dim, eps, kv_fp8=False):
    """Q/K RMSNorm + partial RoPE; Q to `q`, K/V into the arena at [start, start + rows) (FP8-rounded with kv_fp8).

    `rope`: FP32 [positions, rotary_dim], cos in the first half of each row and sin in the second."""
    _qk_rope[(qkv.shape[0],)](qkv, q_weight, k_weight, rope, start, q, k, v, eps, heads, kv_heads, head_dim,
                              rotary_dim, kv_fp8, num_warps=4, **LAUNCH)
    return q


@triton.jit
def _route_rows(L, BIAS, IDS, W, POS, scale, E: tl.constexpr, G: tl.constexpr, TOPG: tl.constexpr, K: tl.constexpr,
                SCALED: tl.constexpr, SLOTS: tl.constexpr):
    """One token: sigmoid scores, biased for selection only; groups ranked by the sum of their two best biased
    scores; top-k experts within the TOPG best groups; weights = unbiased scores renormalized (SCALED: times
    `scale`, rounded to BF16). Experts are listed in ascending order; POS[token, e] is the list position of expert
    e, or -1 (SLOTS). Ties go to the lower index."""
    _wait()
    row = tl.program_id(0)
    cols = tl.arange(0, E)
    s = tl.sigmoid(_bf(tl.load(L + row * E + cols).to(tl.float32)))  # _bf takes FP32 registers
    b = s + tl.load(BIAS + cols)
    EG: tl.constexpr = E // G
    bg = tl.reshape(b, (G, EG))
    j = tl.arange(0, EG)[None, :]
    m1 = tl.max(bg, 1)
    first = tl.min(tl.where(bg == m1[:, None], j, EG), 1)
    score = m1 + tl.max(tl.where(j == first[:, None], float("-inf"), bg), 1)
    g = tl.arange(0, G)
    chosen = g < 0
    for _ in tl.static_range(TOPG):
        avail = tl.where(chosen, float("-inf"), score)
        chosen = chosen | (g == tl.min(tl.where(avail == tl.max(avail, 0), g, G), 0))
    keep = tl.reshape(tl.broadcast_to(chosen[:, None], (G, EG)), (E,))
    masked = tl.where(keep, b, float("-inf"))
    taken = cols < 0
    for _ in tl.static_range(K):
        avail = tl.where(taken, float("-inf"), masked)
        taken = taken | (cols == tl.min(tl.where(avail == tl.max(avail, 0), cols, E), 0))
    top = tl.where(taken, s, 0.0)
    w = top / tl.sum(top, 0)
    if SCALED:
        w = (w * scale).to(tl.bfloat16).to(tl.float32)
    t = taken.to(tl.int32)
    rank = tl.cumsum(t, 0) - t
    tl.store(IDS + row * K + rank, cols, taken)
    tl.store(W + row * K + rank, w, taken)
    if SLOTS:
        tl.store(POS + row * E + cols, tl.where(taken, rank, -1))
    _release()


@triton.jit
def _route_slots(POS, SLOTS, COUNTS, AMAP, T, TP: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
                 M_MAX: tl.constexpr, HAS_AMAP: tl.constexpr):
    """One expert: its tokens numbered in row order (deterministic, no atomics); any T up to TP. AMAP[e, slot]:
    token * K + list position."""
    _wait()
    e = tl.program_id(0)
    rows = tl.arange(0, TP)
    pos = tl.load(POS + rows * E + e, rows < T, -1)
    taken = pos >= 0
    t = taken.to(tl.int32)
    slot = tl.cumsum(t, 0) - t
    tl.store(SLOTS + rows * K + pos, slot, taken)
    if HAS_AMAP:
        tl.store(AMAP + e * M_MAX + slot, rows * K + pos, taken)
    tl.store(COUNTS + e, tl.sum(t, 0))
    _release()


def route(logits, bias, ids, weights, slots, counts, top_k, groups, top_groups, scale=None, amap=None):
    """Grouped sigmoid top-k of FP32 router logits (rounded to BF16 first, as the reference casts them). `scale`:
    the weights come out times `scale` and rounded to BF16 (in FP32), as the TRT-LLM-gen MoE packs them. `slots`
    None: no expert slots or counts (a fused MoE that permutes the rows itself). `amap` [E, M_max]: also each
    expert's slots as token * top_k + list position."""
    t, e = logits.shape
    pos = torch.empty(t, e, device=logits.device, dtype=torch.int32) if slots is not None else ids
    _route_rows[(t,)](logits, bias, ids, weights, pos, scale or 1.0, e, groups, top_groups, top_k, scale is not None,
                      slots is not None, num_warps=4, **LAUNCH)
    if slots is not None:
        _route_slots[(e,)](pos, slots, counts, amap if amap is not None else slots, t, triton.next_power_of_2(t), e,
                           top_k, amap.shape[1] if amap is not None else 0, amap is not None, num_warps=4, **LAUNCH)


@triton.jit
def _moe_swiglu(X, Y, IDS, SLOTS, lo, LOCAL: tl.constexpr, M_MAX: tl.constexpr, F: tl.constexpr, G: tl.constexpr):
    _wait()
    a = tl.program_id(0)
    e = tl.load(IDS + a) - lo
    if (e >= 0) & (e < LOCAL):
        row = e.to(tl.int64) * M_MAX + tl.load(SLOTS + a)
        cols, mask = _row_tile(F, G)
        gate = tl.load(X + row * 2 * F + cols, mask, 0.0).to(tl.float32)
        up = tl.load(X + row * 2 * F + F + cols, mask, 0.0).to(tl.float32)
        tl.store(Y + row * F + cols, (gate * tl.sigmoid(gate) * up).to(tl.bfloat16), mask)
    _release()


def moe_swiglu(x, ids, slots, out, lo=0):
    """SwiGLU of the assigned rows only, in the masked layout [E, M_max, 2F] -> [E, M_max, F]."""
    f = out.shape[2]
    _moe_swiglu[(ids.numel(),)](x, out, ids, slots, lo, x.shape[0], x.shape[1], f, triton.next_power_of_2(f // 128),
                                num_warps=4, **LAUNCH)


@triton.jit
def _moe_gate_up(A, W, Y, COUNTS, lo, M_MAX: tl.constexpr, F: tl.constexpr, H: tl.constexpr, BM: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    """One (expert, BN-column tile): [gate | up] of the expert's rows, rounded to BF16, then SwiGLU, into Y."""
    _wait()
    e, n = tl.program_id(0), tl.program_id(1)
    count = tl.load(COUNTS + lo + e)
    m = tl.arange(0, BM)
    cols = n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    gate_rows = (e.to(tl.int64) * 2 * F + cols)[:, None] * H
    up_rows = gate_rows + F * H
    for m0 in range(0, count, BM):
        rows = (e.to(tl.int64) * M_MAX + m0 + m)[:, None]
        g = tl.zeros((BM, BN), dtype=tl.float32)
        u = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, H, BK):
            a = tl.load(A + rows * H + k0 + k[None, :])
            g += tl.dot(a, tl.trans(tl.load(W + gate_rows + k0 + k[None, :])))
            u += tl.dot(a, tl.trans(tl.load(W + up_rows + k0 + k[None, :])))
        g, u = _bf(g), _bf(u)
        tl.store(Y + rows * F + cols[None, :], (g * tl.sigmoid(g) * u).to(tl.bfloat16), (m0 + m < count)[:, None])
    _release()


def _tile(local_experts):
    """Narrower column tiles when few experts are local (under expert parallelism), so that the few active ones still
    spread over enough programs to keep the weight streams busy."""
    return 64 if local_experts > 64 else 32


def moe_gate_up(a, w13, out, counts, lo=0):
    """SwiGLU(a @ w13[e]^T) of every local expert's assigned rows (masked layout [E, M_max, ...])."""
    e, m_max, h = a.shape
    f, bn = out.shape[2], _tile(a.shape[0])
    _moe_gate_up[(e, f // bn)](a, w13, out, counts, lo, m_max, f, h, 16, bn, 128, num_warps=4, num_stages=4, **LAUNCH)


@triton.jit
def _moe_down(A, W, Y, COUNTS, lo, M_MAX: tl.constexpr, N: tl.constexpr, F: tl.constexpr, BM: tl.constexpr,
              BN: tl.constexpr, BK: tl.constexpr):
    """One (expert, BN-column tile) of the down projection of the expert's rows."""
    _wait()
    e, n = tl.program_id(0), tl.program_id(1)
    count = tl.load(COUNTS + lo + e)
    m = tl.arange(0, BM)
    cols = n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    w_rows = (e.to(tl.int64) * N + cols)[:, None] * F
    for m0 in range(0, count, BM):
        rows = (e.to(tl.int64) * M_MAX + m0 + m)[:, None]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, F, BK):
            acc += tl.dot(tl.load(A + rows * F + k0 + k[None, :]), tl.trans(tl.load(W + w_rows + k0 + k[None, :])))
        tl.store(Y + rows * N + cols[None, :], acc.to(tl.bfloat16), (m0 + m < count)[:, None])
    _release()


def moe_down(a, w2, out, counts, lo=0):
    """a[e] @ w2[e]^T for every local expert's assigned rows (masked layout)."""
    e, m_max, f = a.shape
    n, bn = out.shape[2], _tile(a.shape[0])
    _moe_down[(e, n // bn)](a, w2, out, counts, lo, m_max, n, f, 16, bn, 128, num_warps=4, num_stages=3, **LAUNCH)


@triton.jit
def _moe_combine(D, IDS, SLOTS, W, S, Y, lo, scale, LOCAL: tl.constexpr, M_MAX: tl.constexpr, K: tl.constexpr,
                 N: tl.constexpr, G: tl.constexpr, HAS_S: tl.constexpr):
    _wait()
    row = tl.program_id(0)
    cols, mask = _row_tile(N, G)
    acc = tl.zeros((G, 128), dtype=tl.float32)
    for k in tl.static_range(K):
        e = tl.load(IDS + row * K + k) - lo
        local = (e >= 0) & (e < LOCAL)
        src = e.to(tl.int64) * M_MAX + tl.load(SLOTS + row * K + k)
        y = tl.load(D + src * N + cols, mask & local, 0.0).to(tl.float32)
        acc += _bf(tl.where(local, tl.load(W + row * K + k), 0.0) * y)
    out = _bf(acc * scale)
    if HAS_S:
        out = out + tl.load(S + row * N + cols, mask, 0.0).to(tl.float32)
    tl.store(Y + row * N + cols, out.to(tl.bfloat16), mask)
    _release()


def moe_combine(d, ids, slots, weights, out, scale, shared=None, lo=0):
    """out[token] = bf16(scale * sum_k bf16(w_k * d[expert_k, slot_k])) [+ shared[token]] over this rank's experts."""
    t, k = ids.shape
    n = d.shape[2]
    _moe_combine[(t,)](d, ids, slots, weights, shared if shared is not None else out, out, lo, scale, d.shape[0],
                       d.shape[1], k, n, triton.next_power_of_2(triton.cdiv(n, 128)), shared is not None,
                       num_warps=8, **LAUNCH)
    return out


@triton.jit
def _begin(P, PROMPT_IDS, TOKENS, PROMPT, ACCEPTED, START, END, PAIR_START, PAIR_ENDS, CTR, MASK: tl.constexpr,
           B: tl.constexpr):
    _wait()
    i = tl.arange(0, B)
    offset, count, first = tl.load(P), tl.load(P + 1), tl.load(P + 2)
    is_prompt = i < count
    tl.store(TOKENS + i, tl.where(is_prompt, tl.load(PROMPT_IDS + i, is_prompt, 0), MASK))
    tl.store(PROMPT + i, is_prompt)
    tl.store(ACCEPTED + i, is_prompt)
    tl.store(START, offset)
    tl.store(END, offset + B)
    tl.store(PAIR_START, offset - B)
    tl.store(PAIR_ENDS + tl.arange(0, 2), offset + B * tl.arange(0, 2))
    carry = tl.where(first != 0, 0, tl.load(CTR + CARRY))
    c = tl.arange(0, 16)
    value = tl.where(c == REMAINING, 1, 0)
    value = tl.where(c == CARRY, carry, value)
    value = tl.where(c == CARRY_LOOP, (carry != 0).to(tl.int32), value)
    value = tl.where(c == CARRY_T, tl.where(carry != 0, tl.load(CTR + CARRY_T), 0), value)
    tl.store(CTR + c, value)
    _release()


def begin(params, prompt_ids, tokens, prompt, accepted, start, end, pair_start, pair_ends, ctr, mask_id):
    """Block at params[0] with params[1] fixed prompt-tail rows, the rest `<|mask|>`; counters reset. params[2]
    marks a request's first block (no carried block). pair_start / pair_ends: the two-block window of a fused
    round (the carried block, then this one)."""
    _begin[(1,)](params, prompt_ids, tokens, prompt, accepted, start, end, pair_start, pair_ends, ctr, mask_id,
                 tokens.numel(), num_warps=1, **LAUNCH)


@triton.jit
def _denoise(conf, proposal, token, prompt, accepted, i, THRESH: tl.constexpr, B: tl.constexpr):
    """Accept every unresolved position above the threshold, or else the single most confident one."""
    eligible = ~accepted & ~prompt
    high = eligible & (conf > THRESH)
    score = tl.where(eligible, conf, -1.0)
    best = tl.min(tl.where(eligible & (score == tl.max(score, 0)), i, B), 0)
    return tl.where(tl.sum(high.to(tl.int32), 0) > 0, high, i == best)


@triton.jit
def _decide(STATS, PROPOSAL, TOKENS, PROMPT, ACCEPTED, CARRY_TOKENS, CARRY_PROMPT, CTR, THRESH: tl.constexpr,
            EDIT: tl.constexpr, CHANGE: tl.constexpr, REFINES: tl.constexpr, EOS: tl.constexpr, B: tl.constexpr,
            FUSED: tl.constexpr):
    """state: 0 denoising, 1 refining (edits only), 2 confirm (forward only, writes the final K/V), 3 done."""
    _wait()
    i = tl.arange(0, B)
    state, denoise, refine, terminal = tl.load(CTR + STATE), tl.load(CTR + DENOISE), tl.load(CTR + REFINE), \
        tl.load(CTR + TERMINAL)
    conf = tl.load(STATS + i * 3 + 1)
    proposal = tl.load(PROPOSAL + i)
    token = tl.load(TOKENS + i)
    prompt = tl.load(PROMPT + i) != 0
    accepted = tl.load(ACCEPTED + i) != 0
    accept = (state == 0) & _denoise(conf, proposal, token, prompt, accepted, i, THRESH, B)
    # Editing: positions accepted before this round (all of them once the block is complete).
    editable = tl.where(state == 0, accepted, True) & ~prompt
    edit = CHANGE & (state <= 1) & editable & (conf > EDIT) & (proposal != token)
    token = tl.where(accept | edit, proposal, token)
    accepted = accepted | accept
    tl.store(TOKENS + i, token)
    tl.store(ACCEPTED + i, accepted)
    complete = tl.sum(accepted.to(tl.int32), 0) == B
    eos = tl.sum(((token == EOS) & ~prompt).to(tl.int32), 0) > 0
    edited = tl.sum(edit.to(tl.int32), 0) > 0
    # A finished block's K/V must come from a forward over its final tokens: a refine round without edits, or a
    # confirm round after the last allowed edit. A block holding an EOS when completed ends the request, so it
    # skips the confirm round, as in the reference. FUSED: a finished block without an EOS is carried instead; its
    # refine rounds run inside the next block's first forwards (see decide_pair).
    terminal = tl.where((state == 0) & complete, eos.to(tl.int32), terminal)
    settle = tl.where(terminal != 0, 3, 2)
    refined = refine + (state == 1).to(tl.int32)
    if CHANGE and REFINES > 0:
        after_denoise = tl.where(complete, 1, 0)
    else:
        after_denoise = tl.where(complete, settle, 0)
    if FUSED:
        carried = (state == 0) & complete & (terminal == 0)
        after_denoise = tl.where(carried, 3, after_denoise)
        tl.store(CARRY_TOKENS + i, token, carried)
        tl.store(CARRY_PROMPT + i, prompt, carried)
        tl.store(CTR + CARRY, 1, carried)
        tl.store(CTR + CARRY_NEW, 1, carried)
        tl.store(CTR + CARRY_T, tl.load(CTR + STEP) + 1, carried)
    after_refine = tl.where(~edited, 3, tl.where(refined >= REFINES, settle, 1))
    new = tl.where(state == 0, after_denoise, tl.where(state == 1, after_refine, 3))
    tl.store(CTR + STEP, tl.load(CTR + STEP) + 1)
    tl.store(CTR + REMAINING, (new != 3).to(tl.int32))
    tl.store(CTR + STATE, new)
    tl.store(CTR + DENOISE, denoise + (state == 0).to(tl.int32))
    tl.store(CTR + REFINE, refined)
    tl.store(CTR + TERMINAL, terminal)
    tl.store(CTR + FORWARDS, tl.load(CTR + FORWARDS) + 1)
    _release()


def decide(stats, proposal, tokens, prompt, accepted, carry_tokens, carry_prompt, ctr, config, eos, fused):
    """One round of the paper's LLaDA decoder on the block (see LladaDecode)."""
    _decide[(1,)](stats, proposal, tokens, prompt, accepted, carry_tokens, carry_prompt, ctr, config.prob_threshold,
                  config.edit_threshold, config.allow_accepted_change, config.refine_steps, eos, tokens.numel(), fused,
                  num_warps=1, **LAUNCH)


@triton.jit
def _decide_pair(STATS, PROPOSAL, TOKENS, PROMPT, ACCEPTED, FINAL, CTR, THRESH: tl.constexpr, EDIT: tl.constexpr,
                 REFINES: tl.constexpr, EOS: tl.constexpr, B: tl.constexpr):
    """A fused round over [carried block | this block]: rows [0, B) of every buffer are the carried block."""
    _wait()
    i = tl.arange(0, B)
    carry = tl.load(CTR + CARRY)
    # Carried block: one refine round (edits only), or none once it reached the edit cap (the forward then only
    # rewrites its K/V over the final tokens).
    conf = tl.load(STATS + i * 3 + 1)
    proposal = tl.load(PROPOSAL + i)
    old = tl.load(TOKENS + i)
    old_prompt = tl.load(PROMPT + i) != 0
    edit = (carry == 1) & ~old_prompt & (conf > EDIT) & (proposal != old)
    old = tl.where(edit, proposal, old)
    edited = tl.sum(edit.to(tl.int32), 0) > 0
    edits = tl.load(CTR + CARRY_REFINE) + edited.to(tl.int32)
    settled = ~edited
    tl.store(FINAL + i, old)
    # This block's first denoise round, which the carried block's settling round doubles as.
    conf = tl.load(STATS + (B + i) * 3 + 1)
    proposal = tl.load(PROPOSAL + B + i)
    token = tl.load(TOKENS + B + i)
    prompt = tl.load(PROMPT + B + i) != 0
    accepted = tl.load(ACCEPTED + i) != 0
    accept = settled & _denoise(conf, proposal, token, prompt, accepted, i, THRESH, B)
    token = tl.where(accept, proposal, token)
    accepted = accepted | accept
    tl.store(TOKENS + B + i, token)
    tl.store(ACCEPTED + i, accepted)
    complete = settled & (tl.sum(accepted.to(tl.int32), 0) == B)
    eos = tl.sum(((token == EOS) & ~prompt).to(tl.int32), 0) > 0
    terminal = complete & eos
    carried = complete & ~eos
    tl.store(TOKENS + i, tl.where(carried, token, old))
    tl.store(PROMPT + i, tl.where(carried, prompt, old_prompt))
    state = tl.where(terminal, 1, tl.where(carried, 3, 0))
    carry = tl.where(carried, 1, tl.where(settled, 0, tl.where(edits >= REFINES, 2, 1)))
    tl.store(CTR + CARRY, carry)
    tl.store(CTR + CARRY_NEW, carried.to(tl.int32), carried)
    tl.store(CTR + CARRY_LOOP, edited.to(tl.int32))
    tl.store(CTR + CARRY_STEP, tl.load(CTR + CARRY_STEP) + 1)
    tl.store(CTR + CARRY_T, tl.where(carried, 1, tl.load(CTR + CARRY_T) + 1))
    tl.store(CTR + CARRY_REFINE, tl.where(carried, 0, edits))
    tl.store(CTR + RESOLVED, settled.to(tl.int32))
    tl.store(CTR + STEP, settled.to(tl.int32))
    tl.store(CTR + STATE, state)
    tl.store(CTR + REMAINING, (settled & (state != 3)).to(tl.int32))
    tl.store(CTR + DENOISE, settled.to(tl.int32))
    tl.store(CTR + REFINE, tl.load(CTR + REFINE) + edited.to(tl.int32))
    tl.store(CTR + TERMINAL, terminal.to(tl.int32))
    tl.store(CTR + FORWARDS, tl.load(CTR + FORWARDS) + 1)
    _release()


def decide_pair(stats, proposal, tokens, prompt, accepted, final, ctr, config, eos):
    """Decisions of a fused round. `stats` / `proposal` / `tokens` / `prompt` cover [carried block | this block];
    `final` receives the carried block's tokens after this round (its final tokens once RESOLVED is set)."""
    _decide_pair[(1,)](stats, proposal, tokens, prompt, accepted, final, ctr, config.prob_threshold,
                       config.edit_threshold, config.refine_steps, eos, accepted.numel(), num_warps=1, **LAUNCH)


@triton.jit
def _carry_save(LOGITS, STATS, CTR, V: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    _wait()
    row, part = tl.program_id(0), tl.program_id(1)
    if tl.load(CTR + CARRY_NEW) != 0:
        cols = part * BLOCK + tl.arange(0, BLOCK)
        mask = cols < V
        tl.store(LOGITS + row * V + cols, tl.load(LOGITS + (ROWS + row) * V + cols, mask), mask)
        if part == 0:
            k = tl.arange(0, 4)
            tl.store(STATS + row * 3 + k, tl.load(STATS + (ROWS + row) * 3 + k, k < 3), k < 3)
    _release()


def carry_save(logits, stats, ctr):
    """When a block was just carried: its last logits and statistics (rows [B, 2B)) become the self-conditioning
    source of its refine rounds (rows [0, B))."""
    rows, v = logits.shape[0] // 2, logits.shape[1]
    _carry_save[(rows, triton.cdiv(v, 4096))](logits, stats, ctr, v, rows, 4096, num_warps=8, **LAUNCH)
