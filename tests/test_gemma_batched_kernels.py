"""DiffusionGemma's batched kernels (batched.py) against the single-request kernels, on synthetic data (no model): every
per-row and per-request kernel run over a ragged batch (requests of different canvas lengths at row offsets, in
different stages, and a commit lane) must give each request bit for bit what the single-request kernel gives it alone;
the stage kernels (admit, advance, claim, control) move each request as batched.py describes."""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

dev = "cuda"
i32 = {"device": dev, "dtype": torch.int32}
S, IDLE, V, W = 4096, 7, 262144, 1024  # arena positions per slot, idle slot, vocabulary, sliding window
DENOISE, COMMIT, PREFILL, DONE, IDLE_MODE, ADMIT, SWITCH = range(7)
BLOCKS = (256, 128, 64, 32)
# (slot, rows, canvas start, mode) in row order. Slot 6 commits across the window's edge at a phase whose causal rows
# see none of their query tile's first key tile (with one key split).
REQS = [(2, 64, 700, DENOISE), (5, 32, 1300, DENOISE), (1, 128, 300, DENOISE), (3, 32, 2000, COMMIT),
        (6, 32, 1063, COMMIT)]
LANE = 64  # then a 64-row commit lane: slot 2's previous canvas, at indices -64..-1


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def batch():
    offsets, o = [], 0
    for _, b, _, _ in REQS:
        offsets.append(o)
        o += b
    own, rows = o, o + LANE
    row_slot, row_index = torch.full((rows,), IDLE, **i32), torch.arange(rows, **i32) % 32
    for (slot, b, _, _), off in zip(REQS, offsets):
        row_slot[off:off + b], row_index[off:off + b] = slot, torch.arange(b, **i32)
    row_slot[own:], row_index[own:] = REQS[0][0], torch.arange(LANE, **i32) - LANE
    start, end, left, modes = (torch.zeros(8, **i32) for _ in range(4))
    modes.fill_(IDLE_MODE)
    end[IDLE] = 32
    for slot, b, s, mode in REQS:
        start[slot], end[slot], left[slot], modes[slot] = s, s + b, max(0, s - (W - 1)), mode
    state = torch.stack([t[row_slot.long()] for t in (start, end, left, modes)], 1).contiguous()
    lane = (REQS[0][0], LANE, REQS[0][2] - LANE, COMMIT)
    return types.SimpleNamespace(offsets=offsets, own=own, rows=rows, row_slot=row_slot, row_index=row_index,
                                 start=start, end=end, left=left, modes=modes, state=state,
                                 segments=list(zip(REQS, offsets)) + [(lane, own)])


def one(t, slot):
    return t[slot:slot + 1]


@pytest.mark.parametrize("sliding", [True, False])
def test_attention_slots(batch, sliding):
    """Canvas rows over [left, end), causal rows (commit, lane) within the window, each over its slot's arena; with
    one key split, as batched forwards of >= 512 rows run."""
    from dlmrt.ops.triton.attention import attention

    hq, hk, d, window, bm, warps = (16, 8, 256, W, 64, 4) if sliding else (16, 2, 512, 0, 128, 8)
    tiles = dict(block_m=bm, block_n=64 if sliding else 32, num_warps=warps, programs=1)
    k = torch.randn(8, S, hk, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(8, S, hk, d, device=dev, dtype=torch.bfloat16)
    q = torch.randn(1, batch.rows, hq, d, device=dev, dtype=torch.bfloat16)
    left = batch.left if sliding else None
    got = attention(q, k, v, batch.start, batch.end, False, window=window, left=left, slots=batch.row_slot,
                    index=batch.row_index, block=32, modes=batch.modes, **tiles)
    assert bool(torch.isfinite(got).all())
    for (slot, b, s, mode), off in batch.segments:
        st, en = torch.tensor([s], **i32), torch.tensor([s + b], **i32)
        if mode == COMMIT:
            ref = attention(q[:, off:off + b], one(k, slot), one(v, slot), st, en, True, window=window, **tiles)
        else:
            lf = torch.tensor([max(0, s - (W - 1))], **i32) if sliding else None
            ref = attention(q[:, off:off + b], one(k, slot), one(v, slot), st, en, False, left=lf, **tiles)
        assert torch.equal(got[:, off:off + b], ref), (slot, b, mode)
    with_state = attention(q, k, v, batch.start, batch.end, False, window=window, left=left, slots=batch.row_slot,
                           index=batch.row_index, block=32, modes=batch.modes, state=batch.state, **tiles)
    assert torch.equal(with_state, got)


def test_qkv_rope_slots(batch):
    """Q, and K/V at start[slot] + index of each row's arena (a lane's rows before its slot's start)."""
    from dlmrt.models.diffusion_gemma import kernels as gk

    hq, hk, d = 16, 8, 256
    qkv = torch.randn(batch.rows, (hq + 2 * hk) * d, device=dev, dtype=torch.bfloat16)
    qw, kw = (torch.randn(d, device=dev, dtype=torch.bfloat16) for _ in range(2))
    ang = torch.arange(S, device=dev).float()[:, None] * torch.rand(d // 2, device=dev)[None]
    cos, sin = ang.cos().bfloat16(), ang.sin().bfloat16()
    k, v = torch.zeros(8, S, hk, d, device=dev, dtype=torch.bfloat16), torch.zeros(8, S, hk, d, device=dev,
                                                                                   dtype=torch.bfloat16)
    q = torch.empty(1, batch.rows, hq, d, device=dev, dtype=torch.bfloat16)
    gk.qkv_rope(qkv, qw, kw, cos, sin, batch.start, q, k, v, hq, hk, d, True, 1e-6, split=True, slots=batch.row_slot,
                index=batch.row_index)
    for (slot, b, s, _), off in batch.segments:
        k1, v1 = torch.zeros(1, S, hk, d, device=dev, dtype=torch.bfloat16), torch.zeros(1, S, hk, d, device=dev,
                                                                                         dtype=torch.bfloat16)
        q1 = torch.empty(1, b, hq, d, device=dev, dtype=torch.bfloat16)
        gk.qkv_rope(qkv[off:off + b], qw, kw, cos, sin, torch.tensor([s], **i32), q1, k1, v1, hq, hk, d, True, 1e-6,
                    split=True)
        assert torch.equal(q[:, off:off + b], q1)
        assert torch.equal(k[slot, s:s + b], k1[0, s:s + b]) and torch.equal(v[slot, s:s + b], v1[0, s:s + b])
    k2, v2, q2 = torch.zeros_like(k), torch.zeros_like(v), torch.empty_like(q)
    gk.qkv_rope(qkv, qw, kw, cos, sin, batch.start, q2, k2, v2, hq, hk, d, True, 1e-6, split=True,
                slots=batch.row_slot, index=batch.row_index, state=batch.state)
    assert torch.equal(q2, q) and torch.equal(k2, k) and torch.equal(v2, v)


@pytest.fixture(scope="module")
def decoding(batch):
    """Logits of the requests' rows and per-slot decoder state (rounds 0-2, adaptive temperature)."""
    from dlmrt.models.diffusion_gemma import ROLES

    torch.manual_seed(1)
    own = batch.own
    step = torch.zeros(8, **i32)
    for slot, _, _, _ in REQS:
        step[slot] = slot % 3
    slots = torch.tensor([r[0] for r in REQS] + [IDLE] * 3, **i32)
    offsets, lengths = torch.zeros(8, **i32), torch.zeros(8, **i32)
    for (slot, b, _, _), off in zip(REQS, batch.offsets):
        offsets[slot], lengths[slot] = off, b
    return types.SimpleNamespace(
        dec=ROLES["orl"], logits=(torch.randn(own, V, device=dev) * 4).bfloat16(), step=step,
        seed=torch.full((8,), 1234, **i32), ordinal=torch.arange(8, **i32), remaining=torch.ones(8, **i32),
        temp=torch.full((8,), 0.7, device=dev), h_ref=torch.full((8,), 0.7, device=dev),
        t_prev=torch.full((8,), 0.7, device=dev), accepted=(torch.rand(own, device=dev) < 0.3).to(torch.int8),
        tokens=torch.randint(0, V, (own,), device=dev), out=torch.randint(0, V, (own,), device=dev), slots=slots,
        offsets=offsets, lengths=lengths, stops=("renoise", "renoise", "renoise", "freeze_argmax"))


def test_decoder_per_request(batch, decoding):
    """Adaptive temperature, logit statistics and decide per request (freeze_argmax for 32-row canvases, renoise for
    the others, one launch), against each request decoded alone."""
    import dataclasses

    from dlmrt.models.diffusion_gemma import kernels as gk

    t = decoding
    own = batch.own
    state = [x.clone() for x in (t.temp, t.h_ref, t.t_prev, t.step, t.remaining, t.tokens, t.out, t.accepted)]
    temp, h_ref, t_prev, step, remaining, tokens, out, accepted = (x.clone() for x in state)
    stats = torch.zeros(own, 2, device=dev)
    argmax, sample = torch.zeros(own, device=dev, dtype=torch.int64), torch.zeros(own, device=dev, dtype=torch.int64)
    q = dict(slots=t.slots, offsets=t.offsets, blocks=t.lengths, classes=BLOCKS)
    gk.adaptive_temperature(t.logits, step, accepted, temp, h_ref, t_prev, t.dec, 30.0, **q)
    gk.logit_stats(t.logits, step, t.seed, t.ordinal, t.dec, stats, argmax, sample, 30.0, temp=temp, rand4=True,
                   slots=batch.row_slot[:own], index=batch.row_index[:own])
    gk.decide(stats, argmax, sample, tokens, out, accepted, step, remaining, t.seed, t.ordinal, t.dec, V, stops=t.stops,
              **q)
    for (slot, b, _, _), off in zip(REQS, batch.offsets):
        tp, hr, tq, sp, rm, tk, ot, ac = (x.clone() for x in state)
        tk, ot, ac = tk[off:off + b].clone(), ot[off:off + b].clone(), ac[off:off + b].clone()
        t1, h1, p1, s1, r1, sd, od = (one(x, slot).clone() for x in (tp, hr, tq, sp, rm, t.seed, t.ordinal))
        st1 = torch.zeros(b, 2, device=dev)
        am1, sm1 = torch.zeros(b, device=dev, dtype=torch.int64), torch.zeros(b, device=dev, dtype=torch.int64)
        d1 = dataclasses.replace(t.dec, stop_rule="freeze_argmax" if b == 32 else "renoise")
        gk.adaptive_temperature(t.logits[off:off + b], s1, ac, t1, h1, p1, d1, 30.0)
        gk.logit_stats(t.logits[off:off + b], s1, sd, od, d1, st1, am1, sm1, 30.0, temp=t1, rand4=True)
        gk.decide(st1, am1, sm1, tk, ot, ac, s1, r1, sd, od, d1, V)
        assert torch.equal(one(temp, slot), t1), slot
        assert torch.equal(stats[off:off + b], st1) and torch.equal(argmax[off:off + b], am1), slot
        assert torch.equal(sample[off:off + b], sm1), slot
        assert torch.equal(tokens[off:off + b], tk) and torch.equal(out[off:off + b], ot), slot
        assert torch.equal(accepted[off:off + b], ac), slot
        assert torch.equal(one(step, slot), s1) and torch.equal(one(remaining, slot), r1), slot


def test_row_skipping(batch, decoding):
    """With the forward's row state, the decoder passes compute the rows of denoising requests as without it and leave
    the others untouched; with argmax_only, 32-row canvases draw no sample and nothing else changes."""
    from dlmrt.models.diffusion_gemma import kernels as gk

    t = decoding
    own, state = batch.own, batch.state[:batch.own]
    canvas = state[:, 3] == DENOISE
    full = [torch.empty(own, 2, device=dev), torch.empty(own, device=dev, dtype=torch.int64),
            torch.empty(own, device=dev, dtype=torch.int64)]
    args = dict(temp=t.temp, rand4=True, slots=batch.row_slot[:own], index=batch.row_index[:own])
    gk.logit_stats(t.logits, t.step, t.seed, t.ordinal, t.dec, *full, 30.0, **args)
    skip = [torch.full_like(full[0], -5.0), torch.full_like(full[1], -5), torch.full_like(full[2], -5)]
    gk.logit_stats(t.logits, t.step, t.seed, t.ordinal, t.dec, *skip, 30.0, rows=state, **args)
    for a, b in zip(skip, full):
        assert torch.equal(a[canvas], b[canvas]) and bool((a[~canvas] == -5).all())
    only = [torch.full_like(full[0], -5.0), torch.full_like(full[1], -5), torch.full_like(full[2], -5)]
    gk.logit_stats(t.logits, t.step, t.seed, t.ordinal, t.dec, *only, 30.0, rows=state, argmax_only=32, **args)
    short = canvas & (state[:, 1] - state[:, 0] == 32)
    assert torch.equal(only[0][canvas], full[0][canvas]) and torch.equal(only[1][canvas], full[1][canvas])
    assert torch.equal(only[2][canvas & ~short], full[2][canvas & ~short])
    probs = torch.empty(own, V, device=dev, dtype=torch.bfloat16)
    gk.sc_probs(t.logits, full[0], t.step, t.dec, probs, 30.0, temp=t.temp, slots=batch.row_slot[:own])
    skipped = torch.full((own, V), 7.0, device=dev, dtype=torch.bfloat16)
    gk.sc_probs(t.logits, full[0], t.step, t.dec, skipped, 30.0, temp=t.temp, slots=batch.row_slot[:own],
                modes=batch.modes)
    sc = canvas & (t.step[batch.row_slot[:own].long()] > 0)
    assert torch.equal(skipped[sc], probs[sc]) and bool((skipped[~sc] == 7.0).all())
    for (slot, b, _, _), off in zip(REQS, batch.offsets):
        p1 = torch.empty(b, V, device=dev, dtype=torch.bfloat16)
        gk.sc_probs(t.logits[off:off + b], full[0][off:off + b], one(t.step, slot), t.dec, p1, 30.0,
                    temp=one(t.temp, slot))
        assert torch.equal(probs[off:off + b], p1), slot


@pytest.mark.parametrize("stop_rule", ["renoise", "freeze_argmax"])
def test_decide_nan_rows_slots(stop_rule):
    """A request's rows of NaN logits are re-noised and counted in its slot's launch too."""
    from dlmrt.models.diffusion_gemma import kernels as gk
    from dlmrt.models.diffusion_gemma import GemmaDecode

    b, cap, d = 256, 30.0, GemmaDecode()
    logits = (torch.randn(b, V, device=dev) * 3).to(torch.bfloat16)
    nan = torch.zeros(b, dtype=torch.bool, device=dev)
    nan[b // 8:b // 2:3] = True
    logits[nan] = float("nan")
    step, seed, block = torch.tensor([5], **i32), torch.tensor([1234], **i32), torch.tensor([2], **i32)
    stats = torch.zeros(b, 2, device=dev)
    argmax, sample = torch.zeros(b, device=dev, dtype=torch.int64), torch.zeros(b, device=dev, dtype=torch.int64)
    gk.logit_stats(logits, step, seed, block, d, stats, argmax, sample, cap)
    tokens, canvas = torch.zeros(b, device=dev, dtype=torch.int64), torch.full((b,), -1, device=dev, dtype=torch.int64)
    remaining, bad = torch.tensor([1], **i32), torch.zeros(1, **i32)
    accepted = torch.zeros(b, device=dev, dtype=torch.int8)
    gk.decide(stats, argmax, sample, tokens, canvas, accepted, step, remaining, seed, block, d, V,
              slots=torch.zeros(1, **i32), offsets=torch.zeros(1, **i32), blocks=torch.tensor([b], **i32),
              classes=(b,), stops=(stop_rule,), bad=bad)
    assert int(bad) == int(nan.sum()) and not bool(accepted[nan].any())
    assert bool(((tokens >= 0) & (tokens < V)).all()) and bool(((canvas >= 0) & (canvas < V)).all())


def test_embed_slots(batch):
    """The decoder path for denoising rows (the SC delta from round 1 on), the plain one for commit and lane rows; the
    rows' slot state gathered on the way."""
    from dlmrt.models.diffusion_gemma import kernels as gk

    h, rows = 2816, batch.rows
    table = torch.randn(4096, h, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, 4096, (rows,), device=dev)
    step = torch.ones(8, **i32)
    step[5] = 0
    delta = torch.randn(rows, h, device=dev, dtype=torch.bfloat16)
    out = torch.empty(rows, h, device=dev, dtype=torch.bfloat16)
    gathered = torch.zeros(rows, 4, **i32)
    gk.embed(ids, table, 53.0, out, 1e-6, decoder=True, delta=delta, modes=batch.modes, slots=batch.row_slot, step=step,
             index=batch.row_index, state=(batch.start, batch.end, batch.left, gathered))
    assert torch.equal(gathered, batch.state)
    for (slot, b, _, mode), off in batch.segments:
        h1 = torch.empty(b, h, device=dev, dtype=torch.bfloat16)
        if mode == COMMIT:
            gk.embed(ids[off:off + b], table, 53.0, h1, 1e-6, decoder=False)
        else:
            d1 = delta[off:off + b] if int(step[slot]) > 0 else torch.zeros_like(delta[off:off + b])
            gk.embed(ids[off:off + b], table, 53.0, h1, 1e-6, decoder=True, delta=d1)
        assert torch.equal(out[off:off + b], h1), (slot, mode)


def stage_state(n_slots=8, rows=1024, seq=4608):
    """Slot state for the stage kernels: every slot idle, prompts of 100 tokens, budgets of 4096."""
    return types.SimpleNamespace(
        mode=torch.full((n_slots,), IDLE_MODE, **i32), P=torch.zeros(n_slots, 3, **i32),
        tokens=torch.zeros(rows, device=dev, dtype=torch.int64), canvas=torch.randint(3, 100, (rows,), device=dev),
        accepted=torch.zeros(rows, device=dev, dtype=torch.int8), start=torch.zeros(n_slots, **i32),
        end=torch.zeros(n_slots, **i32), left=torch.zeros(n_slots, **i32), step=torch.full((n_slots,), 3, **i32),
        remaining=torch.zeros(n_slots, **i32), seed=torch.full((n_slots,), 77, **i32),
        ordinal=torch.zeros(n_slots, **i32), slots=torch.full((8,), IDLE, **i32), offsets=torch.zeros(n_slots, **i32),
        lengths=torch.zeros(n_slots, **i32), want=torch.zeros(n_slots, **i32),
        prompt=torch.randint(3, 100, (n_slots, seq), device=dev), plen=torch.full((n_slots,), 100, **i32),
        max_new=torch.full((n_slots,), 4096, **i32), token_out=torch.zeros(n_slots, seq, device=dev, dtype=torch.int64),
        step_out=torch.zeros(n_slots, 160, **i32), block_out=torch.zeros(n_slots, 160, **i32),
        clock_out=torch.zeros(n_slots, 160, **i32), canvases=torch.zeros(n_slots, **i32),
        eos=torch.tensor([1, 106], device=dev), ctrl=torch.tensor([0, 0, 0, 100, 0, 0, 0, 0], **i32),
        counts=torch.zeros(24, **i32), lane=torch.zeros(16 + 2 * (rows // 32), **i32), wants=torch.zeros(8, **i32),
        row_slot=torch.full((rows,), IDLE, **i32), row_index=torch.arange(rows, **i32) % 32)


def place(t, layout):
    """Requests in position order: [(slot, rows, wanted canvas length)], contiguous from row 0. Returns their rows."""
    o = 0
    for p, (s, b, w) in enumerate(layout):
        t.slots[p], t.offsets[s], t.lengths[s], t.want[s] = s, o, b, w
        t.row_slot[o:o + b], t.row_index[o:o + b] = s, torch.arange(b, **i32)
        o += b
    return o


def advance(t, fused=False):
    from dlmrt.models.diffusion_gemma import kernels as gk

    if fused:
        gk.release_lanes(t.lane, t.row_slot, t.row_index, IDLE, 32, t.row_slot.numel())
    gk.advance(t.mode, t.P, t.tokens, t.canvas, t.accepted, t.start, t.end, t.left, t.step, t.remaining, t.seed,
               t.ordinal, t.slots, t.offsets, t.lengths, t.want, t.prompt, t.plen, t.max_new, t.token_out, t.step_out,
               t.block_out, t.canvases, t.eos, t.ctrl, W, V, BLOCKS, 48, clock=(t.counts, t.clock_out),
               lanes=(t.lane, t.wants) if fused else None)
    if fused:
        gk.claim(t.mode, t.P, t.tokens, t.canvas, t.accepted, t.start, t.end, t.left, t.step, t.remaining, t.seed,
                 t.ordinal, t.slots, t.offsets, t.lengths, t.wants, t.lane, t.row_slot, t.row_index, W, V, BLOCKS)
    torch.cuda.synchronize()


def test_advance_stages():
    """A stopped canvas is recorded (tokens, rounds, length, forward clock) and committed next, or ends the request
    with an EOS in its budget; a commit begins the next canvas, or parks the request if it wants another length; the
    prompt's last chunk begins the first canvas."""
    t = stage_state()
    layout = [(2, 64, 64), (5, 32, 32), (4, 32, 32), (6, 32, 64), (1, 128, 128)]
    place(t, layout)
    t.mode[[2, 5, 4]] = DENOISE
    t.mode[6], t.mode[1] = COMMIT, PREFILL
    t.P[2], t.P[5], t.P[4], t.P[6] = (torch.tensor(x, **i32) for x in ([700, 2, 9], [300, 0, 9], [500, 1, 9],
                                                                         [900, 3, 9]))
    t.start[1], t.plen[1] = 0, 100
    t.counts[16] = 41
    t.canvas[int(t.offsets[4]) + 5] = 106  # an EOS in slot 4's canvas
    before = t.canvas.clone()
    advance(t)
    o2, o5 = int(t.offsets[2]), int(t.offsets[5])
    assert torch.equal(t.token_out[2, 600:664], before[o2:o2 + 64]) and torch.equal(t.tokens[o2:o2 + 64],
                                                                                    before[o2:o2 + 64])
    assert t.step_out[2, 2] == 3 and t.block_out[2, 2] == 64 and t.clock_out[2, 2] == 41
    assert torch.equal(t.token_out[5, 200:232], before[o5:o5 + 32])
    assert t.mode.tolist()[:7] == [IDLE_MODE, DENOISE, COMMIT, IDLE_MODE, DONE, COMMIT, SWITCH]
    assert t.canvases[4] == 2 and t.ctrl[2] == 2  # slot 4 finished, slot 6 parked
    assert t.P[6].tolist() == [932, 4, 9]  # slot 6: its committed canvas passed, the next one waits for 64 rows
    assert t.P[1].tolist() == [100, 0, 0] and t.start[1] == 100 and t.step[1] == 0 and t.remaining[1] == 1


def test_admit_and_control():
    """admit: an admitted request starts its prefill with its first prompt rows, a parked one begins its canvas.
    control: run while a request is active and none finished; the decoder / SC flags; requests and rows per mode."""
    from dlmrt.models.diffusion_gemma import kernels as gk

    t = stage_state()
    rows = place(t, [(3, 64, 64), (5, 32, 32)])
    t.mode[3], t.mode[5] = ADMIT, SWITCH
    t.P[5] = torch.tensor([400, 2, 9], **i32)
    gk.admit(t.mode, t.P, t.tokens, t.canvas, t.accepted, t.start, t.end, t.left, t.step, t.remaining, t.seed,
             t.ordinal, t.slots, t.offsets, t.lengths, t.prompt, t.plen, W, V, BLOCKS)
    torch.cuda.synchronize()
    assert torch.equal(t.tokens[:64], torch.cat([t.prompt[3, :64]]))
    assert t.mode[3] == PREFILL and t.start[3] == 0 and t.end[3] == 64
    assert t.mode[5] == DENOISE and t.start[5] == 400 and t.step[5] == 0 and t.ordinal[5] == 2
    t.ctrl.zero_()
    t.ctrl[3] = 1 << 30
    t.counts.zero_()
    gk.control(t.mode, t.step, t.slots, t.lengths, t.ctrl, t.counts, rows, True)
    torch.cuda.synchronize()
    assert t.ctrl[[0, 4, 5]].tolist() == [1, 1, 0]  # run; a canvas round (slot 5, round 0), no SC delta to read
    assert t.counts[[DENOISE, PREFILL, 8 + DENOISE, 8 + PREFILL]].tolist() == [1, 1, 32, 64]
    t.step[5] = 1
    gk.control(t.mode, t.step, t.slots, t.lengths, t.ctrl, t.counts, rows, False)
    torch.cuda.synchronize()
    assert t.ctrl[[0, 1, 5, 6]].tolist() == [1, 1, 1, 1] and t.counts[16] == 1
    t.ctrl[2] = 1 << 30
    gk.control(t.mode, t.step, t.slots, t.lengths, t.ctrl, t.counts, rows, False)
    torch.cuda.synchronize()
    assert t.ctrl[0] == 0  # the finish limit reached


def test_claim_lanes():
    """Fused: stopped canvases of lane length take the launch's lanes in position order (their tokens and row map in
    the lane, the next canvas begun); the rest, longer canvases and requests that want another length commit in their
    rows; the lane counters; bit-identical over repeats."""
    from dlmrt.models.diffusion_gemma import kernels as gk

    layout = [(3, 64, 64), (0, 32, 32), (5, 32, 32), (2, 64, 64), (6, 32, 64), (1, 128, 128), (4, 32, 32)]
    size, lanes = 64, 3

    def run(fused=True, count=lanes):
        torch.manual_seed(3)
        t = stage_state()
        own = place(t, layout)
        t.mode[[s for s, _, _ in layout]] = DENOISE
        t.P[:, 0], t.P[:, 1], t.P[:, 2] = 1000 + 100 * torch.arange(8, **i32), 2, 77
        t.canvas[int(t.offsets[4]) + 5] = 106
        t.lane[:3] = torch.tensor([count, size, own], **i32)
        advance(t, fused)
        if fused:
            gk.control(t.mode, t.step, t.slots, t.lengths, t.ctrl, t.counts, own, False, lane=t.lane, classes=BLOCKS)
            torch.cuda.synchronize()
        return t, own

    t, own = run()
    torch.manual_seed(3)
    c0 = stage_state().canvas
    for j, s in enumerate((3, 0, 5)):  # the first three waiting canvases in position order
        o, b, first = int(t.offsets[s]), int(t.lengths[s]), own + j * size
        assert torch.equal(t.tokens[first:first + b], c0[o:o + b])
        assert t.row_slot[first:first + b].tolist() == [s] * b and t.row_slot[first + b:first + size].tolist() == \
            [IDLE] * (size - b)
        assert t.row_index[first:first + b].tolist() == list(range(-b, 0)) and t.lane[16 + 2 * j:18 + 2 * j].tolist() \
            == [s, b]
        assert t.mode[s] == DENOISE and t.P[s, :2].tolist() == [1000 + 100 * s + b, 3] and t.step[s] == 0
    for s in (2, 6, 1):  # no lane left, wants another length, longer than a lane
        o, b = int(t.offsets[s]), int(t.lengths[s])
        assert t.mode[s] == COMMIT and torch.equal(t.tokens[o:o + b], c0[o:o + b])
    assert t.mode[4] == DONE and t.ctrl[2] == 1
    assert t.lane[3:6].tolist() == [3, 3, 128] and t.lane[8:12].tolist() == [0, 0, 2, 2]
    assert t.lane[6] == lanes * size and t.lane[12:16].tolist() == [0, 0, 2, 3]
    assert t.wants.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    for _ in range(20):
        again, _ = run()
        assert all(torch.equal(getattr(again, k), getattr(t, k)) for k in vars(t))
    plain = run(fused=False)[0]
    for name, other in (("without lanes", plain), ("no lane in the launch", run(count=0)[0])):
        assert other.mode[[s for s, _, _ in layout]].tolist() == [1, 1, 1, 1, 1, 1, 3], name
        assert torch.equal(other.tokens[:own], plain.tokens[:own]), name


def test_prompt_chunks():
    """Prompt chunks of a row count that is no canvas length (96 rows, a 100-token prompt): admit, the next chunk,
    then the request parks for canvas rows, with and without lanes."""
    from dlmrt.models.diffusion_gemma import kernels as gk

    for fused in (False, True):
        t = stage_state()
        place(t, [(6, 96, 32)])
        t.mode[6] = ADMIT
        t.tokens.fill_(-7)
        gk.admit(t.mode, t.P, t.tokens, t.canvas, t.accepted, t.start, t.end, t.left, t.step, t.remaining, t.seed,
                 t.ordinal, t.slots, t.offsets, t.lengths, t.prompt, t.plen, W, V, BLOCKS)
        torch.cuda.synchronize()
        assert torch.equal(t.tokens[:96], t.prompt[6, :96]) and [t.start[6], t.end[6], t.mode[6]] == [0, 96, PREFILL]
        advance(t, fused)
        assert torch.equal(t.tokens[:4], t.prompt[6, 96:100]) and bool((t.tokens[4:96] == 0).all())
        assert [t.start[6], t.end[6], t.mode[6]] == [96, 192, PREFILL]
        advance(t, fused)
        assert t.mode[6] == SWITCH and t.P[6, :2].tolist() == [100, 0] and t.ctrl[2] == 1
