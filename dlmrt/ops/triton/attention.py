"""Flash attention in Triton over a contiguous KV arena: the attention of DiffusionGemma's full layers (and of its
sliding layers off Hopper) and of LLaDA2 off Hopper, where FA3 does not run. Query heads of one KV head are packed
along M; causal, sliding-window, left-bounded and block-causal key ranges; split-KV with a combine pass."""

import torch
import triton
import triton.language as tl

from .common import LAUNCH, _release, _wait


@triton.jit
def _attention(Q, K, V, O, PO, PL, START, END, LEFT, SLOTS, INDEX, MODES, STATE, S, scale, HQ: tl.constexpr,
               HK: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               CAUSAL: tl.constexpr, WINDOW: tl.constexpr, HAS_LEFT: tl.constexpr, SPLITS: tl.constexpr,
               ROWS: tl.constexpr, GRID: tl.constexpr, HAS_SLOTS: tl.constexpr = False, MIXED: tl.constexpr = False,
               HAS_STATE: tl.constexpr = False):
    """Flash attention, FP32 scores times `scale`. Query heads of one KV head are packed along M; each program owns
    one DV-wide slice of the output (the QK product is recomputed per slice) and, with SPLITS > 1, one of SPLITS key
    ranges, written as a normalized FP32 partial and its log-sum-exp. Keys: [LEFT, END) (HAS_LEFT), causal up to
    the query position, and within WINDOW positions of it (causal sliding layers), or block-causal on a GRID-token
    grid (GRID > 0: a row sees every key before the end of its grid block). K/V may be FP8 (read as BF16).
    HAS_SLOTS: row r belongs to the request at slot SLOTS[r], at position START[slot] + INDEX[r] of the slot's arena
    K/V[slot] (S positions); a query tile's rows (BLOCK_M / group) belong to one request, which reads START / END /
    LEFT at its slot. MIXED (with HAS_SLOTS): per tile, causal (within WINDOW) if MODES[slot] is 1 or 2 (commit /
    prefill rows) or its rows sit before START (a commit lane: negative INDEX), else the canvas's [LEFT, END).
    HAS_STATE (with HAS_SLOTS): the slot's START / END / LEFT / MODES read at STATE[r] = (start, end, left, mode) of
    the tile's first row r instead, so the key range does not wait on the slot load."""
    _wait()
    pid_m, kv, program = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    part, split = program % (D // DV), program // (D // DV)
    group: tl.constexpr = HQ // HK
    if HAS_SLOTS:
        first_row = pid_m * (BLOCK_M // group)
        slot = tl.load(SLOTS + first_row)
        q0 = tl.load(INDEX + first_row)  # the tile's first query index within its request's rows
        at = slot  # where START / END / LEFT / MODES hold the request's values
        if HAS_STATE:
            START = STATE + first_row * 4
            END = START + 1
            LEFT = START + 2
            MODES = START + 3
            at = 0
        K += slot.to(tl.int64) * S * HK * D
        V += slot.to(tl.int64) * S * HK * D
        m = tl.arange(0, BLOCK_M)
        row = q0 + m // group
        grow = first_row + m // group
        q1 = q0 + (BLOCK_M - 1) // group
    else:
        slot = 0
        at = 0
        m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row = m // group
        grow = row
        q0 = (pid_m * BLOCK_M) // group
        q1 = (pid_m * BLOCK_M + BLOCK_M - 1) // group
    head = kv * group + m % group
    d = tl.arange(0, D)
    dv = part * DV + tl.arange(0, DV)
    q = tl.load(Q + (grow * HQ + head)[:, None].to(tl.int64) * D + d[None, :])
    start = tl.load(START + at)
    end = tl.load(END + at)
    qpos = start + row
    stop = tl.minimum(end, start + q1 + 1) if CAUSAL else end
    if GRID > 0:
        stop = tl.minimum(end, ((start + q1) // GRID + 1) * GRID)
    first = 0
    left = 0
    if HAS_LEFT:
        left = tl.load(LEFT + at)
        first = left
    if WINDOW > 0 and CAUSAL:
        first = tl.maximum(first, start + q0 - (WINDOW - 1))
    if MIXED:
        mode = tl.load(MODES + at)
        causal = (mode == 1) | (mode == 2) | (q0 < 0)
        stop = tl.where(causal, tl.minimum(end, start + q1 + 1), end)
        first = tl.where(causal, 0, first)
        if WINDOW > 0:
            first = tl.where(causal, tl.maximum(first, start + q0 - (WINDOW - 1)), first)
        # The causal kernel's row masks with per-tile bounds: a row sees keys up to qpos + above, and with WINDOW
        # above low: [left, end) for canvas rows, (qpos - WINDOW, qpos] for causal ones.
        above = tl.where(causal, 0, 1 << 30)
        if WINDOW > 0:
            low = tl.where(causal, qpos - WINDOW, left - 1)
    first = first // BLOCK_N * BLOCK_N
    lo, hi = first, stop
    if SPLITS > 1:
        chunk = tl.cdiv(tl.cdiv(stop - first, SPLITS), BLOCK_N) * BLOCK_N
        lo = first + split * chunk
        hi = tl.minimum(stop, lo + chunk)
    # Without key splits the running max starts finite: a causal row past the sliding window can see none of the
    # first key tile (its range starts at the window of the query tile's first row), and exp(-inf - -inf) is NaN.
    m_i = tl.full((BLOCK_M,), -1.0e30, tl.float32)
    if SPLITS > 1:
        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, DV), tl.float32)
    for n0 in range(lo, hi, BLOCK_N):
        kpos = n0 + tl.arange(0, BLOCK_N)
        live = kpos < end
        if HAS_LEFT and not MIXED:
            live = live & (kpos >= left)
        base = (kpos.to(tl.int64) * HK + kv) * D
        k = tl.load(K + base[:, None] + d[None, :], live[:, None], 0.0).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k)) * scale
        keep = live[None, :]
        if CAUSAL:
            keep = keep & (kpos[None, :] <= qpos[:, None])
            if WINDOW > 0:
                keep = keep & (kpos[None, :] > qpos[:, None] - WINDOW)
        if MIXED:
            keep = keep & (kpos[None, :] <= qpos[:, None] + above)
            if WINDOW > 0:
                keep = keep & (kpos[None, :] > low[:, None])
            elif HAS_LEFT:
                keep = keep & (kpos[None, :] >= tl.where(causal, 0, left))
        if GRID > 0:
            keep = keep & (kpos[None, :] < (qpos[:, None] // GRID + 1) * GRID)
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        if SPLITS > 1:  # a split's keys can all be masked for some causal rows
            m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + base[:, None] + dv[None, :], live[:, None], 0.0).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    index = (grow * HQ + head).to(tl.int64)
    if SPLITS == 1:
        tl.store(O + index[:, None] * D + dv[None, :], (acc / l_i[:, None]).to(tl.bfloat16))
    else:
        tl.store(PO + (split * ROWS + index)[:, None] * D + dv[None, :],
                 tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0))
        if part == 0:
            tl.store(PL + split * ROWS + index, tl.where(l_i > 0, m_i + tl.log(l_i), float("-inf")))
    _release()


@triton.jit
def _attention_combine(PO, PL, O, ROWS: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    _wait()
    r = tl.program_id(0)
    s = tl.arange(0, SPLITS)
    lse = tl.load(PL + s * ROWS + r)
    w = tl.exp(lse - tl.max(lse, 0))
    d = tl.arange(0, D)
    o = tl.load(PO + (s * ROWS + r)[:, None].to(tl.int64) * D + d[None, :])
    tl.store(O + r.to(tl.int64) * D + d, (tl.sum(o * w[:, None], 0) / tl.sum(w, 0)).to(tl.bfloat16))
    _release()


def attention(q, k, v, start, end, causal, out=None, *, window=0, left=None, programs=128, block_m=64, block_n=32,
              dv=256, num_warps=4, num_stages=2, scale=1.0, grid=0, slots=None, index=None, block=0, modes=None,
              state=None):
    """q [1, rows, HQ, D] over the arena k/v [1, S, HK, D] (BF16 or FP8) at [0, end); rows sit at
    [start, start + rows). `left` (a device scalar) drops keys below it; `window` > 0 limits causal rows to their
    last `window` keys; `grid` > 0 makes rows block-causal on that grid (LLaDA's prefill). `scale`: softmax scale.

    Key ranges are split so that ~`programs` programs run (one range when the query tiles already fill the GPU).
    `slots` / `index` (per row): row r at position start[slots[r]] + index[r] over the arena k/v[slots[r]], with
    start / end / left indexed by slot; each request's rows are contiguous and `block`-aligned (a multiple of `block`
    rows from a multiple of `block`). `modes` (with `slots`, `causal` False): per slot, causal rows (within `window`)
    for modes 1-2, else canvas rows over [left, end). `state` (with `slots`; int32 [rows, 4]): each row's slot's
    (start, end, left, mode), read instead of start / end / left / modes at the slot."""
    _, rows, hq, d = q.shape
    hk = k.shape[2]
    out = torch.empty_like(q) if out is None else out
    dv = min(dv, d)
    if ((block if slots is not None else rows) * hq // hk) % block_m or d % dv:
        raise ValueError(f"attention: rows x query group must tile by {block_m} and head_dim by {dv}")
    tiles = rows * hq // hk // block_m
    splits = triton.next_power_of_2(max(1, programs // (tiles * hk * (d // dv))))
    po = pl = out
    if splits > 1:
        po = torch.empty(splits, rows * hq, d, device=q.device, dtype=torch.float32)
        pl = torch.empty(splits, rows * hq, device=q.device, dtype=torch.float32)
    _attention[(tiles, hk, d // dv * splits)](q, k, v, out, po, pl, start, end, start if left is None else left,
                                              slots if slots is not None else start,
                                              index if index is not None else start,
                                              modes if modes is not None else start,
                                              state if state is not None else start, k.shape[1], scale, hq, hk, d,
                                              dv, block_m, block_n, causal, window, left is not None, splits,
                                              rows * hq, grid, slots is not None, modes is not None,
                                              state is not None, num_warps=num_warps, num_stages=num_stages,
                                              **LAUNCH)
    if splits > 1:
        _attention_combine[(rows * hq,)](po, pl, out, rows * hq, d, splits, num_warps=4, **LAUNCH)
    return out
