"""DiffusionGemma with several requests per forward, each at its own stage and with its own canvas length. Every
forward runs a bucket of rows in which request p owns a contiguous range of B_p rows (B_p one of the engine's canvas
lengths, at most four), and each request's rows follow its mode, kept at its KV slot:

    PREFILL      the request's next B_p prompt tokens, causal (GemmaEngine's prefill chunks)
    DENOISE      a round of the request's canvas of B_p positions (self-conditioned, bidirectional over the canvas)
    COMMIT       the stopped canvas's causal re-encode
    SWITCH       between two canvases, parked until the host gives it rows of another length
    DONE / IDLE  rows nobody reads (a finished request, padding)

    one launch = admit -> WHILE(some request runs and none has finished or parked since the launch){
                     embed + forward -> [LM head + decoder] -> advance -> [self-conditioning]}

The bracketed parts run only when some request is denoising (IF nodes): the decoder when this forward has a canvas
round, the self-conditioning (the SC delta of each row's next round, from the logits just computed) when the next
forward has one past its round 0 (whose delta is zero and not read). A forward of prefill / commit rows alone costs
what GemmaEngine's commit costs.

After each forward, `advance` (kernels.py) moves every request on by itself: a stopped canvas is recorded and
committed in the next forward unless it ends the request (an EOS within the token budget, or the last canvas:
GemmaEngine's deferred commit), and a commit or the prompt's last chunk begins the next canvas. No canvas waits for
another, and prompts are prefilled inside the shared forwards. One request alone runs GemmaEngine's forwards in its
order, so its tokens are GemmaEngine's.

Canvas lengths follow the load: before each launch the host asks the policy for the canvas length of a batch of n
requests and sets it as every request's wanted length. A request keeps its rows until its current canvas is
committed; if it then wants another length, it parks (SWITCH), which ends the launch, and the host gives it rows of
the new length (committed context is ordinary causal KV, so nothing moves but its rows). Each canvas length has its
stop rule (freeze-argmax for short canvases, the stock renoise rule for long ones).

The host steps in when a request finishes or parks: it reads the finished request's canvases, admits waiting
requests (within the row budget `max_rows`) into the freed batch positions, moves the last requests into the gaps,
lays out the rows (moving the rows a request carries between forwards when its first row changed) and relaunches the
bucket that covers them; rows past the batch map to an idle slot.
"""

import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from ...ops.extensions import block_graph
from ...runtime.generation import check_prompt
from . import kernels as gk
from .config import GemmaConfig, GemmaDecode
from .model import GemmaModel

DENOISE, COMMIT, PREFILL, DONE, IDLE, ADMIT, SWITCH = range(7)
MODES = ("denoise", "commit", "prefill", "done", "idle")
LIMIT = 1 << 30  # forwards per launch


@dataclass
class _Request:
    rid: int
    slot: int
    admitted: float
    block: int  # its rows (canvas length)
    offset: int = -1  # its first row in the last launch (-1: none yet)
    clock: int = 0  # forwards the run had done when it was admitted


@dataclass
class BatchedResult:
    rid: int
    token_ids: list
    denoise_steps: list
    elapsed_s: float  # from admission to the end of the launch in which the request finished
    prefill_s: float = 0.0  # prefill runs inside the shared forwards
    blocks: list = field(default_factory=list)  # canvas length per canvas
    admitted_forward: int = 0  # forwards the run had done at admission
    canvas_forwards: list = field(default_factory=list)  # per canvas, the run's forwards before the one that ended it

    @property
    def first_forwards(self):
        """Forwards from admission to the end of the first canvas, that forward included."""
        return self.canvas_forwards[0] - self.admitted_forward + 1 if self.canvas_forwards else 0

    @property
    def request_tps(self):
        return len(self.token_ids) / self.elapsed_s


def buckets_for(max_batch):
    out, n = [], 1
    while n < max_batch:
        out.append(n)
        n *= 2
    return out + [max_batch]


# FlashInfer's TRT-LLM MoE token buckets: row counts whose tactic the autotuner finds (in the last cache file it
# loaded); commit lanes add LANE_STEP-row rungs above 4096, whose own tactics nvfp4.keep restores.
MOE_ROWS = [1 << k for k in range(9)] + list(range(512, 2048, 256)) + list(range(2048, 4096, 512)) + \
    [1 << k for k in range(12, 17)]
LANE_STEP = 512
# Lane claims per active request per forward before a run has measured any (freeze-argmax 32-canvases), weighted as
# LANE_WEIGHT request-forwards of measurements.
LANE_PRIOR, LANE_WEIGHT = 0.25, 64
# Forward cost in rows of transformer forward (B200, ORL NVFP4 stage timings): a fixed part, and the LM head, decoder
# and SC per request row.
FORWARD_FIXED, HEAD_COST = 600, 0.7


def row_buckets(blocks, max_batch, max_rows=None, lanes=0):
    """Rows per forward the engine captures a loop for (and the expert kernels are tuned for): N x B for the batch
    buckets N at one canvas length B; at several (or with `lanes` rows of commit lanes on top, or more rows than
    max_batch x B), the MoE token buckets from the shortest length up to `max_rows` (default max_batch x the shortest
    length). Lanes add the rungs (the MoE token buckets to 4096, then every LANE_STEP rows) that are multiples of the
    shortest length, up to the first that holds `lanes` more rows, below `max_rows` too (request rows of 64-canvases
    between 4096 and 8192)."""
    b0 = min(blocks)
    top = max_rows or max_batch * b0
    if len(blocks) == 1 and not lanes and top <= max_batch * b0:
        return [n * b0 for n in buckets_for(max_batch)]
    out = {r for r in MOE_ROWS if b0 <= r <= top} | {top}
    if lanes:
        rungs = sorted({r for r in MOE_ROWS if r < 4096} | set(range(4096, top + lanes + LANE_STEP, LANE_STEP)))
        total = next(r for r in rungs if r >= top + lanes)
        out |= {r for r in rungs if b0 <= r <= total and r % b0 == 0}
    return sorted(out)


def lane_size(blocks, block=64):
    """Rows per commit lane: the longest canvas length <= `block` (0: none, no lanes)."""
    return max([b for b in blocks if b <= block], default=0)


def head_totals(buckets, max_rows, max_batch, b0, cap=0.4, size=0):
    """{head: totals}: the row buckets requests may fill (heads, <= max_rows) and, per head, the launch row counts
    prebuilt for it, up to the first with room for ceil(cap x n) lanes of `size` rows at its most requests n."""
    out = {}
    for head in (r for r in buckets if r <= max_rows):
        most = math.ceil(cap * min(max_batch, head // b0)) * size
        top = next((r for r in buckets if r >= head + most), buckets[-1])
        out[head] = [r for r in buckets if head <= r <= top]
    return out


def lane_rows(max_batch, b0, share=0.4, block=64):
    """Rows of commit lanes an engine holds at most: share x max_batch lanes of `block` rows."""
    return math.ceil(math.ceil(max_batch * share) * block / b0) * b0


class LoadPolicy:
    """The canvas length of the next canvases for a batch of n requests: the first tier (most requests, length) with
    n <= most; the current tier is kept while n is within `slack` requests of its range. The default tiers follow the
    measured B200 frontier (256 up to 2 requests, 128 up to 11, then 32 with the freeze-argmax rule)."""

    def __init__(self, tiers=((2, 256), (11, 128), (LIMIT, 32)), slack=1):
        self.tiers, self.slack, self.tier = tuple(tiers), slack, None

    @property
    def blocks(self):
        return tuple(b for _, b in self.tiers)

    def reset(self):
        self.tier = None

    def __call__(self, n):
        tier = next(i for i, (most, _) in enumerate(self.tiers) if n <= most)
        if self.tier is not None:
            least = self.tiers[self.tier - 1][0] + 1 if self.tier else 1
            if least - self.slack <= n <= self.tiers[self.tier][0] + self.slack:
                tier = self.tier
        self.tier = tier
        return self.tiers[tier][1]


def sizes(blocks, max_batch, max_rows=None, *, fused=False, lane_cap=0.4, lane_min=8, lane_block=64, **_):
    """(the batch's row budget, rows per commit lane (0: no lanes), the row buckets) of an engine with these options
    (see BatchedGemmaEngine; the others do not size it), from which its model is built."""
    b0 = min(blocks)
    max_rows = max_rows or max(max_batch * b0, max(blocks))
    size = lane_size(blocks, lane_block) if fused and max_batch >= lane_min else 0
    lanes = lane_rows(max_batch, b0, lane_cap, size) if size else 0
    return max_rows, size, row_buckets(blocks, max_batch, max_rows, lanes)


class BatchedGemmaEngine:
    def __init__(self, model, decoder=None, max_batch=8, *, blocks=None, stops=None, policy=None, max_rows=None,
                 fused=False, lane_share=None, lane_cap=0.4, lane_min=8, lane_block=64):
        """`blocks`: the canvas lengths requests may use (one to four powers of two; default the checkpoint's
        canvas_length), `stops`: {length: GemmaDecode stop rule} (default the decoder's for every length), `policy`:
        n -> the canvas length for a batch of n requests (default the longest; see LoadPolicy), `max_rows`: the rows a
        batch may hold, which bounds admission (default max_batch x the shortest length, at least the longest).

        `fused`: commit a stopped canvas in a commit lane, in the same forward as the next canvas's first round
        (numerics differ from GemmaEngine's: other row counts). A launch of n >= `lane_min` requests at a canvas
        length <= `lane_block` holds lanes of that length after the requests' rows, up to the row bucket of least
        expected cost for the claims expected per forward (_lane_total; at most ceil(lane_cap x n) lanes unless more
        lane commits are pending): the sum over the requests that may claim one of their canvas length's claim rate,
        the share of its active requests whose canvas stops in a forward, `lane_share` (a number or {length: share})
        or, by default, measured over the run (from LANE_PRIOR). Stopped canvases take lanes in request-position order; one
        that finds none is committed in its own rows as without `fused`. The LM head, decoder and self-conditioning
        skip the lanes. An engine that can hold no lane (max_batch < lane_min, or no canvas length <= lane_block)
        runs as without `fused`.

        `model`: a GemmaModel at TP=1 built with slots >= max_batch + 1 and row_buckets = the row buckets of `sizes`
        for these options, max_rows >= the last of them (see `load`)."""
        self.model = m = model
        self.decoder = decoder or GemmaDecode()
        c = m.config
        self.blocks = tuple(sorted(set(blocks or (c.block_size,)), reverse=True))
        if len(self.blocks) > 4 or any(b < 16 or b & (b - 1) for b in self.blocks):
            raise ValueError("one to four power-of-two canvas lengths >= 16")
        self.b0 = b0 = self.blocks[-1]
        self.stops = tuple((stops or {}).get(b, self.decoder.stop_rule) for b in self.blocks)
        if not set(self.stops) <= set(gk.STOPS):
            raise ValueError(f"stop rules must be among {tuple(gk.STOPS)}")
        # Canvas lengths whose stop rule takes the argmax only (freeze_argmax): their rows draw no sample.
        self.argmax_only = sum(b for b, s in zip(self.blocks, self.stops) if s == "freeze_argmax")
        self.policy = policy or (lambda n: self.blocks[0])
        if not set(getattr(self.policy, "blocks", ())) <= set(self.blocks):
            raise ValueError(f"the policy's canvas lengths {self.policy.blocks} are not all in {self.blocks}")
        self.max_batch = max_batch
        self.lane_min, self.lane_block, self.lane_cap = lane_min, lane_block, lane_cap
        self.max_rows, self.lane_size, self.row_buckets = sizes(
            self.blocks, max_batch, max_rows, fused=fused, lane_cap=lane_cap, lane_min=lane_min, lane_block=lane_block)
        self.fused = self.lane_size > 0
        self.lane_share = lane_share if isinstance(lane_share, dict) or lane_share is None else \
            {b: lane_share for b in self.blocks}
        if m.group is not None or m.slots < max_batch + 1 or m.max_rows < self.row_buckets[-1] or \
                self.max_rows < self.blocks[0]:
            raise ValueError("BatchedGemmaEngine needs a TP=1 model with slots >= max_batch + 1 and max_rows >= the "
                             "engine's largest row bucket, and max_rows >= the longest canvas")
        # Request rows fill a head bucket (whose rows run the LM head, decoder and SC); with lanes, a forward runs
        # one of the head's totals.
        self.totals = head_totals(self.row_buckets, self.max_rows, max_batch, b0, lane_cap, self.lane_size)
        self.heads = list(self.totals)
        # A launch's rows are aligned to the largest of these that divides all its row counts (the shortest canvas in
        # it); from 64 rows on the sliding layers' attention takes its default query tiles (a forward graph per
        # alignment).
        self.aligns = sorted({min(b, 64) for b in self.blocks})
        self.idle = m.slots - 1
        dev = m.device
        rows, n_slots = self.row_buckets[-1], m.slots
        i32 = {"device": dev, "dtype": torch.int32}
        # Rows: request p owns [offset[slot], offset[slot] + length[slot]) for slot = slots[p]; commit lanes follow
        # the head bucket. The LM head, decoder and SC buffers hold the largest head.
        self.tokens = torch.zeros(rows, device=dev, dtype=torch.int64)  # canvas / commit / prompt-chunk input
        self.canvas = torch.zeros(rows, device=dev, dtype=torch.int64)  # argmax canvas
        self.argmax = torch.zeros(self.heads[-1], device=dev, dtype=torch.int64)
        self.sample = torch.zeros(self.heads[-1], device=dev, dtype=torch.int64)
        self.logits = torch.zeros(self.heads[-1], m.vocab, device=dev, dtype=torch.bfloat16)
        self.probs = torch.zeros(self.heads[-1], m.vocab, device=dev, dtype=torch.bfloat16)
        self.stats = torch.zeros(self.heads[-1], 2, device=dev, dtype=torch.float32)
        self.h = torch.zeros(rows, c.hidden_size, device=dev, dtype=torch.bfloat16)
        self.out = torch.zeros(rows, c.hidden_size, device=dev, dtype=torch.bfloat16)  # the forward's final norm
        self.delta = torch.zeros(rows, c.hidden_size, device=dev, dtype=torch.bfloat16)  # SC delta of the next round
        self.accepted = torch.zeros(rows, device=dev, dtype=torch.int8)
        # Rows a request carries from one forward to the next (moved with it).
        self.carried = (self.tokens, self.canvas, self.accepted, self.delta)
        # Per row: the KV slot of its request and its index within the request's rows.
        self.rowmap = torch.zeros(2, rows, **i32)
        self.row_slot, self.row_index = self.rowmap
        self.row_state = torch.zeros(rows, 4, **i32)  # the row's slot's (start, end, leftpad, mode), per forward
        self.rowmap_host = torch.zeros(2, rows, dtype=torch.int32).pin_memory()
        self.pad_index = np.arange(rows, dtype=np.int32) % b0
        self.arange_host = np.arange(rows, dtype=np.int32)
        # Slot state.
        self.params = torch.zeros(n_slots, 3, **i32)  # canvas offset, canvas ordinal, seed
        self.start, self.end, self.leftpad, self.step, self.remaining, self.seed, self.ordinal = (
            torch.zeros(n_slots, **i32) for _ in range(7))
        self.temp, self.h_ref, self.t_prev = (torch.zeros(n_slots, device=dev, dtype=torch.float32) for _ in range(3))
        # WHILE control [run, forwards, finished, finish limit, denoising, conditioned, forward clock, canvas positions
        # re-noised for NaN logits], the slots' modes and, from the 16-byte boundary after them, the commit lanes'
        # state, read back in one copy after each launch. Commit lanes: [lanes, rows per lane, first lane row, lanes
        # claimed after the last forward | lane commits, their rows, lane rows held x forwards, - | per canvas length
        # class c (self.blocks order): canvases that waited for a lane (8 + c), active requests in forwards with lanes
        # for them (12 + c)], then (slot, rows) per claimed lane; the first four set by the host per launch, the rest
        # accumulated over a run.
        self.lane_at = 8 + -(-n_slots // 4) * 4
        self.readback = torch.zeros(self.lane_at + 16 + 2 * (rows // b0), **i32)
        self.ctrl, self.mode, self.lane = self.readback[:8], self.readback[8:8 + n_slots], self.readback[self.lane_at:]
        self.mode.fill_(IDLE)
        self.prompt_len, self.max_new, self.canvases = (torch.zeros(n_slots, **i32) for _ in range(3))
        # [first row, rows (0: none), wanted canvas length] per slot, set by the host for each launch.
        self.layout = torch.zeros(3, n_slots, **i32)
        self.offset, self.length, self.want = self.layout
        self.layout_host = torch.zeros(3, n_slots, dtype=torch.int32).pin_memory()
        self.prompt = torch.zeros(n_slots, m.max_seq_len, device=dev, dtype=torch.int64)
        # Admission staging per slot (a slot's row is reused only after its request finished).
        self.prompt_host = torch.zeros(n_slots, m.max_seq_len, dtype=torch.int64).pin_memory()
        self.params_host = torch.zeros(n_slots, 3, dtype=torch.int32).pin_memory()
        most = (m.max_seq_len + b0 - 1) // b0
        self.token_out = torch.zeros(n_slots, m.max_seq_len, device=dev, dtype=torch.int64)
        self.step_out, self.block_out, self.clock_out = (torch.zeros(n_slots, most, **i32) for _ in range(3))
        self.eos_ids = torch.tensor(c.eos_token_ids, dtype=torch.int64)
        self.eos = self.eos_ids.to(dev)
        # Launch state: position -> slot, WHILE control, forwards per mode (see kernels.control).
        self.slots_host = torch.full((max_batch,), self.idle, dtype=torch.int32).pin_memory()
        self.slots = self.slots_host.to(dev)
        self.ctrl_host = torch.zeros(8, dtype=torch.int32).pin_memory()
        self.counts = torch.zeros(24, **i32)
        # `wants`: per request position, a stopped canvas waiting for a commit lane (kernels._claim).
        self.lane_host = torch.zeros(4, dtype=torch.int32).pin_memory()
        self.lane_stats = [0] * 16
        self.wants = torch.zeros(max_batch, **i32)
        self.trace = []  # per launch of the last run
        # The host writes its tables through numpy views: CPU tensor ops run on PyTorch's OpenMP pool, at a cost that
        # depends on the other processes on the node.
        self.rowmap_np, self.layout_np, self.slots_np, self.lane_np, self.ctrl_np, self.prompt_np, self.params_np = (
            t.numpy() for t in (self.rowmap_host, self.layout_host, self.slots_host, self.lane_host, self.ctrl_host,
                                self.prompt_host, self.params_host))
        with torch.cuda.device(dev):
            self._capture()

    @classmethod
    def load(cls, path, decoder=None, max_batch=8, *, max_seq_len, device=None, precision=None, **options):
        """The engine over GemmaModel.load(path, max_seq_len, ...) at TP=1, built with the KV slots, rows and expert
        tactics these `options` (the constructor's) need. The KV cache holds max_batch + 1 arenas of max_seq_len
        positions."""
        opts = dict(options)
        blocks = tuple(opts.pop("blocks", None) or (GemmaConfig.load(path).block_size,))
        _, _, buckets = sizes(blocks, max_batch, **opts)
        model = GemmaModel.load(path, max_seq_len, device=device, precision=precision, slots=max_batch + 1,
                                max_rows=buckets[-1], row_buckets=buckets)
        return cls(model, decoder, max_batch, blocks=blocks, **opts)

    # --- captured stages ---

    def _positions(self, r):
        return min(self.max_batch, r // self.b0)

    def _requests(self, r):
        """Per-request kernel arguments of head bucket r: positions -> slots, and the slots' rows."""
        return dict(slots=self.slots[:self._positions(r)], offsets=self.offset, blocks=self.length,
                    classes=self.blocks)

    def _control(self, r, head):
        lanes = dict(lane=self.lane, classes=self.blocks) if self.fused else {}
        gk.control(self.mode, self.step, self.slots[:self._positions(r)], self.length, self.ctrl, self.counts, r, head,
                   **lanes)

    def _head(self, r):
        c = self.model.config
        q = self._requests(r)
        gk.admit(self.mode, self.params, self.tokens, self.canvas, self.accepted, self.start, self.end, self.leftpad,
                 self.step, self.remaining, self.seed, self.ordinal, q["slots"], self.offset, self.length, self.prompt,
                 self.prompt_len, c.sliding_window, c.vocab_size, self.blocks)
        self._control(r, head=True)

    def _forward(self, r, align):
        m = self.model
        slots, index, state = self.row_slot[:r], self.row_index[:r], self.row_state[:r]
        gk.embed(self.tokens[:r], m.embedding, m.embed_scale, self.h[:r], m.config.rms_norm_eps, decoder=True,
                 delta=self.delta[:r], modes=self.mode, slots=slots, step=self.step, index=index if self.fused else None,
                 state=(self.start, self.end, self.leftpad, state))
        m.forward(self.h[:r], self.start, self.end, self.leftpad, causal=False, splits=1, slots=slots, index=index,
                  block=align, modes=self.mode, state=state, out=self.out[:r])

    def _decode(self, r):
        m, d = self.model, self.decoder
        c = m.config
        cap = c.final_logit_softcapping
        q = self._requests(r)
        logits, stats = self.logits[:r], self.stats[:r]
        m.logits(self.out[:r], out=logits)
        state = self.row_state[:r]
        if d.adaptive:
            gk.adaptive_temperature(logits, self.step, self.accepted, self.temp, self.h_ref, self.t_prev, d, cap,
                                    rows=state, **q)
        gk.logit_stats(logits, self.step, self.seed, self.ordinal, d, stats, self.argmax[:r], self.sample[:r], cap,
                       offset=m.vocab_offset, temp=self.temp, rand4=m.blackwell, slots=self.row_slot[:r],
                       index=self.row_index[:r], rows=state, argmax_only=self.argmax_only)
        gk.decide(stats, self.argmax, self.sample, self.tokens, self.canvas, self.accepted, self.step, self.remaining,
                  self.seed, self.ordinal, d, c.vocab_size, stops=self.stops, bad=self.ctrl[7:8], **q)

    def _advance(self, r):
        c = self.model.config
        q = self._requests(r)
        if self.fused:
            gk.release_lanes(self.lane, self.row_slot, self.row_index, self.idle, self.b0, self.row_buckets[-1])
        gk.advance(self.mode, self.params, self.tokens, self.canvas, self.accepted, self.start, self.end,
                   self.leftpad, self.step, self.remaining, self.seed, self.ordinal, q["slots"], self.offset,
                   self.length, self.want, self.prompt, self.prompt_len, self.max_new, self.token_out, self.step_out,
                   self.block_out, self.canvases, self.eos, self.ctrl, c.sliding_window, c.vocab_size, self.blocks,
                   self.decoder.max_denoising_steps, clock=(self.counts, self.clock_out),
                   lanes=(self.lane, self.wants) if self.fused else None)
        if self.fused:
            gk.claim(self.mode, self.params, self.tokens, self.canvas, self.accepted, self.start, self.end,
                     self.leftpad, self.step, self.remaining, self.seed, self.ordinal, q["slots"], self.offset,
                     self.length, self.wants, self.lane, self.row_slot, self.row_index, c.sliding_window, c.vocab_size,
                     self.blocks)
        self._control(r, head=False)

    def _self_condition(self, r):
        """Every row's SC delta for its next round, from this forward's logits (a canvas that just began, at step
        0, gets none: embed skips it; the loop skips the pass when every canvas of the next forward is at step 0)."""
        m = self.model
        gk.sc_probs(self.logits[:r], self.stats[:r], self.step, self.decoder, self.probs[:r],
                    m.config.final_logit_softcapping, temp=self.temp, slots=self.row_slot[:r], modes=self.mode)
        m.self_condition(self.probs[:r], out=self.delta[:r])

    def _reset(self):
        """Every slot idle, over a valid key range (padding rows attend to [0, b0) of the idle slot)."""
        self.mode.fill_(IDLE)
        self.params.copy_(torch.tensor([0, -1, 0], dtype=torch.int32).expand_as(self.params))
        for t in (self.start, self.leftpad, self.step, self.remaining, self.ctrl, self.counts, self.tokens,
                  self.layout, self.lane, self.wants):
            t.zero_()
        self.end.fill_(self.b0)
        self.slots.fill_(self.idle)
        self.row_slot.fill_(self.idle)
        self.row_index.copy_(torch.from_numpy(self.pad_index))

    def _capture(self):
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.graphs = {}

        def capture(key, fn):
            for _ in range(2):
                fn()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                fn()
            self.graphs[key] = graph

        self._reset()
        # Per-request stages and the LM head / decoder / SC run over the head rows, the forward over all rows.
        parts = (("head", self._head), ("decode", self._decode), ("advance", self._advance),
                 ("sc", self._self_condition))
        with torch.cuda.stream(stream):
            for r in self.row_buckets:
                if r in self.totals:
                    for name, fn in parts:
                        capture((name, r), lambda fn=fn, r=r: fn(r))
                for a in self.aligns:
                    if r % a == 0:
                        capture(("forward", r, a), lambda r=r, a=a: self._forward(r, a))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self._reset()
        self.loops = {}
        for head, totals in self.totals.items():
            for rows in totals:
                for a in self.aligns:
                    if head % a == 0 and rows % a == 0:
                        self._loop(rows, head, a)

    def _loop(self, rows, head, align):
        """The launch graph of `rows` rows aligned to `align`, whose first `head` rows (the requests' own rows; the
        rest are commit lanes) run the LM head, decoder and self-conditioning (built on first use: the head's totals
        are prebuilt, others serve pending lane commits beyond them)."""
        if (rows, head, align) not in self.loops:
            ctrl = self.ctrl.data_ptr()
            denoising, conditioned = ctrl + 16, ctrl + 20
            raw = {key: g.raw_cuda_graph() for key, g in self.graphs.items()}
            self.loops[rows, head, align] = block_graph().StagedLoopGraph(
                raw[("head", head)], [(raw[("forward", rows, align)], 0), (raw[("decode", head)], denoising),
                                      (raw[("advance", head)], 0), (raw[("sc", head)], conditioned)], ctrl, ctrl + 4,
                LIMIT)
        return self.loops[rows, head, align]

    # --- host loop ---

    def span(self, max_new_tokens):
        """Positions past the prompt a request may write: its canvases until the budget is reached."""
        if len(self.blocks) == 1:
            return (max_new_tokens + self.b0 - 1) // self.b0 * self.b0
        return max_new_tokens - 1 + self.blocks[0]

    def _admit(self, ids, max_new_tokens, seed, slot):
        n = len(ids)
        self.prompt_np[slot, :n] = ids
        self.params_np[slot] = (0, 0, seed)
        self.prompt[slot, :n].copy_(self.prompt_host[slot, :n], non_blocking=True)
        self.params[slot].copy_(self.params_host[slot], non_blocking=True)
        self.prompt_len[slot] = n
        self.max_new[slot] = max_new_tokens
        self.mode[slot] = ADMIT

    def _move(self, moves, lanes=()):
        """Carried rows (src, dst, rows) to their new place, and the tokens of lane commits (src, dst, rows)."""
        def rows(ms, k):
            return torch.from_numpy(np.concatenate([np.arange(m[k], m[k] + m[2]) for m in ms])).to(self.model.device)

        every = list(moves) + list(lanes)
        self.tokens[rows(every, 1)] = self.tokens[rows(every, 0)]
        if moves:
            src, dst = rows(moves, 0), rows(moves, 1)
            for t in self.carried[1:]:
                t[dst] = t[src]

    def _bucket(self, rows):
        return next(b for b in self.heads if b >= rows)

    def _claims(self, batch, want, size):
        """Lane claims expected per forward: the claim rate of each request that may take a lane of `size` rows."""
        if self.lane_share is not None:
            rate = self.lane_share
        else:
            s = self.lane_stats
            rate = {b: (s[8 + c] + LANE_PRIOR * LANE_WEIGHT) / (s[12 + c] + LANE_WEIGHT)
                    for c, b in enumerate(self.blocks)}
        return sum(rate.get(r.block, 0.0) for r in batch if r.block <= size and want[r.slot] == r.block)

    def _lane_total(self, head, size, claims, pending, n):
        """The launch's rows for `head` rows of n requests and lanes of `size` rows: among the head's totals with room
        for the `pending` lane commits (else the first bucket that holds them), the one of least expected cost per
        forward in rows of transformer forward: its lane rows, plus a canvas slot of the forward (FORWARD_FIXED and
        HEAD_COST: the forward's cost over n) per claim past its lanes, which commits in its own rows next; claims
        Poisson with mean `claims`."""
        fits = [t for t in self.totals[head] if (t - head) // size >= pending]
        if not fits:
            return next(b for b in self.row_buckets if b >= head + pending * size)
        slot = (FORWARD_FIXED + (1 + HEAD_COST) * head) / max(1, n)
        most = (fits[-1] - head) // size
        p, below = math.exp(-claims), [0.0]  # below[k] = P(claims < k)
        for k in range(most + 1):
            below.append(below[-1] + p)
            p *= claims / (k + 1)

        def cost(t):
            k = (t - head) // size
            past = claims * (1 - below[max(k - 1, 0)]) - k * (1 - below[k])  # E[max(claims - k, 0)]
            return t - head + max(past, 0.0) * slot

        return min(fits, key=lambda t: (cost(t), t))

    def _layout(self, batch, want, n, target, claimed):
        """Rows in batch order and, with `fused`, commit lanes after them (from the head bucket on), the lane commits
        `claimed` [(slot, rows, first row)] in the previous launch's last forward first; moves the rows that changed
        place; host tables of the launch. Returns (the requests' rows, head rows, launch rows, expected claims)."""
        offset, moves = 0, []
        for r in batch:
            if 0 <= r.offset != offset:
                moves.append((r.offset, offset, r.block))
            r.offset = offset
            offset += r.block
        head = total = self._bucket(offset)
        size, lanes, claims = 0, 0, 0.0
        if self.fused:
            size = target if target <= self.lane_block and n >= self.lane_min else 0
            claims = self._claims(batch, want, size) if size else 0.0
            size = max([size] + [b for _, b, _ in claimed])
            if size:
                total = self._lane_total(head, size, claims, len(claimed), n)
                lanes = (total - head) // size
        placed = [(src, head + j * size, b) for j, (_, b, src) in enumerate(claimed)]
        shifted = [p for p in placed if p[0] != p[1]]
        if moves or shifted:
            self._move(moves, shifted)
        rm = self.rowmap_np
        rm[0, offset:total] = self.idle
        rm[1, offset:total] = self.pad_index[offset:total]
        if claimed:
            k = len(claimed)
            slot, b = np.array([(s, b) for s, b, _ in claimed], dtype=np.int32).T[:, :, None]
            i = self.arange_host[:size]
            on = i < b
            rm[0, head:head + k * size] = np.where(on, slot, self.idle).ravel()
            rm[1, head:head + k * size] = np.where(on, i - b, self.pad_index[head:head + k * size].reshape(k, size)
                                                   ).ravel()
        self.lane_np[:] = lanes, size, head, len(claimed)
        slots = np.array([r.slot for r in batch], dtype=np.int32)
        blocks = np.array([r.block for r in batch], dtype=np.int32)
        firsts = np.array([r.offset for r in batch], dtype=np.int32)
        lay = self.layout_np
        lay[:] = 0
        lay[0, slots] = firsts
        lay[1, slots] = blocks
        lay[2, slots] = [want.get(r.slot, r.block) for r in batch]
        rm[0, :offset] = np.repeat(slots, blocks)
        rm[1, :offset] = self.arange_host[:offset] - np.repeat(firsts, blocks)
        self.slots_np[:] = self.idle
        self.slots_np[:len(batch)] = slots
        return offset, head, total, claims

    def _outputs(self, slots):
        """Per slot (host arrays, one copy): emitted tokens, then per canvas its length, rounds and forward clock, then
        the canvas count."""
        idx = torch.tensor(slots, device=self.model.device)
        return torch.cat([self.token_out[idx], self.block_out[idx].long(), self.step_out[idx].long(),
                          self.clock_out[idx].long(), self.canvases[idx].long()[:, None]], 1).cpu().numpy()

    def _collect(self, r, out, max_new_tokens, eos, now):
        """The request's emitted tokens (`out`: its row of _outputs), cut as GemmaEngine cuts them (canvas by canvas,
        at the budget and EOS)."""
        k, n, most = int(out[-1]), self.token_out.shape[1], self.block_out.shape[1]
        blocks = out[n:n + k].tolist()
        tokens = out[:sum(blocks)].tolist()
        generated, at = [], 0
        for b in blocks:
            block = tokens[at:at + b][: max_new_tokens - len(generated)]
            at += b
            end = next((j for j, t in enumerate(block) if t in eos), None)
            generated.extend(block if end is None else block[: end + 1])
            if end is not None or len(generated) >= max_new_tokens:
                break
        return BatchedResult(r.rid, generated, out[n + most:n + most + k].tolist(), now - r.admitted, blocks=blocks,
                             admitted_forward=r.clock, canvas_forwards=out[n + 2 * most:n + 2 * most + k].tolist())

    @torch.inference_mode()
    def run(self, prompts, max_new_tokens, *, concurrency=None, seed=None, stop_on_eos=True, on_finish=None):
        """Generate for every prompt (lists of token ids), at most `concurrency` at a time: a number (default
        max_batch), or phases [(concurrency, requests), ...] that apply to the requests in prompt order (the last
        phase's concurrency past them). Returns BatchedResults in prompt order; `on_finish(result)` is called as each
        request finishes."""
        m = self.model
        phases = [(concurrency or self.max_batch, len(prompts))] if not isinstance(concurrency, (list, tuple)) \
            else list(concurrency)
        seed = self.decoder.seed if seed is None else seed
        eos = set(m.config.eos_token_ids) if stop_on_eos else set()
        self.eos.copy_(self.eos_ids if stop_on_eos else torch.full_like(self.eos_ids, -2))
        stream = torch.cuda.current_stream().cuda_stream
        pending = list(enumerate(prompts))[::-1]
        free = list(range(self.max_batch))[::-1]
        batch, results, taken, modes, claimed, clock = [], [None] * len(prompts), 0, None, [], 0
        self.trace = []
        started = time.perf_counter()
        self.counts.zero_()
        self.lane.zero_()
        self.lane_stats = [0] * 16
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        def limit():
            k = taken
            for level, count in phases:
                if k < count:
                    return min(level, self.max_batch)
                k -= count
            return min(phases[-1][0], self.max_batch)

        def take(block):
            nonlocal taken
            rid, ids = pending.pop()
            ids = check_prompt(ids, max_new_tokens, m.config.vocab_size)
            if max(len(ids) + self.span(max_new_tokens), -(-len(ids) // block) * block) > m.max_seq_len:
                raise ValueError("Request exceeds max_seq_len")
            slot = free.pop()
            self._admit(ids, max_new_tokens, seed, slot)
            taken += 1
            return _Request(rid, slot, time.perf_counter(), block, clock=clock)

        with torch.cuda.device(m.device):
            while pending or any(r is not None for r in batch):
                alive = sum(r is not None for r in batch)
                target = self.policy(min(max(limit(), alive), alive + len(pending)))
                used = sum(r.block for r in batch if r is not None)
                want = {}
                for r in batch:
                    if r is not None and modes is not None and modes[r.slot] == SWITCH and r.block != target:
                        if used - r.block + target <= self.max_rows:
                            used += target - r.block
                            r.block, r.offset = target, -1
                        else:
                            want[r.slot] = r.block

                def admissible():
                    return pending and alive < limit() and used + target <= self.max_rows

                for p in range(len(batch)):
                    if batch[p] is None and admissible():
                        batch[p] = take(target)
                        alive, used = alive + 1, used + batch[p].block
                p = 0
                while p < len(batch):
                    if batch[p] is None:
                        last = batch.pop()
                        if p < len(batch):
                            batch[p] = last
                        continue
                    p += 1
                while admissible():
                    batch.append(take(target))
                    alive, used = alive + 1, used + batch[-1].block
                for r in batch:
                    want.setdefault(r.slot, target)
                rows, head, bucket, claims = self._layout(batch, want, alive, target, claimed)
                self.rowmap[:, :bucket].copy_(self.rowmap_host[:, :bucket], non_blocking=True)
                self.layout.copy_(self.layout_host, non_blocking=True)
                self.slots.copy_(self.slots_host, non_blocking=True)
                lanes, size = self.lane_np[:2].tolist()
                if self.fused:
                    self.lane[:4].copy_(self.lane_host, non_blocking=True)
                self.ctrl_np[3] = 1
                self.ctrl.copy_(self.ctrl_host, non_blocking=True)
                launched = time.perf_counter()
                # Every request's rows, the lanes and the head start at a multiple of the alignment.
                align = max(a for a in self.aligns if all(
                    k % a == 0 for k in [r.block for r in batch] + [head, bucket] + ([size] if lanes else [])))
                self._loop(bucket, head, align).replay(stream)
                back = self.readback[:self.lane_at + 16 + 2 * lanes if self.fused else 8 + self.mode.numel()].tolist()
                now = time.perf_counter()
                ctrl, modes = back[:8], back[8:]
                at, clock = clock, ctrl[6]
                if self.fused:
                    lane = back[self.lane_at:]
                    self.lane_stats = lane[:16]
                    claimed = [(lane[16 + 2 * j], lane[17 + 2 * j], lane[2] + j * lane[1]) for j in range(lane[3])]
                mix = {}
                for r in batch:
                    mix[r.block] = mix.get(r.block, 0) + 1
                self.trace.append({"n": len(batch), "bucket": bucket, "head": head, "align": align, "rows": rows,
                                   "mix": mix, "target": target, "lanes": lanes, "lane_rows": size,
                                   "claims": claims, "forwards": ctrl[1], "finished": ctrl[2], "nan": ctrl[7],
                                   "graph_s": now - launched, "t0": launched - started, "clock": at})
                if ctrl[7]:
                    warnings.warn(f"launch {len(self.trace)}: {ctrl[7]} canvas positions had NaN logits (re-noised; "
                                  f"slots {[r.slot for r in batch]})", RuntimeWarning, stacklevel=3)
                done = [p for p, r in enumerate(batch) if modes[r.slot] == DONE]
                outs = self._outputs([batch[p].slot for p in done]) if done else ()
                for p, out in zip(done, outs):
                    r = batch[p]
                    res = self._collect(r, out, max_new_tokens, eos, now)
                    results[r.rid] = res
                    if on_finish is not None:
                        on_finish(res)
                    self.mode[r.slot] = IDLE
                    free.append(r.slot)
                    batch[p] = None
        return results

    def mode_shares(self):
        """Share of requests x forwards per mode over the last run (each request once, whatever its rows; padding
        appears in row_shares only); with `fused`, commits in a lane count as requests of their own ("lane")."""
        counts, names = self.counts.tolist()[:len(MODES)], MODES
        if self.fused:
            counts, names = counts + [self.lane_commits()], names + ("lane",)
        total = max(1, sum(counts))
        return {name: c / total for name, c in zip(names, counts)}

    def row_shares(self):
        """Share of rows x forwards per mode over the last run (padding as idle); with `fused`, lane rows claimed
        ("lane") and held unclaimed ("spare")."""
        counts, names = self.counts.tolist()[8:8 + len(MODES)], MODES
        if self.fused:
            held, claimed = self.lane[5:7].tolist()[::-1]
            counts, names = counts + [claimed, held - claimed], names + ("lane", "spare")
        total = max(1, sum(counts))
        return {name: c / total for name, c in zip(names, counts)}

    def lane_commits(self):
        """Canvases committed in a lane over the last run (`fused`)."""
        return int(self.lane[4])

    def first_canvas_s(self, r):
        """Seconds from the start of the launch that admitted `r` (a result of the last run) to the end of its first
        canvas, the forwards of a launch taken as equally long."""
        def end_of(f):
            for t in self.trace:
                if t["clock"] <= f < t["clock"] + max(1, t["forwards"]):
                    return t["t0"] + t["graph_s"] * (f - t["clock"] + 1) / max(1, t["forwards"])
            return math.nan

        if not r.canvas_forwards:
            return math.nan
        return end_of(r.canvas_forwards[0]) - next((t["t0"] for t in self.trace if t["clock"] >= r.admitted_forward),
                                                   math.nan)

    def nan_positions(self):
        """Canvas positions x rounds whose logits were NaN over the last run (re-noised by decide; 0 when healthy)."""
        return sum(t["nan"] for t in self.trace)
