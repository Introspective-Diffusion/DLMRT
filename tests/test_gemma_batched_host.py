"""BatchedGemmaEngine's host loop (run, _layout, _move, _outputs, _collect) on CPU tensors against a Python model of the
device side of a launch (admit, advance, claim, control), without a GPU or a model.

Checked at every forward: each request's rows are contiguous, disjoint, inside the head bucket and aligned to the
launch's alignment; the row maps match the layout and rows past the batch map to the idle slot; the rows a request
carries between forwards (marked with its slot) followed it through every move; the batch stays within max_rows.
With commit lanes: they start at the head bucket and the launch holds them; lane commits claimed in a launch's last
forward reach the next launch's lanes with their tokens and row maps; the launch's rows are one of the head's
prebuilt totals unless pending lane commits need more. With prompt chunks: a request's first rows are min(prefill, its
prompt rounded up) when that exceeds its canvas length. Every request's canvases cover its tokens, and two runs with
the same seed are identical.
"""

import contextlib
import random
import types

import numpy as np
import pytest
import torch

batched = pytest.importorskip("dlmrt.models.diffusion_gemma.batched")

DENOISE, COMMIT, PREFILL, DONE, IDLE, ADMIT, SWITCH = range(7)
EOS = 1


class Device:
    """The device side of one launch (its replay): head, then forwards until a request finishes or parks. Canvases
    stop after 1-5 rounds; a stopped canvas ends its request with probability 0.15 (an EOS) or at its budget."""

    def __init__(self, e, rng):
        self.e, self.rng = e, rng
        self.rounds, self.pos, self.emitted = {}, {}, {}
        self.forwards = self.lane_commits = self.chunks = 0

    @staticmethod
    def marker(s):
        return 1000 + s

    def check(self, bucket, head):
        e = self.e
        if e.fused:
            lanes, size, first, claimed = (int(v) for v in e.lane[:4])
            assert first == head and head + lanes * size <= bucket and claimed <= lanes
            for j in range(claimed):
                s, n = int(e.lane[16 + 2 * j]), int(e.lane[17 + 2 * j])
                o = first + j * size
                assert (e.row_slot[o:o + n] == s).all() and (e.row_index[o:o + n] == torch.arange(n) - n).all()
                assert (e.tokens[o:o + n] == self.marker(s) + 100000).all(), f"slot {s}: lane commit lost"
                assert (e.row_slot[o + n:o + size] == e.idle).all()
            assert (e.row_slot[first + claimed * size:bucket] == e.idle).all()
        seen = torch.zeros(head, dtype=torch.bool)
        used = 0
        for p in range(e._positions(head)):
            s = int(e.slots[p])
            if s == e.idle:
                continue
            o, n = int(e.offset[s]), int(e.length[s])
            assert n in e.blocks or (int(e.mode[s]) == PREFILL and n % e.b0 == 0), (s, n)
            assert o % e.b0 == 0 and o + n <= head and not seen[o:o + n].any(), (s, o, n, head)
            seen[o:o + n] = True
            used += n
            assert (e.row_slot[o:o + n] == s).all() and (e.row_index[o:o + n] == torch.arange(n)).all()
            if int(e.mode[s]) in (DENOISE, COMMIT, PREFILL):
                assert (e.tokens[o:o + n] == self.marker(s)).all(), f"slot {s}: carried rows lost"
        assert used <= e.max_rows
        assert (e.row_slot[:head][~seen] == e.idle).all()

    def replay(self, bucket, head):
        e = self.e
        if e.fused:
            assert bucket in e.totals[head] or int(e.lane[3]) > (max(e.totals[head]) - head) // max(1, int(e.lane[1]))
        else:
            assert bucket == head
        slots = [int(s) for s in e.slots[:e._positions(head)] if int(s) != e.idle]
        for s in slots:  # admit
            o, n = int(e.offset[s]), int(e.length[s])
            if int(e.mode[s]) == ADMIT:
                e.mode[s] = PREFILL
                self.pos[s], self.emitted[s] = 0, 0
                self.chunks += n not in e.blocks
                e.tokens[o:o + n] = self.marker(s)
            elif int(e.mode[s]) == SWITCH:
                self.begin(s, n)
        finished = 0
        self.control(slots)
        while finished == 0 and any(int(e.mode[s]) in (DENOISE, COMMIT, PREFILL) for s in slots):
            self.check(bucket, head)
            self.forwards += 1
            e.ctrl[1] += 1
            if e.fused:  # release_lanes
                lanes, size, first = (int(v) for v in e.lane[:3])
                e.row_slot[first:first + lanes * size] = e.idle
                e.lane[3] = 0
            waiting = []
            for s in slots:  # advance
                n, mode = int(e.length[s]), int(e.mode[s])
                if mode == PREFILL and n not in e.blocks:  # a prompt chunk
                    self.pos[s] += n
                    if self.pos[s] >= int(e.prompt_len[s]):
                        e.params[s, 0], e.params[s, 1] = int(e.prompt_len[s]), 0
                        e.mode[s] = SWITCH
                        finished += 1
                elif mode == DENOISE:
                    self.rounds[s] -= 1
                    if self.rounds[s] == 0:
                        k = int(e.params[s, 1])
                        e.block_out[s, k], e.step_out[s, k], e.clock_out[s, k] = n, 3, int(e.counts[16])
                        e.token_out[s, self.emitted[s]:self.emitted[s] + n] = 7
                        self.emitted[s] += n
                        budget = self.emitted[s] >= int(e.max_new[s])
                        if budget or self.rng.random() < 0.15:
                            if not budget:
                                e.token_out[s, self.emitted[s] - 1] = EOS
                            e.mode[s], e.canvases[s] = DONE, k + 1
                            finished += 1
                        elif e.fused and int(e.want[s]) == n and int(e.lane[0]) > 0 and int(e.lane[1]) >= n:
                            waiting.append(s)
                        else:
                            e.mode[s] = COMMIT
                elif mode == COMMIT:
                    e.params[s, 1] += 1
                    finished += self.next_canvas(s, n)
                elif mode == PREFILL:
                    self.pos[s] += n
                    if self.pos[s] >= int(e.prompt_len[s]):
                        e.params[s, 1] = 0
                        finished += self.next_canvas(s, n)
            for j, s in enumerate(waiting):  # claim, in position order
                n = int(e.length[s])
                e.lane[8 + e.blocks.index(n)] += 1
                if j < int(e.lane[0]):
                    first = int(e.lane[2]) + j * int(e.lane[1])
                    e.tokens[first:first + n] = self.marker(s) + 100000
                    e.row_slot[first:first + n], e.row_index[first:first + n] = s, torch.arange(n) - n
                    e.lane[16 + 2 * j], e.lane[17 + 2 * j] = s, n
                    e.lane[3] += 1
                    e.lane[4] += 1
                    e.lane[5] += n
                    e.params[s, 1] += 1
                    self.lane_commits += 1
                    self.begin(s, n)
                else:
                    e.mode[s] = COMMIT
            e.counts[16] += 1
            e.ctrl[6] = e.counts[16]
            if finished == 0:
                self.control(slots)
        e.ctrl[2] = finished

    def control(self, slots):
        """kernels._control's lane counters for the next forward."""
        e = self.e
        if not e.fused:
            return
        lanes, size = int(e.lane[0]), int(e.lane[1])
        e.lane[6] += lanes * size
        for s in slots:
            n = int(e.length[s])
            if lanes > 0 and size >= n and n in e.blocks and int(e.mode[s]) in (DENOISE, COMMIT, PREFILL):
                e.lane[12 + e.blocks.index(n)] += 1

    def next_canvas(self, s, n):
        if int(self.e.want[s]) == n:
            self.begin(s, n)
            return 0
        self.e.mode[s] = SWITCH
        return 1

    def begin(self, s, n):
        e = self.e
        e.mode[s] = DENOISE
        self.rounds[s] = self.rng.randint(1, 5)
        e.tokens[int(e.offset[s]):int(e.offset[s]) + n] = self.marker(s)


@pytest.fixture
def host(monkeypatch):
    """BatchedGemmaEngine on CPU tensors: no CUDA streams, devices or pinned memory, no stage graphs. One CPU thread:
    the simulation's tensor ops are tiny, and PyTorch's OpenMP pool makes each cost milliseconds on a loaded node."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: types.SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda t: t)

    def capture(self):
        self._reset()
        self.graphs, self.loops = {}, {}

    monkeypatch.setattr(batched.BatchedGemmaEngine, "_capture", capture)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def trial(seed, blocks, max_batch, max_rows, concurrency, count, fused=False, lane_share=None, prefill=None):
    rng = random.Random(seed)
    tiers = tuple((n, b) for n, b in ((2, 256), (11, 128), (40, 64), (batched.LIMIT, 32)) if b in blocks)
    policy = batched.LoadPolicy(tiers) if len(blocks) > 1 else None
    max_rows += prefill or 0
    _, _, buckets = batched.sizes(blocks, max_batch, max_rows, fused=fused, prefill=prefill)
    config = types.SimpleNamespace(block_size=max(blocks), hidden_size=4, eos_token_ids=(EOS,), vocab_size=1 << 18,
                                   sliding_window=1024)
    model = types.SimpleNamespace(config=config, device="cpu", group=None, slots=max_batch + 1, max_rows=buckets[-1],
                                  vocab=8, max_seq_len=8192)
    e = batched.BatchedGemmaEngine(model, None, max_batch, blocks=blocks, policy=policy, max_rows=max_rows, fused=fused,
                                   lane_share=lane_share, prefill=prefill)
    device = Device(e, rng)

    def loop(rows, head, align):
        live = [s for s in e.slots.tolist() if s != e.idle]
        assert align in e.aligns and rows % align == 0 and head % align == 0
        assert all(int(e.length[s]) % align == 0 and int(e.offset[s]) % align == 0 for s in live)
        assert rows in e.row_buckets and head in e.heads
        return types.SimpleNamespace(replay=lambda stream: device.replay(rows, head))

    e._loop = loop
    prompts = [[5] * rng.randint(20, 600) for _ in range(count)]
    budget = 700
    results = e.run(prompts, budget, concurrency=concurrency)
    for r in results:
        assert sum(r.blocks) >= len(r.token_ids) and len(r.token_ids) <= budget
        assert r.token_ids[-1] == EOS or len(r.token_ids) == budget
        assert all(b in e.blocks for b in r.blocks) and r.first_forwards >= 1
        assert r.canvas_forwards == sorted(r.canvas_forwards) and r.canvas_forwards[0] >= r.admitted_forward
    if prefill and prefill > min(blocks) and max_batch > 1:
        assert device.chunks > 0
    return [(t["n"], t["bucket"], t["head"], t["lanes"], t["mix"]) for t in e.trace], \
        [(r.token_ids, r.blocks, r.canvas_forwards) for r in results], device


PHASES = [(1, 4), (4, 16), (16, 64), (64, 256), (16, 64), (1, 4)]
CASES = {
    "adaptive-phased": ((256, 128, 32), 128, 4096, PHASES, 408),
    "four-lengths": ((256, 128, 64, 32), 64, 2048, [(64, 128), (2, 8), (40, 80)], 216),
    "adaptive-128": ((256, 128, 32), 128, 4096, 128, 300),
    "fixed-32": ((32,), 64, 2048, 64, 200),
    "fixed-64": ((64,), 128, 8192, 128, 300),
    "one-request": ((256,), 1, 256, 1, 5),
    "small-switching": ((256, 32), 8, 512, [(1, 3), (8, 40), (1, 3)], 46),
}
OPTIONS = {"plain": {}, "fused": {"fused": True}, "fused-fixed-share": {"fused": True, "lane_share": 0.1},
           "prefill-128": {"prefill": 128}, "fused-prefill-256": {"fused": True, "prefill": 256}}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("option", list(OPTIONS))
def test_host_loop(host, case, option):
    blocks, max_batch, max_rows, concurrency, count = CASES[case]
    options = OPTIONS[option]
    prefill = options.get("prefill")
    if prefill and prefill % min(blocks):
        pytest.skip("prefill is not a multiple of the shortest canvas length")
    launches, results, device = trial(0, blocks, max_batch, max_rows, concurrency, count, **options)
    assert trial(0, blocks, max_batch, max_rows, concurrency, count, **options)[:2] == (launches, results), \
        "not reproducible"
    if options.get("fused") and max_batch >= 8 and min(blocks) <= 64 and case != "small-switching":
        assert device.lane_commits > 0


def test_row_buckets():
    """One canvas length: N x B per batch bucket; several, or lanes: FlashInfer's MoE token buckets, lane rungs."""
    assert batched.row_buckets((32,), 64) == [32, 64, 128, 256, 512, 1024, 2048]
    assert batched.row_buckets((256, 128, 32), 128, 4096) == [32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048,
                                                              2560, 3072, 3584, 4096]
    max_rows, size, buckets = batched.sizes((256, 128, 32), 128, fused=True)
    # 52 lanes of 32 rows on top of 4096 request rows: the first lane rung that holds them.
    assert (max_rows, size) == (4096, 32) and buckets[-1] == 6144 and all(b % 32 == 0 for b in buckets)
    assert batched.sizes((256, 128), 128, fused=True)[1] == 0  # no canvas length <= 64: no lanes
    assert batched.sizes((32,), 4, fused=True)[1] == 0  # fewer than lane_min requests
    assert np.all(np.diff(buckets) > 0)


def test_load_policy():
    """The first tier that holds n requests; the current tier while n is within one request of its range."""
    policy = batched.LoadPolicy()
    assert [policy(n) for n in (1, 2, 3, 4, 11, 12, 13, 12, 11, 10, 3, 2, 1)] == \
        [256, 256, 256, 128, 128, 128, 32, 32, 32, 128, 128, 128, 256]
