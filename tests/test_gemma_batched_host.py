"""BatchedGemmaEngine's host loop (run, _layout, _move, _outputs, _collect) on CPU tensors against a Python model of the
device side of a launch (admit, advance, control), without a GPU or a model.

Checked at every forward: each request's rows are contiguous, disjoint, inside the launch's bucket and aligned to the
launch's alignment; the row maps match the layout and rows past the batch map to the idle slot; the rows a request
carries between forwards (marked with its slot) followed it through every move; the batch stays within max_rows.
Every request's canvases cover its tokens, and two runs with the same seed are identical.
"""

import contextlib
import random
import types

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
        self.forwards = 0

    @staticmethod
    def marker(s):
        return 1000 + s

    def check(self, bucket):
        e = self.e
        seen = torch.zeros(bucket, dtype=torch.bool)
        used = 0
        for p in range(e._positions(bucket)):
            s = int(e.slots[p])
            if s == e.idle:
                continue
            o, n = int(e.offset[s]), int(e.length[s])
            assert n in e.blocks and o % e.b0 == 0 and o + n <= bucket and not seen[o:o + n].any(), (s, o, n, bucket)
            seen[o:o + n] = True
            used += n
            assert (e.row_slot[o:o + n] == s).all() and (e.row_index[o:o + n] == torch.arange(n)).all()
            if int(e.mode[s]) in (DENOISE, COMMIT, PREFILL):
                assert (e.tokens[o:o + n] == self.marker(s)).all(), f"slot {s}: carried rows lost"
        assert used <= e.max_rows
        assert (e.row_slot[:bucket][~seen] == e.idle).all()

    def replay(self, bucket):
        e = self.e
        slots = [int(s) for s in e.slots[:e._positions(bucket)] if int(s) != e.idle]
        for s in slots:  # admit
            o, n = int(e.offset[s]), int(e.length[s])
            if int(e.mode[s]) == ADMIT:
                e.mode[s] = PREFILL
                self.pos[s], self.emitted[s] = 0, 0
                e.tokens[o:o + n] = self.marker(s)
            elif int(e.mode[s]) == SWITCH:
                self.begin(s, n)
        finished = 0
        while finished == 0 and any(int(e.mode[s]) in (DENOISE, COMMIT, PREFILL) for s in slots):
            self.check(bucket)
            self.forwards += 1
            e.ctrl[1] += 1
            for s in slots:  # advance
                n, mode = int(e.length[s]), int(e.mode[s])
                if mode == DENOISE:
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
            e.counts[16] += 1
            e.ctrl[6] = e.counts[16]
        e.ctrl[2] = finished

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


def trial(seed, blocks, max_batch, max_rows, concurrency, count):
    rng = random.Random(seed)
    tiers = tuple((n, b) for n, b in ((2, 256), (11, 128), (40, 64), (batched.LIMIT, 32)) if b in blocks)
    policy = batched.LoadPolicy(tiers) if len(blocks) > 1 else None
    _, buckets = batched.sizes(blocks, max_batch, max_rows)
    config = types.SimpleNamespace(block_size=max(blocks), hidden_size=4, eos_token_ids=(EOS,), vocab_size=1 << 18,
                                   sliding_window=1024)
    model = types.SimpleNamespace(config=config, device="cpu", group=None, slots=max_batch + 1, max_rows=buckets[-1],
                                  vocab=8, max_seq_len=8192)
    e = batched.BatchedGemmaEngine(model, None, max_batch, blocks=blocks, policy=policy, max_rows=max_rows)
    device = Device(e, rng)

    def loop(rows, align):
        live = [s for s in e.slots.tolist() if s != e.idle]
        assert align in e.aligns and rows % align == 0 and rows in e.row_buckets
        assert all(int(e.length[s]) % align == 0 and int(e.offset[s]) % align == 0 for s in live)
        return types.SimpleNamespace(replay=lambda stream: device.replay(rows))

    e._loop = loop
    prompts = [[5] * rng.randint(20, 600) for _ in range(count)]
    budget = 700
    results = e.run(prompts, budget, concurrency=concurrency)
    for r in results:
        assert sum(r.blocks) >= len(r.token_ids) and len(r.token_ids) <= budget
        assert r.token_ids[-1] == EOS or len(r.token_ids) == budget
        assert all(b in e.blocks for b in r.blocks) and r.first_forwards >= 1
        assert r.canvas_forwards == sorted(r.canvas_forwards) and r.canvas_forwards[0] >= r.admitted_forward
    return [(t["n"], t["bucket"], t["mix"]) for t in e.trace], [(r.token_ids, r.blocks, r.canvas_forwards) for r in results]


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


@pytest.mark.parametrize("case", list(CASES))
def test_host_loop(host, case):
    launches, results = trial(0, *CASES[case])
    assert trial(0, *CASES[case]) == (launches, results), "not reproducible"


def test_row_buckets():
    """One canvas length: N x B per batch bucket; several: FlashInfer's MoE token buckets."""
    assert batched.row_buckets((32,), 64) == [32, 64, 128, 256, 512, 1024, 2048]
    assert batched.row_buckets((256, 128, 32), 128, 4096) == [32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048,
                                                              2560, 3072, 3584, 4096]
    assert batched.sizes((256, 128, 32), 128) == (4096, batched.row_buckets((256, 128, 32), 128, 4096))
    assert batched.sizes((256,), 1)[0] == 256


def test_load_policy():
    """The first tier that holds n requests; the current tier while n is within one request of its range."""
    policy = batched.LoadPolicy()
    assert [policy(n) for n in (1, 2, 3, 4, 11, 12, 13, 12, 11, 10, 3, 2, 1)] == \
        [256, 256, 256, 128, 128, 128, 32, 32, 32, 128, 128, 128, 256]
