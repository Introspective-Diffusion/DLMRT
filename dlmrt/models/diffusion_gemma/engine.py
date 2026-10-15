"""DiffusionGemma requests, one at a time. The host launches one graph per canvas; the rounds run on the device.

    block graph = [commit of the previous canvas ->] begin -> WHILE(not stopped and step < max_denoising_steps){round}
    round       = SC(previous logits) -> bidirectional canvas forward -> LM head -> stats -> accept / re-noise / stop
    commit      = causal re-encode of the argmax canvas (writes its K/V); also the prompt prefill, 256 rows at a time

A canvas is committed at the start of the next canvas's graph rather than at the end of its own, so the request's
last canvas, whose K/V nothing reads, is never committed.

Each canvas starts right after the tokens before it (no block alignment). Sliding layers see at most
window - 1 + 256 keys, so one FA3 split count serves every context.
"""

import time
from dataclasses import dataclass

import torch

from ... import envs
from ...ops.extensions import block_graph
from ...ops.triton import core as kernels
from ...runtime.generation import Generation, check_prompt
from . import kernels as gk
from .config import GemmaDecode


@dataclass
class GemmaGeneration(Generation):
    commits: int  # one fewer than canvases: the last canvas is never committed

    @property
    def commit_forwards(self):
        return self.commits


class GemmaEngine:
    def __init__(self, model, decoder=None, *, trace=False):
        """`trace`: also capture a block graph that records every round's input canvas, for `generate(trace=True)`.
        The sliding layers' FA3 KV splits: 1 on one GPU, 4 under TP, where a rank has 1-2 heads, too few CTAs to
        fill the GPU without splitting the keys; DLMRT_DG_SPLITS overrides them."""
        self.model = m = model
        self.decoder = decoder or GemmaDecode()
        self.trace = trace
        default = 1 if m.group is None else 4
        self.splits = envs.DG_SPLITS() or default
        c = m.config
        self.block = b = c.block_size
        dev = m.device
        i32 = {"device": dev, "dtype": torch.int32}
        # params = [canvas offset, canvas ordinal, request seed]
        self.params_host = torch.zeros(3, dtype=torch.int32, pin_memory=True)
        self.params = torch.zeros(3, **i32)
        self.tokens = torch.zeros(b, device=dev, dtype=torch.int64)
        # The argmax canvas: the block's output, the commit's input, and the prefill staging buffer.
        self.canvas = torch.zeros(b, device=dev, dtype=torch.int64)
        self.argmax = torch.zeros(b, device=dev, dtype=torch.int64)
        self.sample = torch.zeros(b, device=dev, dtype=torch.int64)
        self.start, self.end, self.leftpad, self.step, self.remaining, self.seed, self.ordinal = (
            torch.zeros(1, **i32) for _ in range(7))
        self.logits = torch.zeros(b, m.vocab, device=dev, dtype=torch.bfloat16)
        self.probs = torch.zeros(b, m.vocab, device=dev, dtype=torch.bfloat16)
        self.stats = torch.zeros(b, 2, device=dev, dtype=torch.float32)
        self.h = torch.zeros(b, c.hidden_size, device=dev, dtype=torch.bfloat16)
        # Adaptive temperature state: this round's T, the reference entropy, the previous T, accepted positions.
        self.temp, self.h_ref, self.t_prev = (torch.zeros(1, device=dev, dtype=torch.float32) for _ in range(3))
        self.accepted = torch.zeros(b, device=dev, dtype=torch.int8)
        self.out_tokens = torch.zeros(b, dtype=torch.int64, pin_memory=True)
        self.out_counters = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        steps = self.decoder.max_denoising_steps
        self.frames = torch.zeros(steps, b, device=dev, dtype=torch.int64) if trace else None
        self.done = torch.cuda.Event()
        with torch.cuda.device(dev):
            self._capture()

    def _begin(self):
        c = self.model.config
        gk.begin(self.params, self.tokens, self.canvas, self.accepted, self.start, self.end, self.leftpad, self.step,
                 self.remaining, self.seed, self.ordinal, c.sliding_window, c.vocab_size)

    def _round(self):
        m, d = self.model, self.decoder
        c = m.config
        cap = c.final_logit_softcapping
        gk.sc_probs(self.logits, self.stats, self.step, d, self.probs, cap, temp=self.temp)
        gk.embed(self.tokens, m.embedding, m.embed_scale, self.h, c.rms_norm_eps, decoder=True,
                 delta=m.self_condition(self.probs))
        x = m.forward(self.h, self.start, self.end, self.leftpad, causal=False, splits=self.splits)
        m.logits(x, out=self.logits)
        if d.adaptive:
            gk.adaptive_temperature(self.logits, self.step, self.accepted, self.temp, self.h_ref, self.t_prev, d, cap,
                                    group=m.group)
        gk.logit_stats(self.logits, self.step, self.seed, self.ordinal, d, self.stats, self.argmax, self.sample, cap,
                       offset=m.vocab_offset, group=m.group, temp=self.temp, rand4=m.blackwell)
        gk.decide(self.stats, self.argmax, self.sample, self.tokens, self.canvas, self.accepted, self.step,
                  self.remaining, self.seed, self.ordinal, d, c.vocab_size)

    def _traced_round(self):
        kernels.trace(self.tokens, self.step, self.frames)
        self._round()

    def _commit(self):
        m = self.model
        gk.embed(self.canvas, m.embedding, m.embed_scale, self.h, m.config.rms_norm_eps, decoder=False)
        m.forward(self.h, self.start, self.end, self.leftpad, causal=True, splits=self.splits)

    def _carry(self):
        """Commit the previous canvas (start / end still hold its range), then begin the next one."""
        self._commit()
        self._begin()

    def _capture(self):
        self.graphs = {}
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.params.copy_(torch.tensor([0, 0, self.decoder.seed], dtype=torch.int32))
        stages = {"begin": self._begin, "round": self._round, "commit": self._commit, "carry": self._carry}
        if self.trace:
            stages["traced"] = self._traced_round
        with torch.cuda.stream(stream):
            for name, fn in stages.items():
                for _ in range(2):
                    # begin leaves -1 in the argmax canvas, which a second warm-up carry would commit (look up);
                    # when serving, carry always follows a round.
                    self.canvas.clamp_(min=0)
                    fn()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    fn()
                self.graphs[name] = graph
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        begin, carry = (self.graphs[n].raw_cuda_graph() for n in ("begin", "carry"))
        limit = self.decoder.max_denoising_steps

        def block_graphs(round_):
            """(first canvas, later canvases): begin / carry -> WHILE(round), no tail."""
            loop = [(self.graphs[round_].raw_cuda_graph(), self.remaining.data_ptr(), self.step.data_ptr(), limit)]
            return block_graph().LoopGraph(begin, loop, 0), block_graph().LoopGraph(carry, loop, 0)

        self.block_graphs = block_graphs("round")
        self.traced_graphs = block_graphs("traced") if self.trace else None

    def _prefill(self, ids):
        b = self.block
        for offset in range(0, ids.numel(), b):
            rows = min(b, ids.numel() - offset)
            self.canvas.zero_()
            self.canvas[:rows].copy_(ids[offset:offset + rows])
            self.start.fill_(offset)
            self.end.fill_(offset + b)
            self.graphs["commit"].replay()

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=256, *, seed=None, stop_on_eos=True, on_block=None, trace=False):
        """`on_block(token_ids, denoise_steps, elapsed_s, frames)` is called after each canvas (keep it cheap: it is
        timed). With `trace` (an engine built with trace=True), `frames` lists every round's input canvas over the
        emitted positions, random tokens included; otherwise it is None. The emitted tokens are the same either way.
        """
        if trace and not self.trace:
            raise ValueError("generate(trace=True) needs GemmaEngine(..., trace=True)")
        graphs = self.traced_graphs if trace else self.block_graphs
        m = self.model
        with torch.cuda.device(m.device):
            ids = check_prompt(prompt_ids, max_new_tokens, m.config.vocab_size)
            b = self.block
            canvases = (max_new_tokens + b - 1) // b
            if len(ids) + canvases * b > m.max_seq_len:
                raise ValueError("Request exceeds max_seq_len")
            seed = self.decoder.seed if seed is None else seed
            eos = set(m.config.eos_token_ids)
            stream = torch.cuda.current_stream().cuda_stream
            torch.cuda.synchronize()
            started = time.perf_counter()
            self._prefill(torch.tensor(ids, device=m.device, dtype=torch.int64))
            torch.cuda.synchronize()
            prefill_s = time.perf_counter() - started
            generated, steps = [], []
            for ordinal in range(canvases):
                self.params_host[:] = torch.tensor([len(ids) + ordinal * b, ordinal, seed])
                self.params.copy_(self.params_host, non_blocking=True)
                graphs[ordinal > 0].replay(stream)
                self.out_tokens.copy_(self.canvas, non_blocking=True)
                self.out_counters[:1].copy_(self.step, non_blocking=True)
                self.done.record()
                self.done.synchronize()
                steps.append(int(self.out_counters[0]))
                block = self.out_tokens.tolist()[: max_new_tokens - len(generated)]
                end = next((i for i, t in enumerate(block) if t in eos), None) if stop_on_eos else None
                block = block if end is None else block[: end + 1]
                generated.extend(block)
                if on_block is not None:
                    frames = self.frames[:steps[-1], :len(block)].tolist() if trace else None
                    on_block(block, steps[-1], time.perf_counter() - started, frames)
                if end is not None or len(generated) == max_new_tokens:
                    break
            return GemmaGeneration(generated, time.perf_counter() - started, prefill_s, steps, len(steps) - 1)
