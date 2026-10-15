"""LLaDA2 requests, one at a time. The host launches one graph per block; the rounds run on the device.

    block graph = begin -> WHILE(carried block not settled){pair round} -> WHILE(block not done){round} -> snapshot
    round       = SC(previous logits) -> block forward (sees [0, block end)) -> LM head -> stats -> decide
    pair round  = forward over [carried block | this block] -> the carried block's refine round, which, once it
                  makes no edit, doubles as this block's first denoise round (the reference's commit_mode=fused_next)
    prefill     = the whole prompt blocks, up to max_rows per block-causal forward (one graph per chunk size); the
                  prompt tail rides in the first decode block as fixed rows

Blocks sit on the absolute 32-token grid; every forward writes the K/V of the rows it runs. A block's committed K/V
come from the last forward over its final tokens: its settling refine round, run as the first half of the next
block's first forward. A block holding an EOS when completed ends the request and is refined in place.
`fused=False` refines every block in place instead (one more forward per block, same tokens up to BF16 noise).
"""

import time
from dataclasses import dataclass

import torch

from ... import envs
from ...ops.extensions import block_graph
from ...ops.triton import core as kernels
from ...ops.triton import decoding
from ...runtime.generation import Generation, check_prompt
from . import kernels as lk
from .config import LladaDecode

STEP, REMAINING, DENOISE, REFINE, TERMINAL, CARRY, RESOLVED, FORWARDS, CARRY_T = (
    s.value for s in (lk.STEP, lk.REMAINING, lk.DENOISE, lk.REFINE, lk.TERMINAL, lk.CARRY, lk.RESOLVED, lk.FORWARDS,
                      lk.CARRY_T))


@dataclass
class LladaGeneration(Generation):
    refine_steps: list[int]
    forwards: list[int]

    @property
    def commit_forwards(self):
        return 0


class LladaEngine:
    # FA3 KV splits by block end: one block is only (KV heads per rank) x 1 query tile of CTAs, 4 on one GPU and 1
    # under TP=8. DLMRT_LLADA_SPLITS=N fixes one count (a different attention summation order: a numeric resample).
    SPLITS = {1: ((512, 1), (4096, 8), (None, 32)), "tp": ((1024, 16), (None, 64))}

    def __init__(self, model, decoder=None, *, fused=None, trace=False):
        """`trace`: also capture block graphs that record every round's input block, for `generate(trace=True)`."""
        self.model = m = model
        self.decoder = d = decoder or LladaDecode()
        self.fused = (d.allow_accepted_change and d.refine_steps > 0) if fused is None else fused
        if self.fused and not (d.allow_accepted_change and d.refine_steps > 0):
            raise ValueError("Fused rounds need refine rounds (allow_accepted_change, refine_steps > 0)")
        self.trace = trace
        if envs.LLADA_SPLITS() is not None:
            self.split_table = ((None, envs.LLADA_SPLITS()),)
        else:
            self.split_table = self.SPLITS[1 if m.group is None else "tp"]
        c = m.config
        self.block = b = c.block_size
        self.limit = b + d.refine_steps + 1
        dev = m.device
        i32 = {"device": dev, "dtype": torch.int32}
        # params = [block offset, prompt-tail rows in the block, first block of the request]
        self.params_host = torch.zeros(3, dtype=torch.int32, pin_memory=True)
        self.params = torch.zeros(3, **i32)
        self.prompt_ids = torch.zeros(b, device=dev, dtype=torch.int64)
        # Two-block buffers: rows [0, b) the carried block, rows [b, 2b) the current block.
        self.pair_tokens = torch.zeros(2 * b, device=dev, dtype=torch.int64)
        self.pair_prompt = torch.zeros(2 * b, device=dev, dtype=torch.bool)
        self.pair_logits = torch.zeros(2 * b, m.vocab, device=dev, dtype=torch.bfloat16)
        self.pair_stats = torch.zeros(2 * b, 3, device=dev, dtype=torch.float32)
        self.pair_proposal = torch.zeros(2 * b, device=dev, dtype=torch.int64)
        self.pair_h = torch.zeros(2 * b, c.hidden_size, device=dev, dtype=torch.bfloat16)
        self.tokens, self.prompt = self.pair_tokens[b:], self.pair_prompt[b:]
        self.logits, self.stats, self.proposal = self.pair_logits[b:], self.pair_stats[b:], self.pair_proposal[b:]
        self.h = self.pair_h[b:]
        self.accepted = torch.zeros(b, device=dev, dtype=torch.bool)
        self.final = torch.zeros(b, device=dev, dtype=torch.int64)
        self.start, self.end, self.pair_start = (torch.zeros(1, **i32) for _ in range(3))
        self.pair_ends, self.pair_batch = torch.zeros(2, **i32), torch.zeros(2, **i32)
        self.one = torch.ones(1, **i32)
        self.zero = torch.zeros(1, **i32)
        self.ctr = torch.zeros(16, **i32)
        self.snapshot = torch.zeros(16, **i32)
        if m.sc is not None:
            self.probs = torch.zeros(b, m.vocab, device=dev, dtype=torch.bfloat16)
        self.out_tokens = torch.zeros(2 * b, dtype=torch.int64, pin_memory=True)
        self.out_ctr = torch.zeros(16, dtype=torch.int32, pin_memory=True)
        self.frames = torch.zeros(self.limit, b, device=dev, dtype=torch.int64) if trace else None
        rows = m.max_rows // b
        # FA3 splits of a prefill chunk: one CTA per (block, KV head) without splitting.
        self.prefill_splits = 4 if m.group is None else 16
        self.prefill_start = torch.zeros(1, **i32)
        self.prefill_ends = torch.zeros(rows, **i32)
        self.prefill_batch = torch.zeros(rows, **i32)
        self.prefill_ids = torch.zeros(m.max_rows, device=dev, dtype=torch.int64)
        self.prefill_h = torch.zeros(m.max_rows, c.hidden_size, device=dev, dtype=torch.bfloat16)
        self.done = torch.cuda.Event()
        with torch.cuda.device(dev):
            self._capture()

    def splits(self, end):
        return next(s for limit, s in self.split_table if limit is None or end <= limit)

    def _embed(self, ids, out, delta=None):
        c = self.model.config
        kernels.embed_norm(ids, self.model.embedding, out, c.rms_norm_eps, delta=delta,
                           delta_scale=c.self_conditioning_residual_scale * c.self_conditioning_reparam_out_scale,
                           norm=c.self_conditioning_post_norm, out_scale=c.self_conditioning_post_norm_scale)

    def _begin(self):
        lk.begin(self.params, self.prompt_ids, self.tokens, self.prompt, self.accepted, self.start, self.end,
                 self.pair_start, self.pair_ends, self.ctr, self.model.config.mask_token_id)

    def _round(self, splits):
        m, b = self.model, self.block
        step = self.ctr[STEP:STEP + 1]
        delta = None
        if m.sc is not None:
            decoding.sc_probs(self.logits, self.stats, self.prompt, step, self.probs)
            delta = m.self_condition(self.probs)
        self._embed(self.tokens, self.h, delta)
        x = m.forward(self.h, self.start, self.end, splits=splits)
        m.logits(x, out=self.logits)
        decoding.row_stats(self.logits, step, self.decoder.confidence, self.stats, self.proposal, offset=m.vocab_offset,
                          group=m.group)
        lk.decide(self.stats, self.proposal, self.tokens, self.prompt, self.accepted, self.pair_tokens[:b],
                  self.pair_prompt[:b], self.ctr, self.decoder, m.config.eos_token_id, self.fused)

    def _pair_round(self, splits):
        """The carried block (self-conditioned on its previous forward) and this block's first round (none)."""
        m, b = self.model, self.block
        delta = None
        if m.sc is not None:
            decoding.sc_probs(self.pair_logits[:b], self.pair_stats[:b], self.pair_prompt[:b], self.one, self.probs)
            delta = m.self_condition(self.probs, chain=2 * b)
        self._embed(self.pair_tokens[:b], self.pair_h[:b], delta)
        self._embed(self.tokens, self.h)
        x = m.forward(self.pair_h, self.pair_start, None, splits=splits, seqlens=self.pair_ends,
                      batch_idx=self.pair_batch)
        m.logits(x, out=self.pair_logits)
        conf = self.decoder.confidence
        if self.decoder.anneals:
            # Each half at its own round index: the carried block's (it keeps counting), this block's first (zero).
            decoding.row_stats(self.pair_logits[:b], self.ctr[CARRY_T:CARRY_T + 1], conf, self.pair_stats[:b],
                               self.pair_proposal[:b], offset=m.vocab_offset, group=m.group)
            decoding.row_stats(self.pair_logits[b:], self.zero, conf, self.pair_stats[b:], self.pair_proposal[b:],
                               offset=m.vocab_offset, group=m.group)
        else:
            decoding.row_stats(self.pair_logits, self.one, conf, self.pair_stats, self.pair_proposal,
                               offset=m.vocab_offset, group=m.group)
        lk.decide_pair(self.pair_stats, self.pair_proposal, self.pair_tokens, self.pair_prompt, self.accepted,
                       self.final, self.ctr, self.decoder, m.config.eos_token_id)
        if m.sc is not None:
            lk.carry_save(self.pair_logits, self.pair_stats, self.ctr)

    def _traced(self, fn, splits):
        kernels.trace(self.tokens, self.ctr[STEP:STEP + 1], self.frames)
        fn(splits)

    def _snapshot(self):
        if self.fused and self.model.sc is not None:
            lk.carry_save(self.pair_logits, self.pair_stats, self.ctr)
        self.snapshot.copy_(self.ctr)

    def _capture(self):
        self.graphs, self.blocks, self.traced = {}, {}, {}
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.params.copy_(torch.tensor([self.block, 0, 1], dtype=torch.int32))
        split_counts = sorted({s for _, s in self.split_table})
        with torch.cuda.stream(stream):
            for splits in split_counts:
                stages = {"begin": self._begin, "round": lambda: self._round(splits), "snapshot": self._snapshot}
                if self.fused:
                    stages["pair"] = lambda: self._pair_round(splits)
                if self.trace:
                    stages["traced_round"] = lambda: self._traced(self._round, splits)
                    if self.fused:
                        stages["traced_pair"] = lambda: self._traced(self._pair_round, splits)
                for name, fn in stages.items():
                    fn()
                    fn()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, pool=pool, stream=stream):
                        fn()
                    self.graphs[splits, name] = graph
            self.prefill_graphs = {}
            self.prefill_start.fill_(0)
            for blocks in range(1, self.model.max_rows // self.block + 1):
                torch.arange(self.block, (blocks + 1) * self.block, self.block, out=self.prefill_ends[:blocks])
                self._prefill_chunk(blocks)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._prefill_chunk(blocks)
                self.prefill_graphs[blocks] = graph
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        def ptr(slot):
            return self.ctr[slot:].data_ptr()

        def loop_graph(splits, prefix):
            raw = {name: graph.raw_cuda_graph() for (s, name), graph in self.graphs.items() if s == splits}
            loops = []
            if self.fused:
                loops.append((raw[prefix + "pair"], ptr(lk.CARRY_LOOP.value), ptr(lk.CARRY_STEP.value),
                              self.decoder.refine_steps + 2))
            loops.append((raw[prefix + "round"], ptr(REMAINING), ptr(STEP), self.limit))
            return block_graph().LoopGraph(raw["begin"], loops, raw["snapshot"])

        for splits in split_counts:
            self.blocks[splits] = loop_graph(splits, "")
            if self.trace:
                self.traced[splits] = loop_graph(splits, "traced_")

    def _prefill_chunk(self, blocks):
        """`blocks` whole prompt blocks from prefill_ids at prefill_start, block-causal (ends in prefill_ends)."""
        m = self.model
        rows = blocks * self.block
        h = self.prefill_h[:rows]
        kernels.embed_norm(self.prefill_ids[:rows], m.embedding, h, m.config.rms_norm_eps, norm=False)
        m.forward(h, self.prefill_start, None, splits=self.prefill_splits, seqlens=self.prefill_ends[:blocks],
                  batch_idx=self.prefill_batch[:blocks])

    def _prefill(self, ids, prefix):
        """Whole prompt blocks [0, prefix), up to max_rows per forward, one graph per chunk size."""
        m, b = self.model, self.block
        for offset in range(0, prefix, m.max_rows):
            rows = min(m.max_rows, prefix - offset)
            self.prefill_ids[:rows].copy_(ids[offset:offset + rows])
            self.prefill_start.fill_(offset)
            torch.arange(offset + b, offset + rows + 1, b, out=self.prefill_ends[:rows // b])
            self.prefill_graphs[rows // b].replay()

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=256, *, stop_on_eos=True, on_block=None, trace=False):
        """`on_block(token_ids, denoise_steps, elapsed_s, frames)` is called once per block with its final tokens
        (keep it cheap: it is timed); a carried block is reported after the next block settles it, or at the end.
        With `trace` (an engine built with trace=True), `frames` lists every round's input block over the emitted
        positions, `<|mask|>` included; otherwise it is None. The emitted tokens are the same either way.

        A request stops at a block that held an EOS when its last position was accepted and still holds one after
        refinement. A carried block's refinement may still edit it after it was emitted, as in the reference; a
        request that ends on its token budget keeps its last block unrefined."""
        if trace and not self.trace:
            raise ValueError("generate(trace=True) needs LladaEngine(..., trace=True)")
        graphs = self.traced if trace else self.blocks
        m = self.model
        c = m.config
        with torch.cuda.device(m.device):
            ids = check_prompt(prompt_ids, max_new_tokens, c.vocab_size)
            b = self.block
            prefix = len(ids) // b * b
            tail = len(ids) - prefix
            blocks = (tail + max_new_tokens + b - 1) // b
            if prefix + blocks * b > m.max_seq_len:
                raise ValueError("Request exceeds max_seq_len")
            stream = torch.cuda.current_stream().cuda_stream
            torch.cuda.synchronize()
            started = time.perf_counter()
            device_ids = torch.tensor(ids, device=m.device, dtype=torch.int64)
            if prefix:
                self._prefill(device_ids, prefix)
            self.prompt_ids[:tail].copy_(device_ids[prefix:])
            torch.cuda.synchronize()
            prefill_s = time.perf_counter() - started
            generated, denoise, refine, forwards = [], [], [], []
            pending = None  # (position in `generated`, prompt rows, length, denoise steps, frames) of a carried block

            def report(where, length, steps, frames):
                if on_block is not None:
                    on_block(generated[where:where + length], steps, time.perf_counter() - started, frames)

            for ordinal in range(blocks):
                offset, count = prefix + ordinal * b, tail if ordinal == 0 else 0
                self.params_host[:] = torch.tensor([offset, count, int(ordinal == 0)])
                self.params.copy_(self.params_host, non_blocking=True)
                graphs[self.splits(offset + b)].replay(stream)
                self.out_tokens[:b].copy_(self.tokens, non_blocking=True)
                self.out_tokens[b:].copy_(self.final, non_blocking=True)
                self.out_ctr.copy_(self.snapshot, non_blocking=True)
                self.done.record()
                self.done.synchronize()
                if self.out_ctr[REMAINING]:
                    raise RuntimeError("Decoder exceeded its step limit")
                if pending is not None and self.out_ctr[RESOLVED]:
                    where, rows, length = pending[:3]
                    generated[where:where + length] = self.out_tokens[b + rows:b + rows + length].tolist()
                    report(where, length, *pending[3:])
                    pending = None
                denoise.append(int(self.out_ctr[DENOISE]))
                refine.append(int(self.out_ctr[REFINE]))
                forwards.append(int(self.out_ctr[FORWARDS]))
                block = self.out_tokens[count:b].tolist()[: max_new_tokens - len(generated)]
                finished = stop_on_eos and bool(self.out_ctr[TERMINAL]) and c.eos_token_id in block
                block = block[: block.index(c.eos_token_id) + 1] if finished else block
                where = len(generated)
                generated.extend(block)
                steps = int(self.out_ctr[STEP])
                frames = self.frames[:steps, count:count + len(block)].tolist() if trace else None
                if self.fused and self.out_ctr[CARRY]:
                    pending = (where, count, len(block), denoise[-1], frames)
                else:
                    report(where, len(block), denoise[-1], frames)
                if finished or len(generated) == max_new_tokens:
                    break
            if pending is not None:
                report(pending[0], pending[2], *pending[3:])
            return LladaGeneration(generated, time.perf_counter() - started, prefill_s, denoise, refine, forwards)
