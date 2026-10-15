"""One request at a time. The host launches one graph per block; the rounds run on the device.

    block graph = begin -> WHILE(remaining > 0 and step < block){round} -> commit
    round       = SC(previous logits) -> bidirectional forward -> LM head -> stats -> accept/edit
    commit      = clean causal forward (writes the block's immutable K/V) -> boundary logits
"""

import time

import torch

from ...ops.extensions import block_graph
from ...ops.triton import core as kernels
from ...ops.triton import decoding
from ...runtime.generation import Generation, check_prompt
from .config import DecodeConfig
from .model import SPLIT_CONTEXT


class Engine:
    def __init__(self, model, decoder=None, *, trace=False):
        """`trace`: also capture block graphs that record every round's input block, for `generate(trace=True)`."""
        self.model = m = model
        self.decoder = decoder or DecodeConfig()
        self.trace = trace
        c = m.config
        if self.decoder.random_token_high > c.vocab_size:
            raise ValueError("Noise vocabulary exceeds the checkpoint vocabulary")
        self.block = b = c.block_size
        dev = m.device
        i32 = {"device": dev, "dtype": torch.int32}
        # params = [block offset, prompt rows in block, noise counter origin, boundary row]
        self.params_host = torch.zeros(4, dtype=torch.int32, pin_memory=True)
        self.params = torch.zeros(4, **i32)
        self.prompt_ids = torch.zeros(b, device=dev, dtype=torch.int64)
        self.tokens = torch.zeros(b, device=dev, dtype=torch.int64)
        self.prompt = torch.zeros(b, device=dev, dtype=torch.bool)
        self.accepted = torch.zeros_like(self.prompt)
        self.start, self.end, self.noise_start, self.step, self.remaining = (torch.zeros(1, **i32) for _ in range(5))
        self.last = torch.zeros(1, device=dev, dtype=torch.int64)
        # Row b holds the clean boundary distribution used by shifted decisions. Logits and SC
        # probabilities cover this rank's vocabulary shard; stats and proposals are global.
        self.logits = torch.zeros(b + 1, m.vocab, device=dev, dtype=torch.bfloat16)
        self.stats = torch.zeros(b + 1, 3, device=dev, dtype=torch.float32)
        self.proposal = torch.zeros(b + 1, device=dev, dtype=torch.int64)
        self.h = torch.zeros(b, c.hidden_size, device=dev, dtype=torch.bfloat16)
        if m.sc is not None:
            self.probs = torch.zeros(b, m.vocab, device=dev, dtype=torch.bfloat16)
        self.out_tokens = torch.zeros(b, dtype=torch.int64, pin_memory=True)
        self.out_counters = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        # Round r's input block lands in frames[r]; a block runs at most b rounds.
        self.frames = torch.zeros(b, b, device=dev, dtype=torch.int64) if trace else None
        self.done = torch.cuda.Event()
        with torch.cuda.device(dev):
            self._capture()

    def _begin(self):
        decoding.begin_block(self.params, self.prompt_ids, self.tokens, self.prompt, self.accepted, self.start,
                            self.end, self.noise_start, self.step, self.remaining, self.last, self.decoder)

    def _embed(self, delta=None):
        c = self.model.config
        kernels.embed_norm(self.tokens, self.model.embedding, self.h, c.rms_norm_eps, delta=delta,
                           delta_scale=c.self_conditioning_residual_scale * c.self_conditioning_reparam_out_scale,
                           norm=c.self_conditioning_post_norm, out_scale=c.self_conditioning_post_norm_scale)

    def _round(self, splits):
        m, b = self.model, self.block
        delta = None
        if m.sc is not None:
            decoding.sc_probs(self.logits[:b], self.stats[:b], self.prompt, self.step, self.probs)
            delta = m.self_condition(self.probs)
        self._embed(delta)
        x = m.forward(self.h, self.start, self.end, causal=False, splits=splits)
        torch.mm(x, m.lm_head.t(), out=self.logits[:b])
        decoding.row_stats(self.logits, self.step, self.decoder, self.stats, self.proposal,
                          offset=m.vocab_offset, group=m.group)
        decoding.decide(self.stats, self.proposal, self.tokens, self.accepted, self.prompt, self.start,
                       self.noise_start, self.step, self.remaining, self.decoder)

    def _traced_round(self, splits):
        kernels.trace(self.tokens, self.step, self.frames)
        self._round(splits)

    def _commit(self, splits):
        m, b = self.model, self.block
        self._embed()
        x = m.forward(self.h, self.start, self.end, causal=True, splits=splits)
        if self.decoder.logits_shift:
            torch.mm(x.index_select(0, self.last), m.lm_head.t(), out=self.logits[b:])

    def _capture(self):
        b = self.block
        self.graphs, self.blocks, self.traced = {}, {}, {}
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.params.copy_(torch.tensor([0, 0, 0, b - 1], dtype=torch.int32))
        with torch.cuda.stream(stream):
            for splits in self.split_counts:
                stages = {"begin": self._begin, "round": lambda: self._round(splits),
                          "commit": lambda: self._commit(splits)}
                if self.trace:
                    stages["traced"] = lambda: self._traced_round(splits)
                for name, fn in stages.items():
                    fn()
                    fn()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, pool=pool, stream=stream):
                        fn()
                    self.graphs[splits, name] = graph
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for splits in self.split_counts:
            begin, round_, commit = (self.graphs[splits, n].raw_cuda_graph() for n in ("begin", "round", "commit"))
            self.blocks[splits] = block_graph().BlockGraph(begin, round_, commit, self.remaining.data_ptr(),
                                                      self.step.data_ptr(), b)
            if self.trace:
                self.traced[splits] = block_graph().BlockGraph(begin, self.graphs[splits, "traced"].raw_cuda_graph(),
                                                          commit, self.remaining.data_ptr(), self.step.data_ptr(), b)

    @property
    def split_counts(self):
        world = self.model.group.world if self.model.group else 1
        return sorted({1, max(2, world), 2 * world})

    def splits(self, end):
        """FA3 KV splits: 2 x ranks fill the GPU at long context (each rank has 1/world of the heads)."""
        world = self.model.group.world if self.model.group else 1
        if end <= SPLIT_CONTEXT:
            return 1
        return max(2, world) if end <= 4096 else 2 * world

    def _prefill(self, ids, prefix):
        b = self.block
        for offset in range(0, prefix, b):
            rows = min(b, prefix - offset)
            self.tokens.zero_()
            self.tokens[:rows].copy_(ids[offset:offset + rows])
            self.start.fill_(offset)
            self.end.fill_(offset + b)
            self.last.fill_(rows - 1)
            self.graphs[self.splits(offset + b), "commit"].replay()

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=256, *, stop_on_eos=True, on_block=None, trace=False):
        """`on_block(token_ids, denoise_steps, elapsed_s, frames)` is called after each block (keep it cheap: it is
        timed). With `trace` (an engine built with trace=True), `frames` lists every round's input block over the
        emitted positions, random tokens included; otherwise it is None. The emitted tokens are the same either way.
        """
        if trace and not self.trace:
            raise ValueError("generate(trace=True) needs Engine(..., trace=True)")
        graphs = self.traced if trace else self.blocks
        with torch.cuda.device(self.model.device):
            ids = check_prompt(prompt_ids, max_new_tokens, self.model.config.vocab_size)
            b = self.block
            # A prompt of at least one block is prefilled; a shorter one shares the first block.
            prefix = len(ids) if len(ids) >= b else 0
            total = prefix + (len(ids) - prefix + max_new_tokens + b - 1) // b * b
            if total > self.model.max_seq_len:
                raise ValueError("Rounded request exceeds max_seq_len")
            stream = torch.cuda.current_stream().cuda_stream
            torch.cuda.synchronize()
            started = time.perf_counter()
            device_ids = torch.tensor(ids, device=self.model.device, dtype=torch.int64)
            self.logits[b].zero_()
            if prefix:
                self._prefill(device_ids, prefix)
            else:
                self.prompt_ids[:len(ids)].copy_(device_ids)
            torch.cuda.synchronize()
            prefill_s = time.perf_counter() - started
            eos = self.model.config.eos_token_id
            generated, steps = [], []
            for ordinal, offset in enumerate(range(prefix, total, b)):
                count = max(0, len(ids) - offset)
                # Deterministic noise counts logical blocks; the whole prefill counts as one.
                self.params_host[:] = torch.tensor([offset, count, (int(prefix > 0) + ordinal) * b, b - 1])
                self.params.copy_(self.params_host, non_blocking=True)
                graphs[self.splits(offset + b)].replay(stream)
                self.out_tokens.copy_(self.tokens, non_blocking=True)
                self.out_counters[:1].copy_(self.step, non_blocking=True)
                self.out_counters[1:].copy_(self.remaining, non_blocking=True)
                self.done.record()
                self.done.synchronize()
                if self.out_counters[1]:
                    raise RuntimeError("Decoder exceeded its step limit")
                steps.append(int(self.out_counters[0]))
                # Accepted tokens remain editable until the block ends, so EOS is checked per block.
                block = self.out_tokens[count:].tolist()[: max_new_tokens - len(generated)]
                finished = stop_on_eos and eos in block
                block = block[: block.index(eos) + 1] if finished else block
                generated.extend(block)
                if on_block is not None:
                    frames = self.frames[:steps[-1], count:count + len(block)].tolist() if trace else None
                    on_block(block, steps[-1], time.perf_counter() - started, frames)
                if finished or len(generated) == max_new_tokens:
                    break
            return Generation(generated, time.perf_counter() - started, prefill_s, steps)
