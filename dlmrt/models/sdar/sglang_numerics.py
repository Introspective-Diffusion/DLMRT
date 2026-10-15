"""SDAR with SGLang's numerics (`dlmrt bench suite --numerics sglang`): every forward's logits and every decision are
bitwise those of the paper's SGLang runs (SDARPrefixKV: its fused fixed-shape entropy path for ORL, its generic
probability-threshold path for Base) on the same GPU software stack.

The forward runs the paper runtime's arithmetic in its order: sgl_kernel norms, cuBLAS projections and LM head,
FlashInfer FA3 attention (paged over [0, end) for denoise rounds; for the causal commits and the prefill, a causal pass
over the rows and a non-causal pass over their prefix, merged), and its self-conditioning (row softmax, the soft
embedding of the non-prompt rows in chunks of self_conditioning_chunk_size, the gated adapter, the post-norm). Where
faster, kernels that reproduce those bit for bit run instead: DeepGEMM / the TK gate_up + SwiGLU (FAST_ROWS), the
SGLang variant of the post-attention megakernel (ops/csrc/ogud.cu), the paged FA3 numerics through FlashInfer's patched
ragged kernel (ops/fa3_tma.py), and the QK norm + RoPE, softmax, soft-embedding split-K GEMM and elementwise ops in
Triton (ops/triton/sglang_rope.py, sglang_kernels.py). Prompts follow its
layout: a prompt of a block or more is prefilled in 1024-row chunks and the first block starts right after it; a
shorter one shares the first block. A block's rounds run in one of dlmrt's block graphs. BF16 on one GPU; verified on
H100 with torch 2.11, Triton 3.6, FlashInfer 0.6.12 and sgl_kernel 0.4.3, the paper's stack.
"""

import time
from pathlib import Path

import deep_gemm
import flashinfer
import torch
import torch.nn.functional as F
from flashinfer.cascade import merge_state
from safetensors import safe_open
from sgl_kernel import fused_add_rmsnorm, rmsnorm

from ...ops.extensions import block_graph, init_deep_gemm, ogud, tk
from ...ops.fa3_tma import PagedNumericsWrapper
from ...ops.triton.common import PDL
from ...ops.triton.sglang_rope import qk_norm_rope_place
from ...runtime.generation import Generation, check_prompt
from . import sglang_kernels as sk
from .config import DecodeConfig, ModelConfig
from .model import ogud_fits

WORKSPACE_BYTES = 128 << 20
PREFILL_CHUNK = 1024  # rows per prefill forward (SGLang's chunked prefill for these runs)
# A block's 256-row projections without split-K in cuBLAS (QKV, O, gate_up) are each output's K-ordered accumulation,
# which DeepGEMM and the TK gate_up + SwiGLU kernel (both products rounded to BF16, then SiLU-and-multiply in FP32)
# compute bitwise too, faster; down (split-K in cuBLAS) and other row counts stay on cuBLAS.
FAST_ROWS = 256


class SglangSdarModel:
    """BF16 SDAR weights (QKV and gate/up packed), one request, KV slot = position."""

    def __init__(self, path, max_seq_len=65536, device="cuda:0"):
        self.config = c = ModelConfig.load(path)
        self.device, self.dtype, self.group, self.max_seq_len = torch.device(device), torch.bfloat16, None, max_seq_len
        self.hq, self.hk, self.d, self.eps = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rms_norm_eps
        w = {}
        for shard in sorted(Path(path).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as s:
                for key in s.keys():  # noqa: SIM118 - safe_open is not a mapping.
                    w[key] = s.get_tensor(key).to(device, torch.bfloat16)
        self.embedding = w.pop("model.embed_tokens.weight")
        head = w.pop("lm_head.weight", None)
        if head is not None and not torch.equal(head, self.embedding):
            raise ValueError("Tied checkpoint has different embedding and LM-head weights")
        self.final_norm = w.pop("model.norm.weight")
        init_deep_gemm()
        self.layers = []
        for i in range(c.num_hidden_layers):
            p = f"model.layers.{i}."
            self.layers.append({
                "norm": w.pop(p + "input_layernorm.weight"),
                "qkv": torch.cat([w.pop(p + f"self_attn.{n}_proj.weight") for n in "qkv"]),
                "qk_norm": torch.stack([w.pop(p + "self_attn.q_norm.weight"), w.pop(p + "self_attn.k_norm.weight")]),
                "o": w.pop(p + "self_attn.o_proj.weight"),
                "post_norm": w.pop(p + "post_attention_layernorm.weight"),
                "gate_up": torch.cat([w.pop(p + "mlp.gate_proj.weight"), w.pop(p + "mlp.up_proj.weight")]),
                "down": w.pop(p + "mlp.down_proj.weight"),
            })
        self.sc = None
        if c.sc:
            self.sc = {n: w.pop(f"self_conditioning.{n}.weight") for n in ("pre_norm", "gate_proj", "up_proj",
                                                                           "down_proj")}
            # A full block's soft embedding (chunks of self_conditioning_chunk_size rows, split-K in cuBLAS) as one
            # launch reproducing cuBLAS's split; partial blocks keep cuBLAS.
            self.sc_split = sk.cublas_split(c.self_conditioning_chunk_size, self.embedding)
            if self.sc_split is not None:
                self.sc_scratch = sk.split_scratch(c.block_size, c.hidden_size, self.sc_split[0], device)
        if w:
            raise ValueError(f"Unconsumed checkpoint weights: {sorted(w)[:8]}")
        # A block's post-attention chain (O, residual RMSNorm, gate_up + SwiGLU, split-K down) as the bitwise SGLang
        # variant of ogud's persistent kernel (SDAR-4B's shapes on a 132-SM H100); one counter slot per layer.
        self.ogud = ogud_fits(c, device)
        if self.ogud:
            ogud()
            self.post_counters = torch.zeros(c.num_hidden_layers, 512, device=device, dtype=torch.int32)
        self.k = [torch.zeros(max_seq_len, 1, self.hk, self.d, device=device, dtype=self.dtype) for _ in self.layers]
        self.v = [torch.zeros_like(x) for x in self.k]
        # RotaryEmbedding's FP32 cos / sin cache (full rotary), computed on the device.
        inv = 1.0 / c.rope_theta ** (torch.arange(0, self.d, 2, device=device).float() / self.d)
        angle = torch.arange(max_seq_len, device=device).float()[:, None] * inv[None, :]
        self.rope = torch.cat((angle.cos(), angle.sin()), dim=-1)

        def workspace():
            return torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

        def graph_buffers(graph):
            return dict(use_cuda_graph=True, qo_indptr_buf=torch.zeros(2, device=device, dtype=torch.int32),
                        kv_indptr_buf=torch.zeros(2, device=device, dtype=torch.int32)) if graph else {}

        def ragged(graph):
            return flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace(), "NHD", backend="fa3",
                                                                   **graph_buffers(graph))

        # Denoise rounds: SGLang's paged FA3 (page size 1) over [0, block end), as the bitwise-equal ragged kernel on
        # the contiguous cache (ops/fa3_tma.py). Causal passes: block-causal FA3 over the rows' own K/V and, after a
        # prefix, a non-causal pass over it, merged by log-sum-exp. CUDA-graph wrappers for the block graphs (rounds
        # and commit), planned on the host per block; eager ones for prefill chunks.
        self.denoise = PagedNumericsWrapper(workspace(), **graph_buffers(True))
        self.commit = (ragged(True), PagedNumericsWrapper(workspace(), **graph_buffers(True)))
        self.prefill = (ragged(False), PagedNumericsWrapper(workspace()))

    def _plan(self, wrapper, rows, kv, causal):
        wrapper.plan(torch.tensor([0, rows], dtype=torch.int32), torch.tensor([0, kv], dtype=torch.int32), self.hq,
                     self.hk, self.d, causal=causal, q_data_type=self.dtype, kv_data_type=self.dtype)

    def plan_block(self, rows, offset):
        """The block at `offset`: its denoise rounds over [0, offset + rows) and its commit's prefix [0, offset)."""
        self._plan(self.denoise, rows, offset + rows, False)
        if offset:
            self._plan(self.commit[1], rows, offset, False)

    def plan_prefill(self, rows, offset):
        self._plan(self.prefill[0], rows, rows, True)
        if offset:
            self._plan(self.prefill[1], rows, offset, False)

    def _attend(self, i, q, positions, causal):
        """causal: None (a denoise round) or (rows wrapper, prefix wrapper or None)."""
        cache_k, cache_v = (x[i].view(-1, self.hk, self.d) for x in (self.k, self.v))
        if causal is None:
            # into an uninitialized output (SGLang's forward() zero-fills its own; the kernel writes all of it)
            return self.denoise.run(q, cache_k, cache_v, out=torch.empty_like(q))
        n = q.shape[0]
        k = self.k[i].index_select(0, positions).view(n, self.hk, self.d)
        v = self.v[i].index_select(0, positions).view(n, self.hk, self.d)
        rows_wrapper, prefix_wrapper = causal
        if prefix_wrapper is None:
            return rows_wrapper.forward(q, k, v, causal=True)
        rows, rows_lse = rows_wrapper.forward_return_lse(q, k, v, causal=True)
        prefix, prefix_lse = prefix_wrapper.forward_return_lse(q, cache_k, cache_v, causal=False)
        return merge_state(rows, rows_lse, prefix, prefix_lse)[0]

    def _linear(self, x, weight):
        if x.shape[0] != FAST_ROWS:
            return F.linear(x, weight)
        out = torch.empty(x.shape[0], weight.shape[0], device=self.device, dtype=self.dtype)
        deep_gemm.bf16_gemm_nt(x, weight, out)
        return out

    def forward(self, h, positions, *, causal=None, out=None, last=False):
        """h: input embeddings [rows, hidden] at `positions` (int64; consumed); `causal` as in _attend. Returns the
        BF16 LM-head logits into `out`, or with `last` only the last row's (from the full-rows GEMM, as SGLang's
        causal passes)."""
        n = h.shape[0]
        residual = None
        for i, layer in enumerate(self.layers):
            if residual is None:
                residual = h
                x = rmsnorm(h, layer["norm"], self.eps)
            else:
                fused_add_rmsnorm(h, residual, layer["norm"], self.eps)
                x = h
            qkv = self._linear(x, layer["qkv"])
            q = torch.empty(n, self.hq, self.d, device=self.device, dtype=self.dtype)
            qk_norm_rope_place(qkv, q, self.k[i], self.v[i], positions, self.rope, layer["qk_norm"], self.eps,
                               self.hq, self.hk, self.d, self.d, write_back=False)
            attn = self._attend(i, q, positions, causal).view(n, -1)
            if self.ogud and n == FAST_ROWS:
                h = torch.empty_like(residual)
                ogud().post_attention_sglang(attn, residual, layer["o"], layer["post_norm"], self.eps,
                                             layer["gate_up"], layer["down"], torch.empty_like(residual),
                                             torch.empty_like(residual),
                                             torch.empty(n, layer["down"].shape[1], device=self.device,
                                                         dtype=self.dtype), h, self.post_counters, i, PDL)
                continue
            a = self._linear(attn, layer["o"])
            fused_add_rmsnorm(a, residual, layer["post_norm"], self.eps)
            act = torch.empty(n, layer["gate_up"].shape[0] // 2, device=self.device, dtype=self.dtype)
            if n == FAST_ROWS:
                tk().gate_up_swiglu(a, layer["gate_up"], act, PDL, 1, 2, False)
            else:
                sk.silu_mul(F.linear(a, layer["gate_up"]), act)
            h = F.linear(act, layer["down"])
        fused_add_rmsnorm(h, residual, self.final_norm, self.eps)
        if last:
            return F.linear(h, self.embedding)[-1:]
        return torch.mm(h, self.embedding.t(), out=out)

    def embed(self, ids, prev_logits=None, prompt_rows=0, signal=None):
        """SGLang's input embeddings: with `prev_logits` ([rows, vocab], the previous round's), the soft embedding of
        the non-prompt rows (their softmax, then chunks of self_conditioning_chunk_size rows) through the gated adapter
        (into `signal`, its prompt rows zero); then the post-norm. Without, the adapter of a zero signal is the
        identity, so only the post-norm applies."""
        c = self.config
        x = F.embedding(ids, self.embedding)
        if self.sc is None:
            return x
        if prev_logits is not None:
            if prompt_rows:
                signal[:prompt_rows].zero_()
            rows = prev_logits[prompt_rows:]
            probs = torch.empty_like(rows)
            sk.soft_probs(rows, probs)
            chunk = c.self_conditioning_chunk_size
            if not prompt_rows and self.sc_split is not None:
                sk.split_gemm(probs, self.embedding, signal, self.sc_split, *self.sc_scratch)
            else:
                for s in range(0, rows.shape[0], chunk):
                    torch.mm(probs[s:s + chunk], self.embedding, out=signal[prompt_rows + s:prompt_rows + s + chunk])
            s = rmsnorm(signal, self.sc["pre_norm"], self.eps) * c.self_conditioning_prenorm_scale
            gated = F.silu(self._linear(s, self.sc["gate_proj"])) * self._linear(s, self.sc["up_proj"])
            sk.scaled_add_(x, F.linear(gated, self.sc["down_proj"]),
                           c.self_conditioning_residual_scale * c.self_conditioning_reparam_out_scale)
        if c.self_conditioning_post_norm:
            x = F.rms_norm(x, (c.hidden_size,), eps=self.eps) * c.self_conditioning_post_norm_scale
        return x


def temperature_table(decoder, max_steps):
    """FP32 1/T of SGLang's step_linear confidence temperature for each 1-based step (index 0 as step 1), the cap at
    max_steps folded in; computed in python doubles as its post-step graphs do."""
    inv = []
    for step in range(max_steps + 3):
        capped = min(max(step, 1), max_steps)
        t = max(decoder.confidence_temperature_min, decoder.confidence_temperature
                - decoder.confidence_temperature_step_decrement * float(max(capped - 1, 0)))
        inv.append(1.0 / max(float(t), 1e-6))
    return inv


class SglangSdarEngine:
    """One request at a time on SglangSdarModel. Each block is one block graph: the first round without
    self-conditioning, rounds with it while positions remain, then the commit (a causal forward over the block whose
    last row's logits become the next block's boundary). The next block's attention is planned on the host while a
    block runs."""

    def __init__(self, model, decoder=None):
        """Captures the block graphs of prompt-row counts 1 .. block_size - 1 now (that of a block without prompt rows
        on first use)."""
        self.model = m = model
        self.decoder = d = decoder or DecodeConfig()
        if d.accept_rule == "prob_threshold" and (d.confidence_temperature != 1.0 or d.confidence_temperature_min
                                                  != 1.0):
            raise NotImplementedError("SGLang numerics: the probability-threshold decoder at confidence temperature 1")
        if not d.logits_shift or not d.allow_accepted_change:
            raise NotImplementedError("SGLang numerics: the paper's decoders (shifted logits, edits on)")
        c = m.config
        self.block = b = c.block_size
        dev = m.device
        self.tokens = torch.zeros(b, device=dev, dtype=torch.int64)
        self.prompt = torch.zeros(b, device=dev, dtype=torch.bool)
        self.accepted = torch.zeros(b, device=dev, dtype=torch.bool)
        self.step = torch.zeros(1, device=dev, dtype=torch.int32)
        self.remaining = torch.zeros(1, device=dev, dtype=torch.int32)
        self.noise_block = torch.zeros(1, device=dev, dtype=torch.int64)
        self.start = torch.zeros(1, device=dev, dtype=torch.int64)
        self.rows = torch.arange(b, device=dev, dtype=torch.int64)
        self.positions = torch.zeros(b, device=dev, dtype=torch.int64)
        # Row 0: the boundary distribution (the last row of the previous causal pass) for the first position's
        # shifted logits; rows 1..b: the round's.
        self.logits = torch.zeros(b + 1, c.vocab_size, device=dev, dtype=m.dtype)
        self.signal = torch.zeros(b, c.hidden_size, device=dev, dtype=m.dtype) if m.sc is not None else None
        self.proposal = torch.zeros(b, device=dev, dtype=torch.int64)
        self.pmax = torch.zeros(b, device=dev, dtype=torch.float32)
        self.entropy = torch.zeros(b, device=dev, dtype=torch.float32)
        self.scratch = torch.zeros(b, device=dev, dtype=torch.int8)
        self.inv_t = torch.tensor(temperature_table(d, b), device=dev, dtype=torch.float32)
        self.snapshot = torch.zeros(1, device=dev, dtype=torch.int32)
        self.out_tokens = torch.zeros(b, dtype=torch.int64, pin_memory=True)
        self.out_step = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.done = torch.cuda.Event()
        self.graphs, self.blocks = {}, {}
        self.pool = torch.cuda.graph_pool_handle()
        with torch.cuda.device(dev):
            m._plan(m.commit[0], b, b, True)
            m.plan_block(b, b)
            for name, fn in (("first", lambda: self._round(False)), ("round0", lambda: self._round(True)),
                             ("tail", lambda: self._tail(True)), ("tail0", lambda: self._tail(False))):
                self._capture(name, fn)
            for prompt_rows in range(1, b):
                self._block_graph(prompt_rows)
        self.planned = None

    def _capture(self, name, fn):
        stream = torch.cuda.Stream(self.model.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
            fn()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graphs[name] = graph

    def _block_graph(self, prompt_rows):
        """The block graph for a block whose first `prompt_rows` rows hold the prompt (their soft-embedding rows are
        compacted away, which changes the GEMM shapes: one graph per count, captured on first use). Capturing runs
        rounds, so it comes before the block's state is set."""
        if prompt_rows not in self.blocks:
            name = f"round{prompt_rows}"
            if name not in self.graphs:
                self._capture(name, lambda: self._round(True, prompt_rows))
            raw = {k: g.raw_cuda_graph() for k, g in self.graphs.items()}
            # A block with prompt rows is a request's first, at position 0: its commit has no prefix.
            self.blocks[prompt_rows] = block_graph().LoopGraph(
                raw["first"], [(raw[name], self.remaining.data_ptr(), self.step.data_ptr(), self.block)],
                raw["tail0" if prompt_rows else "tail"])
        return self.blocks[prompt_rows]

    def _round(self, conditioned, prompt_rows=0):
        m, b = self.model, self.block
        torch.add(self.rows, self.start, out=self.positions)
        h = m.embed(self.tokens, self.logits[1:] if conditioned else None, prompt_rows, self.signal)
        m.forward(h, self.positions, out=self.logits[1:])
        state = (self.tokens, self.accepted, self.prompt, self.step, self.remaining, self.noise_block)
        eligible = ~self.accepted & ~self.prompt
        if self.decoder.accept_rule == "entropy_budget":
            inv_t = self.inv_t.index_select(0, torch.clamp(self.step + 1, max=self.inv_t.numel() - 1))
            sk.row_stats_temp(self.logits[:b], inv_t, self.proposal, self.pmax, self.entropy)
            sorted_entropy, order = torch.sort(torch.where(eligible, self.entropy, float("inf")), descending=False)
            excl = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy
            sk.decide(state, self.proposal, self.pmax, self.decoder, order=order, sorted_entropy=sorted_entropy,
                      excl=excl, scratch=self.scratch)
        else:
            shifted = self.logits[:b].float()
            proposal = torch.argmax(shifted, dim=-1)
            pp = F.softmax(shifted, dim=-1).gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
            top = torch.topk(torch.where(eligible, pp, torch.full_like(pp, -float("inf"))), k=1).indices
            sk.decide(state, proposal, pp, self.decoder, top=top)

    def _tail(self, prefix):
        """The commit of the block at self.start, then the step count for the host."""
        m = self.model
        torch.add(self.rows, self.start, out=self.positions)
        causal = (m.commit[0], m.commit[1] if prefix else None)
        self.logits[:1].copy_(m.forward(m.embed(self.tokens), self.positions, causal=causal, last=True))
        self.snapshot.copy_(self.step)

    def _plan(self, offset):
        if self.planned != offset and offset + self.block <= self.model.max_seq_len:
            self.model.plan_block(self.block, offset)
            self.planned = offset

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=256, *, stop_on_eos=True):
        m, d, b = self.model, self.decoder, self.block
        c = m.config
        with torch.cuda.device(m.device):
            ids = check_prompt(prompt_ids, max_new_tokens, c.vocab_size)
            prefix = len(ids) if len(ids) >= b else 0
            total = prefix + (len(ids) - prefix + max_new_tokens + b - 1) // b * b
            if total > m.max_seq_len:
                raise ValueError("Request exceeds max_seq_len")
            stream = torch.cuda.current_stream().cuda_stream
            torch.cuda.synchronize()
            started = time.perf_counter()
            device_ids = torch.tensor(ids, device=m.device, dtype=torch.int64)
            self.logits[0].zero_()
            for offset in range(0, prefix, PREFILL_CHUNK):
                rows = min(PREFILL_CHUNK, prefix - offset)
                m.plan_prefill(rows, offset)
                positions = torch.arange(offset, offset + rows, device=m.device, dtype=torch.int64)
                causal = (m.prefill[0], m.prefill[1] if offset else None)
                self.logits[:1].copy_(m.forward(m.embed(device_ids[offset:offset + rows]), positions, causal=causal,
                                                last=True))
            torch.cuda.synchronize()
            prefill_s = time.perf_counter() - started
            generated, steps = [], []
            for ordinal, offset in enumerate(range(prefix, total, b)):
                prompt_rows = max(0, len(ids) - offset)
                noise_block = int(prefix > 0) + ordinal
                graph = self._block_graph(prompt_rows)
                position = self.rows + noise_block * b
                self.tokens.copy_(((position + 1) * 1103515245 + 12345 + d.randtok_seed) % d.random_token_high)
                if prompt_rows:
                    self.tokens[:prompt_rows].copy_(device_ids[offset:])
                torch.lt(self.rows, prompt_rows, out=self.prompt)
                self.accepted.copy_(self.prompt)
                self.step.zero_()
                self.remaining.fill_(b - prompt_rows)
                self.noise_block.fill_(noise_block)
                self.start.fill_(offset)
                self._plan(offset)
                graph.replay(stream)
                self._plan(offset + b)
                self.out_tokens.copy_(self.tokens, non_blocking=True)
                self.out_step.copy_(self.snapshot, non_blocking=True)
                self.done.record()
                self.done.synchronize()
                steps.append(int(self.out_step))
                block = self.out_tokens[prompt_rows:].tolist()[:max_new_tokens - len(generated)]
                finished = stop_on_eos and c.eos_token_id in block
                if finished:
                    block = block[:block.index(c.eos_token_id) + 1]
                generated.extend(block)
                if finished or len(generated) == max_new_tokens:
                    break
            return Generation(generated, time.perf_counter() - started, prefill_s, steps)
