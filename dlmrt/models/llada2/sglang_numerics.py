"""LLaDA2 with SGLang's numerics (`dlmrt bench suite --numerics sglang`): every forward's logits, and so every decision
dlmrt's decoder kernels take from them, are bitwise those of the paper's SGLang runs (SDARPrefixKV,
commit_mode=fused_next) on the same GPU software stack.

The forward runs SGLang's kernels in its call order: sgl_kernel norms and SiLU, F.linear on cuBLAS, FlashInfer FA2
attention planned as SGLang's backend plans dLLM passes, FlashInfer's grouped top-k and SGLang's Triton fused MoE
(vendored: ops/triton/sglang_fused_moe.py); or dlmrt kernels that reproduce theirs bitwise in fewer launches
(sglang_kernels.py). Rounds run in dlmrt's block graphs. BF16 on one GPU; verified on H100 with torch 2.11, Triton
3.6, FlashInfer 0.6.12 and sgl_kernel 0.4.3, the paper's stack.
"""

import json
from pathlib import Path

import flashinfer
import torch
import torch.nn.functional as F
import triton
from flashinfer.fused_moe import fused_topk_deepseek
from safetensors import safe_open
from sgl_kernel import fused_add_rmsnorm, rmsnorm, silu_and_mul

from ...ops.extensions import block_graph
from ...ops.triton import sglang_fused_moe as fused_moe
from ...ops.triton.sglang_rope import qk_norm_rope_place
from . import kernels as lk
from . import sglang_kernels as sk
from .config import MAX_STEPS_PER_BLOCK, LladaConfig
from .engine import LladaEngine
from .model import _normalize

STEP, CARRY_T, REMAINING, CARRY_LOOP, CARRY_STEP = (s.value for s in (lk.STEP, lk.CARRY_T, lk.REMAINING, lk.CARRY_LOOP,
                                                                      lk.CARRY_STEP))

# SGLang's default FlashInfer workspace (SGLANG_FLASHINFER_WORKSPACE_SIZE; LLaDA2 keeps the default).
WORKSPACE_BYTES = 384 << 20
# Fused-MoE tiles for the denoise passes' token counts. Every tile, warps and stages choice of the kernel gives
# bitwise the default's results (each output a K-ordered accumulation; 138 configs checked on real routing).
MOE_TILES = {32: dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3),
             64: dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2)}
# The engine captures the prefill graphs of every prompt up to this many rows at load (warm_prefill).
WARM_ROWS = 1024


def confidence_temperature(decoder, step):
    """SGLang's confidence temperature at 1-based block step `step` (its step_linear schedule, in python floats)."""
    conf = decoder.confidence
    if not decoder.anneals:
        return conf.confidence_temperature
    capped = min(step, MAX_STEPS_PER_BLOCK)
    return max(conf.confidence_temperature_min,
               conf.confidence_temperature - conf.confidence_temperature_step_decrement * float(max(capped - 1, 0)))


class SglangLladaModel:
    """BF16 LLaDA2 weights in SGLang's layout (router and expert bias FP32); one request; KV slot = position."""

    def __init__(self, path, max_seq_len=16384, device="cuda:0"):
        self.config = c = LladaConfig.load(Path(path))
        raw = json.loads((Path(path) / "config.json").read_text())
        if raw.get("quantization_config"):
            raise ValueError("SGLang numerics: BF16 checkpoints only")
        # Graph capture writes the K/V of the first WARM_ROWS positions; the RoPE table covers max_position_embeddings.
        if not WARM_ROWS <= max_seq_len <= raw["max_position_embeddings"]:
            raise ValueError(f"SGLang numerics: max_seq_len must be in [{WARM_ROWS}, {raw['max_position_embeddings']}] "
                             "(the prefill graphs, the RoPE table)")
        self.device, self.dtype, self.group, self.max_seq_len = torch.device(device), torch.bfloat16, None, max_seq_len
        self.precision, self.vocab, self.vocab_offset = "bf16", c.vocab_size, 0
        self.max_rows = 256
        self.hq, self.hk, self.d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.rot, self.eps = c.rotary_dim, c.rms_norm_eps
        e, f_, hidden = c.num_experts, c.moe_intermediate_size, c.hidden_size
        w, stacks = {}, {}
        index = Path(path) / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else \
            [p.name for p in sorted(Path(path).glob("*.safetensors"))]
        for name in files:
            with safe_open(Path(path) / name, framework="pt", device="cpu") as s:
                for key in s.keys():  # noqa: SIM118 - safe_open is not a mapping.
                    if ".mlp.experts." not in key:
                        keep = torch.float32 if key.endswith((".mlp.gate.weight", ".expert_bias")) else torch.bfloat16
                        w[key] = s.get_tensor(key).to(device, keep)
                        continue
                    prefix, rest = key.split(".mlp.experts.")
                    j, proj = rest.split(".")[:2]
                    if prefix not in stacks:
                        stacks[prefix] = (torch.empty(e, 2 * f_, hidden, device=device, dtype=torch.bfloat16),
                                          torch.empty(e, hidden, f_, device=device, dtype=torch.bfloat16), [0])
                    w13, w2, seen = stacks[prefix]
                    j = int(j)
                    {"gate_proj": w13[j, :f_], "up_proj": w13[j, f_:], "down_proj": w2[j]}[proj].copy_(
                        s.get_tensor(key))
                    seen[0] += 1
        for prefix, (_, _, seen) in stacks.items():
            if seen[0] != 3 * e:
                raise ValueError(f"{prefix}: {seen[0]} of {3 * e} expert tensors")
        w = _normalize(w, c)

        def bf(key):
            return w.pop(key).to(torch.bfloat16).contiguous()

        self.embedding = bf("model.word_embeddings.weight")
        self.lm_head = bf("lm_head.weight")
        self.final_norm = bf("model.norm.weight")
        self.layers = []
        for i in range(c.num_hidden_layers):
            p = f"model.layers.{i}."
            layer = {"norm": bf(p + "input_layernorm.weight"), "qkv": bf(p + "attention.query_key_value.weight"),
                     "o": bf(p + "attention.dense.weight"), "post_norm": bf(p + "post_attention_layernorm.weight"),
                     "moe": i >= c.first_k_dense_replace}
            layer["qk_norm"] = torch.stack([bf(p + "attention.query_layernorm.weight"),
                                            bf(p + "attention.key_layernorm.weight")]).contiguous()
            if layer["moe"]:
                layer["router"] = w.pop(p + "mlp.gate.weight").contiguous()
                layer["bias"] = w.pop(p + "mlp.gate.expert_bias").contiguous()
                layer["w13"], layer["w2"], _ = stacks.pop(f"model.layers.{i}")
                s = p + "mlp.shared_experts."
            else:
                s = p + "mlp."
            layer["gate_up"] = torch.cat([bf(s + "gate_proj.weight"), bf(s + "up_proj.weight")]).contiguous()
            layer["down"] = bf(s + "down_proj.weight")
            self.layers.append(layer)
        self.sc = None
        if c.use_self_conditioning:
            if c.self_conditioning_post_norm:
                raise NotImplementedError("self_conditioning_post_norm")
            s = "model.self_conditioning."
            self.sc = {n: bf(s + n + ".weight") for n in ("pre_norm", "gate_proj", "up_proj", "down_proj")}
            self.sc_scales = tuple(float(x) for x in (c.self_conditioning_prenorm_scale,
                                                      c.self_conditioning_residual_scale,
                                                      c.self_conditioning_reparam_out_scale))
        if w or stacks:
            raise ValueError(f"Unconsumed checkpoint weights: {(sorted(w) + sorted(stacks))[:8]}")
        # RotaryEmbedding's FP32 cos / sin cache, computed on the device as SGLang's model loader does.
        with torch.device(device):
            inv = 1.0 / (c.rope_theta ** (torch.arange(0, self.rot, 2, dtype=torch.float) / self.rot))
            t = torch.arange(raw["max_position_embeddings"], dtype=torch.float)
            freqs = torch.einsum("i,j -> ij", t, inv)
            self.cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.k = [torch.zeros(max_seq_len, 1, self.hk, self.d, device=device, dtype=self.dtype) for _ in self.layers]
        self.v = [torch.zeros_like(x) for x in self.k]
        self.side = torch.cuda.Stream(self.device)
        self.positions = torch.arange(max_seq_len, device=device, dtype=torch.int64)
        self.slots = torch.arange(max_seq_len, device=device, dtype=torch.int32)
        self.align_cursor = torch.empty(triton.next_power_of_2(e + 1), dtype=torch.int32, device=device)
        self.router_part = torch.empty(hidden // 64 * 64 * e, dtype=torch.float32, device=device)
        self.router_count = torch.zeros(e // 16, dtype=torch.int32, device=device)
        chunks = triton.cdiv(c.vocab_size, sk.STAT_CHUNK) * 64  # step statistics of up to 64 rows
        self.stat_part = torch.empty(chunks, dtype=torch.float32, device=device)
        self.stat_part_i = torch.empty(chunks, dtype=torch.int32, device=device)
        self.stat_count = torch.zeros(64, dtype=torch.int32, device=device)
        self.stat_xmax = torch.empty(64, dtype=torch.float32, device=device)
        self.prefill_graphs, self.prefill_pool, self.prefill_workspace = {}, None, None
        # CUDA-graph FlashInfer wrappers for the denoise passes, one per batch size as SGLang's graph runner builds
        # them (one block: bs 1; a fused pair: bs 2), each on its own workspace so that both stay planned through a
        # block's graph.
        self.denoise = {}
        for bs in (1, 2):
            self.denoise[bs] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device), "NHD", backend="fa2",
                use_cuda_graph=True, qo_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=device),
                paged_kv_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=device),
                paged_kv_indices_buf=torch.zeros(2 * max_seq_len, dtype=torch.int32, device=device),
                paged_kv_last_page_len_buf=torch.ones(bs, dtype=torch.int32, device=device))
        self.planned, self.planned_copies = None, torch.cuda.Event()

    def plan_denoise(self, ends):
        """DLLM_EXTEND denoise over one block (ends = [block end]) or a fused pair of blocks (ends = [carried block
        end, this block end]) as FlashInferAttnBackend plans it: paged FA2, page size 1, every request sees [0, its
        end). The indptrs stay on the host (the plan reads them there)."""
        b, bs = self.config.block_size, len(ends)
        kv_indptr = torch.tensor([0] + [sum(ends[:i + 1]) for i in range(bs)], dtype=torch.int32)
        qo_indptr = torch.arange(0, (bs + 1) * b, b, dtype=torch.int32)
        kv_indices = self.slots[:ends[0]] if bs == 1 else torch.cat([self.slots[:end] for end in ends])
        last = torch.ones(bs, dtype=torch.int32)
        self.denoise[bs].begin_forward(qo_indptr, kv_indptr, kv_indices, last, self.hq, self.hk, self.d, 1,
                                       q_data_type=self.dtype, kv_data_type=self.dtype, custom_mask=None,
                                       non_blocking=True, fixed_split_size=None)

    def plan_graphs(self, offset):
        """Plan both denoise wrappers for the block at `offset` (the pair one only when a block precedes it). The plans'
        device copies queue on the current stream, behind any graph replay already queued; their pinned host staging
        is reused by the next plan, hence `planned_copies`, which completes once they did."""
        b = self.config.block_size
        self.plan_denoise([offset + b])
        if offset >= b:
            self.plan_denoise([offset, offset + b])
        self.planned = offset
        self.planned_copies.record()

    def _attention(self, i, x, positions, wrapper, prefill):
        layer, n = self.layers[i], x.shape[0]
        qkv = F.linear(x, layer["qkv"])
        _, k, v = qkv.split([self.hq * self.d, self.hk * self.d, self.hk * self.d], dim=-1)
        # q made contiguous and this pass's K/V written (SGLang: q.contiguous(), then set_kv_buffer before a paged
        # pass, after a ragged one; either way the attention reads the same values)
        qc = torch.empty(n, self.hq, self.d, device=x.device, dtype=x.dtype)
        qk_norm_rope_place(qkv, qc, self.k[i], self.v[i], positions, self.cos_sin, layer["qk_norm"], self.eps,
                              self.hq, self.hk, self.d, self.rot)
        if prefill is not None:
            o = prefill.forward(qc, k.view(-1, self.hk, self.d), v.view(-1, self.hk, self.d), causal=False,
                                sm_scale=self.d ** -0.5, logits_soft_cap=0.0)
        else:
            # wrapper.forward(qc, kv, causal=False, sm_scale, window_left=-1, logits_soft_cap=0.0) as SGLang calls
            # it, but into an uninitialized output: forward() zero-fills its own, and the kernels write all of it
            wrapper._causal, wrapper._pos_encoding_mode, wrapper._use_fp16_qk_reduction = False, "NONE", False
            wrapper._window_left, wrapper._logits_soft_cap, wrapper._sm_scale = -1, 0.0, self.d ** -0.5
            wrapper._rope_scale = wrapper._rope_theta = None
            o = wrapper.run(qc, (self.k[i], self.v[i]), out=torch.empty_like(qc))
        return F.linear(o.view(n, self.hq * self.d), layer["o"])

    def _mlp(self, x, layer):
        return F.linear(silu_and_mul(F.linear(x, layer["gate_up"])), layer["down"])

    def _moe(self, x, layer):
        c = self.config
        # The shared expert runs beside the router chain (both latency-bound), joined before the expert kernels.
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            shared = self._mlp(x, layer)
        # Router logits: the FP32 GEMM rounded to BF16, kept as FP32 values.
        m, router = x.shape[0], layer["router"]
        if m in (32, 64):
            logits = torch.empty(m, router.shape[0], device=x.device, dtype=torch.float32)
            sk.router_logits(x, router, logits, self.router_part, self.router_count)
        else:
            logits = F.linear(x.to(torch.float32), router, None)
            sk.round_bf16(logits)
        # select_experts: biased grouped top-k, renormalized, the routed scaling applied by the combine
        weights = torch.empty(m, c.num_experts_per_tok, device=x.device, dtype=torch.float32)
        ids = torch.empty(m, c.num_experts_per_tok, device=x.device, dtype=torch.int32)
        fused_topk_deepseek(logits, layer["bias"], c.n_group, c.topk_group, c.num_experts_per_tok, 1.0, weights, ids,
                            True)
        config = MOE_TILES.get(m) or fused_moe.default_config(m, c.num_experts)
        if ids.numel() > 1024:  # prefill chunks
            align = fused_moe.moe_align_block_size(ids, config["BLOCK_SIZE_M"], c.num_experts)
        else:
            align = sk.moe_align(ids, config["BLOCK_SIZE_M"], c.num_experts, self.align_cursor)
        main.wait_stream(self.side)
        shared.record_stream(main)
        return self._experts(x, layer, weights, ids, config, align, shared)

    def _experts(self, x, layer, weights, ids, config, align, shared):
        """fused_experts_impl's routed experts (BF16, outplace) on the `align` grouping, combined with the shared
        expert's output added as SGLang adds it."""
        m, w1, w2 = x.shape[0], layer["w13"], layer["w2"]
        t, n = ids.numel(), w1.shape[1]
        cache1 = torch.empty(t, n, device=x.device, dtype=x.dtype)
        cache2 = torch.empty(t, n // 2, device=x.device, dtype=x.dtype)
        cache3 = torch.empty(m, ids.shape[1], w2.shape[1], device=x.device, dtype=x.dtype)
        fused_moe.invoke_fused_moe_kernel(x, w1, cache1, weights, ids, *align, False, ids.shape[1], config)
        silu_and_mul(cache1.view(-1, n), cache2)
        fused_moe.invoke_fused_moe_kernel(cache2, w2, cache3, weights, ids, *align, True, 1, config)
        out = torch.empty_like(x)
        sk.moe_sum(cache3, shared, out, self.config.routed_scaling_factor)
        return out

    def forward(self, h, positions, wrapper=None, out=None, prefill=None):
        """h: [rows, hidden] input embeddings at `positions` (consumed); returns the BF16 LM-head logits [rows, vocab]
        (into `out` if given; SGLang widens the same values to FP32, which its step kernels read back exactly).
        `wrapper`: a planned denoise wrapper; `prefill`: a planned ragged wrapper for the prompt prefill, which only
        writes K/V (returns None)."""
        residual = None
        for i, layer in enumerate(self.layers):
            if residual is None:
                residual = h
                x = rmsnorm(h, layer["norm"], self.eps)
            else:
                fused_add_rmsnorm(h, residual, layer["norm"], self.eps)
                x = h
            a = self._attention(i, x, positions, wrapper, prefill)
            if prefill is not None and i == len(self.layers) - 1:
                return None
            fused_add_rmsnorm(a, residual, layer["post_norm"], self.eps)
            h = self._moe(a, layer) if layer["moe"] else self._mlp(a, layer)
        fused_add_rmsnorm(h, residual, self.final_norm, self.eps)
        return torch.matmul(h, self.lm_head.T, out=out)

    def embed(self, ids, prev_logits, sc_rows, first=0):
        """Input embeddings of `ids`, self-conditioned as SGLang's _fused_sc_embeds_fast without host syncs: the
        softmax of rows [first, 32) of the previous forward's logits (`prev_logits` [32, vocab], the leading rows of
        `ids`) pushed back through the embedding table, kept on rows where the bool `sc_rows` holds, then the gated
        residual adapter over every row. Equal to SGLang's on those rows when they are exactly its rows (first = the
        block's prompt rows), or a superset of more than 16 rows (cuBLAS keeps its kernel above M=16); with none, the
        plain embeddings (SGLang skips the adapter)."""
        x = F.embedding(ids, self.embedding)
        if self.sc is None:
            return x
        probs = F.softmax(prev_logits[first:].float(), dim=-1).to(self.embedding.dtype)
        soft = probs.matmul(self.embedding).to(x.dtype)
        last = first + soft.shape[0]
        kept = torch.where(sc_rows[first:last, None], soft, torch.zeros_like(soft))
        if first == 0 and last == x.shape[0]:
            signal = kept
        else:
            signal = torch.zeros_like(x)
            signal[first:last] = kept
        return torch.where(sc_rows.any(), self._adapter(x, signal), x)

    def _adapter(self, x, signal):
        """The gated residual adapter; scales of exactly one are skipped (multiplying by one is exact)."""
        prenorm, residual_scale, out_scale = self.sc_scales
        s = rmsnorm(signal, self.sc["pre_norm"], self.eps)
        if prenorm != 1.0:
            s = s * prenorm
        down = F.linear(F.silu(F.linear(s, self.sc["gate_proj"])) * F.linear(s, self.sc["up_proj"]),
                        self.sc["down_proj"])
        if residual_scale * out_scale != 1.0:
            down = residual_scale * out_scale * down
        return x + down

    def step_stats(self, logits, t, stats, proposal):
        sk.step_stats(logits, t, stats, proposal, self.stat_part, self.stat_part_i, self.stat_count, self.stat_xmax)

    def prefill(self, ids):
        """The prompt blocks' K/V (no LM head: SGLang's prefill logits are never read). One CUDA graph per prompt
        length, captured on first use, over a graph-mode ragged wrapper (bitwise the eager one SGLang runs)."""
        rows = ids.numel()
        entry = self.prefill_graphs.get(rows) or self._capture_prefill(rows)
        wrapper, ids_buf, mask, indptr, graph = entry
        ids_buf.copy_(ids)
        wrapper.begin_forward(indptr, indptr, self.hq, self.hk, self.d, custom_mask=mask, q_data_type=self.dtype)
        graph.replay()

    def warm_prefill(self, max_rows=WARM_ROWS):
        """Capture the prefill graphs of every prompt length up to `max_rows` now (longer ones on first use)."""
        for rows in range(self.config.block_size, max_rows + 1, self.config.block_size):
            if rows not in self.prefill_graphs:
                self._capture_prefill(rows)

    def _capture_prefill(self, rows):
        dev = self.device
        if self.prefill_pool is None:
            self.prefill_pool = torch.cuda.graph_pool_handle()
            self.prefill_workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self.prefill_workspace, "NHD", backend="fa2", use_cuda_graph=True,
            qo_indptr_buf=torch.zeros(2, dtype=torch.int32, device=dev),
            kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device=dev),
            custom_mask_buf=torch.zeros((rows * rows + 7) // 8, dtype=torch.uint8, device=dev),
            mask_indptr_buf=torch.zeros(2, dtype=torch.int32, device=dev))
        block = torch.arange(rows, device=dev) // self.config.block_size
        mask = (block[None, :] <= block[:, None]).reshape(-1)
        indptr = torch.tensor([0, rows], dtype=torch.int32, device=dev)
        ids_buf = torch.zeros(rows, dtype=torch.int64, device=dev)
        wrapper.begin_forward(indptr, indptr, self.hq, self.hk, self.d, custom_mask=mask, q_data_type=self.dtype)

        def run():
            self.forward(F.embedding(ids_buf, self.embedding), self.positions[:rows], prefill=wrapper)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.prefill_pool, stream=stream):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        self.prefill_graphs[rows] = entry = (wrapper, ids_buf, mask, indptr, graph)
        return entry


class _PlannedBlock:
    """A block's LoopGraph, its attention planned on the host first and the next block's planned while it runs. A
    prompt tail of 16 rows or more leaves the first block's self-conditioning soft embedding 16 rows or fewer, where
    cuBLAS picks other kernels than for 32: that block (its rounds) and the next (its pair rounds, which refine it)
    replay graphs captured for that tail."""

    def __init__(self, engine, graph):
        self.e, self.graph = engine, graph

    def replay(self, stream):
        e, b, m = self.e, self.e.block, self.e.model
        offset, count, first = (int(x) for x in e.params_host)
        if first:
            e.first_block = (offset, count)
        start, tail = e.first_block
        graph = self.graph
        if tail in e.tail_blocks and offset in (start, start + b):
            graph = e.tail_blocks[tail][offset != start]
        if m.planned != offset:
            m.plan_graphs(offset)
        graph.replay(stream)
        if offset + 2 * b <= m.max_seq_len:
            m.planned_copies.synchronize()
            m.plan_graphs(offset + b)


class SglangLladaEngine(LladaEngine):
    """LladaEngine on SglangLladaModel: the same decoder kernels and block protocol in dlmrt's block graphs (the
    device-side round loops), SGLang's numerics."""

    def __init__(self, model, decoder=None, *, fused=None, trace=False):
        if trace:
            raise ValueError("SGLang numerics: no traced block graphs")
        super().__init__(model, decoder, fused=fused)

    def _capture(self):
        m, b, d = self.model, self.block, self.decoder
        self.split_table = ((None, 1),)
        dev = m.device
        self.rows = torch.arange(2 * b, device=dev, dtype=torch.int64)
        self.positions = torch.zeros(2 * b, device=dev, dtype=torch.int64)
        # SGLang's step_linear temperature of 1-based block step s at index s - 1 (python doubles rounded to FP32,
        # as row_conf_prob receives its python float); indexed by STEP for a round, CARRY_T for a carried block,
        # both below self.limit (a carried block settles within refine_steps + 1 pair rounds).
        if d.anneals and self.limit > 128:
            raise ValueError(f"SGLang numerics: an annealed confidence temperature allows refine_steps <= {127 - b}")
        self.t_table = torch.tensor([confidence_temperature(d, s) for s in range(1, 129)], device=dev,
                                    dtype=torch.float32)
        self.pair_sc_rows = torch.zeros(2 * b, device=dev, dtype=torch.bool)
        self.first_block = (0, 0)
        self.params.copy_(torch.tensor([2 * b, 0, 0], dtype=torch.int32))
        m.plan_graphs(2 * b)
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        stages = {"begin": self._begin, "round": self._round, "pair": self._pair_round, "snapshot": self._snapshot}
        # first-block variants for prompt tails of 16 rows or more (self-conditioning only)
        tails = range(b // 2, b) if m.sc is not None and self.fused else ()
        for tail in tails:
            stages[f"round{tail}"] = lambda t=tail: self._round(first=t)
            stages[f"pair{tail}"] = lambda t=tail: self._pair_round(first=t)
        self.graphs = {}
        with torch.cuda.stream(stream):
            for name, fn in stages.items():
                fn()
                fn()
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    fn()
                self.graphs[name] = graph
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        raw = {name: graph.raw_cuda_graph() for name, graph in self.graphs.items()}

        def ptr(slot):
            return self.ctr[slot:].data_ptr()

        def loop_graph(pair, round_):
            loops = []
            if self.fused:
                loops.append((raw[pair], ptr(CARRY_LOOP), ptr(CARRY_STEP), d.refine_steps + 2))
            loops.append((raw[round_], ptr(REMAINING), ptr(STEP), self.limit))
            return block_graph().LoopGraph(raw["begin"], loops, raw["snapshot"])

        self.tail_blocks = {t: (loop_graph("pair", f"round{t}"), loop_graph(f"pair{t}", "round")) for t in tails}
        self.blocks = {1: _PlannedBlock(self, loop_graph("pair", "round"))}
        self.traced = {}
        m.warm_prefill()

    def _prefill(self, ids, prefix):
        self.model.prefill(ids[:prefix])

    def _temperature(self, slot):
        if not self.decoder.anneals:
            return self.decoder.confidence.confidence_temperature
        return self.t_table.index_select(0, self.ctr[slot:slot + 1]).reshape(())

    def _round(self, first=0):
        """One round; `first`: the block's prompt rows, for the soft embedding of a first block (see _PlannedBlock)."""
        m, b = self.model, self.block
        positions = self.positions[:b]
        torch.add(self.rows[:b], self.start, out=positions)
        sc_rows = ~self.prompt & (self.ctr[STEP] != 0)
        h = m.embed(self.tokens, self.logits, sc_rows, first)
        logits = m.forward(h, positions, m.denoise[1], out=self.logits)
        m.step_stats(logits, self._temperature(STEP), self.stats, self.proposal)
        lk.decide(self.stats, self.proposal, self.tokens, self.prompt, self.accepted, self.pair_tokens[:b],
                  self.pair_prompt[:b], self.ctr, self.decoder, m.config.eos_token_id, self.fused)

    def _pair_round(self, first=0):
        """One fused pair round; `first`: the carried block's prompt rows (see _PlannedBlock)."""
        m, b = self.model, self.block
        positions = self.positions[:2 * b]
        torch.add(self.rows[:2 * b], self.pair_start, out=positions)
        self.pair_sc_rows[:b] = ~self.pair_prompt[:b]
        h = m.embed(self.pair_tokens, self.pair_logits[:b], self.pair_sc_rows, first)
        logits = m.forward(h, positions, m.denoise[2], out=self.pair_logits)
        m.step_stats(logits[:b], self._temperature(CARRY_T), self.pair_stats[:b], self.pair_proposal[:b])
        m.step_stats(logits[b:], confidence_temperature(self.decoder, 1), self.pair_stats[b:], self.pair_proposal[b:])
        lk.decide_pair(self.pair_stats, self.pair_proposal, self.pair_tokens, self.pair_prompt, self.accepted,
                       self.final, self.ctr, self.decoder, m.config.eos_token_id)
        if m.sc is not None:
            lk.carry_save(self.pair_logits, self.pair_stats, self.ctr)
