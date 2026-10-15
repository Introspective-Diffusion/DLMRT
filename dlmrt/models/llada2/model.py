"""LLaDA2.1-mini (llada2_moe): the block forward.

Per layer: pre-norm attention (fused QKV, per-head Q/K RMSNorm, NeoX RoPE on the first `rotary_dim` of 128 dims),
then a dense SwiGLU MLP (layer 0) or a sigmoid-routed MoE (256 SwiGLU experts, grouped top-8, plus one shared
expert). Attention is block-causal on an absolute 32-token grid: every row of block j sees [0, 32 (j + 1)). A block
forward is 32 rows at a block boundary; the prompt prefill runs several whole blocks per forward. Optional
self-conditioning adapter (the ORL checkpoint): the previous round's softmax pushed through the embedding,
RMSNorm, a SiLU-gated MLP, added to the token embedding.

Precision (`LladaModel.load(precision=...)`, default the checkpoint's own): "bf16", or "nvfp4" for a ModelOpt NVFP4
export (W4A4 routed experts, FP8 KV cache; attention, shared expert, layer-0 MLP, router, embeddings and the SC
adapter BF16). nvfp4 runs on Blackwell FP4 tensor cores: FlashInfer's TRT-LLM-gen MoE on SM 10.x, and on SM 12.x
dlmrt's Triton MoE with its roundings (`dlmrt.ops.triton.nvfp4_moe`; bitwise the B200's in all but about 0.2% of
rows). An export read at bf16 runs its experts dequantized.
"""

import json
from pathlib import Path

import deep_gemm
import torch
from safetensors import safe_open
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

from ... import envs
from ...layers.linear import Linear
from ...ops.extensions import block_graph, init_deep_gemm
from ...ops.triton import core as kernels
from ...ops.triton.attention import attention as triton_attention
from ...ops.triton import b200_attention, gemm, nvfp4_moe
from ...ops.triton.moe import moe_gather
from ...quantization import nvfp4
from ...quantization import precision as precision_switch
from . import kernels as lk
from .config import LladaConfig


# The shared expert on a side stream, concurrent with routing and the routed experts (DLMRT_LLADA_OVERLAP=0: inline).
OVERLAP = envs.LLADA_OVERLAP()

# Attention: FA3 where it runs (Hopper), else the Triton kernel (block-causal on the 32-token grid).
ATTENTION = envs.LLADA_ATTENTION()

# Routed-expert GEMMs of decode rounds (32 / 64 rows): "triton" (small-tile grouped GEMMs, SwiGLU in the gate/up
# epilogue; a few rows per expert) or "deepgemm" (masked grouped GEMMs + a SwiGLU kernel, always used for prefill).
MOE_GEMM = envs.LLADA_MOE()

# Cross-rank sum of the row-parallel outputs under TP (see LladaModel.reduce).
REDUCE = envs.LLADA_REDUCE()

# NVFP4 routed experts on SM 12.x: "triton" (dlmrt's, the TRT-LLM-gen MoE's roundings) or "cutlass" (FlashInfer's).
FP4_MOE = envs.LLADA_FP4_MOE()


def _rope_table(config, max_seq_len, device):
    """FP32 [positions, rotary_dim]: cos of the rotary_dim / 2 frequencies, then sin."""
    r = config.rotary_dim
    inv = 1.0 / config.rope_theta ** (torch.arange(0, r, 2, device=device, dtype=torch.float32) / r)
    angle = torch.arange(max_seq_len, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cat((angle.cos(), angle.sin()), dim=-1).contiguous()


def _normalize(weights, config):
    """Key variants of the RL exports: the adapter at top level, and `expert_bias` concatenated per FSDP rank."""
    for key in [k for k in weights if k.startswith("self_conditioning.")]:
        weights["model." + key] = weights.pop(key)
    e = config.num_experts
    for key in [k for k in weights if k.endswith(".expert_bias")]:
        t = weights[key]
        if t.numel() != e:
            copies = t.view(-1, e)
            if not all(torch.equal(copies[0], c) for c in copies[1:]):
                raise ValueError(f"{key}: the concatenated per-rank copies differ")
            weights[key] = copies[0].clone()
    return weights


class LladaModel:
    """Weights BF16 except the router (FP32 weight and bias, as the reference runs it).

    `group`: a `parallel.Group` to shard over, or None for one GPU. Query heads, dense / shared-expert columns and
    the vocabulary (LM head, SC GEMM) are split Megatron-style, the routed experts whole (expert parallelism); the
    four KV heads are replicated over ranks that share one. The router, the embedding lookup and the SC adapter are
    replicated. Each layer ends in two all-reduces (attention output; dense MLP or experts + shared expert) in
    place in symmetric memory, so every rank holds the same residual stream and makes the same decisions."""

    def __init__(self, config, weights, max_seq_len, *, max_rows=256, group=None, precision="bf16"):
        """`max_rows`: the largest forward (prefill chunk), a multiple of the block size. `weights` as read by
        `load`; `precision` "nvfp4" keeps the KV cache in FP8 E4M3 values and runs the routed experts as W4A4."""
        if max_seq_len < max_rows:
            raise ValueError(f"max_seq_len must be at least {max_rows} (graph capture runs a {max_rows}-row prefill chunk)")
        self.config = c = config
        self.max_seq_len = max_seq_len
        self.max_rows = max_rows
        self.group = group
        self.precision = precision
        self.kv_fp8 = precision == "nvfp4"
        major = torch.cuda.get_device_capability(weights["model.word_embeddings.weight"].device)[0]
        # SM 12.x: every BF16 GEMM in cuBLAS's B200 summation order (bitwise the B200's products).
        self.b200 = major == 12
        self.fa3 = ATTENTION == "fa3" or not ATTENTION and major == 9
        # Triton attention tiles: 64 x 64, and 32 x 32 on SM 12.x (the RTX 5090 results). DLMRT_LLADA_ATTENTION=b200
        # on SM 12.x: the Gluon kernel with the B200's summation order (bitwise its 64 x 64 Triton attention).
        self.attention_tiles = (32, 32) if major == 12 else (64, 64)
        self.b200_order_attention = ATTENTION == "b200" and major == 12
        world, rank = (group.world, group.rank) if group else (1, 0)
        hq, hk, d, e = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.num_experts
        if hq % world or c.vocab_size % world or e % world or (hk % world and world % hk) or \
                c.intermediate_size % (64 * world) or c.moe_intermediate_size % (64 * world):
            raise ValueError(f"Heads, vocabulary, experts and FFN columns must split over {world} ranks")
        self.heads, self.kv_heads = hq // world, max(1, hk // world)
        self.local_experts = e // world
        self.expert_lo = rank * self.local_experts
        init_deep_gemm()
        weights = _normalize(dict(weights), c)

        def take(key, dtype=torch.bfloat16):
            return weights.pop(key).to(dtype).contiguous()

        def shard(t, dim):
            return t.chunk(world, dim)[rank].contiguous() if world > 1 else t

        def mlp(prefix, split=True):
            gate, up, down = (take(prefix + n + "_proj.weight") for n in ("gate", "up", "down"))
            if split:
                gate, up, down = shard(gate, 0), shard(up, 0), shard(down, 1)
            return {"gate_up": Linear(torch.cat([gate, up]), False, b200=self.b200),
                    "down": Linear(down, False, b200=self.b200)}

        def qkv(t):
            """This rank's query heads, then the K and V rows of its KV head."""
            q, k, v = t.split([hq * d, hk * d, hk * d])
            if hk % world == 0:
                return torch.cat([shard(q, 0), shard(k, 0), shard(v, 0)])
            j = rank * hk // world
            return torch.cat([shard(q, 0), k[j * d:(j + 1) * d], v[j * d:(j + 1) * d]])

        self.embedding = take("model.word_embeddings.weight")
        self.device, self.dtype = self.embedding.device, self.embedding.dtype
        self.vocab = c.vocab_size // world
        self.vocab_offset = rank * self.vocab
        # Lookups use the full table; the SC GEMM this rank's vocabulary rows, as the LM head.
        self.sc_embedding = self.embedding[self.vocab_offset:self.vocab_offset + self.vocab]
        self.lm_head = shard(take("lm_head.weight"), 0)
        self.final_norm = take("model.norm.weight")
        self.layers = []
        for i in range(c.num_hidden_layers):
            p = f"model.layers.{i}."
            layer = {
                "norm": take(p + "input_layernorm.weight"),
                "qkv": Linear(qkv(take(p + "attention.query_key_value.weight")), False, b200=self.b200),
                "q_norm": take(p + "attention.query_layernorm.weight"),
                "k_norm": take(p + "attention.key_layernorm.weight"),
                "o": Linear(shard(take(p + "attention.dense.weight"), 1), False, b200=self.b200),
                "post_norm": take(p + "post_attention_layernorm.weight"),
                "moe": i >= c.first_k_dense_replace,
            }
            if layer["moe"]:
                fp4 = weights.pop(p + "mlp.experts.fp4", None)
                w13 = w2 = None
                if fp4 is None:
                    w13, w2 = take(p + "mlp.experts.gate_up_proj"), take(p + "mlp.experts.down_proj")
                    if w13.shape[0] != self.local_experts:  # not already sliced at load
                        w13, w2 = (t[self.expert_lo:self.expert_lo + self.local_experts].contiguous()
                                   for t in (w13, w2))
                layer.update(fp4=fp4)
                router = take(p + "mlp.gate.weight", torch.float32)
                # BF16-exact weights (both released checkpoints): the reference's FP32 GEMM rounded to BF16 is a
                # BF16 GEMM with FP32 accumulation; otherwise keep the FP32 GEMM.
                exact = torch.equal(router.to(torch.bfloat16).float(), router)
                layer.update(router=Linear(router.to(torch.bfloat16), False, b200=self.b200) if exact else router,
                             bias=take(p + "mlp.gate.expert_bias", torch.float32),
                             shared=mlp(p + "mlp.shared_experts."), w13=w13, w2=w2)
            else:
                layer.update(mlp(p + "mlp."))
            self.layers.append(layer)
        self.sc = None
        if c.use_self_conditioning:
            s = "model.self_conditioning."
            self.sc = {"norm": take(s + "pre_norm.weight"), **mlp(s, split=False)}
        if weights:
            raise ValueError(f"Unconsumed checkpoint weights: {sorted(weights)[:8]}")
        self.k = torch.zeros(c.num_hidden_layers, 1, max_seq_len, self.kv_heads, d, device=self.device,
                             dtype=self.dtype)
        self.v = torch.zeros_like(self.k)
        self.rope = _rope_table(c, max_seq_len, self.device)
        # Masked expert layout: every local expert has room for all rows of the largest forward; only the rows
        # assigned to it are computed, so one set of buffers serves every forward size.
        k, f, h = c.num_experts_per_tok, c.moe_intermediate_size, c.hidden_size
        # DeepGEMM's SM100 masked grouped GEMM returns garbage below 128 rows per expert.
        dev, m = {"device": self.device}, max(max_rows, 128)
        fp4 = next((layer["fp4"] for layer in self.layers if layer.get("fp4") is not None), None)
        if fp4 is None:
            self.moe_in = torch.zeros(self.local_experts, m, h, dtype=self.dtype, **dev)
            self.moe_mid = torch.zeros(self.local_experts, m, 2 * f, dtype=self.dtype, **dev)
            self.moe_act = torch.zeros(self.local_experts, m, f, dtype=self.dtype, **dev)
            self.moe_out = torch.zeros(self.local_experts, m, h, dtype=self.dtype, **dev)
        elif fp4.get("backend") == "triton":
            self.fp4_buffers = nvfp4_moe.Buffers(max_rows, h, f, self.local_experts, k, self.device)
            self.expert_map = torch.zeros(e, max_rows, dtype=torch.int32, **dev)
        elif fp4.get("backend") == "cutlass":
            # The CUTLASS MoE's tactics are tuned up to max_rows before any graph is captured.
            self.routed_out = torch.zeros(max_rows, h, dtype=self.dtype, **dev)
            nvfp4.cutlass_tune(fp4, c.num_experts, max_rows, h, k, self.device, activation="swiglu")
        else:
            # TRT-LLM-gen's tactics before any graph is captured: FlashInfer's autotuner profiles every power-of-two
            # row bucket up to max_rows, which covers decode rounds (32 rows), pair rounds (64) and prefill chunks.
            self.packed = torch.zeros(max_rows, k, dtype=torch.int32, **dev)
            self.routed_out = torch.zeros(max_rows, h, dtype=self.dtype, **dev)
            nvfp4.tune(fp4, c.num_experts, self.expert_lo, f, max_rows, h, k, self.device, activation="swiglu")
        self.expert_ids = torch.zeros(max_rows, k, dtype=torch.int32, **dev)
        self.expert_slots = torch.zeros(max_rows, k, dtype=torch.int32, **dev)
        self.expert_weights = torch.zeros(max_rows, k, dtype=torch.float32, **dev)
        self.expert_counts = torch.zeros(e, dtype=torch.int32, **dev)
        self.shared_out = torch.zeros(max_rows, h, dtype=self.dtype, **dev)
        self.side = torch.cuda.Stream(self.device) if OVERLAP else None
        # Row-parallel outputs, one symmetric buffer per forward size (allocated collectively, in order).
        sizes = range(c.block_size, max_rows + 1, c.block_size)
        self.partials = {rows: group.symmetric(f"llada.partial{rows}", (rows, h)) for rows in sizes} if group else {}
        self.reduced = {rows: torch.empty(rows, h, dtype=self.dtype, **dev) for rows in sizes} if group else {}
        block_graph()

    @classmethod
    def load(cls, path, max_seq_len=16384, *, device=None, group=None, precision=None):
        """Experts are stacked into [E, 2F, H] (gate | up) and [E, H, F] as they are read; under expert
        parallelism a rank reads only its own experts. An NVFP4 export's experts come from their per-expert NVFP4
        tensors: as the layout of this GPU's FlashInfer NVFP4 MoE (nvfp4), or dequantized to the BF16 stacks (bf16).
        `precision`: see the module docstring."""
        path = Path(path)
        config = LladaConfig.load(path)
        device = device or (group.device if group else "cuda:0")
        precision = precision_switch.resolve(path, precision, device, backends=("trtllm", "cutlass"))
        fp4_experts = nvfp4.trtllm_experts
        if precision == "nvfp4" and nvfp4.backend(device) == "cutlass":  # backend() imports FlashInfer's MoE
            fp4_experts = nvfp4.triton_experts if FP4_MOE == "triton" else nvfp4.cutlass_experts
        world, rank = (group.world, group.rank) if group else (1, 0)
        e, f, h = config.num_experts // world, config.moe_intermediate_size, config.hidden_size
        lo = rank * e
        index = path / "model.safetensors.index.json"
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else \
            [p.name for p in sorted(path.glob("*.safetensors"))]
        weights, stacks, parts = {}, {}, {}
        fp4_checkpoint = precision_switch.checkpoint_precision(path) == "nvfp4"
        with torch.cuda.device(device):
            for shard in shards:
                with safe_open(path / shard, framework="pt", device="cpu") as source:
                    for key in source.keys():  # noqa: SIM118 - safe_open is not a mapping.
                        expert = nvfp4.EXPERT_KEY.match(key) if fp4_checkpoint else None
                        if expert:
                            if lo <= int(expert[2]) < lo + e:
                                parts[(int(expert[1]), int(expert[2]), expert[3], expert[4])] = source.get_tensor(key)
                            continue
                        if ".mlp.experts." not in key:
                            keep = torch.float32 if key.endswith((".mlp.gate.weight", ".expert_bias")) \
                                else torch.bfloat16
                            weights[key] = source.get_tensor(key).to(device=device, dtype=keep)
                            continue
                        prefix, rest = key.split(".mlp.experts.")
                        expert, proj, _ = rest.split(".")
                        if not lo <= int(expert) < lo + e:
                            continue
                        if prefix not in stacks:
                            stacks[prefix] = (torch.empty(e, 2 * f, h, device=device, dtype=torch.bfloat16),
                                              torch.empty(e, h, f, device=device, dtype=torch.bfloat16), [0])
                        w13, w2, seen = stacks[prefix]
                        j = int(expert) - lo
                        target = {"gate_proj": w13[j, :f], "up_proj": w13[j, f:], "down_proj": w2[j]}[proj]
                        target.copy_(source.get_tensor(key))
                        seen[0] += 1
            for prefix, (w13, w2, seen) in stacks.items():
                if seen[0] != 3 * e:
                    raise ValueError(f"{prefix}: {seen[0]} of {3 * e} expert tensors")
                weights[prefix + ".mlp.experts.gate_up_proj"] = w13
                weights[prefix + ".mlp.experts.down_proj"] = w2
            for i in sorted({layer for layer, *_ in parts}):
                if sum(1 for k in parts if k[0] == i) != 3 * 4 * e:
                    raise ValueError(f"layer {i}: incomplete NVFP4 expert tensors")
                p = f"model.layers.{i}.mlp.experts."
                if precision == "nvfp4":
                    weights[p + "fp4"] = fp4_experts(parts, i, lo, e, device)
                    continue
                weights[p + "gate_up_proj"], weights[p + "down_proj"] = nvfp4.expert_weights(parts, i, lo, e, device)
            return cls(config, weights, max_seq_len, group=group, precision=precision)

    def partial(self, rows):
        """Where a row-parallel product of `rows` rows goes: symmetric memory under TP, else a new tensor."""
        if self.group is None:
            return torch.empty(rows, self.config.hidden_size, device=self.device, dtype=self.dtype)
        return self.partials[rows]

    def reduce(self, partial, site, chain):
        """Sum over ranks (bitwise identical on every rank): NVLink multicast in place (REDUCE=multimem), every
        rank summing all peers' partials in rank order into its own buffer (REDUCE=one_shot), or dlmrt's barrier-free
        push / owner-sum / multicast kernel (REDUCE=fused, `parallel.Group.reduce`). The fused kernel re-arms the
        previous output of its chain (over that output's rows), so every chain must cycle through the same sites: one
        chain per forward size (`chain` = its rows), the SC reduction (the carried block's rows only, in a pair round)
        on the chain of the forward that follows it."""
        if self.group is None:
            return partial
        if REDUCE == "fused":
            return self.group.reduce(partial, site=f"{chain}.{site}", chain=chain)
        if REDUCE == "one_shot":
            out = self.reduced[partial.shape[0]]
            torch.ops.symm_mem.one_shot_all_reduce_out(partial, "sum", self.group.name, out)
            return out
        return self.group.all_reduce_(partial)

    @staticmethod
    def mlp(x, weights, round_silu=False, out=None):
        return weights["down"](kernels.swiglu(weights["gate_up"](x), round_silu=round_silu), out=out)

    def moe(self, x, layer, out):
        """This rank's routed experts (scaled) + its shared-expert columns, BF16 [rows, hidden], into `out`."""
        c = self.config
        rows, lo, n = x.shape[0], self.expert_lo, self.local_experts
        ids, slots, weights = self.expert_ids[:rows], self.expert_slots[:rows], self.expert_weights[:rows]
        expected = max(1, rows * c.num_experts_per_tok // c.num_experts)
        router = layer["router"]
        logits = router(x) if isinstance(router, Linear) else torch.mm(x.float(), router.t())
        backend = layer["fp4"].get("backend") if layer.get("fp4") is not None else None
        cutlass = backend == "cutlass"
        # SM 12.x: the routing weights scaled and rounded to BF16, the values the TRT-LLM-gen MoE is packed with.
        if cutlass:
            lk.route(logits, layer["bias"], ids, weights, None, None, c.num_experts_per_tok, c.n_group,
                     c.topk_group, scale=c.routed_scaling_factor)
        elif backend == "triton":
            lk.route(logits, layer["bias"], ids, weights, slots, self.expert_counts, c.num_experts_per_tok,
                     c.n_group, c.topk_group, scale=c.routed_scaling_factor, amap=self.expert_map)
        else:
            lk.route(logits, layer["bias"], ids, weights, slots, self.expert_counts, c.num_experts_per_tok,
                     c.n_group, c.topk_group)
        counts = self.expert_counts[lo:lo + n]
        shared = self.shared_out[:rows]
        if self.side is not None:  # the shared expert runs beside routing and the expert GEMMs
            main = torch.cuda.current_stream()
            self.side.wait_stream(main)
            with torch.cuda.stream(self.side):
                self.mlp(x, layer["shared"], out=shared)
        else:
            self.mlp(x, layer["shared"], out=shared)
        if backend == "triton":
            # W4A4 on SM 12.x FP4 tensor cores; the shared expert joins in the final sum.
            if self.side is not None:
                main.wait_stream(self.side)
            return nvfp4_moe.moe(x, ids, weights, self.expert_map, self.expert_counts, layer["fp4"], self.fp4_buffers,
                                 out, shared=shared)
        if cutlass:
            # W4A4 on SM 12.x FP4 tensor cores (FlashInfer's CUTLASS MoE).
            routed = nvfp4.cutlass_moe(x, ids, weights, layer["fp4"], self.routed_out[:rows], c.num_experts,
                                       self.max_rows, activation="swiglu")
            if self.side is not None:
                main.wait_stream(self.side)
            return torch.add(routed, shared, out=out)
        if layer.get("fp4") is not None:
            # W4A4 on FP4 tensor cores; the routing weights (scaled) travel packed with the expert ids as BF16.
            packed = self.packed[:rows]
            packed.copy_(nvfp4.pack(ids, weights * c.routed_scaling_factor))
            routed = nvfp4.trtllm_moe(x, packed, layer["fp4"], self.routed_out[:rows], c.num_experts, lo,
                                      c.moe_intermediate_size, activation="swiglu")
            if self.side is not None:
                main.wait_stream(self.side)
            return torch.add(routed, shared, out=out)
        moe_gather(x, ids, slots, self.moe_in, lo)
        if MOE_GEMM == "triton" and rows <= 2 * c.block_size:  # decode rounds; prefill chunks have many rows per expert
            lk.moe_gate_up(self.moe_in, layer["w13"], self.moe_act, self.expert_counts, lo)
            lk.moe_down(self.moe_act, layer["w2"], self.moe_out, self.expert_counts, lo)
        else:
            deep_gemm.m_grouped_bf16_gemm_nt_masked(self.moe_in, layer["w13"], self.moe_mid, counts, expected)
            lk.moe_swiglu(self.moe_mid, ids, slots, self.moe_act, lo)
            deep_gemm.m_grouped_bf16_gemm_nt_masked(self.moe_act, layer["w2"], self.moe_out, counts, expected)
        if self.side is not None:
            main.wait_stream(self.side)
        return lk.moe_combine(self.moe_out, ids, slots, weights, out, c.routed_scaling_factor, shared=shared, lo=lo)

    def self_condition(self, probs, chain=None):
        """SC delta from the previous round's probabilities over this rank's vocabulary columns (zero rows give a
        zero delta); the adapter runs replicated on the summed soft embedding. `chain`: the rows of the forward
        that follows (see reduce)."""
        c = self.config
        rows = probs.shape[0]
        if self.b200:  # cuBLAS's B200 split of this K = vocabulary GEMM
            soft = gemm.mm(probs, self.sc_embedding, self.partial(rows), split=9 if rows <= 96 else 8 if rows <= 128
                           else 4, nn=True)
        else:
            soft = torch.mm(probs, self.sc_embedding, out=self.partial(rows))
        signal = self.reduce(soft, "sc", chain or rows)
        x = kernels.rmsnorm(signal, self.sc["norm"], c.rms_norm_eps, scale=c.self_conditioning_prenorm_scale)
        return self.mlp(x, self.sc, round_silu=True)

    def schedule(self, rows, end, seqlens, splits):
        """FA3 tile schedule of one forward, shared by every layer (one block, or one batch entry per block)."""
        if not self.fa3:
            return None
        b = self.config.block_size
        batch, lengths = (1, end) if seqlens is None else (rows // b, seqlens)
        return get_scheduler_metadata(batch, b if seqlens is not None else rows, self.max_seq_len, self.heads,
                                      self.kv_heads, self.config.head_dim, lengths, qkv_dtype=self.dtype,
                                      causal=False, num_splits=splits)

    def attend(self, i, q, start, end, seqlens, batch_idx, splits, schedule):
        if not self.fa3:
            # One block sees [0, end); several blocks are block-causal up to the last block's end.
            d = self.config.head_dim
            if self.b200_order_attention:
                return b200_attention.attention(q, self.k[i], self.v[i], start, end if seqlens is None else
                                                seqlens[-1:], scale=d ** -0.5,
                                                grid=0 if seqlens is None else self.config.block_size)
            bm, bn = self.attention_tiles
            return triton_attention(q, self.k[i], self.v[i], start, end if seqlens is None else seqlens[-1:], False,
                                    scale=d ** -0.5, grid=0 if seqlens is None else self.config.block_size, dv=d,
                                    block_m=bm, block_n=bn, num_warps=4)
        if seqlens is None:
            return flash_attn_with_kvcache(q, self.k[i], self.v[i], cache_seqlens=end, causal=False,
                                           num_splits=splits, scheduler_metadata=schedule)
        b = self.config.block_size
        return flash_attn_with_kvcache(q.view(-1, b, *q.shape[2:]), self.k[i], self.v[i], cache_seqlens=seqlens,
                                       cache_batch_idx=batch_idx, causal=False, num_splits=splits,
                                       scheduler_metadata=schedule)

    def forward(self, h, start, end, *, splits=1, seqlens=None, batch_idx=None):
        """h: [rows, hidden] block input at [start, start + rows), consumed in place. Returns the final-norm states.

        One block (32 rows) sees [0, end). Several blocks (the prefill) pass `seqlens` (each block's end) and
        `batch_idx` (zeros): each block becomes one FA3 batch entry over the shared arena, i.e. block-causal."""
        c = self.config
        rows, eps = h.shape[0], c.rms_norm_eps
        hq, hk, d = self.heads, self.kv_heads, c.head_dim
        schedule = self.schedule(rows, end, seqlens, splits)
        x = kernels.rmsnorm(h, self.layers[0]["norm"], eps)
        for i, layer in enumerate(self.layers):
            q = torch.empty((1, rows, hq, d), device=h.device, dtype=h.dtype)
            lk.qk_rope(layer["qkv"](x), layer["q_norm"], layer["k_norm"], self.rope, start, q, self.k[i], self.v[i],
                       hq, hk, d, c.rotary_dim, eps, kv_fp8=self.kv_fp8)
            a = self.attend(i, q, start, end, seqlens, batch_idx, splits, schedule).reshape(rows, hq * d)
            o = layer["o"](a, out=self.partial(rows))
            x = kernels.rmsnorm(h, layer["post_norm"], eps, delta=self.reduce(o, f"{i}.attn", rows))
            y = self.moe(x, layer, self.partial(rows)) if layer["moe"] else self.mlp(x, layer, out=self.partial(rows))
            w = self.layers[i + 1]["norm"] if i + 1 < len(self.layers) else self.final_norm
            x = kernels.rmsnorm(h, w, eps, delta=self.reduce(y, f"{i}.ffn", rows))
        return x

    def logits(self, x, out=None):
        """This rank's vocabulary columns of the BF16 LM-head logits."""
        if self.b200:
            return gemm.mm(x, self.lm_head, out)
        return torch.mm(x, self.lm_head.t(), out=out)
