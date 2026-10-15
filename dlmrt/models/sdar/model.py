"""Dense SDAR transformer over one contiguous KV arena, fixed at one 256-row block per forward."""

from pathlib import Path

import torch
from safetensors import safe_open
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

from ... import envs
from ...distributed import parallel
from ...layers.linear import Linear
from ...ops.extensions import block_graph, init_deep_gemm, ogud, tk
from ...ops.triton import core as kernels
from ...ops.triton.common import PDL
from .config import ModelConfig


# FA3 over 64 (head, row-tile) CTAs leaves half of the SMs idle once the prefix
# is long; two KV splits fill them at the cost of a combine kernel.
SPLIT_CONTEXT = envs.SDAR_SPLIT_CONTEXT()

# BF16 MLP of a 256-row block: "fused" (gate_up + SwiGLU kernel, then DeepGEMM down) or "separate" (DeepGEMM,
# Triton SwiGLU, DeepGEMM); bit-identical. The fused kernel runs one 80-column task per CTA, so it is used only
# when that fills the GPU (one GPU, not the per-rank FFN shard under TP).
MLP_KERNEL = envs.SDAR_MLP()

# BF16 post-attention chain of a 256-row block on one GPU (O projection, residual RMSNorm, gate_up + SwiGLU, down) as
# one persistent launch (dlmrt/ops/csrc/ogud.cu), bit-identical to the separate kernels. Built for SDAR-4B's shapes on a
# 132-SM H100.
OGUD = envs.SDAR_OGUD()

PROJECTIONS = ("qkv", "o", "gate_up", "down", "sc")

# FP8 on O saves ~2.6 us per layer but needs a separate activation-quantization kernel (~2 us);
# the SC adapter runs once per round. Both stay BF16 by default.
FP8_DEFAULT = ("qkv", "gate_up", "down")


def ogud_fits(config, device):
    """Whether the post-attention megakernel (ogud.cu, both variants) is built for this model and GPU: SDAR-4B's
    shapes on a 132-SM H100."""
    shapes = (config.hidden_size, config.num_attention_heads * config.head_dim, config.intermediate_size)
    return torch.cuda.get_device_properties(device).multi_processor_count == 132 and shapes == (2560, 4096, 9728)


class Model:
    def __init__(self, config, weights, max_seq_len, *, fp8=False, group=None):
        """`fp8`: False, True (FP8_DEFAULT) or a subset of PROJECTIONS; LM head and SC GEMM stay BF16.

        `group`: a `parallel.Group` to shard over (heads, FFN columns, vocabulary), or None for one GPU.
        """
        self.config = c = config
        self.max_seq_len = max_seq_len
        self.group = group
        world = group.world if group else 1
        self.fp8 = frozenset(FP8_DEFAULT if fp8 is True else fp8 or ())
        if not self.fp8 <= set(PROJECTIONS):
            raise ValueError(f"fp8 projections must be a subset of {PROJECTIONS}")
        if c.num_key_value_heads % world or c.vocab_size % world:
            raise ValueError(f"KV heads and vocabulary must split over {world} ranks")
        self.heads, self.kv_heads = c.num_attention_heads // world, c.num_key_value_heads // world
        # The fused reduction writes BF16 norms; FP8 GEMM inputs fall back to all-reduce + RMSNorm everywhere
        # (mixing would leave the replicated path reading residual rows only their owners updated).
        self.fused = bool(group) and parallel.FUSED and not self.fp8 & {"qkv", "gate_up", "sc"}
        init_deep_gemm()
        weights = dict(weights)

        def take(key, dim=None):
            """A checkpoint tensor; `dim` is the dimension this rank takes its shard of."""
            t = weights.pop(key)
            return group.shard(t, dim) if group and dim is not None else t

        def linear(t, name):
            if name in self.fp8 and t.shape[1] % kernels.GROUP:
                raise ValueError(f"FP8 {name}: {t.shape[1]} input features per rank is not a multiple of 128")
            return Linear(t, name in self.fp8)

        def mlp(prefix, gate_up, down):
            return {"gate_up": linear(torch.cat([take(prefix + "gate_proj.weight", 0),
                                                 take(prefix + "up_proj.weight", 0)]), gate_up),
                    "down": linear(take(prefix + "down_proj.weight", 1), down)}

        self.embedding = take("model.embed_tokens.weight")
        weights.pop("lm_head.weight", None)
        self.device, self.dtype = self.embedding.device, self.embedding.dtype
        if self.dtype != torch.bfloat16:
            raise ValueError("BF16 checkpoints only")
        # Tied embedding: lookups use the full table, the LM head and SC GEMM this rank's vocabulary rows.
        self.vocab = c.vocab_size // world
        self.vocab_offset = group.rank * self.vocab if group else 0
        self.lm_head = self.embedding[self.vocab_offset:self.vocab_offset + self.vocab]
        self.final_norm = take("model.norm.weight")
        self.layers = []
        for i in range(c.num_hidden_layers):
            p = f"model.layers.{i}."
            qkv = torch.cat([take(p + f"self_attn.{n}_proj.weight", 0) for n in "qkv"])
            self.layers.append({
                "norm": take(p + "input_layernorm.weight"),
                "qkv": linear(qkv, "qkv"),
                "q_norm": take(p + "self_attn.q_norm.weight"),
                "k_norm": take(p + "self_attn.k_norm.weight"),
                "o": linear(take(p + "self_attn.o_proj.weight", 1), "o"),
                "post_norm": take(p + "post_attention_layernorm.weight"),
                **mlp(p + "mlp.", "gate_up", "down"),
            })
        self.sc = None
        if c.sc:
            self.sc = {"norm": take("self_conditioning.pre_norm.weight"), **mlp("self_conditioning.", "sc", "sc")}
        if weights:
            raise ValueError(f"Unconsumed checkpoint weights: {sorted(weights)[:8]}")
        shape = (c.num_hidden_layers, 1, max_seq_len, self.kv_heads, c.head_dim)
        self.k = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v = torch.zeros_like(self.k)
        d = c.head_dim
        inv = 1.0 / c.rope_theta ** (torch.arange(0, d, 2, device=self.device).float() / d)
        angle = torch.arange(max_seq_len, device=self.device).float()[:, None] * inv[None, :]
        self.rope = torch.cat((angle.cos(), angle.sin()), dim=-1).contiguous()
        block_graph()
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.fused_mlp = MLP_KERNEL == "fused" and "gate_up" not in self.fp8 and "down" not in self.fp8
        self.fused_mlp_min_f = 80 * sms * 3 // 4
        if self.fused_mlp:
            tk()
        self.ogud = OGUD and group is None and not self.fp8 & {"o", "gate_up", "down"} and ogud_fits(c, self.device)
        if self.ogud:
            ogud()
            # One counter slot per layer: layer i re-arms layer i + 1's slot, whose last user is complete by then.
            self.post_counters = torch.zeros(c.num_hidden_layers, 512, device=self.device, dtype=torch.int32)

    @classmethod
    def load(cls, path, max_seq_len=65536, *, device=None, fp8=False, group=None):
        path = Path(path)
        device = device or (group.device if group else "cuda:0")
        weights = {}
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as source:
                for key in source.keys():  # noqa: SIM118 - safe_open is not a mapping.
                    weights[key] = source.get_tensor(key).to(device=device, dtype=torch.bfloat16)
        with torch.cuda.device(device):
            return cls(ModelConfig.load(path), weights, max_seq_len, fp8=fp8, group=group)

    def partial(self, linear, x, scales=None, site=None):
        """Row-parallel projection: this rank's partial product (symmetric memory under TP), not yet summed.

        `site` is the call site of the reduction that consumes it.
        """
        out = self.group.partial(site, (x.shape[0], linear.n)) if self.group else None
        return linear(x, scales, out=out)

    def norm(self, h, weight, consumer=None, partial=None, scale=1.0, site=None):
        """Input of `consumer` (BF16, or FP8 if it is an FP8 Linear): rmsnorm(h [+= partial summed over ranks]).

        Under TP with the fused reduction only this rank's rows of `h` are updated; every later read of
        `h` goes through here, so the residual stream stays sequence-parallel until the next embedding.
        `site` names the call site, whose output buffer the fused reduction owns.
        """
        eps = self.config.rms_norm_eps
        x, xs = consumer.input(h.shape[0], h.device) if consumer is not None else (None, None)
        if partial is not None and self.group is not None:
            if self.fused:
                return self.group.reduce_norm(partial, h, weight, eps, scale, site=site), None
            self.group.all_reduce_(partial)
        return kernels.rmsnorm(h, weight, eps, delta=partial, scale=scale, out=x, scales=xs), xs

    def mlp(self, x, xs, weights, site, round_silu=False):
        """This rank's partial down(SwiGLU(x @ gate_up)); shared by the layers and the SC adapter."""
        gate_up, down = weights["gate_up"], weights["down"]
        f = gate_up.n // 2
        if self.fused_mlp and x.shape[0] == 256 and f >= self.fused_mlp_min_f:
            a = torch.empty(x.shape[0], f, device=x.device, dtype=torch.bfloat16)
            tk().gate_up_swiglu(x, gate_up.weight, a, PDL, 1, 2, round_silu)
            return self.partial(down, a, None, site)
        a, as_ = down.input(x.shape[0], x.device)
        a = kernels.swiglu(gate_up(x, xs), out=a, scales=as_, round_silu=round_silu)
        return self.partial(down, a, as_, site)

    def post_attention(self, layer, a, h, i):
        """Layer i's down(SwiGLU(rmsnorm(h += a @ o) @ gate_up)) in one launch (one GPU, BF16, 256 rows)."""
        o, x, out = torch.empty_like(h), torch.empty_like(h), torch.empty_like(h)
        y = torch.empty(h.shape[0], layer["down"].k, device=h.device, dtype=h.dtype)
        ogud().post_attention(a, h, layer["o"].weight, layer["post_norm"], self.config.rms_norm_eps, 1.0,
                              layer["gate_up"].weight, layer["down"].weight, o, x, y, out, self.post_counters, i,
                              PDL)
        return out

    def reduce(self, partial, site):
        if self.group is None:
            return partial
        return self.group.reduce(partial, site=site) if self.fused else self.group.all_reduce_(partial)

    def self_condition(self, probs):
        """SC delta from the previous round's probabilities over this rank's vocabulary rows."""
        c, sc = self.config, self.sc
        if self.group is None:
            return self.sc_delta(torch.mm(probs, self.lm_head))
        signal = torch.mm(probs, self.lm_head, out=self.group.partial("sc.norm", (probs.shape[0], c.hidden_size)))
        if self.fused:
            x, xs = self.group.reduce_norm(signal, None, sc["norm"], c.rms_norm_eps,
                                           c.self_conditioning_prenorm_scale, site="sc.norm"), None
        else:
            x, xs = self.norm(self.group.all_reduce_(signal), sc["norm"], sc["gate_up"],
                              scale=c.self_conditioning_prenorm_scale)
        return self.reduce(self.mlp(x, xs, sc, "sc.delta", round_silu=True), "sc.delta")

    def sc_delta(self, signal):
        """Self-conditioning adapter on a full (replicated) soft embedding."""
        c, sc = self.config, self.sc
        x, xs = self.norm(signal, sc["norm"], sc["gate_up"], scale=c.self_conditioning_prenorm_scale)
        return self.reduce(self.mlp(x, xs, sc, "sc.delta", round_silu=True), "sc.delta")

    def forward(self, h, start, end, *, causal, splits):
        """h: [rows, hidden] residual stream, consumed in place. Returns the final-norm hidden states.

        Writes this block's K/V at [start, start + rows); attention covers [0, end).
        """
        c = self.config
        rows = h.shape[0]
        # One FA3 tile schedule per forward, shared by every layer.
        schedule = get_scheduler_metadata(1, rows, self.max_seq_len, self.heads, self.kv_heads,
                                          c.head_dim, end, qkv_dtype=self.dtype, causal=causal, num_splits=splits)
        x, xs = self.norm(h, self.layers[0]["norm"], self.layers[0]["qkv"])
        for i, layer in enumerate(self.layers):
            qkv = layer["qkv"](x, xs)
            q = torch.empty((1, rows, self.heads, c.head_dim), device=h.device, dtype=h.dtype)
            kernels.qk_rope(qkv, layer["q_norm"], layer["k_norm"], self.rope, start, q, self.k[i], self.v[i],
                            self.heads, self.kv_heads, c.rms_norm_eps)
            a = flash_attn_with_kvcache(q, self.k[i], self.v[i], cache_seqlens=end, causal=causal,
                                        num_splits=splits, scheduler_metadata=schedule).view(rows, -1)
            site = f"{i + 1}.input" if i + 1 < len(self.layers) else "final"
            if self.ogud and rows == 256:
                partial = self.post_attention(layer, a, h, i)
            else:
                if layer["o"].fp8:
                    a8, as_ = layer["o"].input(rows, h.device)
                    partial = self.partial(layer["o"], kernels.quant(a, a8, as_), as_, f"{i}.post")
                else:
                    partial = self.partial(layer["o"], a, site=f"{i}.post")
                x, xs = self.norm(h, layer["post_norm"], layer["gate_up"], partial, site=f"{i}.post")
                partial = self.mlp(x, xs, layer, site)
            if i + 1 < len(self.layers):
                x, xs = self.norm(h, self.layers[i + 1]["norm"], self.layers[i + 1]["qkv"], partial, site=site)
        # Every forward ends here, so the next graph's first reduction always re-arms this site's buffer.
        return self.norm(h, self.final_norm, partial=partial, site="final")[0]
