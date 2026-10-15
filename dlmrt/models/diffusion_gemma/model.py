"""DiffusionGemma (26B-A4B): weights, KV cache and the 256-row forward.

Per layer: sandwich RMSNorms, attention on sliding (window 1024, head_dim 256) or full (head_dim 512, value =
unweighted norm of the key projection) layers, a dense GeGLU MLP in parallel with 128 top-8 GeGLU experts,
then the layer scalar. The prompt encoder and the canvas decoder share every weight; the encoder is causal
(sliding layers use a causal window), the canvas is bidirectional over [block start - (W - 1), block end) on
sliding layers and [0, block end) on full layers.
"""

import torch

from ... import envs
from ...layers.linear import Linear
from ...ops.extensions import block_graph, init_deep_gemm
from ...quantization import precision as precision_switch
from . import kernels as gk
from . import precision as precisions
from .attention import Attention
from .config import GemmaConfig
from .experts import CutlassExperts, DeepGemmExperts, TrtllmExperts

OVERLAP = envs.DG_OVERLAP()
# NVFP4 experts on FP4 tensor cores: the post-attention kernel writes their input quantized (no separate pass).
FUSED_QUANT = envs.DG_FUSED_QUANT()


def _rope_table(inv_freq, max_seq_len):
    """cos/sin halves [positions, D/2] in BF16, as the reference casts them."""
    angle = torch.arange(max_seq_len, device=inv_freq.device).float()[:, None] * inv_freq[None, :]
    return angle.cos().to(torch.bfloat16).contiguous(), angle.sin().to(torch.bfloat16).contiguous()


class GemmaModel:
    def __init__(self, config, weights, max_seq_len, group=None, precision="bf16"):
        """`group`: a `parallel.Group` to shard over, or None for one GPU. Attention heads, dense-MLP columns and
        the vocabulary are split Megatron-style, experts are split whole (expert parallelism); the full layers'
        two KV heads are replicated when there are more ranks than KV heads.

        `weights` as read by `precision.read_weights`; `precision` "nvfp4" keeps the KV cache in FP8 E4M3 values and
        runs the experts as W4A4 (precision.py)."""
        self.config = c = config
        self.max_seq_len = max_seq_len
        self.group = group
        self.precision = precision
        self.kv_fp8 = precision == "nvfp4"
        world, rank = (group.world, group.rank) if group else (1, 0)
        if c.num_attention_heads % world or c.num_key_value_heads % world or c.vocab_size % world or \
                c.num_experts % world or world % c.num_global_key_value_heads and \
                c.num_global_key_value_heads % world:
            raise ValueError(f"Heads, vocabulary and experts must split over {world} ranks")
        init_deep_gemm()
        weights = dict(weights)
        # Raw FSDP exports name the shared transformer stack after the encoder (the decoder shares every weight).
        for key in [k for k in weights if k.startswith("model.encoder.language_model.")]:
            alias = "model.decoder." + key[len("model.encoder.language_model."):]
            if alias not in weights:
                weights[alias] = weights.pop(key)

        def take(key):
            return weights.pop(key)

        def shard(t, dim):
            return t.chunk(world, dim)[rank].contiguous() if world > 1 else t

        def heads(t, total, head_dim):
            """This rank's heads of a [total * head_dim, hidden] projection (one replicated head if total < world)."""
            if total % world == 0:
                return shard(t, 0)
            h = rank * total // world
            return t[h * head_dim:(h + 1) * head_dim]

        f = c.intermediate_size
        # Per-rank dense-MLP widths must be multiples of 64 (DeepGEMM K); zero columns add GeGLU(0) * down = 0.
        f_pad = f if f % (64 * world) == 0 else -(-f // (64 * world)) * 64 * world

        def mlp(gate, up, down):
            if f_pad != f:
                gate, up = (torch.nn.functional.pad(t, (0, 0, 0, f_pad - f)) for t in (gate, up))
                down = torch.nn.functional.pad(down, (0, f_pad - f))
            return Linear(torch.cat([shard(gate, 0), shard(up, 0)]), False), Linear(shard(down, 1), False)

        self.embedding = take("model.decoder.embed_tokens.weight")
        self.device, self.dtype = self.embedding.device, self.embedding.dtype
        if self.dtype != torch.bfloat16:
            raise ValueError("BF16 checkpoints only")
        # Tied embedding: lookups use the full table, the LM head and SC GEMM this rank's vocabulary rows.
        self.vocab = c.vocab_size // world
        self.vocab_offset = rank * self.vocab
        self.lm_head = self.embedding[self.vocab_offset:self.vocab_offset + self.vocab]
        self.embed_scale = float(torch.tensor(c.hidden_size ** 0.5, dtype=torch.bfloat16))
        lm_head = weights.pop("lm_head.weight", None)
        if lm_head is not None and not torch.equal(lm_head, self.embedding):
            raise ValueError("lm_head.weight differs from the tied embedding")
        self.final_norm = take("model.decoder.norm.weight")
        self.local_experts = c.num_experts // world
        self.expert_lo = rank * self.local_experts
        self.layers = []
        for i, kind in enumerate(c.layer_types):
            p = f"model.decoder.layers.{i}."
            sliding = kind == "sliding_attention"
            d = c.head_dim if sliding else c.global_head_dim
            kv = c.num_key_value_heads if sliding else c.num_global_key_value_heads
            parts = [heads(take(p + "self_attn.q_proj.weight"), c.num_attention_heads, d),
                     heads(take(p + "self_attn.k_proj.weight"), kv, d)]
            if sliding:
                parts.append(heads(take(p + "self_attn.v_proj.weight"), kv, d))
            scalar = take(p + "layer_scalar")
            encoder_scalar = weights.pop(f"model.encoder.language_model.layers.{i}.layer_scalar", scalar)
            if not torch.equal(encoder_scalar, scalar):
                raise ValueError(f"Layer {i}: distinct encoder/decoder layer scalars are not implemented")
            gate_up, down = mlp(take(p + "mlp.gate_proj.weight"), take(p + "mlp.up_proj.weight"),
                                take(p + "mlp.down_proj.weight"))
            fp4 = weights.pop(p + "experts.fp4", None)
            w13 = w2 = None
            if fp4 is None:
                w13, w2 = take(p + "experts.gate_up_proj"), take(p + "experts.down_proj")
                if w13.shape[0] != self.local_experts:  # not already sliced at load
                    w13, w2 = (t[self.expert_lo:self.expert_lo + self.local_experts].contiguous() for t in (w13, w2))
            self.layers.append({
                "sliding": sliding,
                "heads": c.num_attention_heads // world,
                "kv_heads": max(1, kv // world),
                "head_dim": d,
                "norm": take(p + "input_layernorm.weight"),
                "qkv": Linear(torch.cat(parts), False),
                "q_norm": take(p + "self_attn.q_norm.weight"),
                "k_norm": take(p + "self_attn.k_norm.weight"),
                "o": Linear(shard(take(p + "self_attn.o_proj.weight"), 1), False),
                "post_attn": take(p + "post_attention_layernorm.weight"),
                "pre_mlp": take(p + "pre_feedforward_layernorm.weight"),
                "pre_moe": take(p + "pre_feedforward_layernorm_2.weight"),
                "post_mlp": take(p + "post_feedforward_layernorm_1.weight"),
                "post_moe": take(p + "post_feedforward_layernorm_2.weight"),
                "post_ffn": take(p + "post_feedforward_layernorm.weight"),
                "gate_up": gate_up,
                "down": down,
                "router": Linear(take(p + "router.proj.weight"), False),
                "router_scale": take(p + "router.scale"),
                "expert_scale": take(p + "router.per_expert_scale"),
                "w13": w13,
                "w2": w2,
                "fp4": fp4,
                "scalar": float(scalar.float()),
            })
        s = "model.decoder.self_conditioning."
        self.sc = {"norm": take(s + "pre_norm.weight"),
                   "gate_up": Linear(torch.cat([take(s + "gate_proj.weight"), take(s + "up_proj.weight")]), False),
                   "down": Linear(take(s + "down_proj.weight"), False)}
        weights = {k: v for k, v in weights.items() if not k.startswith(("model.encoder.vision_tower.",
                                                                         "model.encoder.embed_vision."))}
        if weights:
            raise ValueError(f"Unconsumed checkpoint weights: {sorted(weights)[:8]}")
        self.attention = Attention(c, self.device, max_seq_len, self.dtype)
        self.blackwell = torch.cuda.get_device_capability(self.device)[0] >= 10
        # The FP8 KV cache holds FP8 values in BF16 storage: reading FP8 costs the Triton kernel more than the bytes
        # it saves, and FA3 reads BF16.
        self.k, self.v = [], []
        for layer in self.layers:
            shape = (1, max_seq_len, layer["kv_heads"], layer["head_dim"])
            self.k.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
            self.v.append(torch.zeros(shape, device=self.device, dtype=self.dtype))
        d, dg = c.head_dim, c.global_head_dim
        inv = 1.0 / c.rope_theta_sliding ** (torch.arange(0, d, 2, device=self.device).float() / d)
        rotated = int(c.rope_partial_full * dg // 2)
        inv_full = torch.zeros(dg // 2, device=self.device)
        inv_full[:rotated] = 1.0 / c.rope_theta_full ** (torch.arange(0, 2 * rotated, 2, device=self.device).float() / dg)
        self.rope = {True: _rope_table(inv, max_seq_len), False: _rope_table(inv_full, max_seq_len)}
        rows = c.block_size
        fp4 = next((layer["fp4"] for layer in self.layers if layer["fp4"] is not None), None)
        if fp4 is None:
            self.experts = DeepGemmExperts(c, self.expert_lo, self.local_experts, rows, self.device)
        elif fp4.get("backend") == "cutlass":
            self.experts = CutlassExperts(c, rows, self.device, fp4)
        else:
            self.experts = TrtllmExperts(c, self.expert_lo, rows, self.device, fp4)
        # Row-parallel outputs (partial sums under TP, in symmetric memory): attention, [dense MLP; experts], SC.
        if group:
            self.attn_out = group.symmetric("dg.attn", (rows, c.hidden_size))
            self.ffn_out = group.symmetric("dg.ffn", (2 * rows, c.hidden_size))
            self.sc_out = group.symmetric("dg.sc", (rows, c.hidden_size))
        else:
            self.attn_out = self.sc_out = None
            self.ffn_out = torch.empty(2 * rows, c.hidden_size, dtype=self.dtype, device=self.device)
        # The dense MLP runs on a side stream, concurrently with the expert path (disjoint outputs and scratch).
        self.side = torch.cuda.Stream(self.device) if OVERLAP else None
        block_graph()
        if hasattr(self.experts, "calibrate"):
            self.experts.calibrate(self)

    @classmethod
    def load(cls, path, max_seq_len=16384, *, device=None, group=None, precision=None):
        """`precision`: "bf16" or "nvfp4" (an NVFP4 checkpoint), default the checkpoint's own (see precision.py).
        Under TP each rank reads only its experts."""
        config = GemmaConfig.load(path)
        device = device or (group.device if group else "cuda:0")
        precision = precision_switch.resolve(path, precision, device, backends=("trtllm", "cutlass"))
        weights = precisions.read_weights(path, config, precision, device, group)
        with torch.cuda.device(device):
            return cls(config, weights, max_seq_len, group=group, precision=precision)

    def reduce(self, partial, site, chain=0):
        """Sum over ranks. Sites on one chain run in order on one stream; a reduction re-arms the previous
        output of its chain, so every graph ends each chain on the same site."""
        if self.group is None:
            return partial
        return self.group.reduce(partial, site=site, chain=chain)

    def mlp(self, x, weights, out=None):
        return weights["down"](gk.geglu(weights["gate_up"](x)), out=out)

    def moe(self, x, router_in, layer, out):
        """This rank's experts' part of the MoE output, written into `out`."""
        return self.experts(x, layer["router"](router_in), layer, out)

    def self_condition(self, probs):
        """SC delta from the previous round's probabilities over this rank's vocabulary rows: GeGLU adapter
        (replicated) on norm(probs @ E * sqrt(H))."""
        c = self.config
        signal = self.reduce(torch.mm(probs, self.lm_head, out=self.sc_out), "sc")
        x = gk.norm(signal, self.sc["norm"], c.rms_norm_eps, pre=self.embed_scale)
        return self.mlp(x, self.sc)

    def forward(self, h, start, end, leftpad, *, causal, splits):
        """h: [rows, hidden] block input, consumed in place. Returns the final-norm hidden states.

        Writes this block's K/V at [start, start + rows); `leftpad` bounds the canvas's sliding-layer keys.
        """
        c = self.config
        rows = h.shape[0]
        sliding = next(layer for layer in self.layers if layer["sliding"])
        schedule = self.attention.schedule(rows, end, sliding["heads"], sliding["kv_heads"], splits, causal)
        x = gk.norm(h, self.layers[0]["norm"], c.rms_norm_eps)
        scratch = (torch.empty_like(h), torch.empty_like(h), torch.empty_like(h))
        for i in range(len(self.layers)):
            self.layer(i, h, x, scratch, start, end, leftpad, causal=causal, splits=splits, schedule=schedule)
        return x

    def layer(self, i, h, x, scratch, start, end, leftpad, *, causal, splits, schedule):
        """Layer i in place: h is the residual stream, x its input norm on entry and the next layer's on exit.

        `scratch` is three [rows, hidden] buffers (dense-MLP input, expert input, router input); `schedule` is the
        causal FA3 schedule from `forward`, or None. The residual stream stays replicated: every reduction returns
        the full, identical sum on all ranks.
        """
        c, layer = self.config, self.layers[i]
        rows, eps = h.shape[0], c.rms_norm_eps
        x_mlp, x_moe, router_in = scratch
        hq, hk, d = layer["heads"], layer["kv_heads"], layer["head_dim"]
        cos, sin = self.rope[layer["sliding"]]
        q = torch.empty((1, rows, hq, d), device=h.device, dtype=h.dtype)
        gk.qkv_rope(layer["qkv"](x), layer["q_norm"], layer["k_norm"], cos, sin, start, q, self.k[i], self.v[i],
                    hq, hk, d, layer["sliding"], eps, kv_fp8=self.kv_fp8, split=self.blackwell)
        a = self.attention(layer["sliding"], q, self.k[i], self.v[i], start, end, leftpad, causal, splits, schedule)
        attn = self.reduce(layer["o"](a.view(rows, -1), out=self.attn_out), f"{i}.attn")
        fp4 = self.experts.quantized(layer) if FUSED_QUANT and hasattr(self.experts, "quantized") else None
        gk.post_attn(h, attn, layer["post_attn"], layer["pre_mlp"], layer["pre_moe"], layer["router_scale"],
                     x_mlp, x_moe, router_in, eps, fp4=fp4)
        if fp4 is not None:
            x_moe = None
        # The dense half is reduced as soon as the dense MLP is done (on the side stream, hidden behind the
        # expert path), so only the expert half's reduction is exposed. Each row's sum is the same rank-order
        # FP32 sum as in one [2 x rows] reduction (which is ~2% slower on B200 at TP=8).
        if self.side is not None:
            main = torch.cuda.current_stream()
            self.side.wait_stream(main)
            with torch.cuda.stream(self.side):
                y_mlp = self.reduce(self.mlp(x_mlp, layer, out=self.ffn_out[:rows]), f"{i}.mlp", chain=2)
            y_moe = self.reduce(self.moe(x_moe, router_in, layer, out=self.ffn_out[rows:]), f"{i}.moe", chain=1)
            main.wait_stream(self.side)
        else:
            y_mlp = self.reduce(self.mlp(x_mlp, layer, out=self.ffn_out[:rows]), f"{i}.mlp", chain=2)
            y_moe = self.reduce(self.moe(x_moe, router_in, layer, out=self.ffn_out[rows:]), f"{i}.moe", chain=1)
        w_next = self.layers[i + 1]["norm"] if i + 1 < len(self.layers) else self.final_norm
        gk.post_ffn(h, y_mlp, y_moe, layer["post_mlp"], layer["post_moe"], layer["post_ffn"], w_next, x,
                    layer["scalar"], eps)

    def logits(self, x, out=None):
        """Raw BF16 LM-head logits (the soft cap is applied by the decoder statistics)."""
        return torch.mm(x, self.lm_head.t(), out=out)


def softcap(logits, cap):
    return cap * torch.tanh(logits.float() / cap)
