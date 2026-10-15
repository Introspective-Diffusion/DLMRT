"""DiffusionGemma's routed experts (top-8 of 128 GeGLU experts) per backend. Each takes a layer's router logits and
writes this rank's part of the MoE output (experts [lo, lo + count) under expert parallelism).

    DeepGemmExperts  BF16 weights on DeepGEMM's masked grouped GEMMs.
    TrtllmExperts    NVFP4 weights and inputs on FlashInfer's TRT-LLM-gen MoE (SM 10.x FP4 tensor cores).
    CutlassExperts   NVFP4 weights and inputs on FlashInfer's CUTLASS fused MoE (SM 12.x FP4 tensor cores).
"""

import deep_gemm
import torch

from ...ops.triton import moe
from ...quantization import nvfp4


class DeepGemmExperts:
    def __init__(self, config, lo, count, rows, device):
        c = self.config = config
        self.lo, self.count = lo, count
        dev, k = {"device": device, "dtype": torch.bfloat16}, c.top_k_experts
        # Masked expert layout: every expert has room for all rows; only its assigned rows are computed.
        self.moe_in = torch.zeros(count, rows, c.hidden_size, **dev)
        self.moe_mid = torch.zeros(count, rows, 2 * c.moe_intermediate_size, **dev)
        self.moe_act = torch.zeros(count, rows, c.moe_intermediate_size, **dev)
        self.moe_out = torch.zeros(count, rows, c.hidden_size, **dev)
        self.ids = torch.zeros(rows, k, dtype=torch.int32, device=device)
        self.slots = torch.zeros(rows, k, dtype=torch.int32, device=device)
        self.weights = torch.zeros(rows, k, dtype=torch.float32, device=device)
        self.counts = torch.zeros(c.num_experts, dtype=torch.int32, device=device)
        self.expected_m = max(1, rows * k // c.num_experts)

    def __call__(self, x, logits, layer, out):
        lo = self.lo
        moe.route(logits, layer["expert_scale"], self.ids, self.weights, self.slots, self.counts,
                 self.config.top_k_experts)
        counts = self.counts[lo:lo + self.count]
        moe.moe_gather(x, self.ids, self.slots, self.moe_in, lo)
        deep_gemm.m_grouped_bf16_gemm_nt_masked(self.moe_in, layer["w13"], self.moe_mid, counts, self.expected_m)
        moe.moe_geglu(self.moe_mid, self.ids, self.slots, self.moe_act, lo)
        deep_gemm.m_grouped_bf16_gemm_nt_masked(self.moe_act, layer["w2"], self.moe_out, counts, self.expected_m)
        moe.moe_combine(self.moe_out, self.ids, self.slots, self.weights, out, lo)
        return out


class TrtllmExperts:
    def __init__(self, config, lo, rows, device, sample):
        """`sample`: one layer's `experts.fp4` weights, to pick the kernel tactic for `rows` rows before any graph is
        captured (every layer has the same shapes)."""
        c = self.config = config
        self.lo = lo
        self.packed = torch.zeros(rows, c.top_k_experts, dtype=torch.int32, device=device)
        # The expert input as the post-attention kernel writes it: NVFP4 codes and E4M3 scales (see `quantized`).
        self.codes = torch.zeros(rows, c.hidden_size // 2, dtype=torch.uint8, device=device)
        self.scales = torch.zeros(rows, c.hidden_size // 16, dtype=torch.uint8, device=device)
        self.tactics = nvfp4.tune(sample, c.num_experts, lo, c.moe_intermediate_size, rows, c.hidden_size,
                                  c.top_k_experts, device)

    def quantized(self, layer):
        """(global scale, codes, scales): where kernels.post_attn writes this layer's quantized expert input,
        read by __call__ with x None."""
        return layer["fp4"]["a1_inv"], self.codes, self.scales

    def calibrate(self, model):
        """Keep, among the autotuner's fastest tactics in isolation, the fastest in the model's forward, where the
        experts share the GPU with the dense MLP on the side stream (the two orders differ by ~5% of a round). Runs
        once per GPU and shapes (the choice is persisted with the autotuner's), at TP=1 (under TP every rank would
        have to time the same candidates in lockstep)."""
        t = self.tactics
        if model.group is not None or len(t.candidates) < 2:
            return
        c, dev = model.config, model.device
        h0 = torch.randn(c.block_size, c.hidden_size, device=dev, generator=torch.Generator(dev).manual_seed(0))
        h0 = h0.to(torch.bfloat16)
        h = torch.empty_like(h0)
        start, leftpad = torch.zeros(1, device=dev, dtype=torch.int32), torch.zeros(1, device=dev, dtype=torch.int32)
        end = torch.full((1,), c.block_size, device=dev, dtype=torch.int32)

        def forward():
            h.copy_(h0)
            model.forward(h, start, end, leftpad, causal=False, splits=1)

        timed = []
        for tactic in t.candidates:
            nvfp4.pin(t, tactic)
            forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            for _ in range(3):
                graph.replay()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(10):
                graph.replay()
            b.record()
            b.synchronize()
            timed.append((a.elapsed_time(b), tactic))
            del graph
        nvfp4.pin(t, min(timed)[1])

    def __call__(self, x, logits, layer, out):
        """`x` None: the input is already in `codes` / `scales` (see `quantized`)."""
        c = self.config
        moe.route_packed(logits, layer["expert_scale"], self.packed, c.top_k_experts)
        return nvfp4.trtllm_moe(x, self.packed, layer["fp4"], out, c.num_experts, self.lo, c.moe_intermediate_size,
                                quantized=None if x is not None else (self.codes, self.scales))


class CutlassExperts:
    def __init__(self, config, rows, device, sample):
        """`sample`: one layer's `experts.fp4` weights, to tune the kernel tactics for up to `rows` rows before any
        graph is captured (every layer has the same shapes)."""
        c = self.config = config
        self.rows = rows
        self.ids = torch.zeros(rows, c.top_k_experts, dtype=torch.int32, device=device)
        self.weights = torch.zeros(rows, c.top_k_experts, dtype=torch.float32, device=device)
        self.pos = torch.zeros(rows, c.num_experts, dtype=torch.int32, device=device)
        nvfp4.cutlass_tune(sample, c.num_experts, rows, c.hidden_size, c.top_k_experts, device)

    def __call__(self, x, logits, layer, out):
        c, n = self.config, x.shape[0]
        ids, weights = self.ids[:n], self.weights[:n]
        moe.route_topk(logits, layer["expert_scale"], ids, weights, self.pos[:n], c.top_k_experts)
        return nvfp4.cutlass_moe(x, ids, weights, layer["fp4"], out, c.num_experts, self.rows)
