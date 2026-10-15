# Models

dlmrt picks the model family from the checkpoint's `config.json` (`model_type`) and validates the rest of the
config against what the engine implements; unsupported settings fail at load instead of producing wrong output.
`role`, `precision`, `entropy_bound` and `threshold` are available in the Python API
(`dlmrt.LLM(..., role=..., precision=...)`), in `dlmrt serve` and in `dlmrt bench`; the other settings below (edit
threshold, confidence temperature, QwenDLM's `ROLES`, FP8, `--numerics sglang`, `--in-place`, seeds) are
`dlmrt bench` options.

## QwenDLM (`sdar`)

[yifanyu/QwenDLM-4B](https://huggingface.co/yifanyu/QwenDLM-4B) and
[yifanyu/QwenDLM-4B-ORL](https://huggingface.co/yifanyu/QwenDLM-4B-ORL): dense SDAR block diffusion on the Qwen3
architecture (GQA, head_dim 128, per-head Q/K RMSNorm, RoPE). The ORL checkpoint adds a self-conditioning (SC)
adapter: the previous round's softmax pushed through the tied embedding, RMSNorm and a SwiGLU MLP, added to the token
embedding. Other SDAR checkpoints of the same architecture load as well.

- **Decoder** (`DecodeConfig`): 256-token blocks of deterministic noise; each round proposes the argmax at every
  position and accepts the lowest-entropy positions within an entropy budget (`entropy_bound`, default 0.6 in the
  API, `dlmrt serve` and `bench humaneval` whatever the checkpoint) or, with `accept_rule="prob_threshold"`, the
  positions whose confidence clears `prob_threshold`; accepted tokens are edited above a confidence threshold.
  `ROLES` holds the paper's Base and ORL decoders (`dlmrt bench suite --role`; the API and `dlmrt serve` ignore
  `role`). Prompts use the paper's fixed `<|im_start|>user … <|im_start|>assistant\n<think>\n` wrapper
  (`render_prompt`), not the tokenizer's chat template; the model always thinks.
- **Precision**: BF16, or FP8 projections with DeepGEMM's 1×128 activation and 128×128 weight scales
  (`dlmrt bench --fp8 [qkv,o,gate_up,down,sc]`; HumanEval accuracy in [performance](performance.md)).
- **Kernels**: patched DeepGEMM, the ThunderKittens fused gate_up + SwiGLU, and on one H100 the post-attention
  megakernel (built for QwenDLM-4B's shapes).
- **GPUs**: Hopper only: attention runs on sgl_kernel's FA3, and the fused gate_up + SwiGLU and the post-attention
  megakernel are built for sm_90a.
- **SGLang numerics** (`dlmrt bench suite --numerics sglang`; `sglang_numerics.py`): BF16 on one GPU, the paper
  runtime's forward (sgl_kernel norms, cuBLAS projections, FlashInfer FA3 attention, its self-conditioning) and its
  decisions (for ORL the fused row statistics, `torch.sort` / `torch.cumsum` of the entropy budget; for Base the
  softmax confidence and top-1 floor), and its prompt layout (a prompt of a block or more prefilled, a shorter one
  sharing the first block), so 4816 of the 4820 answers are the paper runtime's
  ([performance](performance.md#all-models-bf16-on-one-h100)). Kernels that reproduce those bit for bit
  run where faster: DeepGEMM and the fused gate_up + SwiGLU for the unsplit 256-row GEMMs, a variant of the
  post-attention megakernel with sgl_kernel's RMSNorm reduction and cuBLAS's split-K down projection, FlashInfer's
  ragged FA3 kernel built with the paged kernel's KV tile and 64-row query tiles (`dlmrt/ops/fa3_tma.py`), and Triton
  reproductions of the softmax, the soft-embedding split-K GEMM and the QK norm + RoPE. Verified on H100 with the
  paper's stack (torch 2.11, Triton 3.6, FlashInfer 0.6.12, sgl_kernel 0.4.3).

## LLaDA2 (`llada2_moe`)

[LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini) (masked block diffusion, 32-token blocks) and ORL
fine-tunes with a self-conditioning adapter: [yifanyu/LLaDA2.1-mini-ORL](https://huggingface.co/yifanyu/LLaDA2.1-mini-ORL)
and its NVFP4 export [yifanyu/LLaDA2.1-mini-ORL-NVFP4](https://huggingface.co/yifanyu/LLaDA2.1-mini-ORL-NVFP4).

| | LLaDA2.1-mini |
| --- | --- |
| Layers | 20: layer 0 a dense SwiGLU MLP (5120); layers 1–19 256 SwiGLU experts (512), top-8, plus one shared expert. Router: sigmoid scores (FP32 GEMM, logits rounded to BF16 as the reference does), biased group top-k (best 4 of 8 groups by their top-2 sum), weights renormalized × 2.5 |
| Attention | 16/4 heads, head_dim 128, per-head Q/K RMSNorm, NeoX RoPE on the first 64 dims; block-causal on an absolute 32-token grid; the prompt prefill runs whole blocks as FA3 batch entries over one KV arena |
| Decoder | `<\|mask\|>` noise; argmax proposals with softmax confidence; accept every masked position above the threshold (`role` base: 0.70, orl: 0.85; `threshold` overrides), or else the single most confident one; rewrite accepted tokens whose proposal changed with confidence > `edit_threshold` (0.5); once all are accepted, up to 16 edit-only rounds. Optionally the confidence uses the reference's step_linear temperature max(min, 1 − decrement · r), r the block's round index (`confidence_temperature_min`, `confidence_temperature_step_decrement`; the paper's ORL Speed mode: 0.6, 0.08) |
| Self-conditioning (ORL) | the previous round's softmax · embedding on the non-prompt rows → RMSNorm → SiLU-gated MLP, added to the token embedding |
| Fused rounds | a finished block's refine rounds run as the first half of the next block's first forward (64 rows): one forward fewer per block, the same tokens; the block graph is two device loops |

Under TP: query heads, dense and shared-expert columns and the vocabulary split Megatron-style, 32 whole experts per
rank at TP=8, the 4 KV heads replicated. The router runs as one BF16 GEMM, the shared expert on a side stream, and
decode rounds' routed experts as small-tile Triton grouped GEMMs with SwiGLU in the gate/up epilogue.

- **Prompts**: the tokenizer's chat template (which fixes "detailed thinking off"), encoded as the reference runtime
  does; `thinking` does not apply.
- **Precision** (`precision`): `bf16`, or `nvfp4` for ModelOpt NVFP4 exports: the 19 × 256 routed experts W4A4 and
  an FP8 KV cache with unit scale
  (the Q/K-norm + RoPE kernel writes K and V through FP8 E4M3); attention, the shared expert, layer 0's MLP, the
  router, embeddings and SC stay BF16 as exported. On SM 10.x (B200, B300) the experts run on FlashInfer's TRT-LLM-gen
  MoE with SwiGLU (the routing weights × 2.5 travel packed with the expert ids as BF16; the BF16 shared expert is
  added after). On SM 12.x (RTX 50 series) they run on dlmrt's Triton NVFP4 MoE (`ops/triton/nvfp4_moe.py`) with the
  TRT-LLM-gen MoE's roundings: FlashInfer's input quantization, GEMM1 and SwiGLU in FP32 into the NVFP4
  requantization, GEMM2 rounded to BF16 per expert, the BF16 routing weights. Fed the same inputs it returns the
  B200's bits except in about 0.2% of rows, where the TRT-LLM-gen MoE departs from exact W4A4 arithmetic and dlmrt's
  does not ([performance](performance.md#nvfp4-on-rtx-5090)). `DLMRT_LLADA_FP4_MOE=cutlass` runs FlashInfer's
  CUTLASS MoE instead, which rounds GEMM1 and SwiGLU outputs to BF16 before requantizing. `nvfp4` needs SM 10.x or
  12.x and FlashInfer; on other GPUs run the export at `bf16`, which runs its dequantized experts in BF16. The
  default is the checkpoint's own precision.
- **SM 12.x**: every BF16 GEMM (projections, router, shared expert, layer 0's MLP, LM head, SC) runs on a Triton GEMM in
  cuBLAS's B200 summation order (`ops/triton/gemm.py`: K summed in increasing order in FP32, split into ranges where
  cuBLAS splits it on B200), bitwise the B200's products. Attention runs on the Triton flash attention with 32 × 32
  tiles; `DLMRT_LLADA_ATTENTION=b200` runs a Gluon kernel that sums the probabilities in the B200 Triton kernel's
  register layout instead (`ops/triton/b200_attention.py`, bitwise its 64 × 64 attention).
- **Blackwell**: FA3 runs only on Hopper; attention runs on the Triton flash attention, block-causal on the 32-token
  grid. DeepGEMM's SM100 masked grouped GEMM is exact only from 128 rows per expert, so the masked expert buffers
  hold at least 128 rows.
- **SGLang numerics** (`dlmrt bench suite|tps --numerics sglang`; `sglang_numerics.py`): BF16 on one GPU, the forward in
  the paper runtime's kernels and reduction orders (sgl_kernel norms, cuBLAS projections, FlashInfer FA2 attention and
  grouped top-k, SGLang's Triton fused MoE), or dlmrt kernels that reproduce them bit for bit in fewer launches, inside
  dlmrt's block graphs and decoder; on ORL Speed HumanEval all 164 answers are the paper runtime's. Verified on H100
  with the paper's stack (torch 2.11, Triton 3.6, FlashInfer 0.6.12, sgl_kernel 0.4.3). `--max-seq-len` must lie
  between 1024 (the prefill graphs captured at load) and the checkpoint's `max_position_embeddings` (SGLang's RoPE
  table).

## DiffusionGemma (`diffusion_gemma`)

[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) and its fine-tunes:
uniform-state block diffusion on a Gemma 4 mixture-of-experts backbone. The ORL fine-tune is
[yifanyu/DiffusionGemma-26B-A4B-it-ORL](https://huggingface.co/yifanyu/DiffusionGemma-26B-A4B-it-ORL), with its NVFP4 export
[yifanyu/DiffusionGemma-26B-A4B-it-ORL-NVFP4](https://huggingface.co/yifanyu/DiffusionGemma-26B-A4B-it-ORL-NVFP4).

| | DiffusionGemma |
| --- | --- |
| Layers | 30: 25 sliding-window (1024, 16/8 heads, head_dim 256) and 5 full (16/2 heads, head_dim 512, value = unweighted norm of the key projection) |
| FFN | dense GeGLU (2112) in parallel with 128 top-8 GeGLU experts (704); sandwich norms and a layer scalar |
| Attention | FA3 on the sliding layers on Hopper (causal window for the commit, `cache_leftpad` for the canvas); Triton flash attention for head_dim 512 and on Blackwell (B200, B300, RTX 50 series) |
| Decoder | a canvas of uniform random tokens; each round samples softmax(softcap(z) / T) (Gumbel-max), keeps the lowest-entropy samples within an entropy bound and re-noises the rest; stops when the argmax canvas is stable and its mean entropy < 0.005 (≤ 48 rounds) |

- **Roles**: `base` (entropy bound 0.1, temperature 0.8 → 0.4 linearly) and `orl` (entropy bound 0.12, temperature
  from 0.8 down to 0.5 driven by the raw-logit entropy of the positions not yet accepted). `entropy_bound`
  overrides the role's bound.
- **Prompts**: the tokenizer's chat template; `thinking=False` passes `enable_thinking=False` (the answer then opens
  with an empty thought channel, which the API hides).
- **Precision** (`precision`): `bf16`, or `nvfp4` for ModelOpt NVFP4 exports: W4A4 experts (E2M1 weights with an FP8
  scale per 16 inputs and a per-tensor scale; inputs quantized per 16 against the exported input scale) and an FP8
  KV cache; attention, dense MLP, router, embeddings and SC stay BF16 as exported. The experts run on FP4 tensor
  cores through FlashInfer: on SM 10.x (B200, B300) its TRT-LLM-gen MoE, with the input quantized by the
  post-attention kernel; on SM 12.x (RTX 50 series) its CUTLASS fused MoE, which quantizes the BF16 input itself and
  whose GeGLU uses the erf GELU (the model's is the tanh GELU; the two differ far below the FP4 rounding). The
  kernel tactics are profiled once per GPU, shapes and rank and cached in `$DLMRT_CACHE_DIR`; the CUTLASS MoE keeps
  the fastest deterministic GEMM2 tactic (half of its tactics reduce the experts' rows in the epilogue in arbitrary
  order). `nvfp4` needs one of the two; `bf16` on an NVFP4 export runs its dequantized experts in BF16. The default is
  the checkpoint's own precision.
- **Numerics**: the row kernels round to BF16 where the HF reference does (norm outputs, residual adds, activation
  and RoPE products); `tests/test_gemma_parity.py` checks the forward against the HF implementation within the
  reference's own SDPA-vs-eager spread.
- **Blackwell**: Triton attention with tiles per GPU generation, layer kind and KV heads per rank (B200: accumulators
  in TMEM, DV + BLOCK_N ≤ 512 columns; SM 12.x: about 100 KB of shared memory per block, the full layers' whole
  512-wide value per program in 32 × 32 tiles), cuBLAS dense GEMMs, Q/K-norm + RoPE spread over one program per four
  query heads plus one for K/V (five per row at TP=1), and four columns' sampling noise per Philox call (so Blackwell
  samples a different noise stream than Hopper).

Under TP: attention heads, dense-MLP columns and the vocabulary split Megatron-style, experts whole (16 per GPU at
TP=8, read as slices at load), the dense MLP on a side stream next to the experts, three NVLink reductions per layer
(attention, dense MLP, experts).
