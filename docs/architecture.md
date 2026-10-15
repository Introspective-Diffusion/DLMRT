# Architecture

dlmrt serves one request at a time and optimizes that request's tokens per second. Block-diffusion models generate
a block of tokens (32 or 256) at a time: a block starts as noise and is refined by several bidirectional forward
passes ("denoise rounds") over the whole block until the decoder accepts every position; a causal "commit" pass then
writes the block's keys and values for the blocks after it.

## The block graph

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/block-graph-dark.svg">
  <img alt="The block graph. The host launches one CUDA graph per block (parameters through a pinned H2D copy). On the
  device: begin (prompt tail and deterministic noise), a WHILE loop of denoise rounds decided on the GPU (self-conditioning
  on the previous round's probabilities, embedding, L layers of norm, QKV, Q/K norm with RoPE and the KV write,
  attention, O, norm and MLP or MoE, then the final norm, LM head, row statistics and accept / edit / re-noise), and a
  causal commit that writes the block's K/V. The host then reads back the block's tokens, checks EOS and the budget,
  and streams the block." src="../assets/block-graph-light.svg" width="100%">
</picture>

The host launches one CUDA graph per block and reads back its token ids. The denoise loop is a CUDA conditional
WHILE node (`dlmrt/ops/csrc/block_graph.cu`) whose condition kernel reads device counters after every round, so no
round waits for the CPU. `LoopGraph` chains several WHILE loops (LLaDA2's fused rounds use two). DiffusionGemma
commits each canvas at the start of the next canvas's graph, so a request's last canvas is never committed.

Every kernel in a round is a programmatic dependent launch (PDL): each waits for its predecessor before touching
memory and releases its successor near its end (~0.4 µs saved per kernel boundary): Triton kernels after their last
store, the patched DeepGEMM GEMM and the post-attention megakernel after their mainloop, since their successors are
row kernels that wait anyway. A kernel followed by a GEMM must release only after its last store, and from threads
that do work; an early release makes the next GEMM's CTAs land out of order.

## Code map

| Package | Role |
| --- | --- |
| `dlmrt/models/<family>/config.py` | checkpoint contract (validated against `config.json`), the decoder's settings |
| `dlmrt/models/<family>/model.py` | weights (sharded under TP), the KV cache, the block forward |
| `dlmrt/models/<family>/engine.py` | request state, graph capture, the host loop over blocks |
| `dlmrt/models/{llada2,diffusion_gemma}/kernels.py` | Triton kernels particular to the family |
| `dlmrt/models/diffusion_gemma/{attention,experts,precision}.py` | DiffusionGemma's attention (FA3, Triton) and routed-expert (DeepGEMM, TRT-LLM-gen, CUTLASS) backends, its checkpoint reading per precision |
| `dlmrt/models/{llada2,sdar}/sglang_numerics.py`, `sglang_kernels.py` | the SGLang-numerics modes: the paper runtime's forward and decisions in dlmrt's block graphs, and the kernels that reproduce its reductions bit for bit |
| `dlmrt/ops/triton/` | shared Triton kernels: PDL plumbing and numerics helpers (`common`), norms / SwiGLU / FP8 quantization / Q/K-RoPE / embedding (`core`), the QwenDLM and LLaDA2 decoder (`decoding`), MoE routing and the masked expert layout (`moe`), flash attention where FA3 does not run (`attention`), on SM 12.x the NVFP4 MoE with the TRT-LLM-gen MoE's roundings (`nvfp4_moe`), BF16 GEMMs in cuBLAS's B200 summation order (`gemm`) and a Gluon attention in the B200 kernel's summation order (`b200_attention`), for the SGLang-numerics modes SGLang's fused MoE (`sglang_fused_moe`, vendored) and QK norm + RoPE (`sglang_rope`) |
| `dlmrt/ops/csrc/` | CUDA: the block graph, the fused TP reduction, the ThunderKittens fused gate_up + SwiGLU, the QwenDLM post-attention megakernel (and its SGLang-numerics variant), the patched DeepGEMM kernel |
| `dlmrt/ops/extensions.py` | builds the CUDA extensions on first use; sets up DeepGEMM |
| `dlmrt/ops/fa3_tma.py` | FlashInfer's paged FA3 numerics on a contiguous KV cache through its patched ragged kernel (the SGLang-numerics modes) |
| `dlmrt/layers/linear.py` | `Linear`: DeepGEMM BF16 / FP8 (1×128 activation, 128×128 weight scales), cuBLAS BF16 on Blackwell; on SM 12.x (LLaDA2) a Triton GEMM in cuBLAS's B200 summation order |
| `dlmrt/quantization/nvfp4.py` | ModelOpt NVFP4: dequantization, the experts on FlashInfer's TRT-LLM-gen MoE (SM 10.x) or CUTLASS MoE (SM 12.x), and laid out for `ops/triton/nvfp4_moe` (SM 12.x) |
| `dlmrt/quantization/precision.py` | the `bf16` / `nvfp4` switch of the MoE families: resolving a checkpoint's precision on a device |
| `dlmrt/distributed/parallel.py` | the TP group, symmetric-memory buffers, the fused reduction's call sites |
| `dlmrt/llm.py` | `LLM` and `load_runtime`: one front end over the three families (prompt formats, stop tokens) |
| `dlmrt/entrypoints/` | `dlmrt bench` (per-family benchmark protocols), `dlmrt serve` (streaming HTTP) |

## Kernels

At 256 rows the projections sit at the H100 compute / bandwidth ridge, so neither launch count nor bandwidth alone
bounds them. Measured per QwenDLM-4B layer (H100):

| Shape | cuBLAS BF16 | DeepGEMM BF16 | DeepGEMM FP8 | HBM floor BF16 |
| --- | ---: | ---: | ---: | ---: |
| QKV 2560→6144 | 15.7 µs | 15.6 | 9.9 | 9.4 |
| O 4096→2560 | 13.6 | 12.1 | 9.5 | 6.3 |
| gate_up 2560→19456 | 46.2 | 45.1 | 28.0 | 29.7 |
| down 9728→2560 | 30.8 | 26.8 | 18.8 | 14.9 |

- **GEMMs.** DeepGEMM on Hopper (it supports PDL); its SM90 BF16 kernel is patched (`ops/csrc/deep_gemm`) with L2
  cache policies (weights `evict_first`, activations and outputs `evict_last`) and an early PDL trigger, bit-identical
  (−1 to −1.5% per round). On Blackwell, dense BF16 GEMMs run on cuBLAS (20–40% faster than DeepGEMM at 256 rows);
  LLaDA2 on SM 12.x uses a Triton GEMM in cuBLAS's B200 summation order instead.
- **Fused gate_up + SwiGLU** (ThunderKittens, one GPU): one 256-row × 80-column task per CTA, `[gate | up]` as one
  WGMMA, Triton's sigmoid lowering in the epilogue; bit-identical to GEMM + SwiGLU, −3 to −4% per round.
- **Post-attention megakernel** (QwenDLM, one H100, BF16): O projection, residual RMSNorm, gate_up + SwiGLU and down as
  one persistent launch of 132 CTAs whose phases hand off through per-row-block arrival counters; bit-identical,
  −1.6 to −2.3% per round (`DLMRT_OGUD=0` restores the separate kernels).
- **Attention.** FA3 (`sgl_kernel`) over one contiguous KV arena with device-side sequence lengths and one tile
  schedule per forward, KV splits at long context. DiffusionGemma's head_dim-512 layers, and every layer on
  Blackwell, use a Triton flash attention with tiles tuned per GPU generation and layer kind.
- **Row kernels.** 2-D `[groups, 128]` tiles; FP8 quantization fused into RMSNorm and SwiGLU; Q/K norm + RoPE + KV
  write as one program per row; every global load issued before the first dependent wait.
- **Self-conditioning.** The decoder's log-normalizer is reused for a single-pass BF16 softmax (50 µs instead of an
  FP32 softmax at 412 µs) and one 256-row GEMM against the embedding.
- **MoE.** Routing and the masked expert layout in Triton; experts on DeepGEMM's masked grouped GEMM (BF16), small-tile
  Triton grouped GEMMs for LLaDA2's 32-row decode rounds, or NVFP4 on Blackwell FP4 tensor cores: FlashInfer's
  TRT-LLM-gen MoE on SM 10.x with its input quantized by the preceding kernel; on SM 12.x dlmrt's Triton MoE with the
  TRT-LLM-gen MoE's roundings, bitwise its output in all but about 0.2% of rows (LLaDA2), or FlashInfer's CUTLASS MoE
  on deterministic tactics (DiffusionGemma).

## Tensor parallelism

One process per GPU (`torchrun`) runs the same program: the same block graphs and the same host loop. Layout
(Megatron): QKV and gate_up column-parallel, O and down row-parallel, the tied embedding vocabulary-parallel for the
LM head and the self-conditioning GEMM; MoE experts are split whole across ranks (expert parallelism). Row statistics
over the vocabulary are gathered in rank order, so every rank's decoder sees the same inputs.

Row-parallel partial sums are written into symmetric memory and reduced by `ops/csrc/tp_comm.cu`: pusher CTAs store
each rank's rows into their owners' receive slots over NVLink, owner CTAs sum the `world` slots in rank order (FP32,
deterministic), optionally add the residual and apply RMSNorm, and multicast the row to every rank. Arrival is
detected in-band: buffers start filled with a sentinel no computed BF16 value takes, so there is no barrier or system
fence on the critical path. Every rank receives the bitwise-identical sum, so each rank's device loop takes the same
number of rounds and issues the same collectives with no host coordination after the prompt is shared.

Reductions that may run concurrently use separate chains: each invocation re-arms the previous output buffer of its
chain, so every graph must end each chain on the same call site.
