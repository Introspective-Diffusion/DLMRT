# Performance

Unless noted: one request at a time on an otherwise idle node; request TPS = generated tokens / time from the first
prefill kernel to the last block (tokenization excluded), pooled over requests; round latency = CUDA-event median of
one captured denoise round (self-conditioning → forward → LM head → decide) at a fixed context (DiffusionGemma: every
replay from the same random block and decoder state). Engines are compared on time per forward; models that sample
noise are compared over multiple seeds.

## All models, BF16 on one H100

One H100 per run (TP=1), the papers' prompts, decoders and token budgets: QwenDLM Base probability threshold 0.85
(HumanEval 0.9) with 49152 new tokens (HumanEval 32768), ORL entropy budget 0.4 (GSM8K, MATH-500) or 0.6 (HumanEval,
MBPP) with 32768 new tokens (MBPP 49152); LLaDA2.1-mini Quality mode threshold 0.7 / 0.85 with edits above 0.5, Speed
mode 0.5 / 0.6 with edit threshold 0 (ORL with the step_linear confidence temperature 0.6 / 0.08), 4096 new tokens;
DiffusionGemma 4096 new tokens (HumanEval 2048), each checkpoint's decoder on all four benchmarks, each cell the mean
of three seeds (below the table). Cells: accuracy (%, the papers' isolated scorers; length-capped answers count as
wrong) / TPF (mean over non-length responses of completion tokens / denoise rounds) / request TPS (completion tokens /
generation time, pooled over non-length responses); the papers report accuracy / TPF. GSM8K 1319 problems, MATH-500
500, HumanEval 164, MBPP 427.

| | GSM8K | MATH-500 | HumanEval | MBPP |
| --- | --- | --- | --- | --- |
| QwenDLM 4B Base, paper | 90.4 / 2.4 | 88.0 / 2.1 | 82.9 / 1.8 | 79.2 / 2.1 |
| dlmrt, SGLang numerics | 90.30 / 2.39 / 328 | 88.00 / 2.09 / 191 | 82.93 / 1.80 / 249 | 79.16 / 2.05 / 259 |
| QwenDLM 4B ORL, paper | 92.4 / 5.9 | 92.2 / 5.8 | 89.6 / 11.1 | 79.4 / 8.0 |
| dlmrt, SGLang numerics | 92.42 / 5.93 / 987 | 92.20 / 5.81 / 701 | 89.63 / 11.08 / 1609 | 79.63 / 8.01 / 1222 |
| LLaDA2.1-mini Base, Quality, paper | 91.9 / 5.6 | 84.6 / 8.2 | 86.6 / 11.3 | 77.5 / 7.7 |
| dlmrt, SGLang numerics | 91.89 / 5.61 / 918 | 84.60 / 8.19 / 1320 | 86.59 / 11.32 / 1698 | 77.52 / 7.65 / 1218 |
| LLaDA2.1-mini ORL, Quality, paper | 92.0 / 8.5 | 87.0 / 10.2 | 89.0 / 13.1 | 77.3 / 10.3 |
| dlmrt, SGLang numerics | 92.12 / 8.53 / 1281 | 86.40 / 10.28 / 1627 | 89.02 / 13.06 / 2124 | 77.28 / 10.28 / 1539 |
| LLaDA2.1-mini Base, Speed, paper | 89.69 / 7.3 | 84.0 / 10.2 | 84.15 / 13.9 | 73.54 / 9.5 |
| dlmrt, SGLang numerics | 89.76 / 7.34 / 1095 | 83.60 / 10.23 / 1557 | 84.15 / 13.90 / 1994 | 73.54 / 9.49 / 1384 |
| LLaDA2.1-mini ORL, Speed, paper | 90.3 / 12.0 | 86.0 / 14.0 | 87.8 / 16.8 | 71.43 / 14.1 |
| dlmrt, SGLang numerics | 90.22 / 11.95 / 1619 | 86.00 / 14.02 / 2028 | 87.80 / 16.79 / 2486 | 71.66 / 14.12 / 1953 |
| DiffusionGemma 26B-A4B Base, paper | 95.8 / 26.0 | 92.3 / 23.7 | 90.9 / 24.5 | 87.7 / 23.1 |
| dlmrt | 95.80 / 25.99 / 1271 | 92.27 / 24.01 / 1216 | 90.85 / 24.40 / 1196 | 87.67 / 23.14 / 1082 |
| DiffusionGemma 26B-A4B ORL, paper | 96.0 / 30.0 | 92.3 / 27.4 | 91.9 / 27.9 | 87.4 / 28.7 |
| dlmrt | 95.78 / 29.96 / 1456 | 92.33 / 27.62 / 1415 | 91.87 / 27.77 / 1378 | 87.35 / 28.24 / 1361 |

Against the paper runtime's records, the SGLang-numerics rows give 4816 of the 4820 QwenDLM answers (the other four are
ORL MBPP problems) and all 164 LLaDA2 ORL Speed HumanEval answers.

| DiffusionGemma seeds | GSM8K | MATH-500 | HumanEval | MBPP |
| --- | --- | --- | --- | --- |
| Base | 1234, 1239, 1241 | 1235, 1238, 1239 | 1237, 1238, 1240 | 1234, 1235, 1236 |
| ORL | 1236, 1238, 1241 | 1234, 1235, 1240 | 1235, 1239, 1241 | 1235, 1236, 1240 |

## QwenDLM 4B (H100)

Round latency, against a reference engine that reproduces SGLang's SDAR path:

| Context | Reference engine | dlmrt BF16 | dlmrt FP8 |
| ---: | ---: | ---: | ---: |
| 0 | 6.45 ms | 5.45 (1.18×) | 4.56 (1.41×) |
| 2048 | 7.48 | 6.12 (1.22×) | 5.23 (1.43×) |
| 4096 | 8.53 | 6.50 (1.31×) | 5.56 (1.53×) |
| 8192 | 10.51 | 7.89 (1.33×) | 6.32 (1.66×) |
| 16384 | 14.65 | 8.89 (1.65×) | 7.96 (1.84×) |

With the fused gate_up, the patched DeepGEMM and the post-attention megakernel, BF16 rounds are 5.09 / 5.77 / 6.91 ms
at 0 / 2K / 8K; full HumanEval over six seeds runs at 1640 request TPS on one GPU.

Tensor parallel (fused NVLink reduction):

| Context | TP=1 round | TP=8 round |
| ---: | ---: | ---: |
| 0 | 5.08 ms | 2.52 ms |
| 2K | 5.88 | 2.65 |
| 8K | 7.02 | 2.88 |

Full HumanEval at TP=8: 3770–3900 request TPS (TP=1: 1640).

Accuracy, full HumanEval (paper protocol, isolated scorer), one run per seed; the seed fixes each block's initial
noise, so engines are paired by seed:

| Engine (correct / 164) | 12345 | 12346 | 12347 | 12348 | 12349 | 12350 | Mean | Paired Δ ± s.e. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference (SGLang-exact) | 147 | 139 | 143 | 143 | 139 | 138 | 141.5 | — |
| **dlmrt BF16** | 141 | 142 | 138 | 142 | 145 | 141 | **141.5** | **+0.0 ± 2.0** |
| dlmrt FP8 (qkv, gate_up, down) | 136 | 138 | 137 | 133 | 135 | 141 | 136.7 | −4.8 ± 2.2 |
| dlmrt FP8 (all projections) | 144 | 134 | 136 | 140 | 133 | 137 | 137.3 | −4.2 ± 0.9 |

At TP=8 the same six seeds average 140.2.

QwenDLM 4B ORL (ORL decoder, entropy budget 0.6, 32768 new tokens): rounds 5.04 / 5.70 / 6.79 ms at context 0 / 2K /
8K at TP=1 and 2.65 / 2.75 / 2.96 ms at TP=8; full HumanEval at 1658 request TPS at TP=1 and 3640 at TP=8.

## LLaDA2.1-mini

Accuracy and TPF in the paper-aligned numerics: see [the table at the top](#all-models-bf16-on-one-h100).

ORL checkpoint, Quality decoder:

| H100, BF16 | Round, context 0 / 2K / 8K | Request TPS, full HumanEval |
| --- | ---: | ---: |
| TP=1 | 4.99 / 4.81 / 4.79 ms | 2060 |
| TP=8 | 2.18 / 2.34 / 2.38 ms | 5121 |

### SGLang numerics

`--numerics sglang` with the paper's four decoders: see [the table at the top](#all-models-bf16-on-one-h100). On ORL
Speed HumanEval it is token-identical with the paper runtime (164/164) and as fast as the default numerics (2428–2433
against 2420–2424 TPS).

### NVFP4 on B200

ORL checkpoint and decoder, one B200, the same protocol; NVFP4 exports with routed experts W4A4 and an FP8 KV cache
(accuracy / TPF):

| Precision | GSM8K (1319) | MATH-500 | HumanEval |
| --- | ---: | ---: | ---: |
| BF16 | 91.7% / 8.60 | 84.4% / 10.33 | 84.1% / 13.17 |
| NVFP4, diagonal-Hessian scales | 92.2% / 7.56 | 84.4% / 9.66 | 83.5% / 12.20 |
| NVFP4, local-Hessian scales | 92.0% / 7.54 | 84.0% / 9.72 | 89.0% / 12.23 |
| NVFP4, amax / 6 scales | 92.6% / 7.65 | 83.2% / 9.72 | 87.2% / 12.51 |

NVFP4 lowers tokens per round by 5–12%.

| B200 | TP=1 round (0 / 2K / 8K) | prefill, 256 rows | TP=8 round | Request TPS, TP=1 | TP=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 2.84 / 2.96 / 3.12 ms | 5.01 ms | 1.78 / 1.80 / 1.82 ms | 3633 | 5844 |
| NVFP4, local-Hessian scales | 2.03 / 2.13 / 2.35 ms | 2.58 ms | 1.69 / 1.72 / 1.76 ms | 5046 | 6036 |

(Request TPS: full HumanEval.) At TP=8 (32 experts per rank) the round is latency-bound: NVFP4 shortens it by 5%.

### NVFP4 on RTX 5090

RTX 5090 (SM 12.0, 32 GB), the local-Hessian export, the same protocol. The routed experts run on dlmrt's Triton
NVFP4 MoE with the TRT-LLM-gen MoE's roundings, every BF16 GEMM in cuBLAS's B200 summation order, attention on the
Triton kernel with 32 × 32 tiles. Fed the same inputs, every op of a layer but attention returns the B200's bits,
the routed experts in all but about 0.2% of rows, where the TRT-LLM-gen MoE departs from exact W4A4 arithmetic
(dlmrt's follows the exact arithmetic). The 32 × 32 tiles sum attention in another order than the B200's 64 × 64
kernel; `DLMRT_LLADA_ATTENTION=b200` returns its bits as well, and answers still diverge token-wise from the B200's.
With the default 32 × 32 tiles (accuracy / TPF / request TPS):

| | GSM8K | MATH-500 | HumanEval | MBPP |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 91.89 / 7.58 / 1648 | 84.80 / 9.66 / 2081 | 87.20 / 12.34 / 2788 | 77.28 / 9.72 / 2053 |
| B200 | 92.04 / 7.54 / 2953 | 84.00 / 9.72 / 3882 | 89.02 / 12.23 / 5046 | 75.88 / 9.79 / 3564 |

HumanEval moves by a few points under rounding-level changes on either GPU: with its other attention tiles the
RTX 5090 scores 83.5–85.4 (64 × 64, 64 × 32, 32 × 64) and the B200 84.8–86.0 (32 × 32, 64 × 32, 32 × 64). Over the
four tile sizes HumanEval averages 85.4 on the RTX 5090 and 86.3 on the B200. With 32 × 32 tiles the B200 scores
92.19 / 84.80 / 76.11 on GSM8K / MATH-500 / MBPP.

| RTX 5090 | TP=1 round (0 / 2K / 8K) | prefill, 256 rows | Request TPS |
| --- | ---: | ---: | ---: |
| NVFP4, local-Hessian scales | 3.47 / 3.69 / 4.10 ms | 5.66 ms | 2788 |

## DiffusionGemma 26B-A4B

ORL checkpoint and decoder; request TPS on full HumanEval (chat template with the paper's system / instruction
prompt, 2048 new tokens), seed 1234.

| H100 BF16 round | TP=1 | TP=2 | TP=4 | TP=8 |
| ---: | ---: | ---: | ---: | ---: |
| context 0 | 16.96 ms | 10.69 | 7.11 | 5.36 |
| 2K | 16.20 | 10.22 | 6.95 | 5.37 |
| 8K | 17.51 | 11.07 | 7.39 | 5.53 |

H100 request TPS: 1388 at TP=1, 4384 at TP=8.

| B200 | TP=1 round, context 0 / 2K / 8K | TP=8 round | Request TPS, TP=1 | TP=8 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 9.82 / 9.90 / 10.62 ms | 3.98 / 4.04 / 4.17 ms | 2373 | 5874 |
| NVFP4 experts (diagonal-Hessian scales) | 5.59 / 6.15 / 7.26 ms | 3.14 / 3.27 / 3.39 ms | 3949 | 7026 |

NVFP4 request TPS as in the paper: at TP=1 the mean of seeds 1234–1236, at TP=8 seed 1234.

RTX 5090 (SM 12.0, 32 GB), NVFP4 experts (diagonal-Hessian scales) on FlashInfer's CUTLASS MoE (deterministic
tactics), TP=1: rounds 15.41 / 16.45 / 18.80 ms at context 0 / 2K / 8K, 1490 request TPS. The paper's protocol at
seed 1234 (accuracy / TPF / TPS):

| Seed 1234 | GSM8K | MATH-500 | HumanEval | MBPP |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 95.53 / 29.14 / 1540 | 92.60 / 26.75 / 1408 | 93.90 / 27.78 / 1490 | 87.59 / 28.34 / 1465 |
| B200 | 95.53 / 29.34 / 4225 | 94.20 / 26.77 / 3810 | 92.68 / 27.05 / 3847 | 86.65 / 28.27 / 3881 |
