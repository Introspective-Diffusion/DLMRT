# Configuration

dlmrt reads its environment through `dlmrt/envs.py`, most switches once, at import. The defaults are the tuned
configurations; the switches exist for A/B measurements and for hardware where a path does not apply. Every switch
keeps the engine's results correct; switches marked *bit-identical* do not change a single output token. On/off
switches take `1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off` (any case) and named switches only the values listed
below (the two attention switches also take an empty value: automatic); any other value raises a `ValueError` naming
the variable.

## General

| Variable | Default | Meaning |
| --- | --- | --- |
| `DLMRT_CACHE_DIR` | `~/.cache/dlmrt` | autotuned MoE tactics (per GPU, shapes and rank), the patched DeepGEMM include root and the patched FlashInfer FA3 sources |
| `DLMRT_PDL` | `1` | `0`: launch kernels without programmatic dependent launch (bit-identical) |
| `DLMRT_DEEP_GEMM_PATCH` | `1` | `0`: stock DeepGEMM SM90 BF16 GEMM instead of the patched copy (bit-identical) |

The CUDA extensions build into `$TORCH_EXTENSIONS_DIR` (torch's default `~/.cache/torch_extensions`), Triton
kernels into `$TRITON_CACHE_DIR`, and the SGLang-numerics modes' FlashInfer module into FlashInfer's JIT cache
(`$FLASHINFER_WORKSPACE_BASE`, default `~/.cache/flashinfer`; several minutes on first use).

## Tensor parallelism

| Variable | Default | Meaning |
| --- | --- | --- |
| `DLMRT_TP_FUSED` | `1` | QwenDLM: `0`: torch symmetric-memory all-reduce + RMSNorm instead of the fused reduction (LLaDA2: `DLMRT_LLADA_REDUCE`) |
| `DLMRT_TP_ALL_REDUCE` | `multimem` | the unfused all-reduce (QwenDLM with `DLMRT_TP_FUSED=0`, LLaDA2 with `DLMRT_LLADA_REDUCE=multimem`): `multimem` (NVLink SHARP) or `two_shot` |
| `DLMRT_TP_PUSH_BLOCKS` | `96` | CTAs of the fused reduction that only push partial rows over NVLink |

## QwenDLM (`sdar`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `DLMRT_SPLIT_CONTEXT` | `1536` | context length from which FA3 splits the keys |
| `DLMRT_MLP` | `fused` | `separate`: DeepGEMM gate_up, Triton SwiGLU, DeepGEMM down (bit-identical) |
| `DLMRT_OGUD` | `1` | `0`: separate kernels instead of the post-attention megakernel (bit-identical) |

## LLaDA2

| Variable | Default | Meaning |
| --- | --- | --- |
| `DLMRT_LLADA_OVERLAP` | `1` | `0`: the shared expert inline instead of on a side stream (bit-identical) |
| `DLMRT_LLADA_MOE` | `triton` | routed-expert GEMMs of decode rounds: `triton` (small-tile grouped) or `deepgemm` (masked) |
| `DLMRT_LLADA_REDUCE` | `fused` | cross-rank sum under TP: `fused`, `one_shot`, or `multimem` |
| `DLMRT_LLADA_SPLITS` | by context | force one FA3 key-split count |
| `DLMRT_LLADA_ATTENTION` | FA3 on Hopper, Triton elsewhere | `fa3`, `triton`, or on SM 12.x `b200` (a Gluon kernel in the B200 Triton kernel's summation order) |
| `DLMRT_LLADA_FP4_MOE` | `triton` | NVFP4 routed experts on SM 12.x: `triton` (the TRT-LLM-gen MoE's roundings) or `cutlass` (FlashInfer's CUTLASS MoE) |

## DiffusionGemma

| Variable | Default | Meaning |
| --- | --- | --- |
| `DLMRT_DG_ATTENTION` | FA3 on Hopper, Triton elsewhere | `fa3` or `triton` for the sliding-window layers |
| `DLMRT_DG_FUSED_QUANT` | `1` | `0`: quantize the NVFP4 experts' input in a separate pass (bit-identical) |
| `DLMRT_DG_OVERLAP` | `1` | `0`: the dense MLP inline instead of on a side stream next to the experts (bit-identical) |
| `DLMRT_DG_SPLITS` | `1` on one GPU, `4` under TP | FA3 key splits of the sliding layers |
