# Third-party notices

dlmrt is licensed under the Apache License 2.0 (LICENSE). It includes code from the projects below, under their own
licenses; each directory also carries the license file.

## ThunderKittens

`dlmrt/ops/csrc/tk/` is a vendored subset of [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) (the
Hopper headers the fused gate_up + SwiGLU kernel in `dlmrt/ops/csrc/gate_up_swiglu.cu` is built on).

```
MIT License

Copyright (c) 2024-2026 HazyResearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SGLang and vLLM

`dlmrt/ops/triton/sglang_fused_moe.py` is adapted from the Triton fused-MoE kernel and launcher of
[SGLang](https://github.com/sgl-project/sglang) (`python/sglang/srt/layers/moe/fused_moe_triton/`), which SGLang
adapted from [vLLM](https://github.com/vllm-project/vllm)'s `fused_moe.py`. The row-statistics kernel in
`dlmrt/models/sdar/sglang_kernels.py` is adapted from `_row_stats_temp_ptr_kernel` of the SGLang fork the paper's
runs used (the ORL code's `evaluation/inference/sglang/`, derived from SGLang), which its SDAR engine patch
(`evaluation/runtime/sglang/sdar-engine.patch`) adds to `sglang/srt/dllm/algorithm/fused_step_kernels.py`, a file
upstream SGLang does not have. SGLang, the fork and vLLM are licensed under the Apache License 2.0, the license of
dlmrt (LICENSE).

```
Copyright 2023-2024 SGLang Team
Copyright contributors to the vLLM project
```

## FlashInfer

`dlmrt/ops/fa3_tma.py` builds FlashInfer's SM90 batch-prefill module from the installed
[FlashInfer](https://github.com/flashinfer-ai/flashinfer)'s own sources with a few lines of them patched (tile sizes,
register budget, barrier conditions). FlashInfer is licensed under the Apache License 2.0, the license of dlmrt
(LICENSE); its FA3 kernels derive from FlashAttention-3 (BSD 3-Clause).

```
Copyright (c) 2023-2025 by FlashInfer team.
Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
```

## DeepGEMM

`dlmrt/ops/csrc/deep_gemm/sm90_bf16_gemm.cuh` is a modified copy of the SM90 BF16 GEMM of
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM); dlmrt points DeepGEMM's JIT at it (`dlmrt/ops/extensions.py`).
`dlmrt/ops/csrc/ogud.cu` builds on DeepGEMM / CuTe device code included from the installed DeepGEMM package.

```
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
