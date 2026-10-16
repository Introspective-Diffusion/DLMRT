<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/dlmrt-logo-dark.svg">
  <img alt="dlmrt" src="assets/dlmrt-logo-light.svg" width="400">
</picture>

**A single-request inference engine for block-diffusion language models**

[Paper](https://arxiv.org/abs/2610.20687) · [Project page](https://introspective-diffusion.github.io/orl-dlm) · [ORL code](https://github.com/Introspective-Diffusion/ORL) ·
[Checkpoints](https://huggingface.co/collections/yifanyu/orl-on-policy-residual-learning-for-diffusion-lms)

[Models](docs/models.md) · [Performance](docs/performance.md) · [Benchmarks](docs/benchmarks.md) ·
[Architecture](docs/architecture.md) · [Server](docs/server.md)

</div>

dlmrt runs one request at a time and makes that request fast. Every forward denoises a whole block of tokens, and the
denoising loop runs on the GPU. It serves QwenDLM, LLaDA2.1-mini and DiffusionGemma checkpoints on one GPU or tensor
parallel across an NVLink node, on NVIDIA Hopper (H100) and, for LLaDA2.1-mini and DiffusionGemma, Blackwell (B200,
B300; NVFP4 also on the RTX 5090). It is the inference engine of
[On-Policy Residual Learning for Diffusion Language Models](https://arxiv.org/abs/2610.20687) (ORL).

## Highlights

- **Denoising on the device.** Each block is one CUDA graph, begin → `WHILE(not done){round}` → commit, with a
  conditional node that decides every round on the GPU. The host launches a block and reads back its tokens.
- **The papers' numerics.** Kernels follow each reference implementation's BF16 roundings. For QwenDLM and LLaDA2, the
  SGLang-numerics mode runs the paper runtime's forward pass and decisions: 4,816 of the 4,820 QwenDLM answers and all
  164 LLaDA2.1-mini ORL Speed HumanEval answers are identical to the paper runtime's.
- **Tensor parallel without barriers.** A fused NVLink reduction (push reduce-scatter, rank-ordered FP32 sum,
  multicast all-gather, in-band arrival detection) returns identical values on every rank, so all ranks run the same
  device loop with no host coordination.
- **Overlapped launches.** Every Triton kernel and the patched DeepGEMM GEMMs use programmatic dependent launch: each
  kernel's launch overlaps its predecessor's tail.
- **Low precision.** FP8 block-scaled projections for QwenDLM; NVFP4 experts on Blackwell FP4 tensor cores for LLaDA2
  and DiffusionGemma (FlashInfer's TRT-LLM-gen MoE; on RTX 50-series GPUs dlmrt's Triton MoE with the TRT-LLM-gen
  MoE's roundings for LLaDA2, bitwise the B200's in all but about 0.2% of rows, and FlashInfer's CUTLASS MoE for
  DiffusionGemma), with an FP8 KV cache.

## Supported models

| Family | `model_type` | Checkpoints | Block | Decoder |
| --- | --- | --- | ---: | --- |
| QwenDLM | `sdar` | [yifanyu/QwenDLM-4B](https://huggingface.co/yifanyu/QwenDLM-4B), [yifanyu/QwenDLM-4B-ORL](https://huggingface.co/yifanyu/QwenDLM-4B-ORL): SDAR block diffusion on the Qwen3 architecture, ORL with a self-conditioning adapter | 256 | entropy-budget or probability-threshold acceptance with edits |
| LLaDA2 | `llada2_moe` | [inclusionAI/LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini), [yifanyu/LLaDA2.1-mini-ORL](https://huggingface.co/yifanyu/LLaDA2.1-mini-ORL), [yifanyu/LLaDA2.1-mini-ORL-NVFP4](https://huggingface.co/yifanyu/LLaDA2.1-mini-ORL-NVFP4) | 32 | threshold acceptance, fused refine rounds |
| DiffusionGemma | `diffusion_gemma` | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it), [yifanyu/DiffusionGemma-26B-A4B-it-ORL](https://huggingface.co/yifanyu/DiffusionGemma-26B-A4B-it-ORL), [yifanyu/DiffusionGemma-26B-A4B-it-ORL-NVFP4](https://huggingface.co/yifanyu/DiffusionGemma-26B-A4B-it-ORL-NVFP4) | 256 | uniform-state renoising, linear or entropy-adaptive temperature |

Each family's decoders, precisions and options: [docs/models.md](docs/models.md).

## Performance

ORL checkpoints and decoders: the latency of one denoise round at an empty context, and request TPS on full
HumanEval.

| Model | GPU | Precision | Round, TP=1 | Round, TP=8 | TPS, TP=1 | TPS, TP=8 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| QwenDLM 4B + ORL | H100 | BF16 | 5.04 ms | 2.65 ms | 1609 | 3640 |
| LLaDA2.1-mini + ORL | H100 | BF16 | 4.99 ms | 2.18 ms | 2124 | 5121 |
| LLaDA2.1-mini + ORL | B200 | NVFP4 experts | 2.03 ms | 1.69 ms | 5046 | 6036 |
| LLaDA2.1-mini + ORL | RTX 5090 | NVFP4 experts | 3.47 ms | — | 2788 | — |
| DiffusionGemma 26B-A4B + ORL | H100 | BF16 | 16.96 ms | 5.36 ms | 1378 | 4384 |
| DiffusionGemma 26B-A4B + ORL | B200 | NVFP4 experts | 5.59 ms | 3.14 ms | 3949 | 7026 |
| DiffusionGemma 26B-A4B + ORL | RTX 5090 | NVFP4 experts | 15.41 ms | — | 1490 | — |

Each model runs the paper's decoder (LLaDA2: Quality mode; DiffusionGemma: 2048 new tokens). The TP=1 TPS are the
paper's: on H100, QwenDLM and LLaDA2 run the SGLang-numerics mode and DiffusionGemma averages seeds 1235, 1239 and 1241;
on one B200, DiffusionGemma averages seeds 1234–1236. The round latencies and TP=8 run dlmrt's default numerics
(DiffusionGemma at seed 1234), which reach 1658 (QwenDLM), 2060 (LLaDA2) and 1388 (DiffusionGemma) request TPS at TP=1
on H100. Methodology and every measurement: [docs/performance.md](docs/performance.md).

## Installation

dlmrt runs on Linux with Python 3.10+ and compiles its CUDA extensions on first use. It needs PyTorch 2.11, Triton 3.6
and a CUDA toolkit (12.9+ for Blackwell, 13.0+ for B300).

The SGLang image `lmsysorg/sglang:v0.5.13` ships every dependency; install dlmrt into its system Python:

```bash
git clone https://github.com/Introspective-Diffusion/DLMRT && cd DLMRT
pip install --no-deps --break-system-packages -e .
```

In another environment, pip installs the dependencies:

```bash
pip install -e .                    # torch, triton, DeepGEMM (sgl-deep-gemm), FA3 (sglang-kernel), transformers
pip install -e ".[blackwell]"       # + FlashInfer: NVFP4 experts on Blackwell
pip install -e ".[sglang-numerics]" # + FlashInfer 0.6.12: the QwenDLM / LLaDA2 SGLang-numerics modes
pip install -e ".[bench]"           # + pyarrow: dlmrt bench humaneval, benchmarks/humaneval_score.py
```

## Quickstart

### Python

```python
from dlmrt import LLM

llm = LLM("google/diffusiongemma-26B-A4B-it")  # a local directory or a Hugging Face repo id
out = llm.generate("Write a Python function that checks whether a number is prime.", max_new_tokens=2048)
print(out.text)
print(f"{out.request_tps:.0f} tok/s, {sum(out.denoise_steps)} denoise rounds")
```

For tensor parallelism, run the same program under `torchrun --nproc-per-node 8`: every rank builds the same `LLM`
and makes the same calls ([examples/offline_inference.py](examples/offline_inference.py)).

### Server

Streaming NDJSON over HTTP, one request at a time ([docs/server.md](docs/server.md)):

```bash
torchrun --nproc-per-node 8 -m dlmrt serve --model CKPT --port 18080 [--role orl] [--precision nvfp4]
curl -N localhost:18080/generate -d '{"messages": [{"role": "user", "content": "Hi!"}], "max_new_tokens": 512}'
```

### Benchmarks

```bash
python -m dlmrt bench step --model CKPT    # round latency at fixed contexts
# HumanEval generation (QwenDLM, DiffusionGemma; LLaDA2: docs/benchmarks.md)
torchrun --nproc-per-node 8 -m dlmrt bench humaneval --model CKPT --inputs openai_humaneval.parquet \
    --out results/he --full --repeats 1 --max-new-tokens 2048 [--role orl]
python benchmarks/humaneval_score.py openai_humaneval.parquet results/he   # runs model-written code: use a sandbox
```

## Reproducing the papers

`dlmrt bench suite` runs the papers' generation protocols over their problem files. The commands for every row of
the results table, the settings and the scoring are in
[docs/benchmarks.md](docs/benchmarks.md#reproducing-the-papers-table).

## Documentation

| | |
| --- | --- |
| [Architecture](docs/architecture.md) | the block graph, the device loop, kernels, tensor parallelism |
| [Models](docs/models.md) | the three families, their decoders, precisions and options |
| [Performance](docs/performance.md) | latency, throughput and accuracy |
| [Benchmarks](docs/benchmarks.md) | benchmark protocols and reproducing the papers |
| [Server](docs/server.md) | `dlmrt serve` and its HTTP API |
| [Configuration](docs/configuration.md) | environment variables |
| [Contributing](CONTRIBUTING.md) | development setup, tests and the numerics policy |

## Repository layout

```
dlmrt/
  llm.py           LLM: the Python API (also used by the server)
  models/          sdar/, llada2/, diffusion_gemma/: config, model and engine per family, kernels where it has its own
  layers/          Linear (DeepGEMM BF16 / FP8, cuBLAS, a Triton GEMM in cuBLAS's B200 order)
  quantization/    NVFP4 (ModelOpt exports): dequantization, expert layouts for the TRT-LLM-gen, CUTLASS and Triton MoEs
  distributed/     the tensor / expert parallel group and the fused reduction
  runtime/         shared engine pieces (Generation, check_prompt)
  ops/             triton/ shared kernels, csrc/ CUDA sources, extensions.py native loader,
                   fa3_tma.py FlashInfer's FA3 with the paged kernel's numerics (SGLang-numerics modes)
  entrypoints/     `dlmrt bench`, `dlmrt serve`
  envs.py          every DLMRT_* environment variable
benchmarks/        scoring, TPF / TPS of suite records, token-identity comparison, kernel sweeps
examples/          offline inference
tests/             kernel tests and checkpoint-gated parity tests
tools/             check_imports.py (the static import check CI runs), architecture_svg.py (the block-graph figure)
```

## Citation

```bibtex
@misc{yu2026orl,
  title         = {On-Policy Residual Learning for Diffusion Language Models},
  author        = {Yu, Yifan and Jian, Yuqing and Li, Jisen and Hu, Yuezhou and Zhang, Tianyi and Wu, Xiaoxia and
                   Zhu, Rui-Jie and Wang, Junxiong and Zhou, Zhongzhu and Lai, Fan and Zou, James and Xu, Chenfeng},
  year          = {2026},
  eprint        = {2610.20687},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.20687}
}
```

## License

Apache License 2.0 ([LICENSE](LICENSE)). Vendored third-party code keeps its license: ThunderKittens and a modified
DeepGEMM kernel (MIT), kernels adapted from SGLang (Apache 2.0); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
