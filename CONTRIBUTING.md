# Contributing to dlmrt

## Development setup

```bash
pip install -e ".[dev]"            # inside an environment with CUDA, PyTorch 2.11, Triton 3.6 (e.g. lmsysorg/sglang)
pre-commit install                 # ruff on every commit
```

The first run of anything that touches the GPU compiles dlmrt's CUDA extensions (several minutes for the
ThunderKittens and CuTe kernels) and Triton kernels; both are cached ([configuration](docs/configuration.md)).

## Tests

```bash
python -m pytest tests -q                                    # kernel tests: one GPU, no checkpoint
DLMRT_LLADA_CHECKPOINT=LLADA_CKPT DLMRT_DG_CHECKPOINT=DG_CKPT python -m pytest tests -q
```

Kernel tests compare every kernel against a PyTorch reference (most bit for bit); Hopper- or Blackwell-only kernels
skip elsewhere. Parity tests load a checkpoint on two GPUs and compare the forward with the Hugging Face
implementation; they skip without their `DLMRT_*_CHECKPOINT`.

## The numerics policy

dlmrt's kernels reproduce the reference implementations' BF16 roundings, and decoding is chaotic: a one-ULP change
in a logit changes which positions a round accepts and therefore the rest of the generation. So every change falls
in one of two classes, and the pull request says which:

1. **Bit-identical** (refactors, scheduling, fusions that keep the arithmetic, launch configuration that does not
   change reduction orders). Show token identity: the same protocol and seed (`dlmrt bench humaneval`; LLaDA2:
   `dlmrt bench suite` over HumanEval) on the old and new code, compared with `benchmarks/compare_runs.py` (all tasks
   token- and step-identical), for every model family the change touches, at TP=1 and TP=8.
   `benchmarks/kernel_dump.py` compares DiffusionGemma's row-kernel outputs bit for bit.
2. **Numerics-changing** (different tiles or reduction orders, precision changes). Show accuracy and tokens per denoise
   round over several seeds against the previous version, and time per forward for the speed claim. A single seed
   cannot rank two engines: the seed-to-seed spread is several HumanEval points.

Report speed as round latency (`dlmrt bench step`) and request TPS; for numerics-changing work compare time per
forward, since tokens per denoise round moves with the trajectory.

The SGLang-numerics modes (`--numerics sglang`) admit only the first class: their kernels must reproduce the paper
runtime's ops (`tests/test_*_sglang_numerics.py`), and a `bench suite` run must stay token-identical with the previous
one.

## New kernels

A GPU fault (illegal address, Xid 43) can take a node out of service on managed clusters. Run new kernels and new
code paths under `compute-sanitizer --tool memcheck --report-api-errors no` on one GPU before running them normally.
Kernels that spin on other ranks (the TP reduction) cannot run under the sanitizer: review their indexing with care.

## Code style

- `ruff check .` must pass (configuration in `pyproject.toml`; 128 columns).
- Comments state what the code cannot: a constraint, a measured reason, a numerics requirement. Keep measurements
  that justify a design next to the code they justify.
- Every `DLMRT_*` environment variable goes through `dlmrt/envs.py` and `docs/configuration.md`.
- A new model family gets `dlmrt/models/<family>/{config,model,engine}.py` (and `kernels.py` if it needs its own), an
  entry in `dlmrt/registry.py`, a loader in `dlmrt/llm.py` and its benchmarks in `dlmrt/entrypoints/bench/` (with its
  commands in `dlmrt/entrypoints/bench/__init__.py`).
