"""FlashInfer FA3's paged-prefill attention (BF16, head_dim 128, non-causal, page size 1) bitwise, on a contiguous KV
cache, faster: FlashInfer's ragged kernel (TMA loads instead of the paged kernel's per-row gathers) built with the paged
kernel's KV tile (96), so every query row meets the same KV tiles in the same order, and with 64-row query tiles (one
consumer warpgroup per CTA, twice the CTAs). Built from FlashInfer's own sources by its JIT on first use; the SGLang-
numerics modes run their non-causal attention here.
"""

import hashlib
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import flashinfer
import flashinfer.prefill as fp
import torch

from .. import envs

Q_TILE, KV_TILE = 64, 96
# get_batch_prefill_module's arguments for BF16, int32 indptr, head_dim 128, no RoPE / window / soft cap / FP16 QK.
ARGS = (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, 128, 128, 0, False, False, False)
_get_module = fp.get_batch_prefill_module


def _replace(text, old, new, count=1):
    found = text.count(old)
    if found != count:
        raise RuntimeError(f"FlashInfer {flashinfer.__version__} source differs from the patched one "
                           f"({found} matches, expected {count}): {old[:60]!r}")
    return text.replace(old, new)


def _patch(include):
    hopper = include / "flashinfer" / "attention" / "hopper"
    f = hopper / "prefill_sm90.cuh"
    text = _replace(f.read_text(), "      } else {\n        return std::make_tuple(128, 192);\n      }",
                    f"      }} else {{\n        return std::make_tuple({Q_TILE}, {KV_TILE});\n      }}")
    # One consumer warpgroup: give it the register budget of the two-consumer layout.
    text = _replace(text, "warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 240 : 160>",
                    "warpgroup_reg_alloc<Ktraits::NUM_WARPS <= 12 ? 240 : 160>")
    f.write_text(text)
    # The ping-pong barrier orders two consumer warpgroups' issue; with one it is off (no effect on the arithmetic).
    f = hopper / "mainloop.cuh"
    f.write_text(_replace(f.read_text(), "static constexpr bool UseSchedulerBarrier =\n      "
                          "cutlass::sizeof_bits_v<DTypeQ> == 8 ? HEAD_DIM_VO >= 128 : HEAD_DIM_VO <= 128;",
                          "static constexpr bool UseSchedulerBarrier =\n      size(typename Ktraits::TiledMmaQK{}) > "
                          "cutlass::NumThreadsPerWarpGroup &&\n      (cutlass::sizeof_bits_v<DTypeQ> == 8 ? "
                          "HEAD_DIM_VO >= 128 : HEAD_DIM_VO <= 128);"))
    f = hopper / "named_barrier.cuh"
    f.write_text(_replace(f.read_text(), "static_assert(NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup ||",
                          "static_assert(!UseSchedulerBarrier || NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup ||",
                          count=2))
    # The SM90 plan's query tile: this module's tile schedule is the patched non-causal ragged kernel's only.
    f = include / "flashinfer" / "attention" / "scheduler.cuh"
    text, n = re.subn(r"int cta_tile_q = 128;", f"int cta_tile_q = {Q_TILE};", f.read_text())
    if n != 1:
        raise RuntimeError(f"FlashInfer {flashinfer.__version__}: {n} SM90 query-tile sites, expected 1")
    f.write_text(text)


@lru_cache(maxsize=1)
def module():
    """The patched batch-prefill module, as FlashInfer's wrappers call it (plan / ragged_run)."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.attention import modules as jm
    from flashinfer.jit.core import gen_jit_spec

    stock_uri = jm.get_batch_prefill_uri("fa3", *ARGS)
    uri = f"{stock_uri}_dlmrt_q{Q_TILE}kv{KV_TILE}"
    spec = jm.gen_batch_prefill_module("fa3", *ARGS)  # writes the generated sources
    digest = hashlib.sha256(f"{flashinfer.__version__}:{jit_env.FLASHINFER_INCLUDE_DIR}:{uri}".encode()
                            + Path(__file__).read_bytes()).hexdigest()[:16]
    cache = envs.cache_dir()
    root = cache / f"flashinfer-fa3-{digest}"
    if not root.exists():
        cache.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=cache))
        shutil.copytree(jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer", staging / "include" / "flashinfer")
        _patch(staging / "include")
        (staging / "src").mkdir()
        for source in spec.sources:
            shutil.copy(source, staging / "src" / Path(source).name)
        shutil.copy(jit_env.FLASHINFER_GEN_SRC_DIR / stock_uri / "batch_prefill_sm90_config.inc", staging / "src")
        try:
            staging.rename(root)
        except OSError:  # another process won the race
            shutil.rmtree(staging)
    sources = [root / "src" / Path(source).name for source in spec.sources]
    custom = gen_jit_spec(uri, sources, extra_cuda_cflags=jm.sm90a_nvcc_flags,
                          extra_include_paths=[root / "include", root / "src"])
    stock_gen, stock_get_uri = fp.gen_batch_prefill_module, fp.get_batch_prefill_uri
    fp.gen_batch_prefill_module = lambda backend, *args: custom
    fp.get_batch_prefill_uri = lambda backend, *args: uri
    try:
        return _get_module.__wrapped__("fa3", *ARGS)
    finally:
        fp.gen_batch_prefill_module, fp.get_batch_prefill_uri = stock_gen, stock_get_uri


class PagedNumericsWrapper(flashinfer.BatchPrefillWithRaggedKVCacheWrapper):
    """A ragged FA3 wrapper whose (non-causal) attention is bitwise BatchPrefillWithPagedKVCacheWrapper's with page
    size 1 over the same rows: plan(qo_indptr, kv_indptr, ...) with kv_indptr over a contiguous [slots, heads, dim]
    cache, then run(q, k, v)."""

    def __init__(self, workspace, **kwargs):
        super().__init__(workspace, "NHD", backend="fa3", **kwargs)

    def plan(self, qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk, *, causal=False, **kwargs):
        if causal:
            raise ValueError("PagedNumericsWrapper is non-causal only")
        ours = module()

        def patched(backend, *args):
            if (backend, args) != ("fa3", ARGS):
                raise ValueError(f"PagedNumericsWrapper supports {ARGS}, not {(backend, *args)}")
            return ours

        fp.get_batch_prefill_module = patched
        try:
            super().plan(qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim_qk, causal=False, **kwargs)
        finally:
            fp.get_batch_prefill_module = _get_module
