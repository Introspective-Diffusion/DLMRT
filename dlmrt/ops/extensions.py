"""Native code: CUDA extensions compiled on first use by torch's JIT (into $TORCH_EXTENSIONS_DIR), and DeepGEMM set up
with dlmrt's patched SM90 BF16 GEMM.

    block_graph()      the per-block CUDA graph with a device-side WHILE loop (every model)
    tp_comm()          the barrier-free fused tensor-parallel reduction (sm_90a and later)
    tk()               the ThunderKittens fused gate_up + SwiGLU (sm_90a)
    ogud()             the SDAR post-attention megakernel (sm_90a)
    init_deep_gemm()   DeepGEMM's JIT root and PDL switch; called by every model before its first GEMM
"""

import hashlib
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

from .. import envs
from .triton.common import PDL

CSRC = Path(__file__).with_name("csrc")
# Host flags torch's extension build sets that the device code here needs undone.
_HALF_OPS = ["-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
             "-U__CUDA_NO_HALF2_OPERATORS__"]


def _build(name, sources, cflags, include_paths=()):
    from torch.utils.cpp_extension import load

    return load(name=name, sources=[str(CSRC / s) for s in sources],
                extra_include_paths=[str(p) for p in include_paths], extra_cuda_cflags=["-O3", *cflags])


def _arch():
    """This GPU's arch-specific target (sm_90a, sm_100a, sm_103a): multimem and griddepcontrol need it."""
    return "{}{}a".format(*torch.cuda.get_device_capability())


@lru_cache(maxsize=1)
def block_graph():
    return _build("dlmrt_block_graph", ["block_graph.cu"], ["-std=c++17"])


@lru_cache(maxsize=1)
def tp_comm():
    arch = _arch()
    return _build(f"dlmrt_tp_comm_sm{arch}", ["tp_comm.cu"],
                  ["-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}"])


@lru_cache(maxsize=1)
def tk():
    return _build("dlmrt_tk", ["ops.cpp", "gate_up_swiglu.cu"],
                  ["-std=c++20", "-DKITTENS_HOPPER", "-gencode=arch=compute_90a,code=sm_90a", "--expt-extended-lambda",
                   "--expt-relaxed-constexpr", *_HALF_OPS], [CSRC])


@lru_cache(maxsize=1)
def ogud():
    import deep_gemm

    return _build("dlmrt_ogud", ["ogud.cu"],
                  ["-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "--expt-extended-lambda",
                   "--expt-relaxed-constexpr", "--ptxas-options=--register-usage-level=10",
                   "--diag-suppress=39,161,174,177,186,940", *_HALF_OPS],
                  [CSRC, Path(deep_gemm.__file__).with_name("include")])


def _patch_deep_gemm(deep_gemm):
    """Point DeepGEMM's JIT at a library root whose SM90 BF16 GEMM is dlmrt's patched copy (csrc/deep_gemm):
    DeepGEMM keeps choosing configs, building tensor maps and launching; only the kernel body differs."""
    patched = CSRC / "deep_gemm" / "sm90_bf16_gemm.cuh"
    stock = Path(deep_gemm.__file__).parent / "include"
    digest = hashlib.sha256(patched.read_bytes() + str(stock).encode()).hexdigest()[:16]
    cache = envs.cache_dir()
    root = cache / f"deep_gemm-{digest}"
    if not root.exists():
        cache.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=cache))
        impls = staging / "include" / "deep_gemm" / "impls"
        impls.mkdir(parents=True)
        for entry in stock.iterdir():
            if entry.name != "deep_gemm":
                (staging / "include" / entry.name).symlink_to(entry)
        for entry in (stock / "deep_gemm").iterdir():
            if entry.name != "impls":
                (staging / "include" / "deep_gemm" / entry.name).symlink_to(entry)
        for entry in (stock / "deep_gemm" / "impls").iterdir():
            if entry.name != patched.name:
                (impls / entry.name).symlink_to(entry)
        shutil.copy(patched, impls / patched.name)
        try:
            staging.rename(root)
        except OSError:  # another process won the race
            shutil.rmtree(staging)
    # Separate JIT cache: the cache key does not cover the included headers.
    os.environ["DG_JIT_CACHE_DIR"] = str(root / "jit")
    deep_gemm._C.init(str(root), deep_gemm._find_cuda_home())


@lru_cache(maxsize=1)
def init_deep_gemm():
    """DeepGEMM with dlmrt's patched SM90 BF16 GEMM (DLMRT_DEEP_GEMM_PATCH=0: stock) and the PDL switch. Idempotent;
    must run before DeepGEMM compiles its first kernel."""
    import deep_gemm

    if envs.DEEP_GEMM_PATCH():
        _patch_deep_gemm(deep_gemm)
    deep_gemm._C.set_pdl(PDL)
    return deep_gemm
