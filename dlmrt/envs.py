"""Every environment variable dlmrt reads, in one place (see docs/configuration.md).

The defaults are the tuned configurations; the switches exist for A/B measurements and for hardware where a path does
not apply. Each accessor reads the environment when called (on/off and named switches raise ValueError on a value
they do not list); modules read most switches once, at import (DLMRT_LLADA_SPLITS and DLMRT_DG_SPLITS when an engine
is built, DLMRT_CACHE_DIR and DLMRT_DEEP_GEMM_PATCH on first use).
"""

import os
from pathlib import Path


def _bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}={value!r}: expected 1/true/yes/on or 0/false/no/off")


def _choice(name, default, choices):
    """A string switch; "" (where accepted) means the consumer's automatic choice."""
    value = os.environ.get(name, default)
    if value not in choices:
        raise ValueError(f"{name}={value!r}: expected one of {choices}")
    return value


def _optional_int(name):
    value = os.environ.get(name)
    return int(value) if value else None


# General


def cache_dir():
    """DLMRT_CACHE_DIR: autotuned MoE tactics and the patched DeepGEMM root (default ~/.cache/dlmrt)."""
    return Path(os.environ.get("DLMRT_CACHE_DIR", Path.home() / ".cache" / "dlmrt"))


def PDL():
    """DLMRT_PDL=0: launch kernels without programmatic dependent launch."""
    return _bool("DLMRT_PDL", True)


def DEEP_GEMM_PATCH():
    """DLMRT_DEEP_GEMM_PATCH=0: stock DeepGEMM SM90 BF16 GEMM instead of dlmrt's patched copy."""
    return _bool("DLMRT_DEEP_GEMM_PATCH", True)


# Tensor parallelism


def TP_FUSED():
    """DLMRT_TP_FUSED=0: torch symmetric-memory all-reduce + RMSNorm instead of dlmrt's fused reduction."""
    return _bool("DLMRT_TP_FUSED", True)


def TP_ALL_REDUCE():
    """DLMRT_TP_ALL_REDUCE: multimem (NVLink SHARP) or two_shot, for the unfused all-reduce."""
    return _choice("DLMRT_TP_ALL_REDUCE", "multimem", ("multimem", "two_shot"))


def TP_PUSH_BLOCKS():
    """DLMRT_TP_PUSH_BLOCKS: CTAs of the fused reduction that only push partial rows over NVLink (default 96)."""
    return int(os.environ.get("DLMRT_TP_PUSH_BLOCKS", 96))


# SDAR


def SDAR_SPLIT_CONTEXT():
    """DLMRT_SPLIT_CONTEXT: context length from which FA3 splits the keys (default 1536)."""
    return int(os.environ.get("DLMRT_SPLIT_CONTEXT", 1536))


def SDAR_MLP():
    """DLMRT_MLP: fused (gate_up + SwiGLU kernel, then down) or separate (DeepGEMM, SwiGLU, DeepGEMM)."""
    return _choice("DLMRT_MLP", "fused", ("fused", "separate"))


def SDAR_OGUD():
    """DLMRT_OGUD=0: separate kernels instead of the one-launch post-attention chain (SDAR-4B on H100)."""
    return _bool("DLMRT_OGUD", True)


# LLaDA2


def LLADA_OVERLAP():
    """DLMRT_LLADA_OVERLAP=0: the shared expert inline instead of on a side stream."""
    return _bool("DLMRT_LLADA_OVERLAP", True)


def LLADA_MOE():
    """DLMRT_LLADA_MOE: routed-expert GEMMs of decode rounds, triton (small-tile grouped) or deepgemm (masked)."""
    return _choice("DLMRT_LLADA_MOE", "triton", ("triton", "deepgemm"))


def LLADA_REDUCE():
    """DLMRT_LLADA_REDUCE: cross-rank sum under TP: fused (default), one_shot, or multimem (torch all-reduce)."""
    return _choice("DLMRT_LLADA_REDUCE", "fused", ("fused", "one_shot", "multimem"))


def LLADA_SPLITS():
    """DLMRT_LLADA_SPLITS: force one FA3 key-split count (default: chosen by context length)."""
    return _optional_int("DLMRT_LLADA_SPLITS")


def LLADA_ATTENTION():
    """DLMRT_LLADA_ATTENTION: fa3, triton, or b200 (SM 12.x: the Gluon kernel with the B200's summation order); default
    FA3 on Hopper, the Triton kernel elsewhere."""
    return _choice("DLMRT_LLADA_ATTENTION", "", ("", "fa3", "triton", "b200"))


def LLADA_FP4_MOE():
    """DLMRT_LLADA_FP4_MOE: NVFP4 routed experts on SM 12.x, triton (the TRT-LLM-gen MoE's roundings, default) or
    cutlass (FlashInfer's CUTLASS MoE, which rounds GEMM1 and SwiGLU outputs to BF16)."""
    return _choice("DLMRT_LLADA_FP4_MOE", "triton", ("triton", "cutlass"))


# DiffusionGemma


def DG_ATTENTION():
    """DLMRT_DG_ATTENTION: fa3 or triton for the sliding layers (default: FA3 on Hopper, Triton elsewhere)."""
    return _choice("DLMRT_DG_ATTENTION", "", ("", "fa3", "triton"))


def DG_FUSED_QUANT():
    """DLMRT_DG_FUSED_QUANT=0: quantize the NVFP4 experts' input in a separate pass."""
    return _bool("DLMRT_DG_FUSED_QUANT", True)


def DG_OVERLAP():
    """DLMRT_DG_OVERLAP=0: the dense MLP inline instead of on a side stream next to the experts."""
    return _bool("DLMRT_DG_OVERLAP", True)


def DG_SPLITS():
    """DLMRT_DG_SPLITS: FA3 key splits of the sliding layers (default 1 on one GPU, 4 under TP)."""
    return _optional_int("DLMRT_DG_SPLITS")
