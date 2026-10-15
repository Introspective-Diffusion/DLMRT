"""SDAR (dense, Qwen3 architecture) block-diffusion models with the paper's greedy block decoder.

    config.py           checkpoint contract (ModelConfig), decoder (DecodeConfig, the paper's ROLES), prompt wrapper
                        (render_prompt)
    model.py            Model: weights, KV arena, the 256-row forward (BF16 or FP8 projections), tensor parallel
    engine.py           Engine: one request at a time, one CUDA graph per block
    sglang_numerics.py  SglangSdarModel / SglangSdarEngine: the paper runtime's numerics (--numerics sglang)
    sglang_kernels.py   Triton kernels that reproduce its reductions and decisions bit for bit
"""

from .config import ROLES, DecodeConfig, ModelConfig, render_prompt
from .engine import Engine
from .model import Model

__all__ = ["ROLES", "DecodeConfig", "Engine", "Model", "ModelConfig", "render_prompt"]
