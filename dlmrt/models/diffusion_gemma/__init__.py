"""DiffusionGemma (google/diffusiongemma-26B-A4B-it and its fine-tunes) on dlmrt.

    config.py     checkpoint contract (GemmaConfig), decoder settings (GemmaDecode, ROLES)
    precision.py  the precision switch (bf16 / nvfp4) and the checkpoint reading it governs
    experts.py    routed-expert backends: DeepGEMM BF16, NVFP4 on TRT-LLM-gen (SM 10.x) or CUTLASS (SM 12.x)
    attention.py  attention backends: FA3 (Hopper sliding layers), Triton
    model.py      GemmaModel: weights, KV cache, the forward of one canvas or of several requests' rows
    kernels.py    Triton kernels of the forward, the decoder and the batched engine's stages
    engine.py     GemmaEngine: one request at a time, one CUDA graph per canvas
    batched.py    BatchedGemmaEngine: many requests per forward on one GPU, each at its own stage and canvas length
"""

from ...quantization.precision import PRECISIONS
from .config import ROLES, GemmaConfig, GemmaDecode
from .model import GemmaModel, softcap

__all__ = ["PRECISIONS", "ROLES", "GemmaConfig", "GemmaDecode", "GemmaModel", "softcap"]
