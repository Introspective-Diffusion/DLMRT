"""dlmrt: a single-request inference engine for block-diffusion language models (QwenDLM, LLaDA2, DiffusionGemma),
on one GPU or tensor parallel over the GPUs of an NVLink node."""

from .llm import LLM, RequestOutput
from .runtime.generation import Generation
from .version import __version__

__all__ = ["LLM", "Generation", "RequestOutput", "__version__"]
