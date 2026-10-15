"""The precision switch of the MoE model families (DiffusionGemma, LLaDA2): which numerics a checkpoint runs at.

    bf16    BF16 weights, activations and KV cache: a BF16 checkpoint, or an NVFP4 export's experts dequantized.
    nvfp4   a ModelOpt NVFP4 export as quantized: W4A4 experts (NVFP4 weights and GEMM inputs) on Blackwell FP4 tensor
            cores (FlashInfer's TRT-LLM-gen MoE on SM 10.x; on SM 12.x dlmrt's Triton MoE for LLaDA2 and FlashInfer's
            CUTLASS MoE for DiffusionGemma)
            and an FP8 KV cache. Everything outside the routed experts is BF16 (as exported).

The default is the checkpoint's own precision.
"""

from . import nvfp4

PRECISIONS = ("bf16", "nvfp4")


def checkpoint_precision(path):
    return "nvfp4" if nvfp4.quant_config(path)[0] else "bf16"


def resolve(path, precision, device, backends=("trtllm",)):
    """The precision to run `path` at on `device`: `precision`, or the checkpoint's own when None. `backends`: the
    NVFP4 MoE backends the model family implements (see nvfp4.backend)."""
    checkpoint = checkpoint_precision(path)
    precision = precision or checkpoint
    if precision not in PRECISIONS or precision == "nvfp4" and checkpoint == "bf16":
        raise ValueError(f"precision {precision!r} is not available for a {checkpoint} checkpoint")
    if precision == "nvfp4" and nvfp4.backend(device) not in backends:
        gpus = " or ".join(nvfp4.BACKEND_GPUS[b] for b in backends)
        raise ValueError(f"precision 'nvfp4' of this model needs Blackwell FP4 tensor cores ({gpus}) and FlashInfer; "
                         "run the export at precision 'bf16' on other GPUs")
    return precision
