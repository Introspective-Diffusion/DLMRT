"""Linear layers on DeepGEMM (BF16 / FP8 block-scaled), cuBLAS, and on SM 12.x a Triton GEMM in cuBLAS's B200
summation order."""

import deep_gemm
import torch

from ..ops.extensions import init_deep_gemm
from ..ops.triton import gemm
from ..ops.triton.core import GROUP

# cuBLAS's K splits of BF16 GEMMs on B200 (cuBLAS 13.1) by (N, K): forward rows -> split. Every other shape sums K in
# one range. Measured on LLaDA2's GEMMs against dlmrt.ops.triton.gemm.
B200_SPLITS = {(256, 2048): lambda m: 2 if m <= 128 else 1, (2048, 5120): lambda m: 2 if m <= 64 else 1}


class Linear:
    """y = x @ W^T with DeepGEMM: BF16, or FP8 with 1x128 activation and 128x128 weight scales. BF16 runs on cuBLAS
    on Blackwell, where it beats DeepGEMM's BF16 kernels by 20-40% at a few hundred rows. `b200`: on SM 12.x, BF16
    in cuBLAS's B200 summation order instead (dlmrt.ops.triton.gemm), bitwise the B200's result."""

    def __init__(self, weight, fp8, b200=False):
        init_deep_gemm()
        self.n, self.k = weight.shape
        self.fp8 = fp8
        major = torch.cuda.get_device_capability(weight.device)[0]
        self.b200 = not fp8 and b200 and major == 12
        self.split = B200_SPLITS.get((self.n, self.k), lambda m: 1)
        self.cublas = not fp8 and major >= 10
        if fp8:
            self.weight = deep_gemm.per_block_cast_to_fp8(weight, use_ue8m0=False)
        else:
            self.weight = weight.contiguous()

    def input(self, rows, device):
        """Activation buffers this layer consumes: BF16, or FP8 values plus K-group-major scales."""
        if not self.fp8:
            return None, None
        return (torch.empty(rows, self.k, device=device, dtype=torch.float8_e4m3fn),
                torch.empty(self.k // GROUP, rows, device=device, dtype=torch.float32))

    def __call__(self, x, scales=None, out=None):
        out = torch.empty(x.shape[0], self.n, device=x.device, dtype=torch.bfloat16) if out is None else out
        if self.fp8:
            deep_gemm.fp8_gemm_nt((x, scales.t()), self.weight, out)
        elif self.b200:
            gemm.mm(x, self.weight, out, split=self.split(x.shape[0]))
        elif self.cublas:
            torch.mm(x, self.weight.t(), out=out)
        else:
            deep_gemm.bf16_gemm_nt(x, self.weight, out)
        return out
