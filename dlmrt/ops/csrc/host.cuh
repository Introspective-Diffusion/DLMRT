// Host-side setup shared by the CUDA extensions (no device-library headers): TMA descriptors and launches.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace dlmrt {

// [box_rows, box_cols] boxes of a row-major BF16 [rows, cols] tensor; out-of-range elements read as zeros and are
// dropped on stores. 256B L2 promotion as DeepGEMM: a 64-wide box row is 128B, so the next K step's chunk arrives
// with it.
inline CUtensorMap encode_boxes(const at::Tensor& t, uint64_t rows, uint64_t cols, uint32_t box_rows, uint32_t box_cols,
                                CUtensorMapSwizzle swizzle) {
    using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static EncodeFn encoder = [] {
        EncodeFn fn = nullptr;
        cudaDriverEntryPointQueryResult status;
        TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", reinterpret_cast<void**>(&fn), cudaEnableDefault,
                                            &status) == cudaSuccess && fn, "cuTensorMapEncodeTiled unavailable");
        return fn;
    }();
    TORCH_CHECK(t.is_contiguous() && t.scalar_type() == at::kBFloat16 && cols % 8 == 0);
    CUtensorMap map;
    cuuint64_t shape[2] = {cols, rows};
    cuuint64_t stride[1] = {cols * 2};
    cuuint32_t box[2] = {box_cols, box_rows};
    cuuint32_t elem[2] = {1, 1};
    TORCH_CHECK(encoder(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), shape, stride, box, elem,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "tensor map");
    return map;
}

// Launch with a cluster shape and, when `pdl`, programmatic dependent launch. Launches happen at warm-up and graph
// capture (replays do not come back here) and eagerly in tests and the SGLang-numerics prefill, so the shared-memory
// attribute is simply set on every launch.
template <typename Kernel, typename... Args>
void launch(Kernel kernel, int grid, int threads, int smem, int cluster, bool pdl, Args... args) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {static_cast<unsigned>(cluster), 1, 1};
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cfg.attrs = attrs;
    cfg.numAttrs = pdl ? 2 : 1;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...) == cudaSuccess, "launch");
}

}  // namespace dlmrt
