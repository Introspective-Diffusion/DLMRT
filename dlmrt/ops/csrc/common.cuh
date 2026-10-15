// Shared pieces of the ThunderKittens kernels: TMA, pipeline phase bits, SwiGLU with Triton's rounding, host setup.
#pragma once

#include "host.cuh"
#include "tk/kittens.cuh"

namespace dlmrt {

using namespace kittens;

constexpr int KSTEP = 64;  // K per pipeline stage: one 128B swizzle atom of BF16

// Parity bits of up to 16 barriers per half: half 0 tracks `arrived`, half 1 tracks `finished`.
__device__ __forceinline__ bool phase(uint32_t bits, int half, int i) { return (bits >> (half * 16 + i)) & 1; }
__device__ __forceinline__ void flip(uint32_t& bits, int half, int i) { bits ^= 1u << (half * 16 + i); }

// L2 cache policies (CuTe's SM90 encodings): stream-once weights evict first; activations that many CTAs re-read
// and outputs the next kernel reads evict last.
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ull, EVICT_LAST = 0x14F0000000000000ull;

// One st_bf<rows, 64> tile at (row, K chunk) of a tensor encoded by `encode`.
__device__ __forceinline__ void tma_load(void* dst, const CUtensorMap* desc, int row, int chunk, semaphore& bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&bar))), "r"(chunk * KSTEP), "r"(row)
        : "memory");
}

__device__ __forceinline__ void tma_load(void* dst, const CUtensorMap* desc, int row, int chunk, semaphore& bar,
                                         uint64_t policy) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&bar))), "r"(chunk * KSTEP), "r"(row), "l"(policy)
        : "memory");
}

// A [rows, cols] box of shared memory to (col, row) of a tensor encoded with `encode_plain`, kept in L2 for the next
// kernel; the thread then waits until the box has been read, so the shared memory may be reused or released.
// Out-of-range elements are dropped.
__device__ __forceinline__ void tma_store(const CUtensorMap* desc, const void* src, int col, int row) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%2, %3}], [%1], %4;"
                 :: "l"(reinterpret_cast<uint64_t>(desc)), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src))),
                    "r"(col), "r"(row), "l"(EVICT_LAST)
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;\ncp.async.bulk.wait_group.read 0;" ::: "memory");
}

// Generic-proxy shared-memory writes become visible to TMA (async proxy) reads.
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__device__ __forceinline__ void named_barrier(int id, int threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void grid_dependency_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Triton's tl.sigmoid on sm_90: 1 / (1 + 2^(-x * log2 e)) with ex2.approx and div.full.
__device__ __forceinline__ float triton_sigmoid(float x) {
    float t = (0.0f - x) * __uint_as_float(0x3FB8AA3Bu), e, r;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(1.0f), "f"(e + 1.0f));
    return r;
}

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// silu(bf16(g)) * bf16(u) as GEMM + Triton SwiGLU computes it; ROUND rounds silu to BF16 first
// (the self-conditioning adapter).
template <bool ROUND>
__device__ __forceinline__ float swiglu1(float g, float u) {
    g = bf16_round(g);
    float silu = g * triton_sigmoid(g);
    return (ROUND ? bf16_round(silu) : silu) * bf16_round(u);
}

template <bool ROUND>
__device__ __forceinline__ __nv_bfloat162 swiglu2(float2 g, float2 u) {
    return __floats2bfloat162_rn(swiglu1<ROUND>(g.x, u.x), swiglu1<ROUND>(g.y, u.y));
}

// st_bf<tile_rows, 64> tiles (128B swizzle; lands in shared memory exactly as ThunderKittens' layout).
inline CUtensorMap encode(const at::Tensor& t, uint64_t rows, uint64_t cols, uint32_t tile_rows) {
    TORCH_CHECK(cols % 64 == 0);
    return encode_boxes(t, rows, cols, tile_rows, 64, CU_TENSOR_MAP_SWIZZLE_128B);
}

// Dense row-major [box_rows, box_cols] boxes, for TMA stores from plain shared-memory arrays.
inline CUtensorMap encode_plain(const at::Tensor& t, uint64_t rows, uint64_t cols, uint32_t box_rows,
                                uint32_t box_cols) {
    return encode_boxes(t, rows, cols, box_rows, box_cols, CU_TENSOR_MAP_SWIZZLE_NONE);
}

}  // namespace dlmrt
