// gate_up + SwiGLU in one H100 kernel for a 256-row block:
//   out[m, n] = bf16(silu(bf16(x @ Wg^T)) * bf16(x @ Wu^T)),  W = [Wg; Wu] stored [2F, K].
// Matches the separate GEMM + Triton SwiGLU bit for bit: K-sequential WGMMA with FP32
// accumulation, both products rounded to BF16, then Triton's sigmoid lowering.
//
// One task = all 256 rows x COLS SwiGLU columns, so each weight row is read once and
// ceil(F / COLS) tasks fill the GPU in a single wave. Warpgroup 2 streams A, Wg and Wu
// with TMA (weights before the grid dependency wait); warpgroups 0 and 1 each own 128 rows.
#include "common.cuh"

namespace dlmrt {
namespace {

constexpr int ROWS = 256, COLS = 80, STAGES = 4, THREADS = 384;
using a_tile = st_bf<64, KSTEP>;
using b_tile = st_bf<2 * COLS, KSTEP>;  // rows [0, COLS) gate, [COLS, 2 COLS) up: one WGMMA reads A once

struct stage_tiles {
    a_tile a[ROWS / 64];
    b_tile b;
};
static_assert(COLS * KSTEP * 2 % 1024 == 0, "the up half must start on a 128B-swizzle atom");
constexpr int STAGE_BYTES = sizeof(stage_tiles);

template <int CLUSTER>
__device__ __forceinline__ void tma_load_multicast(void* dst, const CUtensorMap* desc, int row, int chunk, semaphore& bar) {
    if constexpr (CLUSTER == 1) {
        tma_load(dst, desc, row, chunk, bar);
    } else {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4}], [%2], %5;"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(reinterpret_cast<uint64_t>(desc)),
               "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&bar))), "r"(chunk * KSTEP), "r"(row),
               "h"(static_cast<uint16_t>((1 << CLUSTER) - 1))
            : "memory");
    }
}

// Arrive on `bar` in CTA `peer` of the cluster, with the default .release.cta semantics as CUTLASS does.
__device__ __forceinline__ void arrive_remote(semaphore& bar, uint32_t peer) {
    const uint32_t local = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(peer));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}

// Arrive on the same barrier in every CTA of the cluster.
template <int CLUSTER>
__device__ __forceinline__ void arrive_cluster(semaphore& bar) {
    if constexpr (CLUSTER == 1) {
        arrive(bar);
    } else {
#pragma unroll
        for (uint32_t peer = 0; peer < CLUSTER; ++peer) arrive_remote(bar, peer);
    }
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void load_weights(b_tile& b, const CUtensorMap* desc, int col, int F, int k, semaphore& bar) {
    char* base = reinterpret_cast<char*>(&b);
    tma_load(base, desc, col, k, bar, EVICT_FIRST);
    tma_load(base + COLS * KSTEP * 2, desc, F + col, k, bar, EVICT_FIRST);
}

}  // namespace

// MODE (diagnostic): 0 multicast activations + every consumer warp releases the stage in every CTA,
// 1 the same release with CTA-local loads, 2 CTA-local loads and release,
// 3 multicast + one release per consumer warpgroup and peer (warp w signals CTA w),
// 5 no MMA (memory stream only), 6 no loads after the pipeline fill (compute only),
// 7 CTA-local with one step of wgmma kept in flight, 8 no loads at all (launch + MMA + epilogue),
// 9 epilogue without the SwiGLU math, 10 epilogue math without stores, 11 direct global stores instead of the
// shared-memory tile + TMA store.
template <int CLUSTER, int MODE, bool ROUND>
__global__ __launch_bounds__(THREADS, 1) void gate_up_swiglu_kernel(
    const __grid_constant__ CUtensorMap a_desc, const __grid_constant__ CUtensorMap w_desc,
    const __grid_constant__ CUtensorMap out_desc, __nv_bfloat16* __restrict__ out, int F, int K) {
    extern __shared__ int shm[];
    tma_swizzle_allocator allocator(shm);
    stage_tiles(&stages)[STAGES] = allocator.allocate<stage_tiles, STAGES>();
    __shared__ semaphore arrived[STAGES], finished[STAGES];
    if (threadIdx.x == 0) {
        for (int i = 0; i < STAGES; ++i) {
            init_semaphore(arrived[i], 0, 1);
            init_semaphore(finished[i], 0, MODE <= 1 ? 8 * CLUSTER : MODE == 3 ? 2 * CLUSTER : 8);
        }
    }
    __syncthreads();
    if constexpr (CLUSTER > 1) cluster_sync();  // peers' barriers exist before any multicast or remote arrive
    const int rank = CLUSTER > 1 ? cluster_rank() : 0;

    const int wg = warpgroup::groupid(), warp = warpgroup::warpid(), lane = laneid();
    const int col = blockIdx.x * COLS, steps = K / KSTEP;
    uint32_t bits = 0xFFFF0000;
    int stage = 0;

    if (wg == 2) {
        warpgroup::decrease_registers<40>();
        if (warp == 0 && lane == 0) {
            // The weights do not depend on the previous kernel: fill the pipeline with them first.
            const int ahead = MODE == 8 ? 0 : steps < STAGES ? steps : STAGES;
            for (int s = 0; s < ahead; ++s) {
                tma::expect_bytes(arrived[s], STAGE_BYTES);
                load_weights(stages[s].b, &w_desc, col, F, s, arrived[s]);
            }
            grid_dependency_wait();
            for (int k = 0; k < steps; ++k) {
                wait(finished[stage], phase(bits, 1, stage));
                flip(bits, 1, stage);
                if ((MODE == 6 || MODE == 8) && k >= ahead) {
                    arrive(arrived[stage]);
                    stage = (stage + 1) % STAGES;
                    continue;
                }
                if (k >= ahead) {
                    tma::expect_bytes(arrived[stage], STAGE_BYTES);
                    load_weights(stages[stage].b, &w_desc, col, F, k, arrived[stage]);
                }
                // Each CTA fetches its share of the activation tile once and multicasts it to the cluster.
                if constexpr (MODE == 0 || MODE == 3) {
                    for (int i = rank; i < ROWS / 64; i += CLUSTER)
                        tma_load_multicast<CLUSTER>(&stages[stage].a[i], &a_desc, i * 64, k, arrived[stage]);
                } else {
                    for (int i = 0; i < ROWS / 64; ++i)
                        tma_load(&stages[stage].a[i], &a_desc, i * 64, k, arrived[stage], EVICT_LAST);
                }
                stage = (stage + 1) % STAGES;
            }
        }
    } else {
        warpgroup::increase_registers<232>();
        rt_fl<16, 2 * COLS> acc0, acc1;
        zero(acc0);
        zero(acc1);
        int prev = 0;
        for (int k = 0; k < steps; ++k) {
            wait(arrived[stage], phase(bits, 0, stage));
            flip(bits, 0, stage);
            if constexpr (MODE != 5) {
                warpgroup::mma_ABt(acc0, stages[stage].a[2 * wg], stages[stage].b);
                warpgroup::mma_ABt(acc1, stages[stage].a[2 * wg + 1], stages[stage].b);
            }
            if constexpr (MODE == 7) {
                if (k > 0) {
                    warpgroup::mma_async_wait<2>();
                    if (lane == 0) arrive(finished[prev]);
                }
                prev = stage;
            } else {
                if constexpr (MODE != 5) warpgroup::mma_async_wait();
                if (lane == 0) {
                    if constexpr (MODE <= 1) arrive_cluster<CLUSTER>(finished[stage]);
                    else if constexpr (MODE == 3) {
                        if (warp < CLUSTER) arrive_remote(finished[stage], warp);
                    } else arrive(finished[stage]);
                }
            }
            stage = (stage + 1) % STAGES;
        }
        if constexpr (MODE == 7) warpgroup::mma_async_wait();
        // Fragment layout: data[0] (r, c), data[1] (r + 8, c), data[2] (r, c + 8), data[3] (r + 8, c + 8).
        const int r = lane / 4, c = 2 * (lane % 4);
        if constexpr (MODE == 9 || MODE == 10 || MODE == 11) {
            uint32_t sink = 0;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const auto& acc = half ? acc1 : acc0;
                const int row = wg * 128 + half * 64 + warp * 16 + r;
#pragma unroll
                for (int j = 0; j < COLS / 16; ++j) {
                    const int n = col + j * 16 + c;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int rr = row + (e & 1) * 8, nn = n + (e >> 1) * 8;
                        if constexpr (MODE == 9) {
                            if (nn < F)
                                *reinterpret_cast<__nv_bfloat162*>(out + rr * F + nn) =
                                    __float22bfloat162_rn(acc.tiles[0][j].data[e]);
                        } else if constexpr (MODE == 10) {
                            const auto v = swiglu2<ROUND>(acc.tiles[0][j].data[e], acc.tiles[0][j + COLS / 16].data[e]);
                            sink ^= *reinterpret_cast<const uint32_t*>(&v);
                        } else if (nn < F) {
                            *reinterpret_cast<__nv_bfloat162*>(out + rr * F + nn) =
                                swiglu2<ROUND>(acc.tiles[0][j].data[e], acc.tiles[0][j + COLS / 16].data[e]);
                        }
                    }
                }
            }
            if constexpr (MODE == 10)
                if (sink == 0x9E3779B9u) out[threadIdx.x] = __float2bfloat16(1.0f);  // keeps the math alive
            launch_dependents();
        } else {
            // Each warpgroup stages its 128 x COLS half of the tile in the idle pipeline memory and writes it with
            // one TMA store: full-line writes instead of scattered 4-byte stores. Columns past F are dropped.
            named_barrier(1, 256);  // both warpgroups are done reading every stage
            __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(&stages[0]) + wg * 128 * COLS;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const auto& acc = half ? acc1 : acc0;
                const int row = half * 64 + warp * 16 + r;
#pragma unroll
                for (int j = 0; j < COLS / 16; ++j) {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        *reinterpret_cast<__nv_bfloat162*>(tile + (row + (e & 1) * 8) * COLS + j * 16 + c + (e >> 1) * 8) =
                            swiglu2<ROUND>(acc.tiles[0][j].data[e], acc.tiles[0][j + COLS / 16].data[e]);
                }
            }
            fence_async_shared();
            named_barrier(2 + wg, 128);
            if (warp == 0 && lane == 0) {
                tma_store(&out_desc, tile, col, wg * 128);
                // Only after the epilogue: the idle producer threads would otherwise release the next kernel at
                // launch, and a DeepGEMM kernel resident that early runs ~12 us slower.
                launch_dependents();
            }
        }
    }
    if constexpr (CLUSTER > 1) cluster_sync();  // no CTA leaves while peers may still signal it
}

template <int CLUSTER, int MODE, bool ROUND>
static void launch_gate_up(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, bool pdl) {
    TORCH_CHECK(x.size(0) == ROWS && x.size(1) % KSTEP == 0 && w.size(1) == x.size(1) && w.size(0) % 2 == 0);
    const int K = x.size(1), F = w.size(0) / 2;
    TORCH_CHECK(out.size(0) == ROWS && out.size(1) == F && out.is_contiguous());
    const int tasks = (F + COLS - 1) / COLS;
    // Surplus CTAs of the last cluster still share activation loads.
    launch(gate_up_swiglu_kernel<CLUSTER, MODE, ROUND>, (tasks + CLUSTER - 1) / CLUSTER * CLUSTER, THREADS,
           STAGES * static_cast<int>(sizeof(stage_tiles)) + 1024, CLUSTER, pdl, encode(x, ROWS, K, 64),
           encode(w, 2 * F, K, COLS), encode_plain(out, ROWS, F, 128, COLS),
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), F, K);
}

void gate_up_swiglu(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, bool pdl, int cluster, int mode,
                    bool round_silu) {
    if (cluster == 1 && mode == 5) launch_gate_up<1, 5, false>(x, w, out, pdl);
    else if (cluster == 1 && mode == 6) launch_gate_up<1, 6, false>(x, w, out, pdl);
    else if (cluster == 1 && mode == 7) launch_gate_up<1, 7, false>(x, w, out, pdl);
    else if (cluster == 1 && mode == 8) launch_gate_up<1, 8, false>(x, w, out, pdl);
    else if (cluster == 1 && mode == 9) launch_gate_up<1, 9, false>(x, w, out, pdl);
    else if (cluster == 1 && mode == 10) launch_gate_up<1, 10, false>(x, w, out, pdl);
    else if (cluster == 1 && mode == 11) launch_gate_up<1, 11, false>(x, w, out, pdl);
    else if (cluster == 1) round_silu ? launch_gate_up<1, 2, true>(x, w, out, pdl) : launch_gate_up<1, 2, false>(x, w, out, pdl);
    else if (cluster == 2 && mode == 0) launch_gate_up<2, 0, false>(x, w, out, pdl);
    else if (cluster == 2 && mode == 1) launch_gate_up<2, 1, false>(x, w, out, pdl);
    else if (cluster == 2 && mode == 2) launch_gate_up<2, 2, false>(x, w, out, pdl);
    else if (cluster == 2 && mode == 3) launch_gate_up<2, 3, false>(x, w, out, pdl);
    else if (cluster == 4 && mode == 3) launch_gate_up<4, 3, false>(x, w, out, pdl);
    else TORCH_CHECK(false, "unsupported cluster/mode");
}

}  // namespace dlmrt
