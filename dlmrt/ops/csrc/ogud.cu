// Post-attention megakernel of one TP=1 BF16 decoder layer (found by an automated kernel search): O
// projection -> residual add + RMSNorm -> gate_up + SwiGLU -> down as ONE persistent launch of 132 CTAs x 384 threads
// (WG0, WG1 math; WG2 TMA). Each phase reproduces the arithmetic of the separate kernels it replaces (DeepGEMM's k
// order, Triton's RMSNorm reduction order and SwiGLU), so results are bit-identical to them.
//   * CTAs own one O tile (64x80, WG0), one gate_up tile (128x152, WG0+WG1) and one down tile (64x80, WG0); the four
//     CTAs on SMs 128..131 own none.
//   * O D-tile stores -> cntD[mb] (32 arrivals per 64-row M-block) -> WG0/WG1 normalize rows of that M-block.
//   * norm done -> cntN[mb]; the producer issues gate_up's loads once cntN[2mg] and cntN[2mg+1] reach 32.
//   * SwiGLU stores -> cntY[mg] (64 arrivals per 128-row block) -> the down loads of that block.
// Every CTA must be resident at once (one per SM), and the tile map comes from the SM id, so placement does not
// depend on launch order. Every spin is bounded (%globaltimer, 20 ms) and traps.
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

#include "host.cuh"

using namespace deep_gemm;

// Virtual tile id of this CTA from its SM id: 8 consecutive ids are one fixed 8-SM set, within the set TPC order
// t, t+8, t+16, t+24 (smid < 64) or t, t+7, t+14, t+21 (64 <= smid < 120); smids 120..131 keep their id.
__device__ __forceinline__ uint32_t ogu_vid() {
    uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    if (s >= 132u) __trap();
    const uint32_t t = s / 2;
    uint32_t set, k;
    if (t < 32) { set = t % 8; k = t / 8; }
    else if (t < 60) { set = 8 + (t - 32) % 7; k = (t - 32) / 7; }
    else return 120 + (s - 120);
    return set * 8 + k * 2 + (s & 1);
}
namespace ogu {
// DeepGEMM's 2D TMA load with an L2 cache hint: weights stream once (evict first), activations that many CTAs
// re-read evict last.
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, cute::TMA::CacheHintSm90 kHint>
CUTLASS_DEVICE void tma_load(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
                             cutlass::bfloat16_t* smem_ptr, uint32_t inner_idx, uint32_t outer_idx) {
    constexpr uint32_t BLOCK_INNER_ATOM = tma::get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, cutlass::bfloat16_t>();
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i)
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr), static_cast<uint64_t>(kHint),
                                     smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM, inner_idx + i * BLOCK_INNER_ATOM,
                                     outer_idx);
}
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER>
CUTLASS_DEVICE void load_ef(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
                            cutlass::bfloat16_t* smem_ptr, uint32_t inner_idx, uint32_t outer_idx) {
    tma_load<BLOCK_INNER, BLOCK_OUTER, 128, cute::TMA::CacheHintSm90::EVICT_FIRST>(desc_ptr, barrier_ptr, smem_ptr,
                                                                                   inner_idx, outer_idx);
}
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER>
CUTLASS_DEVICE void load_el(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
                            cutlass::bfloat16_t* smem_ptr, uint32_t inner_idx, uint32_t outer_idx) {
    tma_load<BLOCK_INNER, BLOCK_OUTER, 128, cute::TMA::CacheHintSm90::EVICT_LAST>(desc_ptr, barrier_ptr, smem_ptr,
                                                                                  inner_idx, outer_idx);
}
CUTLASS_DEVICE void tma_store_2d_el(void const* desc_ptr, void const* smem_ptr, int32_t crd0, int32_t crd1) {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    const uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    const uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%2, %3}], [%1], %4;"
                 :: "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1), "l"(pol) : "memory");
}
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* v) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
    #pragma unroll
    for (int i = 0; i < 4; ++ i) { const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i])); v[2 * i] = f.x, v[2 * i + 1] = f.y; }
}
__device__ __forceinline__ uint32_t pack_rn(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ void store8(__nv_bfloat16* p, const float* v) {
    *reinterpret_cast<uint4*>(p) = make_uint4(pack_rn(v[0], v[1]), pack_rn(v[2], v[3]), pack_rn(v[4], v[5]), pack_rn(v[6], v[7]));
}
__device__ __forceinline__ float sumsq8(const float* v) {
    float s = __fmul_rn(v[1], v[1]);
    s = __fmaf_rn(v[0], v[0], s);
    #pragma unroll
    for (int i = 2; i < 8; ++ i) s = __fmaf_rn(v[i], v[i], s);
    return s;
}
// FlashInfer's FusedAddRMSNorm (sgl_kernel, fast math) arithmetic: flush-to-zero adds / multiplies / FMAs, the
// per-thread sum of squares FMA'd in element order from zero.
__device__ __forceinline__ float add_ftz(float a, float b) { float r; asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float mul_ftz(float a, float b) { float r; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fma_ftz(float a, float b, float c) { float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float sumsq8_fi(const float* v) {
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++ i) s = fma_ftz(v[i], v[i], s);
    return s;
}
CUTLASS_DEVICE float swiglu1(float g, float u) {
    float y;
    asm("{\n .reg .f32 t, e;\n sub.f32 t, 0f00000000, %1;\n mul.f32 t, t, 0f3FB8AA3B;\n ex2.approx.f32 e, t;\n"
        " add.f32 e, e, 0f3F800000;\n div.full.f32 e, 0f3F800000, e;\n mul.f32 e, e, %1;\n mul.f32 %0, e, %2;\n}"
        : "=f"(y) : "f"(g), "f"(u));
    return y;
}
// Same result as swiglu1 while g > -80 (no denormal exponentials or reciprocals).
CUTLASS_DEVICE float swiglu1_fast(float g, float u) {
    float y;
    asm("{\n .reg .f32 t, e;\n mul.f32 t, %1, 0fBFB8AA3B;\n ex2.approx.ftz.f32 e, t;\n"
        " add.f32 e, e, 0f3F800000;\n rcp.approx.ftz.f32 e, e;\n mul.f32 e, e, %1;\n mul.f32 %0, e, %2;\n}"
        : "=f"(y) : "f"(g), "f"(u));
    return y;
}
template <bool kFast>
CUTLASS_DEVICE uint32_t swiglu2x(uint32_t gb, uint32_t ub) {
    const float2 g = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&gb));
    const float2 u = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&ub));
    float y0, y1;
    if constexpr (kFast) { y0 = swiglu1_fast(g.x, u.x); y1 = swiglu1_fast(g.y, u.y); }
    else { y0 = swiglu1(g.x, u.x); y1 = swiglu1(g.y, u.y); }
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(y1), "f"(y0));
    return r;
}
CUTLASS_DEVICE uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
CUTLASS_DEVICE void st_el_v4(void* ptr, uint4 v) {
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t gtime() { uint64_t v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)); return v; }
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
// Bounded spin until *p == target (counters start at zero every launch); trap after 20 ms or on excess arrivals.
__device__ __forceinline__ void spin_ge(const unsigned* p, unsigned target) {
    { const unsigned v0 = ld_acq(p); if (v0 == target) return; if (v0 > target and v0 < target + 1024u) __trap(); }
    const uint64_t t0 = gtime();
    uint32_t it = 0;
    unsigned v;
    while (int((v = ld_acq(p)) - target) < 0) {
        if ((++ it & 255u) == 0 and gtime() - t0 > 20000000ull) __trap();
    }
    if (v > target) __trap();
}
}  // namespace ogu

constexpr uint32_t GU_PFDN = 1024;   // bytes of every down-weight row prefetched into L2 during gate_up
// O-proj config: 64x80x64, 12 stages, swizzle 128/128/32, 1 math WG
constexpr uint32_t O_BM = 64, O_BN = 80, BK = 64, O_ST = 12, O_K = 4096, O_KB = O_K / BK;
constexpr uint32_t O_SMEM_D = 10240;                         // align(64*80*2, 1024)
constexpr uint32_t O_SA = O_BM * BK * 2, O_SB = O_BN * BK * 2; // 8192, 10240
// gate_up config: 128x152x64, 4 stages, B box 160 x2 halves, 2 math WGs
constexpr uint32_t G_BM = 128, G_BN = 152, G_BOXN = 160, G_ST = 4, G_K = 2560, G_KB = G_K / BK, G_N = 9728, G_M = 256;
constexpr uint32_t G_SA = G_BM * BK * 2, G_SBH = G_BOXN * BK * 2, G_SB = 2 * G_SBH;  // 16384, 20480, 40960
constexpr uint32_t GU_PITCH = G_BN * 2, GU_G_OFF = 0, GU_U_OFF = G_BM * G_BN * 2;
constexpr uint32_t O_BAR_OFF = O_SMEM_D + O_ST * (O_SA + O_SB);      // 231424
constexpr uint32_t G_BAR_OFF = O_BAR_OFF + 2 * O_ST * 8;             // 231616
constexpr uint32_t S_FLAG_OFF = G_BAR_OFF + 2 * G_ST * 8;            // 231680
constexpr uint32_t D_EARLY = 12;                                     // down weight stages issued before y is ready
// down phase: 64x80x64 12-stage layout = the O-proj layout, own barriers after the flags
constexpr uint32_t D_KB = 9728 / BK;
constexpr uint32_t D_BAR_OFF = S_FLAG_OFF + 16;                      // 231696
constexpr uint32_t SMEM_TOTAL = D_BAR_OFF + 2 * O_ST * 8;            // 231888
static_assert(SMEM_TOTAL == 231888, "smem");
constexpr uint32_t EPI_THREADS = 384;                                // warp 11 (down producer) joins the epilogue
static_assert(G_ST * (G_SA + G_SB) <= O_BAR_OFF, "gate_up stages overlay the O region");
constexpr uint32_t N_H = 2560;                                       // hidden
constexpr uint32_t CNT_SLOT = 512;                                   // u32 counters per slot
constexpr int GRID = 132, THREADS = 384;

// down phase as a separate function: keeps its live ranges out of the gate_up mainloop (register cap).
// kSgl: cuBLAS's split-K for this shape, two K halves accumulated separately and summed in FP32.
template <bool kSgl>
static __device__ __forceinline__ void ogu_down_phase(const cute::TmaDescriptor* tm_dcd_p) {
    using WGO = typename mma::sm90::BF16MMASelector<O_BN, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();
        // ================= down tile (WG0): 64x80, K 9728, DeepGEMM k order =================
        // recompute everything from laundered values so nothing is kept live across the gate_up phase
        const uint32_t bx = ogu_vid();
        const uint32_t d_m = ((bx & 1) << 1) | ((bx >> 1) & 1), d_n = bx >> 2;
        uint8_t* sbase = smem_buffer;
        auto d_sa0 = reinterpret_cast<cutlass::bfloat16_t*>(sbase + O_SMEM_D);
        auto d_sb0 = reinterpret_cast<cutlass::bfloat16_t*>(sbase + O_SMEM_D + O_ST * O_SA);
        Barrier* dfull = reinterpret_cast<Barrier*>(sbase + D_BAR_OFF);
        Barrier* dempty = dfull + O_ST;
        auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, O_BM, BK, 128>(d_sa0, 0, 0);
        auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, O_BN, BK, 128>(d_sb0, 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
        float accum[WGO::kNumAccum] = {0};
        float first[kSgl ? WGO::kNumAccum : 1];
        uint32_t s = 0, ph = 0, prev_s = 0;
        for (uint32_t k_block_idx = 0; k_block_idx < D_KB; ++ k_block_idx) {
            if constexpr (kSgl) {
                if (k_block_idx == D_KB / 2) {
                    ptx::warpgroup_wait<0>();
                    #pragma unroll
                    for (uint32_t i = 0; i < WGO::kNumAccum; ++ i) {
                        ptx::warpgroup_fence_operand(accum[i]);
                        first[i] = accum[i];
                        accum[i] = 0.0f;
                        ptx::warpgroup_fence_operand(accum[i]);
                    }
                }
            }
            const auto a_base = a_desc_lo + s * (O_SA / 16);
            const auto b_base = b_desc_lo + s * (O_SB / 16);
            dfull[s].wait(ph);
            #pragma unroll
            for (uint32_t i = 0; i < WGO::kNumAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < BK / WGO::K; ++ k) {
                a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, O_BM, BK, 128, nv_bfloat16>(a_base, 0, k * WGO::K, 0);
                b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, O_BN, BK, 128, nv_bfloat16>(b_base, 0, k * WGO::K, 0);
                WGO::wgmma(a_desc, b_desc, accum, 1);
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGO::kNumAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
            ptx::warpgroup_wait<1>();
            if (k_block_idx > 0 and lane_idx == 0) dempty[prev_s].arrive();
            prev_s = s;
            s = s == O_ST - 1 ? 0 : s + 1; ph ^= s == 0;
        }
        ptx::warpgroup_wait<0>();
        #pragma unroll
        for (uint32_t i = 0; i < WGO::kNumAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
        if (lane_idx == 0) dempty[prev_s].arrive();
        if constexpr (kSgl) {
            #pragma unroll
            for (uint32_t i = 0; i < WGO::kNumAccum; ++ i) accum[i] = __fadd_rn(first[i], accum[i]);
        }
        // PDL: the next layer's RMSNorm may launch once every CTA's down mainloop is done
        if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
        {
            constexpr uint32_t kSwz = 32, TMA_D_BLOCK_N = kSwz / 2, WGMMA_M_PER_WARP = WGO::M / 4;
            constexpr uint32_t kNumBankGroupBytes = 16;
            cutlass::arch::NamedBarrier::sync(128, 3);
            #pragma unroll
            for (uint32_t i = 0; i < WGO::kNumAccum / 4; ++ i) {
                auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                auto bank_group_index = in_atom_offset + lane_idx * (kSwz / kNumBankGroupBytes);
                constexpr bool kHasShortcut = (kSwz / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                col ^= row % (kSwz / 16);
                uint8_t* smem_ptr = sbase + warp_idx * (WGMMA_M_PER_WARP * kSwz) +
                    atom_offset * O_BM * kSwz + row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({accum[i * 4 + 0], accum[i * 4 + 1]}),
                    __float22bfloat162_rn({accum[i * 4 + 2], accum[i * 4 + 3]}), smem_ptr);
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(128, 3);
            if (threadIdx.x < O_BN / TMA_D_BLOCK_N) {
                auto smem_ptr = reinterpret_cast<cutlass::bfloat16_t*>(sbase) + threadIdx.x * TMA_D_BLOCK_N * O_BM;
                ogu::tma_store_2d_el(tm_dcd_p, smem_ptr, d_n * O_BN + threadIdx.x * TMA_D_BLOCK_N, d_m * O_BM);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
}

// kSgl: SGLang's arithmetic (sgl_kernel's fused_add_rmsnorm, cuBLAS's split-K down) instead of dlmrt's Triton norm
// and K-sequential down; O, gate_up and SwiGLU are the same in both.
template <bool kSgl>
__global__ __launch_bounds__(384, 1) void ogu_kernel(
        unsigned* cnt,                     // this launch's counters: cntD[mb] at mb*32, cntN[mb] at 128 + mb*32, cntY[mg] at 256 + mg*32
        unsigned* cnt_next,                // the next launch's slot, zeroed here (the previous launch used it)
        __nv_bfloat16* ON_H,               // residual h (in/out)
        const __nv_bfloat16* ON_D,         // O-proj output (the norm's delta), discarded after use
        const __nv_bfloat16* ON_W,         // norm weight
        __nv_bfloat16* ON_Y,               // x2 (gate_up A)
        const void* ON_U1,                 // attention output (O-proj A), discarded after use
        float on_eps, float on_scale,
        cutlass::bfloat16_t* gu_out,       // SwiGLU output y
        const __grid_constant__ cute::TmaDescriptor tm_oa,
        const __grid_constant__ cute::TmaDescriptor tm_ob,
        const __grid_constant__ cute::TmaDescriptor tm_ocd,
        const __grid_constant__ cute::TmaDescriptor tm_ga,
        const __grid_constant__ cute::TmaDescriptor tm_gb,
        const char* pf_dn,                 // down weight (L2 prefetch)
        const __grid_constant__ cute::TmaDescriptor tm_da,  // y (down A)
        const __grid_constant__ cute::TmaDescriptor tm_db,  // down weight
        const __grid_constant__ cute::TmaDescriptor tm_dcd  // down output
        ) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
    using WGO = typename mma::sm90::BF16MMASelector<O_BN, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
    using WGG = typename mma::sm90::BF16MMASelector<G_BN, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t b = ogu_vid();
    const bool has_tile = b < 128;

    if (warp_idx == 8 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tm_oa); cute::prefetch_tma_descriptor(&tm_ob); cute::prefetch_tma_descriptor(&tm_ocd);
        cute::prefetch_tma_descriptor(&tm_ga); cute::prefetch_tma_descriptor(&tm_gb);
        cute::prefetch_tma_descriptor(&tm_da); cute::prefetch_tma_descriptor(&tm_db); cute::prefetch_tma_descriptor(&tm_dcd);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_d = reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer);
    auto o_sa = [&](uint32_t i) { return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + O_SMEM_D + i * O_SA); };
    auto o_sb = [&](uint32_t i) { return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + O_SMEM_D + O_ST * O_SA + i * O_SB); };
    auto g_sa = [&](uint32_t i) { return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + i * G_SA); };
    auto g_sb = [&](uint32_t i) { return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + G_ST * G_SA + i * G_SB); };
    Barrier* o_full = reinterpret_cast<Barrier*>(smem_buffer + O_BAR_OFF);
    Barrier* o_empty = o_full + O_ST;
    Barrier* g_full = reinterpret_cast<Barrier*>(smem_buffer + G_BAR_OFF);
    Barrier* g_empty = g_full + G_ST;
    volatile uint32_t* s_normed = reinterpret_cast<volatile uint32_t*>(smem_buffer + S_FLAG_OFF);
    Barrier* gu_done = reinterpret_cast<Barrier*>(smem_buffer + S_FLAG_OFF + 8);   // gate_up stages all consumed
    Barrier* d_full = reinterpret_cast<Barrier*>(smem_buffer + D_BAR_OFF);
    Barrier* d_empty = d_full + O_ST;

    if (warp_idx == 9 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < O_ST; ++ i) { o_full[i].init(1); o_empty[i].init(4); }
        #pragma unroll
        for (uint32_t i = 0; i < G_ST; ++ i) { g_full[i].init(1); g_empty[i].init(8); }
        cutlass::arch::fence_barrier_init();
        *s_normed = 0u;
        gu_done->init(1);
        #pragma unroll
        for (uint32_t i = 0; i < O_ST; ++ i) { d_full[i].init(1); d_empty[i].init(4); }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    cudaGridDependencySynchronize();
    // The previous launch (which used cnt_next) is complete once the upstream grids are, so the tile-less last CTA
    // re-arms its 10 counters for the next launch.
    if (b == gridDim.x - 1 and threadIdx.x < 10)
        atomicExch(cnt_next + (threadIdx.x < 4 ? threadIdx.x * 32 : threadIdx.x < 8 ? 128 + (threadIdx.x - 4) * 32 : 256 + (threadIdx.x - 8) * 32), 0u);

    // tile coordinates
    const uint32_t o_m = ((b & 1) << 1) | ((b >> 1) & 1), o_n = b >> 2;
    const uint32_t g_m = b % (G_M / G_BM), g_n = b / (G_M / G_BM);

    auto gu_epilogue = [&]() {
    if (has_tile and threadIdx.x < EPI_THREADS) {
        const uint32_t m_idx = g_m * G_BM, n_idx = g_n * G_BN;
        cutlass::arch::NamedBarrier::sync(EPI_THREADS, 2);
        constexpr uint32_t kChunksPerRow = GU_PITCH / 16, kChunks = G_BM * kChunksPerRow;
        #pragma unroll 1
        for (uint32_t c = threadIdx.x; c < kChunks; c += EPI_THREADS) {
            const uint32_t r = c / kChunksPerRow, q = c % kChunksPerRow;
            const uint4 g4 = *reinterpret_cast<const uint4*>(smem_buffer + GU_G_OFF + c * 16);
            const uint4 u4 = *reinterpret_cast<const uint4*>(smem_buffer + GU_U_OFF + c * 16);
            const uint32_t gw[4] = {g4.x, g4.y, g4.z, g4.w}, uw[4] = {u4.x, u4.y, u4.z, u4.w};
            float gmin = 0.f;
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++ k) {
                const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[k]));
                gmin = fminf(gmin, fminf(gf.x, gf.y));
            }
            uint32_t yw[4];
            if (gmin > -80.f) {
                #pragma unroll
                for (uint32_t k = 0; k < 4; ++ k) yw[k] = ogu::swiglu2x<true>(gw[k], uw[k]);
            } else {
                #pragma unroll
                for (uint32_t k = 0; k < 4; ++ k) yw[k] = ogu::swiglu2x<false>(gw[k], uw[k]);
            }
            ogu::st_el_v4(gu_out + (m_idx + r) * G_N + n_idx + q * 8, make_uint4(yw[0], yw[1], yw[2], yw[3]));
        }
        // y rows -> visible to the down TMA loads of the other CTAs of this 128-row block
        asm volatile("fence.proxy.async.global;" ::: "memory");
        cutlass::arch::NamedBarrier::sync(EPI_THREADS, 2);
        if (threadIdx.x == 0) {
            __threadfence();
            asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(cnt + 256 + g_m * 32) : "memory");
        }
    }
    };
    if (warp_idx >= 8) {
        cutlass::arch::warpgroup_reg_dealloc<48>();
        if (warp_idx == 11) {
            // ---- down loads: wait until gate_up consumed every stage, weight stages early, join the SwiGLU
            // epilogue, then y once the 64 gate_up tiles of this row block are stored ----
            if (has_tile) {
                if (cute::elect_one_sync()) {
                    gu_done->wait(0);
                    #pragma unroll
                    for (uint32_t k = 0; k < D_EARLY; ++ k) {
                        ogu::load_ef<BK, O_BN>(&tm_db, &d_full[k], o_sb(k), k * BK, o_n * O_BN);
                        d_full[k].arrive_and_expect_tx(O_SA + O_SB);
                    }
                }
                __syncwarp();
                gu_epilogue();
            }
            if (cute::elect_one_sync() and has_tile) {
                ogu::spin_ge(cnt + 256 + (o_m >> 1) * 32, 64u);   // y rows of this 64-row block: 64 gate_up tiles
                asm volatile("fence.proxy.async;" ::: "memory");
                #pragma unroll
                for (uint32_t k = 0; k < D_EARLY; ++ k)
                    ogu::load_el<BK, O_BM>(&tm_da, &d_full[k], o_sa(k), k * BK, o_m * O_BM);
                uint32_t s = D_EARLY % O_ST, ph = (D_EARLY / O_ST) & 1;
                for (uint32_t k = D_EARLY; k < D_KB; ++ k) {
                    d_empty[s].wait(ph ^ 1);
                    ogu::load_el<BK, O_BM>(&tm_da, &d_full[s], o_sa(s), k * BK, o_m * O_BM);
                    ogu::load_ef<BK, O_BN>(&tm_db, &d_full[s], o_sb(s), k * BK, o_n * O_BN);
                    d_full[s].arrive_and_expect_tx(O_SA + O_SB);
                    s = s == O_ST - 1 ? 0 : s + 1; ph ^= s == 0;
                }
            }
        } else {
        if (warp_idx == 10 and cute::elect_one_sync() and has_tile) {
            // ---- O-proj loads ----
            uint32_t s = 0, ph = 0;
            for (uint32_t k = 0; k < O_KB; ++ k) {
                o_empty[s].wait(ph ^ 1);
                ogu::load_el<BK, O_BM>(&tm_oa, &o_full[s], o_sa(s), k * BK, o_m * O_BM);
                ogu::load_ef<BK, O_BN>(&tm_ob, &o_full[s], o_sb(s), k * BK, o_n * O_BN);
                o_full[s].arrive_and_expect_tx(O_SA + O_SB);
                s = s == O_ST - 1 ? 0 : s + 1; ph ^= s == 0;
            }
            // drain: every O stage consumed by WG0's WGMMAs before the gate_up stages overlay them
            for (uint32_t i = 0; i < O_ST; ++ i) {
                o_empty[s].wait(ph ^ 1);
                s = s == O_ST - 1 ? 0 : s + 1; ph ^= s == 0;
            }
            // ---- gate_up loads: poll the norm-ready counters first (a poll issued behind a TMA burst returns only
            // after the burst's data), then stage-interleaved A+B ----
            const uint32_t n_idx = g_n * G_BN, m_idx = g_m * G_BM;
            auto issue_b = [&](uint32_t k) {
                ogu::load_ef<BK, G_BOXN>(&tm_gb, &g_full[k], g_sb(k), k * BK, n_idx);
                ogu::load_ef<BK, G_BOXN>(&tm_gb, &g_full[k], g_sb(k) + G_BOXN * BK, k * BK, n_idx + G_N);
            };
            auto issue_a = [&](uint32_t k) {
                ogu::load_el<BK, G_BM>(&tm_ga, &g_full[k], g_sa(k), k * BK, m_idx);
            };
            ogu::spin_ge(cnt + 128 + (2 * g_m) * 32, 32u);
            ogu::spin_ge(cnt + 128 + (2 * g_m + 1) * 32, 32u);
            asm volatile("fence.proxy.async;" ::: "memory");
            #pragma unroll
            for (uint32_t k = 0; k < G_ST; ++ k) { issue_a(k); issue_b(k); g_full[k].arrive_and_expect_tx(G_SA + G_SB); }
            uint32_t gs = 0, gph = 1;
            for (uint32_t k = G_ST; k < G_KB; ++ k) {
                g_empty[gs].wait(gph ^ 1);
                ogu::load_el<BK, G_BM>(&tm_ga, &g_full[gs], g_sa(gs), k * BK, m_idx);
                ogu::load_ef<BK, G_BOXN>(&tm_gb, &g_full[gs], g_sb(gs), k * BK, n_idx);
                ogu::load_ef<BK, G_BOXN>(&tm_gb, &g_full[gs], g_sb(gs) + G_BOXN * BK, k * BK, n_idx + G_N);
                g_full[gs].arrive_and_expect_tx(G_SA + G_SB);
                gs = gs == G_ST - 1 ? 0 : gs + 1; gph ^= gs == 0;
            }
        }
        __syncwarp();
        constexpr int kPfThreads = 96;
        cutlass::arch::NamedBarrier::sync(kPfThreads, 9);
        if (pf_dn != nullptr) {
            constexpr int LPR = GU_PFDN / 128, NL = 2560 * LPR;
            const int nthr = int(gridDim.x) * kPfThreads;
            for (int l = int(b) * kPfThreads + int(threadIdx.x - 256); l < NL; l += nthr)
                asm volatile("prefetch.global.L2::evict_normal [%0];" :: "l"(pf_dn + size_t(l / LPR) * 19456 + (l % LPR) * 128));
        }
        gu_epilogue();
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<224>();
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);

        if (has_tile) {
            // ---- norm row of this WG (WG0 -> r0, WG1 -> r0+1), preloaded before the O phase ----
            const uint32_t t = threadIdx.x % 128, w = t / 32, l = t % 32;
            const uint32_t eA1 = 256 * w + 8 * l, eA2 = 2048 + 256 * w + 8 * l, eB = 256 * (w + 4) + 8 * l;
            const bool two = w < 2;
            const uint32_t row = o_m * O_BM + o_n * 2 + math_wg_idx;
            float wA1[8], wA2[8], wB[8], hA1[8], hA2[8], hB[8];
            ogu::load8(ON_W + eA1, wA1); ogu::load8(ON_W + eB, wB);
            if (two) ogu::load8(ON_W + eA2, wA2);
            ogu::load8(ON_H + row * N_H + eA1, hA1); ogu::load8(ON_H + row * N_H + eB, hB);
            ogu::load8(ON_H + row * N_H + (two ? eA2 : eA1), hA2);

            if (math_wg_idx == 0) {
            // ================= O-proj tile (WG0) =================
            auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, O_BM, BK, 128>(o_sa(0), 0, 0);
            auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, O_BN, BK, 128>(o_sb(0), 0, 0);
            const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
            const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
            float accum[WGO::kNumAccum] = {0};
            uint32_t s = 0, ph = 0, prev_s = 0;
            for (uint32_t k_block_idx = 0; k_block_idx < O_KB; ++ k_block_idx) {
                const auto a_base = a_desc_lo + s * (O_SA / 16);
                const auto b_base = b_desc_lo + s * (O_SB / 16);
                o_full[s].wait(ph);
                #pragma unroll
                for (uint32_t i = 0; i < WGO::kNumAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BK / WGO::K; ++ k) {
                    a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, O_BM, BK, 128, nv_bfloat16>(a_base, 0, k * WGO::K, 0);
                    b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, O_BN, BK, 128, nv_bfloat16>(b_base, 0, k * WGO::K, 0);
                    WGO::wgmma(a_desc, b_desc, accum, 1);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGO::kNumAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<1>();
                if (k_block_idx > 0 and lane_idx == 0) o_empty[prev_s].arrive();
                prev_s = s;
                s = s == O_ST - 1 ? 0 : s + 1; ph ^= s == 0;
            }
            ptx::warpgroup_wait<0>();
            #pragma unroll
            for (uint32_t i = 0; i < WGO::kNumAccum; ++ i) ptx::warpgroup_fence_operand(accum[i]);
            if (lane_idx == 0) o_empty[prev_s].arrive();

            // ---- O epilogue: STSM (swizzle 32) + TMA store, L2 evict_last ----
            {
                constexpr uint32_t kSwz = 32, TMA_D_BLOCK_N = kSwz / 2, WGMMA_M_PER_WARP = WGO::M / 4;
                constexpr uint32_t kNumBankGroupBytes = 16;
                cutlass::arch::NamedBarrier::sync(128, 3);
                #pragma unroll
                for (uint32_t i = 0; i < WGO::kNumAccum / 4; ++ i) {
                    auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                    auto bank_group_index = in_atom_offset + lane_idx * (kSwz / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwz / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                    col ^= row % (kSwz / 16);
                    uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem_d) + warp_idx * (WGMMA_M_PER_WARP * kSwz) +
                        atom_offset * O_BM * kSwz + row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({accum[i * 4 + 0], accum[i * 4 + 1]}),
                        __float22bfloat162_rn({accum[i * 4 + 2], accum[i * 4 + 3]}), smem_ptr);
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(128, 3);
                if (threadIdx.x < O_BN / TMA_D_BLOCK_N) {
                    auto smem_ptr = smem_d + threadIdx.x * TMA_D_BLOCK_N * O_BM;
                    ogu::tma_store_2d_el(&tm_ocd, smem_ptr, o_n * O_BN + threadIdx.x * TMA_D_BLOCK_N, o_m * O_BM);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }

                if (t < O_BN / 16) {
                    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                }
                cutlass::arch::NamedBarrier::sync(128, 4);
                unsigned* cD = cnt + o_m * 32;
                if (t == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(cD) : "memory");
                if (t == 0) ogu::spin_ge(cD, 32u);
            }
            cutlass::arch::NamedBarrier::sync(256, 5);   // WG1 joins once the M-block's D tiles are stored
            // ---- fused post-attention RMSNorm of `row` (Triton's reduction order, 256 virtual threads per row) ----
            {
                {
                    const __nv_bfloat16* dr = ON_D + row * N_H;
                    float dA1[8], dB[8], dA2[8];
                    ogu::load8(dr + eA1, dA1); ogu::load8(dr + eB, dB); ogu::load8(dr + (two ? eA2 : eA1), dA2);
                    if constexpr (kSgl) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++ i) { hA1[i] = ogu::add_ftz(hA1[i], dA1[i]); hB[i] = ogu::add_ftz(hB[i], dB[i]); }
                        if (two) {
                            #pragma unroll
                            for (int i = 0; i < 8; ++ i) hA2[i] = ogu::add_ftz(hA2[i], dA2[i]);
                        }
                    } else {
                        #pragma unroll
                        for (int i = 0; i < 8; ++ i) { hA1[i] = __fadd_rn(hA1[i], dA1[i]); hB[i] = __fadd_rn(hB[i], dB[i]); }
                        if (two) {
                            #pragma unroll
                            for (int i = 0; i < 8; ++ i) hA2[i] = __fadd_rn(hA2[i], dA2[i]);
                        }
                    }
                }
                ogu::store8(ON_H + row * N_H + eA1, hA1); ogu::store8(ON_H + row * N_H + eB, hB);
                if (two) ogu::store8(ON_H + row * N_H + eA2, hA2);
                __syncwarp();
                if ((l & 7) == 0) {
                    const __nv_bfloat16* dr = ON_D + row * N_H;
                    asm volatile("discard.global.L2 [%0], 128;" :: "l"(dr + eA1) : "memory");
                    asm volatile("discard.global.L2 [%0], 128;" :: "l"(dr + eB) : "memory");
                    if (two) asm volatile("discard.global.L2 [%0], 128;" :: "l"(dr + eA2) : "memory");
                }
                if (ON_U1 != nullptr and t < 64)
                    asm volatile("discard.global.L2 [%0], 128;" :: "l"(reinterpret_cast<const char*>(ON_U1) + size_t(row) * 8192 + t * 128) : "memory");
                float* tsum = reinterpret_cast<float*>(smem_d) + math_wg_idx * 16;
                if constexpr (kSgl) {
                    // FlashInfer's 320-thread row (thread t: elements 8t..8t+7): this thread holds threads (w, l),
                    // (w + 4, l) and, for w < 2, (w + 8, l); each warp's xor butterfly, then the warp sums below.
                    float sA1 = ogu::sumsq8_fi(hA1), sB1 = ogu::sumsq8_fi(hB), sA2 = two ? ogu::sumsq8_fi(hA2) : 0.0f;
                    #pragma unroll
                    for (int o = 16; o >= 1; o >>= 1) {
                        sA1 = ogu::add_ftz(sA1, __shfl_xor_sync(0xffffffff, sA1, o));
                        sB1 = ogu::add_ftz(sB1, __shfl_xor_sync(0xffffffff, sB1, o));
                        sA2 = ogu::add_ftz(sA2, __shfl_xor_sync(0xffffffff, sA2, o));
                    }
                    if (l == 0) { tsum[w] = sA1; tsum[w + 4] = sB1; if (two) tsum[w + 8] = sA2; }
                } else {
                    float sA1 = ogu::sumsq8(hA1);
                    float sA2 = two ? ogu::sumsq8(hA2) : 0.0f;
                    float sB1 = ogu::sumsq8(hB);
                    #pragma unroll
                    for (int o = 8; o >= 1; o >>= 1) { sA1 = __fadd_rn(sA1, __shfl_xor_sync(0xffffffff, sA1, o)); sB1 = __fadd_rn(sB1, __shfl_xor_sync(0xffffffff, sB1, o)); }
                    #pragma unroll
                    for (int o = 8; o >= 1; o >>= 1) sA2 = __fadd_rn(sA2, __shfl_xor_sync(0xffffffff, sA2, o));
                    float cA = two ? __fadd_rn(sA1, sA2) : sA1, cB = sB1;
                    cA = __fadd_rn(cA, __shfl_xor_sync(0xffffffff, cA, 16));
                    cB = __fadd_rn(cB, __shfl_xor_sync(0xffffffff, cB, 16));
                    if (l == 0) { tsum[w] = cA; tsum[w + 4] = cB; }
                }
                cutlass::arch::NamedBarrier::sync(128, 6 + math_wg_idx);
                if constexpr (kSgl) {
                    // Warp 0's xor butterfly over the 10 warp sums (lanes 10..31 zero), as fixed pairings.
                    const float* tt = tsum;
                    using ogu::add_ftz;
                    const float total = add_ftz(add_ftz(add_ftz(add_ftz(tt[0], tt[8]), tt[4]), add_ftz(tt[2], tt[6])),
                                                add_ftz(add_ftz(add_ftz(tt[1], tt[9]), tt[5]), add_ftz(tt[3], tt[7])));
                    float rcp, f;
                    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp) : "f"(static_cast<float>(N_H)));
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(f) : "f"(ogu::fma_ftz(total, rcp, on_eps)));
                    float y[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++ i) y[i] = ogu::mul_ftz(add_ftz(0.0f, wA1[i]), ogu::mul_ftz(hA1[i], f));
                    ogu::store8(ON_Y + row * N_H + eA1, y);
                    #pragma unroll
                    for (int i = 0; i < 8; ++ i) y[i] = ogu::mul_ftz(add_ftz(0.0f, wB[i]), ogu::mul_ftz(hB[i], f));
                    ogu::store8(ON_Y + row * N_H + eB, y);
                    if (two) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++ i) y[i] = ogu::mul_ftz(add_ftz(0.0f, wA2[i]), ogu::mul_ftz(hA2[i], f));
                        ogu::store8(ON_Y + row * N_H + eA2, y);
                    }
                } else {
                    const float* tt = tsum;
                    const float total = __fadd_rn(__fadd_rn(__fadd_rn(tt[0], tt[4]), __fadd_rn(tt[2], tt[6])),
                                                  __fadd_rn(__fadd_rn(tt[1], tt[5]), __fadd_rn(tt[3], tt[7])));
                    float f;
                    const float q = __fmaf_rn(total, __int_as_float(0x39cccccd), on_eps);
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(f) : "f"(q));
                    f = __fmul_rn(f, on_scale);
                    float y[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++ i) y[i] = __fmul_rn(__fmul_rn(hA1[i], f), wA1[i]);
                    ogu::store8(ON_Y + row * N_H + eA1, y);
                    #pragma unroll
                    for (int i = 0; i < 8; ++ i) y[i] = __fmul_rn(__fmul_rn(hB[i], f), wB[i]);
                    ogu::store8(ON_Y + row * N_H + eB, y);
                    if (two) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++ i) y[i] = __fmul_rn(__fmul_rn(hA2[i], f), wA2[i]);
                        ogu::store8(ON_Y + row * N_H + eA2, y);
                    }
                }
                // x2 rows (and h rows) -> visible to the async proxy (gate_up TMA) of the other CTAs
                asm volatile("fence.proxy.async.global;" ::: "memory");
                cutlass::arch::NamedBarrier::sync(256, 5);
                if (threadIdx.x == 0) {
                    __threadfence();
                    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(cnt + 128 + o_m * 32) : "memory");
                    *s_normed = 1u;
                }
            }
        }

        if (has_tile) {
            // ================= gate_up + SwiGLU tile (WG0 + WG1) =================
            auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, G_BM, BK, 128>(g_sa(0), math_wg_idx * WGG::M, 0);
            auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, G_BN, BK, 128>(g_sb(0), 0, 0);
            const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
            const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);
            float accum[WGG::kNumAccum] = {0};
            float accum_up[WGG::kNumAccum] = {0};
            uint32_t s = 0, ph = 0;
            for (uint32_t k_block_idx = 0; k_block_idx < G_KB; ++ k_block_idx) {
                const auto a_base = a_desc_lo + s * (G_SA / 16);
                const auto b_base = b_desc_lo + s * (G_SB / 16);
                g_full[s].wait(ph);
                #pragma unroll
                for (uint32_t i = 0; i < WGG::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]), ptx::warpgroup_fence_operand(accum_up[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < BK / WGG::K; ++ k) {
                    a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, G_BM, BK, 128, nv_bfloat16>(a_base, 0, k * WGG::K, 0);
                    b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, G_BN, BK, 128, nv_bfloat16>(b_base, 0, k * WGG::K, 0);
                    WGG::wgmma(a_desc, b_desc, accum, 1);
                    b_desc.reg32_[0] += G_SBH / 16;
                    WGG::wgmma(a_desc, b_desc, accum_up, 1);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGG::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]), ptx::warpgroup_fence_operand(accum_up[i]);
                ptx::warpgroup_wait<0>();
                if (lane_idx == 0) g_empty[s].arrive();
                s = s == G_ST - 1 ? 0 : s + 1; ph ^= s == 0;
            }
            cutlass::arch::NamedBarrier::sync(256, 1);   // both math WGs done reading every stage
            if (threadIdx.x == 0) gu_done->arrive();
            {
                const uint32_t lrow = math_wg_idx * WGG::M + (warp_idx % 4) * 16 + lane_idx / 4;
                #pragma unroll
                for (uint32_t jj = 0; jj < WGG::kNumAccum / 4; ++ jj) {
                    const uint32_t lcol = jj * 8 + (lane_idx % 4) * 2;
                    *reinterpret_cast<uint32_t*>(smem_buffer + GU_G_OFF + lrow * GU_PITCH + lcol * 2) = ogu::pack_bf16x2(accum[jj * 4 + 0], accum[jj * 4 + 1]);
                    *reinterpret_cast<uint32_t*>(smem_buffer + GU_G_OFF + (lrow + 8) * GU_PITCH + lcol * 2) = ogu::pack_bf16x2(accum[jj * 4 + 2], accum[jj * 4 + 3]);
                    *reinterpret_cast<uint32_t*>(smem_buffer + GU_U_OFF + lrow * GU_PITCH + lcol * 2) = ogu::pack_bf16x2(accum_up[jj * 4 + 0], accum_up[jj * 4 + 1]);
                    *reinterpret_cast<uint32_t*>(smem_buffer + GU_U_OFF + (lrow + 8) * GU_PITCH + lcol * 2) = ogu::pack_bf16x2(accum_up[jj * 4 + 2], accum_up[jj * 4 + 3]);
                }
            }
        }
        gu_epilogue();
    if (has_tile and math_wg_idx == 0) ogu_down_phase<kSgl>(&tm_dcd);
    }
#endif
}

namespace dlmrt {

template <bool kSgl>
static void launch_post_attention(const at::Tensor& a, at::Tensor& h, const at::Tensor& w_o, const at::Tensor& norm,
                                  double eps, double scale, const at::Tensor& w_gate_up, const at::Tensor& w_down,
                                  at::Tensor& o, at::Tensor& x, at::Tensor& y, at::Tensor& out, at::Tensor& counters,
                                  int64_t slot, bool pdl) {
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->multiProcessorCount == GRID, "needs a 132-SM H100");
    auto shaped = [](const at::Tensor& t, int64_t rows, int64_t cols) {
        return t.dim() == 2 && t.size(0) == rows && t.size(1) == cols;
    };
    TORCH_CHECK(shaped(a, G_M, O_K) && shaped(h, G_M, N_H) && shaped(w_o, N_H, O_K) && norm.numel() == N_H &&
                shaped(w_gate_up, 2 * G_N, G_K) && shaped(w_down, N_H, G_N) && shaped(o, G_M, N_H) &&
                shaped(x, G_M, N_H) && shaped(y, G_M, G_N) && shaped(out, G_M, N_H), "shapes");
    const int64_t slots = counters.dim() == 2 ? counters.size(0) : 0;
    TORCH_CHECK(counters.scalar_type() == at::kInt && counters.is_contiguous() && slots >= 2 &&
                counters.size(1) == CNT_SLOT && slot >= 0 && slot < slots, "counters");
    TORCH_CHECK(h.is_contiguous() && norm.is_contiguous() && norm.scalar_type() == at::kBFloat16);
    auto bf = [](const at::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
    auto* cnt = reinterpret_cast<unsigned*>(counters.data_ptr<int>());
    const auto sw128 = CU_TENSOR_MAP_SWIZZLE_128B, sw32 = CU_TENSOR_MAP_SWIZZLE_32B;
    launch(ogu_kernel<kSgl>, GRID, THREADS, SMEM_TOTAL, 1, pdl, cnt + slot * CNT_SLOT, cnt + (slot + 1) % slots * CNT_SLOT,
           bf(h), bf(o), bf(norm), bf(x), a.data_ptr(), static_cast<float>(eps), static_cast<float>(scale),
           reinterpret_cast<cutlass::bfloat16_t*>(y.data_ptr()),
           encode_boxes(a, G_M, O_K, O_BM, BK, sw128), encode_boxes(w_o, N_H, O_K, O_BN, BK, sw128),
           encode_boxes(o, G_M, N_H, O_BM, 16, sw32), encode_boxes(x, G_M, N_H, G_BM, BK, sw128),
           encode_boxes(w_gate_up, 2 * G_N, G_K, G_BOXN, BK, sw128),
           reinterpret_cast<const char*>(w_down.data_ptr()),
           encode_boxes(y, G_M, G_N, O_BM, BK, sw128), encode_boxes(w_down, N_H, G_N, O_BN, BK, sw128),
           encode_boxes(out, G_M, N_H, O_BM, 16, sw32));
}

// out = down(SwiGLU(rmsnorm(h += a @ w_o^T) @ w_gate_up^T)) for one 256-row block (Model.forward's post-attention
// chain at TP=1). `o`, `x` and `y` are scratch (O output, normalized rows, SwiGLU output); the kernel discards `a`
// and `o` from L2 after reading them. `counters` is a zero-initialized [slots, 512] int32 tensor: consecutive
// launches must use consecutive slots (mod slots), since each launch re-arms the slot after its own.
void post_attention(const at::Tensor& a, at::Tensor& h, const at::Tensor& w_o, const at::Tensor& norm, double eps,
                    double scale, const at::Tensor& w_gate_up, const at::Tensor& w_down, at::Tensor& o, at::Tensor& x,
                    at::Tensor& y, at::Tensor& out, at::Tensor& counters, int64_t slot, bool pdl) {
    launch_post_attention<false>(a, h, w_o, norm, eps, scale, w_gate_up, w_down, o, x, y, out, counters, slot, pdl);
}

// The same chain bitwise as SGLang computes it (cuBLAS O / gate_up / down, sgl_kernel's fused_add_rmsnorm and
// SiluAndMul); `scale` must be 1.
void post_attention_sglang(const at::Tensor& a, at::Tensor& h, const at::Tensor& w_o, const at::Tensor& norm,
                           double eps, const at::Tensor& w_gate_up, const at::Tensor& w_down, at::Tensor& o,
                           at::Tensor& x, at::Tensor& y, at::Tensor& out, at::Tensor& counters, int64_t slot, bool pdl) {
    launch_post_attention<true>(a, h, w_o, norm, eps, 1.0, w_gate_up, w_down, o, x, y, out, counters, slot, pdl);
}

}  // namespace dlmrt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("post_attention", &dlmrt::post_attention);
    m.def("post_attention_sglang", &dlmrt::post_attention_sglang);
}
