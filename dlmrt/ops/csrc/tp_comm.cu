// Tensor-parallel reduction fused with the residual add and RMSNorm, without barriers.
//
// Every rank holds a partial product P [rows, n] (local memory). Rank r owns rows
// [r * R, (r + 1) * R) with R = rows / world. Pusher CTAs and one owner CTA per owned row index c:
//   1. push: row s * R + c of P goes to rank s's receive slot [r][c] (NVLink stores, pusher CTAs),
//   2. the owner waits until all `world` slots of its row have arrived, sums them in rank order
//      (deterministic, FP32), adds the residual (own rows only: the residual stream is
//      sequence-parallel) and applies RMSNorm,
//   3. multimem.st: the normalized row lands in OUT on every rank (NVSwitch multicast; without multicast,
//      one store per rank),
//   4. every rank waits until all rows of its OUT copy have arrived.
// Arrival is detected in-band (Lamport): receive slots and OUT start filled with a sentinel that no
// computed bf16 value takes (0xFFFF, a negative NaN; arithmetic produces the canonical 0x7FFF), so a
// 16-byte vector has arrived once no lane holds it. No barrier or system fence is on the critical path.
//
// Re-arming, and why it cannot race with remote writes:
//  * receive slots are triple-buffered by invocation e. Invocation e re-arms buffer (e + 2) % 3, which a
//    peer writes next in invocation e + 2; the peer gets there only after receiving this rank's pushes
//    of invocation e + 1, which start after this kernel has completed.
//  * OUT buffers are per call site (the next kernel reads them at a fixed address). Each invocation
//    re-arms the OUT of the previous invocation, whose consumer has completed; peers write that buffer
//    again only after receiving this rank's pushes for it, one full round later.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kVec = 8;  // bf16 per 16-byte access
constexpr uint32_t kSentinel = 0xFFFFFFFFu;

__device__ __forceinline__ bool arrived(const uint4& v) {
    auto ok = [](uint32_t w) { return (w & 0xFFFFu) != 0xFFFFu && (w >> 16) != 0xFFFFu; };
    return ok(v.x) && ok(v.y) && ok(v.z) && ok(v.w);
}

__device__ __forceinline__ uint4 load_volatile(const void* p) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

constexpr int kMaxWorld = 8;
constexpr int kMaxOwners = 32;

// Waits until the 16-byte vectors at base + s * stride (s < world) have all arrived; returns them in v.
// Each round issues every pending load before testing any: testing inside the load loop would make each
// load wait on the previous test through `pending`, one L2 round trip per slot instead of one per round.
__device__ __forceinline__ void poll_all(const __nv_bfloat16* base, size_t stride, int world, uint4 (&v)[kMaxWorld]) {
    unsigned pending = (1u << world) - 1;
    long long start = clock64();
    while (true) {
        const unsigned was = pending;
#pragma unroll
        for (int s = 0; s < kMaxWorld; ++s)
            if (was >> s & 1) v[s] = load_volatile(base + s * stride);
#pragma unroll
        for (int s = 0; s < kMaxWorld; ++s)
            if ((was >> s & 1) && arrived(v[s])) pending &= ~(1u << s);
        if (!pending) return;
        if (clock64() - start > (1ll << 36)) {  // ~30 s: a rank is gone; fail loudly instead of hanging
            printf("tp_comm: rank data never arrived (block %d, thread %d)\n", blockIdx.x, threadIdx.x);
            __trap();
        }
    }
}

__device__ __forceinline__ void multicast_store(void* mc, const uint4& v) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 ::"l"(mc), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ void add(float (&acc)[kVec], const uint4& v) {
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(b[i]);
        acc[2 * i] += f.x;
        acc[2 * i + 1] += f.y;
    }
}

__device__ __forceinline__ uint4 pack(const float (&v)[kVec]) {
    uint4 out;
    __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    return out;
}

__device__ __forceinline__ float block_sum(float x, float* scratch) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    x = lane < (blockDim.x >> 5) ? scratch[lane] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
    __syncthreads();  // scratch is reused by the next row
    return x;
}

struct Args {
    const __nv_bfloat16* partial;  // [rows, n] local
    const int64_t* recv;           // [world] base of each rank's receive buffer [3][world][R][n]
    __nv_bfloat16* h;              // [rows, n] residual (own rows read and written)
    const __nv_bfloat16* weight;   // [n]
    __nv_bfloat16* out;            // [rows, n] local copy of this call site's output
    __nv_bfloat16* out_mc;         // its multicast address, or null without NVLink multicast
    const int64_t* out_peers;      // [world] every rank's copy of OUT (used without multicast)
    __nv_bfloat16* rearm;          // [rearm_vecs * kVec] previous call site's output (local), or null
    int* epochs;                   // [grid] local invocation counters
    long long* stamps;             // [grid, 4] phase timestamps (ns) for diagnostics, or null
    int rank, world, n, rearm_vecs;
    float eps, scale;
};

__device__ __forceinline__ void stamp(const Args& a, int phase) {
    if (a.stamps && threadIdx.x == 0) {
        long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        a.stamps[blockIdx.x * 4 + phase] = t;
    }
}

// P blocks only push and re-arm, so that enough SMs issue the NVLink stores (register-level remote stores
// need many SMs to fill the links); the other min(R, kMaxOwners) blocks own rows. The whole grid is
// resident, so the roles may take any block indices: owners take the low ones, which are dispatched first
// (dispatching a ~128-block grid takes ~1.3 us), so the reducing side is ready when the pushes land.
template <bool NORM, bool RESIDUAL>
__global__ void reduce_kernel(Args a, int R) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    stamp(a, 0);
    __shared__ float scratch[32];
    const int t = threadIdx.x;
    const size_t row_elems = static_cast<size_t>(a.n);
    const size_t slot_elems = static_cast<size_t>(R) * row_elems;   // one rank's rows in a buffer
    const size_t buf_elems = static_cast<size_t>(a.world) * slot_elems;
    const int vecs_per_row = a.n / kVec;

    const int owners = min(R, kMaxOwners), P = gridDim.x - owners;
    // Role index: [0, P) pushers, [P, P + owners) owners.
    const int bid = static_cast<int>(blockIdx.x) < owners ? P + static_cast<int>(blockIdx.x)
                                                          : static_cast<int>(blockIdx.x) - owners;
    if (bid < P) {
        // 1. Push: row s * R + c of the partial goes to rank s, slot [rank][c]. Consecutive logical rows go to
        // consecutive destinations starting at this rank, so the first stores of all ranks spread over all
        // destinations instead of all landing on rank 0.
        const int total = a.world * R * vecs_per_row;
        const int stride = P * blockDim.x;
        const int first = bid * blockDim.x + t;
        auto source = [&](int i, int& s, size_t& off) {
            const int l = i / vecs_per_row, c = l / a.world;
            s = (a.rank + l % a.world) % a.world;
            const size_t col = static_cast<size_t>(i % vecs_per_row) * kVec;
            off = static_cast<size_t>(c) * row_elems + col;  // within the destination's slot for this rank
            return reinterpret_cast<const uint4*>(a.partial + static_cast<size_t>(s * R + c) * row_elems + col);
        };
        // The first kAhead vectors are loaded before the epoch counter: their stores need the epoch's buffer
        // index, but the loads do not, so both round trips overlap.
        constexpr int kAhead = 4;
        uint4 ahead[kAhead];
#pragma unroll
        for (int j = 0; j < kAhead; ++j) {
            const int i = first + j * stride;
            int s;
            size_t off;
            if (i < total) ahead[j] = *source(i, s, off);
        }
        const int e = a.epochs[blockIdx.x] + 1;
        const int buf = e % 3;
        auto destination = [&](int s, size_t off) {
            return reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(a.recv[s]) + buf * buf_elems +
                                            a.rank * slot_elems + off);
        };
#pragma unroll
        for (int j = 0; j < kAhead; ++j) {
            const int i = first + j * stride;
            int s;
            size_t off;
            if (i < total) {
                source(i, s, off);
                *destination(s, off) = ahead[j];
            }
        }
        for (int i = first + kAhead * stride; i < total; i += stride) {
            int s;
            size_t off;
            const uint4 v = *source(i, s, off);
            *destination(s, off) = v;
        }
        stamp(a, 1);
        // Re-arm the receive buffer peers write in invocation e + 2 and the previous call site's output, over
        // that output's own size: calls on one chain may differ in rows.
        const uint4 sentinel = make_uint4(kSentinel, kSentinel, kSentinel, kSentinel);
        auto* stale = reinterpret_cast<__nv_bfloat16*>(a.recv[a.rank]) + ((e + 2) % 3) * buf_elems;
        for (int i = first; i < total; i += stride)
            *reinterpret_cast<uint4*>(stale + static_cast<size_t>(i) * kVec) = sentinel;
        for (int i = first; i < a.rearm_vecs; i += stride)
            *reinterpret_cast<uint4*>(a.rearm + static_cast<size_t>(i) * kVec) = sentinel;
        stamp(a, 2);
        stamp(a, 3);
        if (t == 0) a.epochs[blockIdx.x] = e;
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    // Owner CTAs take rows c = first, first + owners, ... (at most kMaxOwners CTAs, so that together with
    // the pushers the whole grid is resident: owners wait on each other across ranks).
    const int first = bid - P;
    const int e = a.epochs[blockIdx.x] + 1;
    const int buf = e % 3;
    const size_t col = static_cast<size_t>(t) * kVec;
    uint4 slots[kMaxWorld];
    for (int c = first; c < R; c += owners) {
        const size_t off = static_cast<size_t>(a.rank * R + c) * row_elems + col;
        // Local operands first, so their latency overlaps the wait for the peers.
        uint4 hraw = make_uint4(0, 0, 0, 0), wraw = make_uint4(0, 0, 0, 0);
        if constexpr (RESIDUAL) hraw = *reinterpret_cast<const uint4*>(a.h + off);
        if constexpr (NORM) wraw = *reinterpret_cast<const uint4*>(a.weight + col);
        // 2. Sum this rank's row R * rank + c over all ranks, in rank order.
        const auto* mine = reinterpret_cast<const __nv_bfloat16*>(a.recv[a.rank]) + buf * buf_elems;
        poll_all(mine + c * row_elems + col, slot_elems, a.world, slots);
        stamp(a, 1);
        float v[kVec] = {};
#pragma unroll
        for (int s = 0; s < kMaxWorld; ++s)
            if (s < a.world) add(v, slots[s]);
        if constexpr (NORM) {
            // Match the unfused path: the reduced sum is rounded to BF16 before the residual add.
            const uint4 rounded = pack(v);
#pragma unroll
            for (int i = 0; i < kVec; ++i) v[i] = 0.f;
            add(v, rounded);
            if constexpr (RESIDUAL) {
                add(v, hraw);
                *reinterpret_cast<uint4*>(a.h + off) = pack(v);
            }
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < kVec; ++i) ss += v[i] * v[i];
            const float r = rsqrtf(block_sum(ss, scratch) / a.n + a.eps) * a.scale;
            const __nv_bfloat162* w = reinterpret_cast<const __nv_bfloat162*>(&wraw);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f = __bfloat1622float2(w[i]);
                v[2 * i] = v[2 * i] * r * f.x;
                v[2 * i + 1] = v[2 * i + 1] * r * f.y;
            }
        }
        // 3. Broadcast the row: one multicast store, or one store per rank.
        const uint4 packed = pack(v);
        if (a.out_mc) {
            multicast_store(a.out_mc + off, packed);
        } else {
            for (int s = 0; s < a.world; ++s)
                *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(a.out_peers[s]) + off) = packed;
        }
    }
    stamp(a, 2);

    // 4. Wait for rows s * R + c of this rank's OUT copy from every owner s.
    for (int c = first; c < R; c += owners) poll_all(a.out + c * row_elems + col, slot_elems, a.world, slots);
    stamp(a, 3);
    if (t == 0) a.epochs[blockIdx.x] = e;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

template <bool NORM, bool RESIDUAL>
void launch(torch::Tensor partial, torch::Tensor recv, torch::Tensor h, torch::Tensor weight, torch::Tensor out,
            int64_t out_mc, torch::Tensor out_peers, int64_t rearm, int64_t rearm_numel, torch::Tensor epochs,
            int64_t rank, int64_t world, double eps, double scale, int64_t push_blocks, int64_t stamps, bool pdl) {
    const int64_t rows = partial.size(0), n = partial.size(1);
    TORCH_CHECK(n % (kVec * 32) == 0 && n / kVec <= 1024, "hidden size must be a multiple of 256, at most 8192");
    TORCH_CHECK(rows % world == 0 && push_blocks > 0, "rows must split over ranks; push blocks required");
    TORCH_CHECK(world <= kMaxWorld, "at most 8 ranks");
    TORCH_CHECK(rearm_numel % kVec == 0 && (rearm != 0 || rearm_numel == 0), "re-armed output: whole 16-byte vectors");
    const int R = static_cast<int>(rows / world);
    TORCH_CHECK(epochs.numel() >= R + push_blocks, "epoch counters too small");
    Args a{reinterpret_cast<const __nv_bfloat16*>(partial.data_ptr()),
           recv.data_ptr<int64_t>(),
           RESIDUAL ? reinterpret_cast<__nv_bfloat16*>(h.data_ptr()) : nullptr,
           NORM ? reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()) : nullptr,
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
           reinterpret_cast<__nv_bfloat16*>(out_mc),
           out_peers.data_ptr<int64_t>(),
           reinterpret_cast<__nv_bfloat16*>(rearm),
           epochs.data_ptr<int>(),
           reinterpret_cast<long long*>(stamps),
           static_cast<int>(rank), static_cast<int>(world), static_cast<int>(n),
           static_cast<int>(rearm_numel / kVec),
           static_cast<float>(eps), static_cast<float>(scale)};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(std::min(R, kMaxOwners) + push_blocks));
    config.blockDim = dim3(static_cast<unsigned>(n / kVec));
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attr;
    config.numAttrs = pdl ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, reduce_kernel<NORM, RESIDUAL>, a, R));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_norm_residual", &launch<true, true>);
    m.def("reduce_norm", &launch<true, false>);
    m.def("reduce", &launch<false, false>);
}
