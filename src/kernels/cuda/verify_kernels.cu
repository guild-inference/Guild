// src/kernels/cuda/verify_kernels.cu - see include/guild/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include "guild/kernels/verify_kernels.hpp"
#include "guild/kernels/dp4a.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace guild::kernels {
namespace {

__global__ void finite_first_bad_kernel(const float* values, unsigned long long count,
                                       unsigned long long* first_bad) {
    unsigned long long bad = ~0ull;
    for (unsigned long long i = (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (unsigned long long) gridDim.x * blockDim.x) {
        if ((__float_as_uint(values[i]) & 0x7f800000u) == 0x7f800000u) bad = i < bad ? i : bad;
    }
    __shared__ unsigned long long minima[256];
    minima[threadIdx.x] = bad;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            const auto other = minima[threadIdx.x + stride];
            minima[threadIdx.x] = other < minima[threadIdx.x] ? other : minima[threadIdx.x];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && minima[0] != ~0ull) atomicMin(first_bad, minima[0]);
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                            const float* __restrict__ offsets, const int32_t* __restrict__ tokens,
                                            int64_t n, int code_bits, int code_bias, int group_elems,
                                            unsigned long long row_codes, unsigned long long row_groups,
                                            float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), sc[group]);
    out[(size_t) t * n + i] = __fadd_rn(product, of ? of[group] : 0.0f);
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__global__ void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                                    const int32_t* __restrict__ index, int64_t n) {
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[(size_t) idx * stride + i];
}

__global__ void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                   uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}

__global__ void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes) {
    const int k = threadIdx.x;
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__global__ void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                             float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__global__ void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot,
                                  int32_t* __restrict__ dst, int32_t* __restrict__ count) {
    const int i = threadIdx.x;
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
__global__ void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids,
                                   long long n, E* __restrict__ dst) {
    const long long total = n * row_e;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__global__ void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i = threadIdx.x;
    if (i < n) ids[i] = table[ids[i]];
}

__global__ void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                    float* __restrict__ probs) {
    __shared__ float part[32];
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) s += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        probs[t] = 1.0f / tot;
    }
}

__global__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                  const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p) {
    const int row = *row_dev;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += (int64_t) gridDim.x * blockDim.x)
        R_dst[i] = R_src[(size_t) row * stride + i];
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        __threadfence_system();
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
}

__global__ void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

}  // namespace

void finite_first_bad(const float* values, uint64_t count, unsigned long long* first_bad, void* stream) {
    auto cs = static_cast<cudaStream_t>(stream);
    if (cudaMemsetAsync(first_bad, 0xff, sizeof(*first_bad), cs) != cudaSuccess)
        throw std::runtime_error("finite validation status initialization failed");
    if (!count) return;
    const unsigned blocks = unsigned(count / 256 + (count % 256 != 0) < 256
                                      ? count / 256 + (count % 256 != 0) : 256);
    finite_first_bad_kernel<<<blocks, 256, 0, cs>>>(values, count, first_bad);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    fetch_blobs_kernel<<<48 * 8, 256, 0, (cudaStream_t) stream>>>(src, n, (uint4*) dst, (long long) (blob_bytes / 16));
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    rebase_ptrs_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ptr, n, (unsigned long long) base, (long long) blob_bytes);
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    add_streams_broadcast_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                                   (cudaStream_t) stream>>>(h, e, R, n_embd, hc);
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    ident_hits_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, n, slot, dst, count);
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    mtp_select_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(R_src, R_stride, ids, row_dev, R_dst, tok_dst, out, j,
                                                            probs, out_p);
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (row_bytes % 16 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint4*) src, row_bytes / 16, ids, n, (uint4*) dst);
    else if (row_bytes % 4 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint32_t*) src, row_bytes / 4, ids, n, (uint32_t*) dst);
    else
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>(src, row_bytes, ids, n, dst);
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    map_ids_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(ids, table, n);
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    row_top_prob_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n_vocab, ids, probs);
    check("row_top_prob");
}

namespace {
__global__ void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    window_ids_kernel<<<dim3(8, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(steps, window, ids, (long long) ids_stride);
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    dense_steps_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(cells, n, steps);
    check("dense_steps");
}

namespace {
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    while (*flag < value) guild_spin_pause();
    __threadfence_system();
}
}  // namespace

namespace {
// one thread per window entry: up to kVerifyMaxT tokens x 10 routed experts (80), so 128 (#646 had 64: a window of
// 7+ tokens lost its last entries)
constexpr int kResidentPlanMax = 128;
static_assert(kVerifyMaxT * 10 <= kResidentPlanMax, "resident_plan: one thread per entry");
__global__ void __launch_bounds__(kResidentPlanMax) resident_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                     int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                     long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                     uint32_t ring) {
    __shared__ int32_t s_ids[kResidentPlanMax];
    __shared__ unsigned long long s_ptr[kResidentPlanMax];
    __shared__ int32_t s_first[kResidentPlanMax];
    __shared__ int32_t s_cnt[kResidentPlanMax];
    __shared__ int32_t s_gstart[kResidentPlanMax];
    __shared__ int s_bad;
    const int tid = threadIdx.x;
    if (tid == 0) s_bad = 0;
    __syncthreads();

    int32_t eid = -1;
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        const int32_t slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0) {
            atomicOr(&s_bad, 1);
        } else {
            s_ptr[tid] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        }
    }
    __syncthreads();
    if (s_bad) {
        if (tid == 0 && skip != nullptr) *skip = 0;
        return;
    }

    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
        const bool is_first = (first_j == tid);
        s_first[tid] = is_first ? 1 : 0;
        s_cnt[tid] = is_first ? count_same : 0;
    }
    __syncthreads();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (tid < n && first_j == tid) {
        int grp_idx = 0;
        int ent_start = 0;
        for (int j = 0; j < tid; ++j) {
            grp_idx += s_first[j];
            ent_start += s_cnt[j];
        }
        ptr[grp_idx] = s_ptr[tid];
        start[grp_idx] = ent_start;
        s_gstart[tid] = ent_start;
    }
    __syncthreads();

    if (tid < n) {
        const int out_idx = s_gstart[first_j] + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        int groups = 0;
        for (int j = 0; j < n; ++j) groups += s_first[j];
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
        if (skip != nullptr) {
            __threadfence();
            *skip = ring;
        }
    }
}
__global__ void wait_flag_ge_or_kernel(const volatile uint32_t* flag, uint32_t value, const volatile uint32_t* skip) {
    if (*skip == value) return;
    while (*flag < value) guild_spin_pause();
    __threadfence_system();
}
__global__ void copy_i32_unless_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n,
                                       const uint32_t* skip, uint32_t value) {
    if (*skip == value) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
__global__ void copy_or_zero_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                    const uint32_t* skip, uint32_t value) {
    const bool zero = *skip == value;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = zero ? make_float4(0.f, 0.f, 0.f, 0.f) : const_cast<const float4*>(src)[i];
}
}  // namespace

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    resident_plan_kernel<<<1, kResidentPlanMax, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off,
                                                              blob, plan, capx, skip, ring);
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    wait_flag_ge_or_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, skip);
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    copy_i32_unless_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n, skip, value);
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_or_zero_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4, skip,
                                                                    value);
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    wait_flag_ge_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value);
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    embedding_gather_dev_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0,
                                  (cudaStream_t) stream>>>(codes, scales, offsets, tokens, n, code_bits, code_bias,
                                                           group_elems, row_codes, row_groups, out);
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    broadcast_streams_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                               (cudaStream_t) stream>>>(x, R, n_embd, hc);
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    copy_indexed_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, stride, index, n);
    check("copy_indexed");
}

// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler
namespace { __global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t;
#if defined(GUILD_HIP_GFX906)
    t = wall_clock64() * 40ull;   // gfx906: the wall clock runs at 25 MHz (hipDeviceAttributeWallClockRate) -> ns
#elif defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    buf[i] = t;
} }
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    gpu_stamp_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(buf, i);
}

}  // namespace guild::kernels
