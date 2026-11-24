// src/core/verify_combine.cu - the verify window's host-combine add (include/guild/core/verify_combine.hpp).
#include "guild/core/verify_combine.hpp"

#include <cstdint>

namespace guild::core {

#if defined(GUILD_USE_HIP) || defined(GUILD_HIP_GFX906) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

bool combine_sums_f4(const float*, const float*, const float*, float*, int64_t, int, cudaStream_t) {
    return false; // the HIP builds keep the combine kernel (the PTX below is NVIDIA's)
}

#else

namespace {

// native_moe_combine_multi's k = 1 arithmetic as the instructions --use_fast_math gives it (this file is built
// without it): `parts[col] * weights[0]` a flush-to-zero multiply, `sum += shared[col]` a flush-to-zero add
__device__ __forceinline__ float mul_ftz(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float add_ftz(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// one float4 a thread; `sums` with plain loads, as copy_from_mapped_kernel reads mapped rows
__global__ void combine_sums_f4_kernel(const float4* sums, const float* __restrict__ weights,
                                       const float4* __restrict__ shared, float4* __restrict__ out, int64_t row4,
                                       int64_t n4) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4)
        return;
    const float w = weights[i / row4];
    const float4 p = sums[i], s = shared[i];
    float4 o;
    o.x = add_ftz(mul_ftz(p.x, w), s.x);
    o.y = add_ftz(mul_ftz(p.y, w), s.y);
    o.z = add_ftz(mul_ftz(p.z, w), s.z);
    o.w = add_ftz(mul_ftz(p.w, w), s.w);
    out[i] = o;
}

bool aligned16(const void* p) {
    return ((uintptr_t) p & 15) == 0;
}

} // namespace

bool combine_sums_f4(const float* sums, const float* weights, const float* shared, float* out, int64_t n_embd,
                     int n_tok, cudaStream_t stream) {
    if (n_tok < 1 || n_embd < 4 || (n_embd & 3) != 0 || !aligned16(sums) || !aligned16(shared) || !aligned16(out) ||
        weights == nullptr)
        return false;
    const int64_t row4 = n_embd / 4, n4 = row4 * n_tok;
    constexpr int kThreads = 256;
    combine_sums_f4_kernel<<<(unsigned) ((n4 + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        (const float4*) sums, weights, (const float4*) shared, (float4*) out, row4, n4);
    return cudaPeekAtLastError() == cudaSuccess;
}

#endif

} // namespace guild::core
