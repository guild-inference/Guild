// src/prefill/hc_q8.cu - see include/guild/prefill/hc_q8.hpp.
#include "guild/prefill/hc_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace guild::prefill {
namespace {

__global__ void hc_q8_to_bf16_kernel(const int8_t* __restrict__ q, const __half* __restrict__ scales, int64_t n,
                                     uint16_t* __restrict__ out) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float v = __half2float(scales[i >> 5]) * (float) q[i]; // exact: 11 x 8 significant bits
        uint32_t u = __float_as_uint(v);
        u += 0x7fffu + ((u >> 16) & 1u); // to nearest even (v is finite)
        out[i] = (uint16_t) (u >> 16);
    }
}

} // namespace

void hc_q8_to_bf16(const int8_t* planes, int64_t rows, int64_t cols, uint16_t* out, void* stream) {
    const int64_t n = rows * cols;
    const unsigned blocks = (unsigned) std::min<int64_t>((n + 255) / 256, 1024);
    hc_q8_to_bf16_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(planes, reinterpret_cast<const __half*>(planes + n),
                                                                    n, out);
}

} // namespace guild::prefill
