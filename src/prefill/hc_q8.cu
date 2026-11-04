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

__global__ void f16_to_f32_kernel(const __half* __restrict__ in, float* __restrict__ out, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        out[i] = __half2float(in[i]);
}

__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ hi, const uint16_t* __restrict__ lo,
                                   float* __restrict__ out, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        float v = __uint_as_float((uint32_t) hi[i] << 16);
        if (lo)
            v += __uint_as_float((uint32_t) lo[i] << 16);
        out[i] = v;
    }
}

// one thread a block: the scale's two bytes, then the block's 32 values
__global__ void hc_q8_to_blocks_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ scales,
                                       int64_t nblocks, uint8_t* __restrict__ out) {
    for (int64_t b = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; b < nblocks;
         b += (int64_t) gridDim.x * blockDim.x) {
        uint8_t* o = out + b * 34;
        const uint16_t d = scales[b];
        o[0] = (uint8_t) (d & 0xff);
        o[1] = (uint8_t) (d >> 8);
        const int4* src = reinterpret_cast<const int4*>(q + b * 32);
        const int4 v0 = src[0], v1 = src[1];
        const int8_t* c0 = reinterpret_cast<const int8_t*>(&v0);
        const int8_t* c1 = reinterpret_cast<const int8_t*>(&v1);
        for (int i = 0; i < 16; ++i) {
            o[2 + i] = (uint8_t) c0[i];
            o[18 + i] = (uint8_t) c1[i];
        }
    }
}

} // namespace

void bf16_to_f32(const uint16_t* hi, const uint16_t* lo, float* out, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) std::min<int64_t>((n + 255) / 256, 1024);
    bf16_to_f32_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(hi, lo, out, n);
}

void hc_q8_to_blocks(const int8_t* planes, int64_t rows, int64_t cols, void* out, void* stream) {
    const int64_t nb = rows * cols / 32;
    const unsigned blocks = (unsigned) std::min<int64_t>((nb + 255) / 256, 1024);
    hc_q8_to_blocks_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(
        planes, reinterpret_cast<const uint16_t*>(planes + rows * cols), nb, static_cast<uint8_t*>(out));
}

void f16_to_f32(const uint16_t* in, float* out, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) std::min<int64_t>((n + 255) / 256, 1024);
    f16_to_f32_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(reinterpret_cast<const __half*>(in), out, n);
}

void hc_q8_to_bf16(const int8_t* planes, int64_t rows, int64_t cols, uint16_t* out, void* stream) {
    const int64_t n = rows * cols;
    const unsigned blocks = (unsigned) std::min<int64_t>((n + 255) / 256, 1024);
    hc_q8_to_bf16_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(planes, reinterpret_cast<const __half*>(planes + n),
                                                                    n, out);
}

} // namespace guild::prefill
