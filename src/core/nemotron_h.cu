// src/core/nemotron_h.cu - the Nemotron-H engine's own GPU kernels (include/guild/core/nemotron_h_kernels.hpp).
//
// The Mamba-2 scan runs token after token inside one launch (a block per head, the state in registers), in prompt
// windows as in decode: a verify window's rows are bitwise one-token steps, and a prompt costs one pass over its
// tokens per layer (the reference's chunked form computes the same recurrence in another order).
#include "guild/core/nemotron_h_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace guild::core::nhk {

namespace {

inline cudaStream_t S(void* st) {
    return (cudaStream_t) st;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.f + expf(-x));
}

/// a block-wide sum (blockDim.x a multiple of 32, at most 1024); every thread gets it
__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i)
        s += red[i];
    __syncthreads();
    return s;
}
__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = -INFINITY;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i)
        s = fmaxf(s, red[i]);
    __syncthreads();
    return s;
}

// ---- the convolution -----------------------------------------------------------------------------------------------

__global__ void ssm_conv_k(const float* in, int ld_in, float* state, const float* w, const float* b, float* out, int C,
                           int T, float* snaps, int n_snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float h0 = state[c * 3 + 0], h1 = state[c * 3 + 1], h2 = state[c * 3 + 2];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], bc = b ? b[c] : 0.f;
    for (int t = 0; t < T; ++t) {
        const float x = in[(size_t) t * ld_in + c];
        const float acc = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x + bc;
        out[(size_t) t * C + c] = silu(acc);
        h0 = h1;
        h1 = h2;
        h2 = x;
        if (snaps && t < n_snaps) {
            float* s = snaps + (size_t) t * C * 3 + (size_t) c * 3;
            s[0] = h0;
            s[1] = h1;
            s[2] = h2;
        }
    }
    state[c * 3 + 0] = h0;
    state[c * 3 + 1] = h1;
    state[c * 3 + 2] = h2;
}

// ---- the scan: a block of 256 threads per head; thread (p, q) holds s[p][q * 32 .. q * 32 + 31] ---------------------

constexpr int kP = 64, kN = 128, kQ = 4, kPer = kN / kQ;

__global__ void __launch_bounds__(kP* kQ)
    ssm_scan_k(const float* xbc, int ld, const float* dt_raw, int ld_dt, const float* dt_bias, const float* A,
               const float* Dv, float* state, float* y, int H, int G, int T, float dt_min, float* snaps, int n_snaps) {
    const int h = blockIdx.x, g = h / (H / G);
    const int p = threadIdx.x / kQ, q = threadIdx.x % kQ;
    const int HP = H * kP;
    float s[kPer];
    float* sp = state + ((size_t) h * kP + p) * kN + q * kPer;
#pragma unroll
    for (int i = 0; i < kPer; ++i)
        s[i] = sp[i];
    __shared__ float Bs[kN], Cs[kN], xs[kP];
    __shared__ float dts;
    const float a = A[h], d = Dv[h], bias = dt_bias[h];
    for (int t = 0; t < T; ++t) {
        const float* row = xbc + (size_t) t * ld;
        if (threadIdx.x < kN) {
            Bs[threadIdx.x] = row[HP + g * kN + threadIdx.x];
            Cs[threadIdx.x] = row[HP + G * kN + g * kN + threadIdx.x];
        }
        if (threadIdx.x < kP)
            xs[threadIdx.x] = row[h * kP + threadIdx.x];
        if (threadIdx.x == 0) {
            const float r = dt_raw[(size_t) t * ld_dt + h] + bias;
            const float sp_ = r <= 20.f ? log1pf(expf(r)) : r; // torch's softplus (threshold 20)
            dts = fmaxf(sp_, dt_min);
        }
        __syncthreads();
        const float dt = dts, dA = expf(dt * a), x = xs[p];
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const float dB = dt * Bs[q * kPer + i];
            s[i] = s[i] * dA + dB * x;
            acc += s[i] * Cs[q * kPer + i];
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        if (q == 0)
            y[(size_t) t * HP + h * kP + p] = acc + d * x;
        if (snaps && t < n_snaps) {
            float* o = snaps + (size_t) t * H * kP * kN + ((size_t) h * kP + p) * kN + q * kPer;
#pragma unroll
            for (int i = 0; i < kPer; ++i)
                o[i] = s[i];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < kPer; ++i)
        sp[i] = s[i];
}

// ---- norms and activations -----------------------------------------------------------------------------------------

__global__ void gated_group_norm_k(const float* y, const float* z, int ld_z, const float* w, float* out, int I,
                                   int group, float eps) {
    const int t = blockIdx.y, g0 = blockIdx.x * group;
    __shared__ float red[32];
    const float* yr = y + (size_t) t * I + g0;
    const float* zr = z + (size_t) t * ld_z + g0;
    float ss = 0.f;
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = yr[i] * silu(zr[i]);
        ss += v * v;
    }
    const float r = rsqrtf(block_sum(ss, red) / (float) group + eps);
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float v = yr[i] * silu(zr[i]);
        out[(size_t) t * I + g0 + i] = w[g0 + i] * (v * r);
    }
}

__global__ void relu2_k(const float* x, float* y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float r = fmaxf(x[i], 0.f);
        y[i] = r * r;
    }
}

__global__ void layer_norm_k(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        s += xr[i];
    const float mean = block_sum(s, red) / (float) n;
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float r = rsqrtf(block_sum(v, red) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[(size_t) blockIdx.x * n + i] = (xr[i] - mean) * r * (w ? w[i] : 1.f);
}

// ---- attention
// -------------------------------------------------------------------------------------------------------

__global__ void kv_store_k(const float* k, const float* v, __half* kc, __half* vc, int64_t pos0, int kv_dim, int T) {
    const int64_t n = (int64_t) T * kv_dim;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t t = i / kv_dim, c = i % kv_dim;
        kc[(pos0 + t) * kv_dim + c] = __float2half_rn(k[i]);
        vc[(pos0 + t) * kv_dim + c] = __float2half_rn(v[i]);
    }
}

constexpr int kMaxCells = 8192; ///< the scores of one (query, head) in shared memory: 32 KiB

/// a block of 128 threads per (query, head): scores in shared memory, then the weighted sum (a thread per channel)
__global__ void __launch_bounds__(128) gqa_attend_k(const float* q, const __half* kc, const __half* vc, float* out,
                                                    int Hq, int Hk, int64_t pos0, float scale) {
    constexpr int D = 128;
    const int t = blockIdx.y, h = blockIdx.x, hk = h / (Hq / Hk);
    const int64_t cells = pos0 + t + 1;
    extern __shared__ float sc[];
    __shared__ float qs[D];
    __shared__ float red[32];
    qs[threadIdx.x] = q[((size_t) t * Hq + h) * D + threadIdx.x];
    __syncthreads();
    const int kv_dim = Hk * D;
    float m = -INFINITY;
    for (int64_t j = threadIdx.x; j < cells; j += blockDim.x) {
        const __half2* kr = (const __half2*) (kc + j * kv_dim + hk * D);
        float acc = 0.f;
#pragma unroll 8
        for (int i = 0; i < D / 2; ++i) {
            const float2 kk = __half22float2(kr[i]);
            acc += qs[2 * i] * kk.x + qs[2 * i + 1] * kk.y;
        }
        acc *= scale;
        sc[j] = acc;
        m = fmaxf(m, acc);
    }
    m = block_max(m, red);
    float sum = 0.f;
    for (int64_t j = threadIdx.x; j < cells; j += blockDim.x) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red); // its syncthreads also publish sc[]
    float acc = 0.f;
    const int d = threadIdx.x;
    for (int64_t j = 0; j < cells; ++j)
        acc += sc[j] * __half2float(vc[j * kv_dim + hk * D + d]);
    out[((size_t) t * Hq + h) * D + d] = acc / sum;
}

int grid_for(int64_t n) {
    return (int) std::min<int64_t>((n + 255) / 256, 4096);
}

} // namespace

void ssm_conv(const float* in, int ld_in, float* state, const float* w, const float* b, float* out, int C, int T,
              float* snaps, int n_snaps, void* st) {
    ssm_conv_k<<<(C + 127) / 128, 128, 0, S(st)>>>(in, ld_in, state, w, b, out, C, T, snaps, n_snaps);
}

bool ssm_scan_supported(int P, int N) {
    return P == kP && N == kN;
}

void ssm_scan(const float* xbc, int ld, const float* dt_raw, int ld_dt, const float* dt_bias, const float* A,
              const float* D, float* state, float* y, int H, int P, int N, int G, int T, float dt_min, float* snaps,
              int n_snaps, void* st) {
    (void) P;
    (void) N;
    ssm_scan_k<<<H, kP * kQ, 0, S(st)>>>(xbc, ld, dt_raw, ld_dt, dt_bias, A, D, state, y, H, G, T, dt_min, snaps,
                                         n_snaps);
}

void gated_group_norm(const float* y, const float* z, int ld_z, const float* w, float* out, int I, int group, float eps,
                      int T, void* st) {
    gated_group_norm_k<<<dim3(I / group, T), 256, 0, S(st)>>>(y, z, ld_z, w, out, I, group, eps);
}

void relu2(const float* x, float* y, int64_t n, void* st) {
    relu2_k<<<grid_for(n), 256, 0, S(st)>>>(x, y, n);
}

void layer_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* st) {
    layer_norm_k<<<rows, 256, 0, S(st)>>>(x, w, y, n, eps);
}

void kv_store(const float* k, const float* v, void* kc, void* vc, int64_t pos0, int kv_dim, int T, void* st) {
    kv_store_k<<<grid_for((int64_t) T * kv_dim), 256, 0, S(st)>>>(k, v, (__half*) kc, (__half*) vc, pos0, kv_dim, T);
}

int64_t attend_max_cells() {
    return kMaxCells;
}

void gqa_attend(const float* q, const void* kc, const void* vc, float* out, int Hq, int Hk, int D, int64_t pos0, int T,
                float scale, void* st) {
    (void) D;
    const int64_t cells = pos0 + T;
    gqa_attend_k<<<dim3(Hq, T), 128, (size_t) cells * sizeof(float), S(st)>>>(q, (const __half*) kc, (const __half*) vc,
                                                                              out, Hq, Hk, pos0, scale);
}

} // namespace guild::core::nhk
