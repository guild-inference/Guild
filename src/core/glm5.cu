// src/core/glm5.cu - GLM-5.3-Flash's own GPU kernels (see include/guild/core/glm5_kernels.hpp).  The arithmetic
// follows llama.cpp's glm5next graph (src/models/glm5next.cpp, deepseek4.cpp's mHC and delta-net-base.cpp's
// autoregressive KDA step) operation for operation, in FP32.
#include "guild/core/glm5_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace guild::core::glm5k {
namespace {

inline cudaStream_t S(void* st) {
    return (cudaStream_t) st;
}

std::atomic<bool> g_one_wave{true};
bool one_wave() {
    return g_one_wave.load(std::memory_order_relaxed);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// a block-wide sum (blockDim.x a multiple of 32, at most 1024); every thread gets the result
__device__ float block_sum(float v) {
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0)
        red[warp] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i)
        s += red[i];
    return s;
}

__device__ __forceinline__ float sigmoidf_(float x) {
    return 1.f / (1.f + expf(-x));
}

__global__ void rms_norm_k(const float* x, const float* w, float* y, int n, float eps) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = xr[i] * inv * (w ? w[i] : 1.f);
}
// rows of 4096 values or more (the mHC pre-mix's 4 x 4096): 1024 threads and float4 loads.  256 threads read the
// 16384 floats in 16.7 us on a GTX 1070.
__global__ void __launch_bounds__(1024) rms_norm4_k(const float* x, const float* w, float* y, int n, float eps) {
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t) blockIdx.x * n);
    float4* yr = reinterpret_cast<float4*>(y + (size_t) blockIdx.x * n);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        const float4 v = xr[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float inv = rsqrtf(block_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
        float4 v = xr[i];
        const float4 g = w ? w4[i] : make_float4(1.f, 1.f, 1.f, 1.f);
        v.x = v.x * inv * g.x;
        v.y = v.y * inv * g.y;
        v.z = v.z * inv * g.z;
        v.w = v.w * inv * g.w;
        yr[i] = v;
    }
}

// one block per token: the 4x4 Sinkhorn mix in shared memory, then the pre-weighted stream sum and its RMS norm
// The mHC mixes from hc_fn (Q8_0, `rows` x K) and the streams R (K floats a token) in one pass, the norm left as a
// scale: part[t][r][ks] = row r . R[t] over slice ks of kMixSplit, and part[t][rows][ks] the slice's sum of squares
// (hc_pre takes mix = sum_ks part[r] * rsqrt(sum_ks part[rows] / K + eps)).  It replaces an RMS norm, a quantization
// and a 24-row product that kept a block or two of the GPU busy each (~15 us a call on a GTX 1070).
constexpr int kMixSplit = 8;

__global__ void __launch_bounds__(128)
    hc_mix_k(const uint8_t* __restrict__ W, const float* __restrict__ R, float* __restrict__ part, int K, int rows) {
    const int r = blockIdx.x, ks = blockIdx.y, t = blockIdx.z;
    const int slice = K / kMixSplit, k0 = ks * slice;
    const float* x = R + (size_t) t * K + k0;
    float acc = 0.f;
    if (r < rows) {
        const uint8_t* w = W + ((size_t) r * (K / 32) + k0 / 32) * 34;
        for (int i = threadIdx.x; i < slice / 16; i += blockDim.x) { // half blocks of 16 values
            const int b = i >> 1, h = i & 1;
            const uint8_t* blk = w + (size_t) b * 34;
            const float d = __half2float(*reinterpret_cast<const __half*>(blk));
            const uint16_t* q = reinterpret_cast<const uint16_t*>(blk + 2 + 16 * h);
            const float4* xv = reinterpret_cast<const float4*>(x + b * 32 + 16 * h);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 v = xv[j];
                const uint16_t q01 = q[2 * j], q23 = q[2 * j + 1];
                sum += (float) (int8_t) (q01 & 0xff) * v.x + (float) (int8_t) (q01 >> 8) * v.y +
                       (float) (int8_t) (q23 & 0xff) * v.z + (float) (int8_t) (q23 >> 8) * v.w;
            }
            acc += d * sum;
        }
    } else {
        for (int i = threadIdx.x; i < slice / 4; i += blockDim.x) {
            const float4 v = reinterpret_cast<const float4*>(x)[i];
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0)
        part[((size_t) t * (rows + 1) + r) * kMixSplit + ks] = acc;
}

// Two blocks a token (grid.y): block 0 the pre-weighted stream sum and its norm, block 1 the post weights and the
// Sinkhorn mix (comb is read only by hc_post, after the half layer), so neither waits for the other; each computes the
// mixes the same way.  As one block, the stream sum waited for the 20 Sinkhorn rounds on one warp.
__global__ void hc_pre_k(const float* mixes, const float* part, int K, const float* scale, const float* base,
                         const float* R, const float* nw, float* post, float* comb, float* xn, int N, int iters,
                         float hc_eps, float rms_eps, float* xh) {
    extern __shared__ float xs[]; // N floats
    __shared__ float pre_s[kHc], c[kHc * kHc], msh[(2 + kHc) * kHc];
    const int t = blockIdx.x;
    // one block a token (gridDim.y == 1): both halves, the Sinkhorn first
    const bool mixer = blockIdx.y == 1, both = gridDim.y == 1;
    if (mixer && threadIdx.x >= 32)
        return;
    // lanes 0..15 of warp 0 own comb element (dst = lane % 4, src = lane / 4): a row (fixed src) is 4 adjacent lanes
    // (shuffles 1, 2), a column (fixed dst) lanes 4 apart (shuffles 4, 8)
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        constexpr int M = (2 + kHc) * kHc;
        const float* m = mixes + (size_t) t * M;
        if (part) { // hc_mix's slices: the products, then the norm
            const float* pt = part + (size_t) t * (M + 1) * kMixSplit;
            float ss = 0.f;
#pragma unroll
            for (int ks = 0; ks < kMixSplit; ++ks)
                ss += pt[M * kMixSplit + ks];
            const float inv = rsqrtf(ss / K + rms_eps);
            if (lane < M) {
                float v = 0.f;
#pragma unroll
                for (int ks = 0; ks < kMixSplit; ++ks)
                    v += pt[lane * kMixSplit + ks];
                msh[lane] = v * inv;
            }
            __syncwarp();
            m = msh;
        }
        const unsigned mask = 0xffffu;
        if (lane < kHc && !mixer)
            pre_s[lane] = sigmoidf_(m[lane] * scale[0] + base[lane]) + hc_eps;
        if (lane < kHc && (mixer || both))
            post[t * kHc + lane] = 2.f * sigmoidf_(m[kHc + lane] * scale[1] + base[kHc + lane]);
        if (lane < kHc * kHc && (mixer || both)) {
            float v = m[2 * kHc + lane] * scale[2] + base[2 * kHc + lane];
            float mx = fmaxf(v, __shfl_xor_sync(mask, v, 1));
            mx = fmaxf(mx, __shfl_xor_sync(mask, mx, 2));
            v = expf(v - mx);
            float sum = v + __shfl_xor_sync(mask, v, 1);
            sum += __shfl_xor_sync(mask, sum, 2);
            v = v / sum + hc_eps;
            auto cols = [&]() {
                float c4 = v + __shfl_xor_sync(mask, v, 4);
                c4 += __shfl_xor_sync(mask, c4, 8);
                v /= c4 + hc_eps;
            };
            auto rows = [&]() {
                float r4 = v + __shfl_xor_sync(mask, v, 1);
                r4 += __shfl_xor_sync(mask, r4, 2);
                v /= r4 + hc_eps;
            };
            cols();
            for (int i = 1; i < iters; ++i) {
                rows();
                cols();
            }
            comb[t * kHc * kHc + lane] = v;
        }
    }
    if (mixer)
        return;
    __syncthreads();
    const float* Rt = R + (size_t) t * kHc * N;
    float ss = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int h = 0; h < kHc; ++h)
            v += Rt[(size_t) h * N + i] * pre_s[h];
        xs[i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss) / N + rms_eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = xs[i] * inv * nw[i];
        xn[(size_t) t * N + i] = v;
        if (xh) // the same row in mapped host memory, for the CPU's FFN (no copy queued behind this kernel)
            xh[(size_t) t * N + i] = v;
    }
}

// y2 (when given): the layer output is y + y2, added as add_k adds them
__global__ void hc_post_k(const float* y, float* R, const float* post, const float* comb, int N, const float* y2) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    float* Rt = R + (size_t) t * kHc * N;
    const float* p = post + t * kHc;
    const float* c = comb + t * kHc * kHc;
    float r[kHc];
#pragma unroll
    for (int h = 0; h < kHc; ++h)
        r[h] = Rt[(size_t) h * N + i];
    float yi = y[(size_t) t * N + i];
    if (y2)
        yi += y2[(size_t) t * N + i];
#pragma unroll
    for (int dst = 0; dst < kHc; ++dst) {
        float v = yi * p[dst];
#pragma unroll
        for (int src = 0; src < kHc; ++src)
            v += r[src] * c[dst + kHc * src];
        Rt[(size_t) dst * N + i] = v;
    }
}

__global__ void hc_mean_norm_k(const float* R, const float* w, float* out, int N, float eps) {
    extern __shared__ float xs[];
    const int t = blockIdx.x;
    const float* Rt = R + (size_t) t * kHc * N;
    float ss = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int h = 0; h < kHc; ++h)
            v += Rt[(size_t) h * N + i];
        v *= 1.f / kHc;
        xs[i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss) / N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        out[(size_t) t * N + i] = xs[i] * inv * w[i];
}

__global__ void hc_init_k(const float* e, float* R, int N) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const float v = e[(size_t) t * N + i];
#pragma unroll
    for (int h = 0; h < kHc; ++h)
        R[((size_t) t * kHc + h) * N + i] = v;
}

// one thread per channel of q|k|v, the tokens in order
__global__ void kda_conv_k(const float* xq, const float* xk, const float* xv, float* state, const float* wq,
                           const float* wk, const float* wv, float* out, int D, int T, float* mid, int nmid) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = 3 * D;
    if (c >= C)
        return;
    const int part = c / D, cc = c - part * D;
    const float* w = (part == 0 ? wq : part == 1 ? wk : wv) + (size_t) cc * 4;
    const float* x = part == 0 ? xq : part == 1 ? xk : xv;
    float s0 = state[(size_t) c * 3], s1 = state[(size_t) c * 3 + 1], s2 = state[(size_t) c * 3 + 2];
    for (int t = 0; t < T; ++t) {
        const float xi = x[(size_t) t * D + cc];
        const float v = w[0] * s0 + w[1] * s1 + w[2] * s2 + w[3] * xi;
        out[(size_t) t * C + c] = v / (1.f + expf(-v));
        s0 = s1;
        s1 = s2;
        s2 = xi;
        if (t < nmid) { // the state after token t: a verify window's rollback point
            float* m = mid + ((size_t) t * C + c) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    state[(size_t) c * 3] = s0;
    state[(size_t) c * 3 + 1] = s1;
    state[(size_t) c * 3 + 2] = s2;
}

// L2 normalisation of q and k, one block (128 threads) per (head of q|k, token)
__global__ void kda_l2_k(float* qkv, int D, float eps) {
    const int head = blockIdx.x, t = blockIdx.y; // heads 0 .. 2D/128: q heads then k heads
    float* v = qkv + (size_t) t * 3 * D + (size_t) head * kKdaDim;
    const float x = v[threadIdx.x];
    __shared__ float red[4];
    float s = warp_sum(x * x);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = s;
    __syncthreads();
    s = red[0] + red[1] + red[2] + red[3];
    v[threadIdx.x] = x / fmaxf(sqrtf(s), eps);
}

__global__ void kda_gates_k(const float* fpre, const float* dt_b, const float* ssm_a, float* g, const float* bpre,
                            float* beta, int D, int H, int T, float lb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D * T) {
        const int c = i % D;
        const float a = ssm_a[c / kKdaDim];
        g[i] = lb * sigmoidf_(-(fpre[i] + dt_b[c]) * a);
    }
    if (i < H * T)
        beta[i] = sigmoidf_(bpre[i]);
}

// Block (head, tile of 32 value columns): 256 threads, lane = the column in the tile, warp = a slice of 16 keys.  Each
// thread keeps its 16 x 1 piece of S in registers for the whole window; the column sums go through shared memory.
constexpr int kKdaTile = 32, kKdaSlices = 8, kKdaPer = kKdaDim / kKdaSlices;

__global__ void __launch_bounds__(256) kda_step_k(const float* S_in, float* S_out, const float* qkv, const float* g,
                                                  const float* beta, float* o, int H, int T, float* mid) {
    constexpr int DK = kKdaDim;
    const int h = blockIdx.x, lane = threadIdx.x & 31, sl = threadIdx.x >> 5;
    const int j = blockIdx.y * kKdaTile + lane;
    const int D = H * DK;
    __shared__ float sq[DK], sk[DK], sg[DK];
    __shared__ float part[kKdaSlices][kKdaTile];
    const float* Sh = S_in + (size_t) h * DK * DK;
    float col[kKdaPer];
#pragma unroll
    for (int i = 0; i < kKdaPer; ++i)
        col[i] = Sh[(sl * kKdaPer + i) * DK + j];
    const float qscale = rsqrtf((float) DK);
    for (int t = 0; t < T; ++t) {
        const float* row = qkv + (size_t) t * 3 * D;
        __syncthreads();
        if (threadIdx.x < DK) {
            const int i = threadIdx.x;
            sq[i] = row[h * DK + i] * qscale;
            sk[i] = row[D + h * DK + i];
            sg[i] = expf(g[(size_t) t * D + h * DK + i]);
        }
        __syncthreads();
        float ks = 0.f;
#pragma unroll
        for (int i = 0; i < kKdaPer; ++i) {
            const int key = sl * kKdaPer + i;
            col[i] *= sg[key];
            ks += col[i] * sk[key];
        }
        part[sl][lane] = ks;
        __syncthreads();
        float tot = 0.f;
#pragma unroll
        for (int k = 0; k < kKdaSlices; ++k)
            tot += part[k][lane];
        const float d = (row[2 * D + h * DK + j] - tot) * beta[t * H + h];
        __syncthreads();
        float out = 0.f;
#pragma unroll
        for (int i = 0; i < kKdaPer; ++i) {
            const int key = sl * kKdaPer + i;
            col[i] += sk[key] * d;
            out += col[i] * sq[key];
        }
        part[sl][lane] = out;
        __syncthreads();
        if (sl == 0) {
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < kKdaSlices; ++k)
                sum += part[k][lane];
            o[(size_t) t * D + h * DK + j] = sum;
        }
        if (t == 0 && mid) {
            float* Mh = mid + (size_t) h * DK * DK;
#pragma unroll
            for (int i = 0; i < kKdaPer; ++i)
                Mh[(sl * kKdaPer + i) * DK + j] = col[i];
        }
    }
    if (!S_out)
        return;
    float* Oh = S_out + (size_t) h * DK * DK;
#pragma unroll
    for (int i = 0; i < kKdaPer; ++i)
        Oh[(sl * kKdaPer + i) * DK + j] = col[i];
}

__global__ void kda_out_k(const float* o, const float* gate, const float* nw, float* y, int H, float eps) {
    const int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    const size_t off = (size_t) t * H * kKdaDim + (size_t) h * kKdaDim + j;
    const float x = o[off];
    __shared__ float red[4];
    float s = warp_sum(x * x);
    if ((j & 31) == 0)
        red[j >> 5] = s;
    __syncthreads();
    s = red[0] + red[1] + red[2] + red[3];
    y[off] = x * rsqrtf(s / kKdaDim + eps) * nw[j] * sigmoidf_(gate[off]);
}

// GGUF Q8_0: an FP16 scale and 32 int8 values, 34 bytes
__device__ __forceinline__ float q8_0_scale(const uint8_t* b) {
    return __half2float(*reinterpret_cast<const __half*>(b));
}

// one warp per (head, row) and tile of kTile tokens (grid.y), on the split layout (q8_0_split): lane l takes the P =
// n_in / 32 consecutive values l P .. l P + P - 1 (one 8- or 16-byte load), all inside one 32-value block
constexpr int kTile = 8;

// Per-head Q8_0 products in q8_0_split's layout: a lane per row, so no sum across lanes.  The split interleaves 32 rows
// 16 values at a time, so a warp's 16-byte loads read 512 contiguous bytes; the head's activations (TT tokens) sit in
// shared memory, where the warp's lanes all read the same value (a broadcast).  A warp per row with a lane reduction
// per row read the GTX 1070's memory at 46 (rows of 256) and 69 GB/s (512).
template <int TT, int G = 1>
__global__ void __launch_bounds__(256)
    q8_0_heads_k(const int8_t* __restrict__ qs, const __half* __restrict__ d, const float* __restrict__ x,
                 float* __restrict__ y, int H, int n_in, int n_out, int T) {
    extern __shared__ __align__(16) float xs[]; // [TT][n_in]
    const int nw = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * nw * 32 * G, h = row0 / n_out; // a block's rows lie in one head
    const int t0 = blockIdx.y * TT, nt = min(TT, T - t0);
    for (int i = threadIdx.x; i < nt * n_in / 4; i += blockDim.x) {
        const int t = i / (n_in / 4), k = i % (n_in / 4);
        reinterpret_cast<float4*>(xs)[i] = reinterpret_cast<const float4*>(x + ((size_t) (t0 + t) * H + h) * n_in)[k];
    }
    __syncthreads();
    // G groups of 32 rows a warp (group j: row0 / 32 + j * nw + warp), their loads side by side
    const int nc = n_in / 16;
    const int4* wq[G];
    const __half* wd[G];
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int g = row0 / 32 + j * nw + warp;
        wq[j] = reinterpret_cast<const int4*>(qs) + (size_t) g * nc * 32 + lane;
        wd[j] = d + (size_t) g * (n_in / 32) * 32 + lane;
    }
    float acc[G][TT];
#pragma unroll
    for (int j = 0; j < G; ++j)
#pragma unroll
        for (int t = 0; t < TT; ++t)
            acc[j][t] = 0.f;
#pragma unroll 2
    for (int c = 0; c < nc; ++c) {
        int4 v[G];
        float sc[G];
#pragma unroll
        for (int j = 0; j < G; ++j) {
            v[j] = __ldg(wq[j] + (size_t) c * 32);
            sc[j] = __half2float(wd[j][(size_t) (c / 2) * 32]);
        }
#pragma unroll
        for (int j = 0; j < G; ++j) {
            const int8_t* b = reinterpret_cast<const int8_t*>(&v[j]);
            float w[16];
#pragma unroll
            for (int i = 0; i < 16; ++i)
                w[i] = (float) b[i];
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t >= nt)
                    break;
                const float4* xr = reinterpret_cast<const float4*>(xs + t * n_in + c * 16);
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 xv = xr[i];
                    s += w[4 * i] * xv.x + w[4 * i + 1] * xv.y + w[4 * i + 2] * xv.z + w[4 * i + 3] * xv.w;
                }
                acc[j][t] += sc[j] * s;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        const int r = (row0 / 32 + j * nw + warp) * 32 + lane - h * n_out;
#pragma unroll
        for (int t = 0; t < TT; ++t)
            if (t < nt)
                y[((size_t) (t0 + t) * H + h) * n_out + r] = acc[j][t];
    }
}

// Q6_K rows (ggml's layout) times Q8_1 activations, one or two tokens: llama.cpp's vec_dot_q6_K_q8_1 in the order of
// guild::kernels::native_mmvq (a lane per quarter-block slot, four sums over every fourth block, added in order, then
// the lanes), so the result is bitwise that kernel's.  Q6_K's 210-byte blocks are only 2-byte aligned, and the plain
// kernel's 2-byte loads were more load instructions than the bandwidth (151 against Q8_0's 188 GB/s on a GTX 1070):
// here a warp per row stages 8 blocks (1680 bytes) at a time in shared memory with 16-byte loads, the next 8 loading
// while it reads the last, and the block's warps share the activations from shared memory.
constexpr int kQ6Block = 210, kQ6Chunk = 8, kQ6Q81 = 36;
// a warp's staging: a chunk, and the 16-byte pieces' head and tail around a chunk that does not start 16-byte aligned
constexpr int kQ6WarpBytes = kQ6Chunk * kQ6Block + 32;

struct Q81 { // ggml's block_q8_1
    __half2 ds;
    int8_t qs[32];
};

__device__ __forceinline__ int ld_b2(const uint8_t* p, int i32) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return (int) h[2 * i32] | ((int) h[2 * i32 + 1] << 16);
}

// Ids: output row r is the product of row ids[r] of W (a subset of the rows, any order)
template <int T, int NW, bool Ids = false>
__global__ void __launch_bounds__(NW * 32)
    q6k_mmvq_k(const uint8_t* __restrict__ W, const uint8_t* __restrict__ x, float* __restrict__ y, int n_in, int n_out,
               const int32_t* __restrict__ ids = nullptr) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int nb = n_in / 256, xb = n_in / 32, xbytes = T * xb * kQ6Q81;
    for (int o = threadIdx.x * 16; o < xbytes; o += NW * 32 * 16)
        *reinterpret_cast<int4*>(smem + o) = __ldg(reinterpret_cast<const int4*>(x + o));
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * NW + warp;
    if (row >= n_out)
        return;
    const size_t rb = (size_t) nb * kQ6Block;
    const uint8_t* wr = W + (size_t) (Ids ? __ldg(ids + row) : row) * rb;
    uint8_t* sm = smem + xbytes + warp * kQ6WarpBytes;
    const bool v16 = rb % 16 == 0;
    // the lane's fixed slot in every block (iqs = lane)
    const int iqs = lane, bq8 = 4 * (iqs / 16) + (iqs % 16) / 8, so = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8), qh_i = 8 * (iqs / 16) + iqs % 8;
    float acc[T][4];
#pragma unroll
    for (int t = 0; t < T; ++t)
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    constexpr int kV = (kQ6Chunk * kQ6Block / 16 + 31) / 32; // 16-byte pieces a lane carries per chunk (4)
    int4 pf[kV];
    auto fetch = [&](int c0) {
        const int bytes = min(kQ6Chunk, nb - c0) * kQ6Block;
        const uint8_t* src = wr + (size_t) c0 * kQ6Block;
#pragma unroll
        for (int k = 0; k < kV; ++k) {
            const int o = (lane + 32 * k) * 16;
            if (o < bytes)
                pf[k] = __ldg(reinterpret_cast<const int4*>(src + o));
        }
    };
    if (v16)
        fetch(0);
    for (int c0 = 0; c0 < nb; c0 += kQ6Chunk) {
        const int cn = min(kQ6Chunk, nb - c0), bytes = cn * kQ6Block;
        int head = 0; // where the chunk starts in sm
        __syncwarp();
        if (v16) {
#pragma unroll
            for (int k = 0; k < kV; ++k) {
                const int o = (lane + 32 * k) * 16;
                if (o < bytes)
                    *reinterpret_cast<int4*>(sm + o) = pf[k];
            }
            if (c0 + kQ6Chunk < nb)
                fetch(c0 + kQ6Chunk);
        } else {
            // rows that are not 16-byte multiples (DSA q_b: 6 blocks, 1,260 bytes): the 16-byte pieces that cover the
            // chunk, its first byte `head` bytes into the first (2-byte pieces one after another: 171-176 -> 129.5 us
            // on q_b, glm5_kernels_test q6k; the rows' bytes past either end are read, not used)
            const uint8_t* src = wr + (size_t) c0 * kQ6Block;
            head = (int) ((uintptr_t) src & 15);
            const int n16 = (head + bytes + 15) / 16;
            constexpr int kP = (kQ6WarpBytes / 16 + 31) / 32;
            int4 v[kP];
#pragma unroll
            for (int k = 0; k < kP; ++k)
                if (lane + 32 * k < n16)
                    v[k] = __ldg(reinterpret_cast<const int4*>(src - head) + lane + 32 * k);
#pragma unroll
            for (int k = 0; k < kP; ++k)
                if (lane + 32 * k < n16)
                    reinterpret_cast<int4*>(sm)[lane + 32 * k] = v[k];
        }
        __syncwarp();
#pragma unroll 2
        for (int b = 0; b < cn; ++b) {
            const uint8_t* blk = sm + head + b * kQ6Block; // ql[128] qh[64] scales[16] d
            const int kbx = c0 + b;
            const int vl = ld_b2(blk, iqs);
            const int vh = ld_b2(blk + 128, qh_i) >> vh_shift;
            const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + so;
            const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
            int vi[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
                const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
                vi[i] = __vsubss4(vil | vih, 0x20202020);
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const uint8_t* xt = smem + ((size_t) t * xb + kbx * 8) * kQ6Q81;
                float sumf = 0.0f;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const Q81& q = *reinterpret_cast<const Q81*>(xt + (bq8 + 2 * i) * kQ6Q81);
                    const int u = reinterpret_cast<const int*>(q.qs)[iqs % 8];
                    sumf += __low2float(q.ds) * (__dp4a(vi[i], u, 0) * (int) sc[4 * i]);
                }
                acc[t][kbx & 3] += d * sumf;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = acc[t][0];
        v += acc[t][1];
        v += acc[t][2];
        v += acc[t][3];
        v = warp_sum(v);
        if (lane == 0)
            y[(size_t) t * n_out + row] = v;
    }
}

// q6k_mmvq_k with each warp's rows in a loop (row, row + gridDim.x * NW, ...) on as many blocks as fit the GPU at
// once, and the next chunk - the row's next, or the first of the warp's next row - loading while the warp uses the
// current one; rows of any alignment through 16-byte pieces.  For rows of one or two chunks the kernel above has a
// warp load its row, wait, use it and exit: DSA q_b (6 blocks, 1,260 bytes a row) 140.8 -> 102.3 us, the 4096-wide
// rows (KDA q/k/v, the draft head) 2-4% faster (glm5_kernels_test q6k, GTX 1070).  The arithmetic, its order and the
// lane of every value are the kernel above's, so the results are bitwise its own.
template <int T, int NW, bool Ids = false>
__global__ void __launch_bounds__(NW * 32, 4)
    q6k_rows_k(const uint8_t* __restrict__ W, const uint8_t* __restrict__ x, float* __restrict__ y, int n_in, int n_out,
               const int32_t* __restrict__ ids = nullptr) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int nb = n_in / 256, xb = n_in / 32, xbytes = T * xb * kQ6Q81;
    for (int o = threadIdx.x * 16; o < xbytes; o += NW * 32 * 16)
        *reinterpret_cast<int4*>(smem + o) = __ldg(reinterpret_cast<const int4*>(x + o));
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int stride = gridDim.x * NW;
    int row = blockIdx.x * NW + warp;
    if (row >= n_out)
        return;
    const size_t rb = (size_t) nb * kQ6Block;
    uint8_t* sm = smem + xbytes + warp * kQ6WarpBytes;
    const int iqs = lane, bq8 = 4 * (iqs / 16) + (iqs % 16) / 8, so = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8), qh_i = 8 * (iqs / 16) + iqs % 8;
    constexpr int kV = (kQ6WarpBytes / 16 + 31) / 32; // 16-byte pieces a lane carries (4)
    int4 pf[kV];
    int pf_head = 0, pf_n16 = 0;
    auto row_ptr = [&](int r) { return W + (size_t) (Ids ? __ldg(ids + r) : r) * rb; };
    // the 16-byte pieces covering [src, src + bytes), the first `pf_head` bytes into the first
    auto fetch = [&](const uint8_t* src, int bytes) {
        pf_head = (int) ((uintptr_t) src & 15);
        pf_n16 = (pf_head + bytes + 15) / 16;
#pragma unroll
        for (int k = 0; k < kV; ++k)
            if (lane + 32 * k < pf_n16)
                pf[k] = __ldg(reinterpret_cast<const int4*>(src - pf_head) + lane + 32 * k);
    };
    const uint8_t* wr = row_ptr(row);
    fetch(wr, min(kQ6Chunk, nb) * kQ6Block);
    float acc[T][4];
    while (true) {
#pragma unroll
        for (int t = 0; t < T; ++t)
            acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        const int next = row + stride;
        const uint8_t* wn = next < n_out ? row_ptr(next) : nullptr;
        for (int c0 = 0; c0 < nb; c0 += kQ6Chunk) {
            const int cn = min(kQ6Chunk, nb - c0);
            __syncwarp();
            const int head = pf_head, n16 = pf_n16;
#pragma unroll
            for (int k = 0; k < kV; ++k)
                if (lane + 32 * k < n16)
                    reinterpret_cast<int4*>(sm)[lane + 32 * k] = pf[k];
            if (c0 + kQ6Chunk < nb)
                fetch(wr + (size_t) (c0 + kQ6Chunk) * kQ6Block, min(kQ6Chunk, nb - c0 - kQ6Chunk) * kQ6Block);
            else if (wn)
                fetch(wn, min(kQ6Chunk, nb) * kQ6Block);
            __syncwarp();
#pragma unroll 2
            for (int b = 0; b < cn; ++b) {
                const uint8_t* blk = sm + head + b * kQ6Block; // ql[128] qh[64] scales[16] d
                const int kbx = c0 + b;
                const int vl = ld_b2(blk, iqs);
                const int vh = ld_b2(blk + 128, qh_i) >> vh_shift;
                const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + so;
                const float d = __half2float(*reinterpret_cast<const __half*>(blk + 208));
                int vi[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
                    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
                    vi[i] = __vsubss4(vil | vih, 0x20202020);
                }
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const uint8_t* xt = smem + ((size_t) t * xb + kbx * 8) * kQ6Q81;
                    float sumf = 0.0f;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const Q81& q = *reinterpret_cast<const Q81*>(xt + (bq8 + 2 * i) * kQ6Q81);
                        const int u = reinterpret_cast<const int*>(q.qs)[iqs % 8];
                        sumf += __low2float(q.ds) * (__dp4a(vi[i], u, 0) * (int) sc[4 * i]);
                    }
                    acc[t][kbx & 3] += d * sumf;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = acc[t][0];
            v += acc[t][1];
            v += acc[t][2];
            v += acc[t][3];
            v = warp_sum(v);
            if (lane == 0)
                y[(size_t) t * n_out + row] = v;
        }
        if (!wn)
            break;
        row = next;
        wr = wn;
    }
}

// Q5_K rows (ggml's layout) times Q8_1 activations, one or two tokens: native_mmvq's vec_dot_q5_K_q8_1 in the order of
// its one-token kernel (four warps a row, sixteen threads a block at two ints each, eight blocks an iteration: thread
// t's sum takes blocks t / 16, t / 16 + 8, ...; the warps' sums added in order, then the lanes), so every output is
// bitwise that kernel's.  A warp a row, in a loop over rows as q6k_rows_k's, its blocks staged in shared memory eight
// at a time with 16-byte loads (176-byte blocks keep a row 16-byte aligned), the next eight loading while it reads
// these: lanes 0-15 take a pair's even block and 16-31 its odd one, lane l standing for the threads l, 32 + l, 64 + l
// and 96 + l of that kernel, whose sums are acc[0 .. 3].  (native_mmvq's own Q5_K kernel reads each int with its own
// load: 163 against Q6_K's 211 GB/s for 4096 x 24576 on a GTX 1070.)
constexpr int kQ5Block = 176, kQ5Chunk = 8, kQ5WarpBytes = kQ5Chunk * kQ5Block;

// vec_dot_q5_K_q8_1 for the thread whose iqs / 2 is `i2` (0 .. 15): blk is the Q5_K block (dm, scales[12], qh[32],
// qs[128]), xb its 256 values' eight Q8_1 blocks
__device__ __forceinline__ float q5k_dot(const uint8_t* blk, const uint8_t* xb, int i2) {
    const int bq8_offset = 2 * (i2 / 4);
    const int* ql = reinterpret_cast<const int*>(blk + 48 + 16 * bq8_offset + 4 * (i2 % 4));
    const int* qh = reinterpret_cast<const int*>(blk + 16 + 4 * (i2 % 4));
    int vl[2], vh[2], u[4];
    float d8[2];
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(blk + 4);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81& q = *reinterpret_cast<const Q81*>(xb + (bq8_offset + i) * kQ6Q81);
        d8[i] = __low2float(q.ds);
        const int* q8 = reinterpret_cast<const int*>(q.qs) + (i2 % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = __dp4a(v0i, u[2 * i], __dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i], __dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(*reinterpret_cast<const __half2*>(blk));
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

template <int T, int NW>
__global__ void __launch_bounds__(NW * 32, 4) q5k_rows_k(const uint8_t* __restrict__ W, const uint8_t* __restrict__ x,
                                                         float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int nb = n_in / 256, xb = n_in / 32, xbytes = T * xb * kQ6Q81;
    for (int o = threadIdx.x * 16; o < xbytes; o += NW * 32 * 16)
        *reinterpret_cast<int4*>(smem + o) = __ldg(reinterpret_cast<const int4*>(x + o));
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int stride = gridDim.x * NW;
    int row = blockIdx.x * NW + warp;
    if (row >= n_out)
        return;
    const size_t rb = (size_t) nb * kQ5Block;
    uint8_t* sm = smem + xbytes + warp * kQ5WarpBytes;
    const int half = lane >> 4, i2 = lane & 15;
    constexpr int kV = (kQ5WarpBytes / 16 + 31) / 32; // 16-byte pieces a lane carries (3)
    int4 pf[kV];
    int pf_n16 = 0;
    // a chunk's 16-byte pieces (whole: 176-byte blocks)
    auto fetch = [&](const uint8_t* src, int bytes) {
        pf_n16 = bytes / 16;
#pragma unroll
        for (int k = 0; k < kV; ++k)
            if (lane + 32 * k < pf_n16)
                pf[k] = __ldg(reinterpret_cast<const int4*>(src) + lane + 32 * k);
    };
    const uint8_t* wr = W + (size_t) row * rb;
    fetch(wr, min(kQ5Chunk, nb) * kQ5Block);
    float acc[T][4];
    while (true) {
#pragma unroll
        for (int t = 0; t < T; ++t)
            acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        const int next = row + stride;
        const uint8_t* wn = next < n_out ? W + (size_t) next * rb : nullptr;
        for (int c0 = 0; c0 < nb; c0 += kQ5Chunk) {
            const int cn = min(kQ5Chunk, nb - c0);
            __syncwarp();
            const int n16 = pf_n16;
#pragma unroll
            for (int k = 0; k < kV; ++k)
                if (lane + 32 * k < n16)
                    reinterpret_cast<int4*>(sm)[lane + 32 * k] = pf[k];
            if (c0 + kQ5Chunk < nb)
                fetch(wr + (size_t) (c0 + kQ5Chunk) * kQ5Block, min(kQ5Chunk, nb - c0 - kQ5Chunk) * kQ5Block);
            else if (wn)
                fetch(wn, min(kQ5Chunk, nb) * kQ5Block);
            __syncwarp();
#pragma unroll 2
            for (int b = 0; b < cn; b += 2) { // the pair (c0 + b, c0 + b + 1): the oracle's thread sums (b / 2) % 4
                const int bb = b + half;
                if (bb < cn) {
                    const uint8_t* blk = sm + bb * kQ5Block;
                    const int kbx = c0 + bb, a = ((c0 + b) >> 1) & 3;
#pragma unroll
                    for (int t = 0; t < T; ++t)
                        acc[t][a] += q5k_dot(blk, smem + ((size_t) t * xb + kbx * 8) * kQ6Q81, i2);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = acc[t][0];
            v += acc[t][1];
            v += acc[t][2];
            v += acc[t][3];
            v = warp_sum(v);
            if (lane == 0)
                y[(size_t) t * n_out + row] = v;
        }
        if (!wn)
            break;
        row = next;
        wr = wn;
    }
}

// Q8_0 rows times Q8_1 activations, the same way: native_mmvq's order (four lanes a block, a lane's two ints at
// 2 * (lane % 4), blocks lane / 4 + 8 y + 32 i for its four sums y, added in order, then the lanes), so bitwise its
// one-token result, with a warp per row staging 32 blocks (1088 bytes) at a time with 16-byte loads.
constexpr int kQ8Block = 34, kQ8Chunk = 32;

template <int T, int NW>
__global__ void __launch_bounds__(NW * 32) q8_0_mmvq_k(const uint8_t* __restrict__ W, const uint8_t* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int nb = n_in / 32, xbytes = T * nb * kQ6Q81;
    for (int o = threadIdx.x * 16; o < xbytes; o += NW * 32 * 16)
        *reinterpret_cast<int4*>(smem + o) = __ldg(reinterpret_cast<const int4*>(x + o));
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * NW + warp;
    if (row >= n_out)
        return;
    const uint8_t* wr = W + (size_t) row * nb * kQ8Block;
    uint8_t* sm = smem + xbytes + warp * (kQ8Chunk * kQ8Block);
    const int kqs = 2 * (lane % 4), slot = lane / 4;
    float acc[T][4];
#pragma unroll
    for (int t = 0; t < T; ++t)
        acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    constexpr int kV = (kQ8Chunk * kQ8Block / 16 + 31) / 32; // 16-byte pieces a lane carries per chunk (3)
    int4 pf[kV];
    auto fetch = [&](int c0) {
        const int bytes = min(kQ8Chunk, nb - c0) * kQ8Block;
        const uint8_t* src = wr + (size_t) c0 * kQ8Block;
#pragma unroll
        for (int k = 0; k < kV; ++k) {
            const int o = (lane + 32 * k) * 16;
            if (o < bytes)
                pf[k] = __ldg(reinterpret_cast<const int4*>(src + o));
        }
    };
    fetch(0);
    for (int c0 = 0; c0 < nb; c0 += kQ8Chunk) {
        const int bytes = min(kQ8Chunk, nb - c0) * kQ8Block;
        __syncwarp();
#pragma unroll
        for (int k = 0; k < kV; ++k) {
            const int o = (lane + 32 * k) * 16;
            if (o < bytes)
                *reinterpret_cast<int4*>(sm + o) = pf[k];
        }
        if (c0 + kQ8Chunk < nb)
            fetch(c0 + kQ8Chunk);
        __syncwarp();
#pragma unroll
        for (int yy = 0; yy < 4; ++yy) {
            const int b = slot + 8 * yy, kbx = c0 + b;
            if (kbx >= nb)
                break;
            const uint8_t* blk = sm + b * kQ8Block; // d, qs[32]
            const float d0 = __half2float(*reinterpret_cast<const __half*>(blk));
            const int v0 = ld_b2(blk + 2, kqs);
            const int v1 = ld_b2(blk + 2, kqs + 1);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const Q81& q = *reinterpret_cast<const Q81*>(smem + ((size_t) t * nb + kbx) * kQ6Q81);
                const int* u = reinterpret_cast<const int*>(q.qs);
                int sumi = __dp4a(v0, u[kqs], 0);
                sumi = __dp4a(v1, u[kqs + 1], sumi);
                acc[t][yy] += d0 * __low2float(q.ds) * (float) sumi;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = acc[t][0];
        v += acc[t][1];
        v += acc[t][2];
        v += acc[t][3];
        v = warp_sum(v);
        if (lane == 0)
            y[(size_t) t * n_out + row] = v;
    }
}

// Q8_0 rows of 64, 128 or 256 values (the KDA gates' second halves, 128 -> 8192) times Q8_1 activations: a warp stages
// C chunks of 32 blocks (1088 bytes: 32 / nb whole rows each) with 16-byte loads, issued before the activations are
// read, a lane takes one block's 32 products and the row's nb lanes add up with shuffles.  native_mmvq read these rows
// at 48 GB/s.
template <int T, int NB, int C = 1>
__global__ void __launch_bounds__(256)
    q8_0_narrow_k(const uint8_t* __restrict__ W, const uint8_t* __restrict__ x, float* __restrict__ y, int n_out) {
    __shared__ __align__(16) uint8_t stage[8][C][kQ8Chunk * kQ8Block];
    __shared__ __align__(16) uint8_t xs[T * NB * kQ6Q81];
    constexpr int R = 32 / NB; // rows a chunk
    constexpr int KV = (kQ8Chunk * kQ8Block / 16 + 31) / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // the warp's C chunks (chunk j: rows (blockIdx.x * 8 * C + j * 8 + warp) * R ...), loaded before the activations
    int4 v16[C][KV];
    int rows[C];
#pragma unroll
    for (int j = 0; j < C; ++j) {
        const int row0 = (blockIdx.x * 8 * C + j * 8 + warp) * R;
        rows[j] = row0 < n_out ? min(R, n_out - row0) : 0;
        const int bytes = rows[j] * NB * kQ8Block;
        const uint8_t* src = W + (size_t) row0 * NB * kQ8Block;
#pragma unroll
        for (int k = 0; k < KV; ++k)
            if ((lane + 32 * k) * 16 < bytes)
                v16[j][k] = __ldg(reinterpret_cast<const int4*>(src) + lane + 32 * k);
    }
    for (int o = threadIdx.x * 4; o < T * NB * kQ6Q81; o += 256 * 4)
        *reinterpret_cast<int*>(xs + o) = __ldg(reinterpret_cast<const int*>(x + o));
    __syncthreads();
#pragma unroll
    for (int j = 0; j < C; ++j) {
        if (rows[j] == 0)
            break;
        const int row0 = (blockIdx.x * 8 * C + j * 8 + warp) * R, bytes = rows[j] * NB * kQ8Block;
        uint8_t* sm = stage[warp][j];
#pragma unroll
        for (int k = 0; k < KV; ++k)
            if ((lane + 32 * k) * 16 < bytes)
                reinterpret_cast<int4*>(sm)[lane + 32 * k] = v16[j][k];
        __syncwarp();
        const int r = lane / NB, b = lane % NB;
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; ++t)
            acc[t] = 0.f;
        if (r < rows[j]) {
            const uint8_t* blk = sm + lane * kQ8Block;
            const float d0 = __half2float(*reinterpret_cast<const __half*>(blk));
            int v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                v[i] = ld_b2(blk + 2, i);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const Q81& q = *reinterpret_cast<const Q81*>(xs + (t * NB + b) * kQ6Q81);
                const int* u = reinterpret_cast<const int*>(q.qs);
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    sumi = __dp4a(v[i], u[i], sumi);
                acc[t] = d0 * __low2float(q.ds) * (float) sumi;
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
#pragma unroll
            for (int o = NB / 2; o > 0; o >>= 1)
                acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
            if (b == 0 && r < rows[j])
                y[(size_t) t * n_out + row0 + r] = acc[t];
        }
    }
}

// one block (128 threads) per row and tile of kTile tokens: float4 loads, a shuffle-then-shared reduction
__global__ void __launch_bounds__(128)
    f32_matvec_k(const float* W, const float* x, float* y, int n_in, int n_out, int T) {
    const int r = blockIdx.x, t0 = blockIdx.y * kTile;
    T = min(kTile, T - t0);
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t) r * n_in);
    float acc[kTile];
#pragma unroll
    for (int t = 0; t < kTile; ++t)
        acc[t] = 0.f;
    for (int k = threadIdx.x; k < n_in / 4; k += blockDim.x) {
        const float4 w = wr[k];
#pragma unroll
        for (int t = 0; t < kTile; ++t)
            if (t < T) {
                const float4 xv = reinterpret_cast<const float4*>(x + (size_t) (t0 + t) * n_in)[k];
                acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
            }
    }
    __shared__ float red[kTile][4];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
#pragma unroll
    for (int t = 0; t < kTile; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0)
            red[t][wp] = v;
    }
    __syncthreads();
    if (threadIdx.x < T)
        y[(size_t) (t0 + threadIdx.x) * n_out + r] =
            red[threadIdx.x][0] + red[threadIdx.x][1] + red[threadIdx.x][2] + red[threadIdx.x][3];
}

// f32_matvec_k on bf16-valued weights held as their high halves (bf16_matvec): the same sums in the same order
__global__ void __launch_bounds__(128)
    bf16_matvec_k(const uint16_t* W, const float* x, float* y, int n_in, int n_out, int T) {
    const int r = blockIdx.x, t0 = blockIdx.y * kTile;
    T = min(kTile, T - t0);
    const uint2* wr = reinterpret_cast<const uint2*>(W + (size_t) r * n_in);
    float acc[kTile];
#pragma unroll
    for (int t = 0; t < kTile; ++t)
        acc[t] = 0.f;
    for (int k = threadIdx.x; k < n_in / 4; k += blockDim.x) {
        const uint2 u = wr[k];
        const float4 w = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                                     __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
#pragma unroll
        for (int t = 0; t < kTile; ++t)
            if (t < T) {
                const float4 xv = reinterpret_cast<const float4*>(x + (size_t) (t0 + t) * n_in)[k];
                acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
            }
    }
    __shared__ float red[kTile][4];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
#pragma unroll
    for (int t = 0; t < kTile; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0)
            red[t][wp] = v;
    }
    __syncthreads();
    if (threadIdx.x < T)
        y[(size_t) (t0 + threadIdx.x) * n_out + r] =
            red[threadIdx.x][0] + red[threadIdx.x][1] + red[threadIdx.x][2] + red[threadIdx.x][3];
}

// one or two tokens (the routers, 4096 -> 288): a warp per row, four rows a block, eight float4 loads in flight a lane.
// A block per row read the routers at 130 GB/s on a GTX 1070 (288 blocks: a wave and a fifth).
template <int T>
__global__ void __launch_bounds__(128)
    f32_rows_k(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 4 + warp;
    if (r >= n_out)
        return;
    const float4* wr = reinterpret_cast<const float4*>(W + (size_t) r * n_in);
    const int n4 = n_in / 4;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t)
        acc[t] = 0.f;
#pragma unroll 8
    for (int k = lane; k < n4; k += 32) {
        const float4 w = __ldg(wr + k);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (size_t) t * n_in) + k);
            acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0)
            y[(size_t) t * n_out + r] = v;
    }
}

// f32_rows_k on a router whose F32 values are all bf16 values (GLM-5.3-Flash's are), held as their high halves: half
// the bytes.  A lane reads float4 k's four values as a uint2 and widens them exactly, so every product and sum is
// f32_rows_k's, in its order.
template <int T>
__global__ void __launch_bounds__(128) bf16_rows_k(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                   float* __restrict__ y, int n_in, int n_out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 4 + warp;
    if (r >= n_out)
        return;
    const uint2* wr = reinterpret_cast<const uint2*>(W + (size_t) r * n_in);
    const int n4 = n_in / 4;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t)
        acc[t] = 0.f;
#pragma unroll 8
    for (int k = lane; k < n4; k += 32) {
        const uint2 u = __ldg(wr + k);
        const float4 w = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                                     __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (size_t) t * n_in) + k);
            acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0)
            y[(size_t) t * n_out + r] = v;
    }
}

__global__ void kv_store_k(const float* kv, __half* cache, int64_t pos0, int L, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L * T)
        return;
    const int t = i / L, d = i - t * L;
    cache[(pos0 + t) * L + d] = __float2half(kv[i]);
}

// Absorbed MLA attention, flash-decoding style.  Block (split, head group of 8, token): warp w is head
// 8 * group + w; lane l holds q and the accumulator for dims l, l + 32, ... (L / 32 of them).  The block walks its
// share of the token's cells in chunks of kCh, with an online softmax, and writes (max, sum, lat[L]) per head.
constexpr int kCh = 16, kHg = 8, kMaxL = 512;
// splits of a query's cells: many for a short window (the blocks a decode step needs), few for a long one
inline int splits_for(int T) {
    return T >= 32 ? 1 : T >= 16 ? 2 : T >= 4 ? 4 : 16;
}

__global__ void __launch_bounds__(256)
    mla_attend_k(const float* q, const __half* cache, int64_t pos0, int H, int L, float scale, const int32_t* sel,
                 const int32_t* n_sel, int sel_stride, int kSplits, float* part) {
    constexpr int PER = kMaxL / 32;
    __shared__ __half kv_s[kCh][kMaxL];
    const int split = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = hg * kHg + warp;
    const int n_cells = sel ? n_sel[t] : (int) (pos0 + t + 1);
    const int per = (n_cells + kSplits - 1) / kSplits;
    const int c0 = split * per, c1 = min(n_cells, c0 + per);
    const int nl = L / 32;
    float qr[PER], acc[PER];
    const float* qh = q + ((size_t) t * H + h) * L;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        qr[k] = k < nl ? qh[lane + 32 * k] * scale : 0.f;
        acc[k] = 0.f;
    }
    float m = -FLT_MAX, l = 0.f;
    for (int cb = c0; cb < c1; cb += kCh) {
        const int nc = min(kCh, c1 - cb);
        __syncthreads();
        for (int i = threadIdx.x; i < nc * L; i += blockDim.x) {
            const int c = i / L, d = i - c * L;
            const int cell = sel ? sel[(size_t) t * sel_stride + cb + c] : cb + c;
            kv_s[c][d] = cache[(size_t) cell * L + d];
        }
        __syncthreads();
        float s[kCh];
        float mx = m;
#pragma unroll
        for (int c = 0; c < kCh; ++c) {
            float v = 0.f;
            if (c < nc) {
#pragma unroll
                for (int k = 0; k < PER; ++k)
                    if (k < nl)
                        v += qr[k] * __half2float(kv_s[c][lane + 32 * k]);
                v = warp_sum(v);
                mx = fmaxf(mx, v);
            }
            s[c] = v;
        }
        const float corr = expf(m - mx);
        l *= corr;
#pragma unroll
        for (int k = 0; k < PER; ++k)
            acc[k] *= corr;
#pragma unroll
        for (int c = 0; c < kCh; ++c) {
            if (c >= nc)
                break;
            const float p = expf(s[c] - mx);
            l += p;
#pragma unroll
            for (int k = 0; k < PER; ++k)
                if (k < nl)
                    acc[k] += p * __half2float(kv_s[c][lane + 32 * k]);
        }
        m = mx;
    }
    float* out = part + (((size_t) t * H + h) * kSplits + split) * (L + 2);
    if (lane == 0) {
        out[0] = c0 < c1 ? m : -FLT_MAX;
        out[1] = l;
    }
#pragma unroll
    for (int k = 0; k < PER; ++k)
        if (k < nl)
            out[2 + lane + 32 * k] = acc[k];
}

// mla_attend_k's arithmetic with fewer instructions: a chunk's rows go to shared memory as they are (FP16), lane l's
// values (dims l, l + 32, ...) side by side, so a lane reads a cell's 16 values in two 16-byte loads (piece p at
// p ^ ((l >> 2) & 1), so a quarter warp's reads hit 32 banks) instead of sixteen 2-byte loads, and converts them to
// float (exactly) as mla_attend_k does; the rows are read from global memory 16 bytes a thread.  Every product, sum
// and softmax step is mla_attend_k's, in its order.  16 KiB a block; kMinB blocks an SM (80 or 64 registers).  GTX
// 1070, 2,048 selected cells, 64 heads (glm5_kernels_test attend): a token 680 -> 331 us with three blocks (357 with
// four), 128 tokens 64.6 -> 32.2 ms with four (34.0 with three).  Tried: the rows as floats (32 KiB, three blocks):
// 367 us and 39.3 ms.
__device__ __forceinline__ int lane_at_h(int lane, int k) { // k: a value's index in the lane (0 .. PER - 1)
    return lane * (kMaxL / 32) + ((((k >> 3) ^ ((lane >> 2) & 1)) << 3) | (k & 7));
}
template <int kMinB>
__global__ void __launch_bounds__(256, kMinB)
    mla_attend2_k(const float* q, const __half* cache, int64_t pos0, int H, int L, float scale, const int32_t* sel,
                  const int32_t* n_sel, int sel_stride, int kSplits, float* part) {
    constexpr int PER = kMaxL / 32;
    __shared__ __align__(16) __half kv_s[kCh][32 * PER];
    const int split = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = hg * kHg + warp;
    const int n_cells = sel ? n_sel[t] : (int) (pos0 + t + 1);
    const int per = (n_cells + kSplits - 1) / kSplits;
    const int c0 = split * per, c1 = min(n_cells, c0 + per);
    const int nl = L / 32, n8 = L / 8;
    const int32_t* tsel = sel ? sel + (size_t) t * sel_stride : nullptr;
    float qr[PER], acc[PER];
    const float* qh = q + ((size_t) t * H + h) * L;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        qr[k] = k < nl ? qh[lane + 32 * k] * scale : 0.f;
        acc[k] = 0.f;
    }
    float m = -FLT_MAX, l = 0.f;
    for (int cb = c0; cb < c1; cb += kCh) {
        const int nc = min(kCh, c1 - cb);
        __syncthreads();
        for (int i = threadIdx.x; i < nc * n8; i += blockDim.x) {
            const int c = i / n8, d = (i - c * n8) * 8;
            const int cell = tsel ? tsel[cb + c] : cb + c;
            const uint4 v = *(const uint4*) (cache + (size_t) cell * L + d);
            const __half* hv = (const __half*) &v;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                kv_s[c][lane_at_h((d & 31) + j, d >> 5)] = hv[j];
        }
        __syncthreads();
        float s[kCh];
        float mx = m;
#pragma unroll
        for (int c = 0; c < kCh; ++c) {
            float v = 0.f;
            if (c < nc) {
                const uint4 p0 = *(const uint4*) &kv_s[c][lane_at_h(lane, 0)];
                const uint4 p1 = *(const uint4*) &kv_s[c][lane_at_h(lane, 8)];
                const __half* h0 = (const __half*) &p0;
                const __half* h1 = (const __half*) &p1;
#pragma unroll
                for (int k = 0; k < PER; ++k)
                    if (k < nl)
                        v += qr[k] * __half2float(k < 8 ? h0[k] : h1[k - 8]);
                v = warp_sum(v);
                mx = fmaxf(mx, v);
            }
            s[c] = v;
        }
        const float corr = expf(m - mx);
        l *= corr;
#pragma unroll
        for (int k = 0; k < PER; ++k)
            acc[k] *= corr;
#pragma unroll
        for (int c = 0; c < kCh; ++c) {
            if (c >= nc)
                break;
            const float p = expf(s[c] - mx);
            l += p;
            const uint4 p0 = *(const uint4*) &kv_s[c][lane_at_h(lane, 0)];
            const uint4 p1 = *(const uint4*) &kv_s[c][lane_at_h(lane, 8)];
            const __half* h0 = (const __half*) &p0;
            const __half* h1 = (const __half*) &p1;
#pragma unroll
            for (int k = 0; k < PER; ++k)
                if (k < nl)
                    acc[k] += p * __half2float(k < 8 ? h0[k] : h1[k - 8]);
        }
        m = mx;
    }
    float* out = part + (((size_t) t * H + h) * kSplits + split) * (L + 2);
    if (lane == 0) {
        out[0] = c0 < c1 ? m : -FLT_MAX;
        out[1] = l;
    }
#pragma unroll
    for (int k = 0; k < PER; ++k)
        if (k < nl)
            out[2 + lane + 32 * k] = acc[k];
}

__global__ void mla_reduce_k(const float* part, float* lat, int H, int L, int kSplits) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float* p = part + ((size_t) t * H + h) * kSplits * (L + 2);
    float M = -FLT_MAX;
    for (int s = 0; s < kSplits; ++s)
        M = fmaxf(M, p[(size_t) s * (L + 2)]);
    float den = 0.f;
    for (int s = 0; s < kSplits; ++s) {
        const float ms = p[(size_t) s * (L + 2)];
        if (ms > -FLT_MAX)
            den += expf(ms - M) * p[(size_t) s * (L + 2) + 1];
    }
    for (int d = threadIdx.x; d < L; d += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < kSplits; ++s) {
            const float ms = p[(size_t) s * (L + 2)];
            if (ms > -FLT_MAX)
                v += expf(ms - M) * p[(size_t) s * (L + 2) + 2 + d];
        }
        lat[((size_t) t * H + h) * L + d] = v / den;
    }
}

// ---- the DSA indexer ----

// one block (128 threads) per token: LayerNorm of the key, then both rows to FP16 at the cell's row of the rings
// (cell c at row c % ring)
__global__ void idx_store_k(const float* kraw, const float* ln_w, const float* ln_b, const float* gate, __half* kc,
                            __half* gc, int64_t pos0, int ring, float eps) {
    const int t = blockIdx.x, d = threadIdx.x;
    const float x = kraw[(size_t) t * kIdxDim + d];
    __shared__ float red[4];
    float s = warp_sum(x);
    if ((d & 31) == 0)
        red[d >> 5] = s;
    __syncthreads();
    const float mean = (red[0] + red[1] + red[2] + red[3]) / kIdxDim;
    __syncthreads();
    const float c = x - mean;
    s = warp_sum(c * c);
    if ((d & 31) == 0)
        red[d >> 5] = s;
    __syncthreads();
    const float var = (red[0] + red[1] + red[2] + red[3]) / kIdxDim;
    const size_t row = (size_t) ((pos0 + t) % ring) * kIdxDim + d;
    kc[row] = __float2half(c * rsqrtf(var + eps) * ln_w[d] + ln_b[d]);
    gc[row] = __float2half(gate[(size_t) t * kIdxDim + d]);
}

// one block (128 threads, one per dim) per pool completed in the window; a pool's cells are consecutive rows of the
// rings (a ring is a multiple of kKpool rows)
__global__ void idx_pool_k(const __half* kc, const __half* gc, int ring, const float* ape, __half* pooled,
                           int64_t first_pool) {
    const int64_t p = first_pool + blockIdx.x;
    const int d = threadIdx.x;
    const size_t r0 = (size_t) ((p * kKpool) % ring);
    float g[kKpool], mx = -FLT_MAX;
#pragma unroll
    for (int s = 0; s < kKpool; ++s) {
        g[s] = __half2float(gc[(r0 + s) * kIdxDim + d]) + ape[s * kIdxDim + d];
        mx = fmaxf(mx, g[s]);
    }
    float den = 0.f, v = 0.f;
#pragma unroll
    for (int s = 0; s < kKpool; ++s) {
        const float e = expf(g[s] - mx);
        den += e;
        v += e * __half2float(kc[(r0 + s) * kIdxDim + d]);
    }
    // FP16, as the reference's indexer cache keeps it (its scores are taken against the rounded key)
    pooled[(size_t) p * kIdxDim + d] = __float2half(v / den);
}

// block (8 warps) per 8 pools x token: warp w scores pool 8 * blockIdx.x + w over all heads (GUILD_GLM_IDX_LEGACY=1)
__global__ void idx_score_k(const float* iq, const float* w, const __half* pooled, int64_t pos0, int heads,
                            float wscale, float* scores, int stride) {
    const int t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t p = (int64_t) blockIdx.x * 8 + warp;
    const int64_t n_vis = (pos0 + t + 1) / kKpool;
    if (p >= n_vis)
        return;
    float k4[kIdxDim / 32];
#pragma unroll
    for (int i = 0; i < kIdxDim / 32; ++i)
        k4[i] = __half2float(pooled[(size_t) p * kIdxDim + lane + 32 * i]);
    float score = 0.f;
    for (int h = 0; h < heads; ++h) {
        const float* q = iq + ((size_t) t * heads + h) * kIdxDim;
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < kIdxDim / 32; ++i)
            dot += q[lane + 32 * i] * k4[i];
        dot = warp_sum(dot);
        score += w[(size_t) t * heads + h] * wscale * fmaxf(dot, 0.f);
    }
    if (lane == 0)
        scores[(size_t) t * stride + p] = score;
}

// The same scores with the query read once per block, from shared memory, instead of once per pool (16 KiB each).
// Two threads per pool, one per parity of idx_score_k's lane slots: slot j = 2 s + e holds the key dims j, j + 32,
// j + 64 and j + 96, so a head's dot product is idx_score_k's to the bit - the same per-slot sums, warp_sum's
// butterfly over the slots (masks 16, 8, 4 and 2 inside a thread, 1 across the pair), the same sum over the heads.
constexpr int kScorePools = 128, kScoreHeads = 32;
__global__ void __launch_bounds__(2 * kScorePools)
    idx_score_pairs_k(const float* iq, const float* w, const __half* pooled, int64_t pos0, int heads, float wscale,
                      float* scores, int stride) {
    __shared__ float4 qs[kScoreHeads][32]; // [head][slot j]: q[j], q[j + 32], q[j + 64], q[j + 96]
    __shared__ float ws[kScoreHeads];
    const int t = blockIdx.y;
    const int64_t n_vis = (pos0 + t + 1) / kKpool, pool0 = (int64_t) blockIdx.x * kScorePools;
    if (pool0 >= n_vis)
        return;
    const float* q = iq + (size_t) t * heads * kIdxDim;
    for (int i = threadIdx.x; i < heads * 32; i += blockDim.x) {
        const float* qh = q + (size_t) (i >> 5) * kIdxDim + (i & 31);
        qs[i >> 5][i & 31] = make_float4(qh[0], qh[32], qh[64], qh[96]);
    }
    if (threadIdx.x < heads)
        ws[threadIdx.x] = w[(size_t) t * heads + threadIdx.x] * wscale;
    __syncthreads();
    const int64_t p = pool0 + (threadIdx.x >> 1);
    const int e = threadIdx.x & 1;
    // the key: 16 loads of 8 FP16 dims (8 c .. 8 c + 7 = slots 4 (c % 4) .. 4 (c % 4) + 3, both parities, of dim
    // row c / 4); this thread keeps its parity's
    float k[16][4];
    const uint4* kp = reinterpret_cast<const uint4*>(pooled + (size_t) min(p, n_vis - 1) * kIdxDim);
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        const uint4 v = __ldg(kp + c);
        const uint32_t word[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int r = 0; r < 4; ++r)
            k[4 * (c & 3) + r][c >> 2] =
                __half2float(__ushort_as_half((unsigned short) (e ? word[r] >> 16 : word[r] & 0xFFFFu)));
    }
    float score = 0.f;
    for (int h = 0; h < heads; ++h) {
        float d[16];
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const float4 qv = qs[h][2 * s + e];
            float dot = 0.f;
            dot += qv.x * k[s][0];
            dot += qv.y * k[s][1];
            dot += qv.z * k[s][2];
            dot += qv.w * k[s][3];
            d[s] = dot;
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) // slot j and j ^ 16
            d[s] += d[s + 8];
#pragma unroll
        for (int s = 0; s < 4; ++s) // j ^ 8
            d[s] += d[s + 4];
#pragma unroll
        for (int s = 0; s < 2; ++s) // j ^ 4
            d[s] += d[s + 2];
        d[0] += d[1]; // j ^ 2
        const float dot = d[0] + __shfl_xor_sync(0xffffffffu, d[0], 1);
        score += ws[h] * fmaxf(dot, 0.f);
    }
    if (e == 0 && p < n_vis)
        scores[(size_t) t * stride + p] = score;
}

__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block (1024 threads) per token: radix-free top-k by a binary search for the k-th largest key, then a
// deterministic, index-ordered compaction (keys above the threshold, then ties from the lowest pool up)
// (GUILD_GLM_IDX_LEGACY=1)
__global__ void __launch_bounds__(1024) idx_select_k(const float* scores, int stride, int64_t pos0, int k_pools,
                                                     int32_t* sel, int32_t* n_sel, int sel_stride) {
    const int t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int64_t q = pos0 + t;
    const int64_t tail_start = (q + 1) / kKpool * kKpool;
    const int n = (int) (tail_start / kKpool); // visible complete pools
    const float* s = scores + (size_t) t * stride;
    int32_t* out = sel + (size_t) t * sel_stride;
    __shared__ int cnt[1024], cnt2[1024];
    __shared__ uint32_t thr_s;
    __shared__ int need_eq;
    const int per = (n + nt - 1) / nt, i0 = min(n, tid * per), i1 = min(n, i0 + per);
    uint32_t thr = 0;
    bool all = n <= k_pools;
    if (!all) {
        // the largest v with count(key >= v) >= k_pools
        uint32_t lo = 0, hi = 0xFFFFFFFFu;
        while (lo < hi) {
            const uint32_t mid = lo + (uint32_t) (((uint64_t) hi - lo + 1) >> 1);
            int c = 0;
            for (int i = i0; i < i1; ++i)
                c += ord_key(s[i]) >= mid;
            const int total = (int) block_sum((float) c);
            if (total >= k_pools)
                lo = mid;
            else
                hi = mid - 1;
        }
        thr = lo;
    }
    // counts above / at the threshold per thread, then exclusive scans
    int gt = 0, eq = 0;
    for (int i = i0; i < i1; ++i) {
        if (all) {
            ++gt;
            continue;
        }
        const uint32_t kk = ord_key(s[i]);
        gt += kk > thr;
        eq += kk == thr;
    }
    cnt[tid] = gt;
    cnt2[tid] = eq;
    __syncthreads();
    if (tid == 0) {
        int a = 0, b = 0;
        for (int i = 0; i < nt; ++i) {
            const int x = cnt[i], y = cnt2[i];
            cnt[i] = a;
            cnt2[i] = b;
            a += x;
            b += y;
        }
        need_eq = all ? 0 : k_pools - a;
        thr_s = (uint32_t) a; // total above
    }
    __syncthreads();
    const int above = (int) thr_s;
    // pools above the threshold first (index order), then the first need_eq ties
    int o_gt = cnt[tid], o_eq = cnt2[tid];
    for (int i = i0; i < i1; ++i) {
        int slot = -1;
        if (all) {
            slot = o_gt++;
        } else {
            const uint32_t kk = ord_key(s[i]);
            if (kk > thr)
                slot = o_gt++;
            else if (kk == thr) {
                if (o_eq < need_eq)
                    slot = above + o_eq;
                ++o_eq;
            }
        }
        if (slot >= 0)
            for (int r = 0; r < kKpool; ++r)
                out[slot * kKpool + r] = (int32_t) (i * kKpool + r);
    }
    const int chosen = all ? n : k_pools;
    const int n_tail = (int) (q + 1 - tail_start);
    if (tid < n_tail)
        out[chosen * kKpool + tid] = (int32_t) (tail_start + tid);
    if (tid == 0)
        n_sel[t] = chosen * kKpool + n_tail;
}

// an exclusive scan of two counts over the block's threads, in thread order (blockDim.x a multiple of 32, at most
// 1024); `tot` gets the totals
__device__ __forceinline__ void block_scan2(int& a, int& b, int* tot) {
    __shared__ int part[2][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    int ia = a, ib = b;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int x = __shfl_up_sync(0xffffffffu, ia, o), y = __shfl_up_sync(0xffffffffu, ib, o);
        if (lane >= o) {
            ia += x;
            ib += y;
        }
    }
    if (lane == 31) {
        part[0][warp] = ia;
        part[1][warp] = ib;
    }
    __syncthreads();
    if (warp == 0) {
        const int x0 = lane < nw ? part[0][lane] : 0, y0 = lane < nw ? part[1][lane] : 0;
        int x = x0, y = y0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, x, o), v = __shfl_up_sync(0xffffffffu, y, o);
            if (lane >= o) {
                x += u;
                y += v;
            }
        }
        part[0][lane] = x - x0;
        part[1][lane] = y - y0;
        if (lane == 31) {
            tot[0] = x;
            tot[1] = y;
        }
    }
    __syncthreads();
    a = part[0][warp] + ia - a;
    b = part[1][warp] + ib - b;
}

// idx_select_k's selection, faster: the threshold (the k_pools-th largest key) found a byte at a time from the top -
// four histogram passes instead of 32 bisection rounds, each histogram kept per lane so that a warp's increments
// never meet on one address - and the output slots by a parallel scan; the same threshold and the same index-ordered
// compaction, so the same cells in the same order
__global__ void __launch_bounds__(1024) idx_select_radix_k(const float* scores, int stride, int64_t pos0, int k_pools,
                                                           int32_t* sel, int32_t* n_sel, int sel_stride) {
    const int t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x, lane = tid & 31, warp = tid >> 5;
    const int64_t q = pos0 + t;
    const int64_t tail_start = (q + 1) / kKpool * kKpool;
    const int n = (int) (tail_start / kKpool); // visible complete pools
    const float* s = scores + (size_t) t * stride;
    int32_t* out = sel + (size_t) t * sel_stride;
    __shared__ int hist[256][33]; // [digit][lane], padded: the lanes of a warp land in different banks
    __shared__ int bins[256];
    __shared__ uint32_t thr_s;
    __shared__ int want_s, tot[2];
    const bool all = n <= k_pools;
    uint32_t thr = 0;
    if (!all) {
        uint32_t prefix = 0, mask = 0;
        int want = k_pools; // the rank of the threshold among the keys matching the prefix
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256 * 33; i += nt)
                (&hist[0][0])[i] = 0;
            __syncthreads();
            for (int i = tid; i < n; i += nt) {
                const uint32_t kk = ord_key(s[i]);
                if ((kk & mask) == prefix)
                    atomicAdd(&hist[(kk >> shift) & 255u][lane], 1);
            }
            __syncthreads();
            if (tid < 256) {
                int c = 0;
                for (int l = 0; l < 32; ++l)
                    c += hist[tid][l];
                bins[tid] = c;
            }
            __syncthreads();
            if (warp == 0) {
                // lane l: digits 255 - 8 l down to 248 - 8 l, counted from the top digit down
                int c[8], sum = 0;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    c[r] = bins[255 - 8 * lane - r];
                    sum += c[r];
                }
                int incl = sum;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int v = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o)
                        incl += v;
                }
                const unsigned hit = __ballot_sync(0xffffffffu, incl >= want);
                if (lane == __ffs((int) hit) - 1) {
                    int run = incl - sum;
                    for (int r = 0; r < 8; ++r) {
                        if (run + c[r] >= want) {
                            thr_s = prefix | ((uint32_t) (255 - 8 * lane - r) << shift);
                            want_s = want - run;
                            break;
                        }
                        run += c[r];
                    }
                }
            }
            __syncthreads();
            prefix = thr_s;
            want = want_s;
            mask |= 255u << shift;
        }
        thr = prefix;
    }
    // counts above / at the threshold per thread (a contiguous range of pools each), then exclusive scans
    const int per = (n + nt - 1) / nt, i0 = min(n, tid * per), i1 = min(n, i0 + per);
    int gt = 0, eq = 0;
    if (all)
        gt = i1 - i0;
    else
        for (int i = i0; i < i1; ++i) {
            const uint32_t kk = ord_key(s[i]);
            gt += kk > thr;
            eq += kk == thr;
        }
    int o_gt = gt, o_eq = eq;
    block_scan2(o_gt, o_eq, tot);
    const int above = tot[0], need_eq = all ? 0 : k_pools - above;
    // pools above the threshold first (index order), then the first need_eq ties
    for (int i = i0; i < i1; ++i) {
        int slot = -1;
        if (all) {
            slot = o_gt++;
        } else {
            const uint32_t kk = ord_key(s[i]);
            if (kk > thr)
                slot = o_gt++;
            else if (kk == thr) {
                if (o_eq < need_eq)
                    slot = above + o_eq;
                ++o_eq;
            }
        }
        if (slot >= 0)
            for (int r = 0; r < kKpool; ++r)
                out[slot * kKpool + r] = (int32_t) (i * kKpool + r);
    }
    const int chosen = all ? n : k_pools;
    const int n_tail = (int) (q + 1 - tail_start);
    if (tid < n_tail)
        out[chosen * kKpool + tid] = (int32_t) (tail_start + tid);
    if (tid == 0)
        n_sel[t] = chosen * kKpool + n_tail;
}

// one warp per token: k rounds of a warp argmax over the selection scores (ties to the lower expert)
// One warp per token, the scores in registers (E <= 32 * kRouteMax): k rounds of a warp argmax over the biased
// scores, ties to the lower index; the owner lane drops its pick.  (Shared memory and a sync a round took 11 us.)
constexpr int kRouteMax = kMaxExperts / 32;

__global__ void __launch_bounds__(32) route_k(const float* logits, const float* bias, int E, int k, bool norm,
                                              float scale, int32_t* ids, float* w, int32_t* ids_h, float* w_h) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float sg[kRouteMax], sel[kRouteMax];
#pragma unroll
    for (int i = 0; i < kRouteMax; ++i) {
        const int e = lane + 32 * i;
        sg[i] = e < E ? sigmoidf_(logits[(size_t) t * E + e]) : 0.f;
        sel[i] = e < E ? sg[i] + bias[e] : -FLT_MAX;
    }
    float sum = 0.f, mine = 0.f;
    int mine_id = 0;
    for (int j = 0; j < k; ++j) {
        float bv = -FLT_MAX, bw = 0.f;
        int best = E;
#pragma unroll
        for (int i = 0; i < kRouteMax; ++i)
            if (sel[i] > bv) { // indices rise with i: a tie keeps the lower one
                bv = sel[i];
                bw = sg[i];
                best = lane + 32 * i;
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, bv, o), w2 = __shfl_xor_sync(0xffffffffu, bw, o);
            const int b2 = __shfl_xor_sync(0xffffffffu, best, o);
            if (v2 > bv || (v2 == bv && b2 < best)) {
                bv = v2;
                bw = w2;
                best = b2;
            }
        }
        if ((best & 31) == lane) {
#pragma unroll
            for (int i = 0; i < kRouteMax; ++i)
                if (best == lane + 32 * i)
                    sel[i] = -FLT_MAX;
        }
        sum += bw;
        if (lane == j) {
            mine = bw;
            mine_id = best;
        }
    }
    if (lane < k) {
        const float wv = (norm ? mine / sum : mine) * scale;
        ids[t * k + lane] = mine_id;
        w[t * k + lane] = wv;
        if (ids_h) { // mapped host memory, for the CPU's experts
            ids_h[t * k + lane] = mine_id;
            w_h[t * k + lane] = wv;
        }
    }
}

__global__ void swiglu_clamp_k(const float* gate, const float* up, float* y, int64_t n, float lim) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float g = fminf(gate[i], lim);
    const float u = fminf(fmaxf(up[i], -lim), lim);
    y[i] = g / (1.f + expf(-g)) * u;
}

// ids: out[0] = ids[the index] (x holds the rows of a subset, ids their token ids)
__global__ void __launch_bounds__(1024) argmax_k(const float* x, int n, int32_t* out, const int32_t* ids) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -FLT_MAX;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) {
            v = x[i];
            idx = i;
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) {
            v = v2;
            idx = i2;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = v;
        bi[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int) (blockDim.x >> 5); ++w)
            if (bv[w] > v || (bv[w] == v && bi[w] < idx)) {
                v = bv[w];
                idx = bi[w];
            }
        out[0] = ids ? ids[idx] : idx;
    }
}

__global__ void add_k(float* y, const float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += x[i];
}

} // namespace

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* st) {
    if (n >= 4096 && n % 4 == 0)
        rms_norm4_k<<<rows, 1024, 0, S(st)>>>(x, w, y, n, eps);
    else
        rms_norm_k<<<rows, 256, 0, S(st)>>>(x, w, y, n, eps);
}

void hc_pre(const float* mixes, const float* scale, const float* base, const float* R, const float* norm_w, float* post,
            float* comb, float* xn, int N, int T, int iters, float hc_eps, float rms_eps, void* st, float* xn_host) {
    hc_pre_k<<<dim3(T, one_wave() ? 2 : 1), 1024, (size_t) N * sizeof(float), S(st)>>>(
        mixes, nullptr, 0, scale, base, R, norm_w, post, comb, xn, N, iters, hc_eps, rms_eps, xn_host);
}

void hc_mix(const void* hc_fn, const float* R, float* part, int N, int T, void* st) {
    const int K = kHc * N, rows = (2 + kHc) * kHc;
    hc_mix_k<<<dim3(rows + 1, kMixSplit, T), 128, 0, S(st)>>>((const uint8_t*) hc_fn, R, part, K, rows);
}

size_t hc_mix_floats(int T) {
    return (size_t) T * ((2 + kHc) * kHc + 1) * kMixSplit;
}

void hc_pre_mixed(const float* part, const float* scale, const float* base, const float* R, const float* norm_w,
                  float* post, float* comb, float* xn, int N, int T, int iters, float hc_eps, float rms_eps, void* st,
                  float* xn_host) {
    hc_pre_k<<<dim3(T, one_wave() ? 2 : 1), 1024, (size_t) N * sizeof(float), S(st)>>>(
        nullptr, part, kHc * N, scale, base, R, norm_w, post, comb, xn, N, iters, hc_eps, rms_eps, xn_host);
}

void hc_post(const float* y, float* R, const float* post, const float* comb, int N, int T, void* st, const float* y2) {
    hc_post_k<<<dim3((N + 255) / 256, T), 256, 0, S(st)>>>(y, R, post, comb, N, y2);
}

void hc_mean_norm(const float* R, const float* w, float* out, int N, int T, float eps, void* st) {
    hc_mean_norm_k<<<T, 1024, (size_t) N * sizeof(float), S(st)>>>(R, w, out, N, eps);
}

void hc_init(const float* e, float* R, int N, int T, void* st) {
    hc_init_k<<<dim3((N + 255) / 256, T), 256, 0, S(st)>>>(e, R, N);
}

void kda_conv(const float* xq, const float* xk, const float* xv, float* state, const float* wq, const float* wk,
              const float* wv, float* out, int D, int T, float* mid, int nmid, void* st) {
    kda_conv_k<<<(3 * D + 255) / 256, 256, 0, S(st)>>>(xq, xk, xv, state, wq, wk, wv, out, D, T, mid,
                                                       mid ? std::min(nmid, T) : 0);
    kda_l2_k<<<dim3(2 * D / kKdaDim, T), kKdaDim, 0, S(st)>>>(out, D, 1e-6f);
}

void kda_gates(const float* fpre, const float* dt_b, const float* ssm_a, float* g, const float* bpre, float* beta,
               int D, int H, int T, float lower_bound, void* st) {
    kda_gates_k<<<(D * T + 255) / 256, 256, 0, S(st)>>>(fpre, dt_b, ssm_a, g, bpre, beta, D, H, T, lower_bound);
}

void kda_step(const float* S_in, float* S_out, const float* qkv, const float* g, const float* beta, float* o, int H,
              int T, float* mid, void* st) {
    kda_step_k<<<dim3(H, kKdaDim / kKdaTile), 256, 0, S(st)>>>(S_in, S_out, qkv, g, beta, o, H, T, mid);
}

void kda_out(const float* o, const float* gate, const float* norm_w, float* y, int H, int T, float eps, void* st) {
    kda_out_k<<<dim3(H, T), kKdaDim, 0, S(st)>>>(o, gate, norm_w, y, H, eps);
}

void q8_0_heads(const void* W, const float* x, float* y, int H, int n_in, int n_out, int T, void* st) {
    const int8_t* qs = (const int8_t*) W;
    const __half* d = (const __half*) ((const uint8_t*) W + (size_t) H * n_out * n_in);
    // rows a block: 256, or fewer so that a block stays in one head
    const int nw = n_out % 256 == 0 ? 8 : n_out % 128 == 0 ? 4 : n_out % 64 == 0 ? 2 : 1;
    const unsigned blocks = (unsigned) ((size_t) H * n_out / (32 * nw));
    // a token, two groups of 32 rows a warp when a head has 512 or more rows: the absorb's 64 x 512 rows were 1,024
    // warps a launch, more than a GTX 1070 holds (960), so 64 of them ran after the rest (k_b 51.4 -> 45.6 us; two
    // tokens gained nothing, 50.5 -> 51.1)
    const bool two = n_out % 512 == 0 && nw == 8 && T == 1 && one_wave();
    if (T <= 2) {
        const size_t sh = (size_t) T * n_in * 4;
        if (two && T == 1)
            q8_0_heads_k<1, 2><<<dim3(blocks / 2, 1), nw * 32, sh, S(st)>>>(qs, d, x, y, H, n_in, n_out, T);
        else if (two)
            q8_0_heads_k<2, 2><<<dim3(blocks / 2, 1), nw * 32, sh, S(st)>>>(qs, d, x, y, H, n_in, n_out, T);
        else if (T == 1)
            q8_0_heads_k<1><<<dim3(blocks, 1), nw * 32, sh, S(st)>>>(qs, d, x, y, H, n_in, n_out, T);
        else
            q8_0_heads_k<2><<<dim3(blocks, 1), nw * 32, sh, S(st)>>>(qs, d, x, y, H, n_in, n_out, T);
    } else {
        const size_t sh = (size_t) 8 * n_in * 4;
        q8_0_heads_k<8><<<dim3(blocks, (T + 7) / 8), nw * 32, sh, S(st)>>>(qs, d, x, y, H, n_in, n_out, T);
    }
}

bool q8_0_heads_supported(int n_in, int n_out) {
    return n_in % 32 == 0 && n_in >= 32 && n_in <= 1024 && n_out % 32 == 0;
}

void q8_0_split(const uint8_t* q8, uint8_t* out, int64_t rows, int n_in) {
    // values: group of 32 rows, then 16-value chunk, then the row in the group (16 bytes each); scales: group,
    // then 32-value block, then the row
    const int nb = n_in / 32, nc = n_in / 16;
    int8_t* qs = (int8_t*) out;
    uint8_t* d = out + (size_t) rows * n_in;
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t gr = r / 32, l = r % 32;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = q8 + ((size_t) r * nb + b) * 34;
            std::memcpy(d + (((size_t) gr * nb + b) * 32 + l) * 2, blk, 2);
            for (int half = 0; half < 2; ++half) {
                const int c = 2 * b + half;
                std::memcpy(qs + (((size_t) gr * nc + c) * 32 + l) * 16, blk + 2 + 16 * half, 16);
            }
        }
    }
}

bool q6k_mmvq_supported(int n_in, int n_out, int T) {
    // native_mmvq's path for four or more blocks a row; the activations of T tokens fit in shared memory beside the
    // staged rows (48 KiB a block before Volta)
    return n_in % 256 == 0 && n_in >= 1024 && n_out >= 512 && T >= 1 && T <= 2 &&
           (size_t) T * (n_in / 32) * kQ6Q81 + 8 * kQ6WarpBytes <= 48 * 1024;
}

namespace {
// q6k_rows_k's grid: the blocks the GPU holds at once (four an SM at 64 registers), at most a block per 8 rows
unsigned q6k_rows_grid(int n_out) {
    static const int resident = [] {
        int dev = 0, sms = 0, per_sm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        // the widest rows it takes (two chunks, 4096 values) and one token: the most shared memory
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q6k_rows_k<1, 8>, 256,
                                                      (size_t) (4096 / 32) * kQ6Q81 + 8 * kQ6WarpBytes);
        cudaGetLastError();
        return std::max(1, sms) * std::max(1, per_sm);
    }();
    return (unsigned) std::min(resident, (n_out + 7) / 8);
}

std::atomic<bool> g_q6k_rows{true};

bool q6k_rows_on(int n_in, int T) {
    // a token, rows of one or two chunks: 8192 or 16384 values a row (4 and 8 chunks) and two tokens measured no
    // faster this way (KDA out 134 -> 136 us, the two-column head 3335 -> 3328 us)
    return g_q6k_rows.load(std::memory_order_relaxed) && T == 1 && n_in / 256 <= 2 * kQ6Chunk;
}

} // namespace

void set_one_wave(bool on) {
    g_one_wave.store(on, std::memory_order_relaxed);
}

void set_q6k_rows(bool on) {
    g_q6k_rows.store(on, std::memory_order_relaxed);
}

void q6k_mmvq(const void* W, const void* xq8_1, float* y, int n_in, int n_out, int T, void* st) {
    const size_t xbytes = (size_t) T * (n_in / 32) * kQ6Q81;
    const auto* w = (const uint8_t*) W;
    const auto* x = (const uint8_t*) xq8_1;
    // eight rows a block: the activations are read once per block (four rows a block, which two tokens 16384 wide
    // would need, lost to native_mmvq: 117 against 146 GB/s)
    const dim3 grid((unsigned) ((n_out + 7) / 8));
    const size_t sh = xbytes + 8 * kQ6WarpBytes;
    if (q6k_rows_on(n_in, T))
        q6k_rows_k<1, 8><<<q6k_rows_grid(n_out), 256, sh, S(st)>>>(w, x, y, n_in, n_out);
    else if (T == 1)
        q6k_mmvq_k<1, 8><<<grid, 256, sh, S(st)>>>(w, x, y, n_in, n_out);
    else
        q6k_mmvq_k<2, 8><<<grid, 256, sh, S(st)>>>(w, x, y, n_in, n_out);
}

void q6k_mmvq_rows(const void* W, const int32_t* ids, int n_ids, const void* xq8_1, float* y, int n_in, void* st) {
    const size_t sh = (size_t) (n_in / 32) * kQ6Q81 + 8 * kQ6WarpBytes;
    if (q6k_rows_on(n_in, 1))
        q6k_rows_k<1, 8, true>
            <<<q6k_rows_grid(n_ids), 256, sh, S(st)>>>((const uint8_t*) W, (const uint8_t*) xq8_1, y, n_in, n_ids, ids);
    else
        q6k_mmvq_k<1, 8, true><<<(unsigned) ((n_ids + 7) / 8), 256, sh, S(st)>>>(
            (const uint8_t*) W, (const uint8_t*) xq8_1, y, n_in, n_ids, ids);
}

bool q5k_mmvq_supported(int n_in, int n_out, int T) {
    // the activations of T tokens beside eight warps' staged chunks in a block's 48 KiB (before Volta)
    return n_in % 256 == 0 && n_out >= 1 && T >= 1 && T <= 2 &&
           (size_t) T * (n_in / 32) * kQ6Q81 + 8 * kQ5WarpBytes <= 48 * 1024;
}

void q5k_mmvq(const void* W, const void* xq8_1, float* y, int n_in, int n_out, int T, void* st) {
    // as many blocks as the GPU holds at once (the widest rows' shared memory), at most a block per 8 rows
    static const int resident = [] {
        int dev = 0, sms = 0, per_sm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q5k_rows_k<1, 8>, 256,
                                                      (size_t) (16384 / 32) * kQ6Q81 + 8 * kQ5WarpBytes);
        cudaGetLastError();
        return std::max(1, sms) * std::max(1, per_sm);
    }();
    const unsigned grid = (unsigned) std::min(resident, (n_out + 7) / 8);
    const size_t sh = (size_t) T * (n_in / 32) * kQ6Q81 + 8 * kQ5WarpBytes;
    const auto* w = (const uint8_t*) W;
    const auto* x = (const uint8_t*) xq8_1;
    if (T == 1)
        q5k_rows_k<1, 8><<<grid, 256, sh, S(st)>>>(w, x, y, n_in, n_out);
    else
        q5k_rows_k<2, 8><<<grid, 256, sh, S(st)>>>(w, x, y, n_in, n_out);
}

bool q8_0_mmvq_supported(int n_in, int n_out, int T) {
    // fewer than 512 rows are too few blocks of 8 for the GPU (16384 x 24: 12.4 us against native_mmvq's 5.4)
    return n_in % 256 == 0 && n_in >= 1024 && n_out >= 512 && T >= 1 && T <= 2 &&
           (size_t) T * (n_in / 32) * kQ6Q81 + 8 * kQ8Chunk * kQ8Block <= 48 * 1024;
}

void q8_0_mmvq(const void* W, const void* xq8_1, float* y, int n_in, int n_out, int T, void* st) {
    const size_t sh = (size_t) T * (n_in / 32) * kQ6Q81 + 8 * kQ8Chunk * kQ8Block;
    const dim3 grid((unsigned) ((n_out + 7) / 8));
    const auto* w = (const uint8_t*) W;
    const auto* x = (const uint8_t*) xq8_1;
    if (T == 1)
        q8_0_mmvq_k<1, 8><<<grid, 256, sh, S(st)>>>(w, x, y, n_in, n_out);
    else
        q8_0_mmvq_k<2, 8><<<grid, 256, sh, S(st)>>>(w, x, y, n_in, n_out);
}

bool q8_0_narrow_supported(int n_in, int T) {
    return (n_in == 64 || n_in == 128 || n_in == 256) && T >= 1 && T <= 2;
}

void q8_0_narrow(const void* W, const void* xq8_1, float* y, int n_in, int n_out, int T, void* st) {
    // two chunks a warp from 4096 rows: 8192 rows of 128 values were 1,024 warps, more than a GTX 1070 holds (960)
    const int C = one_wave() && n_out >= 4096 ? 2 : 1;
    const int nb = n_in / 32, rows_per_block = 8 * C * (32 / nb);
    const dim3 grid((unsigned) ((n_out + rows_per_block - 1) / rows_per_block));
    const auto* w = (const uint8_t*) W;
    const auto* x = (const uint8_t*) xq8_1;
#define GLM5_NARROW(TT, NB)                                                                                            \
    (C == 2 ? q8_0_narrow_k<TT, NB, 2><<<grid, 256, 0, S(st)>>>(w, x, y, n_out)                                        \
            : q8_0_narrow_k<TT, NB, 1><<<grid, 256, 0, S(st)>>>(w, x, y, n_out))
    if (T == 1)
        nb == 2 ? GLM5_NARROW(1, 2) : nb == 4 ? GLM5_NARROW(1, 4) : GLM5_NARROW(1, 8);
    else
        nb == 2 ? GLM5_NARROW(2, 2) : nb == 4 ? GLM5_NARROW(2, 4) : GLM5_NARROW(2, 8);
#undef GLM5_NARROW
}

void bf16_matvec(const uint16_t* W, const float* x, float* y, int n_in, int n_out, int T, void* st) {
    const unsigned blocks = (unsigned) ((n_out + 3) / 4);
    if (T == 1 && n_in % 4 == 0)
        bf16_rows_k<1><<<blocks, 128, 0, S(st)>>>(W, x, y, n_in, n_out);
    else if (T == 2 && n_in % 4 == 0)
        bf16_rows_k<2><<<blocks, 128, 0, S(st)>>>(W, x, y, n_in, n_out);
    else
        bf16_matvec_k<<<dim3(n_out, (T + kTile - 1) / kTile), 128, 0, S(st)>>>(W, x, y, n_in, n_out, T);
}

void f32_matvec(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* st) {
    const unsigned blocks = (unsigned) ((n_out + 3) / 4);
    if (T == 1 && n_in % 4 == 0)
        f32_rows_k<1><<<blocks, 128, 0, S(st)>>>(W, x, y, n_in, n_out);
    else if (T == 2 && n_in % 4 == 0)
        f32_rows_k<2><<<blocks, 128, 0, S(st)>>>(W, x, y, n_in, n_out);
    else
        f32_matvec_k<<<dim3(n_out, (T + kTile - 1) / kTile), 128, 0, S(st)>>>(W, x, y, n_in, n_out, T);
}

// a warp a row: L FP16 values in 16-byte pieces (L % 8 == 0)
__global__ void kv_gather_k(const __half* near, int64_t n_near, const __half* far, const int32_t* sel,
                            const int32_t* n_sel, int sel_stride, __half* stage, int L) {
    const int q = blockIdx.y, i = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= n_sel[q])
        return;
    const int64_t cell = sel[(size_t) q * sel_stride + i];
    const uint4* src = (const uint4*) (cell < n_near ? near + cell * L : far + (cell - n_near) * L);
    uint4* dst = (uint4*) (stage + ((size_t) q * sel_stride + i) * L);
    for (int k = lane; k < L / 8; k += 32)
        dst[k] = src[k];
}

void kv_gather(const void* near, int64_t n_near, const void* far, const int32_t* sel, const int32_t* n_sel,
               int sel_stride, int nq, void* stage, int L, void* st) {
    kv_gather_k<<<dim3((sel_stride + 7) / 8, nq), 256, 0, S(st)>>>((const __half*) near, n_near, (const __half*) far,
                                                                   sel, n_sel, sel_stride, (__half*) stage, L);
}

__global__ void union_mark_k(const int32_t* sel, const int32_t* n_sel, int sel_stride, uint32_t* mask) {
    const int q = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_sel[q]) {
        const int c = sel[(size_t) q * sel_stride + i];
        atomicOr(&mask[c >> 5], 1u << (c & 31));
    }
}

// one block: base[w] = the marked cells before word w, base[nw] = all of them
__global__ void __launch_bounds__(1024) union_scan_k(const uint32_t* mask, int nw, int32_t* base) {
    __shared__ int tot[2];
    const int per = (nw + blockDim.x - 1) / blockDim.x, w0 = min(nw, (int) threadIdx.x * per), w1 = min(nw, w0 + per);
    int n = 0, unused = 0;
    for (int w = w0; w < w1; ++w)
        n += __popc(mask[w]);
    block_scan2(n, unused, tot);
    for (int w = w0; w < w1; ++w) {
        base[w] = n;
        n += __popc(mask[w]);
    }
    if (threadIdx.x == 0)
        base[nw] = tot[0];
}

__global__ void union_list_k(const uint32_t* mask, const int32_t* base, int nw, int32_t* list) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= nw)
        return;
    uint32_t m = mask[w];
    for (int b = base[w]; m; m &= m - 1)
        list[b++] = w * 32 + __ffs((int) m) - 1;
}

// a warp a stage row, as kv_gather_k
__global__ void union_rows_k(const __half* near, int64_t n_near, const __half* far, const int32_t* list,
                             const int32_t* count, __half* stage, int L) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= *count)
        return;
    const int64_t cell = list[r];
    const uint4* src = (const uint4*) (cell < n_near ? near + cell * L : far + (cell - n_near) * L);
    uint4* dst = (uint4*) (stage + (size_t) r * L);
    for (int k = lane; k < L / 8; k += 32)
        dst[k] = src[k];
}

__global__ void union_remap_k(const int32_t* sel, const int32_t* n_sel, int sel_stride, const uint32_t* mask,
                              const int32_t* base, int32_t* out_sel) {
    const int q = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_sel[q]) {
        const int c = sel[(size_t) q * sel_stride + i], w = c >> 5;
        out_sel[(size_t) q * sel_stride + i] = base[w] + __popc(mask[w] & ((1u << (c & 31)) - 1u));
    }
}

void kv_gather_union(const void* near, int64_t n_near, const void* far, const int32_t* sel, const int32_t* n_sel,
                     int sel_stride, int nq, int64_t n_cells, uint32_t* mask, int32_t* base, int32_t* list,
                     int32_t* out_sel, void* stage, int stage_rows, int L, void* st) {
    const int nw = (int) union_mask_words(n_cells);
    const dim3 per_sel((sel_stride + 255) / 256, nq);
    union_mark_k<<<per_sel, 256, 0, S(st)>>>(sel, n_sel, sel_stride, mask);
    union_scan_k<<<1, 1024, 0, S(st)>>>(mask, nw, base);
    union_list_k<<<(nw + 255) / 256, 256, 0, S(st)>>>(mask, base, nw, list);
    union_rows_k<<<(stage_rows + 7) / 8, 256, 0, S(st)>>>((const __half*) near, n_near, (const __half*) far, list,
                                                          base + nw, (__half*) stage, L);
    union_remap_k<<<per_sel, 256, 0, S(st)>>>(sel, n_sel, sel_stride, mask, base, out_sel);
    cudaMemsetAsync(mask, 0, (size_t) nw * 4, S(st));
}

// one block: each selected cell's slot (a hit keeps its own; a miss takes one no hit holds, in slot order), the
// misses listed for rc_rows_k
__global__ void __launch_bounds__(1024)
    rc_plan_k(const int32_t* sel, const int32_t* n_sel, int slots, int32_t* slot_cell, int32_t* cell_slot,
              int32_t* out_sel, int32_t* misses) {
    extern __shared__ uint32_t used[]; // a bit a slot
    __shared__ int tot[2];
    const int n = n_sel[0], tid = threadIdx.x, nt = blockDim.x, words = (slots + 31) / 32;
    for (int w = tid; w < words; w += nt)
        used[w] = 0;
    __syncthreads();
    // hits: each thread a contiguous range of the selection
    const int per = (n + nt - 1) / nt, i0 = min(n, tid * per), i1 = min(n, i0 + per);
    int miss = 0;
    for (int i = i0; i < i1; ++i) {
        const int c = sel[i], s = cell_slot[c];
        const bool hit = s >= 0 && s < slots && slot_cell[s] == c;
        out_sel[i] = hit ? s : -1;
        if (hit)
            atomicOr(&used[s >> 5], 1u << (s & 31));
        miss += !hit;
    }
    __syncthreads();
    // the free slots, in slot order: each thread a contiguous range of the slots
    const int sper = (slots + nt - 1) / nt, s0 = min(slots, tid * sper), s1 = min(slots, s0 + sper);
    int nfree = 0;
    for (int s = s0; s < s1; ++s)
        nfree += !((used[s >> 5] >> (s & 31)) & 1u);
    int o_miss = miss, o_free = nfree;
    block_scan2(o_miss, o_free, tot);
    // the k-th free slot to the k-th miss: the free slots first listed at misses[slots + k]
    int32_t* free_list = misses + slots;
    for (int s = s0; s < s1; ++s)
        if (!((used[s >> 5] >> (s & 31)) & 1u))
            free_list[o_free++] = s;
    __syncthreads();
    for (int i = i0; i < i1; ++i)
        if (out_sel[i] < 0) {
            const int c = sel[i], s = free_list[o_miss];
            misses[o_miss++] = c;
            out_sel[i] = s;
            slot_cell[s] = c;
            cell_slot[c] = s;
        }
    if (tid == 0)
        misses[2 * slots] = tot[0];
}

// a warp a missed row, into its slot (out_sel tells which: the misses are in selection order, so walk them again)
__global__ void rc_rows_k(const __half* near, int64_t n_near, const __half* far, const int32_t* misses,
                          const int32_t* cell_slot, int slots, __half* rows, int L) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= misses[2 * slots])
        return;
    const int64_t cell = misses[r];
    const uint4* src = (const uint4*) (cell < n_near ? near + cell * L : far + (cell - n_near) * L);
    uint4* dst = (uint4*) (rows + (size_t) cell_slot[cell] * L);
    for (int k = lane; k < L / 8; k += 32)
        dst[k] = src[k];
}

void kv_cache_rows(const void* near, int64_t n_near, const void* far, const int32_t* sel, const int32_t* n_sel,
                   int slots, void* rows, int32_t* slot_cell, int32_t* cell_slot, int32_t* out_sel, int32_t* misses,
                   int L, void* st) {
    rc_plan_k<<<1, 1024, (size_t) (slots + 31) / 32 * 4, S(st)>>>(sel, n_sel, slots, slot_cell, cell_slot, out_sel,
                                                                  misses);
    rc_rows_k<<<(slots + 7) / 8, 256, 0, S(st)>>>((const __half*) near, n_near, (const __half*) far, misses, cell_slot,
                                                  slots, (__half*) rows, L);
}

__global__ void rc_forget_k(int32_t* cell_slot, int64_t c0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        cell_slot[c0 + i] = -1;
}

void kv_cache_forget(int32_t* cell_slot, int64_t c0, int n, void* st) {
    rc_forget_k<<<(n + 255) / 256, 256, 0, S(st)>>>(cell_slot, c0, n);
}

void kv_store(const float* kv, void* cache, int64_t pos0, int L, int T, void* st) {
    kv_store_k<<<(L * T + 255) / 256, 256, 0, S(st)>>>(kv, (__half*) cache, pos0, L, T);
}

int64_t attend_scratch_floats(int64_t, int H, int L, int T) {
    int64_t most = 0;
    for (int t = 1; t <= T; ++t)
        most = std::max<int64_t>(most, (int64_t) t * splits_for(t));
    return most * H * (L + 2);
}

void mla_attend(const float* q, const void* cache, int64_t pos0, int H, int L, int T, float scale, const int32_t* sel,
                const int32_t* n_sel, int sel_stride, float* lat, float* part, void* st, int splits_of, bool legacy) {
    const int splits = splits_for(splits_of > 0 ? splits_of : T);
    if (legacy)
        mla_attend_k<<<dim3(splits, H / kHg, T), 256, 0, S(st)>>>(q, (const __half*) cache, pos0, H, L, scale, sel,
                                                                  n_sel, sel_stride, splits, part);
    else if (splits * (H / kHg) * T >= 256) // a long launch: four blocks an SM (a decode step's 128: three)
        mla_attend2_k<4><<<dim3(splits, H / kHg, T), 256, 0, S(st)>>>(q, (const __half*) cache, pos0, H, L, scale, sel,
                                                                      n_sel, sel_stride, splits, part);
    else
        mla_attend2_k<3><<<dim3(splits, H / kHg, T), 256, 0, S(st)>>>(q, (const __half*) cache, pos0, H, L, scale, sel,
                                                                      n_sel, sel_stride, splits, part);
    mla_reduce_k<<<dim3(H, T), 256, 0, S(st)>>>(part, lat, H, L, splits);
}

void idx_store(const float* kraw, const float* ln_w, const float* ln_b, const float* gate, void* kcache, void* gcache,
               int64_t pos0, int T, int ring, float eps, void* st) {
    idx_store_k<<<T, kIdxDim, 0, S(st)>>>(kraw, ln_w, ln_b, gate, (__half*) kcache, (__half*) gcache, pos0, ring, eps);
}

void idx_pool(const void* kcache, const void* gcache, int ring, const float* ape, void* pooled, int64_t pos0, int T,
              void* st) {
    // pool p is complete once position kKpool * p + kKpool - 1 is stored
    const int64_t first = pos0 / kKpool;      // the first pool whose last member (kKpool p + kKpool - 1) is >= pos0
    const int64_t last = (pos0 + T) / kKpool; // pools [first, last) end inside the window
    if (last > first)
        idx_pool_k<<<(unsigned) (last - first), kIdxDim, 0, S(st)>>>((const __half*) kcache, (const __half*) gcache,
                                                                     ring, ape, (__half*) pooled, first);
}

void idx_score(const float* iq, const float* w, const void* pooled, int64_t pos0, int T, int heads, float wscale,
               float* scores, int stride, bool legacy, void* st) {
    const int64_t n = (pos0 + T) / kKpool;
    if (n <= 0)
        return;
    if (legacy || heads > kScoreHeads)
        idx_score_k<<<dim3((unsigned) ((n + 7) / 8), T), 256, 0, S(st)>>>(iq, w, (const __half*) pooled, pos0, heads,
                                                                          wscale, scores, stride);
    else
        idx_score_pairs_k<<<dim3((unsigned) ((n + kScorePools - 1) / kScorePools), T), 2 * kScorePools, 0, S(st)>>>(
            iq, w, (const __half*) pooled, pos0, heads, wscale, scores, stride);
}

void idx_select(const float* scores, int stride, int64_t pos0, int T, int k_pools, int32_t* sel, int32_t* n_sel,
                int sel_stride, bool legacy, void* st) {
    if (legacy)
        idx_select_k<<<T, 1024, 0, S(st)>>>(scores, stride, pos0, k_pools, sel, n_sel, sel_stride);
    else
        idx_select_radix_k<<<T, 1024, 0, S(st)>>>(scores, stride, pos0, k_pools, sel, n_sel, sel_stride);
}

void route(const float* logits, const float* bias, int E, int k, bool norm, float scale, int32_t* ids, float* w, int T,
           void* st, int32_t* ids_host, float* w_host) {
    route_k<<<T, 32, 0, S(st)>>>(logits, bias, E, k, norm, scale, ids, w, ids_host, w_host);
}

void swiglu_clamp(const float* gate, const float* up, float* y, int n, int T, float lim, void* st) {
    const int64_t total = (int64_t) n * T;
    swiglu_clamp_k<<<(unsigned) ((total + 255) / 256), 256, 0, S(st)>>>(gate, up, y, total, lim);
}

void argmax(const float* x, int n, int32_t* out, void* st, const int32_t* ids) {
    argmax_k<<<1, 1024, 0, S(st)>>>(x, n, out, ids);
}

void add(float* y, const float* x, int64_t n, void* st) {
    add_k<<<(unsigned) ((n + 255) / 256), 256, 0, S(st)>>>(y, x, n);
}

} // namespace guild::core::glm5k
