// src/core/glm5.cu - GLM-5.3-Flash's own GPU kernels (see include/guild/core/glm5_kernels.hpp).  The arithmetic
// follows llama.cpp's glm5next graph (src/models/glm5next.cpp, deepseek4.cpp's mHC and delta-net-base.cpp's
// autoregressive KDA step) operation for operation, in FP32.
#include "guild/core/glm5_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace guild::core::glm5k {
namespace {

inline cudaStream_t S(void* st) {
    return (cudaStream_t) st;
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

// one block per token: the 4x4 Sinkhorn mix in shared memory, then the pre-weighted stream sum and its RMS norm
__global__ void hc_pre_k(const float* mixes, const float* scale, const float* base, const float* R, const float* nw,
                         float* post, float* comb, float* xn, int N, int iters, float hc_eps, float rms_eps) {
    extern __shared__ float xs[]; // N floats
    __shared__ float pre_s[kHc], c[kHc * kHc];
    const int t = blockIdx.x;
    if (threadIdx.x == 0) {
        const float* m = mixes + (size_t) t * (2 + kHc) * kHc;
        for (int h = 0; h < kHc; ++h) {
            pre_s[h] = sigmoidf_(m[h] * scale[0] + base[h]) + hc_eps;
            post[t * kHc + h] = 2.f * sigmoidf_(m[kHc + h] * scale[1] + base[kHc + h]);
        }
        // c[dst + 4 src]: softmax over dst for every src, + eps, one column pass, then rows/columns
        for (int src = 0; src < kHc; ++src) {
            float mx = -FLT_MAX;
            for (int dst = 0; dst < kHc; ++dst) {
                const int i = dst + kHc * src;
                c[i] = m[2 * kHc + i] * scale[2] + base[2 * kHc + i];
                mx = fmaxf(mx, c[i]);
            }
            float s = 0.f;
            for (int dst = 0; dst < kHc; ++dst) {
                c[dst + kHc * src] = expf(c[dst + kHc * src] - mx);
                s += c[dst + kHc * src];
            }
            for (int dst = 0; dst < kHc; ++dst)
                c[dst + kHc * src] = c[dst + kHc * src] / s + hc_eps;
        }
        auto norm_cols = [&]() { // for each dst: divide by the sum over src
            for (int dst = 0; dst < kHc; ++dst) {
                float s = 0.f;
                for (int src = 0; src < kHc; ++src)
                    s += c[dst + kHc * src];
                s += hc_eps;
                for (int src = 0; src < kHc; ++src)
                    c[dst + kHc * src] /= s;
            }
        };
        auto norm_rows = [&]() { // for each src: divide by the sum over dst
            for (int src = 0; src < kHc; ++src) {
                float s = 0.f;
                for (int dst = 0; dst < kHc; ++dst)
                    s += c[dst + kHc * src];
                s += hc_eps;
                for (int dst = 0; dst < kHc; ++dst)
                    c[dst + kHc * src] /= s;
            }
        };
        norm_cols();
        for (int i = 1; i < iters; ++i) {
            norm_rows();
            norm_cols();
        }
        for (int i = 0; i < kHc * kHc; ++i)
            comb[t * kHc * kHc + i] = c[i];
    }
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
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        xn[(size_t) t * N + i] = xs[i] * inv * nw[i];
}

__global__ void hc_post_k(const float* y, float* R, const float* post, const float* comb, int N) {
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
    const float yi = y[(size_t) t * N + i];
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
                           const float* wk, const float* wv, float* out, int D, int T, float* mid) {
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
        if (t == 0 && mid) {
            mid[(size_t) c * 3] = s0;
            mid[(size_t) c * 3 + 1] = s1;
            mid[(size_t) c * 3 + 2] = s2;
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

// One block per head; thread j keeps the value column S[:, j] in registers for the whole window.
__global__ void __launch_bounds__(kKdaDim)
    kda_step_k(float* S, const float* qkv, const float* g, const float* beta, float* o, int H, int T, float* mid) {
    constexpr int DK = kKdaDim;
    const int h = blockIdx.x, j = threadIdx.x;
    const int D = H * DK;
    __shared__ float sq[DK], sk[DK], sg[DK];
    float* Sh = S + (size_t) h * DK * DK;
    float col[DK];
#pragma unroll
    for (int i = 0; i < DK; ++i)
        col[i] = Sh[i * DK + j];
    const float qscale = rsqrtf((float) DK);
    for (int t = 0; t < T; ++t) {
        const float* row = qkv + (size_t) t * 3 * D;
        __syncthreads();
        sq[j] = row[h * DK + j] * qscale;
        sk[j] = row[D + h * DK + j];
        sg[j] = expf(g[(size_t) t * D + h * DK + j]);
        __syncthreads();
        const float v = row[2 * D + h * DK + j];
        const float b = beta[t * H + h];
        float ks = 0.f;
#pragma unroll
        for (int i = 0; i < DK; ++i) {
            col[i] *= sg[i];
            ks += col[i] * sk[i];
        }
        const float d = (v - ks) * b;
        float out = 0.f;
#pragma unroll
        for (int i = 0; i < DK; ++i) {
            col[i] += sk[i] * d;
            out += col[i] * sq[i];
        }
        o[(size_t) t * D + h * DK + j] = out;
        if (t == 0 && mid) {
            float* Mh = mid + (size_t) h * DK * DK;
#pragma unroll
            for (int i = 0; i < DK; ++i)
                Mh[i * DK + j] = col[i];
        }
    }
#pragma unroll
    for (int i = 0; i < DK; ++i)
        Sh[i * DK + j] = col[i];
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

// one warp per (head, row) and tile of kTile tokens (grid.y)
constexpr int kTile = 8;

__global__ void q8_0_heads_k(const uint8_t* W, const float* x, float* y, int H, int n_in, int n_out, int T) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= H * n_out)
        return;
    const int t0 = blockIdx.y * kTile;
    x += (size_t) t0 * H * n_in;
    y += (size_t) t0 * H * n_out;
    T = min(kTile, T - t0);
    const int h = warp / n_out;
    const int nb = n_in / 32;
    const uint8_t* row = W + (size_t) warp * nb * 34;
    float acc[kTile];
#pragma unroll
    for (int t = 0; t < kTile; ++t)
        acc[t] = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        const float w = q8_0_scale(blk) * (float) (int8_t) blk[2 + lane];
        const int k = b * 32 + lane;
#pragma unroll
        for (int t = 0; t < kTile; ++t)
            if (t < T)
                acc[t] += w * x[((size_t) t * H + h) * n_in + k];
    }
#pragma unroll
    for (int t = 0; t < kTile; ++t) {
        if (t >= T)
            break;
        const float s = warp_sum(acc[t]);
        if (lane == 0)
            y[((size_t) t * H + h) * n_out + (warp - h * n_out)] = s;
    }
}

__global__ void f32_matvec_k(const float* W, const float* x, float* y, int n_in, int n_out, int T) {
    const int r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (r >= n_out)
        return;
    const int t0 = blockIdx.y * kTile;
    x += (size_t) t0 * n_in;
    y += (size_t) t0 * n_out;
    T = min(kTile, T - t0);
    const float* wr = W + (size_t) r * n_in;
    float acc[kTile];
#pragma unroll
    for (int t = 0; t < kTile; ++t)
        acc[t] = 0.f;
    for (int k = lane; k < n_in; k += 32) {
        const float w = wr[k];
#pragma unroll
        for (int t = 0; t < kTile; ++t)
            if (t < T)
                acc[t] += w * x[(size_t) t * n_in + k];
    }
#pragma unroll
    for (int t = 0; t < kTile; ++t) {
        if (t >= T)
            break;
        const float s = warp_sum(acc[t]);
        if (lane == 0)
            y[(size_t) t * n_out + r] = s;
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

// one block (128 threads) per token: LayerNorm of the key, then both rows to FP16
__global__ void idx_store_k(const float* kraw, const float* ln_w, const float* ln_b, const float* gate, __half* kc,
                            __half* gc, int64_t pos0, float eps) {
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
    const size_t row = (size_t) (pos0 + t) * kIdxDim + d;
    kc[row] = __float2half(c * rsqrtf(var + eps) * ln_w[d] + ln_b[d]);
    gc[row] = __float2half(gate[(size_t) t * kIdxDim + d]);
}

// one block (128 threads, one per dim) per pool completed in the window
__global__ void idx_pool_k(const __half* kc, const __half* gc, const float* ape, float* pooled, int64_t first_pool) {
    const int64_t p = first_pool + blockIdx.x;
    const int d = threadIdx.x;
    float g[kKpool], mx = -FLT_MAX;
#pragma unroll
    for (int s = 0; s < kKpool; ++s) {
        g[s] = __half2float(gc[(size_t) (p * kKpool + s) * kIdxDim + d]) + ape[s * kIdxDim + d];
        mx = fmaxf(mx, g[s]);
    }
    float den = 0.f, v = 0.f;
#pragma unroll
    for (int s = 0; s < kKpool; ++s) {
        const float e = expf(g[s] - mx);
        den += e;
        v += e * __half2float(kc[(size_t) (p * kKpool + s) * kIdxDim + d]);
    }
    // rounded to FP16 as the reference's indexer cache keeps it (its scores are taken against the rounded key)
    pooled[(size_t) p * kIdxDim + d] = __half2float(__float2half(v / den));
}

// block (8 warps) per 8 pools x token: warp w scores pool 8 * blockIdx.x + w over all heads
__global__ void idx_score_k(const float* iq, const float* w, const float* pooled, int64_t pos0, int heads, float wscale,
                            float* scores, int stride) {
    const int t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t p = (int64_t) blockIdx.x * 8 + warp;
    const int64_t n_vis = (pos0 + t + 1) / kKpool;
    if (p >= n_vis)
        return;
    float k4[kIdxDim / 32];
#pragma unroll
    for (int i = 0; i < kIdxDim / 32; ++i)
        k4[i] = pooled[(size_t) p * kIdxDim + lane + 32 * i];
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

__device__ __forceinline__ uint32_t ord_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block (1024 threads) per token: radix-free top-k by a binary search for the k-th largest key, then a
// deterministic, index-ordered compaction (keys above the threshold, then ties from the lowest pool up)
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

__global__ void route_k(const float* logits, const float* bias, int E, int k, bool norm, float scale, int32_t* ids,
                        float* w) {
    extern __shared__ float sc[]; // 2E: the sigmoid scores, then the selection scores
    const int t = blockIdx.x;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float s = sigmoidf_(logits[(size_t) t * E + e]);
        sc[e] = s;
        sc[E + e] = s + bias[e];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int j = 0; j < k; ++j) {
            int best = 0;
            float bv = -FLT_MAX;
            for (int e = 0; e < E; ++e)
                if (sc[E + e] > bv) {
                    bv = sc[E + e];
                    best = e;
                }
            sc[E + best] = -FLT_MAX;
            ids[t * k + j] = best;
            w[t * k + j] = sc[best];
            sum += sc[best];
        }
        for (int j = 0; j < k; ++j)
            w[t * k + j] = (norm ? w[t * k + j] / sum : w[t * k + j]) * scale;
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

__global__ void __launch_bounds__(1024) argmax_k(const float* x, int n, int32_t* out) {
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
        out[0] = idx;
    }
}

__global__ void add_k(float* y, const float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += x[i];
}

} // namespace

void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* st) {
    rms_norm_k<<<rows, 256, 0, S(st)>>>(x, w, y, n, eps);
}

void hc_pre(const float* mixes, const float* scale, const float* base, const float* R, const float* norm_w, float* post,
            float* comb, float* xn, int N, int T, int iters, float hc_eps, float rms_eps, void* st) {
    hc_pre_k<<<T, 1024, (size_t) N * sizeof(float), S(st)>>>(mixes, scale, base, R, norm_w, post, comb, xn, N, iters,
                                                             hc_eps, rms_eps);
}

void hc_post(const float* y, float* R, const float* post, const float* comb, int N, int T, void* st) {
    hc_post_k<<<dim3((N + 255) / 256, T), 256, 0, S(st)>>>(y, R, post, comb, N);
}

void hc_mean_norm(const float* R, const float* w, float* out, int N, int T, float eps, void* st) {
    hc_mean_norm_k<<<T, 1024, (size_t) N * sizeof(float), S(st)>>>(R, w, out, N, eps);
}

void hc_init(const float* e, float* R, int N, int T, void* st) {
    hc_init_k<<<dim3((N + 255) / 256, T), 256, 0, S(st)>>>(e, R, N);
}

void kda_conv(const float* xq, const float* xk, const float* xv, float* state, const float* wq, const float* wk,
              const float* wv, float* out, int D, int T, float* mid, void* st) {
    kda_conv_k<<<(3 * D + 255) / 256, 256, 0, S(st)>>>(xq, xk, xv, state, wq, wk, wv, out, D, T, mid);
    kda_l2_k<<<dim3(2 * D / kKdaDim, T), kKdaDim, 0, S(st)>>>(out, D, 1e-6f);
}

void kda_gates(const float* fpre, const float* dt_b, const float* ssm_a, float* g, const float* bpre, float* beta,
               int D, int H, int T, float lower_bound, void* st) {
    kda_gates_k<<<(D * T + 255) / 256, 256, 0, S(st)>>>(fpre, dt_b, ssm_a, g, bpre, beta, D, H, T, lower_bound);
}

void kda_step(float* S_, const float* qkv, const float* g, const float* beta, float* o, int H, int T, float* mid,
              void* st) {
    kda_step_k<<<H, kKdaDim, 0, S(st)>>>(S_, qkv, g, beta, o, H, T, mid);
}

void kda_out(const float* o, const float* gate, const float* norm_w, float* y, int H, int T, float eps, void* st) {
    kda_out_k<<<dim3(H, T), kKdaDim, 0, S(st)>>>(o, gate, norm_w, y, H, eps);
}

void q8_0_heads(const void* W, const float* x, float* y, int H, int n_in, int n_out, int T, void* st) {
    const int warps = H * n_out;
    q8_0_heads_k<<<dim3((warps + 7) / 8, (T + kTile - 1) / kTile), 256, 0, S(st)>>>((const uint8_t*) W, x, y, H, n_in,
                                                                                    n_out, T);
}

void f32_matvec(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* st) {
    f32_matvec_k<<<dim3((n_out + 7) / 8, (T + kTile - 1) / kTile), 256, 0, S(st)>>>(W, x, y, n_in, n_out, T);
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
                const int32_t* n_sel, int sel_stride, float* lat, float* part, void* st) {
    const int splits = splits_for(T);
    mla_attend_k<<<dim3(splits, H / kHg, T), 256, 0, S(st)>>>(q, (const __half*) cache, pos0, H, L, scale, sel, n_sel,
                                                              sel_stride, splits, part);
    mla_reduce_k<<<dim3(H, T), 256, 0, S(st)>>>(part, lat, H, L, splits);
}

void idx_store(const float* kraw, const float* ln_w, const float* ln_b, const float* gate, void* kcache, void* gcache,
               int64_t pos0, int T, float eps, void* st) {
    idx_store_k<<<T, kIdxDim, 0, S(st)>>>(kraw, ln_w, ln_b, gate, (__half*) kcache, (__half*) gcache, pos0, eps);
}

void idx_pool(const void* kcache, const void* gcache, const float* ape, float* pooled, int64_t pos0, int T, void* st) {
    // pool p is complete once position kKpool * p + kKpool - 1 is stored
    const int64_t first = pos0 / kKpool;      // the first pool whose last member (kKpool p + kKpool - 1) is >= pos0
    const int64_t last = (pos0 + T) / kKpool; // pools [first, last) end inside the window
    if (last > first)
        idx_pool_k<<<(unsigned) (last - first), kIdxDim, 0, S(st)>>>((const __half*) kcache, (const __half*) gcache,
                                                                     ape, pooled, first);
}

void idx_score(const float* iq, const float* w, const float* pooled, int64_t pos0, int T, int heads, float wscale,
               float* scores, int stride, void* st) {
    const int64_t n = (pos0 + T) / kKpool;
    if (n > 0)
        idx_score_k<<<dim3((unsigned) ((n + 7) / 8), T), 256, 0, S(st)>>>(iq, w, pooled, pos0, heads, wscale, scores,
                                                                          stride);
}

void idx_select(const float* scores, int stride, int64_t pos0, int T, int k_pools, int32_t* sel, int32_t* n_sel,
                int sel_stride, void* st) {
    idx_select_k<<<T, 1024, 0, S(st)>>>(scores, stride, pos0, k_pools, sel, n_sel, sel_stride);
}

void route(const float* logits, const float* bias, int E, int k, bool norm, float scale, int32_t* ids, float* w, int T,
           void* st) {
    route_k<<<T, 256, (size_t) 2 * E * sizeof(float), S(st)>>>(logits, bias, E, k, norm, scale, ids, w);
}

void swiglu_clamp(const float* gate, const float* up, float* y, int n, int T, float lim, void* st) {
    const int64_t total = (int64_t) n * T;
    swiglu_clamp_k<<<(unsigned) ((total + 255) / 256), 256, 0, S(st)>>>(gate, up, y, total, lim);
}

void argmax(const float* x, int n, int32_t* out, void* st) {
    argmax_k<<<1, 1024, 0, S(st)>>>(x, n, out);
}

void add(float* y, const float* x, int64_t n, void* st) {
    add_k<<<(unsigned) ((n + 255) / 256), 256, 0, S(st)>>>(y, x, n);
}

} // namespace guild::core::glm5k
