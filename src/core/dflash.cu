// src/core/dflash.cu - the DFlash 2 drafter's kernels (include/guild/core/dflash.hpp).
#include "guild/core/dflash.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>

namespace guild::core::dflashk {
namespace {

inline cudaStream_t S(void* st) {
    return static_cast<cudaStream_t>(st);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// a block's sum (blockDim.x a multiple of 32, at most 1024)
__device__ float block_sum(float v) {
    __shared__ float part[32];
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0)
        part[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i)
        s += part[i];
    return s;
}

__global__ void stream_mean_k(const float* R, int hc, int n, size_t row_stride, float* out, size_t out_stride) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float* x = R + (size_t) r * row_stride;
    float s = 0.f;
    for (int c = 0; c < hc; ++c)
        s += x[(size_t) c * n + i];
    out[(size_t) r * out_stride + i] = s / (float) hc;
}

__global__ void rms_rows_k(const float* x, const float* w, float* y, int n, float eps) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += xr[i] * xr[i];
    const float rs = rsqrtf(block_sum(ss) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = w[i] * (xr[i] * rs);
}

// a warp per (row, head) of 128 values: lane l holds dims l, l + 32, l + 64, l + 96
__global__ void head_norm_rope_k(float* x, size_t stride, const float* w, int heads, float eps, const int32_t* pos,
                                 float theta) {
    const int r = blockIdx.x, h = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (h >= heads)
        return;
    float* v = x + (size_t) r * stride + (size_t) h * 128;
    float a[4];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[j] = v[lane + 32 * j];
        ss += a[j] * a[j];
    }
    const float rs = rsqrtf(warp_sum(ss) / 128.f + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        a[j] = w[lane + 32 * j] * (a[j] * rs);
    const float p = (float) pos[r];
#pragma unroll
    for (int j = 0; j < 2; ++j) { // dims i = lane + 32 j (< 64) and i + 64
        const int i = lane + 32 * j;
        const float inv = 1.0f / powf(theta, (float) (2 * i) / 128.f);
        float sn, cs;
        sincosf(p * inv, &sn, &cs);
        const float x1 = a[j], x2 = a[j + 2];
        a[j] = x1 * cs - x2 * sn;
        a[j + 2] = x2 * cs + x1 * sn;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        v[lane + 32 * j] = a[j];
}

__global__ void block_conv_k(const float* x, const float* dyn, int dyn_stride, const float* base, float* y, int n,
                             int taps, int group) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n)
        return;
    const int groups = n / group;
    float acc = 0.f;
    for (int k = 0; k < taps && k <= t; ++k) {
        const float v = x[(size_t) (t - k) * n + c];
        acc += base[(size_t) k * n + c] * v;
        acc += dyn[(size_t) t * dyn_stride + (size_t) k * groups + c / group] * v;
    }
    y[(size_t) t * n + c] = acc;
}

__global__ void add_rows_k(const float* x, float* y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += x[i];
}

__global__ void silu_mul_k(const float* gu, float* h, int f) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f)
        return;
    const float g = gu[(size_t) t * 2 * f + i], u = gu[(size_t) t * 2 * f + f + i];
    h[(size_t) t * f + i] = g / (1.f + expf(-g)) * u;
}

__global__ void ring_store_k(const float* k, const float* v, size_t stride, __half* kr, __half* vr, const int32_t* pos,
                             int kv_dim, int ring) {
    const int r = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= kv_dim)
        return;
    const size_t slot = (size_t) (pos[r] % ring);
    kr[slot * kv_dim + j] = __float2half(k[(size_t) r * stride + j]);
    vr[slot * kv_dim + j] = __float2half(v[(size_t) r * stride + j]);
}

__global__ void positions_k(int32_t* pos, int64_t pos0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        pos[i] = (int32_t) (pos0 + i);
}

// ---- block attention, split over the keys (flash-decoding): a block per (kv head, run of kKC keys) keeps the online
// softmax of the kv head's G x B queries (qi = gh * B + b, < kQ) over its run in tiles of kKT keys and writes its
// partial (max, sum, unnormalized output); attn_combine_k merges a query's runs.  Keys: the context positions ctx_lo ..
// ctx_hi - 1 (fp16 ring rows), then the block's B (converted to fp16 like them).
constexpr int kD = 128, kKT = 32, kKC = 256, kQ = 32;
constexpr int kKS = kD + 2; // a K/V tile row in halves: 65 words, so a warp's 8 key rows fall in distinct banks
constexpr int kQS = kD + 2; // a query row in halves

__device__ __forceinline__ const __half2* ring_row2(const __half* ring, int64_t p, int ring_n, int kv_dim, int kvh) {
    return reinterpret_cast<const __half2*>(ring + (size_t) (p % ring_n) * kv_dim + (size_t) kvh * kD);
}

// thread (qi = tid / 8, sub = tid % 8): the scores of keys sub + 8 j (j < 4) of a tile, and output dims
// 2 sub + 16 i + {0, 1} (i < 8); a query's 8 threads are 8 neighbouring lanes of one warp
__global__ void __launch_bounds__(256)
    attn_part_k(const float* q, size_t qs, const __half* kring, const __half* vring, const float* kb, const float* vb,
                size_t kvs, int B, int heads, int kv_heads, int ring_n, int64_t ctx_lo, int64_t ctx_hi, int64_t start,
                int window, float scale, float* part) {
    __shared__ __half sq[kQ][kQS];
    __shared__ __half sk[kKT][kKS], sv[kKT][kKS];
    __shared__ float sp[kQ][kKT + 1];
    const int kvh = blockIdx.x, run = blockIdx.y, G = heads / kv_heads, nq = G * B;
    const int nctx = (int) (ctx_hi - ctx_lo), nk = nctx + B;
    const int kv_dim = kv_heads * kD;
    const int tid = threadIdx.x, qi = tid >> 3, sub = tid & 7;
    for (int e = tid; e < kQ * (kD / 2); e += 256) {
        const int i = e / (kD / 2), d2 = e % (kD / 2);
        __half2 v = __float2half2_rn(0.f);
        if (i < nq) {
            const float* qr = q + (size_t) (i % B) * qs + (size_t) (kvh * G + i / B) * kD;
            v = __floats2half2_rn(qr[2 * d2] * scale, qr[2 * d2 + 1] * scale);
        }
        reinterpret_cast<__half2*>(&sq[i][0])[d2] = v;
    }
    const int b_q = qi < nq ? qi % B : 0;
    float m = -INFINITY, l = 0.f;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
        acc[j] = 0.f;
    const int k_begin = run * kKC, k_end = min(nk, k_begin + kKC);
    for (int t0 = k_begin; t0 < k_end; t0 += kKT) {
        __syncthreads(); // the last tile's readers are done (and sq is written, the first time)
        for (int e = tid; e < kKT * (kD / 2); e += 256) {
            const int kk = e / (kD / 2), d2 = e % (kD / 2), key = t0 + kk;
            __half2 k2 = __float2half2_rn(0.f), v2 = k2;
            if (key < nctx && key < k_end) {
                k2 = ring_row2(kring, ctx_lo + key, ring_n, kv_dim, kvh)[d2];
                v2 = ring_row2(vring, ctx_lo + key, ring_n, kv_dim, kvh)[d2];
            } else if (key < k_end) {
                const float* kr = kb + (size_t) (key - nctx) * kvs + (size_t) kvh * kD;
                const float* vr = vb + (size_t) (key - nctx) * kvs + (size_t) kvh * kD;
                k2 = __floats2half2_rn(kr[2 * d2], kr[2 * d2 + 1]);
                v2 = __floats2half2_rn(vr[2 * d2], vr[2 * d2 + 1]);
            }
            reinterpret_cast<__half2*>(&sk[kk][0])[d2] = k2;
            reinterpret_cast<__half2*>(&sv[kk][0])[d2] = v2;
        }
        __syncthreads();
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        const __half2* qrow = reinterpret_cast<const __half2*>(&sq[qi][0]);
#pragma unroll 4
        for (int d2 = 0; d2 < kD / 2; ++d2) {
            const float2 qf = __half22float2(qrow[d2]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 kf = __half22float2(reinterpret_cast<const __half2*>(&sk[sub + 8 * j][0])[d2]);
                s[j] += qf.x * kf.x + qf.y * kf.y;
            }
        }
        float tmax = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = t0 + sub + 8 * j;
            const bool ok = key < k_end && qi < nq && (key >= nctx || (start + b_q) - (ctx_lo + key) < window);
            s[j] = ok ? s[j] : -INFINITY;
            tmax = fmaxf(tmax, s[j]);
        }
        for (int o = 4; o > 0; o >>= 1)
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
        const float m_new = fmaxf(m, tmax);
        float corr = 1.f, psum = 0.f;
        if (m_new == -INFINITY) { // nothing visible yet
#pragma unroll
            for (int j = 0; j < 4; ++j)
                sp[qi][sub + 8 * j] = 0.f;
        } else {
            corr = __expf(m - m_new); // 0 while m is -inf
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = __expf(s[j] - m_new);
                sp[qi][sub + 8 * j] = p;
                psum += p;
            }
        }
        for (int o = 4; o > 0; o >>= 1)
            psum += __shfl_xor_sync(0xffffffffu, psum, o);
        l = l * corr + psum;
        m = m_new;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc[j] *= corr;
        __syncwarp(); // a query's probabilities come from its own warp
        const int nt = min(kKT, k_end - t0);
        for (int kk = 0; kk < nt; ++kk) {
            const float p = sp[qi][kk];
            const __half2* vrow = reinterpret_cast<const __half2*>(&sv[kk][0]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float2 vf = __half22float2(vrow[sub + 8 * i]);
                acc[2 * i] += p * vf.x;
                acc[2 * i + 1] += p * vf.y;
            }
        }
        __syncwarp();
    }
    if (qi >= nq)
        return;
    float* o = part + (((size_t) kvh * gridDim.y + run) * kQ + qi) * (kD + 2);
    if (sub == 0) {
        o[0] = m;
        o[1] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        o[2 + 2 * sub + 16 * i] = acc[2 * i];
        o[2 + 2 * sub + 16 * i + 1] = acc[2 * i + 1];
    }
}

// a block per (kv head, query), a thread per output dim: the query's runs merged
__global__ void attn_combine_k(const float* part, int runs, int B, int heads, int kv_heads, float* out) {
    const int kvh = blockIdx.x, qi = blockIdx.y, d = threadIdx.x, G = heads / kv_heads;
    const float* base = part + ((size_t) kvh * runs * kQ + qi) * (kD + 2);
    const size_t step = (size_t) kQ * (kD + 2);
    float M = -INFINITY;
    for (int r = 0; r < runs; ++r)
        M = fmaxf(M, base[r * step]);
    float L = 0.f, acc = 0.f;
    if (M != -INFINITY)
        for (int r = 0; r < runs; ++r) {
            const float* o = base + r * step;
            if (o[0] == -INFINITY)
                continue;
            const float w = __expf(o[0] - M);
            L += o[1] * w;
            acc += o[2 + d] * w;
        }
    out[((size_t) (qi % B) * heads + kvh * G + qi / B) * kD + d] = L > 0.f ? acc / L : 0.f;
}

// ---- top k (k <= 16) and log-sum-exp of rows of logits, in three passes: each of kTopBlocks blocks a row takes its
// part's max and sum of exp; then every logit at or above the row's 16th-largest block max - a bound no member of the
// top 16 falls below, since those maxima are 16 distinct logits - is gathered (usually a few dozen); then a block a row
// keeps the best 16 of the gathered ones and merges the parts' sums.
constexpr int kTopBlocks = 64;

// the warp's 16 best of its lanes' descending lists: 16 rounds of the best head (ties: the lower lane), which its lane
// pops; lane r < 16 ends with the r-th best in (bv, bi)
__device__ __forceinline__ void warp_top16(float (&lv)[16], int (&li)[16], float& bv, int& bi) {
    const int lane = threadIdx.x & 31;
    bv = -INFINITY;
    bi = -1;
    for (int r = 0; r < 16; ++r) {
        float v = lv[0];
        int src = lane;
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
            const int s2 = __shfl_xor_sync(0xffffffffu, src, o);
            if (v2 > v || (v2 == v && s2 < src)) {
                v = v2;
                src = s2;
            }
        }
        const int idx = __shfl_sync(0xffffffffu, li[0], src);
        if (lane == r) {
            bv = v;
            bi = idx;
        }
        if (lane == src) {
#pragma unroll
            for (int j = 0; j < 15; ++j) {
                lv[j] = lv[j + 1];
                li[j] = li[j + 1];
            }
            lv[15] = -INFINITY;
            li[15] = -1;
        }
    }
}

__device__ __forceinline__ void lse_merge(float& m, float& s, float m2, float s2) {
    const float M = fmaxf(m, m2);
    if (M == -INFINITY)
        return;
    s = (m == -INFINITY ? 0.f : s * __expf(m - M)) + (m2 == -INFINITY ? 0.f : s2 * __expf(m2 - M));
    m = M;
}

// pass 1: a block's (max, sum of exp) into part[row][blk][2]; VEC: rows of whole float4s (n a multiple of 4)
template <bool VEC>
__global__ void __launch_bounds__(256) topk_max_k(const float* logits, int n, int per, float* part) {
    __shared__ float wm[8], ws[8];
    const int row = blockIdx.y, blk = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const float* x = logits + (size_t) row * n;
    const int i0 = blk * per, i1 = min(n, i0 + per);
    // a pass for the max, then one for the sum (VEC: over the part's float4s, four at once)
    float m = -INFINITY, s = 0.f;
    if constexpr (VEC) {
        const float4* x4 = reinterpret_cast<const float4*>(x + i0);
        const int n4 = (i1 - i0) / 4;
        for (int j = threadIdx.x; j < n4; j += 256) {
            const float4 v = x4[j];
            m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
        if (m != -INFINITY)
            for (int j = threadIdx.x; j < n4; j += 256) {
                const float4 v = x4[j];
                s += __expf(v.x - m) + __expf(v.y - m) + __expf(v.z - m) + __expf(v.w - m);
            }
    } else {
        for (int i = i0 + (int) threadIdx.x; i < i1; i += 256)
            m = fmaxf(m, x[i]);
        if (m != -INFINITY)
            for (int i = i0 + (int) threadIdx.x; i < i1; i += 256)
                s += __expf(x[i] - m);
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
        lse_merge(m, s, m2, s2);
    }
    if (lane == 0) {
        wm[w] = m;
        ws[w] = s;
    }
    __syncthreads();
    if (w != 0)
        return;
    m = lane < 8 ? wm[lane] : -INFINITY;
    s = lane < 8 ? ws[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
        lse_merge(m, s, m2, s2);
    }
    if (lane == 0) {
        part[((size_t) row * gridDim.x + blk) * 2] = m;
        part[((size_t) row * gridDim.x + blk) * 2 + 1] = s;
    }
}

// pass 2: the logits at or above the row's 16th-largest block max into cand[row] (count in cnt[row], capacity n)
template <bool VEC>
__global__ void __launch_bounds__(256)
    topk_gather_k(const float* logits, int n, int per, const float* part, int* cnt, float2* cand) {
    __shared__ float thr;
    const int row = blockIdx.y, blk = blockIdx.x, nblk = gridDim.x;
    if (threadIdx.x < 32) { // the 16th largest of the row's block maxima (two a lane)
        const int lane = threadIdx.x;
        float a = lane < nblk ? part[((size_t) row * nblk + lane) * 2] : -INFINITY;
        float b = lane + 32 < nblk ? part[((size_t) row * nblk + lane + 32) * 2] : -INFINITY;
        if (b > a) {
            const float t = a;
            a = b;
            b = t;
        }
        float t16 = -INFINITY;
        for (int r = 0; r < 16; ++r) {
            float v = a;
            int src = lane;
            for (int o = 16; o > 0; o >>= 1) {
                const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
                const int s2 = __shfl_xor_sync(0xffffffffu, src, o);
                if (v2 > v || (v2 == v && s2 < src)) {
                    v = v2;
                    src = s2;
                }
            }
            t16 = v;
            if (lane == src) {
                a = b;
                b = -INFINITY;
            }
        }
        if (lane == 0)
            thr = t16;
    }
    __syncthreads();
    const float t = thr;
    const float* x = logits + (size_t) row * n;
    const int i0 = blk * per, i1 = min(n, i0 + per);
    auto keep = [&](float v, int i) {
        const int at = atomicAdd(&cnt[row], 1);
        cand[(size_t) row * n + at] = make_float2(v, __int_as_float(i));
    };
    if constexpr (VEC) {
        const float4* x4 = reinterpret_cast<const float4*>(x + i0);
        const int n4 = (i1 - i0) / 4;
        for (int j = threadIdx.x; j < n4; j += 256) {
            const float4 v = x4[j];
            if (fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)) < t)
                continue;
            const float e[4] = {v.x, v.y, v.z, v.w};
            for (int q = 0; q < 4; ++q)
                if (e[q] >= t)
                    keep(e[q], i0 + 4 * j + q);
        }
    } else {
        for (int i = i0 + (int) threadIdx.x; i < i1; i += 256)
            if (x[i] >= t)
                keep(x[i], i);
    }
}

// pass 3: a block a row: the best 16 of the gathered logits (ties: the lower index), the parts' sums merged
__global__ void __launch_bounds__(256) topk_pick_k(const float2* cand, int* cnt, const float* part, int n, int nblk,
                                                   int k, int32_t* idx, float* val, float* lse) {
    __shared__ float wv[8][16];
    __shared__ int wi[8][16];
    const int row = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int c = cnt[row];
    float lv[16];
    int li[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        lv[j] = -INFINITY;
        li[j] = -1;
    }
    for (int e = threadIdx.x; e < c; e += 256) {
        const float2 p = cand[(size_t) row * n + e];
        const int i = __float_as_int(p.y);
        // equal values: the lower index first, as a sort by value then index gives
        const float v = p.x;
        if (v > lv[15] || (v == lv[15] && i < li[15])) {
            float vv = v;
            int ii = i;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (vv > lv[j] || (vv == lv[j] && ii < li[j])) {
                    const float tv = lv[j];
                    const int ti = li[j];
                    lv[j] = vv;
                    li[j] = ii;
                    vv = tv;
                    ii = ti;
                }
        }
    }
    float bv;
    int bi;
    warp_top16(lv, li, bv, bi);
    if (lane < 16) {
        wv[w][lane] = bv;
        wi[w][lane] = bi;
    }
    __syncthreads();
    if (w != 0)
        return;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        lv[j] = lane < 8 ? wv[lane][j] : -INFINITY;
        li[j] = lane < 8 ? wi[lane][j] : -1;
    }
    warp_top16(lv, li, bv, bi);
    if (lane < k) {
        idx[row * k + lane] = bi;
        val[row * k + lane] = bv;
    }
    float m = -INFINITY, s = 0.f;
    for (int b = lane; b < nblk; b += 32)
        lse_merge(m, s, part[((size_t) row * nblk + b) * 2], part[((size_t) row * nblk + b) * 2 + 1]);
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
        lse_merge(m, s, m2, s2);
    }
    if (lane == 0) {
        lse[row] = m + logf(s);
        cnt[row] = 0; // the next call's count starts at zero
    }
}

// ---- the drafter's products: weights as int8 rows (n_in bytes) with a scale a 32 (fp16, n_in / 32 a row), the
// columns as int8 with a float scale a 32 (q8_act).  A half-warp takes R rows at a time, all C columns: lane l the
// 32-value blocks l, l + 16, ... of a chunk of the columns staged in shared memory, the rows' 32 bytes and scales
// straight from global memory, the 16 lanes' sums added at the end.
constexpr int kD8Chunk = 2560;

// the 16-byte chunks of a staged column, swizzled within each group of eight: the lanes of a load phase (the d8 and k4
// kernels read chunks 2 l + j and 4 l + j) then fall in eight different 16-byte bank groups instead of two or four
__device__ __forceinline__ int swz(int chunk) {
    return chunk ^ ((chunk >> 3) & 7);
}

__global__ void q8_act_k(const float* __restrict__ x, int K, int8_t* __restrict__ xq, float* __restrict__ xs,
                         float* __restrict__ xsum) {
    const int c = blockIdx.y, blk = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blk >= K / 32)
        return;
    const float v = x[(size_t) c * K + (size_t) blk * 32 + lane];
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const int q = d > 0.f ? __float2int_rn(v / d) : 0;
    xq[(size_t) c * K + (size_t) blk * 32 + lane] = (int8_t) q;
    const float sum = xsum ? warp_sum((float) q) : 0.f;
    if (lane == 0) {
        xs[(size_t) c * (K / 32) + blk] = d;
        if (xsum)
            xsum[(size_t) c * (K / 32) + blk] = d * sum;
    }
}

__device__ __forceinline__ int dot8(const int4& a, const int4& b, const int4& c, const int4& d) {
    int s = __dp4a(a.x, c.x, 0);
    s = __dp4a(a.y, c.y, s);
    s = __dp4a(a.z, c.z, s);
    s = __dp4a(a.w, c.w, s);
    s = __dp4a(b.x, d.x, s);
    s = __dp4a(b.y, d.y, s);
    s = __dp4a(b.z, d.z, s);
    return __dp4a(b.w, d.w, s);
}

template <int C, int R>
__global__ void __launch_bounds__(256)
    d8_mm_k(const int8_t* __restrict__ wq, const __half* __restrict__ ws, const int8_t* __restrict__ xq,
            const float* __restrict__ xs, float* __restrict__ y, int K, int n_out) {
    __shared__ __align__(16) int8_t sx[C][kD8Chunk];
    __shared__ float sxs[C][kD8Chunk / 32];
    const int hw = threadIdx.x >> 4, lane = threadIdx.x & 15, KB = K / 32;
    const int tasks = n_out / R, nchunk = (K + kD8Chunk - 1) / kD8Chunk;
    auto stage = [&](int k0, int kc) {
        __syncthreads();
        const int n16 = kc / 16;
        for (int e = threadIdx.x; e < C * n16; e += blockDim.x) {
            const int c = e / n16, i = e - c * n16;
            reinterpret_cast<int4*>(&sx[c][0])[swz(i)] = reinterpret_cast<const int4*>(xq + (size_t) c * K + k0)[i];
        }
        const int n32 = kc / 32;
        for (int e = threadIdx.x; e < C * n32; e += blockDim.x) {
            const int c = e / n32, i = e - c * n32;
            sxs[c][i] = xs[(size_t) c * KB + k0 / 32 + i];
        }
        __syncthreads();
    };
    if (nchunk == 1)
        stage(0, K);
    for (int base = blockIdx.x * 16; base < tasks; base += gridDim.x * 16) {
        const int task = base + hw;
        const bool active = task < tasks;
        float acc[R][C];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < C; ++c)
                acc[r][c] = 0.f;
        for (int ch = 0; ch < nchunk; ++ch) {
            const int k0 = ch * kD8Chunk, kc = min(kD8Chunk, K - k0);
            if (nchunk > 1)
                stage(k0, kc);
            if (!active)
                continue;
            const int8_t* wrow = wq + (size_t) task * R * K + k0;
            const __half* wsrow = ws + (size_t) task * R * KB + k0 / 32;
            for (int b = lane; b < kc / 32; b += 16) {
                int4 w0[R], w1[R];
                float wsc[R];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int4* wp = reinterpret_cast<const int4*>(wrow + (size_t) r * K + b * 32);
                    w0[r] = wp[0];
                    w1[r] = wp[1];
                    wsc[r] = __half2float(wsrow[(size_t) r * KB + b]);
                }
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const int4* xp = reinterpret_cast<const int4*>(&sx[c][0]);
                    const int4 x0 = xp[swz(2 * b)], x1 = xp[swz(2 * b + 1)];
                    const float xsc = sxs[c][b];
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        acc[r][c] += (float) dot8(w0[r], w1[r], x0, x1) * (wsc[r] * xsc);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < C; ++c) {
                float v = acc[r][c];
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                acc[r][c] = v;
            }
        if (active && lane == 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int c = 0; c < C; ++c)
                    y[(size_t) c * n_out + (size_t) task * R + r] = acc[r][c];
    }
}

template <int C, int R>
void d8_launch(const int8_t* wq, const __half* ws, const int8_t* xq, const float* xs, float* y, int K, int n_out,
               cudaStream_t st) {
    static int per_sm = 0, sms = 0;
    if (per_sm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, d8_mm_k<C, R>, 256, 0);
        per_sm = per_sm > 0 ? per_sm : 1;
    }
    const int tasks = n_out / R, blocks = std::min((tasks + 15) / 16, std::max(1, sms * per_sm));
    d8_mm_k<C, R><<<blocks, 256, 0, st>>>(wq, ws, xq, xs, y, K, n_out);
}

template <int C>
void d8_launch_r(const int8_t* wq, const __half* ws, const int8_t* xq, const float* xs, float* y, int K, int n_out,
                 cudaStream_t st) {
    // rows a task: 4 where there are enough tasks to fill the card (the 2560-row products: 6.52 -> 6.37 ms a draft
    // against 2), fewer for the small products
    if (n_out % 4 == 0 && n_out >= 2048)
        d8_launch<C, 4>(wq, ws, xq, xs, y, K, n_out, st);
    else if (n_out % 2 == 0 && n_out >= 1024)
        d8_launch<C, 2>(wq, ws, xq, xs, y, K, n_out, st);
    else
        d8_launch<C, 1>(wq, ws, xq, xs, y, K, n_out, st);
}

// ---- the head: ggml's Q4_K rows (256-value blocks of 144 bytes: d, dmin, twelve bytes of 6-bit scales and mins,
// 128 bytes of nibbles - bytes 32 p .. 32 p + 31 holding sub-blocks 2 p (low nibbles) and 2 p + 1 (high)) times C
// columns as q8_act's int8 with scale and scaled sum.  Eight lanes take R rows at a time, lane l the sub-block pairs
// l, l + 8, ... of the row; the columns are staged in shared memory once (K <= kD8Chunk).
struct K4Block {
    __half d, dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

template <int C, int R>
__global__ void __launch_bounds__(256, 2)
    k4_mm_k(const K4Block* __restrict__ w, const int8_t* __restrict__ xq, const float* __restrict__ xs,
            const float* __restrict__ xsum, float* __restrict__ y, int K, int n_out) {
    __shared__ __align__(16) int8_t sx[C][kD8Chunk];
    __shared__ float sxs[C][kD8Chunk / 32], sxm[C][kD8Chunk / 32];
    const int oct = threadIdx.x >> 3, lane = threadIdx.x & 7, KB = K / 32, NSB = K / 256, NP = K / 64;
    const int tasks = n_out / R;
    {
        const int n16 = K / 16;
        for (int e = threadIdx.x; e < C * n16; e += blockDim.x) {
            const int c = e / n16, i = e - c * n16;
            reinterpret_cast<int4*>(&sx[c][0])[swz(i)] = reinterpret_cast<const int4*>(xq + (size_t) c * K)[i];
        }
        for (int e = threadIdx.x; e < C * KB; e += blockDim.x) {
            const int c = e / KB, i = e - c * KB;
            sxs[c][i] = xs[(size_t) c * KB + i];
            sxm[c][i] = xsum[(size_t) c * KB + i];
        }
        __syncthreads();
    }
    for (int task = blockIdx.x * 32 + oct; task < tasks; task += gridDim.x * 32) {
        float acc[R][C];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < C; ++c)
                acc[r][c] = 0.f;
        for (int p = lane; p < NP; p += 8) {
            const int sb = p >> 2, pq = p & 3;
            int4 q0[R], q1[R];
            float d1[R], m1[R], d2[R], m2[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const K4Block* blk = w + (size_t) (task * R + r) * NSB + sb;
                const int4* qp = reinterpret_cast<const int4*>(blk->qs + 32 * pq);
                q0[r] = qp[0];
                q1[r] = qp[1];
                const int4 hdr = __ldg(reinterpret_cast<const int4*>(blk));
                const __half2 dd = *reinterpret_cast<const __half2*>(&hdr.x);
                const float d = __low2float(dd), dmin = __high2float(dd);
                // the 6-bit scale and min of sub-blocks 2 pq and 2 pq + 1 (k4_scale_min without its branch)
                const uint32_t w0 = (uint32_t) hdr.y, w1 = (uint32_t) hdr.z, w2 = (uint32_t) hdr.w;
                const int j0 = 2 * pq;
                auto byte_at = [&](int i) -> int { // scales[i], i < 12
                    const uint32_t wd = i < 4 ? w0 : i < 8 ? w1 : w2;
                    return (int) ((wd >> (8 * (i & 3))) & 0xFF);
                };
                int s0, mm0, s1, mm1;
                if (pq < 2) { // sub-blocks 0..3
                    s0 = byte_at(j0) & 63;
                    mm0 = byte_at(j0 + 4) & 63;
                    s1 = byte_at(j0 + 1) & 63;
                    mm1 = byte_at(j0 + 5) & 63;
                } else {
                    s0 = (byte_at(j0 + 4) & 0xF) | ((byte_at(j0 - 4) >> 6) << 4);
                    mm0 = (byte_at(j0 + 4) >> 4) | ((byte_at(j0) >> 6) << 4);
                    s1 = (byte_at(j0 + 5) & 0xF) | ((byte_at(j0 - 3) >> 6) << 4);
                    mm1 = (byte_at(j0 + 5) >> 4) | ((byte_at(j0 + 1) >> 6) << 4);
                }
                d1[r] = d * (float) s0;
                m1[r] = dmin * (float) mm0;
                d2[r] = d * (float) s1;
                m2[r] = dmin * (float) mm1;
            }
            // the nibbles of every row, once: low ones sub-block 2 pq, high ones 2 pq + 1
            int4 lo0[R], lo1[R], hi0[R], hi1[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                lo0[r] =
                    make_int4(q0[r].x & 0x0F0F0F0F, q0[r].y & 0x0F0F0F0F, q0[r].z & 0x0F0F0F0F, q0[r].w & 0x0F0F0F0F);
                lo1[r] =
                    make_int4(q1[r].x & 0x0F0F0F0F, q1[r].y & 0x0F0F0F0F, q1[r].z & 0x0F0F0F0F, q1[r].w & 0x0F0F0F0F);
                hi0[r] = make_int4((q0[r].x >> 4) & 0x0F0F0F0F, (q0[r].y >> 4) & 0x0F0F0F0F,
                                   (q0[r].z >> 4) & 0x0F0F0F0F, (q0[r].w >> 4) & 0x0F0F0F0F);
                hi1[r] = make_int4((q1[r].x >> 4) & 0x0F0F0F0F, (q1[r].y >> 4) & 0x0F0F0F0F,
                                   (q1[r].z >> 4) & 0x0F0F0F0F, (q1[r].w >> 4) & 0x0F0F0F0F);
            }
            const int xb = 2 * p; // the first sub-block's Q8 block
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const int4* xp = reinterpret_cast<const int4*>(&sx[c][0]);
                const int4 xa0 = xp[swz(2 * xb)], xa1 = xp[swz(2 * xb + 1)], xb0 = xp[swz(2 * xb + 2)],
                           xb1 = xp[swz(2 * xb + 3)];
                const float sa = sxs[c][xb], sb2 = sxs[c][xb + 1], ma = sxm[c][xb], mb = sxm[c][xb + 1];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[r][c] += (float) dot8(lo0[r], lo1[r], xa0, xa1) * (d1[r] * sa) - m1[r] * ma +
                                 (float) dot8(hi0[r], hi1[r], xb0, xb1) * (d2[r] * sb2) - m2[r] * mb;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < C; ++c) {
                float v = acc[r][c];
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                acc[r][c] = v;
            }
        if (lane == 0)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int c = 0; c < C; ++c)
                    y[(size_t) c * n_out + (size_t) task * R + r] = acc[r][c];
    }
}

template <int C, int R>
void k4_launch(const void* w, const int8_t* xq, const float* xs, const float* xsum, float* y, int K, int n_out,
               cudaStream_t st) {
    static int per_sm = 0, sms = 0;
    if (per_sm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k4_mm_k<C, R>, 256, 0);
        per_sm = per_sm > 0 ? per_sm : 1;
    }
    const int tasks = n_out / R, blocks = std::min((tasks + 31) / 32, std::max(1, sms * per_sm));
    k4_mm_k<C, R><<<blocks, 256, 0, st>>>(reinterpret_cast<const K4Block*>(w), xq, xs, xsum, y, K, n_out);
}
} // namespace

void stream_mean(const float* R, int hc, int n, int rows, size_t row_stride, float* out, size_t out_stride, void* st) {
    stream_mean_k<<<dim3((n + 255) / 256, rows), 256, 0, S(st)>>>(R, hc, n, row_stride, out, out_stride);
}
void rms_rows(const float* x, const float* w, float* y, int n, int rows, float eps, void* st) {
    rms_rows_k<<<rows, 512, 0, S(st)>>>(x, w, y, n, eps);
}
void head_norm_rope(float* x, size_t row_stride, const float* w, int rows, int heads, int d, float eps,
                    const int32_t* pos, float theta, void* st) {
    (void) d; // 128
    head_norm_rope_k<<<dim3(rows, (heads + 7) / 8), 256, 0, S(st)>>>(x, row_stride, w, heads, eps, pos, theta);
}
void positions(int32_t* pos, int64_t pos0, int n, void* st) {
    if (n > 0)
        positions_k<<<(n + 255) / 256, 256, 0, S(st)>>>(pos, pos0, n);
}
void block_conv(const float* x, const float* dyn, int dyn_stride, const float* base, float* y, int n, int rows,
                int taps, int group, void* st) {
    block_conv_k<<<dim3((n + 255) / 256, rows), 256, 0, S(st)>>>(x, dyn, dyn_stride, base, y, n, taps, group);
}
void add_rows(const float* x, float* y, int n, int rows, void* st) {
    const size_t total = (size_t) n * rows;
    add_rows_k<<<(unsigned) ((total + 255) / 256), 256, 0, S(st)>>>(x, y, total);
}
void silu_mul(const float* gu, float* h, int f, int rows, void* st) {
    silu_mul_k<<<dim3((f + 255) / 256, rows), 256, 0, S(st)>>>(gu, h, f);
}
void ring_store(const float* k, const float* v, size_t row_stride, uint16_t* kring, uint16_t* vring, const int32_t* pos,
                int rows, int kv_dim, int ring, void* st) {
    ring_store_k<<<dim3((kv_dim + 255) / 256, rows), 256, 0, S(st)>>>(
        k, v, row_stride, reinterpret_cast<__half*>(kring), reinterpret_cast<__half*>(vring), pos, kv_dim, ring);
}
size_t attn_scratch_floats(int B, int kv_heads, int ring) {
    const int runs = (ring + B + kKC - 1) / kKC;
    return (size_t) kv_heads * runs * kQ * (kD + 2);
}
void block_attention(const float* q, size_t q_stride, const uint16_t* kring, const uint16_t* vring, const float* kb,
                     const float* vb, size_t kv_stride, int B, int heads, int kv_heads, int d, int ring, int64_t ctx_lo,
                     int64_t ctx_hi, int64_t start, int window, float* scratch, float* out, void* st) {
    (void) d; // 128
    const int nk = (int) (ctx_hi - ctx_lo) + B, runs = (nk + kKC - 1) / kKC;
    attn_part_k<<<dim3(kv_heads, runs), 256, 0, S(st)>>>(
        q, q_stride, reinterpret_cast<const __half*>(kring), reinterpret_cast<const __half*>(vring), kb, vb, kv_stride,
        B, heads, kv_heads, ring, ctx_lo, ctx_hi, start, window, 1.0f / sqrtf((float) kD), scratch);
    attn_combine_k<<<dim3(kv_heads, (heads / kv_heads) * B), kD, 0, S(st)>>>(scratch, runs, B, heads, kv_heads, out);
}
size_t topk_scratch_floats(int rows, int n) {
    // the parts' (max, sum), the counts, then the gathered (value, index) pairs: room for every logit
    return (size_t) rows * kTopBlocks * 2 + 64 + (size_t) rows * n * 2;
}
void topk_lse(const float* logits, int n, int rows, int k, int32_t* idx, float* val, float* lse, float* scratch,
              void* st) {
    const bool vec = n % 4 == 0; // rows of whole float4s, 16-byte aligned
    const int per = ((n + kTopBlocks - 1) / kTopBlocks + 3) / 4 * 4;
    float* part = scratch;
    int* cnt = reinterpret_cast<int*>(scratch + (size_t) rows * kTopBlocks * 2);
    float2* cand = reinterpret_cast<float2*>(scratch + (size_t) rows * kTopBlocks * 2 + 64);
    const dim3 grid(kTopBlocks, rows);
    if (vec) {
        topk_max_k<true><<<grid, 256, 0, S(st)>>>(logits, n, per, part);
        topk_gather_k<true><<<grid, 256, 0, S(st)>>>(logits, n, per, part, cnt, cand);
    } else {
        topk_max_k<false><<<grid, 256, 0, S(st)>>>(logits, n, per, part);
        topk_gather_k<false><<<grid, 256, 0, S(st)>>>(logits, n, per, part, cnt, cand);
    }
    topk_pick_k<<<rows, 256, 0, S(st)>>>(cand, cnt, part, n, kTopBlocks, k, idx, val, lse);
}

void q8_act(const float* x, int K, int cols, int8_t* xq, float* xs, void* st, float* xsum) {
    q8_act_k<<<dim3((K / 32 + 7) / 8, cols), 256, 0, S(st)>>>(x, K, xq, xs, xsum);
}
bool k4_mm(const void* w, const int8_t* xq, const float* xs, const float* xsum, float* y, int K, int n_out, int cols,
           void* st) {
    if (K % 256 != 0 || K > kD8Chunk || n_out % 2 != 0)
        return false;
        // four rows a task where they divide (the 248320-row head at 7 columns: 6.36 against 7.23 ms a draft with two)
#define K4_CASE(CC)                                                                                                    \
    case CC:                                                                                                           \
        if (n_out % 4 == 0)                                                                                            \
            k4_launch<CC, 4>(w, xq, xs, xsum, y, K, n_out, S(st));                                                     \
        else                                                                                                           \
            k4_launch<CC, 2>(w, xq, xs, xsum, y, K, n_out, S(st));                                                     \
        break;
    switch (cols) {
        K4_CASE(1)
        K4_CASE(2)
        K4_CASE(3)
        K4_CASE(4)
        K4_CASE(5)
        K4_CASE(6)
        K4_CASE(7)
        K4_CASE(8)
    default: return false;
    }
#undef K4_CASE
    return true;
}
bool d8_mm(const int8_t* wq, const uint16_t* ws, const int8_t* xq, const float* xs, float* y, int K, int n_out,
           int cols, void* st) {
    const __half* w = reinterpret_cast<const __half*>(ws);
    switch (cols) {
    case 1: d8_launch_r<1>(wq, w, xq, xs, y, K, n_out, S(st)); break;
    case 2: d8_launch_r<2>(wq, w, xq, xs, y, K, n_out, S(st)); break;
    case 3: d8_launch_r<3>(wq, w, xq, xs, y, K, n_out, S(st)); break;
    case 4: d8_launch_r<4>(wq, w, xq, xs, y, K, n_out, S(st)); break;
    case 5: d8_launch_r<5>(wq, w, xq, xs, y, K, n_out, S(st)); break;
    case 6: d8_launch_r<6>(wq, w, xq, xs, y, K, n_out, S(st)); break;
    case 7: d8_launch_r<7>(wq, w, xq, xs, y, K, n_out, S(st)); break;
    case 8: d8_launch_r<8>(wq, w, xq, xs, y, K, n_out, S(st)); break;
    default: return false;
    }
    return true;
}

} // namespace guild::core::dflashk
