// src/core/drama3_tower.cu - the kernels of Drama 3's codebook tower (include/guild/core/drama3_tower.hpp).
#include "guild/core/drama3_tower_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace guild::core::drama3 {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
    return __uint_as_float((uint32_t) v << 16);
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0)
        red[warp] = v;
    __syncthreads();
    float s = 0.0f;
    for (int w = 0; w < nw; ++w)
        s += red[w];
    return s;
}

// out = x * rsqrt(mean(x^2) + eps) * w, one row of n per block (rows at strides xs / os)
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                int n, float eps, int xs, int os) {
    __shared__ float red[32];
    x += (size_t) blockIdx.x * xs;
    out += (size_t) blockIdx.x * os;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += x[i] * x[i];
    const float inv = rsqrtf(block_sum(ss, red) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[i] = x[i] * inv * w[i];
}

// out = row `row` of a BF16 table (zeros when row < 0), plus, when `n_add` > 0, the sum of the rows in `add_rows`
__global__ void embed_kernel(const uint16_t* __restrict__ table, int row, EmbedAdd add,
                             const uint16_t* __restrict__ add_table, float* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = row >= 0 ? bf16_to_f32(table[(size_t) row * n + i]) : 0.0f;
        if (add.n > 0) {
            float s = 0.0f;
            for (int k = 0; k < add.n; ++k)
                s += bf16_to_f32(add_table[(size_t) add.rows[k] * n + i]);
            v += s;
        }
        out[i] = v;
    }
}

// row r of out = table row rows[r] (zeros when < 0) + the sum of add_table's rows add[r * 9 .. + add_n[r]]
__global__ void embed_rows_kernel(const uint16_t* __restrict__ table, const int32_t* __restrict__ rows,
                                  const int32_t* __restrict__ add, const int32_t* __restrict__ add_n,
                                  const uint16_t* __restrict__ add_table, float* __restrict__ out, int n) {
    const int r = blockIdx.y;
    const int row = rows[r], na = add_n[r];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = row >= 0 ? bf16_to_f32(table[(size_t) row * n + i]) : 0.0f;
        if (na > 0) {
            float sum = 0.0f;
            for (int k = 0; k < na; ++k)
                sum += bf16_to_f32(add_table[(size_t) add[r * 9 + k] * n + i]);
            v += sum;
        }
        out[(size_t) r * n + i] = v;
    }
}

__global__ void add_kernel(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += y[i];
}

// Blocks 0..n_head-1: query head b; n_head..n_head+n_kv-1: key head; the next n_kv: value head.  q/k are
// RMS-normed over the head (scale w, the multiplier 1 + weight), then rotated pairwise on the first rope_dim channels:
// (x[2i], x[2i+1]) by pos * base^(-2i / rope_dim).
__global__ void qk_norm_rope_kernel(const float* __restrict__ qkv, const float* __restrict__ q_w,
                                    const float* __restrict__ k_w, float* __restrict__ q_out,
                                    __half* __restrict__ k_cache, __half* __restrict__ v_cache, QkvShape s,
                                    int64_t pos) {
    __shared__ float red[32];
    __shared__ float buf[512];
    const int b = blockIdx.x, d = threadIdx.x, hd = s.head_dim;
    // row blockIdx.y of a batch: its own qkv row, q row and position
    qkv += (size_t) blockIdx.y * (2 * s.n_head + 2 * s.n_kv) * hd;
    q_out += (size_t) blockIdx.y * s.n_head * hd;
    pos += blockIdx.y;
    const int q_width = s.n_head * hd;
    const bool is_q = b < s.n_head, is_k = !is_q && b < s.n_head + s.n_kv;
    const float* src;
    if (is_q)
        src = qkv + (size_t) b * hd;
    else if (is_k)
        src = qkv + 2 * (size_t) q_width + (size_t) (b - s.n_head) * hd;
    else
        src = qkv + 2 * (size_t) q_width + (size_t) s.n_kv * hd + (size_t) (b - s.n_head - s.n_kv) * hd;
    const float x = d < hd ? src[d] : 0.0f;
    if (!is_q && !is_k) {
        if (d < hd)
            v_cache[((size_t) pos * s.n_kv + (b - s.n_head - s.n_kv)) * hd + d] = __float2half(x);
        return;
    }
    const float inv = rsqrtf(block_sum(x * x, red) / (float) hd + s.eps);
    const float y = d < hd ? x * inv * (is_q ? q_w[d] : k_w[d]) : 0.0f;
    buf[d] = y;
    __syncthreads();
    float r = y;
    if (d < s.rope_dim) {
        const int i = d >> 1;
        const float ang = (float) pos * powf(s.rope_base, -(float) (2 * i) / (float) s.rope_dim);
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float even = buf[2 * i], odd = buf[2 * i + 1];
        r = (d & 1) ? odd * cs + even * sn : even * cs - odd * sn;
    }
    if (d >= hd)
        return;
    if (is_q)
        q_out[(size_t) b * hd + d] = r;
    else
        k_cache[((size_t) pos * s.n_kv + (b - s.n_head)) * hd + d] = __float2half(r);
}

// Split attention, one block per (K/V head, chunk of kChunk positions), for the G = n_head / n_kv query heads of
// that K/V head.  Writes per (kv, chunk, g): the chunk's max score, its sum of exp, and the unnormalized output.
constexpr int kChunk = 64;
constexpr int kMaxG = 8;
constexpr int kMaxHd = 256;
__global__ void attn_partial_kernel(const float* __restrict__ q, const __half* __restrict__ k_cache,
                                    const __half* __restrict__ v_cache, float* __restrict__ part, QkvShape s,
                                    int64_t n_pos) {
    __shared__ float sq[kMaxG][kMaxHd];
    __shared__ float sc[kMaxG][kChunk];
    __shared__ float ml[kMaxG][2];
    const int kv = blockIdx.x, chunk = blockIdx.y, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int hd = s.head_dim, G = s.n_head / s.n_kv;
    const float scale = rsqrtf((float) hd);
    for (int i = t; i < G * hd; i += blockDim.x)
        sq[i / hd][i % hd] = q[(size_t) (kv * G + i / hd) * hd + i % hd];
    __syncthreads();
    const int64_t p0 = (int64_t) chunk * kChunk;
    for (int p = warp; p < kChunk; p += blockDim.x >> 5) {
        const int64_t pos = p0 + p;
        float acc[kMaxG];
#pragma unroll
        for (int g = 0; g < kMaxG; ++g)
            acc[g] = 0.0f;
        if (pos < n_pos) {
            const __half* kr = k_cache + ((size_t) pos * s.n_kv + kv) * hd;
            for (int dd = lane; dd < hd; dd += 32) {
                const float kd = __half2float(kr[dd]);
#pragma unroll
                for (int g = 0; g < kMaxG; ++g)
                    if (g < G)
                        acc[g] += kd * sq[g][dd];
            }
        }
#pragma unroll
        for (int g = 0; g < kMaxG; ++g) {
            float v = acc[g];
            for (int o = 16; o > 0; o >>= 1)
                v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && g < G)
                sc[g][p] = pos < n_pos ? v * scale : -INFINITY;
        }
    }
    __syncthreads();
    if (t < G) {
        float m = -INFINITY;
        for (int p = 0; p < kChunk; ++p)
            m = fmaxf(m, sc[t][p]);
        float l = 0.0f;
        for (int p = 0; p < kChunk; ++p) {
            const float e = sc[t][p] == -INFINITY ? 0.0f : __expf(sc[t][p] - m);
            sc[t][p] = e;
            l += e;
        }
        ml[t][0] = m;
        ml[t][1] = l;
    }
    __syncthreads();
    float* out = part + ((size_t) kv * gridDim.y + chunk) * (size_t) G * (hd + 2);
    if (t < G) {
        out[t * (hd + 2)] = ml[t][0];
        out[t * (hd + 2) + 1] = ml[t][1];
    }
    for (int dd = t; dd < hd; dd += blockDim.x) {
        float acc[kMaxG];
#pragma unroll
        for (int g = 0; g < kMaxG; ++g)
            acc[g] = 0.0f;
        for (int p = 0; p < kChunk && p0 + p < n_pos; ++p) {
            const float vd = __half2float(v_cache[((size_t) (p0 + p) * s.n_kv + kv) * hd + dd]);
#pragma unroll
            for (int g = 0; g < kMaxG; ++g)
                if (g < G)
                    acc[g] += sc[g][p] * vd;
        }
        for (int g = 0; g < G; ++g)
            out[g * (hd + 2) + 2 + dd] = acc[g];
    }
}

// One block per query head: combine the chunks, then out = attention * sigmoid(gate)
__global__ void attn_combine_kernel(const float* __restrict__ part, const float* __restrict__ gate,
                                    float* __restrict__ out, QkvShape s, int n_chunks) {
    const int h = blockIdx.x, hd = s.head_dim, G = s.n_head / s.n_kv, kv = h / G, g = h % G;
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c)
        M = fmaxf(M, part[((size_t) kv * n_chunks + c) * (size_t) G * (hd + 2) + g * (hd + 2)]);
    for (int dd = threadIdx.x; dd < hd; dd += blockDim.x) {
        float L = 0.0f, acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const float* pc = part + ((size_t) kv * n_chunks + c) * (size_t) G * (hd + 2) + g * (hd + 2);
            const float w = pc[0] == -INFINITY ? 0.0f : __expf(pc[0] - M);
            L += pc[1] * w;
            acc += pc[2 + dd] * w;
        }
        const float gt = gate[(size_t) h * hd + dd];
        out[(size_t) h * hd + dd] = (acc / L) * (1.0f / (1.0f + __expf(-gt)));
    }
}

__global__ void backbone_embed_add_kernel(float* __restrict__ emb, const int32_t* __restrict__ tok,
                                          const int32_t* __restrict__ code1, const uint16_t* __restrict__ rows,
                                          int32_t lo, int32_t hi, int n) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t id = tok[t];
    if (i >= n || id < lo || id > hi)
        return;
    const float sqrt2 = 1.41421356237309515f;
    emb[(size_t) t * n + i] = (emb[(size_t) t * n + i] + bf16_to_f32(rows[(size_t) code1[t] * n + i])) / sqrt2;
}

} // namespace

void backbone_embed_add(float* emb, const int32_t* tok, const int32_t* code1, const uint16_t* rows, int32_t lo,
                        int32_t hi, int n, int T, cudaStream_t st) {
    backbone_embed_add_kernel<<<dim3((n + kThreads - 1) / kThreads, T), kThreads, 0, st>>>(emb, tok, code1, rows, lo,
                                                                                           hi, n);
}

void rms_norm(const float* x, const float* w, float* out, int n, float eps, cudaStream_t st, int rows, int xs, int os) {
    rms_norm_kernel<<<rows, kThreads, 0, st>>>(x, w, out, n, eps, xs ? xs : n, os ? os : n);
}

void embed_rows(const uint16_t* table, const int32_t* rows, const int32_t* add, const int32_t* add_n,
                const uint16_t* add_table, float* out, int n, int n_rows, cudaStream_t st) {
    embed_rows_kernel<<<dim3((n + kThreads - 1) / kThreads, n_rows), kThreads, 0, st>>>(table, rows, add, add_n,
                                                                                        add_table, out, n);
}

void embed(const uint16_t* table, int row, const EmbedAdd& add, const uint16_t* add_table, float* out, int n,
           cudaStream_t st) {
    embed_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, st>>>(table, row, add, add_table, out, n);
}

void add(float* x, const float* y, int n, cudaStream_t st) {
    add_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, st>>>(x, y, n);
}

void qk_norm_rope(const float* qkv, const float* q_w, const float* k_w, float* q_out, uint16_t* k_cache,
                  uint16_t* v_cache, const QkvShape& s, int64_t pos, cudaStream_t st, int rows) {
    qk_norm_rope_kernel<<<dim3(s.n_head + 2 * s.n_kv, rows), s.head_dim, 0, st>>>(
        qkv, q_w, k_w, q_out, (__half*) k_cache, (__half*) v_cache, s, pos);
}

int attn_chunks(int64_t n_pos) {
    return (int) ((n_pos + kChunk - 1) / kChunk);
}

size_t attn_part_floats(const QkvShape& s, int64_t max_pos) {
    return (size_t) s.n_kv * attn_chunks(max_pos) * (size_t) (s.n_head / s.n_kv) * (s.head_dim + 2);
}

bool attention(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, const float* gate, float* part,
               float* out, const QkvShape& s, int64_t n_pos, cudaStream_t st) {
    if (s.n_head % s.n_kv || s.n_head / s.n_kv > kMaxG || s.head_dim > kMaxHd)
        return false;
    const int nc = attn_chunks(n_pos);
    attn_partial_kernel<<<dim3(s.n_kv, nc), kThreads, 0, st>>>(q, (const __half*) k_cache, (const __half*) v_cache,
                                                               part, s, n_pos);
    attn_combine_kernel<<<s.n_head, kThreads, 0, st>>>(part, gate, out, s, nc);
    return true;
}

} // namespace guild::core::drama3
