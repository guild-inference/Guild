// src/core/drama3_codec.cu - Fish Audio Drama 3's codec decoder (include/guild/core/drama3_codec.hpp).
//
// Activations are [channels][time] with a row stride `ld`; a buffer that feeds a causal convolution keeps the
// convolution's left context (P = (k - 1) * dilation samples of the previous chunk) in its first P columns.  A
// convolution is one GEMM per tap over a shifted view of that buffer, accumulated; a transposed convolution with
// kernel 2s is one GEMM into [out][k][time] and an overlap-add that carries the last frame's right half.
#include "guild/core/drama3_codec.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace guild::core {

namespace {

constexpr int kT = 256;

struct Fail : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw Fail(std::string("drama3 codec: ") + what + ": " + cudaGetErrorString(e));
}
void ckb(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw Fail(std::string("drama3 codec: cuBLAS ") + what + " failed (" + std::to_string((int) e) + ")");
}

// ---------------------------------------------------------------- kernels
__global__ void vq_sum_kernel(const float* const* tables, const int32_t* codes, int n_cb, int n, int dim, float* out,
                              int ld) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim)
        return;
    float s = 0.0f;
    for (int i = 0; i < n_cb; ++i) {
        const int lim = i == 0 ? 4095 : 1023; // the reference clamps the upper bound only
        int code = codes[t * n_cb + i];
        code = code > lim ? lim : code;
        s += tables[i][(size_t) code * dim + c];
    }
    out[(size_t) c * ld + t] = s;
}

// per column t: out = x * rsqrt(mean_c(x^2) + eps) * w
__global__ void rms_cols_kernel(const float* x, int ldx, const float* w, float* out, int ldo, int C, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float v = x[(size_t) c * ldx + t];
        ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i)
        tot += red[i];
    const float inv = rsqrtf(tot / (float) C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        out[(size_t) c * ldo + t] = x[(size_t) c * ldx + t] * inv * w[c];
}

// per column t: LayerNorm over C (biased variance), affine
__global__ void layer_norm_cols_kernel(const float* x, int ldx, const float* w, const float* b, float* out, int ldo,
                                       int C, float eps) {
    __shared__ float red[2][32];
    const int t = blockIdx.x;
    float s = 0.0f, ss = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float v = x[(size_t) c * ldx + t];
        s += v;
        ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if ((threadIdx.x & 31) == 0) {
        red[0][threadIdx.x >> 5] = s;
        red[1][threadIdx.x >> 5] = ss;
    }
    __syncthreads();
    float S = 0.0f, SS = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) {
        S += red[0][i];
        SS += red[1][i];
    }
    const float mean = S / (float) C, var = fmaxf(SS / (float) C - mean * mean, 0.0f);
    const float inv = rsqrtf(var + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        out[(size_t) c * ldo + t] = (x[(size_t) c * ldx + t] - mean) * inv * w[c] + b[c];
}

// q, k of one column: pairwise RoPE on each head's 64 channels (cos / sin rounded to BF16, as the reference's table),
// q in place, k and v to row `row` of the caches ([pos][dim])
__global__ void rope_append_kernel(float* qkv, int ld, float* kc, float* vc, int row0, int64_t pos0, int dim,
                                   int head_dim, float base) {
    const int t = blockIdx.x, c = threadIdx.x + blockIdx.y * blockDim.x;
    if (c >= dim)
        return;
    const int d = c % head_dim, i = d >> 1, partner = c ^ 1;
    const float freq = 1.0f / powf(base, (float) (2 * i) / (float) head_dim);
    const float ang = (float) (pos0 + t) * freq;
    const float cs = __bfloat162float(__float2bfloat16_rn(cosf(ang)));
    const float sn = __bfloat162float(__float2bfloat16_rn(sinf(ang)));
    const float q = qkv[(size_t) c * ld + t], qp = qkv[(size_t) partner * ld + t];
    const float k = qkv[(size_t) (dim + c) * ld + t], kp = qkv[(size_t) (dim + partner) * ld + t];
    const float v = qkv[(size_t) (2 * dim + c) * ld + t];
    const bool even = (d & 1) == 0;
    const float qr = even ? q * cs - qp * sn : q * cs + qp * sn;
    const float kr = even ? k * cs - kp * sn : k * cs + kp * sn;
    __syncthreads(); // every read of this column's q is done before any write (all of a column is one grid row)
    qkv[(size_t) c * ld + t] = qr;
    kc[(size_t) (row0 + t) * dim + c] = kr;
    vc[(size_t) (row0 + t) * dim + c] = v;
}

// one block per (head, query column): softmax(q k / 8) v over the window ending at the query
__global__ void window_attn_kernel(const float* qkv, int ld, const float* kc, const float* vc, int row0, int window,
                                   int dim, int head_dim, float* out, int ldo) {
    __shared__ float q[128];
    __shared__ float s[256];
    __shared__ float red[32];
    const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const int qrow = row0 + t, j0 = qrow - window + 1 > 0 ? qrow - window + 1 : 0, nk = qrow - j0 + 1;
    if (tid < head_dim)
        q[tid] = qkv[(size_t) (h * head_dim + tid) * ld + t];
    __syncthreads();
    const float scale = rsqrtf((float) head_dim);
    float mx = -INFINITY;
    for (int k = tid; k < nk; k += blockDim.x) {
        const float* kr = kc + (size_t) (j0 + k) * dim + h * head_dim;
        float a = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            a += q[d] * kr[d];
        s[k] = a * scale;
        mx = fmaxf(mx, s[k]);
    }
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((tid & 31) == 0)
        red[tid >> 5] = mx;
    __syncthreads();
    mx = -INFINITY;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i)
        mx = fmaxf(mx, red[i]);
    __syncthreads();
    float sum = 0.0f;
    for (int k = tid; k < nk; k += blockDim.x) {
        s[k] = __expf(s[k] - mx);
        sum += s[k];
    }
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((tid & 31) == 0)
        red[tid >> 5] = sum;
    __syncthreads();
    sum = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i)
        sum += red[i];
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float a = 0.0f;
        for (int k = 0; k < nk; ++k)
            a += s[k] * vc[(size_t) (j0 + k) * dim + h * head_dim + d];
        out[(size_t) (h * head_dim + d) * ldo + t] = a / sum;
    }
}

// x[c][t] += scale[c] * y[c][t]
__global__ void scaled_add_kernel(float* x, int ldx, const float* y, int ldy, const float* scale, int C, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (t < T)
        x[(size_t) c * ldx + t] += scale[c] * y[(size_t) c * ldy + t];
}

// x[c][t] = (scale ? scale[c] : 1) * x[c][t] + y[c][t]
__global__ void scale_add_kernel(float* x, int ldx, const float* scale, const float* y, int ldy, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (t < T)
        x[(size_t) c * ldx + t] = (scale ? scale[c] : 1.0f) * x[(size_t) c * ldx + t] + y[(size_t) c * ldy + t];
}

// a = silu(a) * b
__global__ void swiglu_kernel(float* a, int lda, const float* b, int ldb, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (t >= T)
        return;
    const float g = a[(size_t) c * lda + t];
    a[(size_t) c * lda + t] = g / (1.0f + __expf(-g)) * b[(size_t) c * ldb + t];
}

// x = act(x + bias[c]); act 0 none, 1 exact GELU, 2 tanh
__global__ void bias_act_kernel(float* x, int ld, const float* bias, int T, int act) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (t >= T)
        return;
    float v = x[(size_t) c * ld + t] + (bias ? bias[c] : 0.0f);
    if (act == 1)
        v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    else if (act == 2)
        v = tanhf(v);
    x[(size_t) c * ld + t] = v;
}

// out = x + 1 / (alpha + 1e-9) * sin(alpha x)^2
__global__ void snake_kernel(const float* x, int ldx, const float* alpha, float* out, int ldo, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (t >= T)
        return;
    const float a = alpha[c], v = x[(size_t) c * ldx + t], s = sinf(a * v);
    out[(size_t) c * ldo + t] = v + s * s / (a + 1e-9f);
}

// depthwise causal k7: out[c][t] = b[c] + sum_j w[c][j] in[c][t + j] (in's first 6 columns: the previous chunk)
__global__ void dwconv7_kernel(const float* in, int ldi, const float* w, const float* b, float* out, int ldo, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (t >= T)
        return;
    float a = b[c];
#pragma unroll
    for (int j = 0; j < 7; ++j)
        a += w[c * 7 + j] * in[(size_t) c * ldi + t + j];
    out[(size_t) c * ldo + t] = a;
}

// transposed conv overlap-add: out[o][t s + j] = b[o] + Z[o k + j][t] + (k == 2s: Z[o k + j + s][t - 1], or the
// carried last column of the previous chunk at t = 0)
__global__ void convt_add_kernel(const float* Z, int ldz, const float* bias, const float* carry, int k, int s,
                                 float* out, int ldo, int T) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, o = blockIdx.y;
    if (n >= T * s)
        return;
    const int t = n / s, j = n - t * s;
    float v = bias[o] + Z[(size_t) (o * k + j) * ldz + t];
    if (k == 2 * s)
        v += t > 0 ? Z[(size_t) (o * k + j + s) * ldz + t - 1] : carry[o * s + j];
    out[(size_t) o * ldo + n] = v;
}

__global__ void convt_carry_kernel(const float* Z, int ldz, float* carry, int k, int s, int T, int Cout) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Cout * s)
        return;
    const int o = i / s, j = i - o * s;
    carry[i] = Z[(size_t) (o * k + j + s) * ldz + T - 1];
}

dim3 grid_ct(int T, int C) {
    return dim3((unsigned) ((T + kT - 1) / kT), (unsigned) C);
}

} // namespace

// ---------------------------------------------------------------- the decoder
struct Drama3Codec::Impl {
    struct Tensor {
        const float* p = nullptr;
        int rows = 0, cols = 0;
    };
    // an activation with left context: [C][P + cap], the data at columns [P, P + T)
    struct Act {
        float* p = nullptr;
        int C = 0, P = 0, cap = 0;
        int ld() const {
            return P + cap;
        }
        float* data() const {
            return p + P;
        }
    };
    struct Conv { // causal conv k, dilation d, weights [k][out][in]
        Tensor w, b;
        int k = 1, d = 1, cin = 0, cout = 0;
        Act in; // its input, with (k - 1) d columns of context
    };
    struct ConvT {
        Tensor w, b;
        int k = 0, s = 0, cin = 0, cout = 0;
        float* Z = nullptr; // [cout * k][ldz]
        int ldz = 0;
        float* carry = nullptr; // [cout][s]
    };
    struct RU {
        Tensor snake1, snake2;
        Conv conv1, conv2;
    };
    struct Block {
        Tensor snake;
        ConvT up;
        RU ru[3];
        float* x = nullptr;     // the block's residual stream [cout][cap]
        float* y = nullptr;     // the residual units' scratch [cout][cap]
        float* up_in = nullptr; // Snake(input) for the transposed conv [cin][up_ldi]
        int up_ldi = 0, cap = 0, cout = 0;
    };
    struct PostLayer {
        Tensor wqkv, wo, w1, w2, w3, attn_norm, ffn_norm, attn_scale, ffn_scale;
        float *kc = nullptr, *vc = nullptr;
        int rows = 0; // cached positions (<= window - 1)
    };
    struct Up {
        ConvT ct;
        Tensor dw_w, dw_b, ln_w, ln_b, pw1_w, pw1_b, pw2_w, pw2_b, gamma;
        Act dw_in; // the ConvNeXt input (also its residual), with 6 columns of context
        float *d = nullptr, *ln = nullptr, *h = nullptr, *g = nullptr;
        int cap = 0;
    };

    std::map<std::string, Tensor> tensors;
    void* dense = nullptr;
    cublasHandle_t blas = nullptr;
    cudaStream_t st = nullptr;
    int max_chunk = 0, dim = 1024, heads = 16, head_dim = 64, window = 128;
    float rope_base = 10000.0f, eps = 1e-5f;
    std::vector<const float*> vq;
    const float** vq_dev = nullptr;
    int32_t* codes_dev = nullptr;
    // post_module
    std::vector<PostLayer> post;
    Tensor post_norm;
    float *z = nullptr, *xn = nullptr, *qkv = nullptr, *att = nullptr, *tmp = nullptr, *ff1 = nullptr, *ff3 = nullptr,
          *ktmp = nullptr;
    int64_t pos = 0; // frames decoded since reset
    Up up[2];
    Conv dec_in;
    float* dec_h = nullptr; // dec_in's output [1536][4 max_chunk]
    Block blocks[4];
    Tensor out_snake;
    Conv dec_out;
    float* pcm = nullptr;
    std::vector<void*> allocs;
    uint64_t vram = 0;

    float* alloc(size_t floats) {
        void* p = nullptr;
        ck(cudaMalloc(&p, floats * 4), "allocation");
        ck(cudaMemsetAsync(p, 0, floats * 4, st), "clear");
        allocs.push_back(p);
        vram += floats * 4;
        return (float*) p;
    }
    Act act(int C, int P, int cap) {
        Act a;
        a.C = C;
        a.P = P;
        a.cap = cap;
        a.p = alloc((size_t) C * (P + cap));
        return a;
    }
    const Tensor& get(const std::string& name, int rows, int cols) {
        auto it = tensors.find(name);
        if (it == tensors.end() || (rows >= 0 && it->second.rows != rows) || (cols >= 0 && it->second.cols != cols))
            throw Fail("drama3 codec: dense.txt: " + name + " is missing or of another shape");
        return it->second;
    }

    // Y[o][t] (ld ldy) (+)= sum_c W[o][c] X[c][t] (ld ldx), T columns
    void gemm(const float* X, int ldx, const float* W, int cout, int cin, float* Y, int ldy, int T, float beta) {
        const float one = 1.0f;
        ckb(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, T, cout, cin, &one, X, ldx, W, cin, &beta, Y, ldy), "sgemm");
    }
    void conv(Conv& c, float* out, int ldo, int T) {
        for (int j = 0; j < c.k; ++j)
            gemm(c.in.p + (size_t) j * c.d, c.in.ld(), c.w.p + (size_t) j * c.cout * c.cin, c.cout, c.cin, out, ldo, T,
                 j ? 1.0f : 0.0f);
        bias_act_kernel<<<grid_ct(T, c.cout), kT, 0, st>>>(out, ldo, c.b.p, T, 0);
        shift(c.in, T);
    }
    // keep the last P columns of [context | T new columns] as the next chunk's context
    void shift(Act& a, int T) {
        if (a.P == 0)
            return;
        ck(cudaMemcpy2DAsync(ktmp, (size_t) a.P * 4, a.p + T, (size_t) a.ld() * 4, (size_t) a.P * 4, (size_t) a.C,
                             cudaMemcpyDeviceToDevice, st),
           "context shift");
        ck(cudaMemcpy2DAsync(a.p, (size_t) a.ld() * 4, ktmp, (size_t) a.P * 4, (size_t) a.P * 4, (size_t) a.C,
                             cudaMemcpyDeviceToDevice, st),
           "context shift");
    }
    void conv_t(ConvT& c, const float* X, int ldx, float* out, int ldo, int T) {
        gemm(X, ldx, c.w.p, c.cout * c.k, c.cin, c.Z, c.ldz, T, 0.0f);
        convt_add_kernel<<<grid_ct(T * c.s, c.cout), kT, 0, st>>>(c.Z, c.ldz, c.b.p, c.carry, c.k, c.s, out, ldo, T);
        if (c.k == 2 * c.s)
            convt_carry_kernel<<<(c.cout * c.s + kT - 1) / kT, kT, 0, st>>>(c.Z, c.ldz, c.carry, c.k, c.s, T, c.cout);
    }

    void build() {
        const int N = max_chunk;
        for (int i = 0; i < 10; ++i)
            vq.push_back(get("vq." + std::to_string(i), i == 0 ? 4096 : 1024, dim).p);
        vq_dev = (const float**) alloc(10 * 2); // 10 pointers
        ck(cudaMemcpyAsync(vq_dev, vq.data(), 10 * sizeof(float*), cudaMemcpyHostToDevice, st), "tables");
        codes_dev = (int32_t*) alloc((size_t) N * 10);
        const int ffn = 3 * dim;
        for (size_t l = 0;; ++l) {
            const std::string p = "post." + std::to_string(l) + ".";
            if (!tensors.count(p + "wqkv"))
                break;
            PostLayer L;
            L.wqkv = get(p + "wqkv", 3 * dim, dim);
            L.wo = get(p + "wo", dim, dim);
            L.w1 = get(p + "w1", ffn, dim);
            L.w3 = get(p + "w3", ffn, dim);
            L.w2 = get(p + "w2", dim, ffn);
            L.attn_norm = get(p + "attention_norm", 1, dim);
            L.ffn_norm = get(p + "ffn_norm", 1, dim);
            L.attn_scale = get(p + "attn_scale", 1, dim);
            L.ffn_scale = get(p + "ffn_scale", 1, dim);
            L.kc = alloc((size_t) (window + N) * dim);
            L.vc = alloc((size_t) (window + N) * dim);
            post.push_back(L);
        }
        post_norm = get("post.norm", 1, dim);
        z = alloc((size_t) dim * N);
        xn = alloc((size_t) dim * N);
        qkv = alloc((size_t) 3 * dim * N);
        att = alloc((size_t) dim * N);
        tmp = alloc((size_t) dim * N);
        ff1 = alloc((size_t) ffn * N);
        ff3 = alloc((size_t) ffn * N);
        ktmp = alloc((size_t) (window + N) * dim + (size_t) 1536 * 64); // cache shifts and context shifts
        int T = N;
        for (int u = 0; u < 2; ++u) {
            Up& U = up[u];
            const std::string p = "up." + std::to_string(u) + ".";
            U.ct.w = get(p + "convt.w", -1, dim);
            U.ct.b = get(p + "convt.b", 1, dim);
            U.ct.cin = dim;
            U.ct.cout = dim;
            U.ct.k = U.ct.w.rows / dim;
            U.ct.s = U.ct.k; // k2 s2: no overlap
            U.ct.ldz = T;
            U.ct.Z = alloc((size_t) U.ct.cout * U.ct.k * T);
            T *= U.ct.s;
            U.cap = T;
            U.dw_w = get(p + "dw.w", dim, 7);
            U.dw_b = get(p + "dw.b", 1, dim);
            U.ln_w = get(p + "ln.w", 1, dim);
            U.ln_b = get(p + "ln.b", 1, dim);
            U.pw1_w = get(p + "pw1.w", 4 * dim, dim);
            U.pw1_b = get(p + "pw1.b", 1, 4 * dim);
            U.pw2_w = get(p + "pw2.w", dim, 4 * dim);
            U.pw2_b = get(p + "pw2.b", 1, dim);
            U.gamma = get(p + "gamma", 1, dim);
            U.dw_in = act(dim, 6, T);
            U.d = alloc((size_t) dim * T);
            U.ln = alloc((size_t) dim * T);
            U.h = alloc((size_t) 4 * dim * T);
            U.g = alloc((size_t) dim * T);
        }
        auto make_conv = [&](const std::string& name, int k, int d, int cin, int T_cap) {
            Conv c;
            c.w = get(name + ".w", -1, cin);
            c.cout = c.w.rows / k;
            c.b = get(name + ".b", 1, c.cout);
            c.k = k;
            c.d = d;
            c.cin = cin;
            c.in = act(cin, (k - 1) * d, T_cap);
            return c;
        };
        dec_in = make_conv("dec.in", 7, 1, dim, T);
        dec_h = alloc((size_t) dec_in.cout * T);
        int cin = dec_in.cout;
        for (int bi = 0; bi < 4; ++bi) {
            Block& B = blocks[bi];
            const std::string p = "dec." + std::to_string(bi + 1) + ".";
            B.snake = get(p + "snake", 1, cin);
            B.up.w = get(p + "convt.w", -1, cin);
            B.up.b = get(p + "convt.b", 1, -1);
            B.up.cin = cin;
            B.up.cout = B.up.b.cols;
            B.up.k = B.up.w.rows / B.up.cout;
            B.up.s = B.up.k / 2;
            B.up.ldz = T;
            B.up.Z = alloc((size_t) B.up.cout * B.up.k * T);
            B.up_ldi = T;
            B.up_in = alloc((size_t) cin * T);
            B.up.carry = alloc((size_t) B.up.cout * B.up.s);
            T *= B.up.s;
            B.cap = T;
            B.cout = B.up.cout;
            B.x = alloc((size_t) B.cout * T);
            B.y = alloc((size_t) B.cout * T);
            const int dil[3] = {1, 3, 9};
            for (int r = 0; r < 3; ++r) {
                const std::string q = p + "ru" + std::to_string(r) + ".";
                B.ru[r].snake1 = get(q + "snake1", 1, B.cout);
                B.ru[r].snake2 = get(q + "snake2", 1, B.cout);
                B.ru[r].conv1 = make_conv(q + "conv1", 7, dil[r], B.cout, T);
                B.ru[r].conv2 = make_conv(q + "conv2", 1, 1, B.cout, T);
            }
            cin = B.cout;
        }
        out_snake = get("dec.out.snake", 1, cin);
        dec_out = make_conv("dec.out", 7, 1, cin, T);
        pcm = alloc((size_t) T);
        // the context shift scratch must hold the largest context: C * P floats
        size_t need = 0;
        auto ctx = [&](const Act& a) { need = std::max(need, (size_t) a.C * a.P); };
        ctx(dec_in.in);
        ctx(dec_out.in);
        ctx(up[0].dw_in);
        ctx(up[1].dw_in);
        for (const Block& B : blocks)
            for (const RU& r : B.ru)
                ctx(r.conv1.in);
        if (need > (size_t) (window + N) * dim + (size_t) 1536 * 64)
            ktmp = alloc(need);
    }

    void reset() {
        pos = 0;
        for (PostLayer& L : post)
            L.rows = 0;
        auto clear = [&](void* p, size_t bytes) { ck(cudaMemsetAsync(p, 0, bytes, st), "reset"); };
        auto clear_act = [&](const Act& a) { clear(a.p, (size_t) a.C * a.ld() * 4); };
        clear_act(dec_in.in);
        clear_act(dec_out.in);
        for (Up& U : up)
            clear_act(U.dw_in);
        for (Block& B : blocks) {
            clear(B.up.carry, (size_t) B.up.cout * B.up.s * 4);
            for (RU& r : B.ru) {
                clear_act(r.conv1.in);
                clear_act(r.conv2.in);
            }
        }
    }

    void chunk(const int32_t* codes, int n, float* out) {
        const int ld0 = max_chunk;
        ck(cudaMemcpyAsync(codes_dev, codes, (size_t) n * 10 * 4, cudaMemcpyHostToDevice, st), "codes");
        vq_sum_kernel<<<dim3((dim + kT - 1) / kT, n), kT, 0, st>>>(vq_dev, codes_dev, 10, n, dim, z, ld0);
        // post_module: 8 windowed causal layers over the frames
        for (PostLayer& L : post) {
            rms_cols_kernel<<<n, kT, 0, st>>>(z, ld0, L.attn_norm.p, xn, ld0, dim, eps);
            gemm(xn, ld0, L.wqkv.p, 3 * dim, dim, qkv, ld0, n, 0.0f);
            rope_append_kernel<<<dim3(n, (dim + 1023) / 1024), std::min(dim, 1024), 0, st>>>(
                qkv, ld0, L.kc, L.vc, L.rows, pos, dim, head_dim, rope_base);
            window_attn_kernel<<<dim3(heads, n), 128, 0, st>>>(qkv, ld0, L.kc, L.vc, L.rows, window, dim, head_dim, att,
                                                               ld0);
            gemm(att, ld0, L.wo.p, dim, dim, tmp, ld0, n, 0.0f);
            scaled_add_kernel<<<grid_ct(n, dim), kT, 0, st>>>(z, ld0, tmp, ld0, L.attn_scale.p, dim, n);
            rms_cols_kernel<<<n, kT, 0, st>>>(z, ld0, L.ffn_norm.p, xn, ld0, dim, eps);
            gemm(xn, ld0, L.w1.p, 3 * dim, dim, ff1, ld0, n, 0.0f);
            gemm(xn, ld0, L.w3.p, 3 * dim, dim, ff3, ld0, n, 0.0f);
            swiglu_kernel<<<grid_ct(n, 3 * dim), kT, 0, st>>>(ff1, ld0, ff3, ld0, n);
            gemm(ff1, ld0, L.w2.p, dim, 3 * dim, tmp, ld0, n, 0.0f);
            scaled_add_kernel<<<grid_ct(n, dim), kT, 0, st>>>(z, ld0, tmp, ld0, L.ffn_scale.p, dim, n);
            // keep the last window - 1 positions
            const int total = L.rows + n, keep = std::min(window - 1, total);
            if (keep > 0 && total > keep) {
                for (float* c : {L.kc, L.vc}) {
                    ck(cudaMemcpyAsync(ktmp, c + (size_t) (total - keep) * dim, (size_t) keep * dim * 4,
                                       cudaMemcpyDeviceToDevice, st),
                       "cache shift");
                    ck(cudaMemcpyAsync(c, ktmp, (size_t) keep * dim * 4, cudaMemcpyDeviceToDevice, st), "cache shift");
                }
            }
            L.rows = keep;
        }
        rms_cols_kernel<<<n, kT, 0, st>>>(z, ld0, post_norm.p, xn, ld0, dim, eps);
        // upsample: transposed conv k2 s2, then a ConvNeXt block
        const float* x = xn;
        int ldx = ld0, T = n;
        for (int u = 0; u < 2; ++u) {
            Up& U = up[u];
            conv_t(U.ct, x, ldx, U.dw_in.data(), U.dw_in.ld(), T);
            T *= U.ct.s;
            dwconv7_kernel<<<grid_ct(T, dim), kT, 0, st>>>(U.dw_in.p, U.dw_in.ld(), U.dw_w.p, U.dw_b.p, U.d, U.cap, T);
            layer_norm_cols_kernel<<<T, kT, 0, st>>>(U.d, U.cap, U.ln_w.p, U.ln_b.p, U.ln, U.cap, dim, 1e-6f);
            gemm(U.ln, U.cap, U.pw1_w.p, 4 * dim, dim, U.h, U.cap, T, 0.0f);
            bias_act_kernel<<<grid_ct(T, 4 * dim), kT, 0, st>>>(U.h, U.cap, U.pw1_b.p, T, 1);
            gemm(U.h, U.cap, U.pw2_w.p, dim, 4 * dim, U.g, U.cap, T, 0.0f);
            bias_act_kernel<<<grid_ct(T, dim), kT, 0, st>>>(U.g, U.cap, U.pw2_b.p, T, 0);
            // out = input + gamma * g, written over g (the next stage's input)
            scale_add_kernel<<<grid_ct(T, dim), kT, 0, st>>>(U.g, U.cap, U.gamma.p, U.dw_in.data(), U.dw_in.ld(), T);
            shift(U.dw_in, T);
            x = U.g;
            ldx = U.cap;
        }
        // the decoder: conv k7 1024 -> 1536, four upsampling blocks, Snake, conv k7 -> 1, tanh
        copy_cols(x, ldx, dec_in.in.data(), dec_in.in.ld(), dim, T);
        float* h = dec_h;
        int ldh = dec_in.in.cap;
        conv(dec_in, h, ldh, T);
        for (Block& B : blocks) {
            // Snake into the transposed conv's input (no context: k = 2s carries the overlap itself)
            snake_kernel<<<grid_ct(T, B.up.cin), kT, 0, st>>>(h, ldh, B.snake.p, B.up_in, B.up_ldi, T);
            conv_t(B.up, B.up_in, B.up_ldi, B.x, B.cap, T);
            T *= B.up.s;
            for (RU& r : B.ru) {
                snake_kernel<<<grid_ct(T, B.cout), kT, 0, st>>>(B.x, B.cap, r.snake1.p, r.conv1.in.data(),
                                                                r.conv1.in.ld(), T);
                conv(r.conv1, B.y, B.cap, T);
                snake_kernel<<<grid_ct(T, B.cout), kT, 0, st>>>(B.y, B.cap, r.snake2.p, r.conv2.in.data(),
                                                                r.conv2.in.ld(), T);
                conv(r.conv2, B.y, B.cap, T);
                scale_add_kernel<<<grid_ct(T, B.cout), kT, 0, st>>>(B.x, B.cap, nullptr, B.y, B.cap, T);
            }
            h = B.x;
            ldh = B.cap;
        }
        snake_kernel<<<grid_ct(T, dec_out.cin), kT, 0, st>>>(h, ldh, out_snake.p, dec_out.in.data(), dec_out.in.ld(),
                                                             T);
        conv(dec_out, pcm, T, T);
        bias_act_kernel<<<grid_ct(T, 1), kT, 0, st>>>(pcm, T, nullptr, T, 2);
        ck(cudaMemcpyAsync(out, pcm, (size_t) T * 4, cudaMemcpyDeviceToHost, st), "pcm");
        ck(cudaStreamSynchronize(st), "decode");
        pos += n;
    }
    void copy_cols(const float* src, int lds, float* dst, int ldd, int C, int T) {
        ck(cudaMemcpy2DAsync(dst, (size_t) ldd * 4, src, (size_t) lds * 4, (size_t) T * 4, (size_t) C,
                             cudaMemcpyDeviceToDevice, st),
           "copy");
    }
};

Drama3Codec::~Drama3Codec() {
    if (!impl_)
        return;
    for (void* p : impl_->allocs)
        cudaFree(p);
    cudaFree(impl_->dense);
    if (impl_->blas)
        cublasDestroy(impl_->blas);
    if (impl_->st)
        cudaStreamDestroy(impl_->st);
    delete impl_;
}

bool Drama3Codec::load(const std::string& dir, int max_chunk, std::string& err) {
    try {
        std::map<std::string, std::string> meta;
        {
            std::ifstream in(dir + "/codec.txt");
            if (!in)
                throw Fail("drama3 codec: cannot read " + dir + "/codec.txt (tools/drama3_codec_rt.py writes it)");
            std::string k, v;
            while (in >> k >> v)
                meta[k] = v;
        }
        if (meta["arch"] != "drama3-codec" || meta["n_codebooks"] != "10" || meta["post_dim"] != "1024" ||
            meta["post_heads"] != "16" || meta["rates"] != "8,8,4,2")
            throw Fail("drama3 codec: codec.txt describes another network");
        impl_ = new Impl();
        Impl& m = *impl_;
        m.max_chunk = max_chunk;
        m.window = std::stoi(meta["post_window"]);
        m.rope_base = std::stof(meta["post_rope_base"]);
        m.eps = std::stof(meta["post_eps"]);
        sample_rate_ = std::stoi(meta["sample_rate"]);
        frame_samples_ = std::stoi(meta["frame_samples"]);
        if (m.window > 256)
            throw Fail("drama3 codec: the attention kernel holds at most 256 positions");
        std::ifstream bin(dir + "/dense.bin", std::ios::binary | std::ios::ate);
        if (!bin)
            throw Fail("drama3 codec: cannot read " + dir + "/dense.bin");
        const uint64_t size = (uint64_t) bin.tellg();
        std::vector<char> host(size);
        bin.seekg(0);
        bin.read(host.data(), (std::streamsize) size);
        if (!bin)
            throw Fail("drama3 codec: short read of dense.bin");
        ck(cudaStreamCreateWithFlags(&m.st, cudaStreamNonBlocking), "stream");
        ckb(cublasCreate(&m.blas), "create");
        ckb(cublasSetStream(m.blas, m.st), "set stream");
        ck(cudaMalloc(&m.dense, size), "weights");
        ck(cudaMemcpy(m.dense, host.data(), size, cudaMemcpyHostToDevice), "weights upload");
        m.vram = size;
        std::ifstream idx(dir + "/dense.txt");
        std::string line;
        while (std::getline(idx, line)) {
            std::istringstream ls(line);
            std::string name, kind;
            int64_t rows = 0, cols = 0;
            uint64_t off = 0, bytes = 0;
            if (!(ls >> name >> kind >> rows >> cols >> off >> bytes))
                continue;
            if (kind != "f32" || off + bytes > size || bytes != (uint64_t) rows * cols * 4)
                throw Fail("drama3 codec: dense.txt: bad entry " + name);
            m.tensors[name] = Impl::Tensor{(const float*) ((const char*) m.dense + off), (int) rows, (int) cols};
        }
        m.build();
        m.reset();
        ck(cudaStreamSynchronize(m.st), "load");
        vram_ = m.vram;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

void Drama3Codec::reset() {
    if (impl_) {
        try {
            impl_->reset();
        } catch (...) {}
    }
}

bool Drama3Codec::decode(const int32_t* codes, int n, float* pcm, std::string& err) {
    if (!impl_) {
        err = "drama3 codec: not loaded";
        return false;
    }
    try {
        for (int done = 0; done < n;) {
            const int c = std::min(n - done, impl_->max_chunk);
            impl_->chunk(codes + (size_t) done * 10, c, pcm + (size_t) done * frame_samples_);
            done += c;
        }
        ck(cudaGetLastError(), "decode launch");
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

} // namespace guild::core
