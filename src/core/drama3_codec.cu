// src/core/drama3_codec.cu - Fish Audio Drama 3's codec decoder (include/guild/core/drama3_codec.hpp).
//
// Activations are [channels][time] with a row stride `ld`; a buffer that feeds a causal convolution keeps the
// convolution's left context (P = (k - 1) * dilation samples of the previous chunk) in its first P columns.  A
// convolution is one GEMM per tap over a shifted view of that buffer, accumulated; a transposed convolution with
// kernel 2s is one GEMM into [out][k][time] and an overlap-add that carries the last frame's right half; a strided
// convolution (kernel s or 2s, stride s) splits its input into s phases ([s][in][time / s]) and is one or two GEMMs
// over them.  The encoder (reference voices) runs the same way, its weights on the device only while it encodes.
#include "guild/core/drama3_codec.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstring>
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
    __shared__ float s[512];
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

// in[c][q s + r] (q < n_q) -> ph[(r * C + c)][q]: the s phases of a strided convolution's input
__global__ void phase_split_kernel(const float* in, int ldi, float* ph, int ldp, int C, int s, int n_q) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y; // row = r * C + c
    if (q >= n_q)
        return;
    const int r = row / C, c = row - r * C;
    ph[(size_t) row * ldp + q] = in[(size_t) c * ldi + (size_t) q * s + r];
}

struct VqStage {
    const float *in_w = nullptr, *in_b = nullptr, *cb = nullptr, *out_w = nullptr, *out_b = nullptr;
    int n = 0;
};
struct VqStages {
    VqStage s[10];
    int count = 0;
};

// One block per frame: the residual vector quantizers in order (dac's VectorQuantize): z_e = in_proj(r); the code is
// the nearest codebook row by dist = |E|^2 - 2 E.C + |C|^2 of the l2-normalized E and C (lowest index on a tie, as
// argmax); z_q = z_e + (row - z_e) (the straight-through form); r -= out_proj(z_q).  The first stage is the semantic
// quantizer (r starts as z).
__global__ void vq_encode_kernel(const float* z, int ldz, VqStages S, int dim, int32_t* codes) {
    __shared__ float r[1024];
    __shared__ float ze[8], zq[8];
    __shared__ float bd[256];
    __shared__ int bi[256];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int c = tid; c < dim; c += blockDim.x)
        r[c] = z[(size_t) c * ldz + t];
    __syncthreads();
    for (int i = 0; i < S.count; ++i) {
        const VqStage& q = S.s[i];
        if (warp < 8) {
            float a = 0.0f;
            for (int c = lane; c < dim; c += 32)
                a += q.in_w[(size_t) warp * dim + c] * r[c];
            for (int o = 16; o > 0; o >>= 1)
                a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0)
                ze[warp] = a + q.in_b[warp];
        }
        __syncthreads();
        float E[8], en = 0.0f;
        for (int k = 0; k < 8; ++k)
            en += ze[k] * ze[k];
        en = fmaxf(sqrtf(en), 1e-12f);
        float E2 = 0.0f;
        for (int k = 0; k < 8; ++k) {
            E[k] = ze[k] / en;
            E2 += E[k] * E[k];
        }
        float best = INFINITY;
        int best_i = 0;
        for (int e = tid; e < q.n; e += blockDim.x) {
            const float* cr = q.cb + (size_t) e * 8;
            float cn = 0.0f;
            for (int k = 0; k < 8; ++k)
                cn += cr[k] * cr[k];
            cn = fmaxf(sqrtf(cn), 1e-12f);
            float dot = 0.0f, C2 = 0.0f;
            for (int k = 0; k < 8; ++k) {
                const float ck = cr[k] / cn;
                dot += E[k] * ck;
                C2 += ck * ck;
            }
            const float d = E2 - 2.0f * dot + C2;
            if (d < best) {
                best = d;
                best_i = e;
            }
        }
        bd[tid] = best;
        bi[tid] = best_i;
        __syncthreads();
        for (int w = blockDim.x / 2; w > 0; w >>= 1) {
            if (tid < w && (bd[tid + w] < bd[tid] || (bd[tid + w] == bd[tid] && bi[tid + w] < bi[tid]))) {
                bd[tid] = bd[tid + w];
                bi[tid] = bi[tid + w];
            }
            __syncthreads();
        }
        const int idx = bi[0];
        if (tid < 8)
            zq[tid] = ze[tid] + (q.cb[(size_t) idx * 8 + tid] - ze[tid]);
        __syncthreads();
        for (int c = tid; c < dim; c += blockDim.x) {
            float v = q.out_b[c];
            for (int k = 0; k < 8; ++k)
                v += q.out_w[(size_t) c * 8 + k] * zq[k];
            r[c] -= v;
        }
        if (tid == 0)
            codes[(size_t) t * S.count + i] = idx;
        __syncthreads();
    }
}

dim3 grid_ct(int T, int C) {
    return dim3((unsigned) ((T + kT - 1) / kT), (unsigned) C);
}

bool encode_tensor(const std::string& name) {
    for (const char* p : {"enc.", "enct.", "down.", "pre.", "vqe."})
        if (name.rfind(p, 0) == 0)
            return true;
    return false;
}

} // namespace

// ---------------------------------------------------------------- the network
struct Drama3Codec::Impl {
    struct Tensor {
        const float* p = nullptr;
        int rows = 0, cols = 0;
        uint64_t off = 0; // in dense.bin
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
    struct SConv { // causal strided conv, kernel s or 2s, stride s
        Tensor wa, wb, b;
        int s = 0, cin = 0, cout = 0;
        Act in;              // its input, with s columns of context when the kernel is 2s
        float* ph = nullptr; // [s * cin][ldp]
        int ldp = 0;
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
    struct CNX { // ConvNeXt: depthwise k7, LayerNorm, 1024 -> 4096 GELU -> 1024, gamma, + input
        Tensor dw_w, dw_b, ln_w, ln_b, pw1_w, pw1_b, pw2_w, pw2_b, gamma;
        Act in; // its input (also its residual), with 6 columns of context
        float *d = nullptr, *ln = nullptr, *h = nullptr, *g = nullptr; // g: the output [dim][cap]
        int cap = 0;
    };
    struct Layer {
        Tensor wqkv, wo, w1, w2, w3, attn_norm, ffn_norm, attn_scale, ffn_scale;
        float *kc = nullptr, *vc = nullptr;
        int rows = 0; // cached positions (<= window - 1)
    };
    struct Stack { // a causal window-limited transformer
        std::vector<Layer> layers;
        Tensor norm;
        int window = 0;
        int64_t pos = 0; // positions run since reset
    };
    struct Scratch {
        float *xn = nullptr, *qkv = nullptr, *att = nullptr, *tmp = nullptr, *ff1 = nullptr, *ff3 = nullptr;
        int cols = 0;
    };
    struct DecBlock {
        Tensor snake;
        ConvT up;
        RU ru[3];
        float* x = nullptr;     // the block's residual stream [cout][cap]
        float* y = nullptr;     // the residual units' scratch [cout][cap]
        float* up_in = nullptr; // Snake(input) for the transposed conv [cin][up_ldi]
        int up_ldi = 0, cap = 0, cout = 0;
    };
    struct EncBlock {
        RU ru[3];
        Tensor snake;
        SConv down;
        float* y = nullptr; // the residual units' scratch [cin][cap]
        int cap = 0, cin = 0;
    };
    struct Up {
        ConvT ct;
        CNX cnx;
    };
    // the encode path, built for one encode() and freed after it
    struct Encoder {
        std::vector<void*> allocs;
        void* dense = nullptr;
        Conv enc_in;         // its input holds the samples, with 6 columns of context
        float* x0 = nullptr; // enc_in's output [64][T]
        EncBlock blocks[4];
        float* xb[4] = {}; // each block's output [cout][T / prod(s)]
        Stack t;
        Tensor out_snake;
        Conv enc_out;
        SConv down[2];
        CNX cnx[2];
        Stack pre;
        Scratch sc;
        VqStages vq;
        int32_t* codes = nullptr;
        float *z = nullptr, *ktmp = nullptr;
        int chunk = 0;
    };

    std::map<std::string, Tensor> tensors;
    std::vector<char> host_encode; // the encode tensors (uploaded per encode())
    uint64_t encode_bytes = 0;
    void* dense = nullptr;
    cublasHandle_t blas = nullptr;
    cudaStream_t st = nullptr;
    int max_chunk = 0, dim = 1024, heads = 16, head_dim = 64;
    float rope_base = 10000.0f, eps = 1e-5f;
    std::vector<const float*> vq;
    const float** vq_dev = nullptr;
    int32_t* codes_dev = nullptr;
    Stack post;
    Scratch sc;
    float *z = nullptr, *ktmp = nullptr;
    Up up[2];
    Conv dec_in;
    float* dec_h = nullptr; // dec_in's output [1536][4 max_chunk]
    DecBlock blocks[4];
    Tensor out_snake;
    Conv dec_out;
    float* pcm = nullptr;
    std::vector<void*> allocs;
    std::vector<void*>* list = &allocs; // where alloc() records (an Encoder's own list while it is built)
    uint64_t vram = 0;

    float* alloc(size_t floats) {
        void* p = nullptr;
        ck(cudaMalloc(&p, floats * 4), "allocation");
        ck(cudaMemsetAsync(p, 0, floats * 4, st), "clear");
        list->push_back(p);
        if (list == &allocs)
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
        if (it == tensors.end() || it->second.p == nullptr || (rows >= 0 && it->second.rows != rows) ||
            (cols >= 0 && it->second.cols != cols))
            throw Fail("drama3 codec: dense.txt: " + name + " is missing or of another shape");
        return it->second;
    }

    // ------------------------------------------------------------ the operations
    // Y[o][t] (ld ldy) (+)= sum_c W[o][c] X[c][t] (ld ldx), T columns
    void gemm(const float* X, int ldx, const float* W, int cout, int cin, float* Y, int ldy, int T, float beta) {
        const float one = 1.0f;
        ckb(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, T, cout, cin, &one, X, ldx, W, cin, &beta, Y, ldy), "sgemm");
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
    void conv(Conv& c, float* out, int ldo, int T) {
        for (int j = 0; j < c.k; ++j)
            gemm(c.in.p + (size_t) j * c.d, c.in.ld(), c.w.p + (size_t) j * c.cout * c.cin, c.cout, c.cin, out, ldo, T,
                 j ? 1.0f : 0.0f);
        bias_act_kernel<<<grid_ct(T, c.cout), kT, 0, st>>>(out, ldo, c.b.p, T, 0);
        shift(c.in, T);
    }
    // T input columns -> T / s output columns
    void sconv(SConv& c, float* out, int ldo, int T) {
        const int n_q = (c.in.P + T) / c.s, To = T / c.s;
        phase_split_kernel<<<grid_ct(n_q, c.s * c.cin), kT, 0, st>>>(c.in.p, c.in.ld(), c.ph, c.ldp, c.cin, c.s, n_q);
        // kernel 2s (context s): taps 0..s-1 read phase column t, taps s..2s-1 column t + 1; kernel s: column t
        gemm(c.ph, c.ldp, c.wa.p, c.cout, c.s * c.cin, out, ldo, To, 0.0f);
        if (c.wb.p)
            gemm(c.ph + 1, c.ldp, c.wb.p, c.cout, c.s * c.cin, out, ldo, To, 1.0f);
        bias_act_kernel<<<grid_ct(To, c.cout), kT, 0, st>>>(out, ldo, c.b.p, To, 0);
        shift(c.in, T);
    }
    void conv_t(ConvT& c, const float* X, int ldx, float* out, int ldo, int T) {
        gemm(X, ldx, c.w.p, c.cout * c.k, c.cin, c.Z, c.ldz, T, 0.0f);
        convt_add_kernel<<<grid_ct(T * c.s, c.cout), kT, 0, st>>>(c.Z, c.ldz, c.b.p, c.carry, c.k, c.s, out, ldo, T);
        if (c.k == 2 * c.s)
            convt_carry_kernel<<<(c.cout * c.s + kT - 1) / kT, kT, 0, st>>>(c.Z, c.ldz, c.carry, c.k, c.s, T, c.cout);
    }
    // x (ld cap) += the residual unit's output, in place
    void residual_unit(RU& r, float* x, float* y, int cap, int C, int T) {
        snake_kernel<<<grid_ct(T, C), kT, 0, st>>>(x, cap, r.snake1.p, r.conv1.in.data(), r.conv1.in.ld(), T);
        conv(r.conv1, y, cap, T);
        snake_kernel<<<grid_ct(T, C), kT, 0, st>>>(y, cap, r.snake2.p, r.conv2.in.data(), r.conv2.in.ld(), T);
        conv(r.conv2, y, cap, T);
        scale_add_kernel<<<grid_ct(T, C), kT, 0, st>>>(x, cap, nullptr, y, cap, T);
    }
    // the ConvNeXt block over the T columns already in c.in.data(); the result in c.g (ld c.cap)
    void convnext(CNX& c, int T) {
        dwconv7_kernel<<<grid_ct(T, dim), kT, 0, st>>>(c.in.p, c.in.ld(), c.dw_w.p, c.dw_b.p, c.d, c.cap, T);
        layer_norm_cols_kernel<<<T, kT, 0, st>>>(c.d, c.cap, c.ln_w.p, c.ln_b.p, c.ln, c.cap, dim, 1e-6f);
        gemm(c.ln, c.cap, c.pw1_w.p, 4 * dim, dim, c.h, c.cap, T, 0.0f);
        bias_act_kernel<<<grid_ct(T, 4 * dim), kT, 0, st>>>(c.h, c.cap, c.pw1_b.p, T, 1);
        gemm(c.h, c.cap, c.pw2_w.p, dim, 4 * dim, c.g, c.cap, T, 0.0f);
        bias_act_kernel<<<grid_ct(T, dim), kT, 0, st>>>(c.g, c.cap, c.pw2_b.p, T, 0);
        scale_add_kernel<<<grid_ct(T, dim), kT, 0, st>>>(c.g, c.cap, c.gamma.p, c.in.data(), c.in.ld(), T);
        shift(c.in, T);
    }
    // n columns of x (ld ldx) through the stack, in place; the final norm's output to out (ld ldo)
    void transformer(Stack& S, Scratch& w, float* x, int ldx, int n, float* out, int ldo) {
        const int ffn = 3 * dim, ld = w.cols;
        for (Layer& L : S.layers) {
            rms_cols_kernel<<<n, kT, 0, st>>>(x, ldx, L.attn_norm.p, w.xn, ld, dim, eps);
            gemm(w.xn, ld, L.wqkv.p, 3 * dim, dim, w.qkv, ld, n, 0.0f);
            rope_append_kernel<<<dim3(n, (dim + 1023) / 1024), std::min(dim, 1024), 0, st>>>(
                w.qkv, ld, L.kc, L.vc, L.rows, S.pos, dim, head_dim, rope_base);
            window_attn_kernel<<<dim3(heads, n), 128, 0, st>>>(w.qkv, ld, L.kc, L.vc, L.rows, S.window, dim, head_dim,
                                                               w.att, ld);
            gemm(w.att, ld, L.wo.p, dim, dim, w.tmp, ld, n, 0.0f);
            scaled_add_kernel<<<grid_ct(n, dim), kT, 0, st>>>(x, ldx, w.tmp, ld, L.attn_scale.p, dim, n);
            rms_cols_kernel<<<n, kT, 0, st>>>(x, ldx, L.ffn_norm.p, w.xn, ld, dim, eps);
            gemm(w.xn, ld, L.w1.p, ffn, dim, w.ff1, ld, n, 0.0f);
            gemm(w.xn, ld, L.w3.p, ffn, dim, w.ff3, ld, n, 0.0f);
            swiglu_kernel<<<grid_ct(n, ffn), kT, 0, st>>>(w.ff1, ld, w.ff3, ld, n);
            gemm(w.ff1, ld, L.w2.p, dim, ffn, w.tmp, ld, n, 0.0f);
            scaled_add_kernel<<<grid_ct(n, dim), kT, 0, st>>>(x, ldx, w.tmp, ld, L.ffn_scale.p, dim, n);
            // keep the last window - 1 positions
            const int total = L.rows + n, keep = std::min(S.window - 1, total);
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
        rms_cols_kernel<<<n, kT, 0, st>>>(x, ldx, S.norm.p, out, ldo, dim, eps);
        S.pos += n;
    }

    // ------------------------------------------------------------ building
    Conv make_conv(const std::string& name, int k, int d, int cin, int cap) {
        Conv c;
        c.w = get(name + ".w", -1, cin);
        c.cout = c.w.rows / k;
        c.b = get(name + ".b", 1, c.cout);
        c.k = k;
        c.d = d;
        c.cin = cin;
        c.in = act(cin, (k - 1) * d, cap);
        return c;
    }
    SConv make_sconv(const std::string& name, int s, int cin, int cap) {
        SConv c;
        c.s = s;
        c.cin = cin;
        c.wa = get(name + ".wa", -1, s * cin);
        c.cout = c.wa.rows;
        if (tensors.count(name + ".wb"))
            c.wb = get(name + ".wb", c.cout, s * cin);
        c.b = get(name + ".b", 1, c.cout);
        c.in = act(cin, c.wb.p ? s : 0, cap);
        c.ldp = (c.in.P + cap) / s;
        c.ph = alloc((size_t) s * cin * c.ldp);
        return c;
    }
    RU make_ru(const std::string& p, int C, int dil, int cap) {
        RU r;
        r.snake1 = get(p + "snake1", 1, C);
        r.snake2 = get(p + "snake2", 1, C);
        r.conv1 = make_conv(p + "conv1", 7, dil, C, cap);
        r.conv2 = make_conv(p + "conv2", 1, 1, C, cap);
        return r;
    }
    CNX make_cnx(const std::string& p, int cap) {
        CNX c;
        c.dw_w = get(p + "dw.w", dim, 7);
        c.dw_b = get(p + "dw.b", 1, dim);
        c.ln_w = get(p + "ln.w", 1, dim);
        c.ln_b = get(p + "ln.b", 1, dim);
        c.pw1_w = get(p + "pw1.w", 4 * dim, dim);
        c.pw1_b = get(p + "pw1.b", 1, 4 * dim);
        c.pw2_w = get(p + "pw2.w", dim, 4 * dim);
        c.pw2_b = get(p + "pw2.b", 1, dim);
        c.gamma = get(p + "gamma", 1, dim);
        c.cap = cap;
        c.in = act(dim, 6, cap);
        c.d = alloc((size_t) dim * cap);
        c.ln = alloc((size_t) dim * cap);
        c.h = alloc((size_t) 4 * dim * cap);
        c.g = alloc((size_t) dim * cap);
        return c;
    }
    Stack make_stack(const std::string& name, int window, int cols) {
        Stack S;
        S.window = window;
        const int ffn = 3 * dim;
        for (size_t l = 0;; ++l) {
            const std::string p = name + "." + std::to_string(l) + ".";
            if (!tensors.count(p + "wqkv"))
                break;
            Layer L;
            L.wqkv = get(p + "wqkv", 3 * dim, dim);
            L.wo = get(p + "wo", dim, dim);
            L.w1 = get(p + "w1", ffn, dim);
            L.w3 = get(p + "w3", ffn, dim);
            L.w2 = get(p + "w2", dim, ffn);
            L.attn_norm = get(p + "attention_norm", 1, dim);
            L.ffn_norm = get(p + "ffn_norm", 1, dim);
            L.attn_scale = get(p + "attn_scale", 1, dim);
            L.ffn_scale = get(p + "ffn_scale", 1, dim);
            L.kc = alloc((size_t) (window + cols) * dim);
            L.vc = alloc((size_t) (window + cols) * dim);
            S.layers.push_back(L);
        }
        S.norm = get(name + ".norm", 1, dim);
        return S;
    }
    Scratch make_scratch(int cols) {
        Scratch w;
        w.cols = cols;
        w.xn = alloc((size_t) dim * cols);
        w.qkv = alloc((size_t) 3 * dim * cols);
        w.att = alloc((size_t) dim * cols);
        w.tmp = alloc((size_t) dim * cols);
        w.ff1 = alloc((size_t) 3 * dim * cols);
        w.ff3 = alloc((size_t) 3 * dim * cols);
        return w;
    }

    void build() {
        const int N = max_chunk;
        for (int i = 0; i < 10; ++i)
            vq.push_back(get("vq." + std::to_string(i), i == 0 ? 4096 : 1024, dim).p);
        vq_dev = (const float**) alloc(10 * 2); // 10 pointers
        ck(cudaMemcpyAsync(vq_dev, vq.data(), 10 * sizeof(float*), cudaMemcpyHostToDevice, st), "tables");
        codes_dev = (int32_t*) alloc((size_t) N * 10);
        post = make_stack("post", 128, N);
        sc = make_scratch(N);
        z = alloc((size_t) dim * N);
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
            U.cnx = make_cnx(p, T);
        }
        dec_in = make_conv("dec.in", 7, 1, dim, T);
        dec_h = alloc((size_t) dec_in.cout * T);
        int cin = dec_in.cout;
        for (int bi = 0; bi < 4; ++bi) {
            DecBlock& B = blocks[bi];
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
            B.up.carry = alloc((size_t) B.up.cout * B.up.s);
            B.up_ldi = T;
            B.up_in = alloc((size_t) cin * T);
            T *= B.up.s;
            B.cap = T;
            B.cout = B.up.cout;
            B.x = alloc((size_t) B.cout * T);
            B.y = alloc((size_t) B.cout * T);
            const int dil[3] = {1, 3, 9};
            for (int r = 0; r < 3; ++r)
                B.ru[r] = make_ru(p + "ru" + std::to_string(r) + ".", B.cout, dil[r], T);
            cin = B.cout;
        }
        out_snake = get("dec.out.snake", 1, cin);
        dec_out = make_conv("dec.out", 7, 1, cin, T);
        pcm = alloc((size_t) T);
        // shift scratch: the largest context (C * P) or kept transformer window
        size_t need = (size_t) post.window * dim;
        auto ctx = [&](const Act& a) { need = std::max(need, (size_t) a.C * a.P); };
        ctx(dec_in.in);
        ctx(dec_out.in);
        for (const Up& U : up)
            ctx(U.cnx.in);
        for (const DecBlock& B : blocks)
            for (const RU& r : B.ru)
                ctx(r.conv1.in);
        ktmp = alloc(need);
    }

    void reset() {
        post.pos = 0;
        for (Layer& L : post.layers)
            L.rows = 0;
        auto clear = [&](void* p, size_t bytes) { ck(cudaMemsetAsync(p, 0, bytes, st), "reset"); };
        auto clear_act = [&](const Act& a) { clear(a.p, (size_t) a.C * a.ld() * 4); };
        clear_act(dec_in.in);
        clear_act(dec_out.in);
        for (Up& U : up)
            clear_act(U.cnx.in);
        for (DecBlock& B : blocks) {
            clear(B.up.carry, (size_t) B.up.cout * B.up.s * 4);
            for (RU& r : B.ru) {
                clear_act(r.conv1.in);
                clear_act(r.conv2.in);
            }
        }
    }

    // ------------------------------------------------------------ decode
    void chunk(const int32_t* codes, int n, float* out) {
        const int ld0 = max_chunk;
        ck(cudaMemcpyAsync(codes_dev, codes, (size_t) n * 10 * 4, cudaMemcpyHostToDevice, st), "codes");
        vq_sum_kernel<<<dim3((dim + kT - 1) / kT, n), kT, 0, st>>>(vq_dev, codes_dev, 10, n, dim, z, ld0);
        transformer(post, sc, z, ld0, n, z, ld0); // in place (the final norm reads a column before writing it)
        const float* x = z;
        int ldx = ld0, T = n;
        for (Up& U : up) {
            conv_t(U.ct, x, ldx, U.cnx.in.data(), U.cnx.in.ld(), T);
            T *= U.ct.s;
            convnext(U.cnx, T);
            x = U.cnx.g;
            ldx = U.cnx.cap;
        }
        // the decoder: conv k7 1024 -> 1536, four upsampling blocks, Snake, conv k7 -> 1, tanh
        copy_cols(x, ldx, dec_in.in.data(), dec_in.in.ld(), dim, T);
        float* h = dec_h;
        int ldh = dec_in.in.cap;
        conv(dec_in, h, ldh, T);
        for (DecBlock& B : blocks) {
            snake_kernel<<<grid_ct(T, B.up.cin), kT, 0, st>>>(h, ldh, B.snake.p, B.up_in, B.up_ldi, T);
            conv_t(B.up, B.up_in, B.up_ldi, B.x, B.cap, T);
            T *= B.up.s;
            for (RU& r : B.ru)
                residual_unit(r, B.x, B.y, B.cap, B.cout, T);
            h = B.x;
            ldh = B.cap;
        }
        snake_kernel<<<grid_ct(T, dec_out.cin), kT, 0, st>>>(h, ldh, out_snake.p, dec_out.in.data(), dec_out.in.ld(),
                                                             T);
        conv(dec_out, pcm, T, T);
        bias_act_kernel<<<grid_ct(T, 1), kT, 0, st>>>(pcm, T, nullptr, T, 2);
        ck(cudaMemcpyAsync(out, pcm, (size_t) T * 4, cudaMemcpyDeviceToHost, st), "pcm");
        ck(cudaStreamSynchronize(st), "decode");
    }
    void copy_cols(const float* src, int lds, float* dst, int ldd, int C, int T) {
        ck(cudaMemcpy2DAsync(dst, (size_t) ldd * 4, src, (size_t) lds * 4, (size_t) T * 4, (size_t) C,
                             cudaMemcpyDeviceToDevice, st),
           "copy");
    }

    // ------------------------------------------------------------ encode
    void build_encoder(Encoder& E, int chunk) {
        list = &E.allocs;
        // the encode tensors to the device
        ck(cudaMalloc(&E.dense, encode_bytes), "encoder weights");
        E.allocs.push_back(E.dense);
        ck(cudaMemcpyAsync(E.dense, host_encode.data(), encode_bytes, cudaMemcpyHostToDevice, st), "encoder upload");
        uint64_t at = 0;
        for (auto& [name, t] : tensors) {
            if (!encode_tensor(name))
                continue;
            t.p = (const float*) ((const char*) E.dense + at);
            at += (uint64_t) t.rows * t.cols * 4;
        }
        E.chunk = chunk;
        int T = chunk * 2048;
        E.enc_in = make_conv("enc.in", 7, 1, 1, T);
        E.x0 = alloc((size_t) E.enc_in.cout * T);
        int cin = E.enc_in.cout;
        const int dil[3] = {1, 3, 9};
        for (int b = 0; b < 4; ++b) {
            EncBlock& B = E.blocks[b];
            const std::string p = "enc." + std::to_string(b + 1) + ".";
            B.cin = cin;
            B.cap = T;
            for (int r = 0; r < 3; ++r)
                B.ru[r] = make_ru(p + "ru" + std::to_string(r) + ".", cin, dil[r], T);
            B.y = alloc((size_t) cin * T);
            B.snake = get(p + "snake", 1, cin);
            const int s = (int) (get(p + "down.wa", -1, -1).cols / cin);
            B.down = make_sconv(p + "down", s, cin, T);
            T /= s;
            cin = B.down.cout;
            E.xb[b] = alloc((size_t) cin * T);
        }
        E.sc = make_scratch(T); // the largest transformer chunk: 4 * chunk columns
        E.t = make_stack("enct", 512, T);
        E.out_snake = get("enc.out.snake", 1, cin);
        E.enc_out = make_conv("enc.out", 3, 1, cin, T);
        for (int u = 0; u < 2; ++u) {
            E.down[u] = make_sconv("down." + std::to_string(u) + ".conv", 2, dim, T);
            T /= 2;
            E.cnx[u] = make_cnx("down." + std::to_string(u) + ".", T);
        }
        E.pre = make_stack("pre", 128, T);
        E.z = alloc((size_t) dim * T);
        for (int i = 0; i < 10; ++i) {
            const std::string p = "vqe." + std::to_string(i) + ".";
            VqStage& q = E.vq.s[i];
            q.in_w = get(p + "in.w", 8, dim).p;
            q.in_b = get(p + "in.b", 1, 8).p;
            const Tensor& cb = get(p + "cb", -1, 8);
            q.cb = cb.p;
            q.n = cb.rows;
            q.out_w = get(p + "out.w", dim, 8).p;
            q.out_b = get(p + "out.b", 1, dim).p;
        }
        E.vq.count = 10;
        E.codes = (int32_t*) alloc((size_t) chunk * 10);
        // the shift scratch must hold the encoder's contexts and its 512-position window
        size_t need = (size_t) 512 * dim;
        auto ctx = [&](const Act& a) { need = std::max(need, (size_t) a.C * a.P); };
        for (const EncBlock& B : E.blocks) {
            ctx(B.down.in);
            for (const RU& r : B.ru)
                ctx(r.conv1.in);
        }
        E.ktmp = alloc(need);
        list = &allocs;
    }
    void free_encoder(Encoder& E) {
        cudaStreamSynchronize(st);
        for (void* p : E.allocs)
            cudaFree(p);
        E.allocs.clear();
        for (auto& [name, t] : tensors)
            if (encode_tensor(name))
                t.p = nullptr;
    }
    // n frames (n * 2048 samples at pcm) -> codes [n][10]
    void encode_chunk(Encoder& E, const float* samples, int n, int32_t* out) {
        int T = n * 2048;
        ck(cudaMemcpyAsync(E.enc_in.in.data(), samples, (size_t) T * 4, cudaMemcpyHostToDevice, st), "samples");
        conv(E.enc_in, E.x0, E.enc_in.in.cap, T);
        float* x = E.x0;
        int ldx = E.enc_in.in.cap;
        for (int b = 0; b < 4; ++b) {
            EncBlock& B = E.blocks[b];
            for (RU& r : B.ru)
                residual_unit(r, x, B.y, ldx, B.cin, T);
            snake_kernel<<<grid_ct(T, B.cin), kT, 0, st>>>(x, ldx, B.snake.p, B.down.in.data(), B.down.in.ld(), T);
            const int ld_out = B.cap / B.down.s;
            sconv(B.down, E.xb[b], ld_out, T);
            T /= B.down.s;
            x = E.xb[b];
            ldx = ld_out;
        }
        transformer(E.t, E.sc, x, ldx, T, x, ldx);
        snake_kernel<<<grid_ct(T, dim), kT, 0, st>>>(x, ldx, E.out_snake.p, E.enc_out.in.data(), E.enc_out.in.ld(), T);
        conv(E.enc_out, E.down[0].in.data(), E.down[0].in.ld(), T);
        for (int u = 0; u < 2; ++u) {
            sconv(E.down[u], E.cnx[u].in.data(), E.cnx[u].in.ld(), T);
            T /= 2;
            convnext(E.cnx[u], T);
            if (u == 0)
                copy_cols(E.cnx[0].g, E.cnx[0].cap, E.down[1].in.data(), E.down[1].in.ld(), dim, T);
        }
        transformer(E.pre, E.sc, E.cnx[1].g, E.cnx[1].cap, T, E.z, E.cnx[1].cap);
        vq_encode_kernel<<<T, 256, 0, st>>>(E.z, E.cnx[1].cap, E.vq, dim, E.codes);
        ck(cudaMemcpyAsync(out, E.codes, (size_t) T * 10 * 4, cudaMemcpyDeviceToHost, st), "codes");
        ck(cudaStreamSynchronize(st), "encode");
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
            meta["post_heads"] != "16" || meta["rates"] != "8,8,4,2" || meta["post_window"] != "128")
            throw Fail("drama3 codec: codec.txt describes another network");
        impl_ = new Impl();
        Impl& m = *impl_;
        m.max_chunk = max_chunk;
        m.rope_base = std::stof(meta["post_rope_base"]);
        m.eps = std::stof(meta["post_eps"]);
        sample_rate_ = std::stoi(meta["sample_rate"]);
        frame_samples_ = std::stoi(meta["frame_samples"]);
        can_encode_ = meta["enc_rates"] == "2,4,8,8" && meta["enc_window"] == "512";
        std::ifstream bin(dir + "/dense.bin", std::ios::binary | std::ios::ate);
        if (!bin)
            throw Fail("drama3 codec: cannot read " + dir + "/dense.bin");
        const uint64_t size = (uint64_t) bin.tellg();
        std::vector<char> host(size);
        bin.seekg(0);
        bin.read(host.data(), (std::streamsize) size);
        if (!bin)
            throw Fail("drama3 codec: short read of dense.bin");
        std::ifstream idx(dir + "/dense.txt");
        std::string line;
        uint64_t decode_bytes = 0;
        while (std::getline(idx, line)) {
            std::istringstream ls(line);
            std::string name, kind;
            int64_t rows = 0, cols = 0;
            uint64_t off = 0, bytes = 0;
            if (!(ls >> name >> kind >> rows >> cols >> off >> bytes))
                continue;
            if (kind != "f32" || off + bytes > size || bytes != (uint64_t) rows * cols * 4)
                throw Fail("drama3 codec: dense.txt: bad entry " + name);
            m.tensors[name] = Impl::Tensor{nullptr, (int) rows, (int) cols, off};
            (encode_tensor(name) ? m.encode_bytes : decode_bytes) += bytes;
        }
        ck(cudaStreamCreateWithFlags(&m.st, cudaStreamNonBlocking), "stream");
        ckb(cublasCreate(&m.blas), "create");
        ckb(cublasSetStream(m.blas, m.st), "set stream");
        // the decode tensors stay on the device; the encode tensors stay in host memory until encode()
        ck(cudaMalloc(&m.dense, decode_bytes), "weights");
        std::vector<char> packed(decode_bytes);
        m.host_encode.resize(m.encode_bytes);
        uint64_t at_d = 0, at_e = 0;
        for (auto& [name, t] : m.tensors) {
            const uint64_t bytes = (uint64_t) t.rows * t.cols * 4;
            if (encode_tensor(name)) {
                std::memcpy(m.host_encode.data() + at_e, host.data() + t.off, bytes);
                at_e += bytes;
            } else {
                std::memcpy(packed.data() + at_d, host.data() + t.off, bytes);
                t.p = (const float*) ((const char*) m.dense + at_d);
                at_d += bytes;
            }
        }
        ck(cudaMemcpy(m.dense, packed.data(), decode_bytes, cudaMemcpyHostToDevice), "weights upload");
        m.vram = decode_bytes;
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

bool Drama3Codec::encode(const float* pcm, size_t n_samples, std::vector<int32_t>& codes, std::string& err) {
    if (!impl_ || !can_encode_) {
        err = "drama3 codec: no encoder in these codec files (rerun tools/drama3_codec_rt.py)";
        return false;
    }
    Impl& m = *impl_;
    Impl::Encoder E;
    float* const decode_ktmp = m.ktmp;
    try {
        constexpr int kChunk = 16;
        m.build_encoder(E, kChunk);
        m.ktmp = E.ktmp;
        const size_t F = (size_t) frame_samples_, n = (n_samples + F - 1) / F; // zeros pad the last frame
        codes.assign(n * 10, 0);
        std::vector<float> buf(kChunk * F);
        for (size_t f0 = 0; f0 < n; f0 += kChunk) {
            const int c = (int) std::min<size_t>(kChunk, n - f0);
            std::fill(buf.begin(), buf.end(), 0.0f);
            const size_t have = std::min(n_samples - f0 * F, (size_t) c * F);
            std::memcpy(buf.data(), pcm + f0 * F, have * 4);
            m.encode_chunk(E, buf.data(), c, codes.data() + f0 * 10);
        }
        ck(cudaGetLastError(), "encode launch");
    } catch (const std::exception& e) {
        err = e.what();
        m.list = &m.allocs;
        m.ktmp = decode_ktmp;
        m.free_encoder(E);
        return false;
    }
    m.ktmp = decode_ktmp;
    m.free_encoder(E);
    return true;
}

} // namespace guild::core
