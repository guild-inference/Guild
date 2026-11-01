// src/audio/encoder.cu - Nemotron 3.5 Transcribe's audio model on the GPU (include/guild/audio/encoder.hpp).
//
// Two bidirectional pre-norm transformer encoders over the same spectrogram (the speech encoder on the per-feature
// normalized mel, the speaker model on the raw mel), in transformers' order (Nemotron3_5TranscribeAudioModel):
//   x = stack 8 frames -> Linear -> LayerNorm;  per block: x += o(attn(LN1 x));  x += fc2(gelu(fc1(LN2 x)));  LayerNorm
// The speech encoder normalizes each head's query and key (LayerNorm over the head) before rotating the first half of
// the head; the speaker model rotates whole heads.  Matrices: cuBLAS SGEMM on float32 copies of the BF16 weights,
// widened into one scratch right before each product.
#include "guild/audio/encoder.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace guild::audio {

namespace {

struct Entry {
    std::string kind;
    int64_t rows = 0, cols = 0;
    uint64_t offset = 0, bytes = 0;
    float* wide = nullptr; ///< a BF16 matrix's float32 copy, when keep_wide() kept them
};

__global__ void widen_bf16(const uint16_t* in, float* out, int64_t n) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        out[i] = __uint_as_float(uint32_t(in[i]) << 16);
}

/// LayerNorm of `rows` rows of D floats (biased variance, as torch), in place or not.  Row r starts at
/// (r / per) * outer + (r % per) * inner, the same offset in x and y.
__global__ void layer_norm(const float* x, float* y, const float* w, const float* b, int D, float eps, int per,
                           int64_t outer, int64_t inner) {
    const int r = blockIdx.x;
    const int64_t off = int64_t(r / per) * outer + int64_t(r % per) * inner;
    const float* xr = x + off;
    float* yr = y + off;
    __shared__ float red[32];
    __shared__ float stat;
    auto block_sum = [&](float v) {
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0)
            red[threadIdx.x >> 5] = v;
        __syncthreads();
        if (threadIdx.x < 32) {
            float s = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
            for (int o = 16; o > 0; o >>= 1)
                s += __shfl_xor_sync(0xffffffffu, s, o);
            if (threadIdx.x == 0)
                stat = s;
        }
        __syncthreads();
        float s = stat;
        __syncthreads();
        return s;
    };
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        s += xr[i];
    const float mean = block_sum(s) / float(D);
    float v = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float d = xr[i] - mean;
        v += d * d;
    }
    const float rstd = rsqrtf(block_sum(v) / float(D) + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

/// RoPE (rotate_half form) on the first `rot` channels of each head of the query and key parts of qkv rows
/// [T][3 * D]: cs holds T x rot/2 (cos, sin) pairs.
__global__ void rope(float* qkv, const float2* cs, int T, int heads, int hd, int rot, int D) {
    const int t = blockIdx.x, half = rot / 2;
    for (int i = threadIdx.x; i < 2 * heads * half; i += blockDim.x) {
        const int which = i / (heads * half), h = (i / half) % heads, j = i % half;
        float* v = qkv + int64_t(t) * 3 * D + which * D + h * hd;
        const float2 c = cs[int64_t(t) * half + j];
        const float x1 = v[j], x2 = v[j + half];
        v[j] = __fadd_rn(__fmul_rn(x1, c.x), -__fmul_rn(x2, c.y));
        v[j + half] = __fadd_rn(__fmul_rn(x2, c.x), __fmul_rn(x1, c.y));
    }
}

/// Row softmax of `rows` rows of n floats, in place, over the first `valid` keys (the rest get 0: padding)
__global__ void softmax_rows(float* rows, int n, int valid) {
    float* r = rows + int64_t(blockIdx.x) * n;
    __shared__ float red[32];
    __shared__ float stat;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < valid; i += blockDim.x)
        m = fmaxf(m, r[i]);
    for (int o = 16; o > 0; o >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0)
            stat = v;
    }
    __syncthreads();
    m = stat;
    __syncthreads();
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = i < valid ? expf(r[i] - m) : 0.f;
        r[i] = e;
        s += e;
    }
    for (int o = 16; o > 0; o >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0)
            stat = v;
    }
    __syncthreads();
    const float inv = stat;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        r[i] = r[i] / inv;
}

enum Epilogue {
    kBias = 0,
    kBiasGelu = 1,
    kBiasResidual = 2,
    kBiasRelu = 3
};

/// y[t][n] = op(y[t][n] + b[n]) for kBias / kBiasGelu / kBiasRelu (y holds the product); r[t][n] += y[t][n] + b[n]
/// for kBiasResidual
__global__ void epilogue(float* y, const float* b, float* r, int64_t T, int N, int mode) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < T * N; i += int64_t(gridDim.x) * blockDim.x) {
        float v = b ? y[i] + b[i % N] : y[i];
        if (mode == kBiasGelu)
            v = 0.5f * v * (1.f + erff(v * 0.70710678118654752440f));
        else if (mode == kBiasRelu)
            v = fmaxf(v, 0.f);
        if (mode == kBiasResidual)
            r[i] = r[i] + v;
        else
            y[i] = v;
    }
}

__global__ void relu_inplace(float* y, int64_t n) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        y[i] = fmaxf(y[i], 0.f);
}

/// The sub-pixel convolution's input: rows t of [x[t - 1] | x[t] | x[t + 1]] interleaved as the weight's
/// (in channel, tap) order, zero outside the clip: out[t][c * 3 + k] = x[t + k - 1][c]
__global__ void im2col3(const float* x, float* out, int T, int C) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < int64_t(T) * C * 3;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int t = int(i / (C * 3)), c = int(i % (C * 3)) / 3, k = int(i % 3);
        const int s = t + k - 1;
        out[i] = s >= 0 && s < T ? x[int64_t(s) * C + c] : 0.f;
    }
}

__global__ void add_rows(float* y, const float* x, int64_t n) {
    for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
        y[i] = y[i] + x[i];
}

int grid_for(int64_t n) {
    return int(std::min<int64_t>((n + 255) / 256, 4096));
}

} // namespace

struct AudioEncoder::Impl {
    AudioGeometry g;
    MelConfig mel;
    std::map<std::string, Entry> index;
    uint8_t* dev = nullptr; ///< audio.bin on the device
    uint64_t dev_bytes = 0;
    cudaStream_t st = nullptr;
    cublasHandle_t blas = nullptr;
    float* wide = nullptr; ///< one widened matrix
    int64_t wide_elems = 0;
    float* wide_all = nullptr; ///< every BF16 matrix widened (keep_wide)
    uint64_t wide_all_bytes = 0;
    // activations, sized for diar_chunk tokens
    float *feat = nullptr, *h = nullptr, *a = nullptr, *qkv = nullptr, *att = nullptr, *ff = nullptr, *sc = nullptr,
          *col = nullptr, *up = nullptr, *cls = nullptr, *spk = nullptr, *emb = nullptr;
    float2* cs = nullptr;
    double ms = 0.0;
    std::string err_;

    ~Impl() {
        for (void* p :
             {(void*) dev, (void*) wide, (void*) wide_all, (void*) feat, (void*) h, (void*) a, (void*) qkv, (void*) att,
              (void*) ff, (void*) sc, (void*) col, (void*) up, (void*) cls, (void*) spk, (void*) emb, (void*) cs})
            if (p)
                cudaFree(p);
        if (blas)
            cublasDestroy(blas);
        if (st)
            cudaStreamDestroy(st);
    }

    bool cuda(cudaError_t e, const char* what) {
        if (e == cudaSuccess)
            return true;
        err_ = std::string("audio encoder: ") + what + ": " + cudaGetErrorString(e);
        return false;
    }
    bool blas_ok(cublasStatus_t e, const char* what) {
        if (e == CUBLAS_STATUS_SUCCESS)
            return true;
        err_ = std::string("audio encoder: cuBLAS ") + what + " failed (" + std::to_string(int(e)) + ")";
        return false;
    }

    const Entry* find(const std::string& name) {
        auto it = index.find(name);
        if (it == index.end()) {
            err_ = "audio encoder: no tensor " + name;
            return nullptr;
        }
        return &it->second;
    }
    const float* f32(const std::string& name) {
        const Entry* e = find(name);
        return e ? reinterpret_cast<const float*>(dev + e->offset) : nullptr;
    }

    /// Y[T, N] (row stride ldy) = X[T, K] (row stride ldx) . W^T, W the BF16 matrix `name` [N, K], widened first
    /// (or kept widened)
    bool linear(const std::string& name, const float* X, int64_t ldx, float* Y, int64_t ldy, int T) {
        const Entry* e = find(name);
        if (!e)
            return false;
        const int64_t N = e->rows, K = e->cols;
        const float* W = e->wide;
        if (!W) {
            widen_bf16<<<grid_for(N * K), 256, 0, st>>>(reinterpret_cast<const uint16_t*>(dev + e->offset), wide,
                                                        N * K);
            W = wide;
        }
        const float one = 1.f, zero = 0.f;
        return blas_ok(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, int(N), T, int(K), &one, W, int(K), X, int(ldx),
                                   &zero, Y, int(ldy)),
                       name.c_str());
    }

    /// every BF16 matrix widened once, each at a 256-byte boundary as the per-product scratch is, if the device keeps
    /// `keep_free` bytes free after
    bool keep_wide(uint64_t keep_free) {
        if (wide_all)
            return true;
        uint64_t need = 0;
        for (auto& [name, e] : index)
            if (e.kind == "bf16")
                need += (uint64_t(e.rows * e.cols) * 4 + 255) / 256 * 256;
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b < need + keep_free ||
            cudaMalloc(&wide_all, need) != cudaSuccess) {
            wide_all = nullptr;
            return false;
        }
        wide_all_bytes = need;
        uint8_t* p = reinterpret_cast<uint8_t*>(wide_all);
        for (auto& [name, e] : index)
            if (e.kind == "bf16") {
                e.wide = reinterpret_cast<float*>(p);
                widen_bf16<<<grid_for(e.rows * e.cols), 256, 0, st>>>(reinterpret_cast<const uint16_t*>(dev + e.offset),
                                                                      e.wide, e.rows * e.cols);
                p += (uint64_t(e.rows * e.cols) * 4 + 255) / 256 * 256;
            }
        return cuda(cudaStreamSynchronize(st), "widen");
    }
    /// LayerNorm rows of D at stride D (in place when x == y)
    bool norm(const std::string& tag, const float* x, float* y, int rows, int D) {
        const float *w = f32(tag + ".w"), *b = f32(tag + ".b");
        if (!w || !b)
            return false;
        layer_norm<<<rows, 256, 0, st>>>(x, y, w, b, D, g.ln_eps, 1, D, 0);
        return true;
    }

    /// One encoder: `x` [T][stack * n_mels] -> h [T][D] (the output LayerNorm applied)
    /// `valid`: the tokens with audio; the rest (a group of padding the reference adds when the frames fill whole
    /// groups) are queries but no keys, as the reference's attention mask has them
    bool encoder(const char* tag, const float* x, int T, int valid, int layers, int D, int heads, int ff_dim,
                 bool qk_norm, float rope_fraction, float rope_base) {
        const std::string t(tag);
        const int hd = D / heads, rot = int(float(hd) * rope_fraction);
        // RoPE angles as transformers computes them in float32: inv_freq = 1 / base^(2i / rot), pos * inv_freq
        std::vector<float2> host(size_t(T) * size_t(rot / 2));
        for (int i = 0; i < rot / 2; ++i) {
            const float inv = 1.0f / std::pow(rope_base, float(2 * i) / float(rot));
            for (int p = 0; p < T; ++p) {
                const float ang = float(p) * inv;
                host[size_t(p) * size_t(rot / 2) + size_t(i)] = {float(std::cos(double(ang))),
                                                                 float(std::sin(double(ang)))};
            }
        }
        if (!cuda(cudaMemcpyAsync(cs, host.data(), host.size() * sizeof(float2), cudaMemcpyHostToDevice, st), "rope"))
            return false;
        if (!linear(t + ".stack", x, g.stack * g.n_mels, h, D, T) || !norm(t + ".in_norm", h, h, T, D))
            return false;
        const float one = 1.f, zero = 0.f, scale = 1.0f / std::sqrt(float(hd));
        for (int l = 0; l < layers; ++l) {
            const std::string p = t + "." + std::to_string(l) + ".";
            if (!norm(p + "norm1", h, a, T, D) || !linear(p + "qkv", a, D, qkv, 3 * D, T))
                return false;
            if (qk_norm) {
                const float *qw = f32(p + "q_norm.w"), *qb = f32(p + "q_norm.b"), *kw = f32(p + "k_norm.w"),
                            *kb = f32(p + "k_norm.b");
                if (!qw || !qb || !kw || !kb)
                    return false;
                layer_norm<<<T * heads, 128, 0, st>>>(qkv, qkv, qw, qb, hd, g.ln_eps, heads, 3 * D, hd);
                layer_norm<<<T * heads, 128, 0, st>>>(qkv + D, qkv + D, kw, kb, hd, g.ln_eps, heads, 3 * D, hd);
            }
            rope<<<T, 256, 0, st>>>(qkv, cs, T, heads, hd, rot, D);
            // scores[h] = q_h k_h^T * scale, softmax, out_h = p_h v_h (heads side by side in att)
            if (!blas_ok(cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, T, T, hd, &scale, qkv + D, 3 * D, hd,
                                                   qkv, 3 * D, hd, &zero, sc, T, int64_t(T) * T, heads),
                         "scores"))
                return false;
            softmax_rows<<<T * heads, 256, 0, st>>>(sc, T, valid);
            if (!blas_ok(cublasSgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, T, T, &one, qkv + 2 * D, 3 * D,
                                                   hd, sc, T, int64_t(T) * T, &zero, att, D, hd, heads),
                         "attention"))
                return false;
            const float *ob = f32(p + "out.b"), *b1 = f32(p + "fc1.b"), *b2 = f32(p + "fc2.b");
            if (!ob || !b1 || !b2 || !linear(p + "out", att, D, a, D, T))
                return false;
            epilogue<<<grid_for(int64_t(T) * D), 256, 0, st>>>(a, ob, h, T, D, kBiasResidual);
            if (!norm(p + "norm2", h, a, T, D) || !linear(p + "fc1", a, D, ff, ff_dim, T))
                return false;
            epilogue<<<grid_for(int64_t(T) * ff_dim), 256, 0, st>>>(ff, b1, nullptr, T, ff_dim, kBiasGelu);
            if (!linear(p + "fc2", ff, ff_dim, a, D, T))
                return false;
            epilogue<<<grid_for(int64_t(T) * D), 256, 0, st>>>(a, b2, h, T, D, kBiasResidual);
        }
        return norm(t + ".out_norm", h, h, T, D);
    }

    bool load(const std::string& dir, std::string& err) {
        std::ifstream geo(dir + "/audio.txt");
        if (!geo) {
            err = "audio encoder: cannot read " + dir + "/audio.txt";
            return false;
        }
        std::map<std::string, std::string> kv;
        for (std::string line; std::getline(geo, line);) {
            std::istringstream ls(line);
            std::string k, v;
            if (ls >> k >> v)
                kv[k] = v;
        }
        auto gi = [&](const char* k, int& v) {
            if (kv.count(k))
                v = std::stoi(kv[k]);
        };
        auto gf = [&](const char* k, float& v) {
            if (kv.count(k))
                v = std::stof(kv[k]);
        };
        gi("sample_rate", g.sample_rate), gi("n_fft", g.n_fft), gi("hop", g.hop), gi("win", g.win);
        gi("n_mels", g.n_mels), gi("stack", g.stack), gf("preemphasis", g.preemphasis), gf("log_guard", g.log_guard);
        gf("ln_eps", g.ln_eps), gi("asr_layers", g.asr_layers), gi("asr_dim", g.asr_dim), gi("asr_heads", g.asr_heads);
        gi("asr_ff", g.asr_ff), gf("asr_rope_fraction", g.asr_rope_fraction), gf("asr_rope_base", g.asr_rope_base);
        gi("diar_layers", g.diar_layers), gi("diar_dim", g.diar_dim), gi("diar_heads", g.diar_heads);
        gi("diar_ff", g.diar_ff), gi("diar_hidden", g.diar_hidden), gi("speakers", g.speakers);
        gf("diar_rope_base", g.diar_rope_base), gf("speaker_threshold", g.speaker_threshold);
        gi("diar_chunk", g.diar_chunk), gi("out_dim", g.out_dim);

        std::ifstream idx(dir + "/audio_index.txt");
        for (std::string line; std::getline(idx, line);) {
            std::istringstream ls(line);
            std::string name;
            Entry e;
            if (ls >> name >> e.kind >> e.rows >> e.cols >> e.offset >> e.bytes)
                index[name] = e;
        }
        std::ifstream bin(dir + "/audio.bin", std::ios::binary | std::ios::ate);
        if (index.empty() || !bin) {
            err = "audio encoder: cannot read " + dir + "/audio_index.txt or audio.bin";
            return false;
        }
        dev_bytes = uint64_t(bin.tellg());
        bin.seekg(0);
        std::vector<uint8_t> host(dev_bytes);
        bin.read(reinterpret_cast<char*>(host.data()), std::streamsize(dev_bytes));
        for (auto& [name, e] : index)
            if (e.offset + e.bytes > dev_bytes) {
                err = "audio encoder: " + name + " lies past the end of audio.bin";
                return false;
            }
        // the front end's stored constants
        auto host_f32 = [&](const char* name, std::vector<float>& out) {
            const Entry& e = index[name];
            out.resize(e.bytes / 4);
            std::memcpy(out.data(), host.data() + e.offset, e.bytes);
        };
        if (!index.count("frontend.window") || !index.count("frontend.filters")) {
            err = "audio encoder: the front end's window or filter bank is missing";
            return false;
        }
        mel.rate = g.sample_rate, mel.n_fft = g.n_fft, mel.hop = g.hop, mel.win = g.win, mel.n_mels = g.n_mels;
        mel.stack = g.stack, mel.preemphasis = g.preemphasis, mel.guard = g.log_guard;
        host_f32("frontend.window", mel.window);
        host_f32("frontend.filters", mel.filters);
        for (auto& [name, e] : index)
            if (e.kind == "bf16")
                wide_elems = std::max<int64_t>(wide_elems, e.rows * e.cols);

        const int64_t T = g.diar_chunk, D = std::max(g.asr_dim, g.diar_dim);
        const int64_t F = std::max(g.asr_ff, g.diar_ff), H = std::max(g.asr_heads, g.diar_heads);
        auto alloc = [&](float*& p, int64_t n) { return cuda(cudaMalloc(&p, size_t(n) * sizeof(float)), "alloc"); };
        if (!cuda(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream") ||
            !blas_ok(cublasCreate(&blas), "create") || !blas_ok(cublasSetStream(blas, st), "stream") ||
            !cuda(cudaMalloc(&dev, dev_bytes), "weights") ||
            !cuda(cudaMemcpy(dev, host.data(), dev_bytes, cudaMemcpyHostToDevice), "upload") ||
            !alloc(wide, wide_elems) || !alloc(feat, T * g.stack * g.n_mels) || !alloc(h, T * D) || !alloc(a, T * D) ||
            !alloc(qkv, T * 3 * D) || !alloc(att, T * D) || !alloc(ff, T * F) || !alloc(sc, H * T * T) ||
            !alloc(col, T * 3 * g.diar_hidden) || !alloc(up, T * g.stack * g.diar_hidden) ||
            !alloc(cls, T * g.stack * g.diar_hidden) || !alloc(spk, T * g.asr_dim) || !alloc(emb, T * g.out_dim) ||
            !cuda(cudaMalloc(&cs, size_t(T) * size_t(D) * sizeof(float2)), "rope")) {
            err = err_;
            return false;
        }
        // cuBLAS's default math: no TF32 on these cards, and keep it so where there is (the reference is float32)
        cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);
        return true;
    }

    bool encode(const Features& f, std::vector<float>& embeds, std::string& err, AudioDetail* detail) {
        const auto t0 = std::chrono::steady_clock::now();
        const int T = f.padded / g.stack;
        if (f.n_mels != g.n_mels || f.padded % g.stack != 0 || T <= 0) {
            err = "audio encoder: the features do not match the model's front end";
            return false;
        }
        if (T > g.diar_chunk) {
            err = "audio encoder: " + std::to_string(f.frames * g.hop / g.sample_rate) +
                  " s of audio; this runner reads up to " +
                  std::to_string(g.diar_chunk * g.stack * g.hop / g.sample_rate) +
                  " s in one piece (longer audio needs the reference's chunked reading)";
            return false;
        }
        const int D = g.asr_dim, S = g.speakers, Hd = g.diar_hidden, M = g.stack * g.n_mels;
        const int valid = (f.frames + g.stack - 1) / g.stack; // the audio tokens (the prompt's count)
        std::vector<float> host;
        // the speaker model: raw mel
        bool ok = cuda(cudaMemcpyAsync(feat, f.mel.data(), f.mel.size() * sizeof(float), cudaMemcpyHostToDevice, st),
                       "features") &&
                  encoder("diar", feat, T, valid, g.diar_layers, g.diar_dim, g.diar_heads, g.diar_ff, false, 1.0f,
                          g.diar_rope_base) &&
                  linear("diar.proj", h, g.diar_dim, a, Hd, T);
        const float *pb = f32("diar.proj.b"), *ub = f32("diar.upsample.b"), *db = f32("diar.dense.b"),
                    *ob = f32("diar.out.b");
        ok = ok && pb && ub && db && ob;
        if (ok) {
            epilogue<<<grid_for(int64_t(T) * Hd), 256, 0, st>>>(a, pb, nullptr, T, Hd, kBias);
            im2col3<<<grid_for(int64_t(T) * Hd * 3), 256, 0, st>>>(a, col, T, Hd);
            ok = linear("diar.upsample", col, 3 * Hd, up, g.stack * Hd, T);
        }
        // [T][stack * Hd] is [T * stack][Hd]: relu, dense, relu, out
        const int R = T * g.stack;
        if (ok) {
            epilogue<<<grid_for(int64_t(T) * g.stack * Hd), 256, 0, st>>>(up, ub, nullptr, T, g.stack * Hd, kBiasRelu);
            ok = linear("diar.dense", up, Hd, cls, Hd, R);
        }
        if (ok) {
            epilogue<<<grid_for(int64_t(R) * Hd), 256, 0, st>>>(cls, db, nullptr, R, Hd, kBiasRelu);
            ok = linear("diar.out", cls, Hd, a, S, R);
        }
        if (ok)
            epilogue<<<grid_for(int64_t(R) * S), 256, 0, st>>>(a, ob, nullptr, R, S, kBias);
        std::vector<float> logits(size_t(R) * size_t(S));
        ok = ok &&
             cuda(cudaMemcpyAsync(logits.data(), a, logits.size() * sizeof(float), cudaMemcpyDeviceToHost, st),
                  "logits") &&
             cuda(cudaStreamSynchronize(st), "speaker model");
        // the speech encoder: per-feature normalized mel
        const std::vector<float> norm_mel = normalize_per_feature(f);
        ok = ok &&
             cuda(cudaMemcpyAsync(feat, norm_mel.data(), norm_mel.size() * sizeof(float), cudaMemcpyHostToDevice, st),
                  "features") &&
             encoder("asr", feat, T, valid, g.asr_layers, D, g.asr_heads, g.asr_ff, true, g.asr_rope_fraction,
                     g.asr_rope_base);
        if (ok && detail) {
            detail->asr.resize(size_t(T) * size_t(D));
            ok = cuda(
                cudaMemcpyAsync(detail->asr.data(), h, detail->asr.size() * sizeof(float), cudaMemcpyDeviceToHost, st),
                "asr");
        }
        ok = ok && norm("asr_norm", h, h, T, D);
        if (!ok) {
            err = err_;
            return false;
        }
        // the speaker term (host, float32 as the reference): sigmoid, the mean over each token's frames, > threshold,
        // speaker_norm (LayerNorm over the speakers), times the sinusoidal table of the speaker slots
        std::vector<float> table(size_t(S) * size_t(D));
        const float step = float(-(std::log(10000.0) / double(D)));
        for (int i = 0; i < D / 2; ++i) {
            const float inv = std::exp(float(2 * i) * step);
            for (int s = 0; s < S; ++s) {
                const float x = float(s) * inv;
                table[size_t(s) * size_t(D) + size_t(2 * i)] = float(std::sin(double(x)));
                table[size_t(s) * size_t(D) + size_t(2 * i + 1)] = float(std::cos(double(x)));
            }
        }
        const float* nw_dev = f32("speaker_norm.w");
        std::vector<float> nw((size_t) S), nb((size_t) S);
        if (!nw_dev ||
            !cuda(cudaMemcpy(nw.data(), nw_dev, S * sizeof(float), cudaMemcpyDeviceToHost), "speaker norm") ||
            !cuda(cudaMemcpy(nb.data(), f32("speaker_norm.b"), S * sizeof(float), cudaMemcpyDeviceToHost),
                  "speaker norm")) {
            err = err_;
            return false;
        }
        std::vector<float> term(size_t(T) * size_t(D));
        for (int t = 0; t < T; ++t) {
            float act[64];
            for (int s = 0; s < S; ++s) {
                float sum = 0.f;
                for (int k = 0; k < g.stack; ++k) {
                    const float x = logits[(size_t(t) * size_t(g.stack) + size_t(k)) * size_t(S) + size_t(s)];
                    sum += 1.0f / (1.0f + std::exp(-x));
                }
                act[s] = sum / float(g.stack) > g.speaker_threshold ? 1.f : 0.f;
            }
            float mean = 0.f, var = 0.f;
            for (int s = 0; s < S; ++s)
                mean += act[s];
            mean /= float(S);
            for (int s = 0; s < S; ++s)
                var += (act[s] - mean) * (act[s] - mean);
            const float rstd = 1.0f / std::sqrt(var / float(S) + g.ln_eps);
            for (int s = 0; s < S; ++s)
                act[s] = (act[s] - mean) * rstd * nw[size_t(s)] + nb[size_t(s)];
            for (int d = 0; d < D; ++d) {
                float acc = 0.f;
                for (int s = 0; s < S; ++s)
                    acc += act[s] * table[size_t(s) * size_t(D) + size_t(d)];
                term[size_t(t) * size_t(D) + size_t(d)] = acc;
            }
        }
        const float* pjb = f32("projector.b");
        ok = pjb && cuda(cudaMemcpyAsync(spk, term.data(), term.size() * sizeof(float), cudaMemcpyHostToDevice, st),
                         "speaker term");
        if (ok) {
            add_rows<<<grid_for(int64_t(T) * D), 256, 0, st>>>(h, spk, int64_t(T) * D);
            ok = linear("projector", h, D, emb, g.out_dim, T);
        }
        if (ok) {
            epilogue<<<grid_for(int64_t(T) * g.out_dim), 256, 0, st>>>(emb, pjb, nullptr, T, g.out_dim, kBias);
            embeds.resize(size_t(valid) * size_t(g.out_dim));
            ok = cuda(cudaMemcpyAsync(embeds.data(), emb, embeds.size() * sizeof(float), cudaMemcpyDeviceToHost, st),
                      "embeddings");
        }
        if (ok && detail) {
            detail->speaker_logits = logits;
            detail->audio.resize(size_t(T) * size_t(D));
            ok = cuda(cudaMemcpyAsync(detail->audio.data(), h, detail->audio.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost, st),
                      "audio");
        }
        ok = ok && cuda(cudaStreamSynchronize(st), "encode");
        if (!ok) {
            err = err_;
            return false;
        }
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }
};

AudioEncoder::AudioEncoder() : p_(std::make_unique<Impl>()) {
}
AudioEncoder::~AudioEncoder() = default;
bool AudioEncoder::load(const std::string& dir, std::string& err) {
    return p_->load(dir, err);
}
const AudioGeometry& AudioEncoder::geometry() const {
    return p_->g;
}
const MelConfig& AudioEncoder::mel_config() const {
    return p_->mel;
}
uint64_t AudioEncoder::vram_bytes() const {
    return p_->dev_bytes + uint64_t(p_->wide_elems) * 4 + p_->wide_all_bytes;
}
bool AudioEncoder::keep_wide(uint64_t keep_free) {
    return p_->keep_wide(keep_free);
}
bool AudioEncoder::encode(const Features& f, std::vector<float>& embeds, std::string& err, AudioDetail* detail) {
    return p_->encode(f, embeds, err, detail);
}
double AudioEncoder::last_ms() const {
    return p_->ms;
}

} // namespace guild::audio
