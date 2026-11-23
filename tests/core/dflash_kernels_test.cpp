// tests/core/dflash_kernels_test.cpp - the DFlash drafter's kernels (include/guild/core/dflash.hpp) against host
// references on random data: the activation quantization, the int8 ("D8") and Q4_K products at one to eight columns,
// the top 16 with the log-sum-exp, the block attention over a context ring, and the per-head norm with RoPE; then a
// drafter with random weights, whose drafts for several contexts with one pass of the head (draft_slots, the batch
// slots') must equal draft()'s on each context alone, bit for bit.
// `dflash_kernels_test rates` times the two products instead.
#include "../check.hpp"
#include "guild/core/dflash.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

using namespace guild::core;

namespace {

template <typename T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CHECK(cudaMalloc((void**) &p, std::max<size_t>(1, h.size()) * sizeof(T)) == cudaSuccess);
    if (!h.empty())
        CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}
template <typename T> std::vector<T> host(const T* p, size_t n) {
    std::vector<T> h(n);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    return h;
}

// the largest |a - b| over the largest |b|
double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
    double e = 0, m = 1e-30;
    for (size_t i = 0; i < b.size(); ++i) {
        e = std::max(e, std::fabs((double) a[i] - b[i]));
        m = std::max(m, std::fabs(b[i]));
    }
    return e / m;
}

// q8_act on the device, its int8 values and scales (and scaled sums) back on the host, checked against the rule
struct Act {
    int K = 0, cols = 0;
    int8_t* xq = nullptr;
    float *xs = nullptr, *xm = nullptr;
    std::vector<int8_t> q;
    std::vector<float> s, m;
};
Act quantize(const std::vector<float>& x, int K, int cols) {
    Act a;
    a.K = K;
    a.cols = cols;
    float* dx = dev(x);
    CHECK(cudaMalloc((void**) &a.xq, (size_t) cols * K) == cudaSuccess);
    CHECK(cudaMalloc((void**) &a.xs, (size_t) cols * K / 32 * 4) == cudaSuccess);
    CHECK(cudaMalloc((void**) &a.xm, (size_t) cols * K / 32 * 4) == cudaSuccess);
    dflashk::q8_act(dx, K, cols, a.xq, a.xs, nullptr, a.xm);
    a.q = host(a.xq, (size_t) cols * K);
    a.s = host(a.xs, (size_t) cols * K / 32);
    a.m = host(a.xm, (size_t) cols * K / 32);
    cudaFree(dx);
    for (int c = 0; c < cols; ++c)
        for (int b = 0; b < K / 32; ++b) {
            float amax = 0;
            for (int i = 0; i < 32; ++i)
                amax = std::max(amax, std::fabs(x[(size_t) c * K + b * 32 + i]));
            const float d = amax / 127.f;
            CHECK(a.s[(size_t) c * (K / 32) + b] == d);
            int sum = 0;
            for (int i = 0; i < 32; ++i) {
                const int want = d > 0 ? (int) std::nearbyint(x[(size_t) c * K + b * 32 + i] / d) : 0;
                const int got = a.q[(size_t) c * K + b * 32 + i];
                CHECK(got == want);
                sum += got;
            }
            CHECK(std::fabs(a.m[(size_t) c * (K / 32) + b] - d * (float) sum) <= 1e-6f * (1.f + std::fabs(d * sum)));
        }
    return a;
}

void test_d8(std::mt19937& rng) {
    std::uniform_int_distribution<int> qd(-127, 127);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const int K : {2560, 7680})                     // one staged chunk, then three
        for (const int n_out : {640, 641, 2560, 6144}) { // 641: one row a task
            std::vector<int8_t> wq((size_t) n_out * K);
            std::vector<uint16_t> ws((size_t) n_out * K / 32);
            for (auto& v : wq)
                v = (int8_t) qd(rng);
            for (auto& v : ws)
                v = ggml_fp32_to_fp16(0.001f + 0.01f * std::fabs(nd(rng)));
            std::vector<float> x((size_t) 8 * K);
            for (auto& v : x)
                v = nd(rng);
            const Act a = quantize(x, K, 8);
            int8_t* dwq = dev(wq);
            uint16_t* dws = dev(ws);
            float* y = nullptr;
            CHECK(cudaMalloc((void**) &y, (size_t) 8 * n_out * 4) == cudaSuccess);
            for (int cols = 1; cols <= 8; ++cols) {
                CHECK(dflashk::d8_mm(dwq, dws, a.xq, a.xs, y, K, n_out, cols, nullptr));
                const std::vector<float> got = host(y, (size_t) cols * n_out);
                std::vector<double> want((size_t) cols * n_out);
                for (int c = 0; c < cols; ++c)
                    for (int r = 0; r < n_out; ++r) {
                        double acc = 0;
                        for (int b = 0; b < K / 32; ++b) {
                            long dot = 0;
                            for (int i = 0; i < 32; ++i)
                                dot += (long) wq[(size_t) r * K + b * 32 + i] * a.q[(size_t) c * K + b * 32 + i];
                            acc += (double) dot * ggml_fp16_to_fp32(ws[(size_t) r * (K / 32) + b]) *
                                   a.s[(size_t) c * (K / 32) + b];
                        }
                        want[(size_t) c * n_out + r] = acc;
                    }
                const double e = rel_err(got, want);
                if (e > 2e-6) {
                    std::fprintf(stderr, "d8_mm K %d n_out %d cols %d: relative error %.3g\n", K, n_out, cols, e);
                    CHECK(false);
                }
            }
            cudaFree(dwq);
            cudaFree(dws);
            cudaFree(y);
        }
    std::printf("d8_mm: K 2560 / 7680, 640 / 641 / 2560 / 6144 rows, 1-8 columns match\n");
}

void test_k4(std::mt19937& rng, int K,
             int n_out) { // K past 2560: the columns staged in chunks; n_out odd: a row a task
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> wf((size_t) n_out * K);
    for (auto& v : wf)
        v = nd(rng) * 0.05f;
    const size_t row = ggml_row_size(GGML_TYPE_Q4_K, K);
    std::vector<uint8_t> wq((size_t) n_out * row);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, wf.data(), wq.data(), 0, n_out, K, nullptr);
    std::vector<float> wd((size_t) n_out * K); // what the blocks hold
    const auto* tr = ggml_get_type_traits(GGML_TYPE_Q4_K);
    for (int r = 0; r < n_out; ++r)
        tr->to_float(wq.data() + (size_t) r * row, wd.data() + (size_t) r * K, K);
    std::vector<float> x((size_t) 8 * K);
    for (auto& v : x)
        v = nd(rng);
    const Act a = quantize(x, K, 8);
    uint8_t* dw = dev(wq);
    float* y = nullptr;
    CHECK(cudaMalloc((void**) &y, (size_t) 8 * n_out * 4) == cudaSuccess);
    for (int cols = 1; cols <= 8; ++cols) {
        CHECK(dflashk::k4_mm(dw, a.xq, a.xs, a.xm, y, K, n_out, cols, nullptr));
        const std::vector<float> got = host(y, (size_t) cols * n_out);
        std::vector<double> want((size_t) cols * n_out);
        for (int c = 0; c < cols; ++c)
            for (int r = 0; r < n_out; ++r) {
                double acc = 0;
                for (int k = 0; k < K; ++k)
                    acc +=
                        (double) wd[(size_t) r * K + k] * a.q[(size_t) c * K + k] * a.s[(size_t) c * (K / 32) + k / 32];
                want[(size_t) c * n_out + r] = acc;
            }
        const double e = rel_err(got, want);
        if (e > 2e-5) {
            std::fprintf(stderr, "k4_mm n_out %d cols %d: relative error %.3g\n", n_out, cols, e);
            CHECK(false);
        }
    }
    cudaFree(dw);
    cudaFree(y);
    std::printf("k4_mm: Q4_K %d x %d, 1-8 columns match ggml's dequantized rows\n", n_out, K);
}

void test_topk(std::mt19937& rng, int n) {
    const int rows = 7, k = 16;
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> lg((size_t) rows * n);
    for (auto& v : lg)
        v = nd(rng);
    lg[(size_t) 3 * n + 5] = 40.f; // a clear winner in one row, at the start of the first block's range
    float* d = dev(lg);
    int32_t* di = nullptr;
    float *dv = nullptr, *dl = nullptr, *ds = nullptr;
    CHECK(cudaMalloc((void**) &di, rows * k * 4) == cudaSuccess &&
          cudaMalloc((void**) &dv, rows * k * 4) == cudaSuccess);
    CHECK(cudaMalloc((void**) &dl, rows * 4) == cudaSuccess);
    CHECK(cudaMalloc((void**) &ds, dflashk::topk_scratch_floats(rows, n) * 4) == cudaSuccess);
    CHECK(cudaMemset(ds, 0, dflashk::topk_scratch_floats(rows, n) * 4) == cudaSuccess);
    for (int rep = 0; rep < 2; ++rep) // the second call finds the counts the first left at zero
        dflashk::topk_lse(d, n, rows, k, di, dv, dl, ds, nullptr);
    const auto gi = host(di, rows * k);
    const auto gv = host(dv, rows * k);
    const auto gl = host(dl, rows);
    for (int r = 0; r < rows; ++r) {
        const float* x = lg.data() + (size_t) r * n;
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return x[a] > x[b]; });
        double m = *std::max_element(x, x + n), s = 0;
        for (int i = 0; i < n; ++i)
            s += std::exp((double) x[i] - m);
        for (int j = 0; j < k; ++j) {
            CHECK(gi[(size_t) r * k + j] == idx[(size_t) j]);
            CHECK(gv[(size_t) r * k + j] == x[idx[(size_t) j]]);
        }
        CHECK(std::fabs(gl[(size_t) r] - (m + std::log(s))) < 1e-4 * (1 + std::fabs(m)));
    }
    cudaFree(d);
    cudaFree(di);
    cudaFree(dv);
    cudaFree(dl);
    cudaFree(ds);
    std::printf("topk_lse: 7 rows of %d, the top 16 and log-sum-exp match\n", n);
}

float h16(float v) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
}

void test_attention(std::mt19937& rng) {
    const int B = 8, heads = 32, kvh = 8, D = 128, ring = 2048, window = 2048, G = heads / kvh;
    std::normal_distribution<float> nd(0.f, 1.f);
    // two contexts: 300 positions after a reset at 1000, and 2100 positions (the ring full, the window cutting)
    for (const auto& [lo, hi] : {std::pair<int64_t, int64_t>{1000, 1300}, std::pair<int64_t, int64_t>{0, 2100}}) {
        const int64_t start = hi, ctx_lo = std::max<int64_t>(lo, hi - ring);
        std::vector<float> q((size_t) B * heads * D), kb((size_t) B * kvh * D), vb(kb.size());
        for (auto* v : {&q, &kb, &vb})
            for (auto& e : *v)
                e = nd(rng);
        std::vector<uint16_t> kr((size_t) ring * kvh * D), vr(kr.size());
        for (auto* v : {&kr, &vr})
            for (auto& e : *v)
                e = ggml_fp32_to_fp16(nd(rng));
        float *dq = dev(q), *dkb = dev(kb), *dvb = dev(vb), *dout = nullptr, *dsc = nullptr;
        uint16_t *dkr = dev(kr), *dvr = dev(vr);
        CHECK(cudaMalloc((void**) &dout, q.size() * 4) == cudaSuccess);
        CHECK(cudaMalloc((void**) &dsc, dflashk::attn_scratch_floats(B, kvh, ring) * 4) == cudaSuccess);
        dflashk::block_attention(dq, (size_t) heads * D, dkr, dvr, dkb, dvb, (size_t) kvh * D, B, heads, kvh, D, ring,
                                 ctx_lo, hi, start, window, dsc, dout, nullptr);
        const auto got = host(dout, q.size());
        std::vector<double> want(q.size());
        const float scale = 1.f / std::sqrt((float) D);
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < heads; ++h) {
                const int g = h / G;
                std::vector<double> sc;
                std::vector<const float*> vrows;
                std::vector<float> kv16; // the block's keys and values, as the kernel rounds them
                auto key = [&](int64_t p, int d) {
                    return ggml_fp16_to_fp32(kr[(size_t) (p % ring) * kvh * D + (size_t) g * D + d]);
                };
                for (int64_t p = ctx_lo; p < hi; ++p) {
                    if ((start + b) - p >= window)
                        continue;
                    double s = 0;
                    for (int d = 0; d < D; ++d)
                        s += (double) h16(q[((size_t) b * heads + h) * D + d] * scale) * key(p, d);
                    sc.push_back(s);
                }
                for (int j = 0; j < B; ++j) {
                    double s = 0;
                    for (int d = 0; d < D; ++d)
                        s += (double) h16(q[((size_t) b * heads + h) * D + d] * scale) *
                             h16(kb[((size_t) j * kvh + g) * D + d]);
                    sc.push_back(s);
                }
                const double m = *std::max_element(sc.begin(), sc.end());
                double sum = 0;
                for (auto& s : sc)
                    sum += (s = std::exp(s - m));
                for (int d = 0; d < D; ++d) {
                    double o = 0;
                    size_t i = 0;
                    for (int64_t p = ctx_lo; p < hi; ++p) {
                        if ((start + b) - p >= window)
                            continue;
                        o += sc[i++] * ggml_fp16_to_fp32(vr[(size_t) (p % ring) * kvh * D + (size_t) g * D + d]);
                    }
                    for (int j = 0; j < B; ++j)
                        o += sc[i++] * h16(vb[((size_t) j * kvh + g) * D + d]);
                    want[((size_t) b * heads + h) * D + d] = o / sum;
                }
            }
        const double e = rel_err(got, want);
        if (e > 1e-3) {
            std::fprintf(stderr, "block_attention context %lld..%lld: relative error %.3g\n", (long long) lo,
                         (long long) hi, e);
            CHECK(false);
        }
        for (void* p : {(void*) dq, (void*) dkb, (void*) dvb, (void*) dout, (void*) dsc, (void*) dkr, (void*) dvr})
            cudaFree(p);
    }
    std::printf("block_attention: a 300-position context and a full ring cut by the window match\n");
}

void test_norm_rope(std::mt19937& rng) {
    const int rows = 8, heads = 8, D = 128;
    const size_t stride = (size_t) heads * D + 512; // rows with a gap, as the merged q|k|v rows have
    const float eps = 1e-6f, theta = 10000.f;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(rows * stride), w(D);
    for (auto& v : x)
        v = nd(rng);
    for (auto& v : w)
        v = 1.f + 0.1f * nd(rng);
    std::vector<int32_t> pos(rows);
    for (int r = 0; r < rows; ++r)
        pos[(size_t) r] = 5000 + 7 * r;
    float *dx = dev(x), *dw = dev(w);
    int32_t* dp = dev(pos);
    dflashk::head_norm_rope(dx, stride, dw, rows, heads, D, eps, dp, theta, nullptr);
    const auto got = host(dx, x.size());
    double e = 0;
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < heads; ++h) {
            const float* v = x.data() + (size_t) r * stride + (size_t) h * D;
            double ss = 0;
            for (int d = 0; d < D; ++d)
                ss += (double) v[d] * v[d];
            const double rs = 1.0 / std::sqrt(ss / D + eps);
            std::vector<double> a(D);
            for (int d = 0; d < D; ++d)
                a[(size_t) d] = w[(size_t) d] * v[d] * rs;
            for (int i = 0; i < D / 2; ++i) {
                const double ang = pos[(size_t) r] / std::pow((double) theta, 2.0 * i / D);
                const double c = std::cos(ang), s = std::sin(ang), x1 = a[(size_t) i], x2 = a[(size_t) i + D / 2];
                const double y1 = x1 * c - x2 * s, y2 = x2 * c + x1 * s;
                e = std::max({e, std::fabs(got[(size_t) r * stride + (size_t) h * D + i] - y1),
                              std::fabs(got[(size_t) r * stride + (size_t) h * D + i + D / 2] - y2)});
            }
        }
    CHECK(e < 1e-2);               // the kernel takes the angle in float: near 5000 radians its last bit is 5e-4
    for (int r = 0; r < rows; ++r) // the gap between rows is untouched
        for (size_t i = (size_t) heads * D; i < stride; ++i)
            CHECK(got[(size_t) r * stride + i] == x[(size_t) r * stride + i]);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dp);
    std::printf("head_norm_rope: 8 heads of 128 at positions near 5000 match (max error %.2g)\n", e);
}

// a bf16 value, rounded to nearest even
uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// A DFlash 2 checkpoint of random bf16 weights in `dir`: 2 layers of 512 (4 heads, 2 KV heads, an MLP of 1024), a
// block of 8, a window of 128 (the ring wraps), features of 2 target layers, and a vocabulary of V tokens
void write_checkpoint(const std::string& dir, std::mt19937& rng, int V) {
    const int H = 512, layers = 2, heads = 4, kvh = 2, hd = 128, ffn = 1024, n_feat = 2, rank = 256;
    const int groups = H / 16, kp_rows = 2 * 2 * groups, qd = heads * hd, kvd = kvh * hd;
    {
        std::ofstream c(dir + "/config.json");
        c << "{\"hidden_size\": " << H << ", \"num_hidden_layers\": " << layers
          << ", \"num_attention_heads\": " << heads << ", \"num_key_value_heads\": " << kvh << ", \"head_dim\": " << hd
          << ", \"intermediate_size\": " << ffn
          << ", \"sliding_window\": 128, \"rms_norm_eps\": 1e-6, \"rope_parameters\": {\"rope_theta\": 10000.0}, "
             "\"dflash_config\": {\"block_size\": 8, \"conv_kernel_size\": 2, \"conv_group_size\": 16, "
             "\"selector_rank\": "
          << rank << ", \"selector_top_k\": 16, \"mask_token_id\": " << V - 3 << ", \"target_layer_ids\": [1, 3]}}\n";
        CHECK((bool) c);
    }
    struct T {
        std::string name;
        std::vector<int64_t> shape;
        float mean, sd;
    };
    std::vector<T> ts;
    const float wi = 1.f / std::sqrt((float) H);
    for (int l = 0; l < layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        ts.push_back({p + "input_layernorm.weight", {H}, 1.f, 0.1f});
        ts.push_back({p + "post_attention_layernorm.weight", {H}, 1.f, 0.1f});
        ts.push_back({p + "self_attn.q_norm.weight", {hd}, 1.f, 0.1f});
        ts.push_back({p + "self_attn.k_norm.weight", {hd}, 1.f, 0.1f});
        ts.push_back({p + "attention_conv.base_kernel", {2, 2, H}, 0.f, 0.3f});
        ts.push_back({p + "mlp_conv.base_kernel", {2, 2, H}, 0.f, 0.3f});
        ts.push_back({p + "attention_conv.kernel_projection.weight", {kp_rows, H}, 0.f, 0.3f * wi});
        ts.push_back({p + "mlp_conv.kernel_projection.weight", {kp_rows, H}, 0.f, 0.3f * wi});
        ts.push_back({p + "self_attn.q_proj.weight", {qd, H}, 0.f, wi});
        ts.push_back({p + "self_attn.k_proj.weight", {kvd, H}, 0.f, wi});
        ts.push_back({p + "self_attn.v_proj.weight", {kvd, H}, 0.f, wi});
        ts.push_back({p + "self_attn.o_proj.weight", {H, qd}, 0.f, 1.f / std::sqrt((float) qd)});
        ts.push_back({p + "mlp.gate_proj.weight", {ffn, H}, 0.f, wi});
        ts.push_back({p + "mlp.up_proj.weight", {ffn, H}, 0.f, wi});
        ts.push_back({p + "mlp.down_proj.weight", {H, ffn}, 0.f, 1.f / std::sqrt((float) ffn)});
    }
    ts.push_back({"fc.weight", {H, n_feat * H}, 0.f, 1.f / std::sqrt((float) (n_feat * H))});
    ts.push_back({"hidden_norm.weight", {H}, 1.f, 0.1f});
    ts.push_back({"norm.weight", {H}, 1.f, 0.1f});
    ts.push_back({"candidate_selector.hidden_projection.weight", {rank, H}, 0.f, wi});
    ts.push_back({"candidate_selector.predecessor_codebook", {V, rank}, 0.f, 0.3f});
    ts.push_back({"candidate_selector.successor_codebook", {V, rank}, 0.f, 0.3f});
    std::string hdr = "{";
    std::vector<uint16_t> data;
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const T& t : ts) {
        int64_t n = 1;
        for (int64_t d : t.shape)
            n *= d;
        const size_t begin = data.size() * 2;
        for (int64_t i = 0; i < n; ++i)
            data.push_back(to_bf16(t.mean + t.sd * nd(rng)));
        std::string shape;
        for (int64_t d : t.shape)
            shape += (shape.empty() ? "" : ",") + std::to_string(d);
        hdr += (hdr.size() > 1 ? "," : "") + std::string("\"") + t.name + "\":{\"dtype\":\"BF16\",\"shape\":[" + shape +
               "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(data.size() * 2) + "]}";
    }
    hdr += "}";
    std::ofstream f(dir + "/model.safetensors", std::ios::binary);
    const uint64_t hlen = hdr.size();
    f.write((const char*) &hlen, 8);
    f.write(hdr.data(), (std::streamsize) hdr.size());
    f.write((const char*) data.data(), (std::streamsize)(data.size() * 2));
    CHECK((bool) f);
}

// draft_slots (the batch slots' drafts with one pass of the head) against draft() on each context alone: over 2-4
// contexts (one past its ring, one of a few positions), whole blocks and blocks of the rows wanted, more rows than one
// pass of the head takes, the drafts and their probabilities the same bits
void test_shared_head(std::mt19937& rng, bool int8_weights) {
    const int V = 4098; // even (two rows a task in k4_mm) but not a multiple of 4, as Qwen's 163778-token head
    const int H = 512, n_feat = 2;
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("guild_dflash_test_" + std::to_string((long) getpid()))).string();
    std::filesystem::create_directories(dir);
    write_checkpoint(dir, rng, V);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> head((size_t) V * H); // the target's head, f16 rows in host memory
    for (auto& v : head)
        v = ggml_fp32_to_fp16(nd(rng) / std::sqrt((float) H));
    std::vector<float> emb((size_t) V * H);
    for (auto& v : emb)
        v = nd(rng);
    float* d_emb = dev(emb);
    DFlashTarget t;
    t.head = head.data();
    t.head_on_device = false;
    t.head_type = (int) GGML_TYPE_F16;
    t.head_row_bytes = (size_t) H * 2;
    t.n_vocab = V;
    t.embed = [d_emb](int32_t token, float* out, void* st) {
        cudaMemcpyAsync(out, d_emb + (size_t) token * H, (size_t) H * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) st);
    };
    if (int8_weights)
        setenv("GUILD_DFLASH_WEIGHTS", "int8", 1);
    DFlashDrafter d;
    std::string err;
    const bool loaded = d.load(dir, t, "", err);
    unsetenv("GUILD_DFLASH_WEIGHTS");
    std::filesystem::remove_all(dir);
    if (!loaded)
        std::fprintf(stderr, "dflash load: %s\n", err.c_str());
    CHECK(loaded);
    CHECK(d.block() == 8 && d.add_contexts(3, err));
    cudaStream_t st = nullptr;
    CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess);
    // four contexts: 37 positions; 70 from 1000 (two batches of rows); 5; 300 from 10 (the 128-slot ring wrapped)
    const int64_t pos0[4] = {0, 1000, 0, 10}, len[4] = {37, 70, 5, 300};
    std::vector<float> feats((size_t) 300 * n_feat * H);
    for (auto& v : feats)
        v = nd(rng);
    float* d_feats = dev(feats);
    for (int s = 0; s < 4; ++s) {
        d.use_context(s);
        d.reset(pos0[s]);
        CHECK(d.add_context(d_feats, (size_t) n_feat * H, len[s], pos0[s], st, err));
        CHECK(d.context_end() == pos0[s] + len[s]);
    }
    CHECK(cudaStreamSynchronize(st) == cudaSuccess);
    std::uniform_int_distribution<int> tok(0, V - 1);
    int32_t anchor[4];
    for (auto& a : anchor)
        a = tok(rng);
    // the references: draft() on each context alone, every block size (twice: the same bits each time)
    int32_t ref_d[4][9][8];
    float ref_p[4][9][8];
    for (int s = 0; s < 4; ++s) {
        d.use_context(s);
        for (int B = 2; B <= 8; ++B) {
            CHECK(d.draft(anchor[s], d.context_end(), B, ref_d[s][B], ref_p[s][B], st, err));
            int32_t again_d[8];
            float again_p[8];
            CHECK(d.draft(anchor[s], d.context_end(), B, again_d, again_p, st, err));
            CHECK(std::memcmp(again_d, ref_d[s][B], (size_t) (B - 1) * 4) == 0);
            CHECK(std::memcmp(again_p, ref_p[s][B], (size_t) (B - 1) * 4) == 0);
            for (int i = 0; i < B - 1; ++i)
                CHECK(ref_d[s][B][i] >= 0 && ref_d[s][B][i] < V && std::isfinite(ref_p[s][B][i]) &&
                      ref_p[s][B][i] > 0.f && ref_p[s][B][i] <= 1.f);
        }
    }
    // the references are not constant: the contexts give other drafts, and a block of 2 rows other probabilities
    // than a whole one (its queries see one another)
    bool ctx_matters = false, block_matters = false;
    for (int s = 1; s < 4; ++s)
        ctx_matters |= std::memcmp(ref_d[s][8], ref_d[0][8], 7 * 4) != 0;
    for (int s = 0; s < 4; ++s)
        block_matters |= ref_p[s][2][0] != ref_p[s][8][0];
    CHECK(ctx_matters && block_matters);
    struct Case {
        std::vector<int> sets, n, B; // B 0: the whole block
    };
    const std::vector<Case> cases = {
        {{1, 2}, {2, 2}, {0, 0}},                   // 2 slots, 2 drafts each (a window of 6 rows)
        {{0, 1, 2}, {2, 2, 1}, {0, 0, 0}},          // 3 slots, 8 rows
        {{0, 1, 2, 3}, {1, 1, 1, 1}, {0, 0, 0, 0}}, // 4 slots, a draft each
        {{3, 0, 2, 1}, {3, 2, 2, 1}, {0, 0, 0, 0}}, // 8 rows of drafts: one pass of 8 columns
        {{0, 1, 2, 3}, {7, 7, 7, 7}, {0, 0, 0, 0}}, // more than a pass takes: four passes
        {{2, 3, 0, 1}, {6, 1, 1, 5}, {0, 0, 0, 0}}, // 8 + 5 rows: two passes
        {{0, 1, 2}, {2, 2, 1}, {3, 3, 2}},          // blocks of the rows wanted (GUILD_BATCH_DRAFT_BLOCK=rows)
        {{3, 1, 0, 2}, {1, 4, 2, 7}, {2, 6, 3, 8}}, // mixed block sizes
        {{2}, {7}, {0}},                            // one slot
    };
    int checked = 0;
    for (const Case& c : cases) {
        const int nj = (int) c.sets.size();
        std::vector<DFlashDrafter::SlotDraft> jobs((size_t) nj);
        std::vector<std::array<int32_t, 8>> got_d((size_t) nj);
        std::vector<std::array<float, 8>> got_p((size_t) nj);
        for (int j = 0; j < nj; ++j) {
            DFlashDrafter::SlotDraft& s = jobs[(size_t) j];
            s.set = c.sets[(size_t) j];
            s.anchor = anchor[s.set];
            s.start = pos0[s.set] + len[s.set];
            s.B = c.B[(size_t) j] == 0 ? d.block() : c.B[(size_t) j];
            s.n = c.n[(size_t) j];
            got_d[(size_t) j].fill(-1);
            got_p[(size_t) j].fill(-1.f);
            s.drafts = got_d[(size_t) j].data();
            s.probs = got_p[(size_t) j].data();
        }
        CHECK(d.draft_slots(jobs.data(), nj, st, err));
        for (int j = 0; j < nj; ++j) {
            const DFlashDrafter::SlotDraft& s = jobs[(size_t) j];
            const bool same = std::memcmp(s.drafts, ref_d[s.set][s.B], (size_t) s.n * 4) == 0 &&
                              std::memcmp(s.probs, ref_p[s.set][s.B], (size_t) s.n * 4) == 0;
            if (!same) {
                std::fprintf(stderr, "draft_slots: set %d, block %d, %d drafts differ from draft()'s:\n", s.set, s.B,
                             s.n);
                for (int i = 0; i < s.n; ++i)
                    std::fprintf(stderr, "  %d: %d %.9g against %d %.9g\n", i, s.drafts[i], s.probs[i],
                                 ref_d[s.set][s.B][i], ref_p[s.set][s.B][i]);
            }
            CHECK(same);
            CHECK(s.drafts[s.n] == -1); // nothing past the drafts asked for (n <= 7 of 8)
            ++checked;
        }
    }
    // a job that does not start at its context's end: refused, nothing changed
    DFlashDrafter::SlotDraft bad;
    int32_t bd[8];
    bad.set = 1;
    bad.anchor = anchor[1];
    bad.start = pos0[1] + len[1] - 1;
    bad.B = 8;
    bad.n = 2;
    bad.drafts = bd;
    err.clear();
    CHECK(!d.draft_slots(&bad, 1, st, err) && !err.empty());
    for (int s = 0; s < 4; ++s) {
        d.use_context(s);
        CHECK(d.context_end() == pos0[s] + len[s]);
    }
    cudaStreamDestroy(st);
    cudaFree(d_feats);
    cudaFree(d_emb);
    std::printf("draft_slots (%s weights): %d jobs over 1-4 contexts, the drafts and probabilities equal draft()'s\n",
                int8_weights ? "int8" : "Q4_K", checked);
}

// `dflash_kernels_test rates`: GB/s of the two products at 1-8 columns on random rows (d8_mm 15360 x 2560, k4_mm
// 248320 x 2560), the median of 20 runs after 3
void rates(std::mt19937& rng) {
    const int K = 2560, n8 = 15360, n4 = 248320;
    std::vector<float> x((size_t) 8 * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x)
        v = nd(rng);
    const Act a = quantize(x, K, 8);
    std::vector<int8_t> wq((size_t) n8 * K);
    for (auto& v : wq)
        v = (int8_t) (rng() & 0x7F);
    std::vector<uint16_t> ws((size_t) n8 * K / 32, ggml_fp32_to_fp16(0.01f));
    const size_t row4 = ggml_row_size(GGML_TYPE_Q4_K, K);
    std::vector<uint8_t> w4((size_t) n4 * row4);
    for (auto& v : w4)
        v = (uint8_t) rng();
    for (size_t b = 0; b < w4.size(); b += 144) { // d and dmin: finite halves
        const uint16_t one = ggml_fp32_to_fp16(0.01f);
        std::memcpy(&w4[b], &one, 2);
        std::memcpy(&w4[b + 2], &one, 2);
    }
    int8_t* dwq = dev(wq);
    uint16_t* dws = dev(ws);
    uint8_t* dw4 = dev(w4);
    float* y = nullptr;
    CHECK(cudaMalloc((void**) &y, (size_t) 8 * n4 * 4) == cudaSuccess);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time = [&](auto&& run) {
        std::vector<float> ms;
        for (int i = 0; i < 23; ++i) {
            cudaEventRecord(e0);
            run();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float t = 0;
            cudaEventElapsedTime(&t, e0, e1);
            if (i >= 3)
                ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        return (double) ms[ms.size() / 2];
    };
    std::printf("columns   d8_mm GB/s   k4_mm GB/s\n");
    for (int cols = 1; cols <= 8; ++cols) {
        const double t8 = time([&] { dflashk::d8_mm(dwq, dws, a.xq, a.xs, y, K, n8, cols, nullptr); });
        const double t4 = time([&] { dflashk::k4_mm(dw4, a.xq, a.xs, a.xm, y, K, n4, cols, nullptr); });
        std::printf("%7d %12.0f %12.0f\n", cols, (double) (wq.size() + ws.size() * 2) / t8 / 1e6,
                    (double) w4.size() / t4 / 1e6);
    }
    cudaFree(dwq);
    cudaFree(dws);
    cudaFree(dw4);
    cudaFree(y);
}

} // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    if (argc > 1 && std::strcmp(argv[1], "rates") == 0) {
        std::mt19937 rng(7);
        rates(rng);
        return 0;
    }
    std::mt19937 rng(7);
    test_d8(rng);
    test_k4(rng, 2560, 2048);
    test_k4(rng, 2560, 2047);
    test_k4(rng, 4096, 2048);
    test_topk(rng, 248320); // float4 rows
    test_topk(rng, 106299); // a draft vocabulary of an odd size
    test_attention(rng);
    test_norm_rope(rng);
    test_shared_head(rng, false);
    test_shared_head(rng, true);
    std::printf("dflash kernels: all match\n");
    return 0;
}
