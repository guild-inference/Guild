// tests/core/dflash_kernels_test.cpp - the DFlash drafter's kernels (include/guild/core/dflash.hpp) against host
// references on random data: the activation quantization, the int8 ("D8") and Q4_K products at one to eight columns,
// the top 16 with the log-sum-exp, the block attention over a context ring, and the per-head norm with RoPE.
#include "../check.hpp"
#include "guild/core/dflash.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

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
    for (const int K : {2560, 7680}) // one staged chunk, then three
        for (const int n_out : {640, 2560, 6144}) {
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
    std::printf("d8_mm: K 2560 / 7680, 640 / 2560 / 6144 rows, 1-8 columns match\n");
}

void test_k4(std::mt19937& rng, int K) { // K past 2560: the columns staged in chunks
    const int n_out = 2048;
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
            std::fprintf(stderr, "k4_mm cols %d: relative error %.3g\n", cols, e);
            CHECK(false);
        }
    }
    cudaFree(dw);
    cudaFree(y);
    std::printf("k4_mm: Q4_K 2048 x %d, 1-8 columns match ggml's dequantized rows\n", K);
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

} // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    std::mt19937 rng(7);
    test_d8(rng);
    test_k4(rng, 2560);
    test_k4(rng, 4096);
    test_topk(rng, 248320); // float4 rows
    test_topk(rng, 106299); // a draft vocabulary of an odd size
    test_attention(rng);
    test_norm_rope(rng);
    std::printf("dflash kernels: all match\n");
    return 0;
}
