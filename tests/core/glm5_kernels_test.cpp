// tests/core/glm5_kernels_test.cpp - GLM-5.3-Flash's GPU kernels (src/core/glm5.cu) against plain CPU code that
// follows llama.cpp's glm5next graph, on random inputs.  No model needed.
#include "guild/core/glm5_kernels.hpp"
#include "guild/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = guild::core::glm5k;

static std::mt19937 rng(1234);
static std::vector<float> rnd(size_t n, float s = 1.f) {
    std::normal_distribution<float> d(0.f, s);
    std::vector<float> v(n);
    for (auto& x : v)
        x = d(rng);
    return v;
}
template <class T> static T* up(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> static std::vector<T> down(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaDeviceSynchronize();
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}
static int failures = 0;
static void check(const char* what, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
    double worst = 0, scale = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        worst = std::max(worst, (double) std::fabs(got[i] - ref[i]));
        scale = std::max(scale, (double) std::fabs(ref[i]));
    }
    const bool ok = worst <= tol * std::max(1.0, scale);
    failures += !ok;
    std::printf("%-28s max |diff| %.3g (largest |ref| %.3g) %s\n", what, worst, scale, ok ? "ok" : "FAIL");
}
// the same floats to the bit
static void same_attend(const std::string& what, const std::vector<float>& a, const std::vector<float>& b) {
    const bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    failures += !same;
    std::printf("%-44s %s\n", what.c_str(), same ? "ok (bitwise)" : "FAIL (bits differ)");
}

static float sigm(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static void test_hc() {
    const int N = 4096, T = 3, iters = 20;
    const float eps = 1e-6f, rms = 1e-5f;
    auto R = rnd((size_t) T * 4 * N), mixes = rnd((size_t) T * 24), scale = rnd(3, 0.5f), base = rnd(24, 0.5f),
         nw = rnd(N, 0.2f), y = rnd((size_t) T * N);
    for (auto& x : nw)
        x += 1.f;
    float *dR = up(R), *dm = up(mixes), *ds = up(scale), *db = up(base), *dn = up(nw), *dy = up(y);
    float *dpost, *dcomb, *dxn, *dout;
    cudaMalloc(&dpost, T * 4 * 4);
    cudaMalloc(&dcomb, T * 16 * 4);
    cudaMalloc(&dxn, (size_t) T * N * 4);
    cudaMalloc(&dout, (size_t) T * N * 4);
    k::hc_pre(dm, ds, db, dR, dn, dpost, dcomb, dxn, N, T, iters, eps, rms, nullptr);
    auto xn = down(dxn, (size_t) T * N), post = down(dpost, T * 4), comb = down(dcomb, T * 16);
    k::hc_post(dy, dR, dpost, dcomb, N, T, nullptr);
    auto R2 = down(dR, (size_t) T * 4 * N);
    k::hc_mean_norm(dR, dn, dout, N, T, rms, nullptr);
    auto mean = down(dout, (size_t) T * N);

    std::vector<float> rxn((size_t) T * N), rpost(T * 4), rcomb(T * 16), rR2(R.size()), rmean((size_t) T * N);
    for (int t = 0; t < T; ++t) {
        const float* m = &mixes[(size_t) t * 24];
        float pre[4], c[16];
        for (int h = 0; h < 4; ++h) {
            pre[h] = sigm(m[h] * scale[0] + base[h]) + eps;
            rpost[t * 4 + h] = 2.f * sigm(m[4 + h] * scale[1] + base[4 + h]);
        }
        for (int i = 0; i < 16; ++i)
            c[i] = m[8 + i] * scale[2] + base[8 + i];
        for (int src = 0; src < 4; ++src) { // softmax over dst (ne0)
            float mx = -1e30f, s = 0;
            for (int dst = 0; dst < 4; ++dst)
                mx = std::max(mx, c[dst + 4 * src]);
            for (int dst = 0; dst < 4; ++dst)
                s += (c[dst + 4 * src] = std::exp(c[dst + 4 * src] - mx));
            for (int dst = 0; dst < 4; ++dst)
                c[dst + 4 * src] = c[dst + 4 * src] / s + eps;
        }
        auto cols = [&] {
            for (int dst = 0; dst < 4; ++dst) {
                float s = eps;
                for (int src = 0; src < 4; ++src)
                    s += c[dst + 4 * src];
                for (int src = 0; src < 4; ++src)
                    c[dst + 4 * src] /= s;
            }
        };
        auto rows = [&] {
            for (int src = 0; src < 4; ++src) {
                float s = eps;
                for (int dst = 0; dst < 4; ++dst)
                    s += c[dst + 4 * src];
                for (int dst = 0; dst < 4; ++dst)
                    c[dst + 4 * src] /= s;
            }
        };
        cols();
        for (int i = 1; i < iters; ++i) {
            rows();
            cols();
        }
        std::copy(c, c + 16, &rcomb[t * 16]);
        std::vector<float> x(N);
        double ss = 0;
        for (int i = 0; i < N; ++i) {
            float v = 0;
            for (int h = 0; h < 4; ++h)
                v += R[((size_t) t * 4 + h) * N + i] * pre[h];
            x[i] = v;
            ss += (double) v * v;
        }
        const float inv = 1.f / std::sqrt((float) (ss / N) + rms);
        for (int i = 0; i < N; ++i)
            rxn[(size_t) t * N + i] = x[i] * inv * nw[i];
        for (int i = 0; i < N; ++i)
            for (int dst = 0; dst < 4; ++dst) {
                float v = y[(size_t) t * N + i] * rpost[t * 4 + dst];
                for (int src = 0; src < 4; ++src)
                    v += R[((size_t) t * 4 + src) * N + i] * c[dst + 4 * src];
                rR2[((size_t) t * 4 + dst) * N + i] = v;
            }
        double s2 = 0;
        for (int i = 0; i < N; ++i) {
            float v = 0;
            for (int h = 0; h < 4; ++h)
                v += rR2[((size_t) t * 4 + h) * N + i];
            x[i] = v / 4;
            s2 += (double) x[i] * x[i];
        }
        const float inv2 = 1.f / std::sqrt((float) (s2 / N) + rms);
        for (int i = 0; i < N; ++i)
            rmean[(size_t) t * N + i] = x[i] * inv2 * nw[i];
    }
    check("hc_pre post", post, rpost, 1e-5);
    check("hc_pre comb (Sinkhorn)", comb, rcomb, 1e-5);
    check("hc_pre xn", xn, rxn, 1e-4);
    check("hc_post", R2, rR2, 1e-4);
    check("hc_mean_norm", mean, rmean, 1e-4);
}

// hc_mix + hc_pre_mixed (the fused pre-mix) against hc_pre given mixes = W . rms_norm(R) computed in double
static void test_hc_mix() {
    const int N = 4096, K = 4 * N, T = 3, iters = 20;
    const float eps = 1e-6f, rms = 1e-5f;
    std::uniform_int_distribution<int> qd(-127, 127);
    std::vector<uint8_t> W((size_t) 24 * K / 32 * 34);
    std::vector<float> Wf((size_t) 24 * K);
    for (size_t b = 0; b < (size_t) 24 * K / 32; ++b) {
        const __half d = __float2half(0.0005f * (float) (1 + b % 9));
        std::memcpy(&W[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) {
            const int8_t q = (int8_t) qd(rng);
            W[b * 34 + 2 + i] = (uint8_t) q;
            Wf[b * 32 + i] = __half2float(d) * q;
        }
    }
    auto R = rnd((size_t) T * K), scale = rnd(3, 0.5f), base = rnd(24, 0.5f), nw = rnd(N, 0.2f);
    for (auto& x : nw)
        x += 1.f;
    std::vector<float> mixes((size_t) T * 24);
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int i = 0; i < K; ++i)
            ss += (double) R[(size_t) t * K + i] * R[(size_t) t * K + i];
        const double inv = 1.0 / std::sqrt(ss / K + rms);
        for (int r = 0; r < 24; ++r) {
            double v = 0;
            for (int i = 0; i < K; ++i)
                v += (double) Wf[(size_t) r * K + i] * R[(size_t) t * K + i];
            mixes[(size_t) t * 24 + r] = (float) (v * inv);
        }
    }
    uint8_t* dW = up(W);
    float *dR = up(R), *dm = up(mixes), *ds = up(scale), *db = up(base), *dn = up(nw), *dpart;
    cudaMalloc(&dpart, k::hc_mix_floats(T) * 4);
    float *p1, *c1, *x1, *p2, *c2, *x2;
    for (float** q : {&p1, &p2})
        cudaMalloc(q, T * 4 * 4);
    for (float** q : {&c1, &c2})
        cudaMalloc(q, T * 16 * 4);
    for (float** q : {&x1, &x2})
        cudaMalloc(q, (size_t) T * N * 4);
    k::hc_pre(dm, ds, db, dR, dn, p1, c1, x1, N, T, iters, eps, rms, nullptr);
    k::hc_mix(dW, dR, dpart, N, T, nullptr);
    k::hc_pre_mixed(dpart, ds, db, dR, dn, p2, c2, x2, N, T, iters, eps, rms, nullptr);
    check("hc_mix + hc_pre: post", down(p2, T * 4), down(p1, T * 4), 1e-5);
    check("hc_mix + hc_pre: comb", down(c2, T * 16), down(c1, T * 16), 1e-5);
    check("hc_mix + hc_pre: x", down(x2, (size_t) T * N), down(x1, (size_t) T * N), 1e-5);
}

static void test_kda() {
    const int H = 4, DK = 128, D = H * DK, T = 3;
    auto xq = rnd((size_t) T * D), xk = rnd((size_t) T * D), xv = rnd((size_t) T * D), st0 = rnd((size_t) 3 * D * 3),
         wq = rnd((size_t) D * 4, 0.5f), wk = rnd((size_t) D * 4, 0.5f), wv = rnd((size_t) D * 4, 0.5f),
         fpre = rnd((size_t) T * D), dt = rnd(D, 0.3f), bpre = rnd((size_t) T * H), gate = rnd((size_t) T * D),
         nw = rnd(DK, 0.2f), S0 = rnd((size_t) H * DK * DK, 0.1f);
    std::vector<float> a(H);
    for (auto& v : a)
        v = -std::exp(rnd(1, 0.5f)[0]);
    for (auto& v : nw)
        v += 1.f;
    const float lb = -5.f;
    float *dq = up(xq), *dk = up(xk), *dv = up(xv), *dst = up(st0), *dwq = up(wq), *dwk = up(wk), *dwv = up(wv),
          *dfp = up(fpre), *ddt = up(dt), *da = up(a), *dbp = up(bpre), *dgate = up(gate), *dnw = up(nw), *dS = up(S0);
    float *dout, *dg, *dbeta, *do_, *dy;
    cudaMalloc(&dout, (size_t) T * 3 * D * 4);
    cudaMalloc(&dg, (size_t) T * D * 4);
    cudaMalloc(&dbeta, (size_t) T * H * 4);
    cudaMalloc(&do_, (size_t) T * D * 4);
    cudaMalloc(&dy, (size_t) T * D * 4);
    float *dmid, *dcmid;
    cudaMalloc(&dmid, (size_t) H * DK * DK * 4);
    cudaMalloc(&dcmid, (size_t) 3 * D * 3 * 4);
    float* dS0 = up(S0);
    float* dst0 = up(st0);
    k::kda_conv(dq, dk, dv, dst, dwq, dwk, dwv, dout, D, T, dcmid, nullptr);
    k::kda_gates(dfp, ddt, da, dg, dbp, dbeta, D, H, T, lb, nullptr);
    k::kda_step(dS, dS, dout, dg, dbeta, do_, H, T, dmid, nullptr);
    // the saved first-token state is what a one-token window leaves
    float* dout1;
    cudaMalloc(&dout1, (size_t) 3 * D * 4);
    k::kda_conv(dq, dk, dv, dst0, dwq, dwk, dwv, dout1, D, 1, nullptr, nullptr);
    k::kda_step(dS0, dS0, dout, dg, dbeta, do_ + 0, H, 1, nullptr, nullptr);
    check("kda_step mid state", down(dmid, (size_t) H * DK * DK), down(dS0, (size_t) H * DK * DK), 0);
    { // a verify window split in two (glm5.cpp's kda_mid1): token 0's state to a mid copy only, token 1 from it,
      // against both tokens at once: the same state and outputs
        float *dSx = up(S0), *dSy = up(S0), *dm2, *o2a, *o2b;
        cudaMalloc(&dm2, (size_t) H * DK * DK * 4);
        cudaMalloc(&o2a, (size_t) 2 * D * 4);
        cudaMalloc(&o2b, (size_t) 2 * D * 4);
        k::kda_step(dSx, dSx, dout, dg, dbeta, o2a, H, 2, nullptr, nullptr);
        k::kda_step(dSy, nullptr, dout, dg, dbeta, o2b, H, 1, dm2, nullptr);
        check("kda_step split window: S_in left alone", down(dSy, (size_t) H * DK * DK), S0, 0);
        k::kda_step(dm2, dSy, dout + 3 * D, dg + D, dbeta + H, o2b + D, H, 1, nullptr, nullptr);
        check("kda_step split window: state", down(dSy, (size_t) H * DK * DK), down(dSx, (size_t) H * DK * DK), 0);
        check("kda_step split window: outputs", down(o2b, (size_t) 2 * D), down(o2a, (size_t) 2 * D), 0);
        for (float* p : {dSx, dSy, dm2, o2a, o2b})
            cudaFree(p);
    }
    check("kda_conv mid state", down(dcmid, (size_t) 3 * D * 3), down(dst0, (size_t) 3 * D * 3), 0);
    k::kda_out(do_, dgate, dnw, dy, H, T, 1e-5f, nullptr);
    auto conv = down(dout, (size_t) T * 3 * D), st1 = down(dst, (size_t) 3 * D * 3), g = down(dg, (size_t) T * D),
         y = down(dy, (size_t) T * D), S1 = down(dS, (size_t) H * DK * DK);

    // reference
    std::vector<float> rconv((size_t) T * 3 * D), rst = st0, rg((size_t) T * D), rbeta((size_t) T * H),
                                                  ry((size_t) T * D), S = S0;
    for (int c = 0; c < 3 * D; ++c) {
        const int part = c / D, cc = c % D;
        const auto& w = part == 0 ? wq : part == 1 ? wk : wv;
        const auto& x = part == 0 ? xq : part == 1 ? xk : xv;
        float s[3] = {rst[c * 3], rst[c * 3 + 1], rst[c * 3 + 2]};
        for (int t = 0; t < T; ++t) {
            const float xi = x[(size_t) t * D + cc];
            const float v = w[cc * 4] * s[0] + w[cc * 4 + 1] * s[1] + w[cc * 4 + 2] * s[2] + w[cc * 4 + 3] * xi;
            rconv[(size_t) t * 3 * D + c] = v * sigm(v);
            s[0] = s[1];
            s[1] = s[2];
            s[2] = xi;
        }
        rst[c * 3] = s[0];
        rst[c * 3 + 1] = s[1];
        rst[c * 3 + 2] = s[2];
    }
    for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < 2 * H; ++hh) {
            float* v = &rconv[(size_t) t * 3 * D + hh * DK];
            double ss = 0;
            for (int i = 0; i < DK; ++i)
                ss += (double) v[i] * v[i];
            const float n = std::max((float) std::sqrt(ss), 1e-6f);
            for (int i = 0; i < DK; ++i)
                v[i] /= n;
        }
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < D; ++c)
            rg[(size_t) t * D + c] = lb * sigm(-(fpre[(size_t) t * D + c] + dt[c]) * a[c / DK]);
        for (int h = 0; h < H; ++h)
            rbeta[t * H + h] = sigm(bpre[t * H + h]);
    }
    std::vector<float> o((size_t) T * D);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            const float* q = &rconv[(size_t) t * 3 * D + h * DK];
            const float* kk = &rconv[(size_t) t * 3 * D + D + h * DK];
            const float* v = &rconv[(size_t) t * 3 * D + 2 * D + h * DK];
            float* Sh = &S[(size_t) h * DK * DK]; // [key i][value j]
            for (int i = 0; i < DK; ++i)
                for (int j = 0; j < DK; ++j)
                    Sh[i * DK + j] *= std::exp(rg[(size_t) t * D + h * DK + i]);
            for (int j = 0; j < DK; ++j) {
                float sk = 0;
                for (int i = 0; i < DK; ++i)
                    sk += Sh[i * DK + j] * kk[i];
                const float d = (v[j] - sk) * rbeta[t * H + h];
                for (int i = 0; i < DK; ++i)
                    Sh[i * DK + j] += kk[i] * d;
            }
            for (int j = 0; j < DK; ++j) {
                float s = 0;
                for (int i = 0; i < DK; ++i)
                    s += Sh[i * DK + j] * q[i] / std::sqrt((float) DK);
                o[(size_t) t * D + h * DK + j] = s;
            }
            double ss = 0;
            for (int j = 0; j < DK; ++j)
                ss += (double) o[(size_t) t * D + h * DK + j] * o[(size_t) t * D + h * DK + j];
            const float inv = 1.f / std::sqrt((float) (ss / DK) + 1e-5f);
            for (int j = 0; j < DK; ++j) {
                const size_t idx = (size_t) t * D + h * DK + j;
                ry[idx] = o[idx] * inv * nw[j] * sigm(gate[idx]);
            }
        }
    check("kda_conv (+ L2 norm)", conv, rconv, 1e-5);
    check("kda_conv state", st1, rst, 0);
    check("kda_gates g", g, rg, 1e-5);
    check("kda_step state", S1, S, 1e-4);
    check("kda_out", y, ry, 1e-4);
}

static void test_rms_norm() {
    for (const int n : {512, 4096, 16384}) {
        const int T = 3;
        auto x = rnd((size_t) T * n), w = rnd(n, 0.3f);
        float *dx = up(x), *dw = up(w), *dy;
        cudaMalloc(&dy, (size_t) T * n * 4);
        for (const bool weighted : {false, true}) {
            k::rms_norm(dx, weighted ? dw : nullptr, dy, n, T, 1e-6f, nullptr);
            std::vector<float> ref((size_t) T * n);
            for (int t = 0; t < T; ++t) {
                double ss = 0;
                for (int i = 0; i < n; ++i)
                    ss += (double) x[(size_t) t * n + i] * x[(size_t) t * n + i];
                const double inv = 1.0 / std::sqrt(ss / n + 1e-6);
                for (int i = 0; i < n; ++i)
                    ref[(size_t) t * n + i] = (float) (x[(size_t) t * n + i] * inv * (weighted ? w[i] : 1.f));
            }
            const std::string what = "rms_norm " + std::to_string(n) + (weighted ? " weighted" : "");
            check(what.c_str(), down(dy, (size_t) T * n), ref, 1e-5);
        }
    }
}

static void test_f32_matvec() {
    for (const int T : {11, 1, 2}) {
        const int n_in = 4096, n_out = 288;
        auto W = rnd((size_t) n_out * n_in, 0.05f), x = rnd((size_t) T * n_in);
        float *dW = up(W), *dx = up(x), *dy;
        cudaMalloc(&dy, (size_t) T * n_out * 4);
        k::f32_matvec(dW, dx, dy, n_in, n_out, T, nullptr);
        std::vector<float> ref((size_t) T * n_out);
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < n_out; ++r) {
                double s = 0;
                for (int i = 0; i < n_in; ++i)
                    s += (double) W[(size_t) r * n_in + i] * x[(size_t) t * n_in + i];
                ref[(size_t) t * n_out + r] = (float) s;
            }
        check(("f32_matvec T " + std::to_string(T)).c_str(), down(dy, (size_t) T * n_out), ref, 1e-4);
    }
}

static void test_heads_and_attention() {
    for (const int T : {11, 1, 2}) // two token tiles, a decode step, a verify window
        for (const int n_in : {256, 512, 64}) {
            // q8_0_heads: H heads of [n_out][n_in] Q8_0
            const int H = 8, n_out = 512;
            std::vector<uint8_t> W((size_t) H * n_out * (n_in / 32) * 34);
            std::vector<float> Wf((size_t) H * n_out * n_in);
            std::uniform_int_distribution<int> qd(-127, 127);
            for (size_t r = 0; r < (size_t) H * n_out; ++r)
                for (int b = 0; b < n_in / 32; ++b) {
                    uint8_t* blk = &W[(r * (n_in / 32) + b) * 34];
                    const __half d = __float2half(0.01f + 0.001f * (float) (b % 7));
                    std::memcpy(blk, &d, 2);
                    for (int i = 0; i < 32; ++i) {
                        const int8_t q = (int8_t) qd(rng);
                        blk[2 + i] = (uint8_t) q;
                        Wf[r * n_in + b * 32 + i] = __half2float(d) * q;
                    }
                }
            auto x = rnd((size_t) T * H * n_in);
            std::vector<uint8_t> Ws(W.size());
            k::q8_0_split(W.data(), Ws.data(), (int64_t) H * n_out, n_in);
            uint8_t* dW = up(Ws);
            float* dx = up(x);
            float* dyv;
            cudaMalloc(&dyv, (size_t) T * H * n_out * 4);
            k::q8_0_heads(dW, dx, dyv, H, n_in, n_out, T, nullptr);
            auto yv = down(dyv, (size_t) T * H * n_out);
            std::vector<float> ry((size_t) T * H * n_out);
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < H; ++h)
                    for (int r = 0; r < n_out; ++r) {
                        double s = 0;
                        for (int i = 0; i < n_in; ++i)
                            s += (double) Wf[((size_t) h * n_out + r) * n_in + i] * x[((size_t) t * H + h) * n_in + i];
                        ry[((size_t) t * H + h) * n_out + r] = (float) s;
                    }
            check(("q8_0_heads n_in " + std::to_string(n_in) + " T " + std::to_string(T)).c_str(), yv, ry, 1e-4);
        }
    for (const int AT : {20, 1, 2}) { // a prompt window (2 splits), a decode step, a verify window

        // MLA attention: 64 heads over a latent of 512, causal for a window of 3 at position 37, then a selection
        const int AH = 64, L = 512, pos0 = 37 + 600 * (AT < 4), cells = pos0 + AT;
        auto q = rnd((size_t) AT * AH * L), kvf = rnd((size_t) cells * L);
        std::vector<__half> kvh(kvf.size());
        for (size_t i = 0; i < kvf.size(); ++i) {
            kvh[i] = __float2half(kvf[i]);
            kvf[i] = __half2float(kvh[i]);
        }
        float* dq = up(q);
        __half* dc = up(kvh);
        float *dlat, *dpart;
        cudaMalloc(&dlat, (size_t) AT * AH * L * 4);
        cudaMalloc(&dpart, (size_t) k::attend_scratch_floats(cells, AH, L, AT) * 4);
        const float scale = 1.f / 16.f;
        auto ref_attend = [&](const std::vector<std::vector<int>>& sets) {
            std::vector<float> out((size_t) AT * AH * L);
            for (int t = 0; t < AT; ++t)
                for (int h = 0; h < AH; ++h) {
                    const float* qh = &q[((size_t) t * AH + h) * L];
                    std::vector<double> s;
                    for (int c : sets[t]) {
                        double v = 0;
                        for (int d = 0; d < L; ++d)
                            v += (double) qh[d] * kvf[(size_t) c * L + d];
                        s.push_back(v * scale);
                    }
                    const double mx = *std::max_element(s.begin(), s.end());
                    double den = 0;
                    for (auto& v : s)
                        den += (v = std::exp(v - mx));
                    for (int d = 0; d < L; ++d) {
                        double v = 0;
                        for (size_t i = 0; i < s.size(); ++i)
                            v += s[i] * kvf[(size_t) sets[t][i] * L + d];
                        out[((size_t) t * AH + h) * L + d] = (float) (v / den);
                    }
                }
            return out;
        };
        k::mla_attend(dq, dc, pos0, AH, L, AT, scale, nullptr, nullptr, 0, dlat, dpart, nullptr);
        const auto causal_new = down(dlat, (size_t) AT * AH * L);
        k::mla_attend(dq, dc, pos0, AH, L, AT, scale, nullptr, nullptr, 0, dlat, dpart, nullptr, 0, true);
        same_attend("mla_attend (causal), legacy" + std::string(" T ") + std::to_string(AT), causal_new,
                    down(dlat, (size_t) AT * AH * L));
        k::mla_attend(dq, dc, pos0, AH, L, AT, scale, nullptr, nullptr, 0, dlat, dpart, nullptr);
        std::vector<std::vector<int>> causal(AT);
        for (int t = 0; t < AT; ++t)
            for (int c = 0; c <= pos0 + t; ++c)
                causal[t].push_back(c);
        const std::string at = " T " + std::to_string(AT);
        check(("mla_attend (causal)" + at).c_str(), down(dlat, (size_t) AT * AH * L), ref_attend(causal), 1e-4);
        const int n_sel = 21;
        std::vector<int32_t> sel((size_t) AT * n_sel);
        std::vector<std::vector<int>> sets(AT);
        for (int t = 0; t < AT; ++t)
            for (int i = 0; i < n_sel; ++i) {
                const int c = (i * 7 + t) % cells;
                sel[(size_t) t * n_sel + i] = c;
                sets[t].push_back(c);
            }
        int32_t* dsel = up(sel);
        int32_t* dnsel = up(std::vector<int32_t>(AT, n_sel));
        k::mla_attend(dq, dc, pos0, AH, L, AT, scale, dsel, dnsel, n_sel, dlat, dpart, nullptr);
        const auto sel_new = down(dlat, (size_t) AT * AH * L);
        check(("mla_attend (selected cells)" + at).c_str(), sel_new, ref_attend(sets), 1e-4);
        k::mla_attend(dq, dc, pos0, AH, L, AT, scale, dsel, dnsel, n_sel, dlat, dpart, nullptr, 0, true);
        same_attend("mla_attend (selected cells), legacy" + at, sel_new, down(dlat, (size_t) AT * AH * L));
    }
}

// a window's selected rows, some in VRAM and the rest in mapped host memory, gathered kStage queries at a time (each
// query's rows, or the union of the queries' rows) and attended with the window's split (as glm5.cpp's long context
// does) against the whole window over one VRAM cache: the same bits
static void test_attend_staged() {
    const int AH = 64, L = 512, cells = 6000, near = 2304, kStage = 8, n_max = 2052;
    std::vector<__half> kvh((size_t) cells * L);
    {
        const auto kvf = rnd(kvh.size());
        for (size_t i = 0; i < kvh.size(); ++i)
            kvh[i] = __float2half(kvf[i]);
    }
    __half* dc = up(kvh);
    void* far = nullptr;
    void* dfar = nullptr;
    cudaHostAlloc(&far, (size_t) (cells - near) * L * 2, cudaHostAllocMapped);
    std::memcpy(far, kvh.data() + (size_t) near * L, (size_t) (cells - near) * L * 2);
    cudaHostGetDevicePointer(&dfar, far, 0);
    for (const int AT : {1, 2, 3, 20, 37}) {
        const int64_t pos0 = cells - AT;
        auto q = rnd((size_t) AT * AH * L);
        std::vector<int32_t> sel((size_t) AT * n_max), nsel(AT);
        std::uniform_int_distribution<int> any(0, cells - AT - 1);
        for (int t = 0; t < AT; ++t) {
            nsel[t] = n_max - 3 * t; // the counts differ between queries
            for (int i = 0; i < nsel[t]; ++i)
                sel[(size_t) t * n_max + i] = i < 4 ? (int) pos0 + t - i : any(rng);
        }
        std::vector<int32_t> ident((size_t) kStage * n_max);
        for (size_t i = 0; i < ident.size(); ++i)
            ident[i] = (int32_t) i;
        float* dq = up(q);
        int32_t *dsel = up(sel), *dnsel = up(nsel), *did = up(ident);
        float *dlat, *dpart;
        __half* stage;
        cudaMalloc(&dlat, (size_t) AT * AH * L * 4);
        cudaMalloc(&dpart, (size_t) k::attend_scratch_floats(cells, AH, L, AT) * 4);
        cudaMalloc(&stage, (size_t) kStage * n_max * L * 2);
        const int64_t words = k::union_mask_words(cells);
        uint32_t* mask;
        int32_t *base, *list, *usel;
        cudaMalloc(&mask, (size_t) words * 4);
        cudaMemset(mask, 0, (size_t) words * 4);
        cudaMalloc(&base, (size_t) (words + 1) * 4);
        cudaMalloc(&list, (size_t) kStage * n_max * 4);
        cudaMalloc(&usel, (size_t) kStage * n_max * 4);
        k::mla_attend(dq, dc, pos0, AH, L, AT, 0.05f, dsel, dnsel, n_max, dlat, dpart, nullptr);
        const auto whole = down(dlat, (size_t) AT * AH * L);
        for (const bool uni : {false, true}) {
            cudaMemset(dlat, 0, (size_t) AT * AH * L * 4);
            for (int q0 = 0; q0 < AT; q0 += kStage) {
                const int nq = std::min(kStage, AT - q0);
                const int32_t* qsel = dsel + (size_t) q0 * n_max;
                if (uni)
                    k::kv_gather_union(dc, near, dfar, qsel, dnsel + q0, n_max, nq, cells, mask, base, list, usel,
                                       stage, kStage * n_max, L, nullptr);
                else
                    k::kv_gather(dc, near, dfar, qsel, dnsel + q0, n_max, nq, stage, L, nullptr);
                k::mla_attend(dq + (size_t) q0 * AH * L, stage, pos0 + q0, AH, L, nq, 0.05f, uni ? usel : did,
                              dnsel + q0, n_max, dlat + (size_t) q0 * AH * L, dpart, nullptr, AT);
            }
            same_attend(std::string("mla_attend, rows ") + (uni ? "of the union " : "") +
                            "gathered from VRAM and host T " + std::to_string(AT),
                        whole, down(dlat, (size_t) AT * AH * L));
        }
        for (void* p : {(void*) dq, (void*) dsel, (void*) dnsel, (void*) did, (void*) dlat, (void*) dpart,
                        (void*) stage, (void*) mask, (void*) base, (void*) list, (void*) usel})
            cudaFree(p);
    }
    cudaFree(dc);
    cudaFreeHost(far);
}

// one-token queries through the row cache (kv_cache_rows): eight steps whose selections overlap (each replaces ~10%
// of the last one's cells), some cells written again (and forgotten) between steps, against attention over the whole
// cache in VRAM: the same bits every step
static void test_row_cache() {
    const int AH = 64, L = 512, cells = 6000, near = 2304, n_max = 2052, n = 2048;
    std::vector<__half> kvh((size_t) cells * L);
    auto fill = [&](int c) {
        const auto r = rnd(L);
        for (int d = 0; d < L; ++d)
            kvh[(size_t) c * L + d] = __float2half(r[(size_t) d]);
    };
    for (int c = 0; c < cells; ++c)
        fill(c);
    __half *dc = nullptr, *dnear = nullptr, *rows = nullptr;
    cudaMalloc(&dc, kvh.size() * 2);
    cudaMalloc(&dnear, (size_t) near * L * 2);
    cudaMalloc(&rows, (size_t) n_max * L * 2);
    void *far = nullptr, *dfar = nullptr;
    cudaHostAlloc(&far, (size_t) (cells - near) * L * 2, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&dfar, far, 0);
    auto upload = [&]() {
        cudaMemcpy(dc, kvh.data(), kvh.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dnear, kvh.data(), (size_t) near * L * 2, cudaMemcpyHostToDevice);
        std::memcpy(far, kvh.data() + (size_t) near * L, (size_t) (cells - near) * L * 2);
    };
    upload();
    int32_t *slot_cell, *cell_slot, *rsel, *misses, *dsel, *dn;
    cudaMalloc(&slot_cell, n_max * 4);
    cudaMalloc(&cell_slot, cells * 4);
    cudaMalloc(&rsel, n_max * 4);
    cudaMalloc(&misses, (2 * n_max + 1) * 4);
    cudaMalloc(&dsel, n_max * 4);
    cudaMalloc(&dn, 4);
    cudaMemset(slot_cell, 0xff, n_max * 4);
    cudaMemset(cell_slot, 0xff, cells * 4);
    float *dlat, *dpart;
    cudaMalloc(&dlat, (size_t) AH * L * 4);
    cudaMalloc(&dpart, (size_t) k::attend_scratch_floats(cells, AH, L, 1) * 4);
    std::vector<int32_t> sel;
    std::vector<char> in((size_t) cells, 0);
    std::uniform_int_distribution<int> any(0, cells - 1);
    while ((int) sel.size() < n) {
        const int c = any(rng);
        if (!in[(size_t) c]) {
            in[(size_t) c] = 1;
            sel.push_back(c);
        }
    }
    int same = 0;
    for (int step = 0; step < 8; ++step) {
        if (step > 0) {
            for (int r = 0; r < n / 10; ++r) { // ~10% of the cells change
                int c;
                do
                    c = any(rng);
                while (in[(size_t) c]);
                const size_t i = (size_t) any(rng) % sel.size();
                in[(size_t) sel[i]] = 0;
                in[(size_t) c] = 1;
                sel[i] = c;
            }
            for (int r = 0; r < 40; ++r) { // and some cells are written again, selected or not
                const int c = r < 20 ? sel[(size_t) any(rng) % sel.size()] : any(rng);
                fill(c);
                k::kv_cache_forget(cell_slot, c, 1, nullptr);
            }
            upload();
        }
        cudaMemcpy(dsel, sel.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
        const int32_t nn = n;
        cudaMemcpy(dn, &nn, 4, cudaMemcpyHostToDevice);
        auto q = rnd((size_t) AH * L);
        float* dq = up(q);
        k::mla_attend(dq, dc, cells - 1, AH, L, 1, 0.05f, dsel, dn, n_max, dlat, dpart, nullptr);
        const auto want = down(dlat, (size_t) AH * L);
        k::kv_cache_rows(dnear, near, dfar, dsel, dn, n_max, rows, slot_cell, cell_slot, rsel, misses, L, nullptr);
        k::mla_attend(dq, rows, cells - 1, AH, L, 1, 0.05f, rsel, dn, n_max, dlat, dpart, nullptr);
        same += std::memcmp(want.data(), down(dlat, (size_t) AH * L).data(), want.size() * 4) == 0;
        int32_t nmiss = 0;
        cudaMemcpy(&nmiss, misses + 2 * n_max, 4, cudaMemcpyDeviceToHost);
        if (step == 7)
            std::printf("  row cache, step 8: %d of %d cells copied\n", nmiss, n);
        cudaFree(dq);
    }
    same_attend("mla_attend through the row cache, 8 steps", {(float) same}, {8.f});
    for (void* p : {(void*) dc, (void*) dnear, (void*) rows, (void*) slot_cell, (void*) cell_slot, (void*) rsel,
                    (void*) misses, (void*) dsel, (void*) dn, (void*) dlat, (void*) dpart})
        cudaFree(p);
    cudaFreeHost(far);
}

// `glm5_kernels_test attend`: mla_attend over 2,048 selected cells of a long context, legacy and new, and the
// gather of those rows from mapped host memory
static int bench_attend() {
    const int AH = 64, L = 512, n_max = 2052, n = 2048;
    const int64_t cells = 65536;
    std::vector<__half> kvh((size_t) cells * L);
    for (size_t i = 0; i < kvh.size(); ++i)
        kvh[i] = __float2half((float) ((i * 2654435761u) % 1000) * 0.001f - 0.5f);
    __half* dc = up(kvh);
    void* far = nullptr;
    void* dfar = nullptr;
    cudaHostAlloc(&far, kvh.size() * 2, cudaHostAllocMapped);
    std::memcpy(far, kvh.data(), kvh.size() * 2);
    cudaHostGetDevicePointer(&dfar, far, 0);
    cudaStream_t st;
    cudaStreamCreate(&st);
    const int64_t words = k::union_mask_words(cells);
    uint32_t* mask;
    int32_t *base, *list, *usel;
    cudaMalloc(&mask, (size_t) words * 4);
    cudaMemset(mask, 0, (size_t) words * 4);
    cudaMalloc(&base, (size_t) (words + 1) * 4);
    cudaMalloc(&list, (size_t) 8 * n_max * 4);
    cudaMalloc(&usel, (size_t) 8 * n_max * 4);
    for (const int T : {1, 2, 4, 8, 32, 128}) {
        // a prompt window's queries: each selects 2,048 cells, 1,792 of them shared by all (a pool list that moves
        // little from one token to the next), in ascending order as idx_select gives them
        std::vector<int32_t> sel((size_t) T * n_max), nsel(T, n);
        std::uniform_int_distribution<int> any(0, (int) cells - 200);
        std::vector<int32_t> common(1792);
        for (auto& c : common)
            c = any(rng);
        for (int t = 0; t < T; ++t) {
            std::vector<int32_t> s(common);
            while ((int) s.size() < n)
                s.push_back(any(rng));
            std::sort(s.begin(), s.end());
            std::copy(s.begin(), s.end(), sel.begin() + (size_t) t * n_max);
        }
        auto q = rnd((size_t) T * AH * L);
        float* dq = up(q);
        int32_t *dsel = up(sel), *dnsel = up(nsel);
        float *dlat, *dpart;
        __half* stage;
        cudaMalloc(&dlat, (size_t) T * AH * L * 4);
        cudaMalloc(&dpart, (size_t) k::attend_scratch_floats(cells, AH, L, T) * 4);
        cudaMalloc(&stage, (size_t) std::min(T, 8) * n_max * L * 2);
        cudaEvent_t a, b, c, d, f;
        for (cudaEvent_t* e : {&a, &b, &c, &d, &f})
            cudaEventCreate(e);
        const int reps = T >= 32 ? 5 : 30;
        for (int i = 0; i < 2; ++i) {
            k::mla_attend(dq, dc, cells - 100, AH, L, T, 0.05f, dsel, dnsel, n_max, dlat, dpart, st, 0, true);
            k::mla_attend(dq, dc, cells - 100, AH, L, T, 0.05f, dsel, dnsel, n_max, dlat, dpart, st);
        }
        cudaEventRecord(a, st);
        for (int i = 0; i < reps; ++i)
            k::mla_attend(dq, dc, cells - 100, AH, L, T, 0.05f, dsel, dnsel, n_max, dlat, dpart, st, 0, true);
        cudaEventRecord(b, st);
        for (int i = 0; i < reps; ++i)
            k::mla_attend(dq, dc, cells - 100, AH, L, T, 0.05f, dsel, dnsel, n_max, dlat, dpart, st);
        cudaEventRecord(c, st);
        for (int i = 0; i < reps; ++i)
            for (int q0 = 0; q0 < T; q0 += 8)
                k::kv_gather(dc, 0, dfar, dsel + (size_t) q0 * n_max, dnsel + q0, n_max, std::min(8, T - q0), stage, L,
                             st);
        cudaEventRecord(d, st);
        for (int i = 0; i < reps; ++i)
            for (int q0 = 0; q0 < T; q0 += 8)
                k::kv_gather_union(dc, 0, dfar, dsel + (size_t) q0 * n_max, dnsel + q0, n_max, std::min(8, T - q0),
                                   cells, mask, base, list, usel, stage, 8 * n_max, L, st);
        cudaEventRecord(f, st);
        cudaEventSynchronize(f);
        float ms_l = 0, ms_n = 0, ms_g = 0, ms_u = 0;
        cudaEventElapsedTime(&ms_l, a, b);
        cudaEventElapsedTime(&ms_n, b, c);
        cudaEventElapsedTime(&ms_g, c, d);
        cudaEventElapsedTime(&ms_u, d, f);
        std::printf("attend T %3d: legacy %8.1f us, new %8.1f us; rows from host: each query's %8.1f us (%.1f GB/s), "
                    "the union's %8.1f us\n",
                    T, 1000.0 * ms_l / reps, 1000.0 * ms_n / reps, 1000.0 * ms_g / reps,
                    (double) T * n * L * 2 / (1e6 * ms_g / reps), 1000.0 * ms_u / reps);
        for (void* p : {(void*) dq, (void*) dsel, (void*) dnsel, (void*) dlat, (void*) dpart, (void*) stage})
            cudaFree(p);
    }
    for (void* p : {(void*) dc, (void*) mask, (void*) base, (void*) list, (void*) usel})
        cudaFree(p);
    cudaFreeHost(far);
    return 0;
}

// the indexer's kernels against their legacy versions, which must agree to the bit: scores over `n` pools of random
// FP16 keys (`levels` > 0: every score rounded to one of that many values, for ties), then the selections
template <class T> static bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
static void test_indexer_bits(int n, int T, int levels) {
    const int H = 32, D = k::kIdxDim, K = 512, SEL = 2048 + 8, stride = n + 8;
    std::vector<__half> keys((size_t) n * D);
    {
        auto r = rnd((size_t) n * D);
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = __float2half(r[i]);
    }
    auto iq = rnd((size_t) T * H * D), w = rnd((size_t) T * H);
    __half* dk = up(keys);
    float *diq = up(iq), *dw = up(w), *s0, *s1;
    cudaMalloc(&s0, (size_t) T * stride * 4);
    cudaMalloc(&s1, (size_t) T * stride * 4);
    cudaMemset(s0, 0, (size_t) T * stride * 4);
    cudaMemset(s1, 0, (size_t) T * stride * 4);
    const int64_t pos0 = (int64_t) n * k::kKpool - T + 1; // the last query sees all n pools, the first n - 1 or so
    const float ws = 1.f / std::sqrt((float) (D * H));
    k::idx_score(diq, dw, dk, pos0, T, H, ws, s0, stride, true, nullptr);
    k::idx_score(diq, dw, dk, pos0, T, H, ws, s1, stride, false, nullptr);
    auto g0 = down(s0, (size_t) T * stride), g1 = down(s1, (size_t) T * stride);
    const bool score_ok = same_bits(g0, g1);
    if (levels > 0) { // ties: the legacy scores rounded to a few values, for both selections
        float lo = 1e30f, hi = -1e30f;
        for (float v : g0)
            lo = std::min(lo, v), hi = std::max(hi, v);
        for (float& v : g0)
            v = lo + std::floor((v - lo) / (hi - lo + 1e-30f) * levels) * (hi - lo) / levels;
        cudaMemcpy(s0, g0.data(), g0.size() * 4, cudaMemcpyHostToDevice);
    }
    int32_t *sel0, *sel1, *n0, *n1;
    cudaMalloc(&sel0, (size_t) T * SEL * 4);
    cudaMalloc(&sel1, (size_t) T * SEL * 4);
    cudaMalloc(&n0, T * 4);
    cudaMalloc(&n1, T * 4);
    cudaMemset(sel0, 0xff, (size_t) T * SEL * 4);
    cudaMemset(sel1, 0xff, (size_t) T * SEL * 4);
    k::idx_select(s0, stride, pos0, T, K, sel0, n0, SEL, true, nullptr);
    k::idx_select(s0, stride, pos0, T, K, sel1, n1, SEL, false, nullptr);
    const bool sel_ok =
        same_bits(down(sel0, (size_t) T * SEL), down(sel1, (size_t) T * SEL)) && same_bits(down(n0, T), down(n1, T));
    failures += !score_ok + !sel_ok;
    std::printf("%-28s scores %s, selection %s (%d pools, T %d%s)\n", "idx legacy == new", score_ok ? "ok" : "FAIL",
                sel_ok ? "ok" : "FAIL", n, T, levels > 0 ? (", " + std::to_string(levels) + " levels").c_str() : "");
    for (void* p : {(void*) dk, (void*) diq, (void*) dw, (void*) s0, (void*) s1, (void*) sel0, (void*) sel1, (void*) n0,
                    (void*) n1})
        cudaFree(p);
}

static void test_indexer() {
    const int NP = 2203, T = 3, H = 32, D = k::kIdxDim, K = 512, SEL = 2048 + 8;
    const int RING = (NP + 3) / 4 * 4, SMALL = 16; // the whole context, and a ring of two windows of 8
    auto kraw = rnd((size_t) NP * D), gate = rnd((size_t) NP * D), lw = rnd(D, 0.2f), lb = rnd(D, 0.2f),
         ape = rnd((size_t) k::kKpool * D, 0.5f);
    for (auto& v : lw)
        v += 1.f;
    float *dk = up(kraw), *dg = up(gate), *dlw = up(lw), *dlb = up(lb), *dape = up(ape);
    __half *kc, *gc, *kr, *gr, *pooled, *pooled_r;
    float* scores;
    cudaMalloc(&kc, (size_t) RING * D * 2);
    cudaMalloc(&gc, (size_t) RING * D * 2);
    cudaMalloc(&kr, (size_t) SMALL * D * 2);
    cudaMalloc(&gr, (size_t) SMALL * D * 2);
    cudaMalloc(&pooled, (size_t) (NP / 4 + 1) * D * 2);
    cudaMalloc(&pooled_r, (size_t) (NP / 4 + 1) * D * 2);
    const int stride = NP / 4 + 8;
    cudaMalloc(&scores, (size_t) T * stride * 4);
    for (int p0 = 0; p0 < NP; p0 += 8) { // the cache as decode windows would fill it
        const int n = std::min(8, NP - p0);
        k::idx_store(dk + (size_t) p0 * D, dlw, dlb, dg + (size_t) p0 * D, kc, gc, p0, n, RING, 1e-6f, nullptr);
        k::idx_pool(kc, gc, RING, dape, pooled, p0, n, nullptr);
        k::idx_store(dk + (size_t) p0 * D, dlw, dlb, dg + (size_t) p0 * D, kr, gr, p0, n, SMALL, 1e-6f, nullptr);
        k::idx_pool(kr, gr, SMALL, dape, pooled_r, p0, n, nullptr);
    }
    // reference caches (FP16 rounded, as stored)
    std::vector<float> rk((size_t) NP * D), rg((size_t) NP * D);
    for (int p = 0; p < NP; ++p) {
        double m = 0, v = 0;
        for (int d = 0; d < D; ++d)
            m += kraw[(size_t) p * D + d];
        m /= D;
        for (int d = 0; d < D; ++d)
            v += (kraw[(size_t) p * D + d] - m) * (kraw[(size_t) p * D + d] - m);
        v /= D;
        for (int d = 0; d < D; ++d) {
            const float x = (float) ((kraw[(size_t) p * D + d] - m) / std::sqrt(v + 1e-6)) * lw[d] + lb[d];
            rk[(size_t) p * D + d] = __half2float(__float2half(x));
            rg[(size_t) p * D + d] = __half2float(__float2half(gate[(size_t) p * D + d]));
        }
    }
    const int npools = NP / 4;
    std::vector<float> rp((size_t) npools * D);
    for (int p = 0; p < npools; ++p)
        for (int d = 0; d < D; ++d) {
            float g[4], mx = -1e30f, den = 0, v = 0;
            for (int s2 = 0; s2 < 4; ++s2)
                mx = std::max(mx, g[s2] = rg[(size_t) (p * 4 + s2) * D + d] + ape[s2 * D + d]);
            for (int s2 = 0; s2 < 4; ++s2) {
                const float e = std::exp(g[s2] - mx);
                den += e;
                v += e * rk[(size_t) (p * 4 + s2) * D + d];
            }
            rp[(size_t) p * D + d] = __half2float(__float2half(v / den)); // stored as FP16
        }
    const auto gp = down(pooled, (size_t) npools * D);
    std::vector<float> gpf(gp.size());
    for (size_t i = 0; i < gp.size(); ++i)
        gpf[i] = __half2float(gp[i]);
    check("idx_pool", gpf, rp, 1e-3);
    const bool ring_ok = same_bits(gp, down(pooled_r, (size_t) npools * D));
    failures += !ring_ok;
    std::printf("%-28s %s\n", "idx_pool, ring of 16 rows", ring_ok ? "ok (the same bits)" : "FAIL");

    // a window at the end of the context: scores and the selected cells
    const int pos0 = NP - T;
    auto iq = rnd((size_t) T * H * D), w = rnd((size_t) T * H);
    float *diq = up(iq), *dw = up(w);
    const float ws = 1.f / std::sqrt((float) (D * H));
    k::idx_score(diq, dw, pooled, pos0, T, H, ws, scores, stride, false, nullptr);
    int32_t *sel, *nsel;
    cudaMalloc(&sel, (size_t) T * SEL * 4);
    cudaMalloc(&nsel, T * 4);
    k::idx_select(scores, stride, pos0, T, K, sel, nsel, SEL, false, nullptr);
    auto gs = down(scores, (size_t) T * stride);
    auto gsel = down(sel, (size_t) T * SEL);
    auto gn = down(nsel, T);
    int sel_bad = 0;
    std::vector<float> got_sc, ref_sc;
    for (int t = 0; t < T; ++t) {
        const int q = pos0 + t, tail = (q + 1) / 4 * 4, nv = tail / 4;
        std::vector<std::pair<float, int>> sc;
        for (int p = 0; p < nv; ++p) {
            double sum = 0;
            for (int h = 0; h < H; ++h) {
                double dot = 0;
                for (int d = 0; d < D; ++d)
                    dot += (double) iq[((size_t) t * H + h) * D + d] * rp[(size_t) p * D + d];
                sum += w[(size_t) t * H + h] * ws * std::max(dot, 0.0);
            }
            sc.emplace_back((float) sum, p);
            got_sc.push_back(gs[(size_t) t * stride + p]);
            ref_sc.push_back((float) sum);
        }
        // the reference's choice from the GPU's own scores (so float rounding cannot swap a near tie)
        std::vector<std::pair<float, int>> gsc;
        for (int p = 0; p < nv; ++p)
            gsc.emplace_back(gs[(size_t) t * stride + p], p);
        std::sort(gsc.begin(), gsc.end(),
                  [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        std::vector<int> want;
        for (int i = 0; i < std::min(K, nv); ++i)
            for (int r = 0; r < 4; ++r)
                want.push_back(gsc[(size_t) i].second * 4 + r);
        for (int c = tail; c <= q; ++c)
            want.push_back(c);
        std::vector<int> got(gsel.begin() + (size_t) t * SEL, gsel.begin() + (size_t) t * SEL + gn[t]);
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        sel_bad += got != want;
    }
    // a pooled key element can round to the next FP16 value in one sum order and not the other: ~1e-4 of a score
    check("idx_score", got_sc, ref_sc, 1e-3);
    failures += sel_bad != 0;
    std::printf("%-28s %s (%d, %d, %d cells)\n", "idx_select", sel_bad ? "FAIL" : "ok", gn[0], gn[1], gn[2]);
    // the new kernels against the legacy ones: short and long contexts, a window of 1 to 4 queries, and ties
    test_indexer_bits(npools, T, 0);
    test_indexer_bits(300, 2, 0); // fewer pools than the selection: all of them
    test_indexer_bits(4100, 1, 0);
    test_indexer_bits(8197, 2, 0);
    test_indexer_bits(8197, 4, 7);
    test_indexer_bits(33000, 2, 0);
    test_indexer_bits(5000, 2, 1); // every score equal
}

// `glm5_kernels_test indexer`: the indexer's scores and selection, legacy and new, at 4K to 256K tokens of context
static int bench_indexer() {
    const int H = 32, D = k::kIdxDim, K = 512, SEL = 2048 + 8;
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (const int ctx : {4096, 16384, 32768, 131072, 262144}) {
        const int n = ctx / k::kKpool, stride = n + 8;
        std::vector<__half> keys((size_t) n * D);
        auto r = rnd((size_t) n * D);
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = __float2half(r[i]);
        __half* dk = up(keys);
        auto iq = rnd((size_t) 4 * H * D), w = rnd((size_t) 4 * H);
        float *diq = up(iq), *dw = up(w), *sc;
        cudaMalloc(&sc, (size_t) 4 * stride * 4);
        int32_t *sel, *nsel;
        cudaMalloc(&sel, (size_t) 4 * SEL * 4);
        cudaMalloc(&nsel, 4 * 4);
        for (const int T : {1, 2}) {
            const int64_t pos0 = (int64_t) ctx - T;
            for (const bool legacy : {true, false}) {
                cudaEvent_t a, b, c;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                cudaEventCreate(&c);
                const int reps = 50;
                float ms_s = 0, ms_k = 0;
                for (int i = 0; i < 3; ++i) {
                    k::idx_score(diq, dw, dk, pos0, T, H, 0.01f, sc, stride, legacy, st);
                    k::idx_select(sc, stride, pos0, T, K, sel, nsel, SEL, legacy, st);
                }
                cudaEventRecord(a, st);
                for (int i = 0; i < reps; ++i)
                    k::idx_score(diq, dw, dk, pos0, T, H, 0.01f, sc, stride, legacy, st);
                cudaEventRecord(b, st);
                for (int i = 0; i < reps; ++i)
                    k::idx_select(sc, stride, pos0, T, K, sel, nsel, SEL, legacy, st);
                cudaEventRecord(c, st);
                cudaEventSynchronize(c);
                cudaEventElapsedTime(&ms_s, a, b);
                cudaEventElapsedTime(&ms_k, b, c);
                std::printf("indexer ctx %6d T %d %-6s: idx_score %7.1f us, idx_select %7.1f us\n", ctx, T,
                            legacy ? "legacy" : "new", 1000.0 * ms_s / reps, 1000.0 * ms_k / reps);
            }
        }
        for (void* p : {(void*) dk, (void*) diq, (void*) dw, (void*) sc, (void*) sel, (void*) nsel})
            cudaFree(p);
    }
    return 0;
}

static void test_route() {
    const int E = 288, K = 8, T = 3;
    auto logits = rnd((size_t) T * E), bias = rnd(E, 0.1f);
    float *dl = up(logits), *db = up(bias), *dw;
    int32_t* di;
    cudaMalloc(&dw, T * K * 4);
    cudaMalloc(&di, T * K * 4);
    k::route(dl, db, E, K, true, 2.5f, di, dw, T, nullptr);
    auto w = down(dw, T * K);
    auto ids = down(di, T * K);
    std::vector<float> rw(T * K);
    int id_bad = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<int> o(E);
        for (int e = 0; e < E; ++e)
            o[e] = e;
        std::sort(o.begin(), o.end(),
                  [&](int a, int b) { return sigm(logits[t * E + a]) + bias[a] > sigm(logits[t * E + b]) + bias[b]; });
        float s = 0;
        for (int j = 0; j < K; ++j)
            s += sigm(logits[t * E + o[j]]);
        for (int j = 0; j < K; ++j) {
            rw[t * K + j] = sigm(logits[t * E + o[j]]) / s * 2.5f;
            id_bad += ids[t * K + j] != o[j];
        }
    }
    failures += id_bad != 0;
    std::printf("%-28s %s\n", "route ids", id_bad ? "FAIL" : "ok");
    check("route weights", w, rw, 1e-5);
}

// q6k_mmvq / q8_0_mmvq: bitwise guild::kernels::native_mmvq's one-column result for every token, on random rows
static void test_staged_mmvq(int type) {
    cudaStream_t st;
    cudaStreamCreate(&st);
    struct Shape {
        int n_in, n_out;
    } shapes[] = {{4096, 1000}, {1536, 515}, {8192, 520}, {16384, 513}, {1536, 16384}};
    std::uniform_int_distribution<int> byte(0, 255);
    for (const Shape& sh : shapes) {
        const bool q6 = type == 14;
        const int bs = q6 ? 210 : 34, nb = sh.n_in / (q6 ? 256 : 32);
        std::vector<uint8_t> W((size_t) sh.n_out * nb * bs);
        for (size_t blk = 0; blk < (size_t) sh.n_out * nb; ++blk) {
            uint8_t* b = &W[blk * bs];
            for (int i = 0; i < bs; ++i)
                b[i] = (uint8_t) byte(rng);
            const __half d = __float2half(0.001f * (float) (1 + byte(rng) % 50));
            std::memcpy(b + (q6 ? 208 : 0), &d, 2);
        }
        uint8_t* dW = up(W);
        for (const int T : {1, 2}) {
            if (!(q6 ? k::q6k_mmvq_supported(sh.n_in, sh.n_out, T) : k::q8_0_mmvq_supported(sh.n_in, sh.n_out, T)))
                continue;
            auto x = rnd((size_t) T * sh.n_in);
            float* dx = up(x);
            void* xq;
            cudaMalloc(&xq, guild::kernels::native_q8_1_bytes(sh.n_in, T));
            guild::kernels::native_quantize_q8_1(dx, xq, sh.n_in, T, st);
            float *dy, *dr;
            cudaMalloc(&dy, (size_t) T * sh.n_out * 4);
            cudaMalloc(&dr, (size_t) T * sh.n_out * 4);
            (q6 ? k::q6k_mmvq : k::q8_0_mmvq)(dW, xq, dy, sh.n_in, sh.n_out, T, st);
            const size_t qrow = guild::kernels::native_q8_1_bytes(sh.n_in, 1);
            for (int t = 0; t < T; ++t)
                guild::kernels::native_mmvq(type, dW, (const uint8_t*) xq + (size_t) t * qrow,
                                            dr + (size_t) t * sh.n_out, sh.n_in, sh.n_out, 1, st);
            const auto got = down(dy, (size_t) T * sh.n_out), ref = down(dr, (size_t) T * sh.n_out);
            const std::string what = std::string(q6 ? "q6k_mmvq " : "q8_0_mmvq ") + std::to_string(sh.n_in) + "x" +
                                     std::to_string(sh.n_out) + " T " + std::to_string(T) + " (bitwise)";
            check(what.c_str(), got, ref, 0);
            cudaFree(dx);
            cudaFree(xq);
            cudaFree(dy);
            cudaFree(dr);
        }
        cudaFree(dW);
    }
}

// q6k_mmvq_rows (the draft head over a token subset): bitwise q6k_mmvq's rows, picked in id order and shuffled
static void test_mmvq_rows() {
    cudaStream_t st;
    cudaStreamCreate(&st);
    const int n_in = 4096, n_out = 3001, nb = n_in / 256;
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> W((size_t) n_out * nb * 210);
    for (size_t blk = 0; blk < (size_t) n_out * nb; ++blk) {
        uint8_t* b = &W[blk * 210];
        for (int i = 0; i < 210; ++i)
            b[i] = (uint8_t) byte(rng);
        const __half d = __float2half(0.001f * (float) (1 + byte(rng) % 50));
        std::memcpy(b + 208, &d, 2);
    }
    uint8_t* dW = up(W);
    auto x = rnd((size_t) n_in);
    float* dx = up(x);
    void* xq;
    cudaMalloc(&xq, guild::kernels::native_q8_1_bytes(n_in, 1));
    guild::kernels::native_quantize_q8_1(dx, xq, n_in, 1, st);
    float* dfull;
    cudaMalloc(&dfull, (size_t) n_out * 4);
    k::q6k_mmvq(dW, xq, dfull, n_in, n_out, 1, st);
    const auto full = down(dfull, (size_t) n_out);
    std::vector<int32_t> sorted, shuffled;
    for (int r = 0; r < n_out; ++r)
        if (byte(rng) < 100 || r == n_out - 1)
            sorted.push_back(r);
    shuffled = sorted;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    for (const auto* ids : {&sorted, &shuffled}) {
        int32_t* dids = up(*ids);
        float* dy;
        const int n = (int) ids->size();
        cudaMalloc(&dy, (size_t) n * 4);
        k::q6k_mmvq_rows(dW, dids, n, xq, dy, n_in, st);
        std::vector<float> ref((size_t) n);
        for (int i = 0; i < n; ++i)
            ref[(size_t) i] = full[(size_t) (*ids)[(size_t) i]];
        const std::string what = "q6k_mmvq_rows " + std::to_string(n) + " of " + std::to_string(n_out) +
                                 (ids == &sorted ? " (bitwise)" : " shuffled (bitwise)");
        check(what.c_str(), down(dy, (size_t) n), ref, 0);
        // the argmax over the subset's rows mapped to its id (the first of the largest: with ascending ids, the lowest)
        int32_t* darg;
        cudaMalloc(&darg, 4);
        k::argmax(dy, n, darg, st, dids);
        const int32_t want = (*ids)[(size_t) (std::max_element(ref.begin(), ref.end()) - ref.begin())];
        int32_t got = -1;
        cudaMemcpy(&got, darg, 4, cudaMemcpyDeviceToHost);
        std::printf("%-28s %d (want %d) %s\n", "argmax over ids", got, want, got == want ? "ok" : "FAIL");
        failures += got != want;
        cudaFree(darg);
        cudaFree(dids);
        cudaFree(dy);
    }
    cudaFree(dW);
    cudaFree(dx);
    cudaFree(xq);
    cudaFree(dfull);
}

// q8_0_narrow: rows of 64 / 128 / 256 values against native_mmvq (a different order: within FP32 rounding)
static void test_narrow() {
    cudaStream_t st;
    cudaStreamCreate(&st);
    std::uniform_int_distribution<int> byte(0, 255);
    for (const int n_in : {64, 128, 256})
        for (const int T : {1, 2}) {
            const int n_out = 1000, nb = n_in / 32;
            std::vector<uint8_t> W((size_t) n_out * nb * 34);
            for (size_t blk = 0; blk < (size_t) n_out * nb; ++blk) {
                for (int i = 0; i < 34; ++i)
                    W[blk * 34 + i] = (uint8_t) byte(rng);
                const __half d = __float2half(0.001f * (float) (1 + byte(rng) % 50));
                std::memcpy(&W[blk * 34], &d, 2);
            }
            uint8_t* dW = up(W);
            auto x = rnd((size_t) T * n_in);
            float* dx = up(x);
            void* xq;
            cudaMalloc(&xq, guild::kernels::native_q8_1_bytes(n_in, T));
            guild::kernels::native_quantize_q8_1(dx, xq, n_in, T, st);
            float *dy, *dr;
            cudaMalloc(&dy, (size_t) T * n_out * 4);
            cudaMalloc(&dr, (size_t) T * n_out * 4);
            k::q8_0_narrow(dW, xq, dy, n_in, n_out, T, st);
            guild::kernels::native_mmvq(8, dW, xq, dr, n_in, n_out, T, st);
            const std::string what = "q8_0_narrow " + std::to_string(n_in) + " T " + std::to_string(T);
            check(what.c_str(), down(dy, (size_t) T * n_out), down(dr, (size_t) T * n_out), 1e-5);
            cudaFree(dW);
            cudaFree(dx);
            cudaFree(xq);
            cudaFree(dy);
            cudaFree(dr);
        }
}

// `glm5_kernels_test bench`: the GPU products at GLM-5.3-Flash's shapes (Q8_0 and Q6_K), GB/s of weights read
// `glm5_kernels_test q6k`: glm5k::q6k_mmvq (the attention's Q6_K products) at GLM-5.3-Flash's shapes, GB/s
static int bench_q6k() {
    struct Shape {
        const char* what;
        int n_in, n_out;
    } shapes[] = {{"DSA q_b 1536x16384", 1536, 16384}, {"KDA q/k/v 4096x8192", 4096, 8192},
                  {"KDA out 8192x4096", 8192, 4096},   {"DSA out 16384x4096", 16384, 4096},
                  {"head 4096x154880", 4096, 154880},  {"KDA q|k|v 4096x24576", 4096, 24576},
                  {"DSA q_b 1536x16384", 1536, 16384}}; // again, the clocks up
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (const Shape& sh : shapes) {
        const size_t wb = (size_t) sh.n_out * (sh.n_in / 256) * 210, xb = (size_t) 2 * (sh.n_in / 32) * 36;
        void *w, *x;
        float* y;
        cudaMalloc(&w, wb);
        cudaMemset(w, 0, wb);
        cudaMalloc(&x, xb);
        cudaMemset(x, 0, xb);
        cudaMalloc(&y, (size_t) 2 * sh.n_out * 4);
        for (const int T : {1, 2}) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            for (int i = 0; i < 3; ++i)
                k::q6k_mmvq(w, x, y, sh.n_in, sh.n_out, T, st);
            cudaEventRecord(a, st);
            const int reps = 50;
            for (int i = 0; i < reps; ++i)
                k::q6k_mmvq(w, x, y, sh.n_in, sh.n_out, T, st);
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("q6k_mmvq %-22s T %d: %7.1f us  %6.1f GB/s\n", sh.what, T, 1000.0 * ms / reps,
                        wb / (ms / reps * 1e-3) / 1e9);
        }
        cudaFree(w);
        cudaFree(x);
        cudaFree(y);
    }
    return 0;
}

static int bench() {
    struct Shape {
        const char* what;
        int n_in, n_out;
    } shapes[] = {
        {"hc mix 16384x24", 16384, 24},      {"KDA beta 4096x64", 4096, 64},       {"KDA f_a 4096x128", 4096, 128},
        {"KDA f_b 128x8192", 128, 8192},     {"DSA kv_a 4096x512", 4096, 512},     {"DSA q_a 4096x1536", 4096, 1536},
        {"idx q_b 1536x4096", 1536, 4096},   {"shexp gate 4096x2048", 4096, 2048}, {"shexp down 2048x4096", 2048, 4096},
        {"KDA q/k/v 4096x8192", 4096, 8192}, {"KDA out 8192x4096", 8192, 4096},    {"DSA out 16384x4096", 16384, 4096},
        {"head 4096x154880", 4096, 154880}};
    cudaStream_t st;
    cudaStreamCreate(&st);
    { // the routers: F32 4096 x 288
        float *w, *x, *y;
        cudaMalloc(&w, (size_t) 4096 * 288 * 4);
        cudaMemset(w, 0, (size_t) 4096 * 288 * 4);
        cudaMalloc(&x, 4096 * 2 * 4);
        cudaMemset(x, 0, 4096 * 2 * 4);
        cudaMalloc(&y, 288 * 2 * 4);
        for (const int T : {1, 2}) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            k::f32_matvec(w, x, y, 4096, 288, T, st);
            cudaEventRecord(a, st);
            for (int i = 0; i < 100; ++i)
                k::f32_matvec(w, x, y, 4096, 288, T, st);
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("f32_matvec 4096 x 288 T %d: %6.1f us  %6.1f GB/s\n", T, 10.0 * ms,
                        4096.0 * 288 * 4 / (ms / 100 * 1e-3) / 1e9);
        }
    }
    for (const int n_in : {256, 512}) { // attn_k_b and attn_v_b: 64 heads of [512][256] and [256][512]
        const int H = 64, n_out = 131072 / n_in;
        const size_t wb = (size_t) H * n_out * n_in / 32 * 34;
        void* w;
        float *x, *y;
        cudaMalloc(&w, wb);
        cudaMemset(w, 0, wb);
        cudaMalloc(&x, (size_t) 8 * H * n_in * 4);
        cudaMemset(x, 0, (size_t) 8 * H * n_in * 4);
        cudaMalloc(&y, (size_t) 8 * H * n_out * 4);
        for (const int T : {1, 2, 8}) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            k::q8_0_heads(w, x, y, H, n_in, n_out, T, st);
            cudaEventRecord(a, st);
            const int reps = 50;
            for (int i = 0; i < reps; ++i)
                k::q8_0_heads(w, x, y, H, n_in, n_out, T, st);
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("q8_0_heads 64 x [%d][%d] T %d: %7.1f us  %6.1f GB/s\n", n_out, n_in, T, 1000.0 * ms / reps,
                        wb / (ms / reps * 1e-3) / 1e9);
        }
        cudaFree(w);
        cudaFree(x);
        cudaFree(y);
    }
    for (int type : {8, 14}) {
        for (const Shape& sh : shapes) {
            if (type == 14 && sh.n_in % 256 != 0)
                continue;
            const size_t wb = guild::kernels::native_mmvq_weight_bytes(type, sh.n_in, sh.n_out);
            void *w, *xq;
            float *x, *y;
            cudaMalloc(&w, wb);
            cudaMemset(w, 0, wb);
            cudaMalloc(&x, (size_t) sh.n_in * 4 * 64);
            cudaMemset(x, 0, (size_t) sh.n_in * 4 * 64);
            cudaMalloc(&xq, guild::kernels::native_q8_1_bytes(sh.n_in, 64));
            cudaMalloc(&y, (size_t) sh.n_out * 4 * 64);
            guild::kernels::native_quantize_q8_1(x, xq, sh.n_in, 64, st);
            for (int ncols : {1, 2}) {
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                for (int i = 0; i < 3; ++i)
                    guild::kernels::native_mmvq(type, w, xq, y, sh.n_in, sh.n_out, ncols, st);
                cudaEventRecord(a, st);
                const int reps = 50;
                for (int i = 0; i < reps; ++i)
                    guild::kernels::native_mmvq(type, w, xq, y, sh.n_in, sh.n_out, ncols, st);
                cudaEventRecord(b, st);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                std::printf("%-6s %-22s ncols %d: %7.1f us  %6.1f GB/s\n", type == 8 ? "Q8_0" : "Q6_K", sh.what, ncols,
                            1000.0 * ms / reps, wb / (ms / reps * 1e-3) / 1e9);
                if (type == 8 && k::q8_0_narrow_supported(sh.n_in, ncols)) {
                    cudaEventRecord(a, st);
                    for (int i = 0; i < reps; ++i)
                        k::q8_0_narrow(w, xq, y, sh.n_in, sh.n_out, ncols, st);
                    cudaEventRecord(b, st);
                    cudaEventSynchronize(b);
                    cudaEventElapsedTime(&ms, a, b);
                    std::printf("%-6s %-22s ncols %d: %7.1f us  %6.1f GB/s  (glm5k, narrow)\n", "Q8_0", sh.what, ncols,
                                1000.0 * ms / reps, wb / (ms / reps * 1e-3) / 1e9);
                }
                if (type == 14 ? k::q6k_mmvq_supported(sh.n_in, sh.n_out, ncols)
                               : k::q8_0_mmvq_supported(sh.n_in, sh.n_out, ncols)) {
                    cudaEventRecord(a, st);
                    for (int i = 0; i < reps; ++i)
                        (type == 14 ? k::q6k_mmvq : k::q8_0_mmvq)(w, xq, y, sh.n_in, sh.n_out, ncols, st);
                    cudaEventRecord(b, st);
                    cudaEventSynchronize(b);
                    cudaEventElapsedTime(&ms, a, b);
                    std::printf("%-6s %-22s ncols %d: %7.1f us  %6.1f GB/s  (glm5k, staged)\n",
                                type == 14 ? "Q6_K" : "Q8_0", sh.what, ncols, 1000.0 * ms / reps,
                                wb / (ms / reps * 1e-3) / 1e9);
                }
            }
            cudaFree(w);
            cudaFree(x);
            cudaFree(xq);
            cudaFree(y);
        }
    }
    return 0;
}

// `glm5_kernels_test shapes`: guild::kernels::native_mmvq at the other models' dense shapes (Qwen3.8: Q8_0, 2560
// wide; Ornith: Q4_K / Q6_K, 2048 wide; both heads Q6_K), GB/s of weights read, 1 / 2 / 4 columns
static int bench_shapes() {
    struct Shape {
        const char* what;
        int type, n_in, n_out;
    } shapes[] = {
        {"qwen qkv", 8, 2560, 10240},      {"qwen gate", 8, 2560, 6144},       {"qwen ssm_out", 8, 6144, 2560},
        {"qwen q", 8, 2560, 12288},        {"qwen kv", 8, 2560, 512},          {"qwen shexp gu", 8, 2560, 640},
        {"qwen shexp down", 8, 640, 2560}, {"qwen hc down", 8, 10240, 320},    {"qwen hc up", 8, 320, 10240},
        {"qwen head", 14, 2560, 248320},   {"ornith qkv", 12, 2048, 8192},     {"ornith qkv q6", 14, 2048, 8192},
        {"ornith gate", 12, 2048, 4096},   {"ornith out", 12, 4096, 2048},     {"ornith kv", 12, 2048, 512},
        {"ornith v q6", 14, 2048, 512},    {"ornith shexp gu", 12, 2048, 512}, {"ornith shexp dn", 14, 512, 2048},
        {"ornith head", 14, 2048, 248320}};
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (const Shape& sh : shapes) {
        const size_t wb = guild::kernels::native_mmvq_weight_bytes(sh.type, sh.n_in, sh.n_out);
        // copies of the matrix cycled through, 64 MB or more of them, so the reads come from DRAM and not the L2
        // cache, as in a model's forward pass (a GTX 1070's L2 holds 2 MB)
        const int copies = (int) std::min<size_t>(64, (64u << 20) / wb + 1);
        std::vector<void*> w(copies);
        for (void*& p : w) {
            cudaMalloc(&p, wb);
            cudaMemset(p, 0, wb);
        }
        void* xq;
        float *x, *y;
        cudaMalloc(&x, (size_t) sh.n_in * 8 * 4);
        cudaMemset(x, 0, (size_t) sh.n_in * 8 * 4);
        cudaMalloc(&xq, guild::kernels::native_q8_1_bytes(sh.n_in, 8));
        cudaMalloc(&y, (size_t) sh.n_out * 8 * 4);
        guild::kernels::native_quantize_q8_1(x, xq, sh.n_in, 8, st);
        std::printf("%-16s %-5s %6d x %-6d", sh.what,
                    sh.type == 8    ? "Q8_0"
                    : sh.type == 12 ? "Q4_K"
                                    : "Q6_K",
                    sh.n_in, sh.n_out);
        for (int ncols : {1, 2, 3, 4, 5, 8}) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            guild::kernels::native_mmvq(sh.type, w[0], xq, y, sh.n_in, sh.n_out, ncols, st);
            const int reps = sh.n_out > 100000 ? 10 : 200;
            cudaEventRecord(a, st);
            for (int i = 0; i < reps; ++i)
                guild::kernels::native_mmvq(sh.type, w[i % copies], xq, y, sh.n_in, sh.n_out, ncols, st);
            cudaEventRecord(b, st);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("  T%d %6.1f us %4.0f", ncols, 1000.0 * ms / reps, wb / (ms / reps * 1e-3) / 1e9);
        }
        std::printf("\n");
        for (void* p : w)
            cudaFree(p);
        cudaFree(x);
        cudaFree(xq);
        cudaFree(y);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "bench")
        return bench();
    if (argc > 1 && std::string(argv[1]) == "shapes")
        return bench_shapes();
    if (argc > 1 && std::string(argv[1]) == "indexer")
        return bench_indexer();
    if (argc > 1 && std::string(argv[1]) == "attend")
        return bench_attend();
    if (argc > 1 && std::string(argv[1]) == "q6k")
        return bench_q6k();
    test_hc();
    test_hc_mix();
    test_kda();
    test_heads_and_attention();
    test_route();
    test_rms_norm();
    test_f32_matvec();
    test_staged_mmvq(14);
    test_staged_mmvq(8);
    test_mmvq_rows();
    test_narrow();
    test_indexer();
    test_attend_staged();
    test_row_cache();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(failures ? "%d FAILED\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
