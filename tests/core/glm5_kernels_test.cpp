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
    k::kda_step(dS, dout, dg, dbeta, do_, H, T, dmid, nullptr);
    // the saved first-token state is what a one-token window leaves
    float* dout1;
    cudaMalloc(&dout1, (size_t) 3 * D * 4);
    k::kda_conv(dq, dk, dv, dst0, dwq, dwk, dwv, dout1, D, 1, nullptr, nullptr);
    k::kda_step(dS0, dout, dg, dbeta, do_ + 0, H, 1, nullptr, nullptr);
    check("kda_step mid state", down(dmid, (size_t) H * DK * DK), down(dS0, (size_t) H * DK * DK), 0);
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
    const int n_in = 4096, n_out = 288, T = 11;
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
    check("f32_matvec", down(dy, (size_t) T * n_out), ref, 1e-4);
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
        check(("mla_attend (selected cells)" + at).c_str(), down(dlat, (size_t) AT * AH * L), ref_attend(sets), 1e-4);
    }
}

static void test_indexer() {
    const int NP = 2203, T = 3, H = 32, D = k::kIdxDim, K = 512, SEL = 2048 + 8;
    auto kraw = rnd((size_t) NP * D), gate = rnd((size_t) NP * D), lw = rnd(D, 0.2f), lb = rnd(D, 0.2f),
         ape = rnd((size_t) k::kKpool * D, 0.5f);
    for (auto& v : lw)
        v += 1.f;
    float *dk = up(kraw), *dg = up(gate), *dlw = up(lw), *dlb = up(lb), *dape = up(ape);
    __half *kc, *gc;
    float *pooled, *scores;
    cudaMalloc(&kc, (size_t) NP * D * 2);
    cudaMalloc(&gc, (size_t) NP * D * 2);
    cudaMalloc(&pooled, (size_t) (NP / 4 + 1) * D * 4);
    const int stride = NP / 4 + 8;
    cudaMalloc(&scores, (size_t) T * stride * 4);
    for (int p0 = 0; p0 < NP; p0 += 8) { // the cache as decode windows would fill it
        const int n = std::min(8, NP - p0);
        k::idx_store(dk + (size_t) p0 * D, dlw, dlb, dg + (size_t) p0 * D, kc, gc, p0, n, 1e-6f, nullptr);
        k::idx_pool(kc, gc, dape, pooled, p0, n, nullptr);
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
    check("idx_pool", down(pooled, (size_t) npools * D), rp, 1e-3);

    // a window at the end of the context: scores and the selected cells
    const int pos0 = NP - T;
    auto iq = rnd((size_t) T * H * D), w = rnd((size_t) T * H);
    float *diq = up(iq), *dw = up(w);
    const float ws = 1.f / std::sqrt((float) (D * H));
    k::idx_score(diq, dw, pooled, pos0, T, H, ws, scores, stride, nullptr);
    int32_t *sel, *nsel;
    cudaMalloc(&sel, (size_t) T * SEL * 4);
    cudaMalloc(&nsel, T * 4);
    k::idx_select(scores, stride, pos0, T, K, sel, nsel, SEL, nullptr);
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
    check("idx_score", got_sc, ref_sc, 1e-4);
    failures += sel_bad != 0;
    std::printf("%-28s %s (%d, %d, %d cells)\n", "idx_select", sel_bad ? "FAIL" : "ok", gn[0], gn[1], gn[2]);
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
    } shapes[] = {{4096, 1000}, {1536, 515}, {8192, 520}, {16384, 513}};
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

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "bench")
        return bench();
    test_hc();
    test_kda();
    test_heads_and_attention();
    test_route();
    test_rms_norm();
    test_f32_matvec();
    test_staged_mmvq(14);
    test_staged_mmvq(8);
    test_narrow();
    test_indexer();
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(failures ? "%d FAILED\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
