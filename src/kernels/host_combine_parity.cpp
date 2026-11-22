// src/kernels/host_combine_parity.cpp - the verify window's host combine (include/guild/core/host_combine.hpp)
// against the GPU combine it replaces, BITWISE: the K expert rows weighted and summed on the CPU, then the shared
// expert added by native_moe_combine_multi at k = 1 with a weight of 1.0, against native_moe_combine_multi over the
// K rows.  Rows of plain values, exact zeros of both signs, subnormals (the kernel is built with -ftz=true),
// products that underflow or come near the float range, subnormal and zero weights and shared values; 1-4 tokens,
// widths that are and are not a multiple of the 32-column blocks.  Then the window's mapped forms
// (GUILD_COMBINE_MAPPED): the sums in mapped host memory, some made infinite, NaN or the largest float, read by the
// combine kernel itself (2) and by guild::core::combine_sums_f4 (1, widths of a multiple of 4), against the combine of
// their device copy, bitwise.
#include "guild/core/host_combine.hpp"
#include "guild/core/verify_combine.hpp"
#include "guild/kernels/native_moe.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error = (call);                                                                              \
        if (error != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error));                                        \
            return 2;                                                                                                  \
        }                                                                                                              \
    } while (0)

static int compare(int n_tok, int k, int n, unsigned seed, cudaStream_t st) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pick(0, 99);
    auto value = [&](float scale) {
        const int r = pick(rng);
        if (r < 3)
            return 0.0f;
        if (r < 6)
            return -0.0f;
        if (r < 10)
            return nd(rng) * 1e-39f; // subnormal
        if (r < 13)
            return nd(rng) * 1e-20f; // its products underflow
        if (r < 15)
            return nd(rng) * 1e18f; // its products come near the float range
        return nd(rng) * scale;
    };
    std::vector<float> parts((size_t) n_tok * k * n), w((size_t) n_tok * k), shared((size_t) n_tok * n);
    for (auto& x : parts)
        x = value(0.05f);
    for (int t = 0; t < n_tok; ++t) {
        float sum = 0.0f;
        for (int e = 0; e < k; ++e) {
            const float v = std::exp(nd(rng));
            w[(size_t) t * k + e] = v;
            sum += v;
        }
        for (int e = 0; e < k; ++e)
            w[(size_t) t * k + e] /= sum;
    }
    if (n_tok > 1) { // a subnormal and a zero weight in the second token
        w[(size_t) k + 1] = 3e-39f;
        w[(size_t) k + 2] = 0.0f;
    }
    for (auto& x : shared)
        x = value(0.1f);
    // every 7th column tiny in all K rows (1-4x the smallest normal) and its shared value zero or subnormal: the
    // products are subnormal, which the kernel flushes and plain CPU arithmetic keeps, so the sums differ unless the
    // host combine flushes them too
    for (int t = 0; t < n_tok; ++t)
        for (int c = 0; c < n; c += 7) {
            for (int e = 0; e < k; ++e)
                parts[((size_t) t * k + e) * n + c] = (pick(rng) < 50 ? -1.0f : 1.0f) * 1.2e-38f * (1 + pick(rng) % 4);
            shared[(size_t) t * n + c] = pick(rng) < 50 ? 0.0f : 2e-39f;
        }

    std::vector<float> ysum((size_t) n_tok * n), ones((size_t) n_tok, 1.0f);
    guild::core::host_moe_combine(parts.data(), w.data(), ysum.data(), n_tok, k, n);

    float *d_parts, *d_w, *d_shared, *d_ysum, *d_ones, *d_a, *d_b;
    CHECK(cudaMalloc(&d_parts, parts.size() * 4));
    CHECK(cudaMalloc(&d_w, w.size() * 4));
    CHECK(cudaMalloc(&d_shared, shared.size() * 4));
    CHECK(cudaMalloc(&d_ysum, ysum.size() * 4));
    CHECK(cudaMalloc(&d_ones, ones.size() * 4));
    CHECK(cudaMalloc(&d_a, ysum.size() * 4));
    CHECK(cudaMalloc(&d_b, ysum.size() * 4));
    CHECK(cudaMemcpy(d_parts, parts.data(), parts.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_shared, shared.data(), shared.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ysum, ysum.data(), ysum.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ones, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
    guild::kernels::native_moe_combine_multi(d_parts, d_w, d_shared, d_a, n, k, n_tok, st);   // the GPU combine
    guild::kernels::native_moe_combine_multi(d_ysum, d_ones, d_shared, d_b, n, 1, n_tok, st); // host sums + shared
    CHECK(cudaStreamSynchronize(st));
    std::vector<float> a(ysum.size()), b(ysum.size());
    CHECK(cudaMemcpy(a.data(), d_a, a.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(b.data(), d_b, b.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t ua, ub;
        std::memcpy(&ua, &a[i], 4);
        std::memcpy(&ub, &b[i], 4);
        if (ua != ub) {
            if (bad < 3)
                std::printf("  token %zu column %zu: GPU %08x (%.9g), host %08x (%.9g)\n", i / n, i % n, ua, a[i], ub,
                            b[i]);
            ++bad;
        }
    }
    std::printf("%s %d token(s), k %d, width %d: %d of %zu values differ\n", bad ? "FAIL" : "PASS", n_tok, k, n, bad,
                a.size());
    // the mapped forms: the same sums (every 97th made infinite, NaN or the largest float) in mapped host memory,
    // combined without a copy, against the combine of their device copy
    for (size_t i = 0; i < ysum.size(); i += 97)
        ysum[i] = (i / 97) % 4 == 0   ? INFINITY
                  : (i / 97) % 4 == 1 ? -INFINITY
                  : (i / 97) % 4 == 2 ? NAN
                                      : -3.4028235e38f;
    CHECK(cudaMemcpy(d_ysum, ysum.data(), ysum.size() * 4, cudaMemcpyHostToDevice));
    guild::kernels::native_moe_combine_multi(d_ysum, d_ones, d_shared, d_b, n, 1, n_tok, st); // the copy's combine
    CHECK(cudaStreamSynchronize(st));
    CHECK(cudaMemcpy(b.data(), d_b, b.size() * 4, cudaMemcpyDeviceToHost));
    float *h_map = nullptr, *m_map = nullptr;
    CHECK(cudaHostAlloc((void**) &h_map, ysum.size() * 4, cudaHostAllocMapped));
    CHECK(cudaHostGetDevicePointer((void**) &m_map, h_map, 0));
    std::memcpy(h_map, ysum.data(), ysum.size() * 4);
    int bad_map = 0;
    for (int form = 2; form >= 1; --form) {
        CHECK(cudaMemset(d_a, 0xff, ysum.size() * 4));
        bool ran = true;
        if (form == 2)
            guild::kernels::native_moe_combine_multi(m_map, d_ones, d_shared, d_a, n, 1, n_tok, st);
        else
            ran = guild::core::combine_sums_f4(m_map, d_ones, d_shared, d_a, n, n_tok, st);
        CHECK(cudaStreamSynchronize(st));
        if (!ran) { // only a width that is not a multiple of 4 may refuse
            std::printf("%s %d token(s), width %d: combine_sums_f4 refused\n", (n & 3) ? "PASS" : "FAIL", n_tok, n);
            bad_map += (n & 3) ? 0 : 1;
            continue;
        }
        CHECK(cudaMemcpy(a.data(), d_a, a.size() * 4, cudaMemcpyDeviceToHost));
        int diff = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            uint32_t ua, ub;
            std::memcpy(&ua, &a[i], 4);
            std::memcpy(&ub, &b[i], 4);
            if (ua != ub) {
                if (diff < 3)
                    std::printf("  token %zu column %zu: mapped %08x (%.9g), copied %08x (%.9g)\n", i / n, i % n, ua,
                                a[i], ub, b[i]);
                ++diff;
            }
        }
        std::printf("%s %d token(s), width %d, mapped sums through %s: %d of %zu values differ\n",
                    diff ? "FAIL" : "PASS", n_tok, n, form == 2 ? "the combine kernel" : "combine_sums_f4", diff,
                    a.size());
        bad_map += diff ? 1 : 0;
    }
    cudaFreeHost(h_map);
    for (float* p : {d_parts, d_w, d_shared, d_ysum, d_ones, d_a, d_b})
        cudaFree(p);
    return (bad ? 1 : 0) + bad_map;
}

int main() {
    if (!guild::core::host_combine_supported()) {
        std::printf("host_combine_parity: this CPU has no AVX2/FMA; skipped\n");
        return 77;
    }
    cudaStream_t st = nullptr;
    CHECK(cudaStreamCreate(&st));
    int fails = 0;
    unsigned seed = 1;
    for (int n_tok = 1; n_tok <= 4; ++n_tok)
        for (int n : {2560, 2563, 4096})
            fails += compare(n_tok, 10, n, seed++, st);
    fails += compare(2, 8, 2048, seed++, st);
    std::printf("FAILURES: %d\n", fails);
    return fails ? 1 : 0;
}
