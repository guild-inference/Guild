// tests/core/iq4xs_rows512_test.cpp - the AVX-512 IQ4_XS rows (include/guild/core/iq4xs_rows512.hpp) against
// ggml-cpu's one-token ggml_vec_dot_iq4_xs_q8_K, bitwise, for 1-8 tokens, odd and even row ranges, and blocks of
// random bytes (every code and 6-bit scale, f16 scales from tiny to large) against ggml-quantized activations.
// Exit 77 without AVX-512BW.
//
//   iq4xs_rows512_test --dram [threads] [tokens]   threads 0 .. threads - 1 each streaming 128 MB of its own
//   2,048-value
//                                                  rows in 64-row items in a shuffled order (GLM-5.3-Flash's down
//                                                  rows), these rows against ggml-cpu's dot (all threads' GB/s)
#include "guild/core/iq4xs_rows512.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <random>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<uint8_t> random_rows(std::mt19937& rng, int rows, int n) {
    const size_t rb = ggml_row_size(GGML_TYPE_IQ4_XS, n), bb = ggml_type_size(GGML_TYPE_IQ4_XS);
    std::vector<uint8_t> w(rb * rows);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> lg(-14.f, 2.f);
    for (auto& v : w)
        v = (uint8_t) byte(rng);
    for (size_t b = 0; b < w.size() / bb; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(std::exp2(lg(rng)) * (byte(rng) & 1 ? 1.f : -1.f));
        std::memcpy(w.data() + b * bb, &d, 2);
    }
    return w;
}

std::vector<std::vector<uint8_t>> random_acts(std::mt19937& rng, int n, int nt) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> a((size_t) nt, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_K, n)));
    std::vector<float> x((size_t) n);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x)
            v = nd(rng) * (1.f + (float) t);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), a[(size_t) t].data(), n);
    }
    return a;
}

} // namespace

int main(int argc, char** argv) {
    using namespace guild::core::cpu_moe;
    if (!iq4xs_rows512_ok()) {
        std::printf("skip: no AVX-512BW (or GUILD_FORCE_AVX2=1)\n");
        return 77;
    }
    ggml_cpu_init();
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS)->vec_dot;
    if (argc > 1 && std::string(argv[1]) == "--dram") {
        const int threads = argc > 2 ? std::atoi(argv[2]) : 32, nt = argc > 3 ? std::atoi(argv[3]) : 1, n = 2048;
        const size_t rb = ggml_row_size(GGML_TYPE_IQ4_XS, n);
        const int rows = (int) (((size_t) 128 << 20) / rb), item = 64;
        for (int mode = 0; mode < 2; ++mode) {
            std::vector<double> gbs((size_t) threads);
            std::atomic<int> ready{0};
            std::vector<std::thread> th;
            for (int i = 0; i < threads; ++i)
                th.emplace_back([&, i] {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    CPU_SET(i, &set);
                    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
                    std::mt19937 r((unsigned) i);
                    const auto w = random_rows(r, rows, n);
                    const auto a = random_acts(r, n, nt);
                    std::vector<float> o((size_t) rows * nt);
                    const void* act[8];
                    float* out[8];
                    for (int t = 0; t < nt; ++t) {
                        act[t] = a[(size_t) t].data();
                        out[t] = o.data() + (size_t) t * rows;
                    }
                    std::vector<int> order;
                    for (int q = 0; q < rows; q += item)
                        order.push_back(q);
                    std::shuffle(order.begin(), order.end(), r);
                    ready.fetch_add(1);
                    while (ready.load() < threads) {}
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int k = 0; k < 3; ++k)
                        for (int q0 : order) {
                            const int q1 = std::min(q0 + item, rows);
                            if (mode == 1) {
                                iq4xs_rows512(w.data(), rb, n, act, nt, out, q0, q1);
                                continue;
                            }
                            for (int t = 0; t < nt; ++t)
                                for (int q = q0; q < q1; ++q)
                                    dot(n, &out[t][q], 0, w.data() + (size_t) q * rb, 0, act[t], 0, 1);
                        }
                    gbs[(size_t) i] = 3.0 * (double) w.size() /
                                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                });
            for (auto& t : th)
                t.join();
            double s = 0;
            for (double g : gbs)
                s += g;
            std::printf("dram: %d threads, %d tokens, %s: %.1f GB/s\n", threads, nt,
                        mode ? "iq4xs_rows512" : "ggml vec_dot", s * 1e-9);
        }
        return 0;
    }
    std::mt19937 rng(20261102);
    int failures = 0, checked = 0;
    for (int n : {256, 2048, 4096}) {
        const int rows = 37;
        const size_t rb = ggml_row_size(GGML_TYPE_IQ4_XS, n);
        const auto w = random_rows(rng, rows, n);
        const auto a = random_acts(rng, n, 8);
        for (int nt = 1; nt <= 8; ++nt)
            for (int r0 : {0, 3})
                for (int r1 : {rows, rows - 1, r0 + 1}) {
                    std::vector<float> got((size_t) nt * rows, -1.f);
                    const void* act[8];
                    float* out[8];
                    for (int t = 0; t < nt; ++t) {
                        act[t] = a[(size_t) t].data();
                        out[t] = got.data() + (size_t) t * rows;
                    }
                    iq4xs_rows512(w.data(), rb, n, act, nt, out, r0, r1);
                    for (int t = 0; t < nt; ++t)
                        for (int r = r0; r < r1; ++r) {
                            float want = 0.f;
                            dot(n, &want, 0, w.data() + (size_t) r * rb, 0, act[t], 0, 1);
                            ++checked;
                            if (std::memcmp(&want, &out[t][r], 4) != 0 && failures++ < 10)
                                std::printf("FAIL n %d nt %d rows [%d, %d) row %d token %d: %.9g, ggml %.9g\n", n, nt,
                                            r0, r1, r, t, out[t][r], want);
                        }
                }
    }
    std::printf("%s: %d results, %d differ from ggml-cpu's dot\n", failures ? "FAIL" : "ok", checked, failures);
    return failures ? 1 : 0;
}
