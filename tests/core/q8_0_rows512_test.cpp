// tests/core/q8_0_rows512_test.cpp - the AVX-512 Q8_0 rows (include/guild/core/q8_0_rows512.hpp) against ggml-cpu's
// one-token ggml_vec_dot_q8_0_q8_0, bitwise, for 1-16 tokens, odd and even row ranges, and blocks of random bytes
// (every int8, -128 included, and scales from tiny to large).  Exit 77 without AVX-512BW.
//
//   q8_0_rows512_test [--bench]                 one thread's rows a second against guild-kernels-cpu's AVX2 kq256_rows
//   q8_0_rows512_test --dram [threads] [tokens]  threads 0 .. threads - 1 each streaming 128 MB of its own rows
#include "guild/core/q8_0_rows512.hpp"
#include "guild/kernels/cpu/kq_avx2.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr size_t kBlock = 34;

std::vector<uint8_t> random_rows(std::mt19937& rng, int rows, int n) {
    std::vector<uint8_t> v((size_t) rows * (n / 32) * kBlock);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> lg(-24.f, 4.f);
    for (size_t b = 0; b < v.size() / kBlock; ++b) {
        uint8_t* p = v.data() + b * kBlock;
        const float d = std::exp2(lg(rng)) * (byte(rng) & 1 ? 1.f : -1.f);
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(p, &h, 2);
        for (int i = 0; i < 32; ++i)
            p[2 + i] = (uint8_t) byte(rng);
    }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    using namespace guild::core::cpu_moe;
    if (!q8_0_rows512_ok()) {
        std::printf("skip: no AVX-512BW\n");
        return 77;
    }
    ggml_cpu_init();
    const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::mt19937 rng(20261101);
    int failures = 0, checked = 0;
    for (int n : {32, 64, 1856, 2688}) {
        const int rows = 37;
        const size_t rb = (size_t) (n / 32) * kBlock;
        const std::vector<uint8_t> w = random_rows(rng, rows, n);
        const std::vector<uint8_t> a = random_rows(rng, kQ8Rows512MaxT, n);
        for (int nt = 1; nt <= kQ8Rows512MaxT; ++nt)
            for (int r0 : {0, 3})
                for (int r1 : {rows, rows - 1, r0 + 1}) {
                    std::vector<float> got((size_t) nt * rows, -1.f);
                    const void* act[kQ8Rows512MaxT];
                    float* out[kQ8Rows512MaxT];
                    for (int t = 0; t < nt; ++t) {
                        act[t] = a.data() + (size_t) t * rb;
                        out[t] = got.data() + (size_t) t * rows;
                    }
                    std::vector<uint8_t> pk(q8_0_packed_bytes(n, nt));
                    q8_0_pack(act, nt, n, pk.data());
                    q8_0_rows512(w.data(), rb, n, pk.data(), nt, out, r0, r1);
                    for (int t = 0; t < nt; ++t)
                        for (int r = 0; r < rows; ++r) {
                            float want = -1.f;
                            if (r >= r0 && r < r1)
                                tr->vec_dot(n, &want, 0, w.data() + (size_t) r * rb, 0, act[t], 0, 1);
                            ++checked;
                            if (std::memcmp(&want, &out[t][r], 4) != 0 && failures++ < 10)
                                std::printf("FAIL n %d nt %d rows [%d, %d) token %d row %d: %.9g, ggml %.9g\n", n, nt,
                                            r0, r1, t, r, out[t][r], want);
                        }
                }
    }
    std::printf("%s: %d results, %d differ from ggml-cpu's\n", failures ? "FAIL" : "ok", checked, failures);
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        const int n = 2688, rows = 1856; // a Nemotron-H expert's up matrix
        const size_t rb = (size_t) (n / 32) * kBlock;
        const std::vector<uint8_t> w = random_rows(rng, rows, n);
        const std::vector<uint8_t> a = random_rows(rng, kQ8Rows512MaxT, n);
        std::vector<float> o((size_t) kQ8Rows512MaxT * rows);
        const void* act[kQ8Rows512MaxT];
        float* out[kQ8Rows512MaxT];
        for (int t = 0; t < kQ8Rows512MaxT; ++t) {
            act[t] = a.data() + (size_t) t * rb;
            out[t] = o.data() + (size_t) t * rows;
        }
        for (int nt : {1, 2, 4, 8, 16})
            for (int k = 0; k < 2; ++k) {
                std::vector<uint8_t> pk(q8_0_packed_bytes(n, nt));
                q8_0_pack(act, nt, n, pk.data());
                const int reps = 40;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i)
                    for (int r = 0; r < rows; r += 64) // the expert phases' row blocks
                        if (k)
                            q8_0_rows512(w.data(), rb, n, pk.data(), nt, out, r, std::min(r + 64, rows));
                        else
                            guild::kernels::cpu::kq256_rows(GGML_TYPE_Q8_0, w.data(), rb, n, act, nt, out, r,
                                                            std::min(r + 64, rows));
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("%-13s nt %2d: %6.1f GMAC/s\n", k ? "q8_0_rows512" : "kq256_rows", nt,
                            1e-9 * reps * (double) rows * n * nt / s);
            }
    }
    if (argc > 1 && std::string(argv[1]) == "--dram") {
        const int threads = argc > 2 ? std::atoi(argv[2]) : 32, nt = argc > 3 ? std::atoi(argv[3]) : 1;
        const int n = 2688, rows = 1856 * 24; // 24 experts' up rows a thread: 128 MB
        const size_t rb = (size_t) (n / 32) * kBlock;
        std::vector<std::thread> th;
        std::vector<double> gbs((size_t) threads);
        std::atomic<int> ready{0};
        for (int i = 0; i < threads; ++i)
            th.emplace_back([&, i] {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(i, &set);
                pthread_setaffinity_np(pthread_self(), sizeof set, &set);
                std::mt19937 r(i);
                const std::vector<uint8_t> w = random_rows(r, rows, n), a = random_rows(r, nt, n);
                std::vector<uint8_t> pk(q8_0_packed_bytes(n, nt));
                const void* act[kQ8Rows512MaxT];
                for (int t = 0; t < nt; ++t)
                    act[t] = a.data() + (size_t) t * rb;
                q8_0_pack(act, nt, n, pk.data());
                std::vector<float> o((size_t) nt * rows);
                float* out[kQ8Rows512MaxT];
                for (int t = 0; t < nt; ++t)
                    out[t] = o.data() + (size_t) t * rows;
                ready.fetch_add(1);
                while (ready.load() < threads) {}
                const auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < 3; ++k)
                    for (int rr = 0; rr < rows; rr += 64)
                        q8_0_rows512(w.data(), rb, n, pk.data(), nt, out, rr, std::min(rr + 64, rows));
                gbs[(size_t) i] = 3.0 * (double) w.size() /
                                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            });
        for (auto& t : th)
            t.join();
        double sum = 0;
        for (double g : gbs)
            sum += g;
        std::printf("dram: %d threads, %d tokens: %.1f GB/s\n", threads, nt, sum * 1e-9);
    }
    return failures ? 1 : 0;
}
