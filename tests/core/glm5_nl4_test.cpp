// tests/core/glm5_nl4_test.cpp - the NL4 rows' AVX-512BW path (include/guild/core/glm5_nl4.hpp) against their AVX-2
// rows, bit for bit, for 1-8 tokens, odd and even row ranges, random blocks and activations (every int8, -128
// included).  The AVX-2 rows are ggml-cpu's IQ3_S dot to the bit (guild-kernels-cpu's iq3s_nl4_test).  Exit 77
// without AVX-512BW.
//
//   glm5_nl4_test [--bench]                 one thread's GB/s of rows, AVX-2 against AVX-512, 1/2/4 tokens
//   glm5_nl4_test --dram [threads] [tokens]  threads 0 .. threads - 1 each streaming 128 MB of its own rows
#include "guild/core/glm5_nl4.hpp"

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

constexpr size_t kBlock = 134; // an NL4 block: f16 d, 4 scale bytes, 128 bytes of codes (256 values)
constexpr size_t kQ8K = 292;   // ggml's block_q8_K: float d, 256 int8, 16 int16 sums

uint16_t f2h(float f) { // round to nearest f16 (finite, normal range)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int exp = (int) ((u >> 23) & 0xff) - 127 + 15;
    const uint32_t man = (u >> 13) & 0x3ff;
    return (uint16_t) (sign | (uint32_t) std::clamp(exp, 1, 30) << 10 | man);
}

std::vector<uint8_t> random_rows(std::mt19937& rng, int rows, int n) {
    std::vector<uint8_t> v((size_t) rows * (n / 256) * kBlock);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> lg(-14.f, 2.f);
    for (auto& b : v)
        b = (uint8_t) byte(rng);
    for (size_t k = 0; k < v.size() / kBlock; ++k) {
        const uint16_t d = f2h(std::exp2(lg(rng)) * (byte(rng) & 1 ? 1.f : -1.f));
        std::memcpy(v.data() + k * kBlock, &d, 2);
    }
    return v;
}

std::vector<uint8_t> random_acts(std::mt19937& rng, int nt, int n) {
    std::vector<uint8_t> v((size_t) nt * (n / 256) * kQ8K);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> lg(-10.f, 0.f);
    for (auto& b : v)
        b = (uint8_t) byte(rng);
    for (size_t k = 0; k < v.size() / kQ8K; ++k) {
        const float d = std::exp2(lg(rng));
        std::memcpy(v.data() + k * kQ8K, &d, 4);
    }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    using namespace guild::core::glm5cpu;
    if (!nl4_rows512_ok()) {
        std::printf("skip: no AVX-512BW (or GUILD_NL4_512=0)\n");
        return 77;
    }
    std::mt19937 rng(20261101);
    int failures = 0, checked = 0;
    for (int n : {256, 2048, 4096}) {
        const int rows = 37;
        const size_t rb = (size_t) (n / 256) * kBlock, ab = (size_t) (n / 256) * kQ8K;
        const std::vector<uint8_t> w = random_rows(rng, rows, n), a = random_acts(rng, 8, n);
        for (int nt = 1; nt <= 8; ++nt)
            for (int r0 : {0, 3})
                for (int r1 : {rows, rows - 1, r0 + 1}) {
                    std::vector<float> got((size_t) nt * rows, -1.f), want((size_t) nt * rows, -1.f);
                    const void* act[8];
                    float* og[8];
                    float* ow[8];
                    for (int t = 0; t < nt; ++t) {
                        act[t] = a.data() + (size_t) t * ab;
                        og[t] = got.data() + (size_t) t * rows;
                        ow[t] = want.data() + (size_t) t * rows;
                    }
                    nl4_rows(w.data(), rb, n, act, nt, og, r0, r1);
                    nl4_rows_avx2(w.data(), rb, n, act, nt, ow, r0, r1);
                    for (size_t i = 0; i < got.size(); ++i) {
                        ++checked;
                        if (std::memcmp(&got[i], &want[i], 4) != 0 && failures++ < 10)
                            std::printf("FAIL n %d nt %d rows [%d, %d) at %zu: %.9g, AVX-2 %.9g\n", n, nt, r0, r1, i,
                                        got[i], want[i]);
                    }
                }
    }
    std::printf("%s: %d results, %d differ from the AVX-2 rows'\n", failures ? "FAIL" : "ok", checked, failures);
    const int n =
        std::getenv("NL4_TEST_N") ? std::atoi(std::getenv("NL4_TEST_N")) : 4096; // GLM-5.3-Flash's gate/up rows
    const size_t rb = (size_t) (n / 256) * kBlock, ab = (size_t) (n / 256) * kQ8K;
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        const int rows = 4096; // one expert's gate and up rows: 8.8 MB
        const std::vector<uint8_t> w = random_rows(rng, rows, n), a = random_acts(rng, 4, n);
        std::vector<float> o((size_t) 4 * rows);
        const void* act[4];
        float* out[4];
        for (int t = 0; t < 4; ++t) {
            act[t] = a.data() + (size_t) t * ab;
            out[t] = o.data() + (size_t) t * rows;
        }
        for (int nt : {1, 2, 4})
            for (int k = 0; k < 2; ++k) {
                const int reps = 20;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i)
                    for (int r = 0; r < rows; r += 64)
                        (k ? nl4_rows : nl4_rows_avx2)(w.data(), rb, n, act, nt, out, r, std::min(r + 64, rows));
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("%-8s nt %d: %6.1f GB/s of rows\n", k ? "avx512" : "avx2", nt,
                            1e-9 * reps * (double) w.size() / s);
            }
    }
    if (argc > 1 && std::string(argv[1]) == "--dram") {
        const int threads = argc > 2 ? std::atoi(argv[2]) : 32, nt = argc > 3 ? std::atoi(argv[3]) : 2;
        const bool avx2 = std::getenv("NL4_TEST_AVX2") != nullptr;
        const int rows = (int) ((size_t) 128 << 20) / (int) rb; // 128 MB a thread
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
                const std::vector<uint8_t> w = random_rows(r, rows, n), a = random_acts(r, nt, n);
                std::vector<float> o((size_t) nt * rows);
                const void* act[8];
                float* out[8];
                for (int t = 0; t < nt; ++t) {
                    act[t] = a.data() + (size_t) t * ab;
                    out[t] = o.data() + (size_t) t * rows;
                }
                ready.fetch_add(1);
                while (ready.load() < threads) {}
                const auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < 3; ++k)
                    for (int rr = 0; rr < rows; rr += 64)
                        (avx2 ? nl4_rows_avx2 : nl4_rows)(w.data(), rb, n, act, nt, out, rr, std::min(rr + 64, rows));
                gbs[(size_t) i] = 3.0 * (double) w.size() /
                                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            });
        for (auto& t : th)
            t.join();
        double sum = 0;
        for (double g : gbs)
            sum += g;
        std::printf("dram: %d threads, %d tokens, %s: %.1f GB/s\n", threads, nt, avx2 ? "avx2" : "avx512", sum * 1e-9);
    }
    return failures ? 1 : 0;
}
