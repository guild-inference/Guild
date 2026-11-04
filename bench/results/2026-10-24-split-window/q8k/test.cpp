#include "q8k_avx2.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
extern "C" void quantize_row_q8_K_ref(const float* x, void* y, int64_t k);
int main() {
    const int K = 2560, NB = K / 256;
    std::mt19937 rng(1);
    std::vector<float> x(K);
    std::vector<BlockQ8K> a(NB), b(NB);
    long bad = 0, n = 0;
    for (int it = 0; it < 400000; ++it) {
        int kind = it % 5;
        for (auto& v : x) {
            if (kind == 0) v = std::normal_distribution<float>(0, 1)(rng);
            else if (kind == 1) v = std::normal_distribution<float>(0, 1e-3f)(rng) * (rng() % 50 == 0 ? 300 : 1);
            else if (kind == 2) v = (float) ((int) (rng() % 255) - 127) * 0.5f; // exact halves: ties
            else if (kind == 3) v = (rng() % 7 == 0) ? 0.f : std::uniform_real_distribution<float>(-4, 4)(rng);
            else v = (rng() % 3 == 0) ? -2.0f : ((rng() % 2) ? 2.0f : std::ldexp(1.f, -(int) (rng() % 40)));
        }
        if (it % 97 == 0) for (int j = 256; j < 512; ++j) x[j] = 0;      // all-zero block
        std::memset(a.data(), 0x5a, sizeof(BlockQ8K) * NB);
        std::memset(b.data(), 0x5a, sizeof(BlockQ8K) * NB);
        quantize_row_q8_K_ref(x.data(), a.data(), K);
        q8k_avx2(x.data(), b.data(), K);
        ++n; if (std::memcmp(a.data(), b.data(), sizeof(BlockQ8K) * NB)) ++bad;
    }
    std::printf("rows %ld mismatches %ld\n", n, bad);
    auto t = [&](auto f) { auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 200000; ++i) f(); return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200000; };
    std::printf("ref %.3f us/row  avx2 %.3f us/row\n", t([&] { quantize_row_q8_K_ref(x.data(), a.data(), K); }),
                t([&] { q8k_avx2(x.data(), b.data(), K); }));
}
