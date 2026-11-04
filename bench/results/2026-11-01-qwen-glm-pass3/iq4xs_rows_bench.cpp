// IQ4_XS rows (GLM-5.3-Flash's down: 2,048 values a row) through iq256_rows (AVX2) and iq512_rows (AVX-512 BW), one
// thread: GMAC/s for 1-8 tokens, 16-row items over a 4,096-row matrix (one expert's down, 4.46 MB: L2/L3 misses)
// and over 64 such matrices (DRAM), the way cpu_moe's down items call them
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "guild/kernels/cpu/iq_avx512.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace guild::kernels::cpu;
int main(int argc, char** argv) {
    ggml_cpu_init();
    const int n = 2048, rows = 4096, nmat = argc > 1 ? atoi(argv[1]) : 1;
    const size_t rb = ggml_row_size(GGML_TYPE_IQ4_XS, n);
    std::mt19937 rng(1);
    std::vector<uint8_t> w(rb * rows * nmat);
    for (auto& b : w) b = (uint8_t) rng();
    for (size_t b = 0; b < w.size() / 136; ++b) {
        ggml_fp16_t d = ggml_fp32_to_fp16(1e-3f);
        std::memcpy(w.data() + b * 136, &d, 2);
    }
    std::vector<float> x(n);
    std::normal_distribution<float> nd;
    std::vector<std::vector<uint8_t>> a(8, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_K, n)));
    for (auto& v : a) { for (auto& f : x) f = nd(rng); ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), v.data(), n); }
    std::vector<float> o(8 * rows);
    const void* act[8]; float* out[8];
    for (int t = 0; t < 8; ++t) { act[t] = a[t].data(); out[t] = o.data() + t * rows; }
    for (int nt : {1, 2, 4, 8}) for (int k = 0; k < 2; ++k) {
        auto fn = k ? iq512_rows : iq256_rows;
        const int reps = nmat > 1 ? 2 : 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) for (int m = 0; m < nmat; ++m)
            for (int r0 = 0; r0 < rows; r0 += 16)
                fn(23, w.data() + (size_t) m * rb * rows, rb, n, act, nt, out, r0, r0 + 16);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%s nt %d: %6.1f GMAC/s, %6.1f GB/s of rows\n", k ? "avx512" : "avx2  ", nt,
                    1e-9 * reps * nmat * (double) rows * n * nt / s, 1e-9 * reps * nmat * (double) rows * rb / s);
    }
}
