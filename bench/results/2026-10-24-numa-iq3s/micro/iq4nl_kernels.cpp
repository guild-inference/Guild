// IQ4_NL down rows (2560 rows of 640): AVX2 multi-token kernel vs ggml's per-token dot.
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
namespace cpu = guild::kernels::cpu;
constexpr int H = 2560, FF = 640, NB = 64;
int main() {
    ggml_cpu_init();
    const auto type = GGML_TYPE_IQ4_NL;
    const size_t row = ggml_row_size(type, FF), blob = row * H;
    std::vector<uint8_t> w(blob * NB);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 0.02f);
    std::vector<float> f((size_t) H * FF);
    for (int e = 0; e < NB; ++e) {
        for (auto& v : f) v = nd(rng);
        ggml_quantize_chunk(type, f.data(), w.data() + e * blob, 0, H, FF, nullptr);
    }
    const auto vdt = ggml_get_type_traits_cpu(type)->vec_dot_type;
    const size_t abytes = ggml_row_size(vdt, FF);
    std::vector<uint8_t> act(abytes * 4);
    std::vector<float> x(FF);
    for (int t = 0; t < 4; ++t) { for (auto& v : x) v = nd(rng) * 50; ggml_get_type_traits_cpu(vdt)->from_float(x.data(), act.data() + t * abytes, FF); }
    const void* a[4] = {act.data(), act.data() + abytes, act.data() + 2 * abytes, act.data() + 3 * abytes};
    std::vector<float> o(4 * H);
    float* p[4] = {&o[0], &o[H], &o[2 * H], &o[3 * H]};
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(type)->vec_dot;
    for (int nt = 1; nt <= 4; ++nt) {
        double best[2] = {1e9, 1e9};
        for (int rep = 0; rep < 5; ++rep)
            for (int k = 0; k < 2; ++k) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int e = 0; e < NB; ++e) {
                    const uint8_t* b = w.data() + e * blob;
                    if (k == 0) cpu::iq4nl256_down_rows(b, row, FF, a, nt, p, 0, H);
                    else for (int r = 0; r < H; ++r) for (int t = 0; t < nt; ++t) dot(FF, &p[t][r], 0, b + r * row, 0, a[t], 0, 1);
                }
                best[k] = std::min(best[k], std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
        const double gb = (double) blob * NB / 1e9;
        std::printf("nt=%d  iq4nl avx2 %.2f GB/s  ggml %.2f GB/s\n", nt, gb / best[0], gb / best[1]);
    }
}
