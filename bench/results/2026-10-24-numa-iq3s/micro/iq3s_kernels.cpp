// IQ3_S gate/up rows: AVX-512 kernel vs AVX2 kernel vs ggml's per-token dot, on this CPU.
#include "guild/kernels/cpu/iq_avx512.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace cpu = guild::kernels::cpu;
constexpr int H = 2560, FF = 640, NB = 48; // 48 experts (~67 MB) so the weights come from DRAM
int main() {
    ggml_cpu_init();
    const auto type = GGML_TYPE_IQ3_S;
    ggml_quantize_init(type);
    const size_t row = ggml_row_size(type, H), up_off = row * FF, blob = 2 * up_off;
    std::vector<uint8_t> w(blob * NB);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 0.02f);
    std::vector<float> f((size_t) 2 * FF * H);
    for (int e = 0; e < NB; ++e) {
        for (auto& v : f) v = nd(rng);
        ggml_quantize_chunk(type, f.data(), w.data() + e * blob, 0, 2 * FF, H, nullptr);
    }
    const auto* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t abytes = ggml_row_size(GGML_TYPE_Q8_K, H);
    std::vector<uint8_t> act(abytes * 4);
    std::vector<float> x(H);
    for (int t = 0; t < 4; ++t) {
        for (auto& v : x) v = nd(rng) * 50;
        q8->from_float(x.data(), act.data() + t * abytes, H);
    }
    const void* a[4] = {act.data(), act.data() + abytes, act.data() + 2 * abytes, act.data() + 3 * abytes};
    std::vector<float> o512(4 * FF), o256(4 * FF), og(4 * FF);
    auto ptrs = [](std::vector<float>& v) { return std::vector<float*>{&v[0], &v[FF], &v[2 * FF], &v[3 * FF]}; };
    auto p512 = ptrs(o512), p256 = ptrs(o256), pg = ptrs(og);
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(type)->vec_dot;
    auto ggml_rows = [&](const uint8_t* b, int nt) {
        for (int r = 0; r < FF; ++r)
            for (int t = 0; t < nt; ++t) {
                float g = 0, u = 0;
                dot(H, &g, 0, b + r * row, 0, a[t], 0, 1);
                dot(H, &u, 0, b + up_off + r * row, 0, a[t], 0, 1);
                pg[t][r] = (g / (1.f + std::exp(-g))) * u;
            }
    };
    for (int nt = 1; nt <= 4; ++nt) {
        double best[3] = {1e9, 1e9, 1e9};
        for (int rep = 0; rep < 5; ++rep) {
            for (int k = 0; k < 3; ++k) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int e = 0; e < NB; ++e) {
                    const uint8_t* b = w.data() + e * blob;
                    if (k == 0) cpu::iq512_gu_rows(type, b, row, up_off, H, a, nt, p512.data(), 0, FF);
                    else if (k == 1) cpu::iq256_gu_rows(type, b, row, up_off, H, a, nt, p256.data(), 0, FF);
                    else ggml_rows(b, nt);
                }
                best[k] = std::min(best[k], std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
        }
        const double gb = (double) blob * NB / 1e9;
        int d512 = 0, d256 = 0;
        double m512 = 0;
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < FF; ++r) {
                d512 += std::memcmp(&o512[t * FF + r], &og[t * FF + r], 4) != 0;
                d256 += std::memcmp(&o256[t * FF + r], &og[t * FF + r], 4) != 0;
                m512 = std::max(m512, (double) std::fabs(o512[t * FF + r] - o256[t * FF + r]) / (std::fabs(o256[t * FF + r]) + 1e-6));
            }
        std::printf("nt=%d  avx512 %.2f GB/s  avx2 %.2f GB/s  ggml %.2f GB/s   rows differing from ggml: avx512 %d avx2 %d of %d; max rel 512 vs 256 %.2e\n",
                    nt, gb / best[0], gb / best[1], gb / best[2], d512, d256, nt * FF, m512);
    }
}
