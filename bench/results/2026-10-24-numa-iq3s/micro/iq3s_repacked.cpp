// Repacked IQ3_S rows against ggml's single-token dot (ggml order) and iq256 (multi-token order): bits and speed.
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
constexpr int H = 2560, FF = 640, NB = 48;
int main() {
    ggml_cpu_init();
    const auto type = GGML_TYPE_IQ3_S;
    ggml_quantize_init(type);
    const size_t row = ggml_row_size(type, H), up_off = row * FF, blob = 2 * up_off;
    const size_t nrow = (H / 256) * cpu::kIq3sNl4Block, nup = nrow * FF, nblob = 2 * nup;
    std::vector<uint8_t> w(blob * NB), p(nblob * NB);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 0.02f);
    std::vector<float> f((size_t) 2 * FF * H);
    for (int e = 0; e < NB; ++e) {
        for (auto& v : f) v = nd(rng) * (1 + (e % 5));
        ggml_quantize_chunk(type, f.data(), w.data() + e * blob, 0, 2 * FF, H, nullptr);
        for (int r = 0; r < 2 * FF; ++r)
            if (!cpu::iq3s_nl4_repack(w.data() + e * blob + r * row, H / 256, p.data() + e * nblob + r * nrow)) { std::puts("repack refused"); return 1; }
    }
    const auto* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t abytes = ggml_row_size(GGML_TYPE_Q8_K, H);
    std::vector<uint8_t> act(abytes * 8);
    std::vector<float> x(H);
    std::vector<const void*> a(8);
    for (int t = 0; t < 8; ++t) { for (auto& v : x) v = nd(rng) * 50 * (1 + t); q8->from_float(x.data(), act.data() + t * abytes, H); a[t] = act.data() + t * abytes; }
    std::vector<float> o1(8 * FF), o2(8 * FF);
    std::vector<float*> p1(8), p2(8);
    for (int t = 0; t < 8; ++t) { p1[t] = &o1[t * FF]; p2[t] = &o2[t * FF]; }
    const ggml_vec_dot_t dot = ggml_get_type_traits_cpu(type)->vec_dot;
    int fails = 0;
    for (int e = 0; e < NB; ++e) {
        const uint8_t* b = w.data() + e * blob; const uint8_t* q = p.data() + e * nblob;
        // single token: ggml's dot, row by row
        for (int r = 0; r < FF; ++r) { float g = 0, u = 0; dot(H, &g, 0, b + r * row, 0, a[0], 0, 1); dot(H, &u, 0, b + up_off + r * row, 0, a[0], 0, 1); o1[r] = (g / (1.f + std::exp(-g))) * u; }
        cpu::iq3s_nl4_gu_rows(q, nrow, nup, H, a.data(), 1, p2.data(), 0, FF, true);
        fails += std::memcmp(o1.data(), o2.data(), FF * 4) != 0;
        // 2..8 tokens in ggml order (each token alone through ggml)
        for (int nt = 2; nt <= 8; nt += 3) {
            for (int t = 0; t < nt; ++t) for (int r = 0; r < FF; ++r) { float g = 0, u = 0; dot(H, &g, 0, b + r * row, 0, a[t], 0, 1); dot(H, &u, 0, b + up_off + r * row, 0, a[t], 0, 1); p1[t][r] = (g / (1.f + std::exp(-g))) * u; }
            cpu::iq3s_nl4_gu_rows(q, nrow, nup, H, a.data(), nt, p2.data(), 0, FF, true);
            fails += std::memcmp(o1.data(), o2.data(), nt * FF * 4) != 0;
        }
        // iq256 order
        for (int nt = 1; nt <= 8; ++nt) {
            cpu::iq256_gu_rows(type, b, row, up_off, H, a.data(), nt, p1.data(), 3, FF - 5);
            cpu::iq3s_nl4_gu_rows(q, nrow, nup, H, a.data(), nt, p2.data(), 3, FF - 5, false);
            for (int t = 0; t < nt; ++t) fails += std::memcmp(p1[t] + 3, p2[t] + 3, (FF - 8) * 4) != 0;
        }
    }
    std::printf("bitwise mismatches: %d\n", fails);
    for (int nt = 1; nt <= 4; ++nt) {
        double best[2] = {1e9, 1e9};
        for (int rep = 0; rep < 5; ++rep) for (int k = 0; k < 2; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int e = 0; e < NB; ++e) {
                if (k == 0) cpu::iq256_gu_rows(type, w.data() + e * blob, row, up_off, H, a.data(), nt, p1.data(), 0, FF);
                else cpu::iq3s_nl4_gu_rows(p.data() + e * nblob, nrow, nup, H, a.data(), nt, p2.data(), 0, FF, nt == 1);
            }
            best[k] = std::min(best[k], std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        const double gw = 2.0 * FF * H * NB / 1e9;
        std::printf("nt=%d  IQ3_S iq256 %.2f Gweights/s   repacked %.2f Gweights/s  (%.2fx)\n", nt, gw / best[0], gw / best[1], best[0] / best[1]);
    }
    return fails != 0;
}
