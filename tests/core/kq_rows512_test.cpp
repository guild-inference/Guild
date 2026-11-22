// tests/core/kq_rows512_test.cpp - the AVX-512 Q4_K / Q6_K rows (include/guild/core/kq_rows512.hpp) against
// ggml-cpu's one-token ggml_vec_dot_q4_K_q8_K / ggml_vec_dot_q6_K_q8_K, bitwise, for 1-16 tokens, odd and even row
// ranges, rows of 1, 2 and 8 blocks, blocks of random bytes and activations quantized by ggml's from_float.  The
// gate/up form against two dots and native_gu_rows' SwiGLU expression.  Exit 77 without AVX-512BW.
#include "guild/core/kq_rows512.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

std::vector<uint8_t> random_rows(std::mt19937& rng, int type, int rows, int n) {
    const size_t rb = ggml_row_size((ggml_type) type, n), bb = type == GGML_TYPE_Q4_K ? 144 : 210;
    std::vector<uint8_t> v((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> lg(-14.f, 2.f);
    for (auto& b : v)
        b = (uint8_t) byte(rng);
    for (size_t k = 0; k < v.size() / bb; ++k) { // finite, normal-range fp16 scales
        uint8_t* p = v.data() + k * bb;
        const ggml_fp16_t d = ggml_fp32_to_fp16(std::exp2(lg(rng)) * (byte(rng) & 1 ? 1.f : -1.f));
        const ggml_fp16_t m = ggml_fp32_to_fp16(std::exp2(lg(rng)));
        if (type == GGML_TYPE_Q4_K) {
            std::memcpy(p, &d, 2);
            std::memcpy(p + 2, &m, 2);
        } else
            std::memcpy(p + 208, &d, 2);
    }
    return v;
}

} // namespace

int main() {
    using namespace guild::core::cpu_moe;
    if (!kq_rows512_ok()) {
        std::printf("skip: no AVX-512BW\n");
        return 77;
    }
    ggml_cpu_init();
    const ggml_type_traits_cpu* q8k = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    std::mt19937 rng(20261122);
    std::normal_distribution<float> nd(0.f, 1.f);
    int failures = 0, checked = 0;
    constexpr int kT = 16;
    for (int type : {(int) GGML_TYPE_Q4_K, (int) GGML_TYPE_Q6_K})
        for (int n : {256, 512, 2048}) {
            const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) type);
            const int rows = 37;
            const size_t rb = ggml_row_size((ggml_type) type, n), ab = ggml_row_size(GGML_TYPE_Q8_K, n);
            const std::vector<uint8_t> w = random_rows(rng, type, 2 * rows, n); // [gate rows | up rows]
            std::vector<uint8_t> a((size_t) kT * ab);
            std::vector<float> x(n);
            for (int t = 0; t < kT; ++t) {
                const float s = std::exp2((float) (t % 7) - 3.f);
                for (auto& v : x)
                    v = s * nd(rng);
                q8k->from_float(x.data(), a.data() + (size_t) t * ab, n);
            }
            for (int nt = 1; nt <= kT; ++nt) {
                const void* act[kT];
                for (int t = 0; t < nt; ++t)
                    act[t] = a.data() + (size_t) t * ab;
                for (int r0 : {0, 3})
                    for (int r1 : {rows, rows - 1, r0 + 1}) {
                        std::vector<float> got((size_t) nt * rows, -1.f);
                        float* out[kT];
                        for (int t = 0; t < nt; ++t)
                            out[t] = got.data() + (size_t) t * rows;
                        kq_rows512(type, w.data(), rb, n, act, nt, out, r0, r1);
                        for (int t = 0; t < nt; ++t)
                            for (int r = r0; r < r1; ++r) {
                                float want = 0.f;
                                tr->vec_dot(n, &want, 0, w.data() + (size_t) r * rb, 0, act[t], 0, 1);
                                ++checked;
                                if (std::memcmp(&want, &out[t][r], 4) != 0 && failures++ < 10)
                                    std::printf(
                                        "FAIL type %d n %d nt %d rows [%d, %d) token %d row %d: %.9g, ggml %.9g\n",
                                        type, n, nt, r0, r1, t, r, out[t][r], want);
                            }
                        if (type != GGML_TYPE_Q4_K)
                            continue;
                        // gate row r with up row r: silu(g) * u
                        std::fill(got.begin(), got.end(), -1.f);
                        q4k_gu_rows512(w.data(), rb, (size_t) rows * rb, n, act, nt, out, r0, r1);
                        for (int t = 0; t < nt; ++t)
                            for (int r = r0; r < r1; ++r) {
                                float g = 0.f, u = 0.f;
                                tr->vec_dot(n, &g, 0, w.data() + (size_t) r * rb, 0, act[t], 0, 1);
                                tr->vec_dot(n, &u, 0, w.data() + (size_t) (rows + r) * rb, 0, act[t], 0, 1);
                                const float want = (g / (1.f + std::exp(-g))) * u;
                                ++checked;
                                if (std::memcmp(&want, &out[t][r], 4) != 0 && failures++ < 10)
                                    std::printf(
                                        "FAIL gate/up n %d nt %d rows [%d, %d) token %d row %d: %.9g, ggml %.9g\n", n,
                                        nt, r0, r1, t, r, out[t][r], want);
                            }
                    }
            }
        }
    std::printf("%s: %d results, %d differ from ggml-cpu's\n", failures ? "FAIL" : "ok", checked, failures);
    return failures ? 1 : 0;
}
