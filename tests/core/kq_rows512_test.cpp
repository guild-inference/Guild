// tests/core/kq_rows512_test.cpp - the AVX-512 Q4_K / Q6_K rows (include/guild/core/kq_rows512.hpp) against
// ggml-cpu's one-token ggml_vec_dot_q4_K_q8_K / ggml_vec_dot_q6_K_q8_K, bitwise, for 1-16 tokens, odd and even row
// ranges, rows of 1, 2 and 8 blocks, blocks of random bytes and activations quantized by ggml's from_float.  The
// gate/up form against two dots and native_gu_rows' SwiGLU expression.  Then the pool's table as the engine binds it
// (pool_rows512, pool_rows.hpp) against the library's table it wraps, for every format it reroutes (Q4_K gate/up,
// Q4_K and Q6_K down: Drama 3, Ornith; IQ4_XS gate/up, IQ4_NL down: Qwen3.8-Flash-Next): every (row, token) of
// gate_up and down the same bits for 1-10 and 16 tokens and odd and even row ranges, and nothing written outside the
// rows asked for.  Exit 77 without AVX-512BW.
#include "guild/core/kq_rows512.hpp"
#include "guild/core/pool_rows.hpp"
#include "guild/kernels/cpu/native_expert.hpp"
#include "guild/kernels/cpu/pool.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// a 256- or 32-value block's bytes and where its fp16 scales are (ggml's block layouts)
struct BlockInfo {
    size_t bytes;
    int d_off, dmin_off; // dmin_off < 0: none
};
BlockInfo block_info(int type) {
    switch (type) {
    case GGML_TYPE_Q4_K: return {144, 0, 2};
    case GGML_TYPE_Q6_K: return {210, 208, -1};
    case GGML_TYPE_IQ4_XS: return {136, 0, -1};
    case GGML_TYPE_Q8_0: return {34, 0, -1};
    default: return {18, 0, -1}; // IQ4_NL
    }
}

// rows of random bytes with finite, normal-range fp16 scales
void randomize(std::mt19937& rng, uint8_t* p, size_t bytes, int type) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> lg(-14.f, 2.f);
    for (size_t i = 0; i < bytes; ++i)
        p[i] = (uint8_t) byte(rng);
    const BlockInfo bi = block_info(type);
    for (size_t k = 0; k < bytes / bi.bytes; ++k) {
        uint8_t* b = p + k * bi.bytes;
        const ggml_fp16_t d = ggml_fp32_to_fp16(std::exp2(lg(rng)) * (byte(rng) & 1 ? 1.f : -1.f));
        std::memcpy(b + bi.d_off, &d, 2);
        if (bi.dmin_off >= 0) {
            const ggml_fp16_t m = ggml_fp32_to_fp16(std::exp2(lg(rng)));
            std::memcpy(b + bi.dmin_off, &m, 2);
        }
    }
}

// pool_rows512(base) against base: gate_up and down of one random expert of format (gu, d, n_embd, n_ff)
int check_table(const guild_cpu_expert_ext_v1_t& base, const guild_cpu_expert_ext_v1_t& wrapped, int gu, int d,
                int n_embd, int n_ff, std::mt19937& rng, int& checked) {
    namespace cpu = guild::kernels::cpu;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, d, n_embd, n_ff, f, err)) {
        std::printf("FAIL native_fmt(%d, %d): %s\n", gu, d, err.c_str());
        return 1;
    }
    std::vector<uint8_t> blob(f.bytes);
    randomize(rng, blob.data(), f.down_off, gu);
    randomize(rng, blob.data() + f.down_off, f.bytes - f.down_off, d);
    constexpr int kT = 16;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> xa((size_t) kT * f.act_bytes), ha((size_t) kT * f.h_bytes);
    std::vector<float> x((size_t) n_embd), h((size_t) n_ff);
    for (int t = 0; t < kT; ++t) {
        const float s = std::exp2((float) (t % 5) - 2.f);
        for (auto& v : x)
            v = s * nd(rng);
        for (auto& v : h)
            v = 0.1f * s * nd(rng);
        base.quant_act(&f, x.data(), xa.data() + (size_t) t * f.act_bytes);
        base.quant_h(&f, h.data(), ha.data() + (size_t) t * f.h_bytes);
    }
    const void* xp[kT];
    const void* hp[kT];
    for (int t = 0; t < kT; ++t) {
        xp[t] = xa.data() + (size_t) t * f.act_bytes;
        hp[t] = ha.data() + (size_t) t * f.h_bytes;
    }
    int failures = 0;
    for (int nt : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16})
        for (int down = 0; down < 2; ++down) {
            const int rows = down ? n_embd : n_ff;
            const int ranges[3][2] = {{0, rows}, {3, rows - 1}, {5, 6}};
            for (const auto& rg : ranges) {
                std::vector<float> want((size_t) nt * rows, -7.f), got((size_t) nt * rows, -7.f);
                float* wp[kT];
                float* gp[kT];
                for (int t = 0; t < nt; ++t) {
                    wp[t] = want.data() + (size_t) t * rows;
                    gp[t] = got.data() + (size_t) t * rows;
                }
                if (down) {
                    base.down(&f, blob.data(), hp, nt, wp, rg[0], rg[1]);
                    wrapped.down(&f, blob.data(), hp, nt, gp, rg[0], rg[1]);
                } else {
                    base.gate_up(&f, blob.data(), xp, nt, wp, rg[0], rg[1]);
                    wrapped.gate_up(&f, blob.data(), xp, nt, gp, rg[0], rg[1]);
                }
                for (int t = 0; t < nt; ++t)
                    for (int r = 0; r < rows; ++r) {
                        ++checked;
                        if (std::memcmp(&want[(size_t) t * rows + r], &got[(size_t) t * rows + r], 4) != 0 &&
                            failures++ < 10)
                            std::printf("FAIL table %d/%d %s nt %d rows [%d, %d) token %d row %d: %.9g, library %.9g\n",
                                        gu, d, down ? "down" : "gate_up", nt, rg[0], rg[1], t, r,
                                        got[(size_t) t * rows + r], want[(size_t) t * rows + r]);
                    }
            }
        }
    return failures;
}

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

    // the engine's pool table (model.cpp binds pool_rows512 of the library's) against the library's own
    int tfail = 0, tchecked = 0;
    {
        guild::kernels::cpu::ExpertPool pool(1, false, true);
        const guild_cpu_expert_ext_v1_t base = pool.native_table();
        const guild_cpu_expert_ext_v1_t wrapped = guild::core::pool_rows512(base, false);
        if (wrapped.gate_up == base.gate_up || wrapped.down == base.down) {
            std::printf("FAIL: pool_rows512 did not wrap the table (GUILD_POOL_ROWS512=0?)\n");
            return 1;
        }
        tfail += check_table(base, wrapped, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 2048, 512, rng, tchecked);
        tfail += check_table(base, wrapped, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 2048, 512, rng, tchecked);
        tfail += check_table(base, wrapped, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0, 2560, 640, rng, tchecked);
        tfail += check_table(base, wrapped, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, 2560, 640, rng, tchecked);
    }
    std::printf("%s: %d table results, %d differ from the library's table\n", tfail ? "FAIL" : "ok", tchecked, tfail);
    return failures || tfail ? 1 : 0;
}
