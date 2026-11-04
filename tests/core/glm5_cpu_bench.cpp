// tests/core/glm5_cpu_bench.cpp - the CPU expert kernels on GLM-5.3-Flash's real expert rows, one thread: GB/s of
// weights read by ggml-cpu's single-token vec_dot and by Guild's multi-token row kernels.
//
//   glm5_cpu_bench <gguf shard> [layer]
#include "guild/artifact/gguf_reader.hpp"
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "guild/kernels/cpu/iq_avx512.hpp"
#include "guild/kernels/cpu/kq_avx2.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace kcpu = guild::kernels::cpu;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: glm5_cpu_bench <gguf> [layer]\n");
        return 2;
    }
    const int layer = argc > 2 ? std::atoi(argv[2]) : 10;
    ggml_cpu_init();
    guild::GgufModel m = guild::GgufModel::open(argv[1]);
    std::mt19937 rng(1);
    for (const char* nm : {"ffn_gate_exps", "ffn_down_exps", "ffn_gate_shexp"}) {
        size_t sh = 0;
        const std::string name = "blk." + std::to_string(layer) + "." + nm + ".weight";
        const guild::TensorInfo* t = m.find(name, &sh);
        if (!t) {
            std::printf("%s: missing\n", name.c_str());
            continue;
        }
        const uint8_t* w = m.shard(sh).tensor_data(*t);
        const int type = (int) t->type, n = (int) t->shape[0];
        const size_t rb = ggml_row_size((ggml_type) type, n);
        const int rows = 2048; // one expert's gate rows, or 2048 down rows
        const auto* tr = ggml_get_type_traits_cpu((ggml_type) type);
        std::vector<float> x((size_t) 8 * n);
        std::normal_distribution<float> nd;
        for (auto& v : x)
            v = nd(rng);
        const int at = (int) tr->vec_dot_type;
        const size_t ab = ggml_row_size((ggml_type) at, n);
        std::vector<uint8_t> act((size_t) 8 * ab);
        for (int k = 0; k < 8; ++k)
            ggml_get_type_traits_cpu((ggml_type) at)->from_float(x.data() + (size_t) k * n, act.data() + k * ab, n);
        std::vector<float> out((size_t) 8 * rows);
        // warm the rows into the cache hierarchy's view of memory once
        volatile uint8_t sink = 0;
        for (size_t i = 0; i < (size_t) rows * rb; i += 4096)
            sink ^= w[i];
        auto gbs = [&](double ms) { return (double) rows * rb / (ms * 1e-3) / 1e9; };
        auto t0 = std::chrono::steady_clock::now();
        const int reps = 5;
        for (int r = 0; r < reps; ++r)
            for (int i = 0; i < rows; ++i)
                tr->vec_dot(n, &out[(size_t) i], 0, w + (size_t) i * rb, 0, act.data(), 0, 1);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%-16s %-7s n %5d  ggml vec_dot 1 token: %6.2f ms / %d rows = %5.2f GB/s\n", nm,
                    ggml_type_name((ggml_type) type), n, ms, rows, gbs(ms));
        using RowsFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
        RowsFn fn = nullptr;
        const char* which = "";
        if (kcpu::cpu_avx512_ok() && kcpu::iq512_supported(type)) {
            fn = &kcpu::iq512_rows;
            which = "iq512";
        } else if (kcpu::iq256_supported(type)) {
            fn = &kcpu::iq256_rows;
            which = "iq256";
        } else if (kcpu::kq256_supported(type)) {
            fn = &kcpu::kq256_rows;
            which = "kq256";
        }
        if (!fn)
            continue;
        for (int nt : {1, 2, 4, 8}) {
            const void* a[8];
            float* o[8];
            for (int k = 0; k < nt; ++k) {
                a[k] = act.data() + k * ab;
                o[k] = out.data() + (size_t) k * rows;
            }
            t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r)
                fn(type, w, rb, n, a, nt, o, 0, rows);
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("%-16s %-7s n %5d  %s %d tokens:   %6.2f ms = %5.2f GB/s (%5.2f per token)\n", nm,
                        ggml_type_name((ggml_type) type), n, which, nt, ms, gbs(ms), gbs(ms) * nt);
        }
    }
    // IQ3_S gate/up repacked as NL4 (the grid lookup done at load): one expert's gate and up rows, fused SwiGLU
    {
        size_t sg = 0, su = 0;
        const guild::TensorInfo* tg = m.find("blk." + std::to_string(layer) + ".ffn_gate_exps.weight", &sg);
        const guild::TensorInfo* tu = m.find("blk." + std::to_string(layer) + ".ffn_up_exps.weight", &su);
        if (tg && tu && tg->type == 21) {
            const int n = (int) tg->shape[0], rows = 2048, nb = n / 256;
            const size_t rb = ggml_row_size(GGML_TYPE_IQ3_S, n), nl4 = (size_t) nb * kcpu::kIq3sNl4Block;
            std::vector<uint8_t> blob((size_t) 2 * rows * nl4);
            const uint8_t* g = m.shard(sg).tensor_data(*tg);
            const uint8_t* u = m.shard(su).tensor_data(*tu);
            bool ok = true;
            for (int r = 0; r < rows; ++r) {
                ok &= kcpu::iq3s_nl4_repack(g + (size_t) r * rb, nb, blob.data() + (size_t) r * nl4);
                ok &= kcpu::iq3s_nl4_repack(u + (size_t) r * rb, nb, blob.data() + (size_t) (rows + r) * nl4);
            }
            std::vector<float> x((size_t) 8 * n);
            std::normal_distribution<float> nd;
            for (auto& v : x)
                v = nd(rng);
            const size_t ab = ggml_row_size(GGML_TYPE_Q8_K, n);
            std::vector<uint8_t> act((size_t) 8 * ab);
            for (int k = 0; k < 8; ++k)
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data() + (size_t) k * n, act.data() + k * ab, n);
            std::vector<float> ff((size_t) 8 * rows);
            for (int nt : {1, 2, 4, 8}) {
                const void* a[8];
                float* o[8];
                for (int k = 0; k < nt; ++k) {
                    a[k] = act.data() + k * ab;
                    o[k] = ff.data() + (size_t) k * rows;
                }
                const int reps = 5;
                auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; ++r)
                    kcpu::iq3s_nl4_gu_rows(blob.data(), nl4, (size_t) rows * nl4, n, a, nt, o, 0, rows, nt == 1);
                const double ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
                const double iq3_bytes = 2.0 * rows * rb; // what the same rows weigh as IQ3_S
                std::printf("gate+up IQ3_S as NL4 (repack %s) %d tokens: %6.2f ms = %5.2f GB/s of IQ3_S rows "
                            "(%5.2f per token)\n",
                            ok ? "ok" : "FAILED", nt, ms, iq3_bytes / (ms * 1e-3) / 1e9,
                            iq3_bytes / (ms * 1e-3) / 1e9 * nt);
            }
        }
    }
    return 0;
}
