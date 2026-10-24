// src/kernels/cpu/pool_numa_test.cpp - the pool's native multi-token paths against the kernels called directly.
//
// run_split_multi_native cuts every expert's rows into tasks - by NUMA node when the pool splits them (numa.hpp),
// with the gate/up rows from ExpertJobMulti::gu_blob when the batch's format is a repacked one - and every row must
// still be what the row kernels give for that expert and those tokens, bit for bit.  Random batches (1-40 experts,
// 1-4 tokens each) through the fused single-barrier pipeline and the two-phase one, on IQ3_S/IQ4_NL experts with
// and without the IQ3_S rows repacked (iq_avx2.hpp), checked against single-threaded native_gu_rows /
// native_quant_h / native_down_rows.
//
//   pool_numa_test [batches]     (default 100 per configuration)
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/native_expert.hpp"
#include "guild/kernels/cpu/numa.hpp"
#include "guild/kernels/cpu/pool.hpp"
#if __has_include("guild/kernels/cpu/iq_avx2.hpp")
#include "guild/kernels/cpu/iq_avx2.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = guild::kernels::cpu;

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "pool_numa_test: failed line %d: %s\n", __LINE__, #x);                                \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

int main(int argc, char** argv) {
#if !(GUILD_BUNDLED_CPU_ARITHMETIC && defined(GUILD_NATIVE_EXPERTS))
    std::puts("pool_numa_test: no bundled native CPU kernels, SKIPPED");
    return 77;
#else
    if (!cpu::cpu_avx2_ok()) {
        std::puts("pool_numa_test: no AVX2, SKIPPED");
        return 77;
    }
    const int batches = argc > 1 ? std::atoi(argv[1]) : 100;
    constexpr int H = 2560, FF = 640, NE = 40, MAXNT = 4;
    cpu::NativeFmt f;
    std::string err;
    CHECK(cpu::native_fmt(21, 20, H, FF, f, err)); // IQ3_S gate/up, IQ4_NL down: Qwen's most common layer
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> byte(0, 255);
    // random but valid blocks: any bits decode, the f16 scales are set to small finite values
    std::vector<std::vector<uint8_t>> blobs(NE, std::vector<uint8_t>(f.bytes));
    for (auto& b : blobs) {
        for (auto& v : b)
            v = (uint8_t) byte(rng);
        for (uint64_t r = 0; r < 2 * (uint64_t) FF; ++r)
            for (int blk = 0; blk < H / 256; ++blk) {
                const uint16_t d = 0x1400; // 2^-10
                std::memcpy(b.data() + r * f.gu_row + (uint64_t) blk * 110, &d, 2);
            }
        for (uint64_t r = 0; r < (uint64_t) H; ++r)
            for (int blk = 0; blk < FF / 32; ++blk) {
                const uint16_t d = 0x1c00; // 2^-8
                std::memcpy(b.data() + f.down_off + r * f.d_row + (uint64_t) blk * 18, &d, 2);
            }
    }
    // the repacked gate/up rows and their format, when this build and CPU have them
    std::vector<std::vector<uint8_t>> packed;
    cpu::NativeFmt pf = f;
    bool have_repack = false;
#if defined(GUILD_CPU_HAS_IQ3S_NL4)
    if (cpu::iq3s_nl4_exact()) {
        pf.gu_type = cpu::kIq3sNl4Type;
        pf.gu_row = (H / 256) * cpu::kIq3sNl4Block;
        pf.up_off = pf.gu_row * FF;
        packed.assign(NE, std::vector<uint8_t>(2 * pf.up_off));
        for (int e = 0; e < NE; ++e)
            for (int r = 0; r < 2 * FF; ++r)
                CHECK(cpu::iq3s_nl4_repack(blobs[e].data() + (size_t) r * f.gu_row, H / 256,
                                           packed[e].data() + (size_t) r * pf.gu_row));
        have_repack = true;
    }
#endif
    // MAXNT tokens of quantized activations
    std::normal_distribution<float> nd(0, 1);
    std::vector<uint8_t> act((size_t) MAXNT * cpu::kNativeActBytes);
    std::vector<float> x(H);
    for (int t = 0; t < MAXNT; ++t) {
        for (auto& v : x)
            v = nd(rng);
        cpu::native_quant_act(f, x.data(), act.data() + (size_t) t * cpu::kNativeActBytes);
    }
    cpu::ExpertPool pool;
    std::printf("pool_numa_test: %d workers + host, NUMA split %s, repacked rows %s\n", pool.workers(),
                pool.numa_split() ? "on" : "off (one node, or not requested)", have_repack ? "on" : "not built");
    // reference rows: the kernels on one thread, the token set as the pool sees it
    std::vector<float> want((size_t) NE * MAXNT * H), got((size_t) NE * MAXNT * H);
    std::uniform_int_distribution<int> n_jobs(1, NE), n_tok(1, MAXNT);
    long long rows = 0;
    for (int fused = 0; fused < 2; ++fused)
        for (int repack = 0; repack < (have_repack ? 2 : 1); ++repack) {
            pool.set_fused_pipeline(fused != 0);
            for (int b = 0; b < batches; ++b) {
                const int n = n_jobs(rng);
                std::vector<cpu::ExpertJobMulti> jobs((size_t) n);
                std::vector<int> pick(NE);
                for (int e = 0; e < NE; ++e)
                    pick[e] = e;
                std::shuffle(pick.begin(), pick.end(), rng);
                for (int j = 0; j < n; ++j) {
                    auto& jb = jobs[(size_t) j];
                    const int e = pick[(size_t) j];
                    jb.blob = blobs[(size_t) e].data();
                    jb.gu_blob = repack ? packed[(size_t) e].data() : nullptr;
                    jb.nt = n_tok(rng);
                    for (int t = 0; t < jb.nt; ++t) {
                        jb.nact[t] = act.data() + (size_t) ((t + j) % MAXNT) * cpu::kNativeActBytes;
                        jb.out[t] = got.data() + ((size_t) j * MAXNT + t) * H;
                    }
                }
                pool.run_split_multi_native(repack ? pf : f, jobs.data(), n);
                for (int j = 0; j < n; ++j) {
                    const auto& jb = jobs[(size_t) j];
                    alignas(64) float ff[MAXNT][FF];
                    alignas(64) uint8_t hq[MAXNT][cpu::kNativeHBytes];
                    float* ffp[MAXNT];
                    const void* hqp[MAXNT];
                    float* outp[MAXNT];
                    for (int t = 0; t < jb.nt; ++t) {
                        ffp[t] = ff[t];
                        hqp[t] = hq[t];
                        outp[t] = want.data() + ((size_t) j * MAXNT + t) * H;
                    }
                    cpu::native_gu_rows(f, jb.blob, jb.nact, jb.nt, ffp, 0, FF);
                    for (int t = 0; t < jb.nt; ++t)
                        cpu::native_quant_h(f, ff[t], hq[t]);
                    cpu::native_down_rows(f, jb.blob, hqp, jb.nt, outp, 0, H);
                    for (int t = 0; t < jb.nt; ++t) {
                        if (std::memcmp(outp[t], jb.out[t], H * sizeof(float)) != 0) {
                            std::fprintf(stderr,
                                         "pool_numa_test: batch %d (fused %d, repacked %d): job %d of %d, token %d of "
                                         "%d differs\n",
                                         b, fused, repack, j, n, t, jb.nt);
                            return 1;
                        }
                        ++rows;
                    }
                }
            }
        }
    std::printf("pool_numa_test: %lld output rows bitwise equal to the kernels called directly\n", rows);
    return 0;
#endif
}
