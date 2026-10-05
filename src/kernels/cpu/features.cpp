// Baseline x86 CPU probe: never compile this translation unit with the expert kernel's ISA flags.
#include "guild/kernels/cpu/expert.hpp"
#include <cstdio>
#include <cstdlib>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace guild::kernels::cpu {
const char* CpuFeatures::reason() const {
    if (usable()) return "ok";
    thread_local char buf[240];
    std::snprintf(buf, sizeof buf, "missing %s%s%s%s%s%s%s%s%s",
        avx512f ? "" : "AVX512F ", avx512bw ? "" : "AVX512BW ", avx512dq ? "" : "AVX512DQ ",
        avx512vl ? "" : "AVX512VL ", avx512_vnni ? "" : "AVX512-VNNI ",
        avx512_vbmi ? "" : "AVX512-VBMI ", fma ? "" : "FMA ", f16c ? "" : "F16C ",
        os_avx512 ? "" : "OS AVX-512 state");
    return buf;
}

CpuFeatures cpu_features() {
    auto cpuid = [](unsigned leaf, unsigned sub, unsigned (&r)[4]) {
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, int(leaf), int(sub));
        for (int i = 0; i < 4; ++i) r[i] = unsigned(x[i]);
#else
        __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
    };
    CpuFeatures f;
    unsigned r[4] = {};
    cpuid(0, 0, r);
    if (r[0] < 7) return f;
    cpuid(1, 0, r);
    f.fma = (r[2] >> 12) & 1u;
    f.f16c = (r[2] >> 29) & 1u;
    if ((r[2] & ((1u << 27) | (1u << 28))) == ((1u << 27) | (1u << 28))) {
#if defined(_MSC_VER)
        const auto xcr0 = _xgetbv(0);
#else
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const auto xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
        f.os_avx512 = (xcr0 & 0xE6) == 0xE6; // SSE, YMM, opmask and ZMM state
    }
    cpuid(7, 0, r);
    f.avx512f = (r[1] >> 16) & 1u;
    f.avx512dq = (r[1] >> 17) & 1u;
    f.avx512bw = (r[1] >> 30) & 1u;
    f.avx512vl = (r[1] >> 31) & 1u;
    f.avx512_vnni = (r[2] >> 11) & 1u;
    f.avx512_vbmi = (r[2] >> 1) & 1u;
    return f;
}

void cpu_require_expert_support() {
    const auto f = cpu_features();
    if (f.usable()) return;
    std::fprintf(stderr, "guild: this CPU cannot run the VNNI expert kernel: %s.\n"
        "       The runtime dispatch can use AVX2/native experts; this direct kernel call requires AVX-512.\n", f.reason());
    std::exit(1);
}
} // namespace guild::kernels::cpu
