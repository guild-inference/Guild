#include "guild/kernels/cpu/expert.hpp"
#include <cstring>
#include <initializer_list>
#include <cstdio>

int main() {
    using guild::kernels::cpu::CpuFeatures;
    CpuFeatures all;
    all.avx512f = all.avx512bw = all.avx512dq = all.avx512vl = all.avx512_vnni = all.avx512_vbmi = true;
    all.fma = all.f16c = all.os_avx512 = true;
    if (!all.usable()) return 1;
    for (auto bit : {&CpuFeatures::avx512f, &CpuFeatures::avx512bw, &CpuFeatures::avx512dq,
                    &CpuFeatures::avx512vl, &CpuFeatures::avx512_vnni, &CpuFeatures::avx512_vbmi,
                    &CpuFeatures::fma, &CpuFeatures::f16c, &CpuFeatures::os_avx512}) {
        auto missing = all;
        missing.*bit = false;
        if (missing.usable() || std::strcmp(missing.reason(), "ok") == 0) return 1;
    }
    const auto host = guild::kernels::cpu::cpu_features();
    std::printf("expert ISA probe: %s\n", host.reason());
    return 0;
}
