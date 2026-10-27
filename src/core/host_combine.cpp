// src/core/host_combine.cpp - see include/guild/core/host_combine.hpp.  Built with AVX2 and FMA (CMakeLists.txt);
// callers check host_combine_supported() first.
#include "guild/core/host_combine.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"

#include <immintrin.h>

namespace guild::core {

bool host_combine_supported() {
    return guild::kernels::cpu::cpu_avx2_ok();
}

namespace {

// One token: 32 columns at a time in four independent chains, then 8, then one (the same ops as scalars).
void combine_row(const float* parts, const float* w, float* out, int k, int n) {
    int c = 0;
    for (; c + 32 <= n; c += 32) {
        const __m256 w0 = _mm256_set1_ps(w[0]);
        __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(parts + c), w0);
        __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(parts + c + 8), w0);
        __m256 s2 = _mm256_mul_ps(_mm256_loadu_ps(parts + c + 16), w0);
        __m256 s3 = _mm256_mul_ps(_mm256_loadu_ps(parts + c + 24), w0);
        for (int e = 1; e < k; ++e) {
            const float* p = parts + (size_t) e * (size_t) n + c;
            const __m256 we = _mm256_set1_ps(w[e]);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(p), we, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 8), we, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 16), we, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(p + 24), we, s3);
        }
        _mm256_storeu_ps(out + c, s0);
        _mm256_storeu_ps(out + c + 8, s1);
        _mm256_storeu_ps(out + c + 16, s2);
        _mm256_storeu_ps(out + c + 24, s3);
    }
    for (; c + 8 <= n; c += 8) {
        __m256 s = _mm256_mul_ps(_mm256_loadu_ps(parts + c), _mm256_set1_ps(w[0]));
        for (int e = 1; e < k; ++e)
            s = _mm256_fmadd_ps(_mm256_loadu_ps(parts + (size_t) e * (size_t) n + c), _mm256_set1_ps(w[e]), s);
        _mm256_storeu_ps(out + c, s);
    }
    for (; c < n; ++c) {
        __m128 s = _mm_mul_ss(_mm_set_ss(parts[c]), _mm_set_ss(w[0]));
        for (int e = 1; e < k; ++e)
            s = _mm_fmadd_ss(_mm_set_ss(parts[(size_t) e * (size_t) n + c]), _mm_set_ss(w[e]), s);
        out[c] = _mm_cvtss_f32(s);
    }
}

} // namespace

void host_moe_combine(const float* parts, const float* weights, float* out, int n_tok, int k, int n_embd) {
    // the kernel's -ftz=true: subnormal inputs read as zero (DAZ), subnormal results written as zero (FTZ), signs kept
    const unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040u);
    for (int t = 0; t < n_tok; ++t)
        combine_row(parts + (size_t) t * (size_t) k * (size_t) n_embd, weights + (size_t) t * (size_t) k,
                    out + (size_t) t * (size_t) n_embd, k, n_embd);
    _mm_setcsr(csr);
}

} // namespace guild::core
