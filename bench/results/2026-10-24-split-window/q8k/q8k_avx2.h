#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
struct BlockQ8K { float d; int8_t qs[256]; int16_t bsums[16]; };
__attribute__((target("avx2,fma"))) inline void q8k_avx2(const float* x, void* vy, int64_t k) {
    auto* y = static_cast<BlockQ8K*>(vy);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int64_t i = 0; i < k / 256; ++i, x += 256) {
        __m256 m = _mm256_setzero_ps();
        for (int j = 0; j < 256; j += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + j)));
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) { y[i].d = 0; std::memset(y[i].qs, 0, 256); continue; }
        float max = 0; // the reference keeps the first element of the largest magnitude, sign included
        for (int j = 0; j < 256; ++j) if (std::fabs(x[j]) == amax) { max = x[j]; break; }
        const float iscale = -127.f / max;
        const __m256 vs = _mm256_set1_ps(iscale);
        for (int j = 0; j < 256; j += 32) {
            __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j)));
            __m256i b = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 8)));
            __m256i c = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 16)));
            __m256i d = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + j + 24)));
            const __m256i lim = _mm256_set1_epi32(127);
            a = _mm256_min_epi32(a, lim); b = _mm256_min_epi32(b, lim);
            c = _mm256_min_epi32(c, lim); d = _mm256_min_epi32(d, lim);
            // bsums: 16 values each; a,b = first 16, c,d = next 16
            const __m256i s0 = _mm256_add_epi32(a, b), s1 = _mm256_add_epi32(c, d);
            __m256i ab = _mm256_packs_epi32(a, b), cd = _mm256_packs_epi32(c, d);
            __m256i q = _mm256_packs_epi16(ab, cd);
            q = _mm256_permutevar8x32_epi32(q, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(y[i].qs + j), q);
            __m128i t0 = _mm_add_epi32(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0, 1));
            __m128i t1 = _mm_add_epi32(_mm256_castsi256_si128(s1), _mm256_extracti128_si256(s1, 1));
            t0 = _mm_hadd_epi32(t0, t1); t0 = _mm_hadd_epi32(t0, t0);
            y[i].bsums[j / 16] = (int16_t) _mm_extract_epi32(t0, 0);
            y[i].bsums[j / 16 + 1] = (int16_t) _mm_extract_epi32(t0, 1);
        }
        y[i].d = 1 / iscale;
    }
}
