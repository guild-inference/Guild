// src/core/glm5_nl4.cpp - GLM-5.3-Flash's IQ3_S expert rows in guild-kernels-cpu's NL4 form (iq_avx2.hpp:
// kIq3sNl4Type, 134 bytes per 256 values, the grid lookup done at load), dotted one row at a time against Q8_K
// activations: the gate and the up products separately, so the SwiGLU clamp can follow (the library's NL4 kernel
// fuses them into silu(g) * u).  The arithmetic is the library's nl4_row_dot: one token sums its 8 lanes in ggml's
// order (bitwise ggml-cpu's ggml_vec_dot_iq3_s_q8_K), several tokens in the library's own.  AVX2 + FMA + F16C.
#include "guild/core/glm5_nl4.hpp"

#include <immintrin.h>

#include <cstring>

namespace guild::core::glm5cpu {
namespace {

constexpr int kQK = 256;
constexpr size_t kNl4Block = 134;

struct BlockQ8K { // ggml's block_q8_K
    float d;
    int8_t qs[kQK];
    int16_t bsums[kQK / 16];
};
static_assert(sizeof(BlockQ8K) == 292, "block_q8_K is 292 bytes");

inline float h2f(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

inline float hsum_ggml(__m256 x) { // ggml-cpu's hsum_float_8
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

inline float hsum8(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

template <int NT, bool GGML>
inline void row_dot(const uint8_t* row, int nblocks, const BlockQ8K* const* y, float* res) {
    // code c -> |2c - 15| (the maddubs operand) and its sign, applied to the activation
    const __m256i lut_abs = _mm256_setr_epi8(15, 13, 11, 9, 7, 5, 3, 1, 1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5,
                                             3, 1, 1, 3, 5, 7, 9, 11, 13, 15);
    const __m256i lut_sgn = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t)
        accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * kNl4Block;
        _mm_prefetch((const char*) blk + 2048, _MM_HINT_T0);
        _mm_prefetch((const char*) blk + 2048 + 64, _MM_HINT_T0);
        __m256i acci[NT];
        for (int t = 0; t < NT; ++t)
            acci[t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i bits = _mm256_loadu_si256((const __m256i*) (blk + 6 + 32 * j));
            const uint8_t s = blk[2 + j];
            for (int half = 0; half < 2; ++half) {
                const __m256i c = half ? _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4) : _mm256_and_si256(bits, m4);
                const __m256i g = _mm256_shuffle_epi8(lut_abs, c), sgn = _mm256_shuffle_epi8(lut_sgn, c);
                const __m256i sc = _mm256_set1_epi16(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
                const int off = 64 * j + 32 * half;
                for (int t = 0; t < NT; ++t) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + off));
                    const __m256i ys = _mm256_sign_epi8(yv, sgn);
                    acci[t] = _mm256_add_epi32(acci[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g, ys), sc));
                }
            }
        }
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        const float dx = h2f(dh);
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t)
        res[t] = GGML ? hsum_ggml(accf[t]) : hsum8(accf[t]);
}

template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const BlockQ8K* y[NT];
    for (int t = 0; t < NT; ++t)
        y[t] = (const BlockQ8K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT, NT == 1>(w + (size_t) r * row_bytes, n / kQK, y, res);
        for (int t = 0; t < NT; ++t)
            out[t][r] = res[t];
    }
}

} // namespace

void nl4_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
              int r1) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
        case 1: rows<1>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
        case 2: rows<2>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
        case 3: rows<3>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
        default: rows<4>(w, row_bytes, n, act + t0, out + t0, r0, r1); break;
        }
    }
}

} // namespace guild::core::glm5cpu
