// src/core/iq4xs_rows512.cpp - IQ4_XS rows against several tokens' Q8_K activations, AVX-512BW (iq4xs_rows512.hpp).
//
// ggml-cpu's AVX2 ggml_vec_dot_iq4_xs_q8_K, per 256-value block: each 32-value sub-block's 16 bytes give a 256-bit
// vector of codebook values (the low nibbles' values in the low 128 bits, the high nibbles' in the high), multiplied
// with the activation bytes as |w| times y signed by w (maddubs), then by the sub-block's signed 6-bit scale (madd),
// summed as int32 lanes; per block one fmadd of d_x * d_y; at the end hsum_float_8.  Here the low 256 bits of a
// register hold row r's lanes and the high 256 bits row r + 1's, each lane computed with the same operations (the
// integer sums are exact, so their order and the masked negation in place of vpsignb, which AVX-512 lacks, change
// nothing: the codebook has no 0 and no -128), the activation bytes loaded once for both rows, and the eight scales
// of a block decoded together.  So each row is ggml's one-token dot to the bit, alone or with other tokens.
#include "guild/core/iq4xs_rows512.hpp"
#include "guild/kernels/cpu/features.hpp"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

namespace guild::core::cpu_moe {
namespace {

constexpr int kQK = 256;
constexpr size_t kBlock = 136; // d, scales_h, scales_l[4], qs[128]

struct BlockQ8K { // ggml's block_q8_K
    float d;
    int8_t qs[kQK];
    int16_t bsums[kQK / 16];
};
static_assert(sizeof(BlockQ8K) == 292, "block_q8_K is 292 bytes");

// ggml's kvalues_iq4nl
alignas(16) const int8_t kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

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

// a block's sub-block scales ls - 32 as int16: row 0's eight in elements 0-7, row 1's in 8-15 (ls of sub-block h is
// the nibble h of scales_l, as a 32-bit word, and the bit pair h of scales_h above it)
inline __m256i scales16(const uint8_t* b0, const uint8_t* b1) {
    uint16_t h0, h1;
    uint32_t l0, l1;
    std::memcpy(&h0, b0 + 2, 2);
    std::memcpy(&h1, b1 + 2, 2);
    std::memcpy(&l0, b0 + 4, 4);
    std::memcpy(&l1, b1 + 4, 4);
    const __m512i sl = _mm512_mask_set1_epi32(_mm512_set1_epi32((int) l0), 0xFF00, (int) l1);
    const __m512i sh = _mm512_mask_set1_epi32(_mm512_set1_epi32((int) h0), 0xFF00, (int) h1);
    const __m512i lo_shift = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28);
    const __m512i hi_shift = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i lo = _mm512_and_si512(_mm512_srlv_epi32(sl, lo_shift), _mm512_set1_epi32(0xf));
    const __m512i hi = _mm512_slli_epi32(_mm512_and_si512(_mm512_srlv_epi32(sh, hi_shift), _mm512_set1_epi32(3)), 4);
    return _mm512_cvtepi32_epi16(_mm512_sub_epi32(_mm512_or_si512(lo, hi), _mm512_set1_epi32(32)));
}

template <int NT>
void row2(const uint8_t* row0, const uint8_t* row1, int nblocks, const BlockQ8K* const* y, float* res0, float* res1,
          const char* pf0, const char* pf1) {
    const __m512i values = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) kValues));
    const __m512i m4 = _mm512_set1_epi8(0x0f), zero = _mm512_setzero_si512();
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t)
        accf[t] = _mm512_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* b0 = row0 + (size_t) i * kBlock;
        const uint8_t* b1 = row1 + (size_t) i * kBlock;
        _mm_prefetch(pf0 + (size_t) i * kBlock, _MM_HINT_T0);
        _mm_prefetch(pf0 + (size_t) i * kBlock + 64, _MM_HINT_T0);
        _mm_prefetch(pf1 + (size_t) i * kBlock, _MM_HINT_T0);
        _mm_prefetch(pf1 + (size_t) i * kBlock + 64, _MM_HINT_T0);
        const __m512i sc16 = _mm512_castsi256_si512(scales16(b0, b1));
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t)
            acci[t] = _mm512_setzero_si512();
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            // 128-bit lanes: row 0's 16 bytes (low nibbles), again (high nibbles), row 1's likewise
            const __m128i q0 = _mm_loadu_si128((const __m128i*) (b0 + 8 + 16 * h));
            const __m128i q1 = _mm_loadu_si128((const __m128i*) (b1 + 8 + 16 * h));
            const __m512i bits = _mm512_mask_broadcast_i32x4(_mm512_broadcast_i32x4(q0), 0xFF00, q1);
            const __m512i nib = _mm512_mask_blend_epi64(0xCC, _mm512_and_si512(bits, m4),
                                                        _mm512_and_si512(_mm512_srli_epi16(bits, 4), m4));
            const __m512i w = _mm512_shuffle_epi8(values, nib);
            const __m512i aw = _mm512_abs_epi8(w);
            const __mmask64 neg = _mm512_movepi8_mask(w);
            const __m512i sc = _mm512_permutexvar_epi16(
                _mm512_inserti64x4(_mm512_set1_epi16((short) h), _mm256_set1_epi16((short) (8 + h)), 1), sc16);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (y[t][i].qs + 32 * h)));
                const __m512i ys = _mm512_mask_sub_epi8(yv, neg, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(aw, ys), sc));
            }
        }
        uint16_t d0, d1;
        std::memcpy(&d0, b0, 2);
        std::memcpy(&d1, b1, 2);
        const float x0 = h2f(d0), x1 = h2f(d1);
        for (int t = 0; t < NT; ++t) {
            // ggml: set1(d_x * d_y), each product a float multiply
            const __m512 dd = _mm512_insertf32x8(_mm512_set1_ps(x0 * y[t][i].d), _mm256_set1_ps(x1 * y[t][i].d), 1);
            accf[t] = _mm512_fmadd_ps(dd, _mm512_cvtepi32_ps(acci[t]), accf[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        res0[t] = hsum_ggml(_mm512_castps512_ps256(accf[t]));
        res1[t] = hsum_ggml(_mm512_extractf32x8_ps(accf[t], 1));
    }
}

template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1,
          const uint8_t* next, size_t next_row_bytes) {
    const BlockQ8K* y[NT];
    for (int t = 0; t < NT; ++t)
        y[t] = (const BlockQ8K*) act[t];
    float a[NT], b[NT];
    for (int r = r0; r < r1; r += 2) {
        const uint8_t* row0 = w + (size_t) r * row_bytes;
        const bool two = r + 1 < r1;
        // the DRAM stream: the next two rows while these two are read, the caller's next rows after the last pair
        const bool last = r + 2 >= r1;
        const char* pf = last && next ? (const char*) next : (const char*) row0 + 2 * row_bytes;
        const size_t step = last && next ? next_row_bytes : row_bytes;
        row2<NT>(row0, two ? row0 + row_bytes : row0, n / kQK, y, a, b, pf, pf + step);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = a[t];
            if (two)
                out[t][r + 1] = b[t];
        }
    }
}

} // namespace

bool iq4xs_rows512_ok() {
    return guild::kernels::cpu::cpu_avx512bw_ok(); // GUILD_FORCE_AVX2=1: false
}

void iq4xs_rows512(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                   int r1, const uint8_t* next, size_t next_row_bytes) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        const bool last = t0 + 4 >= nt;
        const uint8_t* nx = last ? next : nullptr;
        switch (k) {
        case 1: rows<1>(w, row_bytes, n, act + t0, out + t0, r0, r1, nx, next_row_bytes); break;
        case 2: rows<2>(w, row_bytes, n, act + t0, out + t0, r0, r1, nx, next_row_bytes); break;
        case 3: rows<3>(w, row_bytes, n, act + t0, out + t0, r0, r1, nx, next_row_bytes); break;
        default: rows<4>(w, row_bytes, n, act + t0, out + t0, r0, r1, nx, next_row_bytes); break;
        }
    }
}

} // namespace guild::core::cpu_moe
