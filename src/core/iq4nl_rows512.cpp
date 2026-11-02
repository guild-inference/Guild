// src/core/iq4nl_rows512.cpp - IQ4_NL rows against several tokens' Q8_0 activations, AVX-512BW (iq4nl_rows512.hpp).
//
// ggml-cpu's AVX2 ggml_vec_dot_iq4_nl_q8_0, per pair of 32-value blocks: each block's 16 bytes give a 256-bit vector
// of codebook values (the low nibbles' values in the low 128 bits, the high nibbles' in the high), multiplied with the
// activation bytes as |w| times y signed by w (maddubs), summed in pairs (madd with ones), and added in float with one
// fmadd of d_y * d_x per block: the even blocks into one accumulator, the odd ones into another; at the end
// hsum_float_8 of their sum.  Here the low 256 bits of a register hold row r's lanes and the high 256 bits row r + 1's,
// each lane computed with the same operations (the integer sums are exact, so the masked negation in place of
// vpsignb, which AVX-512 lacks, changes nothing: the codebook has no 0 and no -128), the activation bytes loaded once
// for both rows.  So each row is ggml's one-token dot to the bit, alone or with other tokens.
#include "guild/core/iq4nl_rows512.hpp"
#include "guild/kernels/cpu/features.hpp"

#include <immintrin.h>

#include <cstring>

namespace guild::core::cpu_moe {
namespace {

constexpr size_t kBlock = 18; // d, qs[16]

struct BlockQ8_0 { // ggml's block_q8_0
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "block_q8_0 is 34 bytes");

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

// block i of rows 0 and 1 as codebook values: row 0's in the low 256 bits, row 1's in the high, each half its block's
// low nibbles then its high nibbles
inline __m512i decode2(const uint8_t* b0, const uint8_t* b1, __m512i values, __m512i m4) {
    const __m128i q0 = _mm_loadu_si128((const __m128i*) (b0 + 2));
    const __m128i q1 = _mm_loadu_si128((const __m128i*) (b1 + 2));
    const __m512i bits = _mm512_mask_broadcast_i32x4(_mm512_broadcast_i32x4(q0), 0xFF00, q1);
    const __m512i nib =
        _mm512_mask_blend_epi64(0xCC, _mm512_and_si512(bits, m4), _mm512_and_si512(_mm512_srli_epi16(bits, 4), m4));
    return _mm512_shuffle_epi8(values, nib);
}

template <int NT>
void row2(const uint8_t* row0, const uint8_t* row1, int nblocks, const BlockQ8_0* const* y, float* res0, float* res1,
          const char* pf0, const char* pf1) {
    const __m512i values = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) kValues));
    const __m512i m4 = _mm512_set1_epi8(0x0f), zero = _mm512_setzero_si512(), ones = _mm512_set1_epi16(1);
    __m512 acc1[NT], acc2[NT];
    for (int t = 0; t < NT; ++t)
        acc1[t] = acc2[t] = _mm512_setzero_ps();
    for (int ib = 0; ib + 1 < nblocks; ib += 2) {
        if ((ib & 3) == 0) { // a 64-byte line covers ~3.5 blocks
            _mm_prefetch(pf0 + (size_t) ib * kBlock, _MM_HINT_T0);
            _mm_prefetch(pf1 + (size_t) ib * kBlock, _MM_HINT_T0);
        }
        const uint8_t *a0 = row0 + (size_t) ib * kBlock, *a1 = row1 + (size_t) ib * kBlock;
        const __m512i w1 = decode2(a0, a1, values, m4), w2 = decode2(a0 + kBlock, a1 + kBlock, values, m4);
        const __m512i aw1 = _mm512_abs_epi8(w1), aw2 = _mm512_abs_epi8(w2);
        const __mmask64 n1 = _mm512_movepi8_mask(w1), n2 = _mm512_movepi8_mask(w2);
        uint16_t h00, h01, h10, h11;
        std::memcpy(&h00, a0, 2);
        std::memcpy(&h10, a1, 2);
        std::memcpy(&h01, a0 + kBlock, 2);
        std::memcpy(&h11, a1 + kBlock, 2);
        const float x00 = h2f(h00), x10 = h2f(h10), x01 = h2f(h01), x11 = h2f(h11);
        for (int t = 0; t < NT; ++t) {
            const BlockQ8_0& yb1 = y[t][ib];
            const BlockQ8_0& yb2 = y[t][ib + 1];
            const __m512i y1 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) yb1.qs));
            const __m512i y2 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) yb2.qs));
            const __m512i p1 =
                _mm512_madd_epi16(_mm512_maddubs_epi16(aw1, _mm512_mask_sub_epi8(y1, n1, zero, y1)), ones);
            const __m512i p2 =
                _mm512_madd_epi16(_mm512_maddubs_epi16(aw2, _mm512_mask_sub_epi8(y2, n2, zero, y2)), ones);
            // ggml: set1(d_y * d_x), each product a float multiply
            const float dy1 = h2f(yb1.d), dy2 = h2f(yb2.d);
            const __m512 d1 = _mm512_insertf32x8(_mm512_set1_ps(dy1 * x00), _mm256_set1_ps(dy1 * x10), 1);
            const __m512 d2 = _mm512_insertf32x8(_mm512_set1_ps(dy2 * x01), _mm256_set1_ps(dy2 * x11), 1);
            acc1[t] = _mm512_fmadd_ps(d1, _mm512_cvtepi32_ps(p1), acc1[t]);
            acc2[t] = _mm512_fmadd_ps(d2, _mm512_cvtepi32_ps(p2), acc2[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m512 s = _mm512_add_ps(acc1[t], acc2[t]); // ggml: hsum_float_8(accum1 + accum2)
        res0[t] = hsum_ggml(_mm512_castps512_ps256(s));
        res1[t] = hsum_ggml(_mm512_extractf32x8_ps(s, 1));
    }
}

template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const BlockQ8_0* y[NT];
    for (int t = 0; t < NT; ++t)
        y[t] = (const BlockQ8_0*) act[t];
    float a[NT], b[NT];
    for (int r = r0; r < r1; r += 2) {
        const uint8_t* row0 = w + (size_t) r * row_bytes;
        const bool two = r + 1 < r1;
        // the rows after this pair, while it is read
        const char* pf = (const char*) row0 + 2 * row_bytes;
        row2<NT>(row0, two ? row0 + row_bytes : row0, n / 32, y, a, b, pf, pf + row_bytes);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = a[t];
            if (two)
                out[t][r + 1] = b[t];
        }
    }
}

} // namespace

bool iq4nl_rows512_ok() {
    return guild::kernels::cpu::cpu_avx512bw_ok(); // GUILD_FORCE_AVX2=1: false
}

void iq4nl_rows512(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
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

} // namespace guild::core::cpu_moe
