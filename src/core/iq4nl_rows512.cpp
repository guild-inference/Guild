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

#include <cstdlib>
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
// their magnitudes: codes 0-7 are the negative values
alignas(16) const int8_t kAbs[16] = {127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113};

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
void rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1,
          int pf_rows) {
    const BlockQ8_0* y[NT];
    for (int t = 0; t < NT; ++t)
        y[t] = (const BlockQ8_0*) act[t];
    float a[NT], b[NT];
    for (int r = r0; r < r1; r += 2) {
        const uint8_t* row0 = w + (size_t) r * row_bytes;
        const bool two = r + 1 < r1;
        // the two rows pf_rows ahead, while this pair is read
        const char* pf = (const char*) row0 + (size_t) pf_rows * row_bytes;
        row2<NT>(row0, two ? row0 + row_bytes : row0, n / 32, y, a, b, pf, pf + row_bytes);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = a[t];
            if (two)
                out[t][r + 1] = b[t];
        }
    }
}

// ---- v2 (GUILD_IQ4NL_V2=0 keeps the kernel above): the same lanes, the same integer sums and the same float steps,
// with the scales' work taken out of the token loop.  Per call, every token's block scales once (dy, floats).  Per
// block pair, the pair of rows' four weight scales by one conversion, spread into the two blocks' scale vectors
// (row 0's d_x in the low 8 lanes, row 1's in the high 8) by two permutes; a block's d_y * d_x is then one multiply of
// that vector by the token's scale, broadcast from memory: the single-precision product ggml forms (fp16 -> fp32 is
// exact, and the multiply rounds once, the same for a * b and b * a), lane for lane.  The high nibbles come from one
// 64-bit variable shift instead of a shift and a blend.  v1 converted d_y for every row pair and built each product
// with two scalar multiplies, two broadcasts and an insert; on a Xeon Gold 6130 the down rows ran compute-bound at
// 145 / 89 / 70 / 56 GB/s for 1-4 tokens on 32 cores (640-value rows), v2 at 170 / 139 / 112 / 95.
constexpr int kMaxBlocks = 128; // rows of up to 4096 values (Qwen3.8-Flash-Next's down rows: 640)

// the codes of block b0 (row 0) and b1 (row 1): 128-bit lanes b0's low nibbles, its high nibbles, b1's likewise
inline __m512i codes2(const uint8_t* b0, const uint8_t* b1, __m512i m4, __m512i shift) {
    const __m512i bits = _mm512_mask_broadcast_i32x4(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) (b0 + 2))),
                                                     0xFF00, _mm_loadu_si128((const __m128i*) (b1 + 2)));
    return _mm512_and_si512(_mm512_srlv_epi64(bits, shift), m4);
}
// |w| and the lanes where w < 0.  One token: from the values (shuffle, abs, sign bits); more: from the codes (a
// shuffle of the magnitudes, and bit 3 clear), one instruction fewer on port 0, where the token loop's multiplies run
// (2-4 tokens 5-6% faster, one the same).  The same bytes and the same mask either way.
template <bool LEAN>
inline void decode2v(const uint8_t* b0, const uint8_t* b1, __m512i values, __m512i absv, __m512i m4, __m512i eight,
                     __m512i shift, __m512i& aw, __mmask64& neg) {
    const __m512i c = codes2(b0, b1, m4, shift);
    if constexpr (LEAN) {
        aw = _mm512_shuffle_epi8(absv, c);
        neg = _mm512_testn_epi8_mask(c, eight);
    } else {
        const __m512i w = _mm512_shuffle_epi8(values, c);
        aw = _mm512_abs_epi8(w);
        neg = _mm512_movepi8_mask(w);
    }
}

// d_x of blocks ib (words 0, 1: rows 0, 1) and ib + 1 (words 2, 3) as floats.  One token: the words inserted into a
// vector; more: assembled in a 64-bit register first (each the faster there, by 2-8%).
template <bool GPR> inline __m128 scales4(const uint8_t* a0, const uint8_t* a1) {
    uint16_t h00, h10, h01, h11;
    std::memcpy(&h00, a0, 2);
    std::memcpy(&h10, a1, 2);
    std::memcpy(&h01, a0 + kBlock, 2);
    std::memcpy(&h11, a1 + kBlock, 2);
    if constexpr (GPR) {
        const uint64_t hw = (uint64_t) h00 | ((uint64_t) h10 << 16) | ((uint64_t) h01 << 32) | ((uint64_t) h11 << 48);
        return _mm_cvtph_ps(_mm_cvtsi64_si128((long long) hw));
    } else {
        return _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, (short) h11, (short) h01, (short) h10, (short) h00));
    }
}

template <int NT>
inline void row2_v2(const uint8_t* row0, const uint8_t* row1, int nblocks, const BlockQ8_0* const* y, const float* dy,
                    float* res0, float* res1, const char* pf0, const char* pf1) {
    const __m512i values = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) kValues));
    const __m512i absv = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) kAbs));
    const __m512i m4 = _mm512_set1_epi8(0x0f), zero = _mm512_setzero_si512(), ones = _mm512_set1_epi16(1);
    const __m512i eight = _mm512_set1_epi8(8);
    const __m512i shift = _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0); // 128-bit lanes 1 and 3: the high nibbles
    const __m512i i1 = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0); // [x00 x 8 | x10 x 8]
    const __m512i i2 = _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2); // [x01 x 8 | x11 x 8]
    __m512 acc1[NT], acc2[NT];
    for (int t = 0; t < NT; ++t)
        acc1[t] = acc2[t] = _mm512_setzero_ps();
    for (int ib = 0; ib + 1 < nblocks; ib += 2) {
        if ((ib & 3) == 0) {
            _mm_prefetch(pf0 + (size_t) ib * kBlock, _MM_HINT_T0);
            _mm_prefetch(pf1 + (size_t) ib * kBlock, _MM_HINT_T0);
        }
        const uint8_t *a0 = row0 + (size_t) ib * kBlock, *a1 = row1 + (size_t) ib * kBlock;
        __m512i aw1, aw2;
        __mmask64 n1, n2;
        decode2v<(NT > 1)>(a0, a1, values, absv, m4, eight, shift, aw1, n1);
        decode2v<(NT > 1)>(a0 + kBlock, a1 + kBlock, values, absv, m4, eight, shift, aw2, n2);
        const __m512 f = _mm512_castps128_ps512(scales4<(NT > 1)>(a0, a1));
        const __m512 x1 = _mm512_permutexvar_ps(i1, f), x2 = _mm512_permutexvar_ps(i2, f);
        for (int t = 0; t < NT; ++t) {
            const BlockQ8_0& yb1 = y[t][ib];
            const BlockQ8_0& yb2 = y[t][ib + 1];
            const __m512i y1 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) yb1.qs));
            const __m512i y2 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) yb2.qs));
            const __m512i p1 =
                _mm512_madd_epi16(_mm512_maddubs_epi16(aw1, _mm512_mask_sub_epi8(y1, n1, zero, y1)), ones);
            const __m512i p2 =
                _mm512_madd_epi16(_mm512_maddubs_epi16(aw2, _mm512_mask_sub_epi8(y2, n2, zero, y2)), ones);
            const float* dt = dy + (size_t) t * nblocks;
            const __m512 d1 = _mm512_mul_ps(x1, _mm512_set1_ps(dt[ib])); // ggml: set1(d_y * d_x)
            const __m512 d2 = _mm512_mul_ps(x2, _mm512_set1_ps(dt[ib + 1]));
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
void rows_v2(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1,
             int pf_rows) {
    const int nb = n / 32;
    const BlockQ8_0* y[NT];
    alignas(64) float dy[NT * kMaxBlocks];
    for (int t = 0; t < NT; ++t) {
        y[t] = (const BlockQ8_0*) act[t];
        for (int b = 0; b < nb; ++b)
            dy[t * nb + b] = h2f(y[t][b].d);
    }
    float a[NT], b[NT];
    for (int r = r0; r < r1; r += 2) {
        const uint8_t* row0 = w + (size_t) r * row_bytes;
        const bool two = r + 1 < r1;
        const char* pf = (const char*) row0 + (size_t) pf_rows * row_bytes;
        row2_v2<NT>(row0, two ? row0 + row_bytes : row0, nb, y, dy, a, b, pf, pf + row_bytes);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = a[t];
            if (two)
                out[t][r + 1] = b[t];
        }
    }
}

bool v2_on() {
    static const bool v = [] {
        const char* e = std::getenv("GUILD_IQ4NL_V2");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return v;
}

// How many rows ahead a row pair's prefetch reads: the two rows from there, every second block pair (a 64-byte line
// holds ~3.5 blocks).  A 640-value row is 360 bytes, so the next pair (2, as before) is only ~720 bytes ahead of the
// loads.  2x Xeon Gold 6130, Qwen3.8-Flash-Next's down rows in 172-row tasks, six rows ahead against two: 196 -> 203
// GB/s at one token and 148 -> 157 at two on 32 threads, 55.8 -> 60.3 and 42.4 -> 45.2 on 8; but in 64-row items in a
// shuffled order (iq4nl_rows512_test --dram, IQ4NL_TEST_N=640) one token loses 2% (the six rows past an item's end
// are read for nothing) and two gain 1%.  The engine's down tasks are 256-640 rows: Qwen3.8-Flash-Next decode (GTX
// 1070, with layer 2's IQ4_XS gate/up rows, GUILD_POOL_IQ4XS) took 28.17 -> 27.79 ms of CPU experts a prose window
// and 33.87 -> 33.13 a code window in two alternated pairs, 28.20 -> 28.08 and 33.64 -> 33.22 in two more; tok/s
// moved within noise (decode is GPU-bound on that PC).  A prefetch has no architectural effect, so the bits are the
// same at any distance.  GUILD_IQ4NL_PF (read once, 0-64; default 6, 2 the earlier distance).
int pf_rows() {
    static const int v = [] {
        const char* e = std::getenv("GUILD_IQ4NL_PF");
        const int k = e ? std::atoi(e) : 6;
        return k >= 0 && k <= 64 ? k : 6;
    }();
    return v;
}

} // namespace

bool iq4nl_rows512_ok() {
    return guild::kernels::cpu::cpu_avx512bw_ok(); // GUILD_FORCE_AVX2=1: false
}

void iq4nl_rows512(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                   int r1) {
    const int pf = pf_rows();
    if (v2_on() && n / 32 <= kMaxBlocks) {
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            switch (k) {
            case 1: rows_v2<1>(w, row_bytes, n, act + t0, out + t0, r0, r1, pf); break;
            case 2: rows_v2<2>(w, row_bytes, n, act + t0, out + t0, r0, r1, pf); break;
            case 3: rows_v2<3>(w, row_bytes, n, act + t0, out + t0, r0, r1, pf); break;
            default: rows_v2<4>(w, row_bytes, n, act + t0, out + t0, r0, r1, pf); break;
            }
        }
        return;
    }
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        switch (k) {
        case 1: rows<1>(w, row_bytes, n, act + t0, out + t0, r0, r1, pf); break;
        case 2: rows<2>(w, row_bytes, n, act + t0, out + t0, r0, r1, pf); break;
        case 3: rows<3>(w, row_bytes, n, act + t0, out + t0, r0, r1, pf); break;
        default: rows<4>(w, row_bytes, n, act + t0, out + t0, r0, r1, pf); break;
        }
    }
}

} // namespace guild::core::cpu_moe
