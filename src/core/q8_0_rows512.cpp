// src/core/q8_0_rows512.cpp - Q8_0 rows against several tokens' Q8_0 activations, AVX-512BW (q8_0_rows512.hpp).
//
// ggml-cpu's AVX2 ggml_vec_dot_q8_0_q8_0, per 32-value block: d = x.d * y.d (a float product), q = the 8 int32 sums
// of 4 adjacent products (|x| times y signed by x, maddubs then madd with ones), acc = fma(d, q, acc); then
// hsum_float_8(acc).  Here the low 256 bits of a register hold row r's 8 lanes and the high 256 bits row r + 1's,
// each lane computed with the same operations (the integer sums are exact, so the masked negation in place of
// vpsignb, which AVX-512 lacks, changes nothing: where x is 0 the product is 0 either way), and each half summed by
// hsum_float_8.  A row pair's block is loaded, its signs and scales taken once for every token.
//
// The weight scales of a pair of blocks (GUILD_Q8_PAIR, read once; default on): the four fp16 scales of blocks ib and
// ib + 1 of both rows assembled in one 64-bit register, converted by one vcvtph2ps and spread into the two blocks'
// [row 0 | row 1] vectors by two vpermps, instead of two scalar conversions, two broadcasts and an insert a block.
// fp16 -> fp32 is exact, so each lane holds the same float as before; the blocks are still taken in order and each
// one's lanes computed as before, so the bits are the same.  An odd last block keeps the one-block build.
#include "guild/core/q8_0_rows512.hpp"
#include "guild/kernels/cpu/features.hpp"

#include <immintrin.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <utility>

#if defined(__GNUC__) || defined(__clang__)
#define GUILD_Q8_ROWS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define GUILD_Q8_ROWS_INLINE __forceinline
#else
#define GUILD_Q8_ROWS_INLINE inline
#endif

namespace guild::core::cpu_moe {
namespace {

struct BlockQ8 { // ggml's block_q8_0
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 34, "block_q8_0 is 34 bytes");

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

/// block ib of rows x0 and x1 against NT tokens, into acc[t]: xd the block's weight scales (row 0's in the low 8
/// lanes, row 1's in the high 8), qy and yd the tokens' values and scales of this block (inlined: the accumulators
/// stay in registers)
template <int NT>
GUILD_Q8_ROWS_INLINE void block2(const BlockQ8* x0, const BlockQ8* x1, int ib, __m512 xd, const int8_t* qy,
                                 const float* yd, __m512* acc) {
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i qx = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) x0[ib].qs)),
                                          _mm256_loadu_si256((const __m256i*) x1[ib].qs), 1);
    const __m512i ax = _mm512_abs_epi8(qx);
    const __mmask64 neg = _mm512_movepi8_mask(qx);
    for (int t = 0; t < NT; ++t) {
        const __m512i y = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (qy + t * 32)));
        const __m512i sy = _mm512_mask_sub_epi8(y, neg, zero, y);
        const __m512 q = _mm512_cvtepi32_ps(_mm512_madd_epi16(ones, _mm512_maddubs_epi16(ax, sy)));
        const __m512 d = _mm512_mul_ps(xd, _mm512_set1_ps(yd[t]));
        acc[t] = _mm512_fmadd_ps(d, q, acc[t]);
    }
}

/// rows x0 and x1 (nb blocks) against NT tokens: s0[t], s1[t].  The tokens' blocks packed block by block (one pointer
/// for all of them): qy[(ib * NT + t) * 32 ...], the int8 values, and yd[ib * NT + t], the scales as floats.  pf (may
/// be null): two rows, pf_step bytes apart, asked for block by block while these two are read.  PAIR: the weight
/// scales a block pair at a time (see the top of the file).
template <int NT, bool PAIR>
void rows2(const BlockQ8* x0, const BlockQ8* x1, int nb, const int8_t* qy, const float* yd, float* s0, float* s1,
           const char* pf, size_t pf_step) {
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t)
        acc[t] = _mm512_setzero_ps();
    int ib = 0;
    if constexpr (PAIR) {
        const __m512i i1 = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0); // [x0[ib].d | x1[ib].d]
        const __m512i i2 = _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2); // block ib + 1's
        for (; ib + 1 < nb; ib += 2) {
            if (pf) {
                _mm_prefetch(pf + (size_t) ib * sizeof(BlockQ8), _MM_HINT_T0);
                _mm_prefetch(pf + pf_step + (size_t) ib * sizeof(BlockQ8), _MM_HINT_T0);
                _mm_prefetch(pf + (size_t) (ib + 1) * sizeof(BlockQ8), _MM_HINT_T0);
                _mm_prefetch(pf + pf_step + (size_t) (ib + 1) * sizeof(BlockQ8), _MM_HINT_T0);
            }
            const uint64_t hw = (uint64_t) x0[ib].d | ((uint64_t) x1[ib].d << 16) | ((uint64_t) x0[ib + 1].d << 32) |
                                ((uint64_t) x1[ib + 1].d << 48);
            const __m512 f = _mm512_castps128_ps512(_mm_cvtph_ps(_mm_cvtsi64_si128((long long) hw)));
            block2<NT>(x0, x1, ib, _mm512_permutexvar_ps(i1, f), qy + (size_t) ib * NT * 32, yd + (size_t) ib * NT,
                       acc);
            block2<NT>(x0, x1, ib + 1, _mm512_permutexvar_ps(i2, f), qy + (size_t) (ib + 1) * NT * 32,
                       yd + (size_t) (ib + 1) * NT, acc);
        }
    }
    for (; ib < nb; ++ib) {
        if (pf) {
            _mm_prefetch(pf + (size_t) ib * sizeof(BlockQ8), _MM_HINT_T0);
            _mm_prefetch(pf + pf_step + (size_t) ib * sizeof(BlockQ8), _MM_HINT_T0);
        }
        const __m512 xd = _mm512_insertf32x8(_mm512_set1_ps(h2f(x0[ib].d)), _mm256_set1_ps(h2f(x1[ib].d)), 1);
        block2<NT>(x0, x1, ib, xd, qy + (size_t) ib * NT * 32, yd + (size_t) ib * NT, acc);
    }
    for (int t = 0; t < NT; ++t) {
        s0[t] = hsum_ggml(_mm512_castps512_ps256(acc[t]));
        s1[t] = hsum_ggml(_mm512_extractf32x8_ps(acc[t], 1));
    }
}

using Rows2Fn = void (*)(const BlockQ8*, const BlockQ8*, int, const int8_t*, const float*, float*, float*, const char*,
                         size_t);

template <bool PAIR, int... I> constexpr auto table(std::integer_sequence<int, I...>) {
    return std::array<Rows2Fn, sizeof...(I)>{&rows2<I + 1, PAIR>...};
}

bool pair_on() {
    static const bool v = [] {
        const char* e = std::getenv("GUILD_Q8_PAIR");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return v;
}

} // namespace

bool q8_0_rows512_ok() {
    return guild::kernels::cpu::cpu_avx512bw_ok(); // GUILD_FORCE_AVX2=1: false
}

size_t q8_0_packed_bytes(int n, int nt) {
    return ((size_t) nt * (n / 32) * 36 + 63) / 64 * 64;
}

void q8_0_pack(const void* const* act, int nt, int n, uint8_t* packed) {
    const int nb = n / 32;
    float* yd = (float*) (packed + (size_t) nt * nb * 32);
    for (int t = 0; t < nt; ++t) {
        const BlockQ8* y = (const BlockQ8*) act[t];
        for (int ib = 0; ib < nb; ++ib) {
            std::memcpy(packed + ((size_t) ib * nt + t) * 32, y[ib].qs, 32);
            yd[(size_t) ib * nt + t] = h2f(y[ib].d);
        }
    }
}

void q8_0_rows512(const uint8_t* w, size_t row_bytes, int n, const uint8_t* packed, int nt, float* const* out, int r0,
                  int r1, const uint8_t* next, size_t next_row_bytes) {
    static constexpr auto fns = table<false>(std::make_integer_sequence<int, kQ8Rows512MaxT>{});
    static constexpr auto fns_pair = table<true>(std::make_integer_sequence<int, kQ8Rows512MaxT>{});
    const int nb = n / 32;
    const int8_t* qy = (const int8_t*) packed;
    const float* yd = (const float*) (packed + (size_t) nt * nb * 32);
    const Rows2Fn f = (pair_on() ? fns_pair : fns)[(size_t) nt - 1];
    for (int r = r0; r < r1; r += 2) {
        const BlockQ8* x0 = (const BlockQ8*) (w + (size_t) r * row_bytes);
        const bool two = r + 1 < r1;
        const BlockQ8* x1 = two ? (const BlockQ8*) ((const uint8_t*) x0 + row_bytes) : x0;
        float s0[kQ8Rows512MaxT], s1[kQ8Rows512MaxT];
        // the DRAM stream: the next two rows while these are read (2x Xeon Gold 6130, every core streaming its own
        // rows, one token: 187 GB/s without, 215 with), then the caller's next rows
        const bool own = r + 3 < r1;
        const char* pf = own ? (const char*) x0 + 2 * row_bytes : (const char*) next;
        f(x0, x1, nb, qy, yd, s0, s1, pf, own ? row_bytes : next_row_bytes);
        for (int t = 0; t < nt; ++t) {
            out[t][r] = s0[t];
            if (two)
                out[t][r + 1] = s1[t];
        }
    }
}

} // namespace guild::core::cpu_moe
