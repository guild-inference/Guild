// src/core/kq_rows512.cpp - Q4_K and Q6_K expert rows two to an AVX-512BW pass, bitwise ggml-cpu's AVX2 dots
// (kq_rows512.hpp).
//
// ggml-cpu's AVX2 ggml_vec_dot_q4_K_q8_K / ggml_vec_dot_q6_K_q8_K, per 256-value block: the integer products summed
// as 8 int32 lanes (maddubs, then madd with the sub-block scales), fma(d_x * d_y, cvt(sumi), acc) on 8 lanes, Q4_K's
// mins as fma(-d_y * dmin_x, cvt(mins . bsums), acc_m) on 4 lanes; at the end hsum_float_8(acc) (+ the 4-lane sum of
// acc_m).  Here two rows go through one pass (gate row r with up row r, or down rows r and r + 1): the weights'
// nibbles and scales are built in 512 bits directly (a broadcast load, a 64-bit variable shift and an and for a
// 64-value group; one in-lane byte shuffle for its scales), the tokens' bsums hadd once a call, and the next rows are
// asked for while these are read.  Every int32 lane is the same exact sum as ggml's (integer sums do not round, and
// no maddubs pair saturates: Q4_K pairs stay below 2 * 15 * 128, Q6_K below 2 * 63 * 128), and every float step is
// ggml's, in ggml's order; so each (row, token) is ggml's one-token dot to the bit, alone or with other tokens.
//
// Ported from the round-18 scout prototype (kexp2.cpp z3pf): 2x Xeon Gold 6130, Drama 3's rows, one core:
// Q4_K gate/up 11.89 -> 13.21 GB/s, Q4_K down 8.80 -> 11.03, Q6_K down 15.23 -> 22.37 at one token (ggml's dot);
// 32 threads streaming: 199.6 -> 213.3, 176.3 -> 205.6, 209.2 -> 223.4 GB/s.
#include "guild/core/kq_rows512.hpp"
#include "guild/kernels/cpu/features.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace guild::core::cpu_moe {
namespace {

constexpr int QK_K = 256;
struct block_q4_K { // ggml's block_q4_K
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K is 144 bytes");
struct block_q6_K { // ggml's block_q6_K
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K is 210 bytes");
struct block_q8_K { // ggml's block_q8_K
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K is 292 bytes");

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

constexpr int kMaxT = 8;       // tokens a pass
constexpr int kMaxBlocks = 64; // rows of up to 16,384 values

// ---- Q4_K
// a 64-value group's scales: int16 element 2j (low nibbles) and 2j + 1 (high nibbles) in every 128-bit lane, low 256
// bits the first, high 256 the second (the order of the [low | high] nibble vector)
alignas(64) const uint8_t k4idx[4][64] = {
#define KQ(a) a, a + 1, a, a + 1, a, a + 1, a, a + 1, a, a + 1, a, a + 1, a, a + 1, a, a + 1
    {KQ(0), KQ(0), KQ(2), KQ(2)},
    {KQ(4), KQ(4), KQ(6), KQ(6)},
    {KQ(8), KQ(8), KQ(10), KQ(10)},
    {KQ(12), KQ(12), KQ(14), KQ(14)}};
#undef KQ

// ggml's utmp unpacking of the 12 scale bytes: utmp[0..1] the 8 scales, utmp[2..3] the 8 mins
inline void q4k_utmp(const uint8_t* sc_bytes, uint32_t utmp[4]) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    std::memcpy(utmp, sc_bytes, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

template <int NT>
void q4k_2rows(const block_q4_K* x0, const block_q4_K* x1, int nb, const __m128i* q8s, const void* const* act,
               float* res0, float* res1, const char* pf0, const char* pf1) {
    const __m512i m4 = _mm512_set1_epi8(0xF);
    const __m512i nshift = _mm512_setr_epi64(0, 0, 0, 0, 4, 4, 4, 4);
    __m256 acc0[NT], acc1[NT];
    __m128 acc_m0[NT], acc_m1[NT];
    for (int t = 0; t < NT; ++t) {
        acc0[t] = _mm256_setzero_ps();
        acc_m0[t] = _mm_setzero_ps();
        acc1[t] = _mm256_setzero_ps();
        acc_m1[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        // the DRAM stream: the next two rows' block while this one is read (a block is 144 bytes, three lines)
        const char* p0 = pf0 + (size_t) i * sizeof(block_q4_K);
        const char* p1 = pf1 + (size_t) i * sizeof(block_q4_K);
        _mm_prefetch(p0, _MM_HINT_T0);
        _mm_prefetch(p0 + 64, _MM_HINT_T0);
        _mm_prefetch(p0 + 128, _MM_HINT_T0);
        _mm_prefetch(p1, _MM_HINT_T0);
        _mm_prefetch(p1 + 64, _MM_HINT_T0);
        _mm_prefetch(p1 + 128, _MM_HINT_T0);
        uint16_t hd0[2], hd1[2];
        std::memcpy(hd0, &x0[i], 4);
        std::memcpy(hd1, &x1[i], 4);
        const float xd0 = h2f(hd0[0]), xdmin0 = h2f(hd0[1]);
        const float xd1 = h2f(hd1[0]), xdmin1 = h2f(hd1[1]);
        uint32_t u0[4], u1[4];
        q4k_utmp(x0[i].scales, u0);
        q4k_utmp(x1[i].scales, u1);
        const __m256i ms0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) u0)); // ggml's mins_and_scales
        const __m256i ms1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) u1));
        const __m128i mins0 = _mm256_extracti128_si256(ms0, 1), mins1 = _mm256_extracti128_si256(ms1, 1);
        const __m512i sc0 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(ms0)); // the 8 scales in every lane
        const __m512i sc1 = _mm512_broadcast_i32x4(_mm256_castsi256_si128(ms1));
        __m512i s0[NT], s1[NT];
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const __m128i q = q8s[(size_t) i * NT + t];
            acc_m0[t] = _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin0), _mm_cvtepi32_ps(_mm_madd_epi16(mins0, q)), acc_m0[t]);
            acc_m1[t] = _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin1), _mm_cvtepi32_ps(_mm_madd_epi16(mins1, q)), acc_m1[t]);
            s0[t] = _mm512_setzero_si512();
            s1[t] = _mm512_setzero_si512();
        }
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const __m512i idx = _mm512_load_si512((const void*) k4idx[j]);
            const __m512i scl0 = _mm512_shuffle_epi8(sc0, idx), scl1 = _mm512_shuffle_epi8(sc1, idx);
            const __m512i qb0 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (x0[i].qs + 32 * j)));
            const __m512i qb1 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (x1[i].qs + 32 * j)));
            const __m512i q40 = _mm512_and_si512(_mm512_srlv_epi64(qb0, nshift), m4); // [low nibbles | high]
            const __m512i q41 = _mm512_and_si512(_mm512_srlv_epi64(qb1, nshift), m4);
            for (int t = 0; t < NT; ++t) {
                const __m512i q8 = _mm512_loadu_si512((const void*) (((const block_q8_K*) act[t] + i)->qs + 64 * j));
                s0[t] = _mm512_add_epi32(s0[t], _mm512_madd_epi16(scl0, _mm512_maddubs_epi16(q40, q8)));
                s1[t] = _mm512_add_epi32(s1[t], _mm512_madd_epi16(scl1, _mm512_maddubs_epi16(q41, q8)));
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float yd = ((const block_q8_K*) act[t] + i)->d;
            const __m256i a = _mm256_add_epi32(_mm512_castsi512_si256(s0[t]), _mm512_extracti32x8_epi32(s0[t], 1));
            const __m256i b = _mm256_add_epi32(_mm512_castsi512_si256(s1[t]), _mm512_extracti32x8_epi32(s1[t], 1));
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * xd0), _mm256_cvtepi32_ps(a), acc0[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * xd1), _mm256_cvtepi32_ps(b), acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m128 m0 = acc_m0[t];
        m0 = _mm_add_ps(m0, _mm_movehl_ps(m0, m0));
        m0 = _mm_add_ss(m0, _mm_movehdup_ps(m0));
        res0[t] = hsum_ggml(acc0[t]) + _mm_cvtss_f32(m0);
        __m128 m1 = acc_m1[t];
        m1 = _mm_add_ps(m1, _mm_movehl_ps(m1, m1));
        m1 = _mm_add_ss(m1, _mm_movehdup_ps(m1));
        res1[t] = hsum_ggml(acc1[t]) + _mm_cvtss_f32(m1);
    }
}

// the tokens' bsums, hadd as ggml does, once a call: [block][token]
void q8s_of(const void* const* act, int nt, int nb, __m128i* q8s) {
    for (int i = 0; i < nb; ++i)
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            q8s[(size_t) i * nt + t] =
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        }
}

void q4k_two(const uint8_t* a, const uint8_t* b, int nb, const void* const* act, int m, const __m128i* q8s, float* ra,
             float* rb, const char* pf0, const char* pf1) {
    const block_q4_K *x0 = (const block_q4_K*) a, *x1 = (const block_q4_K*) b;
    switch (m) {
    case 1: q4k_2rows<1>(x0, x1, nb, q8s, act, ra, rb, pf0, pf1); break;
    case 2: q4k_2rows<2>(x0, x1, nb, q8s, act, ra, rb, pf0, pf1); break;
    case 3: q4k_2rows<3>(x0, x1, nb, q8s, act, ra, rb, pf0, pf1); break;
    case 4: q4k_2rows<4>(x0, x1, nb, q8s, act, ra, rb, pf0, pf1); break;
    case 5: q4k_2rows<5>(x0, x1, nb, q8s, act, ra, rb, pf0, pf1); break;
    case 6: q4k_2rows<6>(x0, x1, nb, q8s, act, ra, rb, pf0, pf1); break;
    case 7: q4k_2rows<7>(x0, x1, nb, q8s, act, ra, rb, pf0, pf1); break;
    default: q4k_2rows<8>(x0, x1, nb, q8s, act, ra, rb, pf0, pf1); break;
    }
}

// ---- Q6_K
// a 64-value group's 4 int8 scales spread to 8 values each (cvtepi8_epi16 then gives the int16 lanes)
alignas(32) const uint8_t q6idx[4][32] = {
#define S8(a) a, a, a, a, a, a, a, a
    {S8(0), S8(1), S8(2), S8(3)},
    {S8(4), S8(5), S8(6), S8(7)},
    {S8(8), S8(9), S8(10), S8(11)},
    {S8(12), S8(13), S8(14), S8(15)}};
#undef S8

template <int NT>
void q6k_2rows(const block_q6_K* x0, const block_q6_K* x1, int nb, const void* const* act, float* res0, float* res1,
               const char* pf0, const char* pf1) {
    const __m512i m15 = _mm512_set1_epi8(0xF), m48 = _mm512_set1_epi8(48);
    const __m512i sl01 = _mm512_setr_epi64(4, 4, 4, 4, 2, 2, 2, 2), sr23 = _mm512_setr_epi64(0, 0, 0, 0, 2, 2, 2, 2);
    __m256 acc0[NT], acc1[NT];
    for (int t = 0; t < NT; ++t) {
        acc0[t] = _mm256_setzero_ps();
        acc1[t] = _mm256_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        const char* p0 = pf0 + (size_t) i * sizeof(block_q6_K); // 210 bytes, four lines
        const char* p1 = pf1 + (size_t) i * sizeof(block_q6_K);
        _mm_prefetch(p0, _MM_HINT_T0);
        _mm_prefetch(p0 + 64, _MM_HINT_T0);
        _mm_prefetch(p0 + 128, _MM_HINT_T0);
        _mm_prefetch(p0 + 192, _MM_HINT_T0);
        _mm_prefetch(p1, _MM_HINT_T0);
        _mm_prefetch(p1 + 64, _MM_HINT_T0);
        _mm_prefetch(p1 + 128, _MM_HINT_T0);
        _mm_prefetch(p1 + 192, _MM_HINT_T0);
        const block_q6_K* xs[2] = {&x0[i], &x1[i]};
        float xd[2];
        __m256i s16[2];
        __m512i q01[2][2], q23[2][2], c01[2][2], c23[2][2];
#pragma GCC unroll 2
        for (int r = 0; r < 2; ++r) {
            const block_q6_K* x = xs[r];
            uint16_t dh;
            std::memcpy(&dh, &x->d, 2);
            xd[r] = h2f(dh);
            const __m128i scales = _mm_loadu_si128((const __m128i*) x->scales);
            s16[r] = _mm256_cvtepi8_epi16(scales);
            const __m256i sdup = _mm256_broadcastsi128_si256(scales);
#pragma GCC unroll 2
            for (int j = 0; j < 2; ++j) {
                const __m512i L = _mm512_loadu_si512((const void*) (x->ql + 64 * j));
                const __m512i H = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (x->qh + 32 * j)));
                q01[r][j] =
                    _mm512_or_si512(_mm512_and_si512(L, m15), _mm512_and_si512(_mm512_sllv_epi64(H, sl01), m48));
                q23[r][j] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(L, 4), m15),
                                            _mm512_and_si512(_mm512_srlv_epi64(H, sr23), m48));
                c01[r][j] =
                    _mm512_cvtepi8_epi16(_mm256_shuffle_epi8(sdup, _mm256_load_si256((const __m256i*) q6idx[2 * j])));
                c23[r][j] = _mm512_cvtepi8_epi16(
                    _mm256_shuffle_epi8(sdup, _mm256_load_si256((const __m256i*) q6idx[2 * j + 1])));
            }
        }
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
#pragma GCC unroll 2
            for (int j = 0; j < 2; ++j) {
                const __m512i y01 = _mm512_loadu_si512((const void*) (y->qs + 128 * j + 0));
                const __m512i y23 = _mm512_loadu_si512((const void*) (y->qs + 128 * j + 64));
                a0 = _mm512_add_epi32(
                    a0, _mm512_add_epi32(_mm512_madd_epi16(c01[0][j], _mm512_maddubs_epi16(q01[0][j], y01)),
                                         _mm512_madd_epi16(c23[0][j], _mm512_maddubs_epi16(q23[0][j], y23))));
                a1 = _mm512_add_epi32(
                    a1, _mm512_add_epi32(_mm512_madd_epi16(c01[1][j], _mm512_maddubs_epi16(q01[1][j], y01)),
                                         _mm512_madd_epi16(c23[1][j], _mm512_maddubs_epi16(q23[1][j], y23))));
            }
            // ggml: the 32 * scale * bsum correction (q8sclsub) subtracted from the lanes
            const __m256i sub0 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, s16[0]), 5);
            const __m256i sub1 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, s16[1]), 5);
            const __m256i v0 =
                _mm256_sub_epi32(_mm256_add_epi32(_mm512_castsi512_si256(a0), _mm512_extracti32x8_epi32(a0, 1)), sub0);
            const __m256i v1 =
                _mm256_sub_epi32(_mm256_add_epi32(_mm512_castsi512_si256(a1), _mm512_extracti32x8_epi32(a1, 1)), sub1);
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd[0]), _mm256_cvtepi32_ps(v0), acc0[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd[1]), _mm256_cvtepi32_ps(v1), acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        res0[t] = hsum_ggml(acc0[t]);
        res1[t] = hsum_ggml(acc1[t]);
    }
}

void q6k_two(const uint8_t* a, const uint8_t* b, int nb, const void* const* act, int m, float* ra, float* rb,
             const char* pf0, const char* pf1) {
    const block_q6_K *x0 = (const block_q6_K*) a, *x1 = (const block_q6_K*) b;
    switch (m) {
    case 1: q6k_2rows<1>(x0, x1, nb, act, ra, rb, pf0, pf1); break;
    case 2: q6k_2rows<2>(x0, x1, nb, act, ra, rb, pf0, pf1); break;
    case 3: q6k_2rows<3>(x0, x1, nb, act, ra, rb, pf0, pf1); break;
    case 4: q6k_2rows<4>(x0, x1, nb, act, ra, rb, pf0, pf1); break;
    case 5: q6k_2rows<5>(x0, x1, nb, act, ra, rb, pf0, pf1); break;
    case 6: q6k_2rows<6>(x0, x1, nb, act, ra, rb, pf0, pf1); break;
    case 7: q6k_2rows<7>(x0, x1, nb, act, ra, rb, pf0, pf1); break;
    default: q6k_2rows<8>(x0, x1, nb, act, ra, rb, pf0, pf1); break;
    }
}

} // namespace

bool kq_rows512_ok() {
    return guild::kernels::cpu::cpu_avx512bw_ok(); // GUILD_FORCE_AVX2=1: false
}

bool kq_rows512_supported(int type, int n) {
    return (type == 12 || type == 14) && n % QK_K == 0 && n / QK_K <= kMaxBlocks;
}

void kq_rows512(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const int m = nt - t0 < kMaxT ? nt - t0 : kMaxT;
        alignas(16) __m128i q8s[kMaxBlocks * kMaxT];
        if (type == 12)
            q8s_of(act + t0, m, nb, q8s);
        for (int r = r0; r < r1; r += 2) {
            const uint8_t* w0 = w + (size_t) r * row_bytes;
            const bool pair = r + 1 < r1;
            const uint8_t* w1 = pair ? w0 + row_bytes : w0;
            const char* pf = (const char*) w0 + 2 * row_bytes; // the next pair
            float a[kMaxT], b[kMaxT];
            if (type == 12)
                q4k_two(w0, w1, nb, act + t0, m, q8s, a, b, pf, pf + row_bytes);
            else
                q6k_two(w0, w1, nb, act + t0, m, a, b, pf, pf + row_bytes);
            for (int t = 0; t < m; ++t) {
                out[t0 + t][r] = a[t];
                if (pair)
                    out[t0 + t][r + 1] = b[t];
            }
        }
    }
}

void q4k_gu_rows512(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                    float* const* ff, int r0, int r1) {
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const int m = nt - t0 < kMaxT ? nt - t0 : kMaxT;
        alignas(16) __m128i q8s[kMaxBlocks * kMaxT];
        q8s_of(act + t0, m, nb, q8s);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* g = blob + (size_t) r * gu_row;
            const uint8_t* u = blob + up_off + (size_t) r * gu_row;
            float a[kMaxT], b[kMaxT];
            q4k_two(g, u, nb, act + t0, m, q8s, a, b, (const char*) g + 2 * gu_row, (const char*) u + 2 * gu_row);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
            for (int t = 0; t < m; ++t)
                ff[t0 + t][r] = (a[t] / (1.f + std::exp(-a[t]))) * b[t];
        }
    }
}

} // namespace guild::core::cpu_moe
