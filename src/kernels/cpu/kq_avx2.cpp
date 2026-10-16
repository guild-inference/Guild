// src/kernels/cpu/kq_avx2.cpp - Unsloth UD-Q4_K_XL's expert rows (Q4_K gate/up, Q5_1 and Q8_0 down) for several
// tokens at once, BIT-EXACT against ggml-cpu's own dot products.
//
// ggml-cpu is built for AVX2 here (GUILD_PORTABLE: GGML_AVX2=ON, GGML_AVX512=OFF), so its vec_dot for these formats
// is the __AVX2__ branch of ggml-cpu/arch/x86/quants.c: ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q5_1_q8_1,
// ggml_vec_dot_q8_0_q8_0.  The functions below are those branches with the loop over tokens moved inside the loop
// over blocks: everything that depends only on the weights (the 4-bit unpacking, the 6-bit scales and mins, the
// fifth bits, the absolute values of the Q8_0 weights, the fp16 scales) is done once per block, and every token then
// runs exactly ggml's integer chain and ggml's float operations in ggml's order - so each token's result is ggml's
// to the bit (native_expert_parity checks it), whatever the number of tokens.  Unlike the i-quant kernels (#152)
// the group size therefore never changes an answer.
//
// The gain is the weight-side work and the weight bytes, read once per verify window instead of once per token.
#include "guild/kernels/cpu/kq_avx2.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace guild::kernels::cpu {
namespace {

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} / {d, m} / {d, s} a block starts with (a union in ggml-common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 / non-VNNI forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                                0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}
inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}
inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}
alignas(32) const uint8_t k_shuffle[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
alignas(16) const uint8_t k_shuffle_q6[128] = {
     0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2,  3, 3, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 4, 4,  5, 5, 5, 5, 5, 5, 5, 5,
     6, 6, 6, 6, 6, 6, 6, 6,  7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8,  9, 9, 9, 9, 9, 9, 9, 9,
    10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
    14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle + i); }
inline __m128i get_scale_shuffle_q6(int i) { return _mm_loadu_si128((const __m128i*) k_shuffle_q6 + i); }

inline void unpack_q4k_scales(const uint8_t* sc_bytes, __m128i& mins, __m256i& scales) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t utmp[4];
    std::memcpy(utmp, sc_bytes, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                       (int) utmp[1], (int) utmp[0]));
    mins = _mm256_extracti128_si256(mins_and_scales, 1);
    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
}

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
void q4k_dot_avx2(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[MAXT];
    __m128 acc_m[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        __m128i mins;
        __m256i scales;
        unpack_q4k_scales(x[i].scales, mins, scales);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

#define GUILD_Q4K_2ROWS_BODY()                                                                                         \
    const __m256i m4 = _mm256_set1_epi8(0xF);                                                                          \
    __m256 acc0[MAXT], acc1[MAXT];                                                                                     \
    __m128 acc_m0[MAXT], acc_m1[MAXT];                                                                                 \
    for (int t = 0; t < nt; ++t) {                                                                                     \
        acc0[t] = _mm256_setzero_ps(); acc_m0[t] = _mm_setzero_ps();                                                   \
        acc1[t] = _mm256_setzero_ps(); acc_m1[t] = _mm_setzero_ps();                                                   \
    }                                                                                                                  \
    for (int i = 0; i < nb; ++i) {                                                                                     \
        const float xd0 = half_at(&x0[i], 0), xdmin0 = half_at(&x0[i], 1);                                             \
        const float xd1 = half_at(&x1[i], 0), xdmin1 = half_at(&x1[i], 1);                                             \
        __m128i mins0, mins1;                                                                                          \
        __m256i scales0, scales1;                                                                                      \
        unpack_q4k_scales(x0[i].scales, mins0, scales0);                                                               \
        unpack_q4k_scales(x1[i].scales, mins1, scales1);                                                               \
        __m256i sumi0[MAXT], sumi1[MAXT];                                                                              \
        for (int t = 0; t < nt; ++t) {                                                                                 \
            const block_q8_K* y = (const block_q8_K*) act[t] + i;                                                      \
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);                                      \
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),                                    \
                                               _mm256_extracti128_si256(q8sums, 1));                                   \
            acc_m0[t] = _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin0), _mm_cvtepi32_ps(_mm_madd_epi16(mins0, q8s)), acc_m0[t]); \
            acc_m1[t] = _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin1), _mm_cvtepi32_ps(_mm_madd_epi16(mins1, q8s)), acc_m1[t]); \
            sumi0[t] = _mm256_setzero_si256();                                                                         \
            sumi1[t] = _mm256_setzero_si256();                                                                         \
        }                                                                                                              \
        for (int j = 0; j < QK_K / 64; ++j) {                                                                          \
            const __m256i sh_l = get_scale_shuffle_k4(2 * j + 0);                                                      \
            const __m256i sh_h = get_scale_shuffle_k4(2 * j + 1);                                                      \
            const __m256i sc0_l = _mm256_shuffle_epi8(scales0, sh_l);                                                  \
            const __m256i sc0_h = _mm256_shuffle_epi8(scales0, sh_h);                                                  \
            const __m256i sc1_l = _mm256_shuffle_epi8(scales1, sh_l);                                                  \
            const __m256i sc1_h = _mm256_shuffle_epi8(scales1, sh_h);                                                  \
            const __m256i qb0 = _mm256_loadu_si256((const __m256i*) (x0[i].qs + 32 * j));                              \
            const __m256i qb1 = _mm256_loadu_si256((const __m256i*) (x1[i].qs + 32 * j));                              \
            const __m256i q40_l = _mm256_and_si256(qb0, m4);                                                           \
            const __m256i q40_h = _mm256_and_si256(_mm256_srli_epi16(qb0, 4), m4);                                     \
            const __m256i q41_l = _mm256_and_si256(qb1, m4);                                                           \
            const __m256i q41_h = _mm256_and_si256(_mm256_srli_epi16(qb1, 4), m4);                                     \
            for (int t = 0; t < nt; ++t) {                                                                             \
                const int8_t* q8 = ((const block_q8_K*) act[t] + i)->qs + 64 * j;                                      \
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);                                           \
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) (q8 + 32));                                    \
                const __m256i p0_l = _mm256_madd_epi16(sc0_l, _mm256_maddubs_epi16(q40_l, q8l));                       \
                const __m256i p0_h = _mm256_madd_epi16(sc0_h, _mm256_maddubs_epi16(q40_h, q8h));                       \
                sumi0[t] = _mm256_add_epi32(sumi0[t], _mm256_add_epi32(p0_l, p0_h));                                   \
                const __m256i p1_l = _mm256_madd_epi16(sc1_l, _mm256_maddubs_epi16(q41_l, q8l));                       \
                const __m256i p1_h = _mm256_madd_epi16(sc1_h, _mm256_maddubs_epi16(q41_h, q8h));                       \
                sumi1[t] = _mm256_add_epi32(sumi1[t], _mm256_add_epi32(p1_l, p1_h));                                   \
            }                                                                                                          \
        }                                                                                                              \
        for (int t = 0; t < nt; ++t) {                                                                                 \
            const float yd = ((const block_q8_K*) act[t] + i)->d;                                                      \
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * xd0), _mm256_cvtepi32_ps(sumi0[t]), acc0[t]);                \
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(yd * xd1), _mm256_cvtepi32_ps(sumi1[t]), acc1[t]);                \
        }                                                                                                              \
    }                                                                                                                  \
    for (int t = 0; t < nt; ++t) {                                                                                     \
        __m128 m0 = acc_m0[t];                                                                                         \
        m0 = _mm_add_ps(m0, _mm_movehl_ps(m0, m0));                                                                    \
        m0 = _mm_add_ss(m0, _mm_movehdup_ps(m0));                                                                      \
        res0[t] = hsum_float_8(acc0[t]) + _mm_cvtss_f32(m0);                                                           \
        __m128 m1 = acc_m1[t];                                                                                         \
        m1 = _mm_add_ps(m1, _mm_movehl_ps(m1, m1));                                                                    \
        m1 = _mm_add_ss(m1, _mm_movehdup_ps(m1));                                                                      \
        res1[t] = hsum_float_8(acc1[t]) + _mm_cvtss_f32(m1);                                                           \
    }

void q4k_dot_2rows_avx2(const block_q4_K* x0, const block_q4_K* x1, int nb, const void* const* act, int nt,
                        float* res0, float* res1) {
    GUILD_Q4K_2ROWS_BODY()
}

#define GUILD_Q6K_1ROW_BODY()                                                                                          \
    const __m256i m15 = _mm256_set1_epi8(0xF);                                                                         \
    const __m256i m3 = _mm256_set1_epi8(3);                                                                            \
    const __m256i m12 = _mm256_set1_epi8(12);                                                                          \
    const __m256i m48 = _mm256_set1_epi8(48);                                                                          \
    const __m256i m_neg64 = _mm256_set1_epi8(-64);                                                                     \
    __m256 acc[MAXT];                                                                                                  \
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();                                                         \
    for (int i = 0; i < nb; ++i) {                                                                                     \
        const float xd = h2f(x[i].d);                                                                                  \
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);                                          \
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);                                                        \
        __m256i sumi[MAXT];                                                                                            \
        for (int t = 0; t < nt; ++t) sumi[t] = _mm256_setzero_si256();                                                 \
        for (int j = 0; j < QK_K / 128; ++j) {                                                                         \
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) (x[i].ql + 64 * j + 0));                       \
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) (x[i].ql + 64 * j + 32));                      \
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) (x[i].qh + 32 * j));                           \
            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15),                                       \
                                                 _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4));                 \
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15),                                       \
                                                 _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m12), 2));                \
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15),                 \
                                                 _mm256_and_si256(q4bitsH, m48));                                      \
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15),                 \
                                                 _mm256_srli_epi16(_mm256_and_si256(q4bitsH, m_neg64), 2));            \
            const __m256i sc0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(4 * j + 0)));       \
            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(4 * j + 1)));       \
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(4 * j + 2)));       \
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(4 * j + 3)));       \
            for (int t = 0; t < nt; ++t) {                                                                             \
                const int8_t* q8 = ((const block_q8_K*) act[t] + i)->qs + 128 * j;                                     \
                const __m256i p0 = _mm256_madd_epi16(sc0, _mm256_maddubs_epi16(q4_0, _mm256_loadu_si256((const __m256i*) (q8 + 0))));  \
                const __m256i p1 = _mm256_madd_epi16(sc1, _mm256_maddubs_epi16(q4_1, _mm256_loadu_si256((const __m256i*) (q8 + 32)))); \
                const __m256i p2 = _mm256_madd_epi16(sc2, _mm256_maddubs_epi16(q4_2, _mm256_loadu_si256((const __m256i*) (q8 + 64)))); \
                const __m256i p3 = _mm256_madd_epi16(sc3, _mm256_maddubs_epi16(q4_3, _mm256_loadu_si256((const __m256i*) (q8 + 96)))); \
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p0, p1));                                         \
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p2, p3));                                         \
            }                                                                                                          \
        }                                                                                                              \
        for (int t = 0; t < nt; ++t) {                                                                                 \
            const block_q8_K* y = (const block_q8_K*) act[t] + i;                                                      \
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);                                      \
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);                       \
            const __m256i s = _mm256_sub_epi32(sumi[t], q8sclsub);                                                     \
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd), _mm256_cvtepi32_ps(s), acc[t]);                        \
        }                                                                                                              \
    }                                                                                                                  \
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);

#define GUILD_Q6K_2ROWS_BODY()                                                                                         \
    const __m256i m15 = _mm256_set1_epi8(0xF);                                                                         \
    const __m256i m3 = _mm256_set1_epi8(3);                                                                            \
    const __m256i m12 = _mm256_set1_epi8(12);                                                                          \
    const __m256i m48 = _mm256_set1_epi8(48);                                                                          \
    const __m256i m_neg64 = _mm256_set1_epi8(-64);                                                                     \
    __m256 acc0[MAXT], acc1[MAXT];                                                                                     \
    for (int t = 0; t < nt; ++t) { acc0[t] = _mm256_setzero_ps(); acc1[t] = _mm256_setzero_ps(); }                     \
    for (int i = 0; i < nb; ++i) {                                                                                     \
        const float xd0 = h2f(x0[i].d), xd1 = h2f(x1[i].d);                                                            \
        const __m128i scales0 = _mm_loadu_si128((const __m128i*) x0[i].scales);                                        \
        const __m128i scales1 = _mm_loadu_si128((const __m128i*) x1[i].scales);                                        \
        __m256i sumi0[MAXT], sumi1[MAXT];                                                                              \
        for (int t = 0; t < nt; ++t) { sumi0[t] = _mm256_setzero_si256(); sumi1[t] = _mm256_setzero_si256(); }         \
        for (int j = 0; j < QK_K / 128; ++j) {                                                                         \
            const __m256i qb0_1 = _mm256_loadu_si256((const __m256i*) (x0[i].ql + 64 * j + 0));                        \
            const __m256i qb0_2 = _mm256_loadu_si256((const __m256i*) (x0[i].ql + 64 * j + 32));                       \
            const __m256i qb0_H = _mm256_loadu_si256((const __m256i*) (x0[i].qh + 32 * j));                            \
            const __m256i qb1_1 = _mm256_loadu_si256((const __m256i*) (x1[i].ql + 64 * j + 0));                        \
            const __m256i qb1_2 = _mm256_loadu_si256((const __m256i*) (x1[i].ql + 64 * j + 32));                       \
            const __m256i qb1_H = _mm256_loadu_si256((const __m256i*) (x1[i].qh + 32 * j));                            \
            const __m128i sh0 = get_scale_shuffle_q6(4 * j + 0);                                                       \
            const __m128i sh1 = get_scale_shuffle_q6(4 * j + 1);                                                       \
            const __m256i q40_0 = _mm256_or_si256(_mm256_and_si256(qb0_1, m15),                                        \
                                                  _mm256_slli_epi16(_mm256_and_si256(qb0_H, m3), 4));                  \
            const __m256i q40_1 = _mm256_or_si256(_mm256_and_si256(qb0_2, m15),                                        \
                                                  _mm256_slli_epi16(_mm256_and_si256(qb0_H, m12), 2));                 \
            const __m256i sc0_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh0));                                \
            const __m256i sc0_1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh1));                                \
            const __m256i q41_0 = _mm256_or_si256(_mm256_and_si256(qb1_1, m15),                                        \
                                                  _mm256_slli_epi16(_mm256_and_si256(qb1_H, m3), 4));                  \
            const __m256i q41_1 = _mm256_or_si256(_mm256_and_si256(qb1_2, m15),                                        \
                                                  _mm256_slli_epi16(_mm256_and_si256(qb1_H, m12), 2));                 \
            const __m256i sc1_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales1, sh0));                                \
            const __m256i sc1_1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales1, sh1));                                \
            for (int t = 0; t < nt; ++t) {                                                                             \
                const int8_t* q8 = ((const block_q8_K*) act[t] + i)->qs + 128 * j;                                     \
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) (q8 + 0));                                    \
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));                                   \
                const __m256i p0_0 = _mm256_madd_epi16(sc0_0, _mm256_maddubs_epi16(q40_0, q8_0));                      \
                const __m256i p0_1 = _mm256_madd_epi16(sc0_1, _mm256_maddubs_epi16(q40_1, q8_1));                      \
                sumi0[t] = _mm256_add_epi32(sumi0[t], _mm256_add_epi32(p0_0, p0_1));                                   \
                const __m256i p1_0 = _mm256_madd_epi16(sc1_0, _mm256_maddubs_epi16(q41_0, q8_0));                      \
                const __m256i p1_1 = _mm256_madd_epi16(sc1_1, _mm256_maddubs_epi16(q41_1, q8_1));                      \
                sumi1[t] = _mm256_add_epi32(sumi1[t], _mm256_add_epi32(p1_0, p1_1));                                   \
            }                                                                                                          \
            const __m128i sh2 = get_scale_shuffle_q6(4 * j + 2);                                                       \
            const __m128i sh3 = get_scale_shuffle_q6(4 * j + 3);                                                       \
            const __m256i q40_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb0_1, 4), m15),                  \
                                                  _mm256_and_si256(qb0_H, m48));                                       \
            const __m256i q40_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb0_2, 4), m15),                  \
                                                  _mm256_srli_epi16(_mm256_and_si256(qb0_H, m_neg64), 2));             \
            const __m256i sc0_2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh2));                                \
            const __m256i sc0_3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh3));                                \
            const __m256i q41_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb1_1, 4), m15),                  \
                                                  _mm256_and_si256(qb1_H, m48));                                       \
            const __m256i q41_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb1_2, 4), m15),                  \
                                                  _mm256_srli_epi16(_mm256_and_si256(qb1_H, m_neg64), 2));             \
            const __m256i sc1_2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales1, sh2));                                \
            const __m256i sc1_3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales1, sh3));                                \
            for (int t = 0; t < nt; ++t) {                                                                             \
                const int8_t* q8 = ((const block_q8_K*) act[t] + i)->qs + 128 * j;                                     \
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) (q8 + 64));                                   \
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) (q8 + 96));                                   \
                const __m256i p0_2 = _mm256_madd_epi16(sc0_2, _mm256_maddubs_epi16(q40_2, q8_2));                      \
                const __m256i p0_3 = _mm256_madd_epi16(sc0_3, _mm256_maddubs_epi16(q40_3, q8_3));                      \
                sumi0[t] = _mm256_add_epi32(sumi0[t], _mm256_add_epi32(p0_2, p0_3));                                   \
                const __m256i p1_2 = _mm256_madd_epi16(sc1_2, _mm256_maddubs_epi16(q41_2, q8_2));                      \
                const __m256i p1_3 = _mm256_madd_epi16(sc1_3, _mm256_maddubs_epi16(q41_3, q8_3));                      \
                sumi1[t] = _mm256_add_epi32(sumi1[t], _mm256_add_epi32(p1_2, p1_3));                                   \
            }                                                                                                          \
        }                                                                                                              \
        const __m256i scales0_16 = _mm256_cvtepi8_epi16(scales0);                                                      \
        const __m256i scales1_16 = _mm256_cvtepi8_epi16(scales1);                                                      \
        for (int t = 0; t < nt; ++t) {                                                                                 \
            const block_q8_K* y = (const block_q8_K*) act[t] + i;                                                      \
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);                                      \
            const __m256i q8sub0 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales0_16), 5);                        \
            const __m256i q8sub1 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales1_16), 5);                        \
            const __m256i s0 = _mm256_sub_epi32(sumi0[t], q8sub0);                                                     \
            const __m256i s1 = _mm256_sub_epi32(sumi1[t], q8sub1);                                                     \
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd0), _mm256_cvtepi32_ps(s0), acc0[t]);                    \
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd1), _mm256_cvtepi32_ps(s1), acc1[t]);                    \
        }                                                                                                              \
    }                                                                                                                  \
    for (int t = 0; t < nt; ++t) {                                                                                     \
        res0[t] = hsum_float_8(acc0[t]);                                                                               \
        res1[t] = hsum_float_8(acc1[t]);                                                                               \
    }

void q6k_dot_avx2(const block_q6_K* x, int nb, const void* const* act, int nt, float* res) {
    GUILD_Q6K_1ROW_BODY()
}

void q6k_dot_2rows_avx2(const block_q6_K* x0, const block_q6_K* x1, int nb, const void* const* act, int nt,
                        float* res0, float* res1) {
    GUILD_Q6K_2ROWS_BODY()
}

#if defined(__GNUC__) || defined(__clang__)
template <int NT>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c")))
void q4k_dot_avx512bw_nt(const block_q4_K* x, int nb, const void* const* act, float* res) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[NT];
    __m128 acc_m[NT];
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        __m128i mins;
        __m256i scales;
        unpack_q4k_scales(x[i].scales, mins, scales);
        __m512i scale_lh[4], q4_lh[4];
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const __m256i sl = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i sh = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            scale_lh[j] = _mm512_inserti64x4(_mm512_castsi256_si512(sl), sh, 1);
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) (x[i].qs + 32 * j));
            q4_lh[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(q4bits, m4)),
                                          _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4), 1);
        }
#pragma GCC unroll 8
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s)), acc_m[t]);
            __m512i sumi512 = _mm512_setzero_si512();
            const int8_t* q8 = y->qs;
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const __m512i q8_lh = _mm512_loadu_si512((const void*) (q8 + 64 * j));
                sumi512 = _mm512_add_epi32(sumi512, _mm512_madd_epi16(scale_lh[j], _mm512_maddubs_epi16(q4_lh[j], q8_lh)));
            }
            const __m256i sumi = _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti32x8_epi32(sumi512, 1));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

template <int NT>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c")))
void q4k_dot_2rows_avx512bw_nt(const block_q4_K* x0, const block_q4_K* x1, int nb, const void* const* act,
                               float* res0, float* res1) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc0[NT], acc1[NT];
    __m128 acc_m0[NT], acc_m1[NT];
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) {
        acc0[t] = _mm256_setzero_ps(); acc_m0[t] = _mm_setzero_ps();
        acc1[t] = _mm256_setzero_ps(); acc_m1[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        const float xd0 = half_at(&x0[i], 0), xdmin0 = half_at(&x0[i], 1);
        const float xd1 = half_at(&x1[i], 0), xdmin1 = half_at(&x1[i], 1);
        __m128i mins0, mins1;
        __m256i scales0, scales1;
        unpack_q4k_scales(x0[i].scales, mins0, scales0);
        unpack_q4k_scales(x1[i].scales, mins1, scales1);
        __m512i scale0_lh[4], q40_lh[4], scale1_lh[4], q41_lh[4];
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            const __m256i sh_l = get_scale_shuffle_k4(2 * j + 0);
            const __m256i sh_h = get_scale_shuffle_k4(2 * j + 1);
            scale0_lh[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_shuffle_epi8(scales0, sh_l)),
                                              _mm256_shuffle_epi8(scales0, sh_h), 1);
            scale1_lh[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_shuffle_epi8(scales1, sh_l)),
                                              _mm256_shuffle_epi8(scales1, sh_h), 1);
            const __m256i qb0 = _mm256_loadu_si256((const __m256i*) (x0[i].qs + 32 * j));
            const __m256i qb1 = _mm256_loadu_si256((const __m256i*) (x1[i].qs + 32 * j));
            q40_lh[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(qb0, m4)),
                                           _mm256_and_si256(_mm256_srli_epi16(qb0, 4), m4), 1);
            q41_lh[j] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(qb1, m4)),
                                           _mm256_and_si256(_mm256_srli_epi16(qb1, 4), m4), 1);
        }
#pragma GCC unroll 8
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            acc_m0[t] = _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin0), _mm_cvtepi32_ps(_mm_madd_epi16(mins0, q8s)), acc_m0[t]);
            acc_m1[t] = _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin1), _mm_cvtepi32_ps(_mm_madd_epi16(mins1, q8s)), acc_m1[t]);
            __m512i sumi0_512 = _mm512_setzero_si512(), sumi1_512 = _mm512_setzero_si512();
            const int8_t* q8 = y->qs;
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const __m512i q8_lh = _mm512_loadu_si512((const void*) (q8 + 64 * j));
                sumi0_512 = _mm512_add_epi32(sumi0_512, _mm512_madd_epi16(scale0_lh[j], _mm512_maddubs_epi16(q40_lh[j], q8_lh)));
                sumi1_512 = _mm512_add_epi32(sumi1_512, _mm512_madd_epi16(scale1_lh[j], _mm512_maddubs_epi16(q41_lh[j], q8_lh)));
            }
            const __m256i sumi0 = _mm256_add_epi32(_mm512_castsi512_si256(sumi0_512), _mm512_extracti32x8_epi32(sumi0_512, 1));
            const __m256i sumi1 = _mm256_add_epi32(_mm512_castsi512_si256(sumi1_512), _mm512_extracti32x8_epi32(sumi1_512, 1));
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd0), _mm256_cvtepi32_ps(sumi0), acc0[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd1), _mm256_cvtepi32_ps(sumi1), acc1[t]);
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) {
        __m128 m0 = acc_m0[t];
        m0 = _mm_add_ps(m0, _mm_movehl_ps(m0, m0));
        m0 = _mm_add_ss(m0, _mm_movehdup_ps(m0));
        res0[t] = hsum_float_8(acc0[t]) + _mm_cvtss_f32(m0);
        __m128 m1 = acc_m1[t];
        m1 = _mm_add_ps(m1, _mm_movehl_ps(m1, m1));
        m1 = _mm_add_ss(m1, _mm_movehdup_ps(m1));
        res1[t] = hsum_float_8(acc1[t]) + _mm_cvtss_f32(m1);
    }
}

template <int NT>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c")))
void q6k_dot_avx512bw_nt(const block_q6_K* x, int nb, const void* const* act, float* res) {
    const __m256i m15 = _mm256_set1_epi8(0xF);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m_neg64 = _mm256_set1_epi8(-64);
    __m256 acc[NT];
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        __m512i q4_01[2], q4_23[2], sc_01[2], sc_23[2];
#pragma GCC unroll 2
        for (int j = 0; j < 2; ++j) {
            const __m256i qb1 = _mm256_loadu_si256((const __m256i*) (x[i].ql + 64 * j + 0));
            const __m256i qb2 = _mm256_loadu_si256((const __m256i*) (x[i].ql + 64 * j + 32));
            const __m256i qbH = _mm256_loadu_si256((const __m256i*) (x[i].qh + 32 * j));
            q4_01[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_or_si256(_mm256_and_si256(qb1, m15), _mm256_slli_epi16(_mm256_and_si256(qbH, m3), 4))),
                _mm256_or_si256(_mm256_and_si256(qb2, m15), _mm256_slli_epi16(_mm256_and_si256(qbH, m12), 2)), 1);
            q4_23[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb1, 4), m15), _mm256_and_si256(qbH, m48))),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb2, 4), m15),
                                _mm256_srli_epi16(_mm256_and_si256(qbH, m_neg64), 2)), 1);
            sc_01[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(4 * j + 0)))),
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(4 * j + 1))), 1);
            sc_23[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(4 * j + 2)))),
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(4 * j + 3))), 1);
        }
#pragma GCC unroll 8
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            __m512i sumi512 = _mm512_setzero_si512();
            const int8_t* q8 = y->qs;
#pragma GCC unroll 2
            for (int j = 0; j < 2; ++j) {
                const __m512i q8_01 = _mm512_loadu_si512((const void*) (q8 + 128 * j + 0));
                const __m512i q8_23 = _mm512_loadu_si512((const void*) (q8 + 128 * j + 64));
                const __m512i p01 = _mm512_madd_epi16(sc_01[j], _mm512_maddubs_epi16(q4_01[j], q8_01));
                const __m512i p23 = _mm512_madd_epi16(sc_23[j], _mm512_maddubs_epi16(q4_23[j], q8_23));
                sumi512 = _mm512_add_epi32(sumi512, _mm512_add_epi32(p01, p23));
            }
            const __m256i sumi = _mm256_sub_epi32(
                _mm256_add_epi32(_mm512_castsi512_si256(sumi512), _mm512_extracti32x8_epi32(sumi512, 1)), q8sclsub);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) res[t] = hsum_float_8(acc[t]);
}

template <int NT>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c")))
void q6k_dot_2rows_avx512bw_nt(const block_q6_K* x0, const block_q6_K* x1, int nb, const void* const* act,
                               float* res0, float* res1) {
    const __m256i m15 = _mm256_set1_epi8(0xF);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m_neg64 = _mm256_set1_epi8(-64);
    __m256 acc0[NT], acc1[NT];
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) { acc0[t] = _mm256_setzero_ps(); acc1[t] = _mm256_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd0 = h2f(x0[i].d), xd1 = h2f(x1[i].d);
        const __m128i scales0 = _mm_loadu_si128((const __m128i*) x0[i].scales);
        const __m128i scales1 = _mm_loadu_si128((const __m128i*) x1[i].scales);
        const __m256i scales0_16 = _mm256_cvtepi8_epi16(scales0);
        const __m256i scales1_16 = _mm256_cvtepi8_epi16(scales1);
        __m512i q40_01[2], q40_23[2], sc0_01[2], sc0_23[2];
        __m512i q41_01[2], q41_23[2], sc1_01[2], sc1_23[2];
#pragma GCC unroll 2
        for (int j = 0; j < 2; ++j) {
            const __m256i qb0_1 = _mm256_loadu_si256((const __m256i*) (x0[i].ql + 64 * j + 0));
            const __m256i qb0_2 = _mm256_loadu_si256((const __m256i*) (x0[i].ql + 64 * j + 32));
            const __m256i qb0_H = _mm256_loadu_si256((const __m256i*) (x0[i].qh + 32 * j));
            const __m256i qb1_1 = _mm256_loadu_si256((const __m256i*) (x1[i].ql + 64 * j + 0));
            const __m256i qb1_2 = _mm256_loadu_si256((const __m256i*) (x1[i].ql + 64 * j + 32));
            const __m256i qb1_H = _mm256_loadu_si256((const __m256i*) (x1[i].qh + 32 * j));
            q40_01[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_or_si256(_mm256_and_si256(qb0_1, m15), _mm256_slli_epi16(_mm256_and_si256(qb0_H, m3), 4))),
                _mm256_or_si256(_mm256_and_si256(qb0_2, m15), _mm256_slli_epi16(_mm256_and_si256(qb0_H, m12), 2)), 1);
            q40_23[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb0_1, 4), m15), _mm256_and_si256(qb0_H, m48))),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb0_2, 4), m15),
                                _mm256_srli_epi16(_mm256_and_si256(qb0_H, m_neg64), 2)), 1);
            q41_01[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_or_si256(_mm256_and_si256(qb1_1, m15), _mm256_slli_epi16(_mm256_and_si256(qb1_H, m3), 4))),
                _mm256_or_si256(_mm256_and_si256(qb1_2, m15), _mm256_slli_epi16(_mm256_and_si256(qb1_H, m12), 2)), 1);
            q41_23[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb1_1, 4), m15), _mm256_and_si256(qb1_H, m48))),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb1_2, 4), m15),
                                _mm256_srli_epi16(_mm256_and_si256(qb1_H, m_neg64), 2)), 1);
            const __m128i sh0 = get_scale_shuffle_q6(4 * j + 0), sh1 = get_scale_shuffle_q6(4 * j + 1);
            const __m128i sh2 = get_scale_shuffle_q6(4 * j + 2), sh3 = get_scale_shuffle_q6(4 * j + 3);
            sc0_01[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh0))),
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh1)), 1);
            sc0_23[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh2))),
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh3)), 1);
            sc1_01[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales1, sh0))),
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales1, sh1)), 1);
            sc1_23[j] = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales1, sh2))),
                _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales1, sh3)), 1);
        }
#pragma GCC unroll 8
        for (int t = 0; t < NT; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m256i q8sub0 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales0_16), 5);
            const __m256i q8sub1 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales1_16), 5);
            __m512i sumi0_512 = _mm512_setzero_si512(), sumi1_512 = _mm512_setzero_si512();
            const int8_t* q8 = y->qs;
#pragma GCC unroll 2
            for (int j = 0; j < 2; ++j) {
                const __m512i q8_01 = _mm512_loadu_si512((const void*) (q8 + 128 * j + 0));
                const __m512i q8_23 = _mm512_loadu_si512((const void*) (q8 + 128 * j + 64));
                const __m512i p0_01 = _mm512_madd_epi16(sc0_01[j], _mm512_maddubs_epi16(q40_01[j], q8_01));
                const __m512i p0_23 = _mm512_madd_epi16(sc0_23[j], _mm512_maddubs_epi16(q40_23[j], q8_23));
                sumi0_512 = _mm512_add_epi32(sumi0_512, _mm512_add_epi32(p0_01, p0_23));
                const __m512i p1_01 = _mm512_madd_epi16(sc1_01[j], _mm512_maddubs_epi16(q41_01[j], q8_01));
                const __m512i p1_23 = _mm512_madd_epi16(sc1_23[j], _mm512_maddubs_epi16(q41_23[j], q8_23));
                sumi1_512 = _mm512_add_epi32(sumi1_512, _mm512_add_epi32(p1_01, p1_23));
            }
            const __m256i sumi0 = _mm256_sub_epi32(
                _mm256_add_epi32(_mm512_castsi512_si256(sumi0_512), _mm512_extracti32x8_epi32(sumi0_512, 1)), q8sub0);
            const __m256i sumi1 = _mm256_sub_epi32(
                _mm256_add_epi32(_mm512_castsi512_si256(sumi1_512), _mm512_extracti32x8_epi32(sumi1_512, 1)), q8sub1);
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd0), _mm256_cvtepi32_ps(sumi0), acc0[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd1), _mm256_cvtepi32_ps(sumi1), acc1[t]);
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t) {
        res0[t] = hsum_float_8(acc0[t]);
        res1[t] = hsum_float_8(acc1[t]);
    }
}

void q4k_dot_avx512bw(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    switch (nt) {
        case 1: q4k_dot_avx512bw_nt<1>(x, nb, act, res); break;
        case 2: q4k_dot_avx512bw_nt<2>(x, nb, act, res); break;
        case 3: q4k_dot_avx512bw_nt<3>(x, nb, act, res); break;
        case 4: q4k_dot_avx512bw_nt<4>(x, nb, act, res); break;
        case 5: q4k_dot_avx512bw_nt<5>(x, nb, act, res); break;
        case 6: q4k_dot_avx512bw_nt<6>(x, nb, act, res); break;
        case 7: q4k_dot_avx512bw_nt<7>(x, nb, act, res); break;
        default: q4k_dot_avx512bw_nt<8>(x, nb, act, res); break;
    }
}

void q4k_dot_2rows_avx512bw(const block_q4_K* x0, const block_q4_K* x1, int nb, const void* const* act, int nt,
                            float* res0, float* res1) {
    switch (nt) {
        case 1: q4k_dot_2rows_avx512bw_nt<1>(x0, x1, nb, act, res0, res1); break;
        case 2: q4k_dot_2rows_avx512bw_nt<2>(x0, x1, nb, act, res0, res1); break;
        case 3: q4k_dot_2rows_avx512bw_nt<3>(x0, x1, nb, act, res0, res1); break;
        case 4: q4k_dot_2rows_avx512bw_nt<4>(x0, x1, nb, act, res0, res1); break;
        case 5: q4k_dot_2rows_avx512bw_nt<5>(x0, x1, nb, act, res0, res1); break;
        case 6: q4k_dot_2rows_avx512bw_nt<6>(x0, x1, nb, act, res0, res1); break;
        case 7: q4k_dot_2rows_avx512bw_nt<7>(x0, x1, nb, act, res0, res1); break;
        default: q4k_dot_2rows_avx512bw_nt<8>(x0, x1, nb, act, res0, res1); break;
    }
}

void q6k_dot_avx512bw(const block_q6_K* x, int nb, const void* const* act, int nt, float* res) {
    switch (nt) {
        case 1: q6k_dot_avx512bw_nt<1>(x, nb, act, res); break;
        case 2: q6k_dot_avx512bw_nt<2>(x, nb, act, res); break;
        case 3: q6k_dot_avx512bw_nt<3>(x, nb, act, res); break;
        case 4: q6k_dot_avx512bw_nt<4>(x, nb, act, res); break;
        case 5: q6k_dot_avx512bw_nt<5>(x, nb, act, res); break;
        case 6: q6k_dot_avx512bw_nt<6>(x, nb, act, res); break;
        case 7: q6k_dot_avx512bw_nt<7>(x, nb, act, res); break;
        default: q6k_dot_avx512bw_nt<8>(x, nb, act, res); break;
    }
}

void q6k_dot_2rows_avx512bw(const block_q6_K* x0, const block_q6_K* x1, int nb, const void* const* act, int nt,
                            float* res0, float* res1) {
    switch (nt) {
        case 1: q6k_dot_2rows_avx512bw_nt<1>(x0, x1, nb, act, res0, res1); break;
        case 2: q6k_dot_2rows_avx512bw_nt<2>(x0, x1, nb, act, res0, res1); break;
        case 3: q6k_dot_2rows_avx512bw_nt<3>(x0, x1, nb, act, res0, res1); break;
        case 4: q6k_dot_2rows_avx512bw_nt<4>(x0, x1, nb, act, res0, res1); break;
        case 5: q6k_dot_2rows_avx512bw_nt<5>(x0, x1, nb, act, res0, res1); break;
        case 6: q6k_dot_2rows_avx512bw_nt<6>(x0, x1, nb, act, res0, res1); break;
        case 7: q6k_dot_2rows_avx512bw_nt<7>(x0, x1, nb, act, res0, res1); break;
        default: q6k_dot_2rows_avx512bw_nt<8>(x0, x1, nb, act, res0, res1); break;
    }
}
#endif

void q4k_dot(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
#if defined(__GNUC__) || defined(__clang__)
    if (cpu_avx512bw_ok()) {
        q4k_dot_avx512bw(x, nb, act, nt, res);
        return;
    }
#endif
    q4k_dot_avx2(x, nb, act, nt, res);
}

void q4k_dot_2rows(const block_q4_K* x0, const block_q4_K* x1, int nb, const void* const* act, int nt,
                   float* res0, float* res1) {
#if defined(__GNUC__) || defined(__clang__)
    if (cpu_avx512bw_ok()) {
        q4k_dot_2rows_avx512bw(x0, x1, nb, act, nt, res0, res1);
        return;
    }
#endif
    q4k_dot_2rows_avx2(x0, x1, nb, act, nt, res0, res1);
}

void q6k_dot(const block_q6_K* x, int nb, const void* const* act, int nt, float* res) {
#if defined(__GNUC__) || defined(__clang__)
    if (cpu_avx512bw_ok()) {
        q6k_dot_avx512bw(x, nb, act, nt, res);
        return;
    }
#endif
    q6k_dot_avx2(x, nb, act, nt, res);
}

void q6k_dot_2rows(const block_q6_K* x0, const block_q6_K* x1, int nb, const void* const* act, int nt,
                   float* res0, float* res1) {
#if defined(__GNUC__) || defined(__clang__)
    if (cpu_avx512bw_ok()) {
        q6k_dot_2rows_avx512bw(x0, x1, nb, act, nt, res0, res1);
        return;
    }
#endif
    q6k_dot_2rows_avx2(x0, x1, nb, act, nt, res0, res1);
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, __AVX2__)
void q5_1_dot(const block_q5_1* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    float summs[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(half_at(&x[ib], 0));
        const float xm = half_at(&x[ib], 1);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        for (int t = 0; t < nt; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * half_at(y, 1);
            const __m256 dy = _mm256_set1_ps(half_at(y, 0));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = mul_sum_us8_pairs_float(qx, qy);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, __AVX2__; mul_sum_i8_pairs_float without AVX-VNNI-INT8)
void q8_0_dot(const block_q8_0* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * h2f(y->d));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy);
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

void dot_rows(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (type) {
        case 12: q4k_dot((const block_q4_K*) row, n / QK_K, act, nt, res); break;
        case 14: q6k_dot((const block_q6_K*) row, n / QK_K, act, nt, res); break;
        case 7: q5_1_dot((const block_q5_1*) row, n / QK5_1, act, nt, res); break;
        case 8: q8_0_dot((const block_q8_0*) row, n / QK8_0, act, nt, res); break;
        default: break;
    }
}

void dot_2rows(int type, const uint8_t* row0, const uint8_t* row1, int n, const void* const* act, int nt,
               float* res0, float* res1) {
    switch (type) {
        case 12: q4k_dot_2rows((const block_q4_K*) row0, (const block_q4_K*) row1, n / QK_K, act, nt, res0, res1); break;
        case 14: q6k_dot_2rows((const block_q6_K*) row0, (const block_q6_K*) row1, n / QK_K, act, nt, res0, res1); break;
        default:
            dot_rows(type, row0, n, act, nt, res0);
            dot_rows(type, row1, n, act, nt, res1);
            break;
    }
}

}  // namespace

bool kq256_supported(int type) noexcept { return type == 12 || type == 14 || type == 7 || type == 8; }

void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
    // each row is read once for all `nt` (<= 8) tokens: the router (2.6 MB per layer) streams once per prediction
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc[8];
        for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
        for (int c = 0; c + 8 <= cols; c += 8) {
            const __m256 wf = _mm256_castsi256_ps(
                _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c))), 16));
            for (int t = 0; t < nt; ++t) acc[t] = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + (size_t) t * cols + c), acc[t]);
        }
        for (int t = 0; t < nt; ++t) out[(size_t) t * rows + r] = hsum_float_8(acc[t]);   // cols % 8 == 0 (2560)
    }
}

void bf16_rows_dot(const uint16_t* w, int rows, int cols, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        const uint16_t* wr = w + (size_t) r * (size_t) cols;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            const __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (wr + c)));
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(h, 16)), _mm256_loadu_ps(x + c), acc);
        }
        float s = hsum_float_8(acc);
        for (; c < cols; ++c) {
            const uint32_t b = (uint32_t) wr[c] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            s += f * x[c];
        }
        out[r] = s;
    }
}

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * gu_row;
        const uint8_t* ur = blob + up_off + (size_t) r * gu_row;
        for (int t0 = 0; t0 < nt; t0 += MAXT) {
            const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
            float g[MAXT], u[MAXT];
            dot_2rows(type, gr, ur, n, act + t0, m, g, u);
            // native_gu_rows' own SwiGLU expression: the same bits as the per-token ggml path
            for (int t = 0; t < m; ++t) ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    int r = r0;
    for (; r + 2 <= r1; r += 2) {
        const uint8_t* w0 = w + (size_t) r * row_bytes;
        const uint8_t* w1 = w0 + row_bytes;
        for (int t0 = 0; t0 < nt; t0 += MAXT) {
            const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
            float s0[MAXT], s1[MAXT];
            dot_2rows(type, w0, w1, n, act + t0, m, s0, s1);
            for (int t = 0; t < m; ++t) {
                out[t0 + t][r] = s0[t];
                out[t0 + t][r + 1] = s1[t];
            }
        }
    }
    for (; r < r1; ++r) {
        const uint8_t* w0 = w + (size_t) r * row_bytes;
        for (int t0 = 0; t0 < nt; t0 += MAXT) {
            const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
            float s[MAXT];
            dot_rows(type, w0, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

}  // namespace guild::kernels::cpu
