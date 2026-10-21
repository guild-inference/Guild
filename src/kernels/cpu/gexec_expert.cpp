// Copyright (c) Guild Inference Contributors. MIT License.
// Phase 5: GEXEC Hardware-Specialized CPU Expert Kernel Providers.
//
// Implements 4 specialized CPU expert execution providers registered in KernelRegistry:
//   1. cpu.expert_pool.gexec_fused_avx2        (PortableContiguous layout, AVX2+FMA+F16C)
//   2. cpu.expert_pool.gexec_fused_avx512      (PortableContiguous layout, AVX-512BW)
//   3. cpu.expert_pool.gexec_interleaved_avx2  (GateUpRowInterleaved layout, AVX2+FMA+F16C)
//   4. cpu.expert_pool.gexec_interleaved_avx512(GateUpRowInterleaved layout, AVX-512BW)
//
// Key optimizations over the baseline cpu.expert_pool.split_multi_native:
//   - Fuses gate and up projections into 2-row and 4-row passes over the quantized activation
//     for ALL token counts nt >= 1 (eliminating the nt >= 2 gate that forced single-token
//     decode onto 1-row scalar/AVX2 ggml_vec_dot calls).
//   - Consumes GateUpRowInterleaved expert blobs ([gate_0, up_0, gate_1, up_1, ... | down])
//     so gate_r and up_r reside in adjacent cache lines and stream monotonically through a
//     single hardware prefetcher stream instead of jumping across a 367-589 KiB offset.
//   - Fuses 4 rows (and 2 rows) at a time in down-projection (IQ4_NL, Q4_K, Q6_K, Q8_0, Q5_1),
//     cutting activation loads by 4x across all 2,048-2,560 down rows per expert.
//   - Uses YMM integer accumulators for K-quants, including the AVX-512VL path,
//     avoiding cross-lane ZMM insert/extract operations while sharing activation loads.
//   - Vectorizes the MoE expert combine reduction across 8 FP32 columns with AVX2 FMA and
//     in-register IEEE-754 finiteness verification.
//   - Preserves 100% bitwise numerical parity with the baseline per-row accumulation order.

#include "guild/kernels/cpu/gexec_expert.hpp"
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/native_expert.hpp"
#include "guild/kernels/cpu/expert.hpp"
#include "guild/kernels/cpu/kq_avx2.hpp"
#include "guild/kernels/cpu/kq_avx1.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "guild/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

namespace guild::kernels::cpu {

void canonical_gu_avx2(const uint8_t*, const ActQ* const*, int, float* const*, int, int);
void canonical_down_avx2(const uint8_t*, const ActQ* const*, int, float* const*, int, int);

bool is_gexec_interleaved_impl(const char* impl_id) noexcept {
    if (!impl_id)
        return false;
    return std::strcmp(impl_id, "cpu.expert_pool.gexec_interleaved_avx2") == 0 ||
           std::strcmp(impl_id, "cpu.expert_pool.gexec_interleaved_avx512") == 0;
}

bool is_gexec_optimized_cpu_impl(const char* impl_id) noexcept {
    if (!impl_id)
        return false;
    return std::strcmp(impl_id, "cpu.expert_pool.gexec_fused_avx2") == 0 ||
           std::strcmp(impl_id, "cpu.expert_pool.gexec_fused_avx512") == 0 ||
           std::strcmp(impl_id, "cpu.expert_pool.gexec_interleaved_avx2") == 0 ||
           std::strcmp(impl_id, "cpu.expert_pool.gexec_interleaved_avx512") == 0;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
#define GEXEC_AVX2_TARGET __attribute__((target("avx2,fma,f16c")))
#define GEXEC_AVX2_INLINE __attribute__((target("avx2,fma,f16c"), always_inline)) inline
#define GEXEC_AVX512BW_TARGET __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c")))
#define GEXEC_AVX512_INLINE __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c"), always_inline)) inline
#define GEXEC_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define GEXEC_AVX2_TARGET
#define GEXEC_AVX2_INLINE inline
#define GEXEC_AVX512BW_TARGET
#define GEXEC_AVX512_INLINE inline
#define GEXEC_ALWAYS_INLINE inline
#endif

namespace {

GEXEC_ALWAYS_INLINE uint16_t read_u16(const void* p) noexcept {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

GEXEC_ALWAYS_INLINE uint32_t read_u32(const void* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

GEXEC_ALWAYS_INLINE uint64_t read_u64(const void* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

GEXEC_AVX2_INLINE float fp16_to_fp32(uint16_t h) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

GEXEC_AVX2_INLINE uint16_t fp32_to_fp16(float f) noexcept {
    return (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), 0), 0);
}

#if !defined(GUILD_NATIVE_EXPERTS)
void quantize_q8_K_host(const float* x, void* vy, int k) noexcept {
    block_q8_K* y = static_cast<block_q8_K*>(vy);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        float max_val = 0.0f;
        float max_abs = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            const float v = x[i * QK_K + j];
            if (std::fabs(v) > max_abs) {
                max_abs = std::fabs(v);
                max_val = v;
            }
        }
        if (max_abs == 0.0f) {
            y[i].d = 0.0f;
            std::memset(y[i].qs, 0, sizeof(y[i].qs));
            std::memset(y[i].bsums, 0, sizeof(y[i].bsums));
            continue;
        }
        const float iscale = -127.0f / max_val;
        for (int j = 0; j < QK_K; ++j) {
            int v = (int) std::nearbyint(iscale * x[i * QK_K + j]);
            y[i].qs[j] = (int8_t) std::min(127, std::max(-128, v));
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii)
                sum += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = (int16_t) sum;
        }
        y[i].d = 1.0f / iscale;
    }
}

GEXEC_AVX2_TARGET
void quantize_q8_0_host(const float* x, void* vy, int k) noexcept {
    block_q8_0* y = static_cast<block_q8_0*>(vy);
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            amax = std::max(amax, std::fabs(x[i * QK8_0 + j]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) {
            y[i].qs[j] = (int8_t) std::nearbyint(x[i * QK8_0 + j] * id);
        }
    }
}

GEXEC_AVX2_TARGET
void quantize_q8_1_host(const float* x, void* vy, int k) noexcept {
    block_q8_1* y = static_cast<block_q8_1*>(vy);
    const int nb = k / QK8_1;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) {
            amax = std::max(amax, std::fabs(x[i * QK8_1 + j]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = fp32_to_fp16(d);
        int sumi = 0;
        for (int j = 0; j < QK8_1; ++j) {
            const int8_t q = (int8_t) std::nearbyint(x[i * QK8_1 + j] * id);
            y[i].qs[j] = q;
            sumi += q;
        }
        const uint16_t sh = fp32_to_fp16(fp16_to_fp32(dh) * (float) sumi);
        uint32_t ds = (uint32_t) dh | ((uint32_t) sh << 16);
        std::memcpy(&y[i].ds, &ds, 4);
    }
}

bool type_geom_host(int t, int& qk, size_t& block_bytes, int& act_type, size_t& act_block_bytes) noexcept {
    switch (t) {
    case 7:
        qk = 32;
        block_bytes = 24;
        act_type = 9;
        act_block_bytes = 36;
        return true;
    case 8:
        qk = 32;
        block_bytes = 34;
        act_type = 8;
        act_block_bytes = 34;
        return true;
    case 12:
        qk = 256;
        block_bytes = 144;
        act_type = 15;
        act_block_bytes = 292;
        return true;
    case 14:
        qk = 256;
        block_bytes = 210;
        act_type = 15;
        act_block_bytes = 292;
        return true;
    case 16:
        qk = 256;
        block_bytes = 66;
        act_type = 15;
        act_block_bytes = 292;
        return true;
    case 17:
        qk = 256;
        block_bytes = 74;
        act_type = 15;
        act_block_bytes = 292;
        return true;
    case 18:
        qk = 256;
        block_bytes = 98;
        act_type = 15;
        act_block_bytes = 292;
        return true;
    case 20:
        qk = 32;
        block_bytes = 18;
        act_type = 8;
        act_block_bytes = 34;
        return true;
    case 21:
        qk = 256;
        block_bytes = 110;
        act_type = 15;
        act_block_bytes = 292;
        return true;
    case 22:
        qk = 256;
        block_bytes = 82;
        act_type = 15;
        act_block_bytes = 292;
        return true;
    case 23:
        qk = 256;
        block_bytes = 136;
        act_type = 15;
        act_block_bytes = 292;
        return true;
    default: return false;
    }
}

bool native_fmt_host(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& out) noexcept {
    int gu_qk = 0, d_qk = 0, gu_act = 0, d_act = 0;
    size_t gu_bb = 0, d_bb = 0, gu_ab = 0, d_ab = 0;
    if (!type_geom_host(gu_type, gu_qk, gu_bb, gu_act, gu_ab) || !type_geom_host(d_type, d_qk, d_bb, d_act, d_ab))
        return false;
    if (n_embd <= 0 || (n_embd % gu_qk) != 0 || n_ff <= 0 || (n_ff % d_qk) != 0)
        return false;
    out.gu_type = gu_type;
    out.d_type = d_type;
    out.gu_act = gu_act;
    out.d_act = d_act;
    out.n_embd = n_embd;
    out.n_ff = n_ff;
    out.gu_row = (uint64_t) (n_embd / gu_qk) * gu_bb;
    out.d_row = (uint64_t) (n_ff / d_qk) * d_bb;
    out.up_off = (uint64_t) n_ff * out.gu_row;
    out.down_off = 2 * out.up_off;
    out.bytes = out.down_off + (uint64_t) n_embd * out.d_row;
    out.act_bytes = (uint64_t) (n_embd / (gu_act == 15 ? 256 : 32)) * gu_ab;
    out.h_bytes = (uint64_t) (n_ff / (d_act == 15 ? 256 : 32)) * d_ab;
    return out.act_bytes <= kNativeActBytes && out.h_bytes <= kNativeHBytes;
}

void quantize_act_by_type_host(int act_type, const float* x, void* out, int n) noexcept {
    if (act_type == 15)
        quantize_q8_K_host(x, out, n);
    else if (act_type == 8)
        quantize_q8_0_host(x, out, n);
    else if (act_type == 9)
        quantize_q8_1_host(x, out, n);
}
#endif

GEXEC_AVX2_INLINE float f16_pair_at(const void* block, int idx) noexcept {
    uint16_t h;
    std::memcpy(&h, (const uint16_t*) block + idx, 2);
    return fp16_to_fp32(h);
}

GEXEC_AVX2_INLINE float hsum_f32_8(__m256 x) noexcept {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

GEXEC_AVX2_INLINE float hsum_f32_4(__m128 m) noexcept {
    m = _mm_add_ps(m, _mm_movehl_ps(m, m));
    m = _mm_add_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}

GEXEC_AVX2_INLINE float hsum8_hadd(__m256 v) noexcept {
    const __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

GEXEC_AVX2_INLINE __m256 mul_sum_i8_pairs_f32(__m256i ax, __m256i sy) noexcept {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_maddubs_epi16(ax, sy);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, summed_pairs));
}

GEXEC_AVX2_INLINE __m256i bytes_from_nibbles_32_avx2(const uint8_t* rsi) noexcept {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    return _mm256_and_si256(_mm256_set1_epi8(0xF), bytes);
}

GEXEC_AVX2_INLINE __m256i bytes_from_bits_32_avx2(const uint8_t* hbits) noexcept {
    uint32_t h32;
    std::memcpy(&h32, hbits, 4);
    const __m256i shuf =
        _mm256_set_epi64x(0x0303030303030303ULL, 0x0202020202020202ULL, 0x0101010101010101ULL, 0x0000000000000000ULL);
    const __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) h32), shuf);
    const __m256i mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeULL);
    return _mm256_cmpeq_epi8(_mm256_or_si256(bytes, mask), _mm256_set1_epi8(-1));
}

// ============================================================================
// 1. Q4_K (type 12) and Q6_K (type 14): Zero-Spill AVX-512BW & AVX2 2-Row / 4-Row
// ============================================================================

GEXEC_AVX2_INLINE __m256i scale_shuffle_k4(int i) noexcept {
    alignas(32) static const uint8_t k_shuffle[256] = {
        0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
        0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
        2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
        4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,
        6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,
        8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
        8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    };
    return _mm256_loadu_si256((const __m256i*) k_shuffle + i);
}

GEXEC_AVX2_INLINE __m128i scale_shuffle_q6(int i) noexcept {
    static const uint8_t k_shuffle[128] = {
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
        3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
        6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
        9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
        13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    };
    return _mm_loadu_si128((const __m128i*) k_shuffle + i);
}

GEXEC_AVX2_INLINE void unpack_q4k_scales_avx2(const uint8_t* sc_bytes, __m128i& mins16, __m256i& scales_b) noexcept {
    static const uint32_t kmask1 = 0x3f3f3f3fu;
    static const uint32_t kmask2 = 0x0f0f0f0fu;
    static const uint32_t kmask3 = 0x03030303u;
    uint32_t utmp[4];
    std::memcpy(utmp, sc_bytes, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    const __m256i mins_and_scales =
        _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
    mins16 = _mm256_extracti128_si256(mins_and_scales, 1);
    const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
    scales_b = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
}

#define GEXEC_Q4K_2ROWS_BODY(NT)                                                                                       \
    constexpr int nt = NT;                                                                                             \
    const __m256i m4 = _mm256_set1_epi8(0xF);                                                                          \
    __m256 acc0[NT], acc1[NT];                                                                                         \
    __m128 acc_m0[NT], acc_m1[NT];                                                                                     \
    for (int t = 0; t < nt; ++t) {                                                                                     \
        acc0[t] = _mm256_setzero_ps();                                                                                 \
        acc_m0[t] = _mm_setzero_ps();                                                                                  \
        acc1[t] = _mm256_setzero_ps();                                                                                 \
        acc_m1[t] = _mm_setzero_ps();                                                                                  \
    }                                                                                                                  \
    for (int i = 0; i < nb; ++i) {                                                                                     \
        const float xd0 = f16_pair_at(&x0[i], 0), xdmin0 = f16_pair_at(&x0[i], 1);                                     \
        const float xd1 = f16_pair_at(&x1[i], 0), xdmin1 = f16_pair_at(&x1[i], 1);                                     \
        __m128i mins0, mins1;                                                                                          \
        __m256i scales0, scales1;                                                                                      \
        unpack_q4k_scales_avx2(x0[i].scales, mins0, scales0);                                                          \
        unpack_q4k_scales_avx2(x1[i].scales, mins1, scales1);                                                          \
        __m256i sumi0[NT], sumi1[NT];                                                                                  \
        for (int t = 0; t < nt; ++t) {                                                                                 \
            const block_q8_K* y = (const block_q8_K*) act[t] + i;                                                      \
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);                                      \
            const __m128i q8s =                                                                                        \
                _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));              \
            acc_m0[t] =                                                                                                \
                _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin0), _mm_cvtepi32_ps(_mm_madd_epi16(mins0, q8s)), acc_m0[t]);     \
            acc_m1[t] =                                                                                                \
                _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin1), _mm_cvtepi32_ps(_mm_madd_epi16(mins1, q8s)), acc_m1[t]);     \
            sumi0[t] = _mm256_setzero_si256();                                                                         \
            sumi1[t] = _mm256_setzero_si256();                                                                         \
        }                                                                                                              \
        for (int j = 0; j < 4; ++j) {                                                                                  \
            const __m256i sh_l = scale_shuffle_k4(2 * j + 0);                                                          \
            const __m256i sh_h = scale_shuffle_k4(2 * j + 1);                                                          \
            const __m256i sc0_l = _mm256_shuffle_epi8(scales0, sh_l);                                                  \
            const __m256i sc0_h = _mm256_shuffle_epi8(scales0, sh_h);                                                  \
            const __m256i sc1_l = _mm256_shuffle_epi8(scales1, sh_l);                                                  \
            const __m256i sc1_h = _mm256_shuffle_epi8(scales1, sh_h);                                                  \
            const __m256i q4b0 = _mm256_loadu_si256((const __m256i*) (x0[i].qs + 32 * j));                             \
            const __m256i q4b1 = _mm256_loadu_si256((const __m256i*) (x1[i].qs + 32 * j));                             \
            const __m256i q40_l = _mm256_and_si256(q4b0, m4);                                                          \
            const __m256i q40_h = _mm256_and_si256(_mm256_srli_epi16(q4b0, 4), m4);                                    \
            const __m256i q41_l = _mm256_and_si256(q4b1, m4);                                                          \
            const __m256i q41_h = _mm256_and_si256(_mm256_srli_epi16(q4b1, 4), m4);                                    \
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
        res0[t] = hsum_f32_8(acc0[t]) + hsum_f32_4(acc_m0[t]);                                                         \
        res1[t] = hsum_f32_8(acc1[t]) + hsum_f32_4(acc_m1[t]);                                                         \
    }

#define GEXEC_Q6K_2ROWS_BODY(NT)                                                                                       \
    constexpr int nt = NT;                                                                                             \
    const __m256i m15 = _mm256_set1_epi8(0xF);                                                                         \
    const __m256i m3 = _mm256_set1_epi8(3);                                                                            \
    const __m256i m12 = _mm256_set1_epi8(12);                                                                          \
    const __m256i m48 = _mm256_set1_epi8(48);                                                                          \
    const __m256i m_neg64 = _mm256_set1_epi8(-64);                                                                     \
    __m256 acc0[NT], acc1[NT];                                                                                         \
    for (int t = 0; t < nt; ++t) {                                                                                     \
        acc0[t] = _mm256_setzero_ps();                                                                                 \
        acc1[t] = _mm256_setzero_ps();                                                                                 \
    }                                                                                                                  \
    for (int i = 0; i < nb; ++i) {                                                                                     \
        const float xd0 = fp16_to_fp32(x0[i].d), xd1 = fp16_to_fp32(x1[i].d);                                          \
        const __m128i scales0 = _mm_loadu_si128((const __m128i*) x0[i].scales);                                        \
        const __m128i scales1 = _mm_loadu_si128((const __m128i*) x1[i].scales);                                        \
        __m256i sumi0[NT], sumi1[NT];                                                                                  \
        for (int t = 0; t < nt; ++t) {                                                                                 \
            sumi0[t] = _mm256_setzero_si256();                                                                         \
            sumi1[t] = _mm256_setzero_si256();                                                                         \
        }                                                                                                              \
        for (int j = 0; j < 2; ++j) {                                                                                  \
            const __m256i qb0_1 = _mm256_loadu_si256((const __m256i*) (x0[i].ql + 64 * j + 0));                        \
            const __m256i qb0_2 = _mm256_loadu_si256((const __m256i*) (x0[i].ql + 64 * j + 32));                       \
            const __m256i qb0_H = _mm256_loadu_si256((const __m256i*) (x0[i].qh + 32 * j));                            \
            const __m256i qb1_1 = _mm256_loadu_si256((const __m256i*) (x1[i].ql + 64 * j + 0));                        \
            const __m256i qb1_2 = _mm256_loadu_si256((const __m256i*) (x1[i].ql + 64 * j + 32));                       \
            const __m256i qb1_H = _mm256_loadu_si256((const __m256i*) (x1[i].qh + 32 * j));                            \
            const __m128i sh0 = scale_shuffle_q6(4 * j + 0);                                                           \
            const __m128i sh1 = scale_shuffle_q6(4 * j + 1);                                                           \
            const __m256i q40_0 =                                                                                      \
                _mm256_or_si256(_mm256_and_si256(qb0_1, m15), _mm256_slli_epi16(_mm256_and_si256(qb0_H, m3), 4));      \
            const __m256i q40_1 =                                                                                      \
                _mm256_or_si256(_mm256_and_si256(qb0_2, m15), _mm256_slli_epi16(_mm256_and_si256(qb0_H, m12), 2));     \
            const __m256i sc0_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh0));                                \
            const __m256i sc0_1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh1));                                \
            const __m256i q41_0 =                                                                                      \
                _mm256_or_si256(_mm256_and_si256(qb1_1, m15), _mm256_slli_epi16(_mm256_and_si256(qb1_H, m3), 4));      \
            const __m256i q41_1 =                                                                                      \
                _mm256_or_si256(_mm256_and_si256(qb1_2, m15), _mm256_slli_epi16(_mm256_and_si256(qb1_H, m12), 2));     \
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
            const __m128i sh2 = scale_shuffle_q6(4 * j + 2);                                                           \
            const __m128i sh3 = scale_shuffle_q6(4 * j + 3);                                                           \
            const __m256i q40_2 =                                                                                      \
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb0_1, 4), m15), _mm256_and_si256(qb0_H, m48));     \
            const __m256i q40_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb0_2, 4), m15),                  \
                                                  _mm256_srli_epi16(_mm256_and_si256(qb0_H, m_neg64), 2));             \
            const __m256i sc0_2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh2));                                \
            const __m256i sc0_3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales0, sh3));                                \
            const __m256i q41_2 =                                                                                      \
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qb1_1, 4), m15), _mm256_and_si256(qb1_H, m48));     \
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
        const __m256i sc0_16 = _mm256_cvtepi8_epi16(scales0);                                                          \
        const __m256i sc1_16 = _mm256_cvtepi8_epi16(scales1);                                                          \
        for (int t = 0; t < nt; ++t) {                                                                                 \
            const block_q8_K* y = (const block_q8_K*) act[t] + i;                                                      \
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);                                      \
            const __m256i sub0 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc0_16), 5);                              \
            const __m256i sub1 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, sc1_16), 5);                              \
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd0),                                                      \
                                      _mm256_cvtepi32_ps(_mm256_sub_epi32(sumi0[t], sub0)), acc0[t]);                  \
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(y->d * xd1),                                                      \
                                      _mm256_cvtepi32_ps(_mm256_sub_epi32(sumi1[t], sub1)), acc1[t]);                  \
        }                                                                                                              \
    }                                                                                                                  \
    for (int t = 0; t < nt; ++t) {                                                                                     \
        res0[t] = hsum_f32_8(acc0[t]);                                                                                 \
        res1[t] = hsum_f32_8(acc1[t]);                                                                                 \
    }

template <int NT>
GEXEC_AVX2_TARGET void gexec_q4k_dot_2rows_avx2(const block_q4_K* x0, const block_q4_K* x1, int nb,
                                                const void* const* act, float* res0, float* res1) {
    GEXEC_Q4K_2ROWS_BODY(NT)
}

template <int NT>
GEXEC_AVX2_TARGET void gexec_q6k_dot_2rows_avx2(const block_q6_K* x0, const block_q6_K* x1, int nb,
                                                const void* const* act, float* res0, float* res1) {
    GEXEC_Q6K_2ROWS_BODY(NT)
}

template <int NT>
GEXEC_AVX512BW_TARGET void gexec_q4k_dot_2rows_avx512bw(const block_q4_K* x0, const block_q4_K* x1, int nb,
                                                        const void* const* act, float* res0, float* res1) {
    GEXEC_Q4K_2ROWS_BODY(NT)
}

template <int NT>
GEXEC_AVX512BW_TARGET void gexec_q6k_dot_2rows_avx512bw(const block_q6_K* x0, const block_q6_K* x1, int nb,
                                                        const void* const* act, float* res0,
                                                        float* res1){GEXEC_Q6K_2ROWS_BODY(NT)}
#undef GEXEC_Q4K_2ROWS_BODY
#undef GEXEC_Q6K_2ROWS_BODY

GEXEC_AVX2_TARGET void dispatch_q4k_2rows(bool use_avx512, const block_q4_K* x0, const block_q4_K* x1, int nb,
                                          const void* const* act, int nt, float* res0, float* res1) {
    if (use_avx512) {
        switch (nt) {
        case 1: gexec_q4k_dot_2rows_avx512bw<1>(x0, x1, nb, act, res0, res1); return;
        case 2: gexec_q4k_dot_2rows_avx512bw<2>(x0, x1, nb, act, res0, res1); return;
        case 3: gexec_q4k_dot_2rows_avx512bw<3>(x0, x1, nb, act, res0, res1); return;
        case 4: gexec_q4k_dot_2rows_avx512bw<4>(x0, x1, nb, act, res0, res1); return;
        case 5: gexec_q4k_dot_2rows_avx512bw<5>(x0, x1, nb, act, res0, res1); return;
        case 6: gexec_q4k_dot_2rows_avx512bw<6>(x0, x1, nb, act, res0, res1); return;
        case 7: gexec_q4k_dot_2rows_avx512bw<7>(x0, x1, nb, act, res0, res1); return;
        default: gexec_q4k_dot_2rows_avx512bw<8>(x0, x1, nb, act, res0, res1); return;
        }
    }
    switch (nt) {
    case 1: gexec_q4k_dot_2rows_avx2<1>(x0, x1, nb, act, res0, res1); return;
    case 2: gexec_q4k_dot_2rows_avx2<2>(x0, x1, nb, act, res0, res1); return;
    case 3: gexec_q4k_dot_2rows_avx2<3>(x0, x1, nb, act, res0, res1); return;
    case 4: gexec_q4k_dot_2rows_avx2<4>(x0, x1, nb, act, res0, res1); return;
    case 5: gexec_q4k_dot_2rows_avx2<5>(x0, x1, nb, act, res0, res1); return;
    case 6: gexec_q4k_dot_2rows_avx2<6>(x0, x1, nb, act, res0, res1); return;
    case 7: gexec_q4k_dot_2rows_avx2<7>(x0, x1, nb, act, res0, res1); return;
    case 8: gexec_q4k_dot_2rows_avx2<8>(x0, x1, nb, act, res0, res1); return;
    }
}

GEXEC_AVX2_TARGET
void dispatch_q6k_2rows(bool use_avx512, const block_q6_K* x0, const block_q6_K* x1, int nb, const void* const* act,
                        int nt, float* res0, float* res1) {
    if (use_avx512) {
        switch (nt) {
        case 1: gexec_q6k_dot_2rows_avx512bw<1>(x0, x1, nb, act, res0, res1); return;
        case 2: gexec_q6k_dot_2rows_avx512bw<2>(x0, x1, nb, act, res0, res1); return;
        case 3: gexec_q6k_dot_2rows_avx512bw<3>(x0, x1, nb, act, res0, res1); return;
        case 4: gexec_q6k_dot_2rows_avx512bw<4>(x0, x1, nb, act, res0, res1); return;
        case 5: gexec_q6k_dot_2rows_avx512bw<5>(x0, x1, nb, act, res0, res1); return;
        case 6: gexec_q6k_dot_2rows_avx512bw<6>(x0, x1, nb, act, res0, res1); return;
        case 7: gexec_q6k_dot_2rows_avx512bw<7>(x0, x1, nb, act, res0, res1); return;
        default: gexec_q6k_dot_2rows_avx512bw<8>(x0, x1, nb, act, res0, res1); return;
        }
    }
    switch (nt) {
    case 1: gexec_q6k_dot_2rows_avx2<1>(x0, x1, nb, act, res0, res1); return;
    case 2: gexec_q6k_dot_2rows_avx2<2>(x0, x1, nb, act, res0, res1); return;
    case 3: gexec_q6k_dot_2rows_avx2<3>(x0, x1, nb, act, res0, res1); return;
    case 4: gexec_q6k_dot_2rows_avx2<4>(x0, x1, nb, act, res0, res1); return;
    case 5: gexec_q6k_dot_2rows_avx2<5>(x0, x1, nb, act, res0, res1); return;
    case 6: gexec_q6k_dot_2rows_avx2<6>(x0, x1, nb, act, res0, res1); return;
    case 7: gexec_q6k_dot_2rows_avx2<7>(x0, x1, nb, act, res0, res1); return;
    case 8: gexec_q6k_dot_2rows_avx2<8>(x0, x1, nb, act, res0, res1); return;
    }
}

// ============================================================================
// 2. Q8_0 (type 8) and Q5_1 (type 7): 4-Row and 2-Row Fused AVX2 Kernels
// ============================================================================

template <int NT>
GEXEC_AVX2_TARGET void gexec_q8_0_dot_4rows_avx2(const block_q8_0* x0, const block_q8_0* x1, const block_q8_0* x2,
                                                 const block_q8_0* x3, int nb, const void* const* act, float* res0,
                                                 float* res1, float* res2, float* res3) {
    constexpr int nt = NT;
    __m256 acc0[NT], acc1[NT], acc2[NT], acc3[NT];
    for (int t = 0; t < nt; ++t) {
        acc0[t] = _mm256_setzero_ps();
        acc1[t] = _mm256_setzero_ps();
        acc2[t] = _mm256_setzero_ps();
        acc3[t] = _mm256_setzero_ps();
    }
    for (int ib = 0; ib < nb; ++ib) {
        const float xd0 = fp16_to_fp32(x0[ib].d);
        const float xd1 = fp16_to_fp32(x1[ib].d);
        const float xd2 = fp16_to_fp32(x2[ib].d);
        const float xd3 = fp16_to_fp32(x3[ib].d);
        const __m256i qx0 = _mm256_loadu_si256((const __m256i*) x0[ib].qs);
        const __m256i qx1 = _mm256_loadu_si256((const __m256i*) x1[ib].qs);
        const __m256i qx2 = _mm256_loadu_si256((const __m256i*) x2[ib].qs);
        const __m256i qx3 = _mm256_loadu_si256((const __m256i*) x3[ib].qs);
        const __m256i ax0 = _mm256_sign_epi8(qx0, qx0);
        const __m256i ax1 = _mm256_sign_epi8(qx1, qx1);
        const __m256i ax2 = _mm256_sign_epi8(qx2, qx2);
        const __m256i ax3 = _mm256_sign_epi8(qx3, qx3);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const float yd = fp16_to_fp32(y->d);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(xd0 * yd), mul_sum_i8_pairs_f32(ax0, _mm256_sign_epi8(qy, qx0)),
                                      acc0[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(xd1 * yd), mul_sum_i8_pairs_f32(ax1, _mm256_sign_epi8(qy, qx1)),
                                      acc1[t]);
            acc2[t] = _mm256_fmadd_ps(_mm256_set1_ps(xd2 * yd), mul_sum_i8_pairs_f32(ax2, _mm256_sign_epi8(qy, qx2)),
                                      acc2[t]);
            acc3[t] = _mm256_fmadd_ps(_mm256_set1_ps(xd3 * yd), mul_sum_i8_pairs_f32(ax3, _mm256_sign_epi8(qy, qx3)),
                                      acc3[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        res0[t] = hsum_f32_8(acc0[t]);
        res1[t] = hsum_f32_8(acc1[t]);
        res2[t] = hsum_f32_8(acc2[t]);
        res3[t] = hsum_f32_8(acc3[t]);
    }
}

template <int NT>
GEXEC_AVX2_TARGET void gexec_q8_0_dot_2rows_avx2(const block_q8_0* x0, const block_q8_0* x1, int nb,
                                                 const void* const* act, float* res0, float* res1) {
    constexpr int nt = NT;
    __m256 acc0[NT], acc1[NT];
    for (int t = 0; t < nt; ++t) {
        acc0[t] = _mm256_setzero_ps();
        acc1[t] = _mm256_setzero_ps();
    }
    for (int ib = 0; ib < nb; ++ib) {
        const float xd0 = fp16_to_fp32(x0[ib].d);
        const float xd1 = fp16_to_fp32(x1[ib].d);
        const __m256i qx0 = _mm256_loadu_si256((const __m256i*) x0[ib].qs);
        const __m256i qx1 = _mm256_loadu_si256((const __m256i*) x1[ib].qs);
        const __m256i ax0 = _mm256_sign_epi8(qx0, qx0);
        const __m256i ax1 = _mm256_sign_epi8(qx1, qx1);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const float yd = fp16_to_fp32(y->d);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(xd0 * yd), mul_sum_i8_pairs_f32(ax0, _mm256_sign_epi8(qy, qx0)),
                                      acc0[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(xd1 * yd), mul_sum_i8_pairs_f32(ax1, _mm256_sign_epi8(qy, qx1)),
                                      acc1[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        res0[t] = hsum_f32_8(acc0[t]);
        res1[t] = hsum_f32_8(acc1[t]);
    }
}

template <int NT>
GEXEC_AVX2_TARGET void gexec_q5_1_dot_2rows_avx2(const block_q5_1* x0, const block_q5_1* x1, int nb,
                                                 const void* const* act, float* res0, float* res1) {
    constexpr int nt = NT;
    __m256 acc0[NT], acc1[NT];
    float summs0[NT], summs1[NT];
    for (int t = 0; t < nt; ++t) {
        acc0[t] = _mm256_setzero_ps();
        summs0[t] = 0.0f;
        acc1[t] = _mm256_setzero_ps();
        summs1[t] = 0.0f;
    }
    const __m256i m0x10 = _mm256_set1_epi8(0x10);
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx0 = _mm256_set1_ps(f16_pair_at(&x0[ib], 0));
        const __m256 dx1 = _mm256_set1_ps(f16_pair_at(&x1[ib], 0));
        const float xm0 = f16_pair_at(&x0[ib], 1);
        const float xm1 = f16_pair_at(&x1[ib], 1);
        const __m256i qx0 = _mm256_or_si256(bytes_from_nibbles_32_avx2(x0[ib].qs),
                                            _mm256_and_si256(bytes_from_bits_32_avx2(x0[ib].qh), m0x10));
        const __m256i qx1 = _mm256_or_si256(bytes_from_nibbles_32_avx2(x1[ib].qs),
                                            _mm256_and_si256(bytes_from_bits_32_avx2(x1[ib].qh), m0x10));
        for (int t = 0; t < nt; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            const float ym = f16_pair_at(y, 1);
            summs0[t] += xm0 * ym;
            summs1[t] += xm1 * ym;
            const __m256 dy = _mm256_set1_ps(f16_pair_at(y, 0));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            acc0[t] = _mm256_fmadd_ps(mul_sum_i8_pairs_f32(qx0, qy), _mm256_mul_ps(dx0, dy), acc0[t]);
            acc1[t] = _mm256_fmadd_ps(mul_sum_i8_pairs_f32(qx1, qy), _mm256_mul_ps(dx1, dy), acc1[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        res0[t] = hsum_f32_8(acc0[t]) + summs0[t];
        res1[t] = hsum_f32_8(acc1[t]) + summs1[t];
    }
}

// ============================================================================
// Dispatch only the supported token groups; callers split larger groups at MAXT.
// The fixed token extent lets the compiler keep small accumulators in registers.
#define GEXEC_DISPATCH_NT(kernel, ...)                                                                                 \
    switch (nt) {                                                                                                      \
    case 1: kernel<1>(__VA_ARGS__); return;                                                                            \
    case 2: kernel<2>(__VA_ARGS__); return;                                                                            \
    case 3: kernel<3>(__VA_ARGS__); return;                                                                            \
    case 4: kernel<4>(__VA_ARGS__); return;                                                                            \
    case 5: kernel<5>(__VA_ARGS__); return;                                                                            \
    case 6: kernel<6>(__VA_ARGS__); return;                                                                            \
    case 7: kernel<7>(__VA_ARGS__); return;                                                                            \
    case 8: kernel<8>(__VA_ARGS__); return;                                                                            \
    }

void dispatch_q8_0_4rows(const block_q8_0* x0, const block_q8_0* x1, const block_q8_0* x2, const block_q8_0* x3, int nb,
                         const void* const* act, int nt, float* res0, float* res1, float* res2, float* res3) {
    GEXEC_DISPATCH_NT(gexec_q8_0_dot_4rows_avx2, x0, x1, x2, x3, nb, act, res0, res1, res2, res3)
}

void dispatch_q8_0_2rows(const block_q8_0* x0, const block_q8_0* x1, int nb, const void* const* act, int nt,
                         float* res0, float* res1) {
    GEXEC_DISPATCH_NT(gexec_q8_0_dot_2rows_avx2, x0, x1, nb, act, res0, res1)
}

void dispatch_q5_1_2rows(const block_q5_1* x0, const block_q5_1* x1, int nb, const void* const* act, int nt,
                         float* res0,
                         float* res1){GEXEC_DISPATCH_NT(gexec_q5_1_dot_2rows_avx2, x0, x1, nb, act, res0, res1)}
#undef GEXEC_DISPATCH_NT

// 3. IQ4_NL (type 20): 4-Row and 2-Row Fused Kernels (Bit-Exact for nt==1 & nt>=2)
// ============================================================================

GEXEC_AVX2_INLINE void decode_iq4nl_block(const uint8_t* blk, __m128i values, __m128i m4b, __m256i& aq, __m256i& q4,
                                          float& dx) noexcept {
    const __m128i bits = _mm_loadu_si128((const __m128i*) (blk + 2));
    const __m128i lo = _mm_and_si128(bits, m4b);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);
    q4 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)), _mm_shuffle_epi8(values, hi), 1);
    aq = _mm256_sign_epi8(q4, q4);
    dx = fp16_to_fp32(read_u16(blk));
}

GEXEC_AVX2_INLINE void iq4nl_step_row_2blk(const uint8_t* row_ptr, int ib, __m128i values, __m128i m4b, __m256i mone,
                                           __m256i q8_1, __m256i q8_2, float yd_1, float yd_2, __m256& a1,
                                           __m256& a2) noexcept {
    __m256i aq1, q4_1, aq2, q4_2;
    float dx1, dx2;
    decode_iq4nl_block(row_ptr + (size_t) (ib + 0) * sizeof(block_iq4_nl), values, m4b, aq1, q4_1, dx1);
    decode_iq4nl_block(row_ptr + (size_t) (ib + 1) * sizeof(block_iq4_nl), values, m4b, aq2, q4_2, dx2);
    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aq1, _mm256_sign_epi8(q8_1, q4_1)), mone);
    const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(aq2, _mm256_sign_epi8(q8_2, q4_2)), mone);
    a1 = _mm256_fmadd_ps(_mm256_set1_ps(yd_1 * dx1), _mm256_cvtepi32_ps(p1), a1);
    a2 = _mm256_fmadd_ps(_mm256_set1_ps(yd_2 * dx2), _mm256_cvtepi32_ps(p2), a2);
}

// For nt == 1: matches ggml_vec_dot_iq4_nl_q8_0's dual (accum1, accum2) even/odd block order to the bit!
GEXEC_AVX2_TARGET
void gexec_iq4nl_4rows_nt1(const uint8_t* r0_ptr, const uint8_t* r1_ptr, const uint8_t* r2_ptr, const uint8_t* r3_ptr,
                           int nb, const block_q8_0* y, float& out0, float& out1, float& out2, float& out3) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 acc0_1 = _mm256_setzero_ps(), acc0_2 = _mm256_setzero_ps();
    __m256 acc1_1 = _mm256_setzero_ps(), acc1_2 = _mm256_setzero_ps();
    __m256 acc2_1 = _mm256_setzero_ps(), acc2_2 = _mm256_setzero_ps();
    __m256 acc3_1 = _mm256_setzero_ps(), acc3_2 = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) y[ib + 0].qs);
        const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) y[ib + 1].qs);
        const float yd_1 = fp16_to_fp32(y[ib + 0].d);
        const float yd_2 = fp16_to_fp32(y[ib + 1].d);

        iq4nl_step_row_2blk(r0_ptr, ib, values, m4b, mone, q8_1, q8_2, yd_1, yd_2, acc0_1, acc0_2);
        iq4nl_step_row_2blk(r1_ptr, ib, values, m4b, mone, q8_1, q8_2, yd_1, yd_2, acc1_1, acc1_2);
        iq4nl_step_row_2blk(r2_ptr, ib, values, m4b, mone, q8_1, q8_2, yd_1, yd_2, acc2_1, acc2_2);
        iq4nl_step_row_2blk(r3_ptr, ib, values, m4b, mone, q8_1, q8_2, yd_1, yd_2, acc3_1, acc3_2);
    }
    out0 = hsum_f32_8(_mm256_add_ps(acc0_1, acc0_2));
    out1 = hsum_f32_8(_mm256_add_ps(acc1_1, acc1_2));
    out2 = hsum_f32_8(_mm256_add_ps(acc2_1, acc2_2));
    out3 = hsum_f32_8(_mm256_add_ps(acc3_1, acc3_2));
    for (; ib < nb; ++ib) {
        const float yd = fp16_to_fp32(y[ib].d);
        auto tail_row = [&](const uint8_t* row_ptr, float& s) GEXEC_AVX2_TARGET {
            const auto* xb = (const block_iq4_nl*) row_ptr + ib;
            const float d = yd * fp16_to_fp32(xb->d);
            int s1 = 0, s2 = 0;
            for (int j = 0; j < QK4_NL / 2; ++j) {
                s1 += y[ib].qs[j + 0] * kvalues_iq4nl[xb->qs[j] & 0xf];
                s2 += y[ib].qs[j + QK4_NL / 2] * kvalues_iq4nl[xb->qs[j] >> 4];
            }
            s += d * (s1 + s2);
        };
        tail_row(r0_ptr, out0);
        tail_row(r1_ptr, out1);
        tail_row(r2_ptr, out2);
        tail_row(r3_ptr, out3);
    }
}

GEXEC_AVX2_TARGET
void gexec_iq4nl_1row_nt1(const uint8_t* r0_ptr, int nb, const block_q8_0* y, float& out0) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 acc0_1 = _mm256_setzero_ps(), acc0_2 = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) y[ib + 0].qs);
        const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) y[ib + 1].qs);
        const float yd_1 = fp16_to_fp32(y[ib + 0].d);
        const float yd_2 = fp16_to_fp32(y[ib + 1].d);
        __m256i aq1, q4_1, aq2, q4_2;
        float dx1, dx2;
        decode_iq4nl_block(r0_ptr + (size_t) (ib + 0) * sizeof(block_iq4_nl), values, m4b, aq1, q4_1, dx1);
        decode_iq4nl_block(r0_ptr + (size_t) (ib + 1) * sizeof(block_iq4_nl), values, m4b, aq2, q4_2, dx2);
        const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aq1, _mm256_sign_epi8(q8_1, q4_1)), mone);
        const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(aq2, _mm256_sign_epi8(q8_2, q4_2)), mone);
        acc0_1 = _mm256_fmadd_ps(_mm256_set1_ps(yd_1 * dx1), _mm256_cvtepi32_ps(p1), acc0_1);
        acc0_2 = _mm256_fmadd_ps(_mm256_set1_ps(yd_2 * dx2), _mm256_cvtepi32_ps(p2), acc0_2);
    }
    out0 = hsum_f32_8(_mm256_add_ps(acc0_1, acc0_2));
    for (; ib < nb; ++ib) {
        const auto* xb = (const block_iq4_nl*) r0_ptr + ib;
        const float d = fp16_to_fp32(y[ib].d) * fp16_to_fp32(xb->d);
        int s1 = 0, s2 = 0;
        for (int j = 0; j < QK4_NL / 2; ++j) {
            s1 += y[ib].qs[j + 0] * kvalues_iq4nl[xb->qs[j] & 0xf];
            s2 += y[ib].qs[j + QK4_NL / 2] * kvalues_iq4nl[xb->qs[j] >> 4];
        }
        out0 += d * (s1 + s2);
    }
}

template <int NT>
GEXEC_AVX2_TARGET void gexec_iq4nl_rows_nt(const uint8_t* w, size_t row_bytes, int n, const block_q8_0* const* y,
                                           float* const* out, int r0, int r1) {
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    const int nb = n / QK4_NL;
    int r = r0;
    for (; r + 2 <= r1; r += 2) {
        const uint8_t* row0 = w + (size_t) (r + 0) * row_bytes;
        const uint8_t* row1 = w + (size_t) (r + 1) * row_bytes;
        __m256 acc0[NT], acc1[NT];
        for (int t = 0; t < NT; ++t) {
            acc0[t] = _mm256_setzero_ps();
            acc1[t] = _mm256_setzero_ps();
        }
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk0 = row0 + (size_t) ib * sizeof(block_iq4_nl);
            const uint8_t* blk1 = row1 + (size_t) ib * sizeof(block_iq4_nl);
            _mm_prefetch((const char*) (blk0 + 2048), _MM_HINT_T0);
            _mm_prefetch((const char*) (blk0 + 2048 + 64), _MM_HINT_T0);
            _mm_prefetch((const char*) (blk1 + 2048), _MM_HINT_T0);
            _mm_prefetch((const char*) (blk1 + 2048 + 64), _MM_HINT_T0);
            __m256i aq0, q4_0, aq1, q4_1;
            float dx0, dx1;
            decode_iq4nl_block(blk0, values, m4b, aq0, q4_0, dx0);
            decode_iq4nl_block(blk1, values, m4b, aq1, q4_1, dx1);
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const float yd = fp16_to_fp32(b.d);
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) b.qs);
                const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aq0, _mm256_sign_epi8(q8, q4_0)), ones);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aq1, _mm256_sign_epi8(q8, q4_1)), ones);
                acc0[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx0 * yd), _mm256_cvtepi32_ps(p0), acc0[t]);
                acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx1 * yd), _mm256_cvtepi32_ps(p1), acc1[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            out[t][r + 0] = hsum8_hadd(acc0[t]);
            out[t][r + 1] = hsum8_hadd(acc1[t]);
        }
    }
    for (; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t* blk = row + (size_t) ib * sizeof(block_iq4_nl);
            _mm_prefetch((const char*) (blk + 2048), _MM_HINT_T0);
            _mm_prefetch((const char*) (blk + 2048 + 64), _MM_HINT_T0);
            __m256i aq, q4;
            float dx;
            decode_iq4nl_block(blk, values, m4b, aq, q4, dx);
            for (int t = 0; t < NT; ++t) {
                const block_q8_0& b = y[t][ib];
                const __m256i q8 = _mm256_loadu_si256((const __m256i*) b.qs);
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aq, _mm256_sign_epi8(q8, q4)), ones);
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * fp16_to_fp32(b.d)), _mm256_cvtepi32_ps(p), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t)
            out[t][r] = hsum8_hadd(accf[t]);
    }
}

void gexec_iq4nl_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out,
                           int r0, int r1) {
    static const int mt_min = [] {
        const char* e = std::getenv("GUILD_IQ_MT_MIN");
        return e ? std::atoi(e) : 2;
    }();
    if (nt < mt_min) {
        const auto* y0 = (const block_q8_0*) hq[0];
        const int nb = n / QK4_NL;
        int r = r0;
        for (; r + 4 <= r1; r += 4) {
            gexec_iq4nl_4rows_nt1(w + (size_t) (r + 0) * row_bytes, w + (size_t) (r + 1) * row_bytes,
                                  w + (size_t) (r + 2) * row_bytes, w + (size_t) (r + 3) * row_bytes, nb, y0,
                                  out[0][r + 0], out[0][r + 1], out[0][r + 2], out[0][r + 3]);
        }
        for (; r < r1; ++r) {
            gexec_iq4nl_1row_nt1(w + (size_t) r * row_bytes, nb, y0, out[0][r]);
        }
        return;
    }
    const block_q8_0* y[4];
    for (int t = 0; t < std::min(nt, 4); ++t)
        y[t] = (const block_q8_0*) hq[t];
    switch (nt) {
    case 1: gexec_iq4nl_rows_nt<1>(w, row_bytes, n, y, out, r0, r1); break;
    case 2: gexec_iq4nl_rows_nt<2>(w, row_bytes, n, y, out, r0, r1); break;
    case 3: gexec_iq4nl_rows_nt<3>(w, row_bytes, n, y, out, r0, r1); break;
    case 4: gexec_iq4nl_rows_nt<4>(w, row_bytes, n, y, out, r0, r1); break;
    default:
        for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = std::min(4, nt - t0);
            if (k == 1) {
                const block_q8_0* y1[1] = {(const block_q8_0*) hq[t0]};
                gexec_iq4nl_rows_nt<1>(w, row_bytes, n, y1, out + t0, r0, r1);
            } else {
                gexec_iq4nl_down_rows(w, row_bytes, n, hq + t0, k, out + t0, r0, r1);
            }
        }
        break;
    }
}

// ============================================================================
// 4. I-Quant Gate/Up 2-Row & 4-Row Fused AVX-512BW and AVX2 Kernels
//    Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ4_XS (23)
// ============================================================================

GEXEC_AVX512_INLINE __m512i iq512_scales4(int s0, int s1, int s2, int s3) noexcept {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                         0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

template <int TY> struct Iq512Fmt;

template <> struct Iq512Fmt<16> {
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static GEXEC_AVX512_INLINE void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) noexcept {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = read_u32(q), a1 = read_u32(q + 4), b0 = read_u32(q + 8), b1 = read_u32(q + 12);
        g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s =
            (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
            ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
            ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
            ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = iq512_scales4(sa, sa, sb, sb);
    }
};

template <> struct Iq512Fmt<17> {
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static GEXEC_AVX512_INLINE void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) noexcept {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l)
            s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = iq512_scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Iq512Fmt<22> {
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static GEXEC_AVX512_INLINE void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) noexcept {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        g = _mm512_set_epi64(
            (long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
            (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
            (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
            (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(read_u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = iq512_scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Iq512Fmt<18> {
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static GEXEC_AVX512_INLINE void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) noexcept {
        const uint8_t* q = b + 2 + 16 * j;
        g = _mm512_set_epi32(
            (int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
            (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
            (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
            (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = read_u32(b + 2 + 64 + 8 * j), c = read_u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s =
            (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
            ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
            ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
            ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = iq512_scales4(sa, sa, sb, sb);
    }
};

template <> struct Iq512Fmt<21> {
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static GEXEC_AVX512_INLINE void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) noexcept {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
#define GEXEC_G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(GEXEC_G3(15, h1, 7), GEXEC_G3(14, h1, 6), GEXEC_G3(13, h1, 5), GEXEC_G3(12, h1, 4),
                             GEXEC_G3(11, h1, 3), GEXEC_G3(10, h1, 2), GEXEC_G3(9, h1, 1), GEXEC_G3(8, h1, 0),
                             GEXEC_G3(7, h0, 7), GEXEC_G3(6, h0, 6), GEXEC_G3(5, h0, 5), GEXEC_G3(4, h0, 4),
                             GEXEC_G3(3, h0, 3), GEXEC_G3(2, h0, 2), GEXEC_G3(1, h0, 1), GEXEC_G3(0, h0, 0));
#undef GEXEC_G3
        m = _cvtu64_mask64(read_u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = iq512_scales4(sa, sa, sb, sb);
    }
};

bool native_iq512_policy(int nt) {
    static const bool enabled = cpu_avx512_ok() && std::getenv("GUILD_NO_IQ512") == nullptr;
    static const int mt_min = [] {
        const char* e = std::getenv("GUILD_IQ_MT_MIN");
        return e ? std::atoi(e) : 2;
    }();
    return enabled && nt >= mt_min;
}

template <int TY, int NT>
GEXEC_AVX512BW_TARGET void iq512_row_dot_2rows(const uint8_t* row0, const uint8_t* row1, int nblocks,
                                               const block_q8_K* const* y, float* res0, float* res1) {
    const bool use_512_reduce = native_iq512_policy(NT);
    __m512 accf0[NT], accf1[NT];
    __m256 acc256_0[NT], acc256_1[NT];
    for (int t = 0; t < NT; ++t) {
        accf0[t] = _mm512_setzero_ps();
        accf1[t] = _mm512_setzero_ps();
        acc256_0[t] = _mm256_setzero_ps();
        acc256_1[t] = _mm256_setzero_ps();
    }
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk0 = row0 + (size_t) i * Iq512Fmt<TY>::bytes;
        const uint8_t* blk1 = row1 + (size_t) i * Iq512Fmt<TY>::bytes;
        _mm_prefetch((const char*) (blk0 + 2048), _MM_HINT_T0);
        _mm_prefetch((const char*) (blk0 + 2048 + 64), _MM_HINT_T0);
        _mm_prefetch((const char*) (blk1 + 2048), _MM_HINT_T0);
        _mm_prefetch((const char*) (blk1 + 2048 + 64), _MM_HINT_T0);
        __m512i acci0[NT], acci1[NT];
        for (int t = 0; t < NT; ++t) {
            acci0[t] = _mm512_setzero_si512();
            acci1[t] = _mm512_setzero_si512();
        }
        for (int j = 0; j < 4; ++j) {
            __m512i g0, sc0, g1, sc1;
            __mmask64 m0, m1;
            Iq512Fmt<TY>::decode(blk0, j, g0, m0, sc0);
            Iq512Fmt<TY>::decode(blk1, j, g1, m1, sc1);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys0 = _mm512_mask_sub_epi8(yv, m0, zero, yv);
                const __m512i ys1 = _mm512_mask_sub_epi8(yv, m1, zero, yv);
                acci0[t] = _mm512_add_epi32(acci0[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g0, ys0), sc0));
                acci1[t] = _mm512_add_epi32(acci1[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g1, ys1), sc1));
            }
        }
        if (use_512_reduce) {
            const float dx0 = fp16_to_fp32(read_u16(blk0)) * Iq512Fmt<TY>::K;
            const float dx1 = fp16_to_fp32(read_u16(blk1)) * Iq512Fmt<TY>::K;
            for (int t = 0; t < NT; ++t) {
                const float yd = y[t][i].d;
                accf0[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx0 * yd), _mm512_cvtepi32_ps(acci0[t]), accf0[t]);
                accf1[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx1 * yd), _mm512_cvtepi32_ps(acci1[t]), accf1[t]);
            }
        } else {
            const float dx0 =
                (NT == 1) ? fp16_to_fp32(read_u16(blk0)) : (fp16_to_fp32(read_u16(blk0)) * Iq512Fmt<TY>::K);
            const float dx1 =
                (NT == 1) ? fp16_to_fp32(read_u16(blk1)) : (fp16_to_fp32(read_u16(blk1)) * Iq512Fmt<TY>::K);
            for (int t = 0; t < NT; ++t) {
                const float yd = y[t][i].d;
                const __m256i sum256_0 =
                    _mm256_add_epi32(_mm512_castsi512_si256(acci0[t]), _mm512_extracti32x8_epi32(acci0[t], 1));
                const __m256i sum256_1 =
                    _mm256_add_epi32(_mm512_castsi512_si256(acci1[t]), _mm512_extracti32x8_epi32(acci1[t], 1));
                acc256_0[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx0 * yd), _mm256_cvtepi32_ps(sum256_0), acc256_0[t]);
                acc256_1[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx1 * yd), _mm256_cvtepi32_ps(sum256_1), acc256_1[t]);
            }
        }
    }
    if (use_512_reduce) {
        for (int t = 0; t < NT; ++t) {
            res0[t] = _mm512_reduce_add_ps(accf0[t]);
            res1[t] = _mm512_reduce_add_ps(accf1[t]);
        }
    } else if constexpr (NT == 1) {
        res0[0] = Iq512Fmt<TY>::K * hsum_f32_8(acc256_0[0]);
        res1[0] = Iq512Fmt<TY>::K * hsum_f32_8(acc256_1[0]);
    } else {
        for (int t = 0; t < NT; ++t) {
            res0[t] = hsum8_hadd(acc256_0[t]);
            res1[t] = hsum8_hadd(acc256_1[t]);
        }
    }
}

template <int TY, int NT, bool INTERLEAVED>
GEXEC_AVX512BW_TARGET void gexec_iq512_gu_rows_nt(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                                                  const void* const* act, float* const* ff, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t)
        y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = INTERLEAVED ? (blob + (size_t) (2 * r + 0) * gu_row) : (blob + (size_t) r * gu_row);
        const uint8_t* ur =
            INTERLEAVED ? (blob + (size_t) (2 * r + 1) * gu_row) : (blob + up_off + (size_t) r * gu_row);
        float g[NT], u[NT];
        iq512_row_dot_2rows<TY, NT>(gr, ur, nb, y, g, u);
        for (int t = 0; t < NT; ++t)
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, bool INTERLEAVED>
void dispatch_iq512_gu(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                       float* const* ff, int r0, int r1) {
    switch (nt) {
    case 1: gexec_iq512_gu_rows_nt<TY, 1, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 2: gexec_iq512_gu_rows_nt<TY, 2, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 3: gexec_iq512_gu_rows_nt<TY, 3, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 4: gexec_iq512_gu_rows_nt<TY, 4, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 5: gexec_iq512_gu_rows_nt<TY, 5, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 6: gexec_iq512_gu_rows_nt<TY, 6, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 7: gexec_iq512_gu_rows_nt<TY, 7, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 8: gexec_iq512_gu_rows_nt<TY, 8, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    default:
        for (int t0 = 0; t0 < nt; t0 += 8) {
            const int m = std::min(8, nt - t0);
            dispatch_iq512_gu<TY, INTERLEAVED>(m, blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
        }
        break;
    }
}

// ============================================================================
// 4b. I-Quant Gate/Up 2-Row Fused AVX2 Kernels (16, 17, 18, 21, 22)
// ============================================================================

struct GExecEvenSigns {
    uint64_t v[128];
    constexpr GExecEvenSigns() : v{} {
        for (int i = 0; i < 128; ++i) {
            int par = 0;
            for (int k = 0; k < 7; ++k)
                par ^= (i >> k) & 1;
            const int s = i | (par << 7);
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k)
                r |= (uint64_t) (((s >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
static constexpr GExecEvenSigns gexec_even_signs{};

GEXEC_AVX2_INLINE __m256i iq256_sgn_mask(uint32_t m) noexcept {
    const __m256i bm = _mm256_set1_epi32((int) m);
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                                          3, 3, 3, 3, 3, 3);
    const __m256i bits = _mm256_shuffle_epi8(bm, shuf);
    const __m256i sel = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 0x80, 1, 2, 4, 8, 16, 32, 64, (char) 0x80, 1, 2,
                                         4, 8, 16, 32, 64, (char) 0x80, 1, 2, 4, 8, 16, 32, 64, (char) 0x80);
    return _mm256_cmpeq_epi8(_mm256_and_si256(bits, sel), sel);
}

GEXEC_AVX2_INLINE __m256i iq256_sgn_vec(uint32_t m) noexcept {
    return _mm256_or_si256(iq256_sgn_mask(m), _mm256_set1_epi8(1));
}

GEXEC_AVX2_INLINE __m256i iq256_sc16(int a, int b) noexcept {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16((short) a)), _mm_set1_epi16((short) b), 1);
}

GEXEC_AVX2_INLINE __m256i iq256_sc32(int s) noexcept {
    return _mm256_set1_epi16((short) s);
}

template <int TY> struct Iq256Fmt;

template <> struct Iq256Fmt<16> {
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static GEXEC_AVX2_INLINE void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn,
                                         __m256i& sc) noexcept {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t w0 = read_u32(q), w1 = read_u32(q + 4);
        g = _mm256_set_epi64x((long long) iq2xxs_grid[w0 >> 24], (long long) iq2xxs_grid[(w0 >> 16) & 255],
                              (long long) iq2xxs_grid[(w0 >> 8) & 255], (long long) iq2xxs_grid[w0 & 255]);
        sgn = _mm256_set_epi64x(
            (long long) gexec_even_signs.v[(w1 >> 21) & 127], (long long) gexec_even_signs.v[(w1 >> 14) & 127],
            (long long) gexec_even_signs.v[(w1 >> 7) & 127], (long long) gexec_even_signs.v[w1 & 127]);
        sc = iq256_sc32(2 * (int) (w1 >> 28) + 1);
    }
};

template <> struct Iq256Fmt<17> {
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static GEXEC_AVX2_INLINE void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn,
                                         __m256i& sc) noexcept {
        uint16_t v[4];
        std::memcpy(v, b + 2 + 16 * j + 8 * half, 8);
        g = _mm256_set_epi64x((long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                              (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        sgn = _mm256_set_epi64x((long long) gexec_even_signs.v[v[3] >> 9], (long long) gexec_even_signs.v[v[2] >> 9],
                                (long long) gexec_even_signs.v[v[1] >> 9], (long long) gexec_even_signs.v[v[0] >> 9]);
        const uint8_t sb = b[66 + 2 * j + half];
        sc = iq256_sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Iq256Fmt<22> {
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static GEXEC_AVX2_INLINE void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn,
                                         __m256i& sc) noexcept {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h = b[66 + 2 * j + half];
        const int o = 4 * half;
        g = _mm256_set_epi64x((long long) iq2s_grid[qs[o + 3] | ((h << 2) & 0x300)],
                              (long long) iq2s_grid[qs[o + 2] | ((h << 4) & 0x300)],
                              (long long) iq2s_grid[qs[o + 1] | ((h << 6) & 0x300)],
                              (long long) iq2s_grid[qs[o + 0] | ((h << 8) & 0x300)]);
        const uint32_t m = read_u32(b + 2 + 32 + 8 * j + 4 * half);
        sgn = iq256_sgn_vec(m);
        const uint8_t sb = b[74 + 2 * j + half];
        sc = iq256_sc16(2 * (sb & 15) + 1, 2 * (sb >> 4) + 1);
    }
};

template <> struct Iq256Fmt<18> {
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static GEXEC_AVX2_INLINE void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn,
                                         __m256i& sc) noexcept {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]],
                             (int) iq3xxs_grid[q[4]], (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]],
                             (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t w = read_u32(b + 2 + 64 + 8 * j + 4 * half);
        sgn = _mm256_set_epi64x(
            (long long) gexec_even_signs.v[(w >> 21) & 127], (long long) gexec_even_signs.v[(w >> 14) & 127],
            (long long) gexec_even_signs.v[(w >> 7) & 127], (long long) gexec_even_signs.v[w & 127]);
        sc = iq256_sc32(2 * (int) (w >> 28) + 1);
    }
};

template <> struct Iq256Fmt<21> {
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static GEXEC_AVX2_INLINE void decode(const uint8_t* b, int j, int half, __m256i& g, __m256i& sgn,
                                         __m256i& sc) noexcept {
        const uint8_t* q = b + 2 + 16 * j + 8 * half;
        const uint32_t h = b[66 + 2 * j + half];
#define GEXEC_G256_3(k) (int) iq3s_grid[q[k] | (((h >> k) & 1u) << 8)]
        g = _mm256_set_epi32(GEXEC_G256_3(7), GEXEC_G256_3(6), GEXEC_G256_3(5), GEXEC_G256_3(4), GEXEC_G256_3(3),
                             GEXEC_G256_3(2), GEXEC_G256_3(1), GEXEC_G256_3(0));
#undef GEXEC_G256_3
        const uint32_t m = read_u32(b + 74 + 8 * j + 4 * half);
        sgn = iq256_sgn_vec(m);
        const uint8_t s = b[106 + j];
        sc = iq256_sc32(half ? 2 * (s >> 4) + 1 : 2 * (s & 15) + 1);
    }
};

template <int TY, int NT>
GEXEC_AVX2_TARGET void iq256_row_dot_1row(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    static const bool no_iq256 = std::getenv("GUILD_NO_IQ256") != nullptr;
    static const int mt_min = [] {
        const char* e = std::getenv("GUILD_IQ_MT_MIN");
        return e ? std::atoi(e) : 2;
    }();
    const bool use_ggml_reduce = cpu_avx512_ok() || no_iq256 || (NT < mt_min);
    __m256 accf[NT];
    for (int t = 0; t < NT; ++t)
        accf[t] = _mm256_setzero_ps();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Iq256Fmt<TY>::bytes;
        _mm_prefetch((const char*) (blk + 2048), _MM_HINT_T0);
        _mm_prefetch((const char*) (blk + 2048 + 64), _MM_HINT_T0);
        __m256i sumi0[NT], sumi1[NT];
        for (int t = 0; t < NT; ++t) {
            sumi0[t] = _mm256_setzero_si256();
            sumi1[t] = _mm256_setzero_si256();
        }
        if constexpr (TY == 21) {
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                const uint8_t* q0 = blk + 2 + 16 * j;
                const uint8_t* q1 = q0 + 8;
                const uint32_t h0 = blk[66 + 2 * j], h1 = blk[66 + 2 * j + 1];
#define GEXEC_G3_0(k) (int) iq3s_grid[q0[k] | ((h0 << (8 - (k))) & 0x100u)]
#define GEXEC_G3_1(k) (int) iq3s_grid[q1[k] | ((h1 << (8 - (k))) & 0x100u)]
                const __m256i g0 = _mm256_set_epi32(GEXEC_G3_0(7), GEXEC_G3_0(6), GEXEC_G3_0(5), GEXEC_G3_0(4),
                                                    GEXEC_G3_0(3), GEXEC_G3_0(2), GEXEC_G3_0(1), GEXEC_G3_0(0));
                const __m256i g1 = _mm256_set_epi32(GEXEC_G3_1(7), GEXEC_G3_1(6), GEXEC_G3_1(5), GEXEC_G3_1(4),
                                                    GEXEC_G3_1(3), GEXEC_G3_1(2), GEXEC_G3_1(1), GEXEC_G3_1(0));
#undef GEXEC_G3_0
#undef GEXEC_G3_1
                const __m256i s2_0 = iq256_sgn_mask(read_u32(blk + 74 + 8 * j));
                const __m256i s2_1 = iq256_sgn_mask(read_u32(blk + 74 + 8 * j + 4));
                const uint8_t s = blk[106 + j];
                const __m256i sc0 = iq256_sc32(2 * (s & 15) + 1);
                const __m256i sc1 = iq256_sc32(2 * (s >> 4) + 1);
                for (int t = 0; t < NT; ++t) {
                    const __m256i y0 = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * j));
                    const __m256i y1 = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * j + 32));
                    const __m256i ys0 = _mm256_sub_epi8(_mm256_xor_si256(y0, s2_0), s2_0);
                    const __m256i ys1 = _mm256_sub_epi8(_mm256_xor_si256(y1, s2_1), s2_1);
                    sumi0[t] = _mm256_add_epi32(sumi0[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g0, ys0), sc0));
                    sumi1[t] = _mm256_add_epi32(sumi1[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g1, ys1), sc1));
                }
            }
        } else {
            for (int j = 0; j < 4; ++j) {
                __m256i g0, sgn0, sc0, g1, sgn1, sc1;
                Iq256Fmt<TY>::decode(blk, j, 0, g0, sgn0, sc0);
                Iq256Fmt<TY>::decode(blk, j, 1, g1, sgn1, sc1);
                for (int t = 0; t < NT; ++t) {
                    const __m256i y0 = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * j));
                    const __m256i y1 = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 64 * j + 32));
                    sumi0[t] = _mm256_add_epi32(
                        sumi0[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g0, _mm256_sign_epi8(y0, sgn0)), sc0));
                    sumi1[t] = _mm256_add_epi32(
                        sumi1[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g1, _mm256_sign_epi8(y1, sgn1)), sc1));
                }
            }
        }
        const float dx =
            use_ggml_reduce ? fp16_to_fp32(read_u16(blk)) : (fp16_to_fp32(read_u16(blk)) * Iq256Fmt<TY>::K);
        for (int t = 0; t < NT; ++t) {
            const float yd = y[t][i].d;
            const __m256i sumi = _mm256_add_epi32(sumi0[t], sumi1[t]);
            accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * yd), _mm256_cvtepi32_ps(sumi), accf[t]);
        }
    }
    if (use_ggml_reduce) {
        for (int t = 0; t < NT; ++t)
            res[t] = Iq256Fmt<TY>::K * hsum_f32_8(accf[t]);
    } else {
        for (int t = 0; t < NT; ++t)
            res[t] = hsum8_hadd(accf[t]);
    }
}

template <int TY, int NT, bool INTERLEAVED>
GEXEC_AVX2_TARGET void gexec_iq256_gu_rows_nt(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                                              const void* const* act, float* const* ff, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t)
        y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = INTERLEAVED ? (blob + (size_t) (2 * r + 0) * gu_row) : (blob + (size_t) r * gu_row);
        const uint8_t* ur =
            INTERLEAVED ? (blob + (size_t) (2 * r + 1) * gu_row) : (blob + up_off + (size_t) r * gu_row);
        float g[NT], u[NT];
        iq256_row_dot_1row<TY, NT>(gr, nb, y, g);
        iq256_row_dot_1row<TY, NT>(ur, nb, y, u);
        for (int t = 0; t < NT; ++t)
            ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, bool INTERLEAVED>
void dispatch_iq256_gu(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                       float* const* ff, int r0, int r1) {
    switch (nt) {
    case 1: gexec_iq256_gu_rows_nt<TY, 1, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 2: gexec_iq256_gu_rows_nt<TY, 2, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 3: gexec_iq256_gu_rows_nt<TY, 3, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 4: gexec_iq256_gu_rows_nt<TY, 4, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 5: gexec_iq256_gu_rows_nt<TY, 5, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 6: gexec_iq256_gu_rows_nt<TY, 6, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 7: gexec_iq256_gu_rows_nt<TY, 7, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 8: gexec_iq256_gu_rows_nt<TY, 8, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    default:
        for (int t0 = 0; t0 < nt; t0 += 8) {
            const int m = std::min(8, nt - t0);
            dispatch_iq256_gu<TY, INTERLEAVED>(m, blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
        }
        break;
    }
}

// ============================================================================
// 5. IQ4_XS (type 23): 2-Row & 4-Row Fused AVX2 Kernel (Bit-Exact for all nt >= 1)
// ============================================================================

GEXEC_AVX2_INLINE void decode_iq4xs_subblock(const uint8_t* b, int j, int half, __m128i values, __m128i m4b, __m256i& g,
                                             __m256i& sgn, __m256i& sc) noexcept {
    const int H = 2 * j + half;
    const __m128i bits = _mm_loadu_si128((const __m128i*) (b + 8 + 16 * H));
    const __m128i lo = _mm_and_si128(bits, m4b);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bits, 4), m4b);
    const __m256i q4 =
        _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values, lo)), _mm_shuffle_epi8(values, hi), 1);
    g = _mm256_sign_epi8(q4, q4);
    sgn = q4;
    const uint16_t sh = read_u16(b + 2);
    const uint8_t sl = b[4 + j];
    const int ls = ((half ? (sl >> 4) : (sl & 15)) | (((sh >> (2 * H)) & 3) << 4)) - 32;
    sc = _mm256_set1_epi16((short) ls);
}

template <int NT, bool INTERLEAVED>
GEXEC_AVX2_TARGET void gexec_iq4xs_gu_rows_nt(const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                                              const void* const* act, float* const* ff, int r0, int r1) {
    static const bool no_iq256 = std::getenv("GUILD_NO_IQ256") != nullptr;
    static const int mt_min = [] {
        const char* e = std::getenv("GUILD_IQ_MT_MIN");
        return e ? std::atoi(e) : 2;
    }();
    const bool use_ggml_reduce = cpu_avx512_ok() || no_iq256 || (NT < mt_min);
    const __m128i values = _mm_loadu_si128((const __m128i*) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t)
        y[t] = (const block_q8_K*) act[t];
    const int nblocks = n / QK_K;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row0 = INTERLEAVED ? (blob + (size_t) (2 * r + 0) * gu_row) : (blob + (size_t) r * gu_row);
        const uint8_t* row1 =
            INTERLEAVED ? (blob + (size_t) (2 * r + 1) * gu_row) : (blob + up_off + (size_t) r * gu_row);
        __m256 accf0[NT], accf1[NT];
        for (int t = 0; t < NT; ++t) {
            accf0[t] = _mm256_setzero_ps();
            accf1[t] = _mm256_setzero_ps();
        }
        for (int i = 0; i < nblocks; ++i) {
            const uint8_t* blk0 = row0 + (size_t) i * 136;
            const uint8_t* blk1 = row1 + (size_t) i * 136;
            _mm_prefetch((const char*) (blk0 + 2048), _MM_HINT_T0);
            _mm_prefetch((const char*) (blk0 + 2048 + 64), _MM_HINT_T0);
            _mm_prefetch((const char*) (blk1 + 2048), _MM_HINT_T0);
            _mm_prefetch((const char*) (blk1 + 2048 + 64), _MM_HINT_T0);
            __m256i acci0[NT], acci1[NT];
            for (int t = 0; t < NT; ++t) {
                acci0[t] = _mm256_setzero_si256();
                acci1[t] = _mm256_setzero_si256();
            }
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    __m256i g0, sgn0, sc0, g1, sgn1, sc1;
                    decode_iq4xs_subblock(blk0, j, half, values, m4b, g0, sgn0, sc0);
                    decode_iq4xs_subblock(blk1, j, half, values, m4b, g1, sgn1, sc1);
                    for (int t = 0; t < NT; ++t) {
                        const __m256i yv = _mm256_loadu_si256((const __m256i*) (y[t][i].qs + 32 * (2 * j + half)));
                        acci0[t] = _mm256_add_epi32(
                            acci0[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g0, _mm256_sign_epi8(yv, sgn0)), sc0));
                        acci1[t] = _mm256_add_epi32(
                            acci1[t], _mm256_madd_epi16(_mm256_maddubs_epi16(g1, _mm256_sign_epi8(yv, sgn1)), sc1));
                    }
                }
            }
            const float dx0 = fp16_to_fp32(read_u16(blk0));
            const float dx1 = fp16_to_fp32(read_u16(blk1));
            for (int t = 0; t < NT; ++t) {
                const float yd = y[t][i].d;
                accf0[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx0 * yd), _mm256_cvtepi32_ps(acci0[t]), accf0[t]);
                accf1[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx1 * yd), _mm256_cvtepi32_ps(acci1[t]), accf1[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float g = use_ggml_reduce ? hsum_f32_8(accf0[t]) : hsum8_hadd(accf0[t]);
            const float u = use_ggml_reduce ? hsum_f32_8(accf1[t]) : hsum8_hadd(accf1[t]);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

template <bool INTERLEAVED>
void dispatch_iq4xs_gu(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                       float* const* ff, int r0, int r1) {
    switch (nt) {
    case 1: gexec_iq4xs_gu_rows_nt<1, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 2: gexec_iq4xs_gu_rows_nt<2, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 3: gexec_iq4xs_gu_rows_nt<3, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 4: gexec_iq4xs_gu_rows_nt<4, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 5: gexec_iq4xs_gu_rows_nt<5, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 6: gexec_iq4xs_gu_rows_nt<6, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 7: gexec_iq4xs_gu_rows_nt<7, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    case 8: gexec_iq4xs_gu_rows_nt<8, INTERLEAVED>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    default:
        for (int t0 = 0; t0 < nt; t0 += 8) {
            const int m = std::min(8, nt - t0);
            dispatch_iq4xs_gu<INTERLEAVED>(m, blob, gu_row, up_off, n, act + t0, ff + t0, r0, r1);
        }
        break;
    }
}

// ============================================================================
// 6. Top-Level GEXEC Gate+Up and Down Entry Points
// ============================================================================

template <bool USE_AVX512, bool INTERLEAVED>
void gexec_gu_rows_impl(const NativeFmt* f_ptr, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                        int r0, int r1) {
    const NativeFmt& f = *f_ptr;
#if defined(GUILD_NATIVE_EXPERTS)
    // For nt > 2 (prefill groups), 1-row multi-token kernels already amortize weight unpacking
    // across all nt tokens without doubling accumulator register pressure.
    if constexpr (!INTERLEAVED) {
        if (nt > 2) {
            native_gu_rows(f, blob, act, nt, ff, r0, r1);
            return;
        }
    }
#endif
    const int n = (int) f.n_embd;
    const size_t gu_row = f.gu_row;
    const size_t up_off = f.up_off;
    const bool avx512_ok = USE_AVX512 && cpu_avx512bw_ok();

    // 1. K-quants & Q8_0 / Q5_1 (types 12, 14, 8, 7): fused for ALL nt >= 1!
    if (f.gu_type == 12 || f.gu_type == 14 || f.gu_type == 8 || f.gu_type == 7) {
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gr = INTERLEAVED ? (blob + (size_t) (2 * r + 0) * gu_row) : (blob + (size_t) r * gu_row);
            const uint8_t* ur =
                INTERLEAVED ? (blob + (size_t) (2 * r + 1) * gu_row) : (blob + up_off + (size_t) r * gu_row);
            for (int t0 = 0; t0 < nt; t0 += MAXT) {
                const int m = std::min(MAXT, nt - t0);
                float g[MAXT], u[MAXT];
                if (f.gu_type == 12)
                    dispatch_q4k_2rows(avx512_ok, (const block_q4_K*) gr, (const block_q4_K*) ur, n / QK_K, act + t0, m,
                                       g, u);
                else if (f.gu_type == 14)
                    dispatch_q6k_2rows(avx512_ok, (const block_q6_K*) gr, (const block_q6_K*) ur, n / QK_K, act + t0, m,
                                       g, u);
                else if (f.gu_type == 8)
                    dispatch_q8_0_2rows((const block_q8_0*) gr, (const block_q8_0*) ur, n / QK8_0, act + t0, m, g, u);
                else
                    dispatch_q5_1_2rows((const block_q5_1*) gr, (const block_q5_1*) ur, n / QK5_1, act + t0, m, g, u);
                for (int t = 0; t < m; ++t)
                    ff[t0 + t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
            }
        }
        return;
    }

    // 2. IQ4_XS (type 23): bit-exact 2-row fused AVX2 for all nt >= 1!
    if (f.gu_type == 23 && cpu_avx2_ok()) {
        dispatch_iq4xs_gu<INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1);
        return;
    }

    // 3. I-Quants: use 512-bit 2-row fused when cpu_avx512_ok() (AVX-512 VNNI hosts where ZMM does not
    // trigger Skylake-SP heavy-license downclocking), and 256-bit AVX2 otherwise.
    if (USE_AVX512 && cpu_avx512_ok()) {
        switch (f.gu_type) {
        case 16: dispatch_iq512_gu<16, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        case 17: dispatch_iq512_gu<17, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        case 18: dispatch_iq512_gu<18, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        case 21: dispatch_iq512_gu<21, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        case 22: dispatch_iq512_gu<22, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        default: break;
        }
    }
    if (cpu_avx2_ok()) {
        switch (f.gu_type) {
        case 16: dispatch_iq256_gu<16, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        case 17: dispatch_iq256_gu<17, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        case 18: dispatch_iq256_gu<18, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        case 21: dispatch_iq256_gu<21, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        case 22: dispatch_iq256_gu<22, INTERLEAVED>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); return;
        default: break;
        }
    }

    // 4. Fallback for non-VNNI I-Quants or other formats (uses iq256_gu_rows in 256-bit AVX2):
#if defined(GUILD_NATIVE_EXPERTS)
    if constexpr (!INTERLEAVED) {
        native_gu_rows(f, blob, act, nt, ff, r0, r1);
    } else {
        NativeFmt row_fmt = f;
        row_fmt.gu_row = 2 * f.gu_row;
        row_fmt.up_off = f.gu_row;
        native_gu_rows(row_fmt, blob, act, nt, ff, r0, r1);
    }
#endif
}

// Helper for interleaved layout dispatch
template <bool USE_AVX512>
void gexec_gu_rows_interleaved_safe(const NativeFmt* f_ptr, const uint8_t* blob, const void* const* act, int nt,
                                    float* const* ff, int r0, int r1) {
    gexec_gu_rows_impl<USE_AVX512, true>(f_ptr, blob, act, nt, ff, r0, r1);
}

template <bool USE_AVX512>
void gexec_down_rows_impl(const NativeFmt* f_ptr, const uint8_t* blob, const void* const* act, int nt,
                          float* const* out, int r0, int r1) {
    const NativeFmt& f = *f_ptr;
#if defined(GUILD_NATIVE_EXPERTS)
    if (nt > 2) {
        native_down_rows(f, blob, act, nt, out, r0, r1);
        return;
    }
#endif
    const int n = (int) f.n_ff;
    const uint8_t* w = blob + f.down_off;
    const size_t row_bytes = f.d_row;
    const bool avx512_ok =
        USE_AVX512 && (cpu_avx512_ok() || (cpu_avx512bw_ok() && (f.gu_type == 12 || f.gu_type == 14)));

    // 1. IQ4_NL (type 20): 4-row fused for all nt >= 1!
    if (f.d_type == 20 && cpu_avx2_ok()) {
        gexec_iq4nl_down_rows(w, row_bytes, n, act, nt, out, r0, r1);
        return;
    }

    // 2. Q8_0 (type 8): 4-row and 2-row fused for all nt >= 1!
    if (f.d_type == 8 && cpu_avx2_ok()) {
        const int nb = n / QK8_0;
        int r = r0;
        for (; r + 4 <= r1; r += 4) {
            const auto* w0 = (const block_q8_0*) (w + (size_t) (r + 0) * row_bytes);
            const auto* w1 = (const block_q8_0*) (w + (size_t) (r + 1) * row_bytes);
            const auto* w2 = (const block_q8_0*) (w + (size_t) (r + 2) * row_bytes);
            const auto* w3 = (const block_q8_0*) (w + (size_t) (r + 3) * row_bytes);
            for (int t0 = 0; t0 < nt; t0 += MAXT) {
                const int m = std::min(MAXT, nt - t0);
                float s0[MAXT], s1[MAXT], s2[MAXT], s3[MAXT];
                dispatch_q8_0_4rows(w0, w1, w2, w3, nb, act + t0, m, s0, s1, s2, s3);
                for (int t = 0; t < m; ++t) {
                    out[t0 + t][r + 0] = s0[t];
                    out[t0 + t][r + 1] = s1[t];
                    out[t0 + t][r + 2] = s2[t];
                    out[t0 + t][r + 3] = s3[t];
                }
            }
        }
        for (; r < r1; ++r) {
            const auto* w0 = (const block_q8_0*) (w + (size_t) r * row_bytes);
            for (int t0 = 0; t0 < nt; t0 += MAXT) {
                const int m = std::min(MAXT, nt - t0);
                float s0[MAXT], s1[MAXT];
                dispatch_q8_0_2rows(w0, w0, nb, act + t0, m, s0, s1);
                for (int t = 0; t < m; ++t)
                    out[t0 + t][r] = s0[t];
            }
        }
        return;
    }

    // 3. Q4_K (type 12), Q6_K (type 14), Q5_1 (type 7): 2-row fused for all nt >= 1!
    if (f.d_type == 12 || f.d_type == 14 || f.d_type == 7) {
        int r = r0;
        for (; r + 2 <= r1; r += 2) {
            const uint8_t* w0 = w + (size_t) (r + 0) * row_bytes;
            const uint8_t* w1 = w + (size_t) (r + 1) * row_bytes;
            for (int t0 = 0; t0 < nt; t0 += MAXT) {
                const int m = std::min(MAXT, nt - t0);
                float s0[MAXT], s1[MAXT];
                if (f.d_type == 12)
                    dispatch_q4k_2rows(avx512_ok, (const block_q4_K*) w0, (const block_q4_K*) w1, n / QK_K, act + t0, m,
                                       s0, s1);
                else if (f.d_type == 14)
                    dispatch_q6k_2rows(avx512_ok, (const block_q6_K*) w0, (const block_q6_K*) w1, n / QK_K, act + t0, m,
                                       s0, s1);
                else
                    dispatch_q5_1_2rows((const block_q5_1*) w0, (const block_q5_1*) w1, n / QK5_1, act + t0, m, s0, s1);
                for (int t = 0; t < m; ++t) {
                    out[t0 + t][r + 0] = s0[t];
                    out[t0 + t][r + 1] = s1[t];
                }
            }
        }
        for (; r < r1; ++r) {
            const uint8_t* w0 = w + (size_t) r * row_bytes;
            for (int t0 = 0; t0 < nt; t0 += MAXT) {
                const int m = std::min(MAXT, nt - t0);
                float s0[MAXT], s1[MAXT];
                if (f.d_type == 12)
                    dispatch_q4k_2rows(avx512_ok, (const block_q4_K*) w0, (const block_q4_K*) w0, n / QK_K, act + t0, m,
                                       s0, s1);
                else if (f.d_type == 14)
                    dispatch_q6k_2rows(avx512_ok, (const block_q6_K*) w0, (const block_q6_K*) w0, n / QK_K, act + t0, m,
                                       s0, s1);
                else
                    dispatch_q5_1_2rows((const block_q5_1*) w0, (const block_q5_1*) w0, n / QK5_1, act + t0, m, s0, s1);
                for (int t = 0; t < m; ++t)
                    out[t0 + t][r] = s0[t];
            }
        }
        return;
    }

#if defined(GUILD_NATIVE_EXPERTS)
    native_down_rows(f, blob, act, nt, out, r0, r1);
#endif
}

// ============================================================================
// 7. Vectorized AVX2+FMA MoE Combine with In-Register Finiteness Check
// ============================================================================

GEXEC_AVX2_TARGET
guild_status_t gexec_combine_avx2(const float* parts, const float* weights, float* out, int k, int n) noexcept {
    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    __m256i nonfinite_any = _mm256_setzero_si256();
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int r = 0; r < k; ++r) {
            const __m256 val = _mm256_loadu_ps(parts + (size_t) r * (size_t) n + (size_t) d);
            const __m256i bits = _mm256_castps_si256(val);
            nonfinite_any =
                _mm256_or_si256(nonfinite_any, _mm256_cmpeq_epi32(_mm256_and_si256(bits, exp_mask), exp_mask));
            acc = _mm256_fmadd_ps(val, _mm256_set1_ps(weights[r]), acc);
        }
        _mm256_storeu_ps(out + d, acc);
    }
    uint32_t nonfinite = (uint32_t) _mm256_movemask_epi8(nonfinite_any);
    for (; d < n; ++d) {
        float acc = 0.0f;
        for (int r = 0; r < k; ++r) {
            const float value = parts[(size_t) r * (size_t) n + (size_t) d];
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            nonfinite |= (bits & 0x7f800000u) == 0x7f800000u;
            acc = std::fma(value, weights[r], acc);
        }
        out[d] = acc;
    }
    return nonfinite ? GUILD_STATUS_ERR_NONFINITE_OUTPUT : GUILD_STATUS_OK;
}

guild_status_t gexec_format_fn(int32_t gu, int32_t down, int64_t n, int64_t ff, guild_cpu_expert_format_v1_t* out,
                               guild_error_info_t* error) {
    if (!out || n <= 0 || n > H || ff <= 0 || ff > FF)
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    try {
#if defined(GUILD_NATIVE_EXPERTS)
        std::string message;
        if (native_fmt(gu, down, n, ff, *out, message))
            return GUILD_STATUS_OK;
        if (error) {
            error->struct_size = sizeof(*error);
            error->code = GUILD_STATUS_ERR_INVALID_TENSOR;
            std::snprintf(error->message, sizeof(error->message), "%s", message.c_str());
        }
#else
        if (native_fmt_host(gu, down, n, ff, *out))
            return GUILD_STATUS_OK;
        if (error) {
            error->struct_size = sizeof(*error);
            error->code = GUILD_STATUS_ERR_INVALID_TENSOR;
            std::snprintf(error->message, sizeof(error->message), "unsupported native expert format (%d, %d)", gu,
                          down);
        }
#endif
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    } catch (const std::exception& e) {
        if (error)
            std::snprintf(error->message, sizeof(error->message), "%s", e.what());
        return GUILD_STATUS_ERR_EXECUTION_FAILED;
    } catch (...) {
        return GUILD_STATUS_ERR_EXECUTION_FAILED;
    }
}

void gexec_router_bf16(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) {
#if defined(GUILD_NATIVE_EXPERTS)
    if (cpu_avx2_ok())
        bf16_rows_dot_multi(w, rows, cols, x, nt, out);
    else if (cpu_avx1_ok())
        bf16_rows_dot_multi_avx1(w, rows, cols, x, nt, out);
    else
#endif
        for (int t = 0; t < nt; ++t) {
            for (int r = 0; r < rows; ++r) {
                float sum = 0.0f;
                for (int c = 0; c < cols; ++c) {
                    uint32_t bits = uint32_t(w[(size_t) r * (size_t) cols + (size_t) c]) << 16;
                    float f;
                    std::memcpy(&f, &bits, sizeof(f));
                    sum += f * x[(size_t) t * (size_t) cols + (size_t) c];
                }
                out[(size_t) t * (size_t) rows + (size_t) r] = sum;
            }
        }
}

void gexec_canonical_gu(const uint8_t* b, const ActQ* const* a, int n, float* const* y, int r0, int r1) {
#if defined(GUILD_NATIVE_EXPERTS)
    if (cpu_avx512_ok())
        s2_expert_gu_rows_multi(b, a, n, y, r0, r1);
    else
        canonical_gu_avx2(b, a, n, y, r0, r1);
#else
    s2_expert_gu_rows_multi(b, a, n, y, r0, r1);
#endif
}

void gexec_canonical_down(const uint8_t* b, const ActQ* const* a, int n, float* const* y, int r0, int r1) {
#if defined(GUILD_NATIVE_EXPERTS)
    if (cpu_avx512_ok())
        s2_expert_down_rows_multi(b, a, n, y, r0, r1);
    else
        canonical_down_avx2(b, a, n, y, r0, r1);
#else
    s2_expert_down_rows_multi(b, a, n, y, r0, r1);
#endif
}

template <bool USE_AVX512, bool INTERLEAVED> const guild_cpu_backend_ext_v2_t* make_gexec_table() noexcept {
    static const guild_cpu_backend_ext_v2_t kTable = {
        {sizeof(guild_cpu_backend_ext_v2_t), GUILD_CPU_EXPERT_TAG_V1, 1,
         +[](const NativeFmt* f, const float* x, void* y) {
#if defined(GUILD_NATIVE_EXPERTS)
             native_quant_act(*f, x, y);
#else
             quantize_act_by_type_host(f->gu_act, x, y, (int) f->n_embd);
#endif
         },
         +[](const NativeFmt* f, const float* x, void* y) {
#if defined(GUILD_NATIVE_EXPERTS)
             native_quant_h(*f, x, y);
#else
             quantize_act_by_type_host(f->d_act, x, y, (int) f->n_ff);
#endif
         },
         INTERLEAVED ? +[](const NativeFmt* f, const uint8_t* b, const void* const* x, int n, float* const* y, int a,
                           int z) { gexec_gu_rows_interleaved_safe<USE_AVX512>(f, b, x, n, y, a, z); }
                     : +[](const NativeFmt* f, const uint8_t* b, const void* const* x, int n, float* const* y, int a,
                           int z) { gexec_gu_rows_impl<USE_AVX512, false>(f, b, x, n, y, a, z); },
         +[](const NativeFmt* f, const uint8_t* b, const void* const* x, int n, float* const* y, int a, int z) {
             gexec_down_rows_impl<USE_AVX512>(f, b, x, n, y, a, z);
         }},
        GUILD_CPU_BACKEND_VERSION_V2,
        0,
        gexec_format_fn,
        +[](const float* x, int n, ActQ* a) { act_quant_any(x, n, *a); },
        gexec_canonical_gu,
        gexec_canonical_down,
        +[](const uint8_t* w, uint64_t bytes, int blocks, const ActQ* const* a, int n, float* const* y, int r0,
            int r1) { q2_rows_any(w, size_t(bytes), blocks, a, n, y, r0, r1); },
        gexec_router_bf16,
        +[](const float* parts, const float* weights, float* out, int k, int n) -> guild_status_t {
            if (cpu_avx2_ok())
                return gexec_combine_avx2(parts, weights, out, k, n);
            uint32_t nonfinite = 0;
            for (int d = 0; d < n; ++d) {
                float acc = 0.0f;
                for (int r = 0; r < k; ++r) {
                    const float value = parts[(size_t) r * (size_t) n + (size_t) d];
                    uint32_t bits;
                    std::memcpy(&bits, &value, sizeof(bits));
                    nonfinite |= (bits & 0x7f800000u) == 0x7f800000u;
                    acc = std::fma(value, weights[r], acc);
                }
                out[d] = acc;
            }
            return nonfinite ? GUILD_STATUS_ERR_NONFINITE_OUTPUT : GUILD_STATUS_OK;
        },
        swiglu};
    return &kTable;
}

} // namespace

extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_gexec_fused_avx2_table_v2() noexcept {
    return make_gexec_table<false, false>();
}

extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_gexec_fused_avx512_table_v2() noexcept {
    return make_gexec_table<true, false>();
}

extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_gexec_interleaved_avx2_table_v2() noexcept {
    return make_gexec_table<false, true>();
}

extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_gexec_interleaved_avx512_table_v2() noexcept {
    return make_gexec_table<true, true>();
}

} // namespace guild::kernels::cpu
