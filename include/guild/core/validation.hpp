#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace guild::core {

// A finite argmax does not establish that its source distribution was finite.
// Check the entire output, including entries the sampler would not select.
inline bool validate_finite(const float* values, size_t count, const std::string& name, std::string& err) {
    if (!values || count == 0) {
        err = name + ": missing output";
        return false;
    }
    static_assert(sizeof(float) == sizeof(uint32_t), "validation requires IEEE binary32 floats");
    size_t i = 0;
#if defined(__AVX2__)
    const __m256i exp_mask = _mm256_set1_epi32(0x7f800000);
    for (; count - i >= 32; i += 32) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*) (values + i + 0));
        const __m256i v1 = _mm256_loadu_si256((const __m256i*) (values + i + 8));
        const __m256i v2 = _mm256_loadu_si256((const __m256i*) (values + i + 16));
        const __m256i v3 = _mm256_loadu_si256((const __m256i*) (values + i + 24));
        const __m256i b0 = _mm256_cmpeq_epi32(_mm256_and_si256(v0, exp_mask), exp_mask);
        const __m256i b1 = _mm256_cmpeq_epi32(_mm256_and_si256(v1, exp_mask), exp_mask);
        const __m256i b2 = _mm256_cmpeq_epi32(_mm256_and_si256(v2, exp_mask), exp_mask);
        const __m256i b3 = _mm256_cmpeq_epi32(_mm256_and_si256(v3, exp_mask), exp_mask);
        const __m256i bad = _mm256_or_si256(_mm256_or_si256(b0, b1), _mm256_or_si256(b2, b3));
        if (_mm256_movemask_epi8(bad) != 0) {
            for (size_t j = 0; j < 32; ++j) {
                uint32_t bits;
                std::memcpy(&bits, values + i + j, sizeof(bits));
                if ((bits & 0x7f800000u) == 0x7f800000u) {
                    err = name + ": nonfinite value at element " + std::to_string(i + j);
                    return false;
                }
            }
        }
    }
#endif
    // A reduction over fixed-size blocks can vectorize on the baseline CPU ISA;
    // an early-return isfinite loop scans millions of expert/logit values serially.
    // Check exponent bits so compiler floating-point assumptions cannot hide NaNs.
    for (; count - i >= 64; i += 64) {
        uint32_t nonfinite = 0;
        for (size_t j = 0; j < 64; ++j) {
            uint32_t bits;
            std::memcpy(&bits, values + i + j, sizeof(bits));
            nonfinite |= (bits & 0x7f800000u) == 0x7f800000u;
        }
        if (nonfinite) {
            for (size_t j = 0; j < 64; ++j) {
                uint32_t bits;
                std::memcpy(&bits, values + i + j, sizeof(bits));
                if ((bits & 0x7f800000u) == 0x7f800000u) {
                    err = name + ": nonfinite value at element " + std::to_string(i + j);
                    return false;
                }
            }
        }
    }
    for (; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, values + i, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            err = name + ": nonfinite value at element " + std::to_string(i);
            return false;
        }
    }
    return true;
}

} // namespace guild::core
