#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

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
