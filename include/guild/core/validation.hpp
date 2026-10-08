#pragma once

#include <cmath>
#include <cstddef>
#include <string>

namespace guild::core {

// A finite argmax does not establish that its source distribution was finite.
// Check the entire output, including entries the sampler would not select.
inline bool validate_finite(const float* values, size_t count, const std::string& name, std::string& err) {
    if (!values || count == 0) {
        err = name + ": missing output";
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            err = name + ": nonfinite value at element " + std::to_string(i);
            return false;
        }
    }
    return true;
}

} // namespace guild::core
