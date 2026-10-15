#include "../check.hpp"
#include "guild/core/validation.hpp"
#include <limits>
#include <vector>

int main() {
    float logits[4] = {1, 9, -2, 3};
    std::string err;
    CHECK(guild::core::validate_finite(logits, 4, "logits", err));
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
        logits[2] = bad; // not the previously selected argmax
        CHECK(!guild::core::validate_finite(logits, 4, "logits", err));
        CHECK(err.find("nonfinite") != std::string::npos);
    }
    CHECK(!guild::core::validate_finite(nullptr, 4, "logits", err));
    std::vector<float> large(131, std::numeric_limits<float>::max());
    large[0] = -0.0f;
    large[63] = std::numeric_limits<float>::lowest();
    large[64] = std::numeric_limits<float>::denorm_min();
    CHECK(guild::core::validate_finite(large.data(), large.size(), "large", err));
    for (size_t index : {size_t(0), size_t(63), size_t(64), size_t(127), size_t(130)}) {
        const float previous = large[index];
        for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
            large[index] = bad;
            CHECK(!guild::core::validate_finite(large.data(), large.size(), "large", err));
            CHECK(err == "large: nonfinite value at element " + std::to_string(index));
        }
        large[index] = previous;
    }
    std::puts("finite_output_test: the entire distribution is checked, including unselected entries");
}
