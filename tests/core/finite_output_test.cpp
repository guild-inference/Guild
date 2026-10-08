#include "../check.hpp"
#include "guild/core/validation.hpp"
#include <limits>

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
    std::puts("finite_output_test: the entire distribution is checked, including unselected entries");
}
