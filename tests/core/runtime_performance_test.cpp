#include "../check.hpp"
#include "guild/runtime/performance.hpp"
#include <limits>

class Source final : public guild::core::ExpertSource {
public:
    bool mapped = true;
    const uint8_t* blob(int64_t, int64_t) override {
        return nullptr;
    }
    bool pcie_layer(int64_t) const override {
        return mapped;
    }
};

int main() {
    using namespace guild;
    kernels::cpu::ExpertLayout layout;
    layout.native = true;
    layout.n_layers = 3;
    layout.n_expert = 4;
    layout.fmt.resize(3);
    std::string err;
    CHECK(kernels::cpu::native_fmt(8, 8, 256, 256, layout.fmt[0], err));
    CHECK(kernels::cpu::native_fmt(8, 8, 512, 256, layout.fmt[1], err));
    CHECK(kernels::cpu::native_fmt(1, 1, 256, 256, layout.fmt[2], err));
    layout.bytes = {layout.fmt[0].bytes, layout.fmt[1].bytes, layout.fmt[2].bytes};
    Source source;
    kernels::cpu::ExpertPool pool(1, false, true);
    runtime::ExpertPerformance performance;
    CHECK(runtime::calibrate_experts(layout, source, pool, 0.218, nullptr, performance, err));
    CHECK(performance.gpu_layers == std::vector<uint8_t>({1, 1, 0}));
    CHECK(performance.pcie_numerators == std::vector<int>({56, 56, 0}));
    // Fractions do not depend on expert count or a model family label.
    layout.n_expert = 512;
    CHECK(runtime::calibrate_experts(layout, source, pool, 0.218, nullptr, performance, err));
    CHECK(performance.pcie_numerators[0] == 56);
    layout.n_expert = 4;
    const std::vector<std::pair<int32_t, int32_t>> ranking = {{0, 0}, {2, 0}, {0, 0}, {1, 0}, {0, 1}, {8, 0}};
    const uint64_t small = (layout.bytes[0] + 255) & ~uint64_t(255);
    const auto cache = runtime::size_expert_cache(layout, ranking, performance.gpu_layers, 3 * small, 99);
    CHECK(cache.pairs == (std::vector<std::pair<int32_t, int32_t>>{{0, 0}, {1, 0}}));
    CHECK(cache.bytes <= 3 * small && cache.slot_bytes[0] != cache.slot_bytes[1]);
    CHECK(cache.pairs.size() > (3 * small) / layout.bytes[1]);
    CHECK(runtime::size_expert_cache(layout, ranking, performance.gpu_layers, UINT64_MAX, 1).pairs.size() == 1);
    CHECK(runtime::size_expert_cache(layout, ranking, performance.gpu_layers, small - 1, 99).pairs.empty());
    source.mapped = false;
    CHECK(runtime::calibrate_experts(layout, source, pool, -1, nullptr, performance, err));
    CHECK(performance.measured_formats == 0 && performance.pcie_numerators == std::vector<int>({0, 0, 0}));
    CHECK(!runtime::calibrate_experts(layout, source, pool, std::numeric_limits<double>::quiet_NaN(), nullptr,
                                      performance, err));
    CHECK(!runtime::calibrate_experts(layout, source, pool, -0.1, nullptr, performance, err));
    std::puts(
        "runtime_performance_test: format capabilities, variable cache bytes, limits and explicit fractions PASS");
}
