#pragma once

#include "guild/core/expert_source.hpp"
#include "guild/core/verify.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace guild::runtime {

struct ExpertPerformance {
    std::vector<uint8_t> gpu_layers;
    std::vector<int> pcie_numerators;
    double cpu_ms = 0;
    double gpu_ms = 0;
    double copy_ms = 0;
    int measured_formats = 0;
};

/// Probe supported formats with the actual CPU pool, GPU kernels and resident blobs.
/// No model-name dispatch. Explicit fractions skip timing and preserve their value.
bool calibrate_experts(const kernels::cpu::ExpertLayout& layout, core::ExpertSource& source,
                       kernels::cpu::ExpertPool& pool, double fraction, void* stream,
                       ExpertPerformance& result, std::string& err);

/// Refine the split on complete verification windows, without emitting tokens.
bool tune_expert_split(core::Verifier& verifier, core::SessionState& state, const core::ModelGeometry& geometry,
                       core::ExpertDispatch& dispatch, core::PoolMultiFn pool, void* user, int max_t,
                       void* stream, ExpertPerformance& performance, std::string& err);

struct CacheSizing {
    std::vector<std::pair<int32_t, int32_t>> pairs;
    std::vector<int64_t> slot_bytes;
    uint64_t bytes = 0;
};

/// Ranked, format-capable pairs under an actual byte budget. Explicit counts are
/// counts; auto mode fills a byte budget with each layer's real aligned blob size.
CacheSizing size_expert_cache(const kernels::cpu::ExpertLayout& layout,
                             const std::vector<std::pair<int32_t, int32_t>>& ranking,
                             const std::vector<uint8_t>& gpu_layers, uint64_t budget,
                             int64_t max_slots);

} // namespace guild::runtime
