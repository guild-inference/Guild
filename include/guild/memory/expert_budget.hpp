#pragma once

#include "guild/model/model_descriptor.hpp"

#include <cstdint>
#include <optional>

namespace guild::memory {

struct ExpertTierDistribution {
    int64_t total_experts = 0;
    uint64_t total_bytes = 0;
    uint64_t bytes_per_expert = 0;

    int64_t gpu_cache_slots = 0;
    uint64_t gpu_cache_bytes = 0;

    int64_t ram_resident_slots = 0;
    uint64_t ram_resident_bytes = 0;

    int64_t file_slots = 0;
    uint64_t file_bytes = 0;
};

uint64_t calculate_routed_expert_footprint(const model::ModelDescriptor& desc);

ExpertTierDistribution calculate_expert_distribution(
    const model::ModelDescriptor& desc,
    uint64_t available_ram_bytes,
    uint64_t free_vram_for_cache,
    std::optional<uint64_t> explicit_ram_budget_bytes = std::nullopt,
    std::optional<int64_t> explicit_gpu_slots = std::nullopt);

}  // namespace guild::memory
