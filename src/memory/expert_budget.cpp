#include "guild/memory/expert_budget.hpp"

#include <algorithm>

namespace guild::memory {

uint64_t calculate_routed_expert_footprint(const model::ModelDescriptor& desc) {
    int64_t total_experts = desc.moe.total_routed_experts(desc.attn.n_layers);
    if (total_experts <= 0) {
        if (desc.archetype == model::ModelArchetype::Qwen4Exp || desc.arch_name == "qwen4exp") {
            total_experts = 48 * 512;
        } else {
            total_experts = 1;
        }
    }

    if (desc.moe.expert_blob_bytes > 0) {
        return (uint64_t) total_experts * (uint64_t) desc.moe.expert_blob_bytes;
    }

    // Default expert blob sizing for known archetypes
    if (desc.archetype == model::ModelArchetype::Qwen4Exp || desc.arch_name == "qwen4exp") {
        // UD-IQ4_XS footprint: ~55.43 GiB (59,518,476,288 bytes across 24,576 experts = 2,421,813 bytes/expert)
        return (uint64_t) total_experts * 2421813ULL;
    }

    // Standard SwiGLU 4-bit approximation: 3 * ff * embd * ~0.55 bytes
    const int64_t ff = desc.moe.expert_dim_ff > 0 ? desc.moe.expert_dim_ff : 640;
    const int64_t embd = desc.attn.n_embd > 0 ? desc.attn.n_embd : 2560;
    const uint64_t elems = 3ULL * (uint64_t) ff * (uint64_t) embd;
    const uint64_t approx_bytes_per_expert = (elems * 55ULL) / 100ULL;
    return (uint64_t) total_experts * approx_bytes_per_expert;
}

ExpertTierDistribution calculate_expert_distribution(
    const model::ModelDescriptor& desc,
    uint64_t available_ram_bytes,
    uint64_t free_vram_for_cache,
    std::optional<uint64_t> explicit_ram_budget_bytes,
    std::optional<int64_t> explicit_gpu_slots)
{
    ExpertTierDistribution dist;
    dist.total_experts = desc.moe.total_routed_experts(desc.attn.n_layers);
    if (dist.total_experts <= 0) {
        dist.total_experts = 24576; // Qwen default
    }

    dist.total_bytes = calculate_routed_expert_footprint(desc);
    dist.bytes_per_expert = dist.total_bytes / (uint64_t) dist.total_experts;
    if (dist.bytes_per_expert == 0) dist.bytes_per_expert = 1;

    // 1. GPU Cache allocation
    if (explicit_gpu_slots.has_value()) {
        int64_t slots = explicit_gpu_slots.value();
        slots = std::max<int64_t>(0, std::min<int64_t>(slots, dist.total_experts));
        uint64_t max_fit = free_vram_for_cache / dist.bytes_per_expert;
        dist.gpu_cache_slots = std::min<int64_t>(slots, (int64_t) max_fit);
    } else {
        // Size GPU expert cache only from genuinely free VRAM
        if (free_vram_for_cache >= dist.bytes_per_expert) {
            uint64_t slots = free_vram_for_cache / dist.bytes_per_expert;
            dist.gpu_cache_slots = (int64_t) std::min<uint64_t>(slots, (uint64_t) dist.total_experts);
        } else {
            dist.gpu_cache_slots = 0;
        }
    }
    dist.gpu_cache_bytes = (uint64_t) dist.gpu_cache_slots * dist.bytes_per_expert;

    const int64_t remaining_experts = dist.total_experts - dist.gpu_cache_slots;

    // 2. RAM Tier allocation
    uint64_t ram_budget = explicit_ram_budget_bytes.has_value()
                              ? explicit_ram_budget_bytes.value()
                              : available_ram_bytes;

    uint64_t ram_slots = ram_budget / dist.bytes_per_expert;
    dist.ram_resident_slots = std::min<int64_t>((int64_t) ram_slots, remaining_experts);
    dist.ram_resident_bytes = (uint64_t) dist.ram_resident_slots * dist.bytes_per_expert;

    // 3. File Tier allocation
    dist.file_slots = remaining_experts - dist.ram_resident_slots;
    dist.file_bytes = dist.total_bytes - dist.gpu_cache_bytes - dist.ram_resident_bytes;

    return dist;
}

}  // namespace guild::memory
