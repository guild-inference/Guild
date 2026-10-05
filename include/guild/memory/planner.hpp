#pragma once

#include "guild/hardware/hardware_info.hpp"
#include "guild/memory/expert_budget.hpp"
#include "guild/memory/kv_budget.hpp"
#include "guild/memory/plan.hpp"
#include "guild/model/model_descriptor.hpp"

#include <cstdint>
#include <optional>

namespace guild::memory {

struct PlannerOptions {
    std::optional<int64_t> context_length;
    std::optional<KvPrecision> kv_format;
    std::optional<KvMode> kv_mode;
    std::optional<int64_t> resident_budget_gib;
    std::optional<int64_t> gpu_cache_slots;
    std::optional<int64_t> prefill_chunk;
    std::optional<int> spec_tokens;
    bool enable_mtp = true;
};

class MemoryPlanner {
public:
    static ExecutionPlan plan(
        const model::ModelDescriptor& desc,
        const hardware::HardwareInfo& hw,
        const PlannerOptions& options = {});

    static PlanValidation validate(
        const ExecutionPlan& plan,
        const hardware::HardwareInfo& hw,
        const model::ModelDescriptor& desc);

    static uint64_t estimate_dense_gpu_bytes(const model::ModelDescriptor& desc);
    static uint64_t calculate_prompt_scratch_bytes(const model::ModelDescriptor& desc, int64_t prefill_chunk);
    static uint64_t calculate_safety_reserve_bytes(const hardware::HardwareInfo& hw);
};

}  // namespace guild::memory
