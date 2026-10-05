#pragma once

#include "guild/memory/plan.hpp"
#include "guild/model/model_descriptor.hpp"

#include <cstdint>

namespace guild::memory {

struct KvBudget {
    KvPrecision precision = KvPrecision::FP16;
    KvMode mode = KvMode::HostOnly;

    int64_t cell_bytes = 2048;
    int64_t draft_cell_bytes = 2048;
    int64_t staging_cells = 16448;

    uint64_t full_host_bytes = 0;
    uint64_t persistent_gpu_bytes = 0;
    uint64_t staging_gpu_bytes = 0;
    uint64_t pinned_host_bytes = 0;
};

int64_t get_kv_cell_bytes(KvPrecision prec, const model::ModelDescriptor& desc);
int64_t get_kv_draft_cell_bytes(KvPrecision prec, const model::ModelDescriptor& desc);
int64_t get_staging_cells(const model::ModelDescriptor& desc);

KvBudget calculate_kv_budget(
    const model::ModelDescriptor& desc,
    int64_t context_tokens,
    KvPrecision precision,
    KvMode mode,
    int64_t resident_window_tokens = 0);

}  // namespace guild::memory
