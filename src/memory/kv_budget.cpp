#include "guild/memory/kv_budget.hpp"

#include <algorithm>

namespace guild::memory {

int64_t get_kv_cell_bytes(KvPrecision prec, const model::ModelDescriptor& desc) {
    const int64_t kv_heads = desc.attn.n_kv_heads > 0 ? desc.attn.n_kv_heads : 2;
    const int64_t head_dim = desc.attn.head_dim > 0 ? desc.attn.head_dim : 256;

    switch (prec) {
    case KvPrecision::FP16:
        // 2 (K and V) * n_kv_heads * head_dim * 2 bytes
        return 2 * kv_heads * head_dim * 2;
    case KvPrecision::INT8: {
        // Grouped-64 INT8: 1 byte per element + 2 bytes FP16 scale per 64 elements
        const int64_t elems = 2 * kv_heads * head_dim;
        const int64_t scales = (elems / 64) * 2;
        return elems + scales;
    }
    case KvPrecision::Q4_0: {
        // Q4_0: two 144-byte rows per head for head_dim 256, or (head_dim / 2 + scales)
        if (head_dim == 256) {
            return kv_heads * 288;
        }
        const int64_t elems = 2 * kv_heads * head_dim;
        return (elems / 2) + (elems / 32) * 2;
    }
    case KvPrecision::K8V4: {
        // INT8 K half + Q4_0 V half
        const int64_t k_bytes = kv_heads * head_dim + (kv_heads * head_dim / 64) * 2;
        const int64_t v_bytes = (head_dim == 256) ? (kv_heads * 144) : (kv_heads * head_dim / 2 + (kv_heads * head_dim / 32) * 2);
        return k_bytes + v_bytes;
    }
    default:
        return 2 * kv_heads * head_dim * 2;
    }
}

int64_t get_kv_draft_cell_bytes(KvPrecision prec, const model::ModelDescriptor& desc) {
    if (prec == KvPrecision::K8V4) {
        // MTP uses plain INT8 under K8V4
        return get_kv_cell_bytes(KvPrecision::INT8, desc);
    }
    return get_kv_cell_bytes(prec, desc);
}

int64_t get_staging_cells(const model::ModelDescriptor& /*desc*/) {
    // 8 maximum verify rows, 2,051 selected cells per row, 4-cell pages, plus threshold/tail pages:
    // 8 * (2051 / 4 + 2) * 4 = 16,448 cells
    return 8 * (2051 / 4 + 2) * 4;
}

KvBudget calculate_kv_budget(
    const model::ModelDescriptor& desc,
    int64_t context_tokens,
    KvPrecision precision,
    KvMode mode,
    int64_t resident_window_tokens)
{
    KvBudget b;
    b.precision = precision;
    b.mode = mode;
    b.cell_bytes = get_kv_cell_bytes(precision, desc);
    b.draft_cell_bytes = get_kv_draft_cell_bytes(precision, desc);
    b.staging_cells = get_staging_cells(desc);

    const int64_t full_attn_layers = desc.attn.n_full_attn_layers() > 0 ? desc.attn.n_full_attn_layers() : 12;
    b.full_host_bytes = (uint64_t) context_tokens * (uint64_t) full_attn_layers * (uint64_t) b.cell_bytes;

    if (mode == KvMode::Resident) {
        b.persistent_gpu_bytes = b.full_host_bytes;
        b.staging_gpu_bytes = 0;
        b.pinned_host_bytes = 0;
    } else if (mode == KvMode::Streaming) {
        const int64_t window = resident_window_tokens > 0 ? resident_window_tokens : 32768;
        const int64_t gpu_tokens = std::min(context_tokens, window);
        b.persistent_gpu_bytes = (uint64_t) gpu_tokens * (uint64_t) full_attn_layers * (uint64_t) b.cell_bytes;
        b.staging_gpu_bytes = 0;
        b.pinned_host_bytes = b.full_host_bytes;
    } else { // HostOnly
        b.persistent_gpu_bytes = 0;
        b.staging_gpu_bytes = (uint64_t) b.staging_cells * (uint64_t) b.cell_bytes;
        b.pinned_host_bytes = b.full_host_bytes;
    }

    return b;
}

}  // namespace guild::memory
