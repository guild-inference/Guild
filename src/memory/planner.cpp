#include "guild/memory/planner.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace guild::memory {

std::string ExecutionPlan::to_human_string() const {
    std::ostringstream oss;
    char buf[128];

    oss << "Memory plan\n";
    std::snprintf(buf, sizeof(buf), "  Host RAM        %.0f GiB\n", (double) total_host_ram_bytes / (1024.0 * 1024.0 * 1024.0));
    oss << buf;
    std::snprintf(buf, sizeof(buf), "  GPU VRAM        %.1f GiB\n\n", (double) total_gpu_vram_bytes / (1024.0 * 1024.0 * 1024.0));
    oss << buf;

    oss << "  Experts\n";
    std::snprintf(buf, sizeof(buf), "    total         %.2f GiB\n", (double) routed_expert_total_bytes / (1024.0 * 1024.0 * 1024.0));
    oss << buf;
    if (routed_experts_in_gpu == 0) {
        oss << "    GPU           0\n";
    } else {
        std::snprintf(buf, sizeof(buf), "    GPU           %lld (%.2f GiB)\n", (long long) routed_experts_in_gpu,
                      (double) routed_experts_gpu_bytes / (1024.0 * 1024.0 * 1024.0));
        oss << buf;
    }
    if (routed_experts_in_ram == total_routed_experts) {
        std::snprintf(buf, sizeof(buf), "    RAM           %.2f GiB\n", (double) routed_experts_ram_bytes / (1024.0 * 1024.0 * 1024.0));
        oss << buf;
    } else {
        std::snprintf(buf, sizeof(buf), "    RAM           %lld / %lld (%.2f GiB)\n", (long long) routed_experts_in_ram,
                      (long long) total_routed_experts, (double) routed_experts_ram_bytes / (1024.0 * 1024.0 * 1024.0));
        oss << buf;
    }
    if (routed_experts_on_file == 0) {
        oss << "    file          0\n\n";
    } else {
        std::snprintf(buf, sizeof(buf), "    file          %lld (%.2f GiB)\n\n", (long long) routed_experts_on_file,
                      (double) routed_experts_file_bytes / (1024.0 * 1024.0 * 1024.0));
        oss << buf;
    }

    oss << "  KV\n";
    oss << "    format        " << kv_precision_to_string(kv_format) << "\n";
    oss << "    context       " << context_length << "\n";
    std::snprintf(buf, sizeof(buf), "    host          %.2f GiB\n", (double) full_host_kv_bytes / (1024.0 * 1024.0 * 1024.0));
    oss << buf;
    oss << "    GPU mode      " << kv_mode_to_string(kv_mode) << "\n";
    if (host_only_kv) {
        std::snprintf(buf, sizeof(buf), "    staging       %.1f MiB\n\n", (double) kv_staging_bytes / (1024.0 * 1024.0));
    } else {
        std::snprintf(buf, sizeof(buf), "    staging       -\n\n");
    }
    oss << buf;

    oss << "  Decode\n";
    if (mtp_spec_tokens > 0) {
        oss << "    MTP           spec " << mtp_spec_tokens << "\n";
    } else {
        oss << "    MTP           disabled\n";
    }
    oss << "    prefill       " << prefill_chunk << "\n";
    std::snprintf(buf, sizeof(buf), "    GPU reserve   %.0f MiB\n", (double) safety_reserve_bytes / (1024.0 * 1024.0));
    oss << buf;
    std::snprintf(buf, sizeof(buf), "    free VRAM     %.1f GiB\n", (double) remaining_vram_bytes / (1024.0 * 1024.0 * 1024.0));
    oss << buf;

    if (!warnings.empty()) {
        oss << "\n  Warnings:\n";
        for (const auto& w : warnings) {
            oss << "    - " << w << "\n";
        }
    }

    return oss.str();
}

std::string ExecutionPlan::to_json_string() const {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"context_length\": " << context_length << ",\n"
        << "  \"kv_format\": \"" << kv_precision_to_string(kv_format) << "\",\n"
        << "  \"kv_mode\": \"" << kv_mode_to_string(kv_mode) << "\",\n"
        << "  \"host_only_kv\": " << (host_only_kv ? "true" : "false") << ",\n"
        << "  \"full_host_kv_bytes\": " << full_host_kv_bytes << ",\n"
        << "  \"persistent_gpu_kv_bytes\": " << persistent_gpu_kv_bytes << ",\n"
        << "  \"kv_staging_bytes\": " << kv_staging_bytes << ",\n"
        << "  \"routed_expert_total_bytes\": " << routed_expert_total_bytes << ",\n"
        << "  \"total_routed_experts\": " << total_routed_experts << ",\n"
        << "  \"routed_experts_in_ram\": " << routed_experts_in_ram << ",\n"
        << "  \"routed_experts_ram_bytes\": " << routed_experts_ram_bytes << ",\n"
        << "  \"routed_experts_in_gpu\": " << routed_experts_in_gpu << ",\n"
        << "  \"routed_experts_gpu_bytes\": " << routed_experts_gpu_bytes << ",\n"
        << "  \"routed_experts_on_file\": " << routed_experts_on_file << ",\n"
        << "  \"routed_experts_file_bytes\": " << routed_experts_file_bytes << ",\n"
        << "  \"pinned_host_bytes\": " << pinned_host_bytes << ",\n"
        << "  \"dense_gpu_bytes\": " << dense_gpu_bytes << ",\n"
        << "  \"mtp_bytes\": " << mtp_bytes << ",\n"
        << "  \"mtp_spec_tokens\": " << mtp_spec_tokens << ",\n"
        << "  \"prompt_scratch_bytes\": " << prompt_scratch_bytes << ",\n"
        << "  \"safety_reserve_bytes\": " << safety_reserve_bytes << ",\n"
        << "  \"prefill_chunk\": " << prefill_chunk << ",\n"
        << "  \"remaining_host_ram_bytes\": " << remaining_host_ram_bytes << ",\n"
        << "  \"remaining_vram_bytes\": " << remaining_vram_bytes << ",\n"
        << "  \"total_host_ram_bytes\": " << total_host_ram_bytes << ",\n"
        << "  \"total_gpu_vram_bytes\": " << total_gpu_vram_bytes << ",\n"
        << "  \"warnings\": [";
    for (size_t i = 0; i < warnings.size(); ++i) {
        oss << "\n    \"" << warnings[i] << "\"";
        if (i + 1 < warnings.size()) oss << ",";
    }
    if (!warnings.empty()) oss << "\n  ";
    oss << "]\n}\n";
    return oss.str();
}

uint64_t MemoryPlanner::estimate_dense_gpu_bytes(const model::ModelDescriptor& desc) {
    if (desc.archetype == model::ModelArchetype::Qwen4Exp || desc.arch_name == "qwen4exp") {
        // Qwen3.8-Flash-Next dense weights (projections, recurrent conv, attention, embed, head, router, PLE)
        // Measured footprint: ~3.75 GiB (4,026,531,840 bytes)
        return 4026531840ULL;
    }

    const int64_t embd = desc.attn.n_embd > 0 ? desc.attn.n_embd : 2560;
    const int64_t vocab = desc.attn.vocab_size > 0 ? desc.attn.vocab_size : 151936;
    const int64_t layers = desc.attn.n_layers > 0 ? desc.attn.n_layers : 48;

    // Embedding & head in Q8_0 (1 byte per element + scales)
    const uint64_t token_embd_head = 2ULL * (uint64_t) vocab * (uint64_t) embd;
    // Layer dense projections in FP16 / 4-bit (approx 4 * embd * embd per layer)
    const uint64_t layer_dense = (uint64_t) layers * 4ULL * (uint64_t) embd * (uint64_t) embd / 2ULL;

    return token_embd_head + layer_dense;
}

uint64_t MemoryPlanner::calculate_prompt_scratch_bytes(const model::ModelDescriptor& desc, int64_t prefill_chunk) {
    if (prefill_chunk <= 0) prefill_chunk = 512;

    // Base MMQ workspaces, cuBLAS workspaces, and temporary quantization buffers
    const uint64_t base = 350ULL * 1024ULL * 1024ULL; // 350 MiB

    // Intermediate activations across layers for the chunk
    const int64_t embd = desc.attn.n_embd > 0 ? desc.attn.n_embd : 2560;
    const uint64_t per_tok = (uint64_t) embd * (uint64_t) sizeof(float) * 24ULL;
    const uint64_t chunk_scratch = ((uint64_t) prefill_chunk * per_tok) / 4ULL;

    // Attention matrix scratch for larger chunks
    uint64_t attn_matrix_scratch = 0;
    if (prefill_chunk > 512) {
        attn_matrix_scratch = (uint64_t) prefill_chunk * (uint64_t) prefill_chunk * 12ULL * sizeof(float);
    }

    return base + chunk_scratch + attn_matrix_scratch;
}

uint64_t MemoryPlanner::calculate_safety_reserve_bytes(const hardware::HardwareInfo& hw) {
    if (hw.gpu_vram_bytes <= 8589934592ULL) { // <= 8 GiB
        return 512ULL * 1024ULL * 1024ULL;   // 512 MiB
    } else if (hw.gpu_vram_bytes <= 17179869184ULL) { // <= 16 GiB
        return 768ULL * 1024ULL * 1024ULL;   // 768 MiB
    } else {
        return 1024ULL * 1024ULL * 1024ULL;  // 1024 MiB
    }
}

ExecutionPlan MemoryPlanner::plan(
    const model::ModelDescriptor& desc,
    const hardware::HardwareInfo& hw,
    const PlannerOptions& options)
{
    ExecutionPlan p;

    // 1. Context length & KV precision (preserves explicit requests)
    const int64_t ctx = options.context_length.value_or(desc.attn.context_length > 0 ? desc.attn.context_length : 262144);
    const KvPrecision kv_prec = options.kv_format.value_or(KvPrecision::FP16);

    // 2. Fixed GPU requirements & baselines
    const uint64_t dense_gpu_bytes = estimate_dense_gpu_bytes(desc);
    const uint64_t safety_reserve = calculate_safety_reserve_bytes(hw);

    int64_t prefill_chunk = options.prefill_chunk.value_or((hw.gpu_vram_bytes <= 8589934592ULL) ? 512 : 2048);
    uint64_t prompt_scratch = calculate_prompt_scratch_bytes(desc, prefill_chunk);

    bool enable_mtp = options.enable_mtp;
    int mtp_spec_tokens = options.spec_tokens.value_or(4);
    uint64_t mtp_bytes = enable_mtp ? (250ULL * 1024ULL * 1024ULL) : 0ULL;

    // 3. KV mode decision
    KvMode kv_mode = KvMode::HostOnly;
    if (options.kv_mode.has_value()) {
        kv_mode = options.kv_mode.value();
    } else {
        // Automatic decision: prefer normal KV residency if it comfortably fits
        const KvBudget res_kv = calculate_kv_budget(desc, ctx, kv_prec, KvMode::Resident);
        const uint64_t req_resident = dense_gpu_bytes + mtp_bytes + res_kv.persistent_gpu_bytes + prompt_scratch + safety_reserve;

        if (req_resident <= hw.gpu_vram_bytes && (hw.gpu_vram_bytes > 8589934592ULL || ctx <= 32768)) {
            kv_mode = KvMode::Resident;
        } else {
            // Otherwise use host-only KV
            kv_mode = KvMode::HostOnly;
        }
    }

    // 4. Iterative degradation on tight VRAM (Rules 5, 6, 7)
    KvBudget kv_b = calculate_kv_budget(desc, ctx, kv_prec, kv_mode);
    uint64_t total_gpu = dense_gpu_bytes + mtp_bytes + kv_b.persistent_gpu_bytes + kv_b.staging_gpu_bytes + prompt_scratch + safety_reserve;

    // Rule 6: Reduce prefill chunk if necessary
    if (total_gpu > hw.gpu_vram_bytes && prefill_chunk > 512 && !options.prefill_chunk.has_value()) {
        prefill_chunk = 512;
        prompt_scratch = calculate_prompt_scratch_bytes(desc, prefill_chunk);
        total_gpu = dense_gpu_bytes + mtp_bytes + kv_b.persistent_gpu_bytes + kv_b.staging_gpu_bytes + prompt_scratch + safety_reserve;
        p.warnings.push_back("Reduced prefill chunk to 512 to preserve prompt scratch in VRAM.");
    }

    // Rule 5: Retain MTP/speculation if prompt scratch still fits; otherwise disable MTP
    if (total_gpu > hw.gpu_vram_bytes && enable_mtp) {
        enable_mtp = false;
        mtp_bytes = 0;
        mtp_spec_tokens = 0;
        total_gpu = dense_gpu_bytes + mtp_bytes + kv_b.persistent_gpu_bytes + kv_b.staging_gpu_bytes + prompt_scratch + safety_reserve;
        p.warnings.push_back("Disabled speculative decoding (MTP) to fit prompt scratch in VRAM.");
    }

    // Fallback to host-only KV if resident was auto-chosen and still doesn't fit
    if (total_gpu > hw.gpu_vram_bytes && kv_mode != KvMode::HostOnly && !options.kv_mode.has_value()) {
        kv_mode = KvMode::HostOnly;
        kv_b = calculate_kv_budget(desc, ctx, kv_prec, kv_mode);
        total_gpu = dense_gpu_bytes + mtp_bytes + kv_b.persistent_gpu_bytes + kv_b.staging_gpu_bytes + prompt_scratch + safety_reserve;
        p.warnings.push_back("Switched to host-only KV mode to free VRAM for prompt scratch.");
    }

    // 5. Expert Tier Distribution
    const uint64_t fixed_gpu_allocated = dense_gpu_bytes + mtp_bytes + kv_b.persistent_gpu_bytes + kv_b.staging_gpu_bytes + prompt_scratch + safety_reserve;
    const uint64_t free_vram = (hw.gpu_vram_bytes > fixed_gpu_allocated) ? (hw.gpu_vram_bytes - fixed_gpu_allocated) : 0ULL;

    // Size GPU expert cache only from genuinely free VRAM
    uint64_t free_vram_for_cache = free_vram;
    if (hw.gpu_vram_bytes <= 8589934592ULL && !options.gpu_cache_slots.has_value()) {
        // On 8 GB cards preserve remaining VRAM for allocator headroom
        free_vram_for_cache = 0;
    }

    const uint64_t host_headroom = 10ULL * 1024ULL * 1024ULL * 1024ULL; // 10 GiB OS/engine buffer
    const uint64_t available_ram = (hw.ram_total_bytes > kv_b.pinned_host_bytes + host_headroom)
                                       ? (hw.ram_total_bytes - kv_b.pinned_host_bytes - host_headroom)
                                       : 0ULL;

    std::optional<uint64_t> explicit_ram_bytes;
    if (options.resident_budget_gib.has_value()) {
        explicit_ram_bytes = (uint64_t) options.resident_budget_gib.value() * 1024ULL * 1024ULL * 1024ULL;
    }

    const ExpertTierDistribution expert_dist = calculate_expert_distribution(
        desc, available_ram, free_vram_for_cache, explicit_ram_bytes, options.gpu_cache_slots);

    // 6. Populate ExecutionPlan
    p.context_length = ctx;
    p.kv_format = kv_prec;
    p.kv_mode = kv_mode;
    p.host_only_kv = (kv_mode == KvMode::HostOnly);

    p.full_host_kv_bytes = kv_b.full_host_bytes;
    p.persistent_gpu_kv_bytes = kv_b.persistent_gpu_bytes;
    p.kv_staging_bytes = kv_b.staging_gpu_bytes;

    p.routed_expert_total_bytes = expert_dist.total_bytes;
    p.total_routed_experts = expert_dist.total_experts;
    p.routed_experts_in_ram = expert_dist.ram_resident_slots;
    p.routed_experts_ram_bytes = expert_dist.ram_resident_bytes;
    p.routed_experts_in_gpu = expert_dist.gpu_cache_slots;
    p.routed_experts_gpu_bytes = expert_dist.gpu_cache_bytes;
    p.routed_experts_on_file = expert_dist.file_slots;
    p.routed_experts_file_bytes = expert_dist.file_bytes;

    p.pinned_host_bytes = kv_b.pinned_host_bytes;
    p.dense_gpu_bytes = dense_gpu_bytes;
    p.mtp_bytes = mtp_bytes;
    p.mtp_spec_tokens = mtp_spec_tokens;
    p.prompt_scratch_bytes = prompt_scratch;
    p.safety_reserve_bytes = safety_reserve;
    p.prefill_chunk = prefill_chunk;

    p.total_host_ram_bytes = hw.ram_total_bytes;
    p.total_gpu_vram_bytes = hw.gpu_vram_bytes;

    const uint64_t total_vram_used = p.dense_gpu_bytes + p.mtp_bytes + p.persistent_gpu_kv_bytes +
                                    p.kv_staging_bytes + p.routed_experts_gpu_bytes +
                                    p.prompt_scratch_bytes + p.safety_reserve_bytes;
    p.remaining_vram_bytes = (hw.gpu_vram_bytes > total_vram_used) ? (hw.gpu_vram_bytes - total_vram_used) : 0ULL;

    const uint64_t total_ram_used = p.pinned_host_bytes + p.routed_experts_ram_bytes + host_headroom;
    p.remaining_host_ram_bytes = (hw.ram_total_bytes > total_ram_used) ? (hw.ram_total_bytes - total_ram_used) : 0ULL;

    return p;
}

PlanValidation MemoryPlanner::validate(
    const ExecutionPlan& plan,
    const hardware::HardwareInfo& hw,
    const model::ModelDescriptor& /*desc*/)
{
    PlanValidation val;

    if (plan.context_length <= 0) {
        val.valid = false;
        val.errors.push_back("Context length must be greater than 0.");
    }

    // Check worst-case GPU memory allocation
    const uint64_t total_gpu_needed = plan.dense_gpu_bytes + plan.mtp_bytes +
                                     plan.persistent_gpu_kv_bytes + plan.kv_staging_bytes +
                                     plan.routed_experts_gpu_bytes + plan.prompt_scratch_bytes +
                                     plan.safety_reserve_bytes;

    if (total_gpu_needed > hw.gpu_vram_bytes) {
        val.valid = false;
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "Estimated worst-case GPU allocation (%.2f GiB) exceeds VRAM (%.2f GiB): prompt scratch cannot allocate.",
                      (double) total_gpu_needed / (1024.0 * 1024.0 * 1024.0),
                      (double) hw.gpu_vram_bytes / (1024.0 * 1024.0 * 1024.0));
        val.errors.push_back(buf);
    }

    // Verify prompt scratch allocation headroom explicitly
    const uint64_t fixed_pre_scratch = plan.dense_gpu_bytes + plan.mtp_bytes +
                                      plan.persistent_gpu_kv_bytes + plan.kv_staging_bytes +
                                      plan.routed_experts_gpu_bytes + plan.safety_reserve_bytes;

    if (hw.gpu_vram_bytes <= fixed_pre_scratch ||
        (hw.gpu_vram_bytes - fixed_pre_scratch) < plan.prompt_scratch_bytes)
    {
        val.valid = false;
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "Insufficient VRAM for prompt prefill scratch (needs %.1f MiB, only %.1f MiB available).",
                      (double) plan.prompt_scratch_bytes / (1024.0 * 1024.0),
                      hw.gpu_vram_bytes > fixed_pre_scratch ? (double)(hw.gpu_vram_bytes - fixed_pre_scratch) / (1024.0 * 1024.0) : 0.0);
        val.errors.push_back(buf);
    }

    // Check Host RAM overcommit
    const uint64_t total_host_needed = plan.pinned_host_bytes + plan.routed_experts_ram_bytes;
    if (total_host_needed > hw.ram_total_bytes) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
                      "Host memory requirements (%.1f GiB) exceed physical RAM (%.1f GiB). OS swapping may occur.",
                      (double) total_host_needed / (1024.0 * 1024.0 * 1024.0),
                      (double) hw.ram_total_bytes / (1024.0 * 1024.0 * 1024.0));
        val.warnings.push_back(buf);
    }

    return val;
}

}  // namespace guild::memory
