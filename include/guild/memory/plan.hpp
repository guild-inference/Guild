#pragma once

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace guild::memory {

enum class KvMode {
    Resident,    ///< Full KV resident in GPU VRAM
    Streaming,   ///< GPU resident sliding window with host RAM backing
    HostOnly     ///< Full KV in pinned host RAM, bounded GPU staging pool
};

inline const char* kv_mode_to_string(KvMode mode) {
    switch (mode) {
    case KvMode::Resident: return "resident";
    case KvMode::Streaming: return "streaming";
    case KvMode::HostOnly: return "host-only";
    default: return "unknown";
    }
}

inline KvMode kv_mode_from_string(const std::string& str) {
    if (str == "resident") return KvMode::Resident;
    if (str == "streaming") return KvMode::Streaming;
    if (str == "host-only" || str == "host_only") return KvMode::HostOnly;
    return KvMode::HostOnly;
}

enum class KvPrecision {
    FP16,
    INT8,
    Q4_0,
    K8V4
};

inline const char* kv_precision_to_string(KvPrecision prec) {
    switch (prec) {
    case KvPrecision::FP16: return "FP16";
    case KvPrecision::INT8: return "INT8";
    case KvPrecision::Q4_0: return "Q4_0";
    case KvPrecision::K8V4: return "K8V4";
    default: return "FP16";
    }
}

inline KvPrecision kv_precision_from_string(const std::string& str) {
    if (str == "fp16" || str == "FP16") return KvPrecision::FP16;
    if (str == "int8" || str == "INT8" || str == "q8_0") return KvPrecision::INT8;
    if (str == "q4_0" || str == "Q4_0") return KvPrecision::Q4_0;
    if (str == "k8v4" || str == "K8V4") return KvPrecision::K8V4;
    return KvPrecision::FP16;
}

struct PlanValidation {
    bool valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    bool ok() const { return valid && errors.empty(); }
};

struct ExecutionPlan {
    // Context and KV configuration
    int64_t context_length = 0;
    KvPrecision kv_format = KvPrecision::FP16;
    KvMode kv_mode = KvMode::HostOnly;
    bool host_only_kv = false;

    // KV Cache byte sizing
    uint64_t full_host_kv_bytes = 0;
    uint64_t persistent_gpu_kv_bytes = 0;
    uint64_t kv_staging_bytes = 0;

    // Routed experts tier distribution
    uint64_t routed_expert_total_bytes = 0;
    int64_t total_routed_experts = 0;

    int64_t routed_experts_in_ram = 0;
    uint64_t routed_experts_ram_bytes = 0;

    int64_t routed_experts_in_gpu = 0;
    uint64_t routed_experts_gpu_bytes = 0;

    int64_t routed_experts_on_file = 0;
    uint64_t routed_experts_file_bytes = 0;

    // Memory footprint components
    uint64_t pinned_host_bytes = 0;
    uint64_t dense_gpu_bytes = 0;

    // Speculative decoding / MTP
    uint64_t mtp_bytes = 0;
    int mtp_spec_tokens = 0;

    // Prefill scratch and safety margins
    uint64_t prompt_scratch_bytes = 0;
    uint64_t safety_reserve_bytes = 0;
    int64_t prefill_chunk = 512;

    // Warnings and degradations
    std::vector<std::string> warnings;

    // Remaining capacity estimates
    uint64_t remaining_host_ram_bytes = 0;
    uint64_t remaining_vram_bytes = 0;

    // Total detected capacity
    uint64_t total_host_ram_bytes = 0;
    uint64_t total_gpu_vram_bytes = 0;

    std::string to_human_string() const;
    std::string to_json_string() const;
};

}  // namespace guild::memory
