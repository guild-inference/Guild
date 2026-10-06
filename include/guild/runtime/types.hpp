#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace guild::runtime {

struct ModelPaths {
    std::string primary_model_path;
    std::vector<std::string> additional_shards;
    std::string pack_dir;
    std::string expert_profile_path;
    std::string mtp_dir;
    std::string tokenizer_dir;
    std::string ple_gguf;
};

struct RuntimeConfig {
    int64_t max_context = 262144;
    std::string kv_format = "fp16";
    bool kv_host_only = true;
    int64_t kv_resident = 0;
    uint64_t resident_budget_gib = 56;
    int spec = 4;
    float spec_min_p = 0.5f;
    int expert_cache_slots = 0; // -1 for auto, 0 for off, >0 explicit
    int64_t prefill_chunk = 512;
    int pool_workers = 0;
    int vram_reserve_mib = 512;
    std::vector<int64_t> eos_token_ids = {151643, 151645, 151644}; // Qwen default EOS/im_end
};

struct GenerationRequest {
    std::string request_id;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;
    int max_new_tokens = 512;
    float temperature = 0.0f; // 0.0 = greedy
    float top_p = 1.0f;
    int top_k = 0;
    uint64_t seed = 0;
    std::vector<std::string> stop;
    std::shared_ptr<std::atomic<bool>> cancel_flag;
};

struct TokenEvent {
    int32_t token_id = 0;
    std::string text;
    bool is_special = false;
};

using TokenStreamCallback = std::function<bool(const TokenEvent& token)>;

struct PrefillProgressEvent {
    int64_t pos = 0;
    int64_t total = 0;
    double ms = 0.0;
    double tok_s = 0.0;
};

using PrefillProgressCallback = std::function<void(const PrefillProgressEvent& ev)>;

struct RuntimeTelemetry {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double prompt_ms = 0.0;
    double decode_ms = 0.0;
    double prompt_tok_s = 0.0;
    double decode_tok_s = 0.0;
    std::string finish_reason = "stop"; // "stop", "length", "cancel", "error"
    int drafts_accepted = 0;
    int drafts_offered = 0;
    int64_t ram_expert_hits = 0;
    int64_t file_expert_reads = 0;
    int64_t gpu_cache_hits = 0;
    uint64_t kv_usage_bytes = 0;
    int64_t current_context_length = 0;
    int64_t prefill_chunk = 0;
    double request_latency_s = 0.0;
};

} // namespace guild::runtime
