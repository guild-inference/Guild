#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

namespace guild::server {

struct RequestMetrics {
    std::string method;
    std::string path;
    int status_code = 200;
    double duration_s = 0.0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double prompt_ms = 0.0;
    double decode_ms = 0.0;
    double prompt_tok_s = 0.0;
    double decode_tok_s = 0.0;
    int64_t ram_expert_hits = 0;
    int64_t file_expert_reads = 0;
    int64_t gpu_cache_hits = 0;
    int drafts_accepted = 0;
    int drafts_offered = 0;
    int64_t context_tokens = 0;
    bool streamed = false;
    std::string outcome = "ok"; // "ok", "disconnected", "error"
};

struct TelemetrySnapshot {
    int64_t active_requests = 0;
    int64_t completed_requests = 0;
    int64_t prompt_tokens_processed = 0;
    int64_t generated_tokens_produced = 0;
    double prompt_tok_s_last = 0.0;
    double prompt_tok_s_mean = 0.0;
    double decode_tok_s_last = 0.0;
    double decode_tok_s_mean = 0.0;
    int64_t ram_expert_hits = 0;
    int64_t file_expert_reads = 0;
    int64_t gpu_cache_hits = 0;
    int64_t drafts_accepted = 0;
    int64_t drafts_offered = 0;
    int64_t context_usage = 0;
};

class Telemetry {
public:
    Telemetry() = default;

    void record_request_start();
    void record_request_finish(const RequestMetrics& m);

    TelemetrySnapshot snapshot() const;

    static std::string format_live_line(const RequestMetrics& m);
    static std::string format_log_line(const RequestMetrics& m);
    static std::string format_jsonl(const RequestMetrics& m);

private:
    std::atomic<int64_t> active_requests_{0};
    std::atomic<int64_t> completed_requests_{0};
    std::atomic<int64_t> prompt_tokens_processed_{0};
    std::atomic<int64_t> generated_tokens_produced_{0};
    std::atomic<int64_t> ram_expert_hits_{0};
    std::atomic<int64_t> file_expert_reads_{0};
    std::atomic<int64_t> gpu_cache_hits_{0};
    std::atomic<int64_t> drafts_accepted_{0};
    std::atomic<int64_t> drafts_offered_{0};
    std::atomic<int64_t> context_usage_{0};

    mutable std::mutex stats_mutex_;
    double total_prompt_time_s_{0.0};
    double total_decode_time_s_{0.0};
    double prompt_tok_s_last_{0.0};
    double decode_tok_s_last_{0.0};
};

} // namespace guild::server
