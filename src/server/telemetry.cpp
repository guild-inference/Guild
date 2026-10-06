#include "guild/server/telemetry.hpp"
#include "guild/cli/ansi.hpp"
#include "guild/server/json.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>

namespace guild::server {

void Telemetry::record_request_start() {
    active_requests_++;
}

void Telemetry::record_request_finish(const RequestMetrics& m) {
    if (active_requests_ > 0) active_requests_--;
    completed_requests_++;
    prompt_tokens_processed_ += m.prompt_tokens;
    generated_tokens_produced_ += m.completion_tokens;
    ram_expert_hits_ += m.ram_expert_hits;
    file_expert_reads_ += m.file_expert_reads;
    gpu_cache_hits_ += m.gpu_cache_hits;
    drafts_accepted_ += m.drafts_accepted;
    drafts_offered_ += m.drafts_offered;
    context_usage_ = m.context_tokens;

    std::lock_guard<std::mutex> lock(stats_mutex_);
    if (m.prompt_tok_s > 0.0) prompt_tok_s_last_ = m.prompt_tok_s;
    if (m.decode_tok_s > 0.0) decode_tok_s_last_ = m.decode_tok_s;
    if (m.prompt_ms > 0.0) total_prompt_time_s_ += m.prompt_ms / 1000.0;
    if (m.decode_ms > 0.0) total_decode_time_s_ += m.decode_ms / 1000.0;
}

TelemetrySnapshot Telemetry::snapshot() const {
    TelemetrySnapshot s;
    s.active_requests = active_requests_.load();
    s.completed_requests = completed_requests_.load();
    s.prompt_tokens_processed = prompt_tokens_processed_.load();
    s.generated_tokens_produced = generated_tokens_produced_.load();
    s.ram_expert_hits = ram_expert_hits_.load();
    s.file_expert_reads = file_expert_reads_.load();
    s.gpu_cache_hits = gpu_cache_hits_.load();
    s.drafts_accepted = drafts_accepted_.load();
    s.drafts_offered = drafts_offered_.load();
    s.context_usage = context_usage_.load();

    std::lock_guard<std::mutex> lock(stats_mutex_);
    s.prompt_tok_s_last = prompt_tok_s_last_;
    s.decode_tok_s_last = decode_tok_s_last_;
    s.prompt_tok_s_mean = (total_prompt_time_s_ > 0.0)
                              ? (static_cast<double>(s.prompt_tokens_processed) / total_prompt_time_s_)
                              : 0.0;
    s.decode_tok_s_mean = (total_decode_time_s_ > 0.0)
                              ? (static_cast<double>(s.generated_tokens_produced) / total_decode_time_s_)
                              : 0.0;
    return s;
}

std::string Telemetry::format_live_line(const RequestMetrics& m) {
    using namespace guild::cli::ansi;
    std::ostringstream ss;

    ss << bold() << m.method << " " << reset() << std::left << std::setw(24) << m.path << "  ";

    if (m.status_code >= 200 && m.status_code < 300) {
        ss << green() << bold() << m.status_code << reset();
    } else if (m.status_code >= 400 && m.status_code < 500) {
        ss << yellow() << bold() << m.status_code << reset();
    } else {
        ss << red() << bold() << m.status_code << reset();
    }

    ss << "  " << std::right << std::setw(6) << std::fixed << std::setprecision(2) << m.duration_s << "s";

    if (m.completion_tokens > 0) {
        ss << "  " << std::setw(5) << m.completion_tokens << " tok";
    } else if (m.prompt_tokens > 0) {
        ss << "  " << std::setw(5) << m.prompt_tokens << " ptok";
    } else {
        ss << "         ";
    }

    if (m.decode_tok_s > 0.0) {
        ss << "  " << cyan() << std::fixed << std::setprecision(1) << m.decode_tok_s << " tok/s" << reset();
    } else if (m.prompt_tok_s > 0.0) {
        ss << "  " << cyan() << std::fixed << std::setprecision(1) << m.prompt_tok_s << " ptok/s" << reset();
    }

    if (m.outcome == "disconnected") {
        ss << "  " << yellow() << "[disconnected]" << reset();
    } else if (m.outcome == "error") {
        ss << "  " << red() << "[error]" << reset();
    }

    return ss.str();
}

std::string Telemetry::format_log_line(const RequestMetrics& m) {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char time_str[32];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

    std::ostringstream ss;
    ss << "[" << time_str << "] " << m.method << " " << m.path << " " << m.status_code
       << " " << std::fixed << std::setprecision(2) << m.duration_s << "s";

    if (m.completion_tokens > 0) {
        ss << " " << m.completion_tokens << " tok";
    }
    if (m.decode_tok_s > 0.0) {
        ss << " " << std::fixed << std::setprecision(1) << m.decode_tok_s << " tok/s";
    }
    if (m.outcome != "ok") {
        ss << " [" << m.outcome << "]";
    }
    return ss.str();
}

std::string Telemetry::format_jsonl(const RequestMetrics& m) {
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

    std::ostringstream ss;
    ss << "{\"timestamp\":" << now_ms
       << ",\"method\":\"" << json::JsonValue::escape_string(m.method) << "\""
       << ",\"path\":\"" << json::JsonValue::escape_string(m.path) << "\""
       << ",\"status\":" << m.status_code
       << ",\"duration_s\":" << std::fixed << std::setprecision(3) << m.duration_s
       << ",\"prompt_tokens\":" << m.prompt_tokens
       << ",\"completion_tokens\":" << m.completion_tokens
       << ",\"prompt_tok_s\":" << std::setprecision(1) << m.prompt_tok_s
       << ",\"decode_tok_s\":" << std::setprecision(1) << m.decode_tok_s
       << ",\"ram_expert_hits\":" << m.ram_expert_hits
       << ",\"file_expert_reads\":" << m.file_expert_reads
       << ",\"gpu_cache_hits\":" << m.gpu_cache_hits
       << ",\"drafts_accepted\":" << m.drafts_accepted
       << ",\"drafts_offered\":" << m.drafts_offered
       << ",\"context_tokens\":" << m.context_tokens
       << ",\"stream\":" << (m.streamed ? "true" : "false")
       << ",\"outcome\":\"" << json::JsonValue::escape_string(m.outcome) << "\"}";
    return ss.str();
}

} // namespace guild::server
