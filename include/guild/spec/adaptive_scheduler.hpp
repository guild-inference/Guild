// include/guild/spec/adaptive_scheduler.hpp - Phase 6.6: cost-aware adaptive speculative decoding scheduler.
//
// Selects the token-generation strategy (Normal, MTP(1..K), Lookup(1..K), Retrieval(1..K)) that minimizes
// expected wall-clock milliseconds per committed target-model output token:
//
//   C(s, k) = E[iteration wall-clock time (ms)] / E[committed output tokens]
//
// where iteration wall-clock time includes drafting/lookup/retrieval overhead, target-model MoE verification,
// CUDA/CPU synchronization, and KV/GDN commit or rollback, and committed output tokens count actual target-model
// tokens emitted (accepted draft prefix + 1 target/bonus token).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace guild::spec {

enum class DecodeStrategyMode : uint8_t {
    Auto = 0,      // Legacy default: MTP if loaded and spec > 0, else Normal
    Normal = 1,    // Fixed autoregressive decoding (k = 0, window T = 1)
    Mtp = 2,       // Fixed MTP speculative decoding (up to max_draft)
    Lookup = 3,    // Fixed prompt-lookup speculative decoding (up to max_draft)
    Retrieval = 4, // Fixed session-local retrieval speculative decoding (up to max_draft)
    Adaptive = 5   // Dynamic cost-aware selection across Normal, Mtp, Lookup, and Retrieval
};

enum class CandidateSource : uint8_t {
    Normal = 0,
    Mtp = 1,
    Lookup = 2,
    Retrieval = 3
};

const char* decode_strategy_mode_to_string(DecodeStrategyMode mode) noexcept;
bool parse_decode_strategy_mode(const std::string& s, DecodeStrategyMode& out) noexcept;
const char* candidate_source_to_string(CandidateSource src) noexcept;

/// Hardware- and model-specific speculative cost calibration (persisted in GEXEC or initialized from priors).
struct GExecSpecCalibration {
    static constexpr int kMaxWindowT = 8;

    bool calibrated = false;
    // Total verification + commit wall-clock time (ms) for window sizes T = 1..8 (index 0 unused).
    double verify_window_ms[kMaxWindowT + 1] = {0.0, 20.0, 27.6, 35.4, 43.2, 51.0, 58.8, 66.6, 74.4};
    // CPU MoE expert dispatch time (ms) for window sizes T = 1..8.
    double verify_cpu_expert_ms[kMaxWindowT + 1] = {0.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0};
    // Ratio of distinct routed experts U(T) / U(1) for T = 1..8.
    double distinct_expert_ratio[kMaxWindowT + 1] = {0.0, 1.00, 1.72, 2.38, 2.98, 3.54, 4.06, 4.55, 5.01};
    double mtp_draft_step_ms = 1.60;
    double mtp_kv_catchup_ms = 0.35;
    double lookup_draft_ms = 0.015;
    double retrieval_draft_ms = 0.030;
    double kv_commit_ms = 0.12;
    double verify_noise_cv = 0.035;
    double concurrency_slope = 0.15;
    double context_attn_ms_per_1k = 0.15;
};

struct SchedulerConfig {
    DecodeStrategyMode mode = DecodeStrategyMode::Auto;
    int max_draft = 4;               // Maximum draft tokens k (window T = 1 + k <= 8)
    int cost_window = 16;            // Effective sliding/decay window for cumulative cost ratios
    double hysteresis_margin = 0.05; // Cost improvement margin (noise-calibrated when left at default 0.05)
    int min_dwell_steps = 3;         // Minimum iterations before switching away from a working strategy
    int reexplore_interval = 8;      // Maximum steps in Normal before decaying rejection streak
    float mtp_min_prob = 0.5f;       // Minimum confidence for MTP draft step 1
};

struct ProposalAvailability {
    bool mtp_available = false;
    int mtp_max_k = 0;
    float mtp_first_prob = 1.0f;

    int lookup_k = 0;
    int lookup_match_len = 0;
    int lookup_freq = 1;
    int lookup_candidates = 1;
    int lookup_consensus_k = -1; // -1 = use lookup_k; 0 = divergent variable slot; >0 = consensus prefix length

    int retrieval_k = 0;
    int retrieval_match_len = 0;
    int retrieval_candidates = 0;
    float retrieval_confidence = 0.0f;
    bool retrieval_from_corpus = false;

    int64_t context_tokens = 0;
    int active_sessions = 1;
    int remaining_tokens = 128;
};

struct ScheduleDecision {
    CandidateSource source = CandidateSource::Normal;
    int draft_k = 0; // Draft tokens to verify (window T = 1 + draft_k)
    double expected_tokens = 1.0;
    double expected_iteration_ms = 20.0;
    double expected_ms_per_token = 20.0;
    double normal_ms_per_token = 20.0;
    double estimated_speedup = 1.0;
    bool switched = false;
    const char* reason = "init";
};

struct IterationObservation {
    CandidateSource source = CandidateSource::Normal;
    int proposed_k = 0;
    int accepted_k = 0;
    int committed_tokens = 1;
    int match_len = 0;
    double draft_ms = 0.0;
    double lookup_probe_ms = 0.0;
    double retrieval_probe_ms = 0.0;
    double verify_ms = 0.0;
    double commit_ms = 0.0;
    double cpu_expert_ms = 0.0;
    double scheduler_ms = 0.0;
    int64_t distinct_experts = 0;
};

struct SchedulerTelemetry {
    std::string mode_name = "auto";
    std::string initial_strategy = "normal";
    std::string final_strategy = "normal";
    std::string last_switch_reason = "none";

    int64_t iterations = 0;
    int64_t normal_iterations = 0;
    int64_t mtp_iterations = 0;
    int64_t lookup_iterations = 0;
    int64_t retrieval_iterations = 0;

    int64_t total_proposed_drafts = 0;
    int64_t total_accepted_drafts = 0;
    int64_t total_committed_tokens = 0;

    double avg_proposed_draft_len = 0.0;
    double avg_accepted_prefix_len = 0.0;
    double acceptance_rate = 0.0;

    double total_iteration_ms = 0.0;
    double total_draft_ms = 0.0;
    double total_lookup_ms = 0.0;
    double total_retrieval_ms = 0.0;
    double total_verify_ms = 0.0;
    double total_commit_ms = 0.0;
    double total_scheduler_ms = 0.0;

    double measured_ms_per_token = 0.0;
    double estimated_normal_ms_per_token = 0.0;
    double estimated_speedup_vs_normal = 1.0;

    // Phase 6.6.1: Prediction accuracy & noise-calibrated hysteresis diagnostics
    double mean_abs_cost_pred_error_ms = 0.0;
    double mean_abs_acc_pred_error = 0.0;
    double effective_hysteresis_margin = 0.05;
    double measured_verify_cv = 0.035;

    int64_t strategy_switches = 0;
    int64_t fallback_to_normal_count = 0;
};

class AdaptiveDecodeScheduler {
public:
    static constexpr int kMaxDraft = 7;
    static constexpr int kMaxWindowT = 8;
    static constexpr int kMatchBuckets = 4;

    explicit AdaptiveDecodeScheduler(SchedulerConfig cfg = {}, GExecSpecCalibration cal = {});

    void configure(const SchedulerConfig& cfg);
    void set_calibration(const GExecSpecCalibration& cal);
    void reset_session_stats();

    const SchedulerConfig& config() const {
        return cfg_;
    }
    const GExecSpecCalibration& calibration() const {
        return cal_;
    }

    /// Seed the live T=1 verification latency and CPU expert split from the request's initial single-token window.
    void seed_single_token_verify(double verify_ms, int64_t distinct_experts = 0, double cpu_expert_ms = 0.0);

    /// Choose the best decoding strategy and draft depth for the upcoming verification window.
    ScheduleDecision choose(const ProposalAvailability& avail);

    /// Record the measured wall-clock outcome of a completed verification window.
    void observe(const IterationObservation& obs);

    /// Estimated verification + commit wall-clock cost (ms) for a window of `T` tokens (1..8).
    double estimate_verify_ms(int T, int active_sessions = 1) const;

    /// Estimated ms per committed token for a strategy at depth `k`.
    double estimate_strategy_cost(CandidateSource src, int k, int match_len, float confidence, bool mtp_active_in_model,
                                  int active_sessions, double* out_expected_tokens = nullptr,
                                  double* out_expected_iter_ms = nullptr, int lookup_candidates = 1) const;

    /// Noise-calibrated switching hysteresis margin for the current hardware/session.
    double effective_hysteresis_margin() const noexcept;

    CandidateSource current_source() const {
        return current_source_;
    }
    int current_k() const {
        return current_k_;
    }
    int64_t switches() const {
        return switches_;
    }
    int64_t fallbacks() const {
        return fallbacks_;
    }

    SchedulerTelemetry snapshot() const;

private:
    static int match_bucket(int match_len) noexcept;
    static double continuous_match_prior(int match_len, float freq_or_conf, int candidates, bool is_retrieval) noexcept;

    SchedulerConfig cfg_;
    GExecSpecCalibration cal_;

    // Online per-window-size T = 1..8 verification + commit latency (decayed cumulative ms and rounds).
    double ver_decayed_ms_[kMaxWindowT + 1] = {};
    double ver_decayed_rounds_[kMaxWindowT + 1] = {};
    double ver_decayed_experts_[kMaxWindowT + 1] = {};
    int64_t ver_obs_count_[kMaxWindowT + 1] = {};
    double live_cpu_fraction_ = -1.0; // Measured CPU expert fraction of T=1 verify (-1 = use calibrated curve)
    double online_verify_cv_ = 0.035; // Online relative dispersion of verification window latency

    // Online drafting latencies (EMA in ms).
    double mtp_step_ms_ = 1.60;
    double mtp_catchup_ms_ = 0.35;
    double lookup_ms_ = 0.015;
    double retrieval_ms_ = 0.030;

    // MTP conditional acceptance Beta-Binomial posterior (decayed successes and trials per draft step 0..kMaxDraft-1).
    double mtp_cond_p_[kMaxDraft] = {};
    double mtp_succ_[kMaxDraft] = {};
    double mtp_trials_[kMaxDraft] = {};

    // Lookup and Retrieval per-bucket token acceptance Beta-Binomial posteriors (decayed accepted / decayed proposed).
    double lookup_acc_[kMatchBuckets] = {};
    double lookup_prop_[kMatchBuckets] = {};
    double retrieval_acc_[kMatchBuckets] = {};
    double retrieval_prop_[kMatchBuckets] = {};

    // Per-strategy decayed cumulative wall-clock time (ms) and committed tokens for ratio estimation.
    double strat_decayed_ms_[4] = {};
    double strat_decayed_tok_[4] = {};
    double strat_total_ms_[4] = {};
    int64_t strat_total_tok_[4] = {};
    int64_t strat_iters_[4] = {};
    int consecutive_rejects_[4] = {};

    // State machine for hysteresis & exploration.
    bool has_started_ = false;
    CandidateSource initial_source_ = CandidateSource::Normal;
    CandidateSource current_source_ = CandidateSource::Normal;
    int current_k_ = 0;
    int dwell_steps_ = 0;
    int steps_in_normal_ = 0;
    int64_t switches_ = 0;
    int64_t fallbacks_ = 0;
    const char* last_switch_reason_ = "none";

    // Last decision's predictions (for online prediction error tracking).
    double last_pred_ms_per_tok_ = 0.0;
    double last_pred_exp_tok_ = 1.0;
    double sum_abs_cost_err_ms_ = 0.0;
    double sum_abs_acc_err_ = 0.0;
    int64_t pred_eval_count_ = 0;

    // Cumulative telemetry counters.
    int64_t total_iters_ = 0;
    int64_t total_proposed_ = 0;
    int64_t total_accepted_ = 0;
    int64_t total_committed_ = 0;
    double total_iter_ms_ = 0.0;
    double total_mtp_draft_ms_ = 0.0;
    double total_lookup_ms_ = 0.0;
    double total_retrieval_ms_ = 0.0;
    double total_verify_ms_ = 0.0;
    double total_commit_ms_ = 0.0;
    double total_scheduler_ms_ = 0.0;
};

} // namespace guild::spec
