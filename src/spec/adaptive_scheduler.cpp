// src/spec/adaptive_scheduler.cpp - see include/guild/spec/adaptive_scheduler.hpp.
#include "guild/spec/adaptive_scheduler.hpp"

#include <algorithm>
#include <cmath>

namespace guild::spec {

namespace {
constexpr double kDefaultMtpCondP[AdaptiveDecodeScheduler::kMaxDraft] = {0.78, 0.74, 0.70, 0.66, 0.62, 0.58, 0.54};
constexpr double kMtpPriorTrials = 3.0;
constexpr double kLookupPriorWeight = 2.5;
} // namespace

const char* decode_strategy_mode_to_string(DecodeStrategyMode mode) noexcept {
    switch (mode) {
    case DecodeStrategyMode::Auto: return "auto";
    case DecodeStrategyMode::Normal: return "normal";
    case DecodeStrategyMode::Mtp: return "mtp";
    case DecodeStrategyMode::Lookup: return "lookup";
    case DecodeStrategyMode::Retrieval: return "retrieval";
    case DecodeStrategyMode::Adaptive: return "adaptive";
    }
    return "auto";
}

bool parse_decode_strategy_mode(const std::string& s, DecodeStrategyMode& out) noexcept {
    if (s == "auto" || s == "default") {
        out = DecodeStrategyMode::Auto;
        return true;
    }
    if (s == "normal" || s == "none" || s == "off" || s == "0") {
        out = DecodeStrategyMode::Normal;
        return true;
    }
    if (s == "mtp") {
        out = DecodeStrategyMode::Mtp;
        return true;
    }
    if (s == "lookup" || s == "prompt-lookup" || s == "suffix") {
        out = DecodeStrategyMode::Lookup;
        return true;
    }
    if (s == "retrieval" || s == "rest") {
        out = DecodeStrategyMode::Retrieval;
        return true;
    }
    if (s == "adaptive" || s == "cost-aware") {
        out = DecodeStrategyMode::Adaptive;
        return true;
    }
    return false;
}

const char* candidate_source_to_string(CandidateSource src) noexcept {
    switch (src) {
    case CandidateSource::Normal: return "normal";
    case CandidateSource::Mtp: return "mtp";
    case CandidateSource::Lookup: return "lookup";
    case CandidateSource::Retrieval: return "retrieval";
    }
    return "normal";
}

AdaptiveDecodeScheduler::AdaptiveDecodeScheduler(SchedulerConfig cfg, GExecSpecCalibration cal) {
    configure(cfg);
    set_calibration(cal);
    reset_session_stats();
}

void AdaptiveDecodeScheduler::configure(const SchedulerConfig& cfg) {
    cfg_ = cfg;
    cfg_.max_draft = std::clamp(cfg_.max_draft, 0, kMaxDraft);
    cfg_.cost_window = std::max(4, cfg_.cost_window);
    cfg_.hysteresis_margin = std::clamp(cfg_.hysteresis_margin, 0.0, 0.5);
    cfg_.min_dwell_steps = std::max(1, cfg_.min_dwell_steps);
    cfg_.reexplore_interval = std::max(2, cfg_.reexplore_interval);
}

void AdaptiveDecodeScheduler::set_calibration(const GExecSpecCalibration& cal) {
    cal_ = cal;
    for (int t = 1; t <= kMaxWindowT; ++t) {
        if (cal_.verify_window_ms[t] <= 0.0)
            cal_.verify_window_ms[t] = 20.0 + 7.6 * (double) (t - 1);
        if (cal_.verify_cpu_expert_ms[t] <= 0.0)
            cal_.verify_cpu_expert_ms[t] = 14.0 + 7.0 * (double) (t - 1);
        if (cal_.distinct_expert_ratio[t] <= 0.0)
            cal_.distinct_expert_ratio[t] = 1.0 + 0.65 * (double) (t - 1);
    }
    cal_.verify_noise_cv = std::clamp(cal_.verify_noise_cv > 0.0 ? cal_.verify_noise_cv : 0.035, 0.01, 0.25);
    cal_.concurrency_slope = std::clamp(cal_.concurrency_slope > 0.0 ? cal_.concurrency_slope : 0.15, 0.02, 1.0);
    mtp_step_ms_ = std::max(0.05, cal_.mtp_draft_step_ms);
    mtp_catchup_ms_ = std::max(0.01, cal_.mtp_kv_catchup_ms);
    lookup_ms_ = std::max(0.001, cal_.lookup_draft_ms);
    retrieval_ms_ = std::max(0.001, cal_.retrieval_draft_ms);
    online_verify_cv_ = cal_.verify_noise_cv;
}

void AdaptiveDecodeScheduler::reset_session_stats() {
    for (int t = 0; t <= kMaxWindowT; ++t) {
        ver_decayed_ms_[t] = 0.0;
        ver_decayed_rounds_[t] = 0.0;
        ver_decayed_experts_[t] = 0.0;
        ver_obs_count_[t] = 0;
    }
    live_cpu_fraction_ = -1.0;
    online_verify_cv_ = cal_.verify_noise_cv;

    mtp_step_ms_ = std::max(0.05, cal_.mtp_draft_step_ms);
    mtp_catchup_ms_ = std::max(0.01, cal_.mtp_kv_catchup_ms);
    lookup_ms_ = std::max(0.001, cal_.lookup_draft_ms);
    retrieval_ms_ = std::max(0.001, cal_.retrieval_draft_ms);

    for (int i = 0; i < kMaxDraft; ++i) {
        mtp_cond_p_[i] = kDefaultMtpCondP[i];
        mtp_succ_[i] = kDefaultMtpCondP[i] * kMtpPriorTrials;
        mtp_trials_[i] = kMtpPriorTrials;
    }

    for (int b = 0; b < kMatchBuckets; ++b) {
        lookup_acc_[b] = 0.0;
        lookup_prop_[b] = 0.0;
        retrieval_acc_[b] = 0.0;
        retrieval_prop_[b] = 0.0;
    }

    for (int s = 0; s < 4; ++s) {
        strat_decayed_ms_[s] = 0.0;
        strat_decayed_tok_[s] = 0.0;
        strat_total_ms_[s] = 0.0;
        strat_total_tok_[s] = 0;
        strat_iters_[s] = 0;
        consecutive_rejects_[s] = 0;
    }

    has_started_ = false;
    initial_source_ = CandidateSource::Normal;
    current_source_ = CandidateSource::Normal;
    current_k_ = 0;
    dwell_steps_ = 0;
    steps_in_normal_ = 0;
    switches_ = 0;
    fallbacks_ = 0;
    last_switch_reason_ = "none";

    last_pred_ms_per_tok_ = 0.0;
    last_pred_exp_tok_ = 1.0;
    sum_abs_cost_err_ms_ = 0.0;
    sum_abs_acc_err_ = 0.0;
    pred_eval_count_ = 0;

    total_iters_ = 0;
    total_proposed_ = 0;
    total_accepted_ = 0;
    total_committed_ = 0;
    total_iter_ms_ = 0.0;
    total_mtp_draft_ms_ = 0.0;
    total_lookup_ms_ = 0.0;
    total_retrieval_ms_ = 0.0;
    total_verify_ms_ = 0.0;
    total_commit_ms_ = 0.0;
    total_scheduler_ms_ = 0.0;
}

int AdaptiveDecodeScheduler::match_bucket(int match_len) noexcept {
    if (match_len <= 3)
        return 0;
    if (match_len <= 6)
        return 1;
    if (match_len <= 12)
        return 2;
    return 3;
}

double AdaptiveDecodeScheduler::continuous_match_prior(int match_len, float freq_or_conf, int candidates,
                                                       bool is_retrieval) noexcept {
    if (is_retrieval) {
        const double conf = std::clamp((double) freq_or_conf, 0.1, 1.0);
        double base = 0.32;
        if (match_len <= 3) {
            base = (candidates >= 2 && conf >= 0.70) ? 0.78 : 0.32;
        } else {
            base = 1.0 - (0.65 / (1.0 + 0.65 * (double) (match_len - 3)));
        }
        return std::clamp(base * (0.78 + 0.22 * conf), 0.05, 0.97);
    }

    const int cands = std::max(1, candidates);
    const double freq = std::max(1.0, (double) freq_or_conf);
    const double agree_ratio = std::clamp(freq / (double) cands, 0.0, 1.0);

    // If 3+ earlier occurrences of this suffix were all followed by different tokens, it is a variable/counter slot.
    if (cands >= 3 && freq <= 1.0)
        return 0.15;

    // Unanimous continuation agreement across multiple historical occurrences (invariant template/boilerplate span).
    if (freq >= 2.0 && agree_ratio >= 0.99) {
        return std::min(0.96, 0.91 + 0.02 * std::min(3.0, freq - 1.0));
    }

    if (match_len <= 3) {
        if (freq >= 2.0 && agree_ratio >= 0.66)
            return 0.84;
        return 0.32;
    }

    if (match_len == 4) {
        if (freq >= 2.0 && agree_ratio >= 0.66)
            return 0.88;
        return 0.48;
    }

    double base = 1.0 - (0.42 / (1.0 + 0.65 * (double) (match_len - 4)));
    if (freq >= 2.0 && agree_ratio >= 0.66) {
        base = std::min(0.97, base + 0.08 * agree_ratio);
    } else if (cands >= 2 && agree_ratio < 0.66) {
        base *= (0.45 + 0.55 * agree_ratio);
    }
    return std::clamp(base, 0.05, 0.98);
}

double AdaptiveDecodeScheduler::effective_hysteresis_margin() const noexcept {
    if (std::abs(cfg_.hysteresis_margin - 0.05) > 1e-6)
        return cfg_.hysteresis_margin;
    return std::clamp(1.25 * online_verify_cv_, 0.018, 0.10);
}

void AdaptiveDecodeScheduler::seed_single_token_verify(double verify_ms, int64_t distinct_experts,
                                                       double cpu_expert_ms) {
    if (verify_ms <= 0.05)
        return;
    ver_decayed_ms_[1] = verify_ms;
    ver_decayed_rounds_[1] = 1.0;
    if (distinct_experts > 0)
        ver_decayed_experts_[1] = (double) distinct_experts;
    ver_obs_count_[1] = 1;
    if (cpu_expert_ms > 0.05 && cpu_expert_ms < verify_ms)
        live_cpu_fraction_ = std::clamp(cpu_expert_ms / verify_ms, 0.15, 0.90);
}

double AdaptiveDecodeScheduler::estimate_verify_ms(int T, int active_sessions) const {
    const int tc = std::clamp(T, 1, kMaxWindowT);
    double est = cal_.verify_window_ms[tc];

    if (ver_obs_count_[tc] > 0 && ver_decayed_rounds_[tc] > 1e-9) {
        est = ver_decayed_ms_[tc] / ver_decayed_rounds_[tc];
    } else if (ver_obs_count_[1] > 0 && ver_decayed_rounds_[1] > 1e-9 && live_cpu_fraction_ > 0.0 &&
               cal_.verify_cpu_expert_ms[1] > 1e-6) {
        // Decompose live T=1 latency into measured CPU expert share and GPU dense/attention share,
        // scaling each component by its hardware-calibrated expansion curve.
        const double t1 = ver_decayed_ms_[1] / ver_decayed_rounds_[1];
        const double cpu_ratio = cal_.verify_cpu_expert_ms[tc] / cal_.verify_cpu_expert_ms[1];
        const double gpu_ratio = 1.0 + 0.12 * (double) (tc - 1);
        est = t1 * (live_cpu_fraction_ * cpu_ratio + (1.0 - live_cpu_fraction_) * gpu_ratio);
    } else {
        // Infer from any observed window sizes using the calibrated MoE window cost curve.
        double weighted_scale = 0.0;
        double weight_sum = 0.0;
        for (int j = 1; j <= kMaxWindowT; ++j) {
            if (ver_obs_count_[j] > 0 && ver_decayed_rounds_[j] > 1e-9 && cal_.verify_window_ms[j] > 1e-6) {
                const double obs_j = ver_decayed_ms_[j] / ver_decayed_rounds_[j];
                const double w = std::min<double>(16.0, (double) ver_obs_count_[j]);
                weighted_scale += w * (obs_j / cal_.verify_window_ms[j]);
                weight_sum += w;
            }
        }
        if (weight_sum > 0.0)
            est = cal_.verify_window_ms[tc] * (weighted_scale / weight_sum);
    }

    // Under multi-session concurrency, multi-token windows increase CPU expert thread contention.
    if (active_sessions > 1 && tc > 1) {
        const double t1 = (tc == 1) ? est : estimate_verify_ms(1, 1);
        const double extra = std::max(0.0, est - t1);
        const double conc_factor = 1.0 + cal_.concurrency_slope * (double) (active_sessions - 1);
        est = t1 + extra * conc_factor;
    }
    return std::max(0.05, est);
}

double AdaptiveDecodeScheduler::estimate_strategy_cost(CandidateSource src, int k, int match_len, float confidence,
                                                       bool mtp_active_in_model, int active_sessions,
                                                       double* out_expected_tokens, double* out_expected_iter_ms,
                                                       int lookup_candidates) const {
    const int kc = std::clamp(k, 0, kMaxDraft);
    double exp_tok = 1.0;
    double exp_ms = estimate_verify_ms(1, active_sessions) + (mtp_active_in_model ? mtp_catchup_ms_ : 0.0);

    if (src == CandidateSource::Normal || kc == 0) {
        if (out_expected_tokens)
            *out_expected_tokens = 1.0;
        if (out_expected_iter_ms)
            *out_expected_iter_ms = exp_ms;
        return exp_ms;
    }

    const int sidx = (int) src;
    const int rej = consecutive_rejects_[sidx];
    const double virtual_fail = (rej > 0) ? (0.55 * (double) rej * (double) (rej + 1)) : 0.0;

    if (src == CandidateSource::Mtp) {
        double reach = 1.0;
        exp_tok = 1.0;
        for (int i = 0; i < kc; ++i) {
            const double denom = std::max(1.0, mtp_trials_[i] + virtual_fail);
            double p = mtp_succ_[i] / denom;
            if (i == 0 && confidence > 0.0f && confidence < 1.0f) {
                const double streak_scale = mtp_trials_[0] / denom;
                p = 0.75 * p + 0.25 * (double) confidence * streak_scale;
            }
            reach *= std::clamp(p, 0.01, 0.99);
            exp_tok += reach;
        }
        exp_ms = estimate_verify_ms(kc + 1, active_sessions) + (double) kc * mtp_step_ms_;
    } else if (src == CandidateSource::Lookup) {
        const int b = match_bucket(match_len);
        const double q0 = continuous_match_prior(match_len, confidence, lookup_candidates, false);
        const bool high_consensus = (match_len >= 6) || (lookup_candidates >= 3 && confidence >= 3.0f);
        const double eff_fail = high_consensus ? (0.35 * virtual_fail) : virtual_fail;
        const double q = std::clamp((kLookupPriorWeight * q0 + lookup_acc_[b]) /
                                        std::max(1e-6, kLookupPriorWeight + lookup_prop_[b] + eff_fail),
                                    0.01, 0.99);
        double reach = 1.0;
        exp_tok = 1.0;
        for (int i = 0; i < kc; ++i) {
            reach *= q;
            exp_tok += reach;
        }
        exp_ms =
            estimate_verify_ms(kc + 1, active_sessions) + lookup_ms_ + (mtp_active_in_model ? mtp_catchup_ms_ : 0.0);
    } else if (src == CandidateSource::Retrieval) {
        const int b = match_bucket(match_len);
        const double q0 = continuous_match_prior(match_len, confidence, lookup_candidates, true);
        const bool high_consensus = (match_len >= 6) || (lookup_candidates >= 3 && confidence >= 0.80f);
        const double eff_fail = high_consensus ? (0.35 * virtual_fail) : virtual_fail;
        const double q = std::clamp((kLookupPriorWeight * q0 + retrieval_acc_[b]) /
                                        std::max(1e-6, kLookupPriorWeight + retrieval_prop_[b] + eff_fail),
                                    0.01, 0.99);
        double reach = 1.0;
        exp_tok = 1.0;
        for (int i = 0; i < kc; ++i) {
            reach *= q;
            exp_tok += reach;
        }
        exp_ms =
            estimate_verify_ms(kc + 1, active_sessions) + retrieval_ms_ + (mtp_active_in_model ? mtp_catchup_ms_ : 0.0);
    }

    double cost = exp_ms / std::max(1.0, exp_tok);

    // Blend with realized cumulative time-to-output ratio when empirical data shows higher cost than the analytical
    // acceptance model. This guarantees an optimistic acceptance model cannot mask high verification cost.
    if (strat_iters_[sidx] >= 2 && strat_decayed_tok_[sidx] >= 1.0) {
        const double emp_ratio = strat_decayed_ms_[sidx] / strat_decayed_tok_[sidx];
        if (emp_ratio > cost) {
            const double blend = std::min(0.60, 0.12 * (double) strat_iters_[sidx]);
            cost = (1.0 - blend) * cost + blend * emp_ratio;
        }
    }

    if (out_expected_tokens)
        *out_expected_tokens = exp_tok;
    if (out_expected_iter_ms)
        *out_expected_iter_ms = exp_ms;
    return cost;
}

ScheduleDecision AdaptiveDecodeScheduler::choose(const ProposalAvailability& avail) {
    ScheduleDecision d;
    const bool mtp_ok = avail.mtp_available && avail.mtp_max_k > 0 && cfg_.max_draft > 0;
    const double normal_cost =
        estimate_strategy_cost(CandidateSource::Normal, 0, 0, 1.0f, avail.mtp_available, avail.active_sessions,
                               &d.expected_tokens, &d.expected_iteration_ms);
    d.normal_ms_per_token = normal_cost;
    d.expected_ms_per_token = normal_cost;
    d.estimated_speedup = 1.0;

    auto finalize = [&](CandidateSource src, int k, double exp_tok, double exp_ms, double cost, const char* why) {
        d.source = src;
        d.draft_k = (src == CandidateSource::Normal) ? 0 : k;
        d.expected_tokens = exp_tok;
        d.expected_iteration_ms = exp_ms;
        d.expected_ms_per_token = cost;
        d.estimated_speedup = (cost > 1e-6) ? (normal_cost / cost) : 1.0;
        d.reason = why;

        last_pred_ms_per_tok_ = cost;
        last_pred_exp_tok_ = exp_tok;

        if (!has_started_) {
            has_started_ = true;
            initial_source_ = d.source;
            current_source_ = d.source;
            current_k_ = d.draft_k;
            dwell_steps_ = 1;
            d.switched = false;
        } else if (d.source != current_source_) {
            d.switched = true;
            ++switches_;
            if (d.source == CandidateSource::Normal && current_source_ != CandidateSource::Normal)
                ++fallbacks_;
            last_switch_reason_ = why;
            current_source_ = d.source;
            current_k_ = d.draft_k;
            dwell_steps_ = 1;
        } else {
            d.switched = false;
            current_k_ = d.draft_k;
            ++dwell_steps_;
        }
        if (d.source == CandidateSource::Normal)
            ++steps_in_normal_;
        else
            steps_in_normal_ = 0;
        return d;
    };

    // 1. Fixed modes
    if (cfg_.mode == DecodeStrategyMode::Normal || cfg_.max_draft <= 0) {
        return finalize(CandidateSource::Normal, 0, 1.0, normal_cost, normal_cost, "fixed_normal");
    }

    if (cfg_.mode == DecodeStrategyMode::Auto) {
        if (mtp_ok && avail.mtp_first_prob >= cfg_.mtp_min_prob) {
            const int k = std::min({cfg_.max_draft, avail.mtp_max_k, kMaxDraft});
            double et = 1.0, ems = normal_cost;
            const double c = estimate_strategy_cost(CandidateSource::Mtp, k, 0, avail.mtp_first_prob,
                                                    avail.mtp_available, avail.active_sessions, &et, &ems);
            return finalize(CandidateSource::Mtp, k, et, ems, c, "auto_mtp");
        }
        return finalize(CandidateSource::Normal, 0, 1.0, normal_cost, normal_cost,
                        mtp_ok ? "mtp_low_conf_fallback" : "auto_normal");
    }

    if (cfg_.mode == DecodeStrategyMode::Mtp) {
        if (mtp_ok && avail.mtp_first_prob >= cfg_.mtp_min_prob) {
            const int k = std::min({cfg_.max_draft, avail.mtp_max_k, kMaxDraft});
            double et = 1.0, ems = normal_cost;
            const double c = estimate_strategy_cost(CandidateSource::Mtp, k, 0, avail.mtp_first_prob,
                                                    avail.mtp_available, avail.active_sessions, &et, &ems);
            return finalize(CandidateSource::Mtp, k, et, ems, c, "fixed_mtp");
        }
        return finalize(CandidateSource::Normal, 0, 1.0, normal_cost, normal_cost, "mtp_unavailable_or_low_conf");
    }

    if (cfg_.mode == DecodeStrategyMode::Lookup) {
        const int k = std::min({cfg_.max_draft, avail.lookup_k, kMaxDraft});
        if (k > 0) {
            double et = 1.0, ems = normal_cost;
            const double c =
                estimate_strategy_cost(CandidateSource::Lookup, k, avail.lookup_match_len, (float) avail.lookup_freq,
                                       avail.mtp_available, avail.active_sessions, &et, &ems, avail.lookup_candidates);
            return finalize(CandidateSource::Lookup, k, et, ems, c, "fixed_lookup");
        }
        return finalize(CandidateSource::Normal, 0, 1.0, normal_cost, normal_cost, "lookup_no_match_fallback");
    }

    if (cfg_.mode == DecodeStrategyMode::Retrieval) {
        const int k = std::min({cfg_.max_draft, avail.retrieval_k, kMaxDraft});
        if (k > 0) {
            double et = 1.0, ems = normal_cost;
            const double c = estimate_strategy_cost(CandidateSource::Retrieval, k, avail.retrieval_match_len,
                                                    avail.retrieval_confidence, avail.mtp_available,
                                                    avail.active_sessions, &et, &ems, avail.retrieval_candidates);
            return finalize(CandidateSource::Retrieval, k, et, ems, c, "fixed_retrieval");
        }
        return finalize(CandidateSource::Normal, 0, 1.0, normal_cost, normal_cost, "retrieval_no_match_fallback");
    }

    // 2. Adaptive cost-aware selection across Normal, Mtp(1..K), Lookup(1..K), Retrieval(1..K)
    // Scale re-exploration interval: neural MTP recovers in 2*r (<=6) steps; pattern drafters recover in 3 steps when
    // session-proven and 16 steps when unproven/hostile.
    if (current_source_ == CandidateSource::Normal) {
        bool decayed_any = false;
        for (int s = 1; s < 4; ++s) {
            if (consecutive_rejects_[s] <= 0)
                continue;
            const bool proven = (strat_total_tok_[s] > strat_iters_[s] + 1);
            const int wait_steps =
                (s == (int) CandidateSource::Mtp) ? std::clamp(2 * consecutive_rejects_[s], 2, 6) : (proven ? 3 : 16);
            if (steps_in_normal_ >= wait_steps) {
                --consecutive_rejects_[s];
                decayed_any = true;
            }
        }
        if (decayed_any)
            steps_in_normal_ = 0;
    }

    CandidateSource best_src = CandidateSource::Normal;
    int best_k = 0;
    double best_cost = normal_cost;
    double best_et = 1.0;
    double best_ems = normal_cost;
    const char* best_reason = "normal_lowest_cost";

    bool cur_eligible = (current_source_ == CandidateSource::Normal);
    int cur_best_k = 0;
    double cur_best_cost = normal_cost;
    double cur_best_et = 1.0;
    double cur_best_ems = normal_cost;

    auto consider = [&](CandidateSource src, int max_avail_k, int match_len, float conf, int cands,
                        const char* reason_tag) {
        const int lim_k = std::min({cfg_.max_draft, max_avail_k, kMaxDraft});
        if (lim_k <= 0)
            return;
        for (int k = 1; k <= lim_k; ++k) {
            double et = 1.0, ems = normal_cost;
            const double c = estimate_strategy_cost(src, k, match_len, conf, avail.mtp_available, avail.active_sessions,
                                                    &et, &ems, cands);
            if (src == current_source_) {
                if (!cur_eligible || c < cur_best_cost) {
                    cur_eligible = true;
                    cur_best_k = k;
                    cur_best_cost = c;
                    cur_best_et = et;
                    cur_best_ems = ems;
                }
            }
            if (c < best_cost) {
                best_src = src;
                best_k = k;
                best_cost = c;
                best_et = et;
                best_ems = ems;
                best_reason = reason_tag;
            }
        }
    };

    if (mtp_ok && avail.mtp_first_prob >= cfg_.mtp_min_prob) {
        consider(CandidateSource::Mtp, avail.mtp_max_k, 0, avail.mtp_first_prob, 1, "adaptive_mtp_lowest_cost");
    }

    const int eff_lookup_k =
        (avail.lookup_consensus_k >= 0) ? std::min(avail.lookup_k, avail.lookup_consensus_k) : avail.lookup_k;
    if (eff_lookup_k > 0) {
        consider(CandidateSource::Lookup, eff_lookup_k, avail.lookup_match_len, (float) avail.lookup_freq,
                 avail.lookup_candidates, "adaptive_lookup_lowest_cost");
    }

    // Only consider Retrieval separately in Adaptive mode when it retrieves from the session corpus or finds a strictly
    // longer match than Lookup, avoiding redundant ping-ponging on identical active-stream matches.
    const bool retrieval_distinct =
        avail.retrieval_k > 0 &&
        (avail.retrieval_from_corpus || avail.retrieval_match_len > avail.lookup_match_len || eff_lookup_k == 0);
    if (retrieval_distinct) {
        consider(CandidateSource::Retrieval, avail.retrieval_k, avail.retrieval_match_len, avail.retrieval_confidence,
                 avail.retrieval_candidates, "adaptive_retrieval_lowest_cost");
    }

    double margin = effective_hysteresis_margin();
    if (avail.remaining_tokens > 0 && avail.remaining_tokens < 4) {
        margin = std::min(0.20, margin * 1.5);
    }

    // Require speculative candidates to beat Normal by the noise-calibrated margin when currently in Normal.
    if (best_src != CandidateSource::Normal && best_cost > normal_cost * (1.0 - margin)) {
        return finalize(CandidateSource::Normal, 0, 1.0, normal_cost, normal_cost, "spec_within_noise_of_normal");
    }

    // Hysteresis check when switching away from a working current strategy.
    if (has_started_ && best_src != current_source_ && cur_eligible) {
        const int cur_rej = consecutive_rejects_[(int) current_source_];
        const bool urgent_backoff = (current_source_ != CandidateSource::Normal && cur_rej >= 2);
        const bool high_conf_pattern =
            (best_src == CandidateSource::Lookup || best_src == CandidateSource::Retrieval) &&
            (avail.lookup_match_len >= 5 || avail.retrieval_match_len >= 5 ||
             (avail.lookup_freq >= 3 && eff_lookup_k >= 2));
        if (!urgent_backoff && !high_conf_pattern) {
            const bool significant_gain = (best_cost <= cur_best_cost * (1.0 - margin));
            if (!significant_gain || dwell_steps_ < cfg_.min_dwell_steps) {
                return finalize(current_source_, cur_best_k, cur_best_et, cur_best_ems, cur_best_cost,
                                "hysteresis_keep_current");
            }
        } else if (urgent_backoff && best_src == CandidateSource::Normal) {
            best_reason = "rejection_backoff_to_normal";
        }
    }

    return finalize(best_src, best_k, best_et, best_ems, best_cost, best_reason);
}

void AdaptiveDecodeScheduler::observe(const IterationObservation& obs) {
    const int k = std::clamp(obs.proposed_k, 0, kMaxDraft);
    const int a = std::clamp(obs.accepted_k, 0, k);
    const int T = std::clamp(k + 1, 1, kMaxWindowT);
    const int committed = std::max(1, obs.committed_tokens);
    const double ver_total_ms = std::max(0.0, obs.verify_ms + obs.commit_ms);
    const double iter_total_ms = std::max(0.001, obs.draft_ms + ver_total_ms + obs.scheduler_ms);
    const double decay = 1.0 - (1.0 / (double) std::max(4, cfg_.cost_window));

    // Track online prediction error and verification timing noise CV before updating posteriors.
    if (ver_obs_count_[T] > 0 && ver_decayed_rounds_[T] > 1e-6) {
        const double prev_ver_est = ver_decayed_ms_[T] / ver_decayed_rounds_[T];
        if (prev_ver_est > 0.1) {
            const double rel_dev = std::clamp(std::abs(ver_total_ms - prev_ver_est) / prev_ver_est, 0.0, 0.50);
            online_verify_cv_ = 0.85 * online_verify_cv_ + 0.15 * rel_dev;
        }
    }
    if (last_pred_ms_per_tok_ > 0.0) {
        const double actual_ms_per_tok = iter_total_ms / (double) committed;
        sum_abs_cost_err_ms_ += std::abs(actual_ms_per_tok - last_pred_ms_per_tok_);
        if (k > 0)
            sum_abs_acc_err_ += std::abs((double) a - (last_pred_exp_tok_ - 1.0));
        ++pred_eval_count_;
    }

    // 1. Update per-window-size verification cost and distinct MoE expert counts.
    ver_decayed_ms_[T] = ver_decayed_ms_[T] * decay + ver_total_ms;
    ver_decayed_rounds_[T] = ver_decayed_rounds_[T] * decay + 1.0;
    if (obs.distinct_experts > 0)
        ver_decayed_experts_[T] = ver_decayed_experts_[T] * decay + (double) obs.distinct_experts;
    ++ver_obs_count_[T];

    if (T == 1 && obs.cpu_expert_ms > 0.05 && obs.cpu_expert_ms < ver_total_ms) {
        const double frac = std::clamp(obs.cpu_expert_ms / ver_total_ms, 0.15, 0.90);
        live_cpu_fraction_ = (live_cpu_fraction_ < 0.0) ? frac : (0.80 * live_cpu_fraction_ + 0.20 * frac);
    }

    // Refine MoE distinct-expert ratio if both T=1 and T>1 have been observed.
    if (T > 1 && ver_obs_count_[1] > 0 && ver_decayed_experts_[1] > 1.0 && ver_decayed_experts_[T] > 1.0) {
        const double e1 = ver_decayed_experts_[1] / ver_decayed_rounds_[1];
        const double eT = ver_decayed_experts_[T] / ver_decayed_rounds_[T];
        if (e1 > 0.0)
            cal_.distinct_expert_ratio[T] =
                0.8 * cal_.distinct_expert_ratio[T] + 0.2 * std::clamp(eT / e1, 1.0, (double) T);
    }

    // 2. Update drafting overhead EMAs and Beta-Binomial conditional acceptance posteriors.
    const int sidx = (int) obs.source;
    if (obs.source == CandidateSource::Mtp && k > 0) {
        const double mtp_only_ms = std::max(0.0, obs.draft_ms - obs.lookup_probe_ms - obs.retrieval_probe_ms);
        if (mtp_only_ms > 0.0)
            mtp_step_ms_ = 0.85 * mtp_step_ms_ + 0.15 * (mtp_only_ms / (double) k);
        const int checked = std::min(k, a + 1);
        for (int i = 0; i < checked; ++i) {
            const double hit = (i < a) ? 1.0 : 0.0;
            mtp_succ_[i] = mtp_succ_[i] * decay + hit;
            mtp_trials_[i] = mtp_trials_[i] * decay + 1.0;
            mtp_cond_p_[i] = mtp_succ_[i] / std::max(1e-6, mtp_trials_[i]);
        }
    } else if (obs.source == CandidateSource::Lookup && k > 0) {
        if (obs.lookup_probe_ms > 0.0)
            lookup_ms_ = 0.85 * lookup_ms_ + 0.15 * obs.lookup_probe_ms;
        const int b = match_bucket(obs.match_len);
        const int checked = std::min(k, a + 1);
        lookup_acc_[b] = lookup_acc_[b] * decay + (double) a;
        lookup_prop_[b] = lookup_prop_[b] * decay + (double) checked;
        // Share n-gram acceptance evidence with Retrieval's same match-length bucket.
        retrieval_acc_[b] = retrieval_acc_[b] * decay + 0.5 * (double) a;
        retrieval_prop_[b] = retrieval_prop_[b] * decay + 0.5 * (double) checked;
    } else if (obs.source == CandidateSource::Retrieval && k > 0) {
        if (obs.retrieval_probe_ms > 0.0)
            retrieval_ms_ = 0.85 * retrieval_ms_ + 0.15 * obs.retrieval_probe_ms;
        const int b = match_bucket(obs.match_len);
        const int checked = std::min(k, a + 1);
        retrieval_acc_[b] = retrieval_acc_[b] * decay + (double) a;
        retrieval_prop_[b] = retrieval_prop_[b] * decay + (double) checked;
        lookup_acc_[b] = lookup_acc_[b] * decay + 0.5 * (double) a;
        lookup_prop_[b] = lookup_prop_[b] * decay + 0.5 * (double) checked;
    } else if (obs.source == CandidateSource::Normal) {
        const double catchup = std::max(0.0, obs.draft_ms - obs.lookup_probe_ms - obs.retrieval_probe_ms);
        if (catchup > 0.0)
            mtp_catchup_ms_ = 0.85 * mtp_catchup_ms_ + 0.15 * catchup;
    }

    // 3. Track consecutive rejections and cumulative time/token ratios per strategy.
    if (obs.source != CandidateSource::Normal && k > 0) {
        if (a == 0)
            ++consecutive_rejects_[sidx];
        else
            consecutive_rejects_[sidx] = 0;
    }

    strat_decayed_ms_[sidx] = strat_decayed_ms_[sidx] * decay + iter_total_ms;
    strat_decayed_tok_[sidx] = strat_decayed_tok_[sidx] * decay + (double) committed;
    strat_total_ms_[sidx] += iter_total_ms;
    strat_total_tok_[sidx] += committed;
    ++strat_iters_[sidx];

    // 4. Update global telemetry counters.
    ++total_iters_;
    total_proposed_ += k;
    total_accepted_ += a;
    total_committed_ += committed;
    total_iter_ms_ += iter_total_ms;
    if (obs.lookup_probe_ms > 0.0 || obs.retrieval_probe_ms > 0.0) {
        total_lookup_ms_ += obs.lookup_probe_ms;
        total_retrieval_ms_ += obs.retrieval_probe_ms;
        if (obs.source == CandidateSource::Mtp) {
            total_mtp_draft_ms_ += std::max(0.0, obs.draft_ms - obs.lookup_probe_ms - obs.retrieval_probe_ms);
        }
    } else {
        if (obs.source == CandidateSource::Mtp)
            total_mtp_draft_ms_ += obs.draft_ms;
        else if (obs.source == CandidateSource::Lookup)
            total_lookup_ms_ += obs.draft_ms;
        else if (obs.source == CandidateSource::Retrieval)
            total_retrieval_ms_ += obs.draft_ms;
    }
    total_verify_ms_ += obs.verify_ms;
    total_commit_ms_ += obs.commit_ms;
    total_scheduler_ms_ += obs.scheduler_ms;
}

SchedulerTelemetry AdaptiveDecodeScheduler::snapshot() const {
    SchedulerTelemetry t;
    t.mode_name = decode_strategy_mode_to_string(cfg_.mode);
    t.initial_strategy = candidate_source_to_string(initial_source_);
    t.final_strategy = candidate_source_to_string(current_source_);
    t.last_switch_reason = last_switch_reason_ ? last_switch_reason_ : "none";

    t.iterations = total_iters_;
    t.normal_iterations = strat_iters_[(int) CandidateSource::Normal];
    t.mtp_iterations = strat_iters_[(int) CandidateSource::Mtp];
    t.lookup_iterations = strat_iters_[(int) CandidateSource::Lookup];
    t.retrieval_iterations = strat_iters_[(int) CandidateSource::Retrieval];

    t.total_proposed_drafts = total_proposed_;
    t.total_accepted_drafts = total_accepted_;
    t.total_committed_tokens = total_committed_;

    const int64_t spec_iters = t.mtp_iterations + t.lookup_iterations + t.retrieval_iterations;
    t.avg_proposed_draft_len = spec_iters > 0 ? (double) total_proposed_ / (double) spec_iters : 0.0;
    t.avg_accepted_prefix_len = spec_iters > 0 ? (double) total_accepted_ / (double) spec_iters : 0.0;
    t.acceptance_rate = total_proposed_ > 0 ? (double) total_accepted_ / (double) total_proposed_ : 0.0;

    t.total_iteration_ms = total_iter_ms_;
    t.total_draft_ms = total_mtp_draft_ms_;
    t.total_lookup_ms = total_lookup_ms_;
    t.total_retrieval_ms = total_retrieval_ms_;
    t.total_verify_ms = total_verify_ms_;
    t.total_commit_ms = total_commit_ms_;
    t.total_scheduler_ms = total_scheduler_ms_;

    t.measured_ms_per_token = total_committed_ > 0 ? (total_iter_ms_ / (double) total_committed_) : 0.0;
    t.estimated_normal_ms_per_token = estimate_verify_ms(1, 1);
    t.estimated_speedup_vs_normal =
        (t.measured_ms_per_token > 1e-6) ? (t.estimated_normal_ms_per_token / t.measured_ms_per_token) : 1.0;

    t.mean_abs_cost_pred_error_ms = pred_eval_count_ > 0 ? (sum_abs_cost_err_ms_ / (double) pred_eval_count_) : 0.0;
    t.mean_abs_acc_pred_error = pred_eval_count_ > 0 ? (sum_abs_acc_err_ / (double) pred_eval_count_) : 0.0;
    t.effective_hysteresis_margin = effective_hysteresis_margin();
    t.measured_verify_cv = online_verify_cv_;

    t.strategy_switches = switches_;
    t.fallback_to_normal_count = fallbacks_;
    return t;
}

} // namespace guild::spec
