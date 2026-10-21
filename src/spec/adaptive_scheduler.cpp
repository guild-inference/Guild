// src/spec/adaptive_scheduler.cpp - see include/guild/spec/adaptive_scheduler.hpp.
#include "guild/spec/adaptive_scheduler.hpp"

#include <algorithm>
#include <cmath>

namespace guild::spec {

namespace {
constexpr double kDefaultMtpCondP[AdaptiveDecodeScheduler::kMaxDraft] = {0.78, 0.74, 0.70, 0.66, 0.62, 0.58, 0.54};
constexpr double kPriorLookupQ[AdaptiveDecodeScheduler::kMatchBuckets] = {0.32, 0.78, 0.90, 0.95};
constexpr double kPriorRetrievalQ[AdaptiveDecodeScheduler::kMatchBuckets] = {0.32, 0.75, 0.88, 0.94};
constexpr double kPriorWeight = 3.0;
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
    cfg_.reexplore_interval = std::max(4, cfg_.reexplore_interval);
}

void AdaptiveDecodeScheduler::set_calibration(const GExecSpecCalibration& cal) {
    cal_ = cal;
    for (int t = 1; t <= kMaxWindowT; ++t) {
        if (cal_.verify_window_ms[t] <= 0.0)
            cal_.verify_window_ms[t] = 20.0 + 7.6 * (double) (t - 1);
        if (cal_.distinct_expert_ratio[t] <= 0.0)
            cal_.distinct_expert_ratio[t] = 1.0 + 0.65 * (double) (t - 1);
    }
    mtp_step_ms_ = std::max(0.05, cal_.mtp_draft_step_ms);
    mtp_catchup_ms_ = std::max(0.01, cal_.mtp_kv_catchup_ms);
    lookup_ms_ = std::max(0.001, cal_.lookup_draft_ms);
    retrieval_ms_ = std::max(0.001, cal_.retrieval_draft_ms);
}

void AdaptiveDecodeScheduler::reset_session_stats() {
    for (int t = 0; t <= kMaxWindowT; ++t) {
        ver_decayed_ms_[t] = 0.0;
        ver_decayed_rounds_[t] = 0.0;
        ver_decayed_experts_[t] = 0.0;
        ver_obs_count_[t] = 0;
    }
    mtp_step_ms_ = std::max(0.05, cal_.mtp_draft_step_ms);
    mtp_catchup_ms_ = std::max(0.01, cal_.mtp_kv_catchup_ms);
    lookup_ms_ = std::max(0.001, cal_.lookup_draft_ms);
    retrieval_ms_ = std::max(0.001, cal_.retrieval_draft_ms);

    for (int i = 0; i < kMaxDraft; ++i)
        mtp_cond_p_[i] = kDefaultMtpCondP[i];

    for (int b = 0; b < kMatchBuckets; ++b) {
        lookup_acc_[b] = kPriorLookupQ[b] * kPriorWeight;
        lookup_prop_[b] = kPriorWeight;
        retrieval_acc_[b] = kPriorRetrievalQ[b] * kPriorWeight;
        retrieval_prop_[b] = kPriorWeight;
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

void AdaptiveDecodeScheduler::seed_single_token_verify(double verify_ms, int64_t distinct_experts) {
    if (verify_ms <= 0.05)
        return;
    ver_decayed_ms_[1] = verify_ms;
    ver_decayed_rounds_[1] = 1.0;
    if (distinct_experts > 0)
        ver_decayed_experts_[1] = (double) distinct_experts;
    ver_obs_count_[1] = 1;
}

double AdaptiveDecodeScheduler::estimate_verify_ms(int T, int active_sessions) const {
    const int tc = std::clamp(T, 1, kMaxWindowT);
    double est = cal_.verify_window_ms[tc];

    if (ver_obs_count_[tc] > 0 && ver_decayed_rounds_[tc] > 1e-9) {
        est = ver_decayed_ms_[tc] / ver_decayed_rounds_[tc];
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
        const double conc_factor = 1.0 + 0.15 * (double) (active_sessions - 1);
        est = t1 + extra * conc_factor;
    }
    return std::max(0.05, est);
}

double AdaptiveDecodeScheduler::estimate_strategy_cost(CandidateSource src, int k, int match_len, float confidence,
                                                       bool mtp_active_in_model, int active_sessions,
                                                       double* out_expected_tokens,
                                                       double* out_expected_iter_ms) const {
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
    const double rej_penalty = (rej > 0) ? std::pow(0.68, (double) std::min(rej, 6)) : 1.0;

    if (src == CandidateSource::Mtp) {
        double reach = 1.0;
        exp_tok = 1.0;
        for (int i = 0; i < kc; ++i) {
            double p = mtp_cond_p_[i] * rej_penalty;
            if (i == 0 && confidence > 0.0f && confidence < 1.0f)
                p = 0.65 * p + 0.35 * (double) confidence * rej_penalty;
            reach *= std::clamp(p, 0.01, 0.99);
            exp_tok += reach;
        }
        exp_ms = estimate_verify_ms(kc + 1, active_sessions) + (double) kc * mtp_step_ms_;
    } else if (src == CandidateSource::Lookup) {
        const int b = match_bucket(match_len);
        double q = lookup_acc_[b] / std::max(1e-6, lookup_prop_[b]);
        if (confidence > 1.0f) { // lookup_freq > 1
            const double freq_boost = (b == 0) ? 0.22 : 0.04;
            q = std::min(0.98, q + freq_boost * std::min(2.0, (double) (confidence - 1.0f)));
        }
        q = std::clamp(q * rej_penalty, 0.01, 0.99);
        double reach = 1.0;
        exp_tok = 1.0;
        for (int i = 0; i < kc; ++i) {
            reach *= q;
            exp_tok += reach;
        }
        exp_ms = estimate_verify_ms(kc + 1, active_sessions) + lookup_ms_ +
                 (mtp_active_in_model ? mtp_catchup_ms_ : 0.0);
    } else if (src == CandidateSource::Retrieval) {
        const int b = match_bucket(match_len);
        double q = retrieval_acc_[b] / std::max(1e-6, retrieval_prop_[b]);
        const double conf_scale = std::clamp(0.75 + 0.25 * (double) confidence, 0.50, 1.05);
        q = std::clamp(q * conf_scale * rej_penalty, 0.01, 0.99);
        double reach = 1.0;
        exp_tok = 1.0;
        for (int i = 0; i < kc; ++i) {
            reach *= q;
            exp_tok += reach;
        }
        exp_ms = estimate_verify_ms(kc + 1, active_sessions) + retrieval_ms_ +
                 (mtp_active_in_model ? mtp_catchup_ms_ : 0.0);
    }

    double cost = exp_ms / std::max(1.0, exp_tok);

    // Blend with realized cumulative time-to-output ratio when empirical data shows higher cost than the analytical
    // acceptance model. This guarantees an optimistic acceptance model cannot mask high verification cost.
    if (strat_iters_[sidx] >= 2 && strat_decayed_tok_[sidx] >= 1.0) {
        const double emp_ratio = strat_decayed_ms_[sidx] / strat_decayed_tok_[sidx];
        if (emp_ratio > cost) {
            const double blend = std::min(0.65, 0.15 * (double) strat_iters_[sidx]);
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
            const double c = estimate_strategy_cost(CandidateSource::Lookup, k, avail.lookup_match_len,
                                                    (float) avail.lookup_freq, avail.mtp_available,
                                                    avail.active_sessions, &et, &ems);
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
                                                    avail.active_sessions, &et, &ems);
            return finalize(CandidateSource::Retrieval, k, et, ems, c, "fixed_retrieval");
        }
        return finalize(CandidateSource::Normal, 0, 1.0, normal_cost, normal_cost, "retrieval_no_match_fallback");
    }

    // 2. Adaptive cost-aware selection across Normal, Mtp(1..K), Lookup(1..K), Retrieval(1..K)
    // Gentle decay of consecutive rejection penalty after prolonged stay in Normal so recovery is possible.
    if (current_source_ == CandidateSource::Normal && steps_in_normal_ >= cfg_.reexplore_interval) {
        for (int s = 1; s < 4; ++s) {
            if (consecutive_rejects_[s] > 0)
                --consecutive_rejects_[s];
        }
        steps_in_normal_ = 0;
    }

    CandidateSource best_src = CandidateSource::Normal;
    int best_k = 0;
    double best_cost = normal_cost;
    double best_et = 1.0;
    double best_ems = normal_cost;
    const char* best_reason = "normal_lowest_cost";

    // Also track current strategy's best available (k, cost) if current_source_ is still eligible.
    bool cur_eligible = (current_source_ == CandidateSource::Normal);
    int cur_best_k = 0;
    double cur_best_cost = normal_cost;
    double cur_best_et = 1.0;
    double cur_best_ems = normal_cost;

    auto consider = [&](CandidateSource src, int max_avail_k, int match_len, float conf, const char* reason_tag) {
        const int lim_k = std::min({cfg_.max_draft, max_avail_k, kMaxDraft});
        if (lim_k <= 0)
            return;
        for (int k = 1; k <= lim_k; ++k) {
            double et = 1.0, ems = normal_cost;
            const double c = estimate_strategy_cost(src, k, match_len, conf, avail.mtp_available, avail.active_sessions,
                                                    &et, &ems);
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
        consider(CandidateSource::Mtp, avail.mtp_max_k, 0, avail.mtp_first_prob, "adaptive_mtp_lowest_cost");
    }
    if (avail.lookup_k > 0) {
        consider(CandidateSource::Lookup, avail.lookup_k, avail.lookup_match_len, (float) avail.lookup_freq,
                 "adaptive_lookup_lowest_cost");
    }
    if (avail.retrieval_k > 0) {
        consider(CandidateSource::Retrieval, avail.retrieval_k, avail.retrieval_match_len, avail.retrieval_confidence,
                 "adaptive_retrieval_lowest_cost");
    }

    // Require speculative candidates to beat Normal by hysteresis_margin when currently in Normal.
    if (best_src != CandidateSource::Normal && best_cost > normal_cost * (1.0 - cfg_.hysteresis_margin)) {
        return finalize(CandidateSource::Normal, 0, 1.0, normal_cost, normal_cost, "spec_within_noise_of_normal");
    }

    // Hysteresis check when switching away from a working current strategy.
    if (has_started_ && best_src != current_source_ && cur_eligible) {
        const int cur_rej = consecutive_rejects_[(int) current_source_];
        const bool urgent_backoff = (current_source_ != CandidateSource::Normal && cur_rej >= 2);
        if (!urgent_backoff) {
            const bool significant_gain = (best_cost <= cur_best_cost * (1.0 - cfg_.hysteresis_margin));
            if (!significant_gain || dwell_steps_ < cfg_.min_dwell_steps) {
                return finalize(current_source_, cur_best_k, cur_best_et, cur_best_ems, cur_best_cost,
                                "hysteresis_keep_current");
            }
        } else if (best_src == CandidateSource::Normal) {
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

    // 1. Update per-window-size verification cost and distinct MoE expert counts.
    ver_decayed_ms_[T] = ver_decayed_ms_[T] * decay + ver_total_ms;
    ver_decayed_rounds_[T] = ver_decayed_rounds_[T] * decay + 1.0;
    if (obs.distinct_experts > 0)
        ver_decayed_experts_[T] = ver_decayed_experts_[T] * decay + (double) obs.distinct_experts;
    ++ver_obs_count_[T];

    // Refine MoE distinct-expert ratio if both T=1 and T>1 have been observed.
    if (T > 1 && ver_obs_count_[1] > 0 && ver_decayed_experts_[1] > 1.0 && ver_decayed_experts_[T] > 1.0) {
        const double e1 = ver_decayed_experts_[1] / ver_decayed_rounds_[1];
        const double eT = ver_decayed_experts_[T] / ver_decayed_rounds_[T];
        if (e1 > 0.0)
            cal_.distinct_expert_ratio[T] = 0.8 * cal_.distinct_expert_ratio[T] + 0.2 * std::clamp(eT / e1, 1.0, (double) T);
    }

    // 2. Update drafting overhead EMAs and conditional acceptance rates.
    const int sidx = (int) obs.source;
    if (obs.source == CandidateSource::Mtp && k > 0) {
        if (obs.draft_ms > 0.0)
            mtp_step_ms_ = 0.85 * mtp_step_ms_ + 0.15 * (obs.draft_ms / (double) k);
        const double alpha = 0.12;
        const int checked = std::min(k, a + 1);
        for (int i = 0; i < checked; ++i) {
            const double hit = (i < a) ? 1.0 : 0.0;
            mtp_cond_p_[i] = (1.0 - alpha) * mtp_cond_p_[i] + alpha * hit;
        }
    } else if (obs.source == CandidateSource::Lookup && k > 0) {
        if (obs.draft_ms > 0.0)
            lookup_ms_ = 0.85 * lookup_ms_ + 0.15 * obs.draft_ms;
        const int b = match_bucket(obs.match_len);
        const int checked = std::min(k, a + 1);
        lookup_acc_[b] = lookup_acc_[b] * decay + (double) a;
        lookup_prop_[b] = lookup_prop_[b] * decay + (double) checked;
    } else if (obs.source == CandidateSource::Retrieval && k > 0) {
        if (obs.draft_ms > 0.0)
            retrieval_ms_ = 0.85 * retrieval_ms_ + 0.15 * obs.draft_ms;
        const int b = match_bucket(obs.match_len);
        const int checked = std::min(k, a + 1);
        retrieval_acc_[b] = retrieval_acc_[b] * decay + (double) a;
        retrieval_prop_[b] = retrieval_prop_[b] * decay + (double) checked;
    } else if (obs.source == CandidateSource::Normal) {
        if (obs.draft_ms > 0.0)
            mtp_catchup_ms_ = 0.85 * mtp_catchup_ms_ + 0.15 * obs.draft_ms;
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

    t.strategy_switches = switches_;
    t.fallback_to_normal_count = fallbacks_;
    return t;
}

} // namespace guild::spec
