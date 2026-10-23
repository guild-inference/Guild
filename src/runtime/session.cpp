#include "guild/runtime/session.hpp"
#include "guild/runtime/model.hpp"

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "model_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <vector>

// GuildSession is the per-request handle.  What is per-session lives in Impl below (the speculative drafting state,
// the cancellation flag); everything else - weights, expert tiers, KV staging, the verifier, the prefill engine - is
// owned by the model and shared.  Generation is serialized by the model-wide `generation_mutex`, so a process runs
// one generation at a time and a session only selects which state that generation uses.
//
// Two invariants hold for every entry point:
//   * `model_impl->ready` is fail-closed.  Starting a generation clears and re-checks it, and any inference-time
//     failure leaves it false so later requests are refused instead of running on a half-valid model.
//   * Session lifetime pins what it uses.  The runner and kernel leases taken in the constructor keep the packages
//     the model was loaded with from being disabled or unloaded while this session exists.

namespace guild::runtime {

struct GuildSession::Impl {
    std::shared_ptr<GuildModel::Impl> model_impl;
    GuildModel* model = nullptr;
    int64_t max_context = 262144;
    std::unique_ptr<spec::SuffixDrafter> sfx;
    std::unique_ptr<spec::RetrievalDrafter> retrieval;
    std::unique_ptr<spec::AdaptiveDecodeScheduler> scheduler;
    std::unique_ptr<spec::DraftPolicy> policy;
    std::mutex spec_mutex;
    model::SessionRunnerLease runner_lease;
    kernels::SessionKernelLease kernel_lease;
    std::atomic<bool> cancelled{false};
};

GuildSession::GuildSession(GuildModel* model, int64_t max_context)
    : model_(model),
      max_context_(max_context),
      impl_(std::make_unique<Impl>()) {
    impl_->model = model;
    impl_->model_impl = model ? model->impl_ : nullptr;
    auto* m_impl = impl_->model_impl.get();
    if (m_impl)
        m_impl->active_sessions.fetch_add(1);
    impl_->max_context = max_context > 0 ? max_context : (m_impl ? m_impl->config.max_context : 262144);
    const size_t lookup_win =
        (size_t) std::clamp<int64_t>(m_impl ? m_impl->config.spec_lookup_window : 65536, 256, 1 << 20);
    // Speculative state is sized for this session's context; the leases are re-taken here (not inherited) so that
    // a session created after the model was loaded is just as protected against a package swap as the model is.
    impl_->sfx = std::make_unique<spec::SuffixDrafter>(3, 64, lookup_win);
    impl_->sfx->set_max_history(lookup_win);
    impl_->retrieval = std::make_unique<spec::RetrievalDrafter>(3, 64, lookup_win, lookup_win);
    spec::SchedulerConfig scfg;
    if (m_impl) {
        spec::parse_decode_strategy_mode(m_impl->config.decode_strategy, scfg.mode);
        scfg.max_draft = std::max(0, m_impl->config.spec - 1);
        scfg.cost_window = std::max(4, m_impl->config.spec_cost_window);
        scfg.mtp_min_prob = m_impl->config.spec_min_p;
        impl_->scheduler = std::make_unique<spec::AdaptiveDecodeScheduler>(scfg, m_impl->spec_calibration);
    } else {
        impl_->scheduler = std::make_unique<spec::AdaptiveDecodeScheduler>(scfg);
    }
    impl_->policy = std::make_unique<spec::DraftPolicy>(m_impl ? m_impl->config.spec : 2);
    impl_->runner_lease = model::SessionRunnerLease(
        &model::ModelRunnerRegistry::instance(),
        (m_impl && m_impl->runner && m_impl->runner->package_id()) ? m_impl->runner->package_id() : "");
    impl_->kernel_lease = kernels::SessionKernelLease(
        &kernels::KernelRegistry::instance(), m_impl ? m_impl->kernel_lease.package_ids() : std::vector<std::string>{});
}

GuildSession::~GuildSession() {
    if (impl_ && impl_->model_impl)
        impl_->model_impl->active_sessions.fetch_sub(1);
}

void GuildSession::reset() {
    if (!impl_)
        return;
    {
        std::lock_guard<std::mutex> slock(impl_->spec_mutex);
        if (impl_->sfx)
            impl_->sfx->reset();
        if (impl_->retrieval)
            impl_->retrieval->reset_all();
        if (impl_->scheduler)
            impl_->scheduler->reset_session_stats();
    }
    if (!impl_->model_impl)
        return;
    auto* m_impl = impl_->model_impl.get();
    // The model-wide generation lock: reset rewrites state (KV, MTP, cache counters) that an in-flight generation
    // on another session would be reading.
    std::lock_guard<std::mutex> lock(m_impl->generation_mutex);
    if (!m_impl->ready || !m_impl->ready->load())
        return;
    core::ScopedModelExecution exec_scope(m_impl->exec_state);
    std::optional<kernels::ScopedCudaAttn> attn_scope;
    if (m_impl->cuda_attn)
        attn_scope.emplace(*m_impl->cuda_attn);
    std::optional<kernels::ScopedCudaMoE> moe_scope;
    if (m_impl->cuda_moe)
        moe_scope.emplace(*m_impl->cuda_moe);
    std::string error;
    if (!m_impl->ver->wait_commit(error) || (m_impl->mtp && !m_impl->mtp->idle(error))) {
        m_impl->ready->store(false);
        return;
    }
    core::session_zero(m_impl->ss, m_impl->g, nullptr, m_impl->main_stream);
    if (m_impl->mtp)
        core::qsa_state_zero(m_impl->mtp->kv_state(), m_impl->g, m_impl->main_stream);
    if (cudaStreamSynchronize(m_impl->main_stream) != cudaSuccess)
        m_impl->ready->store(false);
}

void GuildSession::ingest_retrieval_tokens(const int32_t* ids, size_t n) {
    if (!impl_ || !impl_->retrieval || ids == nullptr || n == 0)
        return;
    std::lock_guard<std::mutex> slock(impl_->spec_mutex);
    impl_->retrieval->ingest_corpus(ids, n);
}

void GuildSession::clear_retrieval_corpus() {
    if (!impl_ || !impl_->retrieval)
        return;
    std::lock_guard<std::mutex> slock(impl_->spec_mutex);
    impl_->retrieval->clear_corpus();
}

size_t GuildSession::retrieval_corpus_tokens() const {
    if (!impl_ || !impl_->retrieval)
        return 0;
    std::lock_guard<std::mutex> slock(impl_->spec_mutex);
    return impl_->retrieval->corpus_size();
}

// Lock-free on purpose: cancel is called from a signal handler path and from the server's shutdown thread while a
// generation is blocked inside the prefill/decode loops, which poll this flag.
void GuildSession::cancel() {
    if (impl_) {
        impl_->cancelled.store(true);
    }
}

// One generation, start to finish.  The shape is: validate the request against the model's real limits, take the
// scopes that pin the CUDA bridges for the duration, reset the KV/MTP state, run prefill, then decode until EOS,
// `max_new_tokens`, cancellation or a runtime failure.  Any failure after the first device work marks the model
// not-ready (see `runtime_failure`) - continuing on a model whose commit path already failed would produce garbage.
bool GuildSession::generate(const GenerationRequest& req, TokenStreamCallback on_token, RuntimeTelemetry& telemetry,
                            std::string& error_msg, PrefillProgressCallback on_prefill) try {
    telemetry = RuntimeTelemetry{};
    telemetry.finish_reason = "error";
    error_msg.clear();
    if (!impl_ || !impl_->model_impl) {
        error_msg = "session uninitialized";
        return false;
    }

    auto* m_impl = impl_->model_impl.get();
    std::lock_guard<std::mutex> lock(m_impl->generation_mutex);
    if (!m_impl->ready || !m_impl->ready->load()) {
        error_msg = "model is not ready or an earlier inference failed; reload the model";
        return false;
    }
    auto runtime_failure = [&]() {
        m_impl->ready->store(false);
        return false;
    };
    if (m_impl->ple_required && !m_impl->ss.ple.ready()) {
        error_msg = "required PLE operation is no longer connected; reload the model";
        return runtime_failure();
    }

    impl_->cancelled.store(false);

    // Resolve prompt tokens
    std::vector<int64_t> ids;
    if (!req.prompt_tokens.empty()) {
        ids.assign(req.prompt_tokens.begin(), req.prompt_tokens.end());
    } else if (!req.prompt.empty()) {
        std::vector<int32_t> enc_ids;
        if (!m_impl->tokenizer || !m_impl->tokenizer->encode(req.prompt, enc_ids, true, error_msg)) {
            return false;
        }
        ids.assign(enc_ids.begin(), enc_ids.end());
    }

    if (ids.empty()) {
        error_msg = "prompt cannot be empty";
        return false;
    }

    const int64_t n_prompt = static_cast<int64_t>(ids.size());
    if (req.max_new_tokens <= 0 || !std::isfinite(req.temperature) || req.temperature < 0 ||
        !std::isfinite(req.top_p) || req.top_p <= 0 || req.top_p > 1 || req.top_k < 0) {
        error_msg = "invalid generation length or sampling parameters";
        return false;
    }
    if (req.max_new_tokens > impl_->max_context || n_prompt > impl_->max_context - req.max_new_tokens) {
        error_msg = "prompt (" + std::to_string(n_prompt) + ") + max_tokens (" + std::to_string(req.max_new_tokens) +
                    ") exceeds max context (" + std::to_string(impl_->max_context) + ")";
        return false;
    }
    for (int64_t id : ids) {
        if (id < 0 || id >= m_impl->ver->vocab()) {
            error_msg = "prompt token id " + std::to_string(id) + " is outside the model vocabulary";
            return false;
        }
    }

    if (m_impl->g.is_dense_attention() && n_prompt + req.max_new_tokens > m_impl->ver->attention_capacity()) {
        error_msg = "dense attention request exceeds the allocated single-tile history capacity; multi-tile execution "
                    "is unavailable";
        return false;
    }

    struct RequestAffinity {
        kernels::cpu::ThreadAffinity previous;
        ~RequestAffinity() {
            kernels::cpu::restore_thread_affinity(previous);
        }
    } affinity;
    // Pin the calling thread to the pool's host core for the duration of the request (GUILD_HOST_AFFINITY=0 opts
    // out).  The expert workers run on dedicated cores; leaving the request thread free to migrate between sockets
    // makes the host-side marshalling unpredictable and steals cycles from the pool.
    const char* host_affinity = std::getenv("GUILD_HOST_AFFINITY");
    if (!host_affinity || std::atoi(host_affinity) != 0)
        affinity.previous = kernels::cpu::pin_current_thread_if_allowed(m_impl->pool->host_core());
    core::ScopedModelExecution exec_scope(m_impl->exec_state);
    std::optional<kernels::ScopedCudaDense> dense_scope;
    if (m_impl->cuda_dense)
        dense_scope.emplace(*m_impl->cuda_dense);
    std::optional<kernels::ScopedCudaAttn> attn_scope;
    if (m_impl->cuda_attn)
        attn_scope.emplace(*m_impl->cuda_attn);
    std::optional<kernels::ScopedCudaMoE> moe_scope;
    if (m_impl->cuda_moe)
        moe_scope.emplace(*m_impl->cuda_moe);

    // Reset KV state for fresh generation
    if (!m_impl->ver->wait_commit(error_msg))
        return runtime_failure();
    if (m_impl->mtp && !m_impl->mtp->idle(error_msg))
        return runtime_failure();
    core::session_zero(m_impl->ss, m_impl->g, nullptr, m_impl->main_stream);
    if (m_impl->mtp)
        core::qsa_state_zero(m_impl->mtp->kv_state(), m_impl->g, m_impl->main_stream);
    if (cudaStreamSynchronize(m_impl->main_stream) != cudaSuccess) {
        error_msg = "failed to reset inference state";
        return runtime_failure();
    }

    // Resolve speculative decoding strategy and maximum draft depth for this request
    spec::DecodeStrategyMode eff_mode = spec::DecodeStrategyMode::Auto;
    spec::parse_decode_strategy_mode(m_impl->config.decode_strategy, eff_mode);
    if (!req.decode_strategy.empty() && !spec::parse_decode_strategy_mode(req.decode_strategy, eff_mode)) {
        error_msg = "invalid request decode_strategy '" + req.decode_strategy + "'";
        return false;
    }

    const int S = std::max(1, m_impl->config.spec);
    int max_draft = 0;
    if (req.spec_max_draft >= 0) {
        max_draft = std::min(req.spec_max_draft, S - 1);
    } else if (m_impl->config.spec_max_draft > 0) {
        max_draft = std::min(m_impl->config.spec_max_draft, S - 1);
    } else if (eff_mode == spec::DecodeStrategyMode::Auto) {
        max_draft = (m_impl->plan.mtp_spec_tokens > 0 && m_impl->mtp != nullptr) ? (S - 1) : 0;
    } else if (eff_mode == spec::DecodeStrategyMode::Normal) {
        max_draft = 0;
    } else {
        max_draft = S - 1;
    }

    const bool use_mtp = (m_impl->mtp != nullptr) && (max_draft > 0) &&
                         (eff_mode == spec::DecodeStrategyMode::Auto || eff_mode == spec::DecodeStrategyMode::Mtp ||
                          eff_mode == spec::DecodeStrategyMode::Adaptive);
    const bool need_lookup = (max_draft > 0) && (eff_mode == spec::DecodeStrategyMode::Lookup ||
                                                 eff_mode == spec::DecodeStrategyMode::Adaptive);
    const bool need_retrieval = (max_draft > 0) && (eff_mode == spec::DecodeStrategyMode::Retrieval ||
                                                    eff_mode == spec::DecodeStrategyMode::Adaptive);

    {
        std::lock_guard<std::mutex> slock(impl_->spec_mutex);
        spec::SchedulerConfig scfg;
        scfg.mode = eff_mode;
        scfg.max_draft = max_draft;
        scfg.cost_window = std::max(4, m_impl->config.spec_cost_window);
        scfg.mtp_min_prob = m_impl->config.spec_min_p;
        impl_->scheduler->configure(scfg);
        impl_->scheduler->reset_session_stats();

        if (impl_->retrieval)
            impl_->retrieval->reset_stream();
        if (impl_->sfx)
            impl_->sfx->reset();

        if (need_lookup || need_retrieval || (impl_->retrieval && impl_->retrieval->corpus_size() > 0)) {
            std::vector<int32_t> prompt_i32(ids.size());
            for (size_t i = 0; i < ids.size(); ++i)
                prompt_i32[i] = static_cast<int32_t>(ids[i]);
            if (need_lookup && impl_->sfx)
                impl_->sfx->append(prompt_i32.data(), prompt_i32.size());
            if ((need_retrieval || (impl_->retrieval && impl_->retrieval->corpus_size() > 0)) && impl_->retrieval)
                impl_->retrieval->append_stream(prompt_i32.data(), prompt_i32.size());
        }
    }

    // Reset prefill stats
    auto t_req_start = Clock::now();
    auto t_prefill_start = Clock::now();

    if (use_mtp)
        m_impl->mtp->set_prompt_len(n_prompt);
    if (m_impl->prefill) {
        m_impl->prefill->should_stop = [this, &req] {
            return impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load());
        };

        if (use_mtp) {
            m_impl->prefill->on_chunk = [this, m_impl, &ids](const float* R_rows, int64_t T, int64_t p0,
                                                             std::string& e) -> bool {
                std::vector<int32_t> next_tokens(static_cast<size_t>(T));
                for (int64_t i = 0; i < T; ++i) {
                    if (p0 + i + 1 < static_cast<int64_t>(ids.size())) {
                        next_tokens[static_cast<size_t>(i)] =
                            static_cast<int32_t>(ids[static_cast<size_t>(p0 + i + 1)]);
                    }
                }
                if (m_impl->prefill->draft_kv(*m_impl->mtp, R_rows, next_tokens.data(), T, p0, e))
                    return true;
                return e.empty() && m_impl->mtp->prefill(R_rows, next_tokens.data(), T, p0, e);
            };
        }

        // Prefill prompt tokens [0, n_prompt - 1)
        if (n_prompt > 1) {
            if (!m_impl->prefill->run(ids.data(), n_prompt - 1, 0, error_msg)) {
                if (error_msg == "cancelled") {
                    telemetry.finish_reason = "cancel";
                    return true;
                }
                return runtime_failure();
            }
        }
    } else if (m_impl->batched_prefill && std::getenv("GUILD_PREFILL_SEQUENTIAL") == nullptr) {
        m_impl->batched_prefill->should_stop = [this, &req] {
            return impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load());
        };
        if (n_prompt > 1) {
            if (!m_impl->batched_prefill->run(ids.data(), n_prompt - 1, 0, error_msg)) {
                if (error_msg == "cancelled") {
                    telemetry.finish_reason = "cancel";
                    return true;
                }
                return runtime_failure();
            }
        }
    } else {
        // Architecture-independent prompt ingestion via Verifier (sequential fallback)
        if (n_prompt > 1) {
            std::vector<int32_t> sink_out(1);
            for (int64_t pos = 0; pos < n_prompt - 1; ++pos) {
                if (impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load())) {
                    telemetry.finish_reason = "cancel";
                    return true;
                }
                int32_t prompt_tok = static_cast<int32_t>(ids[static_cast<size_t>(pos)]);
                m_impl->drive.d.layers = 0;
                m_impl->drive.d.experts = 0;
                if (!m_impl->ver->run(1, &prompt_tok, pos, drive_pool_multi, &m_impl->drive, sink_out.data(),
                                      error_msg)) {
                    return runtime_failure();
                }
                if (!m_impl->ver->commit(1, error_msg)) {
                    return runtime_failure();
                }
                if (use_mtp) {
                    const int32_t next = int32_t(ids[size_t(pos + 1)]);
                    if (!m_impl->mtp->prefill(m_impl->ver->final_R_all(), &next, 1, pos, error_msg))
                        return runtime_failure();
                }
            }
        }
    }

    auto t_prefill_end = Clock::now();
    telemetry.prompt_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    telemetry.prompt_tokens = static_cast<int>(n_prompt);
    telemetry.prompt_tok_s =
        (n_prompt > 1 && telemetry.prompt_ms > 0) ? ((n_prompt - 1) / (telemetry.prompt_ms / 1000.0)) : 0.0;
    if (m_impl->batched_prefill && std::getenv("GUILD_PREFILL_SEQUENTIAL") == nullptr) {
        const auto& st = m_impl->batched_prefill->stats();
        telemetry.prefill_embed_ms = st.ms_embed;
        telemetry.prefill_gpu_dense_ms = st.gpu_dense_ms;
        telemetry.prefill_gpu_attn_ms = st.gpu_attention_ms;
        telemetry.prefill_gpu_gdn_ms = st.gpu_gdn_ms;
        telemetry.prefill_host_wait_ms = st.host_wait_ms;
        telemetry.prefill_cpu_group_ms = st.cpu_group_ms;
        telemetry.prefill_cpu_quant_ms = st.ms_cpu_quant;
        telemetry.prefill_cpu_gu_ms = st.ms_cpu_gu;
        telemetry.prefill_cpu_down_ms = st.ms_cpu_down;
    }

    if (on_prefill) {
        PrefillProgressEvent pev;
        pev.pos = n_prompt - 1;
        pev.total = n_prompt;
        pev.ms = telemetry.prompt_ms;
        pev.tok_s = telemetry.prompt_tok_s;
        on_prefill(pev);
    }

    // Speculative decode loop starting from last prompt token
    auto t_decode_start = Clock::now();
    const bool decode_timing = std::getenv("GUILD_DECODE_TIMING") != nullptr;
    const int64_t windows_before = m_impl->ver->windows;
    const int64_t pcie_before = m_impl->drive.d.pcie_experts;
    const double commit_before = m_impl->ver->ms_commit;
    const double wait_before = m_impl->ver->ms_wait;
    const double host_before = m_impl->ver->ms_host;
    const double capture_before = m_impl->ver->ms_capture;
    const double pool_before = m_impl->drive.cpu_ms;
    const double validation_before = m_impl->drive.validation_ms;
    const double gu_before = m_impl->pool ? m_impl->pool->ms_multi_gu : 0.0;
    const double quant_before = m_impl->pool ? m_impl->pool->ms_multi_q : 0.0;
    const double down_before = m_impl->pool ? m_impl->pool->ms_multi_down : 0.0;
    const double draft_before = use_mtp ? m_impl->mtp->ms_draft : 0;
    const auto logits_bytes_before = m_impl->ver->logits_readback_bytes;
    const double logits_validation_before = m_impl->ver->ms_logit_validation;
    const auto cache_hits_before = m_impl->drive.d.cache_hits;
    const auto cache_lookups_before = m_impl->drive.d.cache_hits + m_impl->drive.d.cache_refused;
    const auto ram_before = m_impl->src.ram_reads(), files_before = m_impl->src.file_reads();
    const auto file_bytes_before = m_impl->src.file_blob_bytes();
    const auto offload_before = m_impl->drive.d.offload_entries;
    int64_t verified_entries = 0;
    int32_t x = static_cast<int32_t>(ids.back());
    int64_t p = n_prompt - 1;

    const bool spec_trace = m_impl->config.spec_trace || (std::getenv("GUILD_SPEC_TRACE") != nullptr &&
                                                          std::atoi(std::getenv("GUILD_SPEC_TRACE")) != 0);
    std::vector<int32_t> window(static_cast<size_t>(S), 0);
    std::vector<int32_t> drafts(static_cast<size_t>(S), 0);
    std::vector<int32_t> lookup_drafts(static_cast<size_t>(S), 0);
    std::vector<int32_t> retrieval_drafts(static_cast<size_t>(S), 0);
    std::vector<int32_t> outv(static_cast<size_t>(S), 0);
    std::vector<float> dprob(static_cast<size_t>(S), 1.0f);

    bool first_window = true;
    int produced_n = 0;
    bool eos = false;
    bool client_aborted = false;

    // Set sampling parameters and optional repetition/frequency/presence penalty history
    const int pen_h = (req.penalty_last_n > 0 &&
                       (req.penalty_repeat != 1.0f || req.penalty_freq != 0.0f || req.penalty_present != 0.0f))
                          ? std::min(req.penalty_last_n, 4096)
                          : 0;
    kernels::SamplerParams sp_params;
    sp_params.temperature = req.temperature;
    sp_params.top_p = req.top_p;
    sp_params.min_p = req.min_p;
    sp_params.top_k = req.top_k;
    sp_params.penalty_last_n = pen_h;
    sp_params.penalty_repeat = req.penalty_repeat;
    sp_params.penalty_freq = req.penalty_freq;
    sp_params.penalty_present = req.penalty_present;
    sp_params.seed = req.seed;
    sp_params.greedy = (req.temperature <= 0.0f) && (pen_h == 0);
    m_impl->ver->set_sampling(sp_params);
    if (use_mtp) {
        m_impl->mtp->set_draft_sampling(sp_params);
        m_impl->mtp->set_max_drafts(max_draft);
    }

    struct PenaltyHistoryGuard {
        core::Verifier* ver = nullptr;
        int32_t* d_rows = nullptr;
        ~PenaltyHistoryGuard() {
            if (ver)
                ver->set_history(nullptr, 0);
            if (d_rows)
                cudaFree(d_rows);
        }
    } pen_guard{m_impl->ver.get(), nullptr};
    std::vector<int32_t> pen_consumed;
    std::vector<int32_t> pen_rows_host;
    if (pen_h > 0) {
        if (cudaMalloc((void**) &pen_guard.d_rows, (size_t) kernels::kVerifyMaxT * (size_t) pen_h * sizeof(int32_t)) !=
            cudaSuccess) {
            error_msg = "failed to allocate device penalty history buffer";
            return runtime_failure();
        }
        pen_rows_host.assign((size_t) kernels::kVerifyMaxT * (size_t) pen_h, -1);
        pen_consumed.reserve((size_t) (n_prompt + req.max_new_tokens));
        for (int64_t i = 0; i < n_prompt - 1; ++i)
            pen_consumed.push_back(static_cast<int32_t>(ids[(size_t) i]));
        m_impl->ver->set_history(pen_guard.d_rows, pen_h);
    } else {
        m_impl->ver->set_history(nullptr, 0);
    }

    std::string text_accum;
    auto decoder = m_impl->tokenizer->create_incremental_decoder(false);

    spec::CandidateSource cur_source = spec::CandidateSource::Normal;
    int cur_proposed_k = 0;
    int cur_match_len = 0;
    double cur_draft_ms = 0.0;
    double cur_lookup_ms = 0.0;
    double cur_retrieval_ms = 0.0;
    double cur_sched_ms = 0.0;
    float mtp_prob_ema = 0.85f;
    const int mtp_cap_k = (m_impl->plan.mtp_spec_tokens > 0)
                              ? std::min(max_draft, std::max(1, m_impl->plan.mtp_spec_tokens - 1))
                              : max_draft;

    while (produced_n < req.max_new_tokens && !eos) {
        if (impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load())) {
            client_aborted = true;
            break;
        }

        int T = first_window ? 1 : (1 + cur_proposed_k);
        T = std::clamp(T, 1, S);
        T = std::min(T, req.max_new_tokens - produced_n);
        const int actual_proposed_k = T - 1;

        if (p + T > impl_->max_context) {
            telemetry.finish_reason = "length";
            break;
        }

        window[0] = x;
        for (int i = 1; i < T; ++i) {
            window[static_cast<size_t>(i)] = drafts[static_cast<size_t>(i - 1)];
        }

        if (pen_h > 0) {
            kernels::penalty_rows(pen_consumed.data(), (int64_t) pen_consumed.size(), window.data(), T, pen_h,
                                  pen_rows_host.data());
            if (cudaMemcpy(pen_guard.d_rows, pen_rows_host.data(), (size_t) T * (size_t) pen_h * sizeof(int32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                error_msg = "failed to upload penalty history rows";
                return runtime_failure();
            }
        }

        m_impl->drive.d.layers = 0;
        m_impl->drive.d.experts = 0;
        const int64_t misses_w0 = m_impl->drive.d.multi_misses;
        const double cpu_w0 = m_impl->drive.cpu_ms;
        const double cap_w0 = m_impl->ver->ms_capture;
        const auto t_ver0 = Clock::now();
        if (!m_impl->ver->run(T, window.data(), p, drive_pool_multi, &m_impl->drive, outv.data(), error_msg)) {
            return runtime_failure();
        }
        const auto t_ver1 = Clock::now();
        verified_entries += int64_t(T) * m_impl->K * m_impl->g.n_layers;

        int a = 0;
        while (a < T - 1 && window[static_cast<size_t>(a + 1)] == outv[static_cast<size_t>(a)])
            ++a;

        const auto t_com0 = Clock::now();
        if (!m_impl->ver->commit(a + 1, error_msg)) {
            return runtime_failure();
        }
        const auto t_com1 = Clock::now();

        const double cap_delta_ms = std::max(0.0, m_impl->ver->ms_capture - cap_w0);
        const double raw_ver_ms = std::chrono::duration<double, std::milli>(t_ver1 - t_ver0).count();
        const double exec_ver_ms = std::max(0.05, raw_ver_ms - cap_delta_ms);
        const double com_ms = std::chrono::duration<double, std::milli>(t_com1 - t_com0).count();
        const double win_cpu_ms = std::max(0.0, m_impl->drive.cpu_ms - cpu_w0);
        const int64_t win_distinct_experts = std::max<int64_t>(0, m_impl->drive.d.multi_misses - misses_w0);

        const bool was_first_window = first_window;
        if (first_window) {
            telemetry.first_window_capture_ms = cap_delta_ms;
            telemetry.ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_req_start).count();
            impl_->scheduler->seed_single_token_verify(exec_ver_ms + com_ms, win_distinct_experts, win_cpu_ms);
        }

        telemetry.drafts_offered += actual_proposed_k;
        telemetry.drafts_accepted += a;
        first_window = false;

        if (pen_h > 0) {
            for (int i = 0; i <= a; ++i)
                pen_consumed.push_back(window[(size_t) i]);
        }

        // Emit accepted tokens
        int committed_in_win = 0;
        const bool track_stream =
            need_lookup || need_retrieval || (impl_->retrieval && impl_->retrieval->corpus_size() > 0);
        for (int i = 0; i <= a && produced_n < req.max_new_tokens && !eos; ++i) {
            int32_t tid = outv[static_cast<size_t>(i)];

            // Check EOS before decoding so EOS control tokens (<|im_end|>, <|endoftext|>) are not emitted as text
            eos = std::find(m_impl->config.eos_token_ids.begin(), m_impl->config.eos_token_ids.end(),
                            static_cast<int64_t>(tid)) != m_impl->config.eos_token_ids.end();
            std::string tok_str = eos ? "" : decoder->add(tid);
            text_accum += tok_str;

            // Check custom stop sequence
            for (const auto& stop_word : req.stop) {
                if (!stop_word.empty() && text_accum.size() >= stop_word.size()) {
                    if (text_accum.substr(text_accum.size() - stop_word.size()) == stop_word) {
                        eos = true;
                        break;
                    }
                }
            }

            TokenEvent te{tid, tok_str, eos};
            if (on_token && !on_token(te)) {
                client_aborted = true;
                break;
            }

            ++produced_n;
            ++committed_in_win;
            if (need_lookup && impl_->sfx)
                impl_->sfx->append(tid);
            if (track_stream && impl_->retrieval)
                impl_->retrieval->append_stream(tid);
        }

        if (!was_first_window) {
            spec::IterationObservation obs;
            obs.source = cur_source;
            obs.proposed_k = actual_proposed_k;
            obs.accepted_k = a;
            obs.committed_tokens = std::max(1, committed_in_win);
            obs.match_len = cur_match_len;
            obs.draft_ms = cur_draft_ms;
            obs.lookup_probe_ms = cur_lookup_ms;
            obs.retrieval_probe_ms = cur_retrieval_ms;
            obs.verify_ms = exec_ver_ms;
            obs.commit_ms = com_ms;
            obs.cpu_expert_ms = win_cpu_ms;
            obs.scheduler_ms = cur_sched_ms;
            obs.distinct_experts = win_distinct_experts;
            impl_->scheduler->observe(obs);
        }

        if (client_aborted || eos || produced_n >= req.max_new_tokens)
            break;

        // Safe verification boundary: select strategy and prepare draft tokens for the next window
        int lookup_k = 0;
        double lookup_ms = 0.0;
        if (need_lookup && impl_->sfx) {
            const auto tl0 = Clock::now();
            lookup_k = impl_->sfx->propose(max_draft, lookup_drafts.data());
            lookup_ms = std::chrono::duration<double, std::milli>(Clock::now() - tl0).count();
        }

        int retrieval_k = 0;
        double retrieval_ms = 0.0;
        const bool do_retrieval_probe =
            need_retrieval && impl_->retrieval &&
            (eff_mode == spec::DecodeStrategyMode::Retrieval || impl_->retrieval->corpus_size() > 0);
        if (do_retrieval_probe) {
            const auto tr0 = Clock::now();
            retrieval_k = impl_->retrieval->propose(max_draft, retrieval_drafts.data());
            retrieval_ms = std::chrono::duration<double, std::milli>(Clock::now() - tr0).count();
        }

        const auto ts0 = Clock::now();
        spec::ProposalAvailability avail;
        avail.mtp_available = use_mtp;
        avail.mtp_max_k = use_mtp ? mtp_cap_k : 0;
        avail.mtp_first_prob = (eff_mode == spec::DecodeStrategyMode::Mtp || eff_mode == spec::DecodeStrategyMode::Auto)
                                   ? 1.0f
                                   : mtp_prob_ema;
        avail.lookup_k = lookup_k;
        avail.lookup_match_len = (need_lookup && impl_->sfx) ? impl_->sfx->last_match() : 0;
        avail.lookup_freq = (need_lookup && impl_->sfx) ? impl_->sfx->last_freq() : 1;
        avail.lookup_candidates = (need_lookup && impl_->sfx) ? impl_->sfx->last_candidates() : 1;
        avail.lookup_consensus_k = (need_lookup && impl_->sfx) ? impl_->sfx->last_consensus_k() : -1;
        avail.retrieval_k = retrieval_k;
        avail.retrieval_match_len = do_retrieval_probe ? impl_->retrieval->last_match() : 0;
        avail.retrieval_candidates = do_retrieval_probe ? impl_->retrieval->last_candidates() : 0;
        avail.retrieval_confidence = do_retrieval_probe ? impl_->retrieval->last_confidence() : 0.0f;
        avail.retrieval_from_corpus = do_retrieval_probe ? impl_->retrieval->last_from_corpus() : false;
        avail.context_tokens = p + a + 1;
        avail.active_sessions = std::max(1, m_impl->active_sessions.load());
        avail.remaining_tokens = std::max(0, req.max_new_tokens - produced_n);

        const spec::ScheduleDecision dec = impl_->scheduler->choose(avail);
        cur_sched_ms = std::chrono::duration<double, std::milli>(Clock::now() - ts0).count();

        cur_source = dec.source;
        cur_proposed_k = dec.draft_k;
        cur_match_len = 0;
        cur_lookup_ms = lookup_ms;
        cur_retrieval_ms = retrieval_ms;
        cur_draft_ms = lookup_ms + retrieval_ms;

        if (dec.source == spec::CandidateSource::Mtp && dec.draft_k > 0 && use_mtp) {
            m_impl->mtp->set_max_drafts(dec.draft_k);
            if (pen_h > 0 && m_impl->mtp->coupled()) {
                m_impl->mtp->set_draft_history(pen_consumed.data(), (int64_t) pen_consumed.size(),
                                               outv[static_cast<size_t>(a)]);
            }
            int n_drafted = 0;
            const double mtp_ms_before = m_impl->mtp->ms_draft;
            if (!m_impl->mtp->draft(T, outv.data(), p, a, drafts.data(), error_msg, dprob.data(),
                                    m_impl->config.spec_min_p, &n_drafted)) {
                return runtime_failure();
            }
            cur_draft_ms += std::max(0.0, m_impl->mtp->ms_draft - mtp_ms_before);
            mtp_prob_ema = 0.70f * mtp_prob_ema + 0.30f * dprob[0];

            // Honor MTP confidence threshold (spec_min_p)
            int valid_mtp_k = 0;
            if (m_impl->config.spec_min_p > 0.0f) {
                while (valid_mtp_k < dec.draft_k && valid_mtp_k < n_drafted &&
                       dprob[(size_t) valid_mtp_k] >= m_impl->config.spec_min_p) {
                    ++valid_mtp_k;
                }
            } else {
                valid_mtp_k = std::min(dec.draft_k, n_drafted);
            }
            cur_proposed_k = valid_mtp_k;
            if (cur_proposed_k == 0) {
                // MTP draft confidence fell below spec_min_p; in Adaptive mode, allow high-confidence Lookup fallback.
                if (eff_mode == spec::DecodeStrategyMode::Adaptive && lookup_k > 0 &&
                    (avail.lookup_match_len >= 4 || (avail.lookup_freq >= 3 && avail.lookup_consensus_k >= 2))) {
                    cur_source = spec::CandidateSource::Lookup;
                    const int cons_k = (avail.lookup_consensus_k > 0) ? avail.lookup_consensus_k : lookup_k;
                    cur_proposed_k = std::min({max_draft, lookup_k, cons_k});
                    cur_match_len = avail.lookup_match_len;
                    for (int i = 0; i < cur_proposed_k; ++i)
                        drafts[(size_t) i] = lookup_drafts[(size_t) i];
                } else {
                    cur_source = spec::CandidateSource::Normal;
                }
            }
        } else if (dec.source == spec::CandidateSource::Lookup && dec.draft_k > 0) {
            mtp_prob_ema = 0.85f * mtp_prob_ema + 0.15f * 0.85f;
            cur_proposed_k = std::min(dec.draft_k, lookup_k);
            cur_match_len = avail.lookup_match_len;
            for (int i = 0; i < cur_proposed_k; ++i)
                drafts[(size_t) i] = lookup_drafts[(size_t) i];
            if (use_mtp) {
                const double mtp_pf_before = m_impl->mtp->ms_prefill;
                if (!m_impl->mtp->prefill(m_impl->ver->final_R_all(), outv.data(), a + 1, p, error_msg))
                    return runtime_failure();
                cur_draft_ms += std::max(0.0, m_impl->mtp->ms_prefill - mtp_pf_before);
            }
        } else if (dec.source == spec::CandidateSource::Retrieval && dec.draft_k > 0) {
            mtp_prob_ema = 0.85f * mtp_prob_ema + 0.15f * 0.85f;
            cur_proposed_k = std::min(dec.draft_k, retrieval_k);
            cur_match_len = avail.retrieval_match_len;
            for (int i = 0; i < cur_proposed_k; ++i)
                drafts[(size_t) i] = retrieval_drafts[(size_t) i];
            if (use_mtp) {
                const double mtp_pf_before = m_impl->mtp->ms_prefill;
                if (!m_impl->mtp->prefill(m_impl->ver->final_R_all(), outv.data(), a + 1, p, error_msg))
                    return runtime_failure();
                cur_draft_ms += std::max(0.0, m_impl->mtp->ms_prefill - mtp_pf_before);
            }
        } else {
            mtp_prob_ema = 0.85f * mtp_prob_ema + 0.15f * 0.85f;
            cur_source = spec::CandidateSource::Normal;
            cur_proposed_k = 0;
            if (use_mtp) {
                const double mtp_pf_before = m_impl->mtp->ms_prefill;
                if (!m_impl->mtp->prefill(m_impl->ver->final_R_all(), outv.data(), a + 1, p, error_msg))
                    return runtime_failure();
                cur_draft_ms += std::max(0.0, m_impl->mtp->ms_prefill - mtp_pf_before);
            }
        }

        if (spec_trace) {
            std::fprintf(stderr,
                         "guild spec trace [pos=%lld]: src=%s k=%d exp_tok=%.2f exp_ms/tok=%.2f norm_ms/tok=%.2f "
                         "speedup=%.2fx switched=%d reason=%s\n",
                         (long long) (p + a + 1), spec::candidate_source_to_string(cur_source), cur_proposed_k,
                         dec.expected_tokens, dec.expected_ms_per_token, dec.normal_ms_per_token, dec.estimated_speedup,
                         (int) dec.switched, dec.reason);
        }

        p += (a + 1);
        x = outv[static_cast<size_t>(a)];
    }

    // A successful request must leave completed state before reset, another session, or teardown.
    if (!m_impl->ver->wait_commit(error_msg))
        return runtime_failure();
    std::string flushed = decoder->flush();
    if (!flushed.empty() && on_token && !client_aborted) {
        on_token(TokenEvent{0, flushed, false});
    }

    auto t_decode_end = Clock::now();
    telemetry.completion_tokens = produced_n;
    telemetry.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    telemetry.decode_tok_s =
        (produced_n > 0 && telemetry.decode_ms > 0) ? (produced_n / (telemetry.decode_ms / 1000.0)) : 0.0;
    telemetry.current_context_length = p;
    telemetry.prefill_chunk =
        m_impl->batched_prefill ? m_impl->batched_prefill->batch_size() : m_impl->config.prefill_chunk;
    telemetry.ram_expert_hits = m_impl->src.ram_reads() - ram_before;
    telemetry.file_expert_reads = m_impl->src.file_reads() - files_before;
    telemetry.file_read_bytes = m_impl->src.file_blob_bytes() - file_bytes_before;
    telemetry.gpu_cache_hits =
        m_impl->ver->all_experts_resident() ? verified_entries : m_impl->drive.d.cache_hits - cache_hits_before;
    telemetry.gpu_cache_lookups =
        m_impl->ver->all_experts_resident()
            ? verified_entries
            : m_impl->drive.d.cache_hits + m_impl->drive.d.cache_refused - cache_lookups_before;
    telemetry.offload_entries = m_impl->drive.d.offload_entries - offload_before;
    telemetry.request_latency_s = std::chrono::duration<double>(t_decode_end - t_req_start).count();
    telemetry.decode_windows = m_impl->ver->windows - windows_before;
    telemetry.decode_mtp_draft_ms = use_mtp ? (m_impl->mtp->ms_draft - draft_before) : 0.0;
    telemetry.decode_commit_ms = m_impl->ver->ms_commit - commit_before;
    telemetry.decode_host_wait_ms = m_impl->ver->ms_wait - wait_before;
    telemetry.decode_ple_ms = m_impl->ver->ms_host - host_before;
    telemetry.decode_cpu_pool_ms = m_impl->drive.cpu_ms - pool_before;
    telemetry.decode_cpu_validation_ms = m_impl->drive.validation_ms - validation_before;
    telemetry.decode_cpu_gu_ms = m_impl->pool ? (m_impl->pool->ms_multi_gu - gu_before) : 0.0;
    telemetry.decode_cpu_quant_ms = m_impl->pool ? (m_impl->pool->ms_multi_q - quant_before) : 0.0;
    telemetry.decode_cpu_down_ms = m_impl->pool ? (m_impl->pool->ms_multi_down - down_before) : 0.0;
    telemetry.decode_logit_validation_ms = m_impl->ver->ms_logit_validation - logits_validation_before;
    if (impl_->scheduler) {
        const auto st = impl_->scheduler->snapshot();
        telemetry.decode_strategy_mode = st.mode_name;
        telemetry.decode_strategy_initial = st.initial_strategy;
        telemetry.decode_strategy_final = st.final_strategy;
        telemetry.spec_fallback_reason = st.last_switch_reason;
        telemetry.decode_lookup_draft_ms = st.total_lookup_ms;
        telemetry.decode_retrieval_draft_ms = st.total_retrieval_ms;
        telemetry.decode_scheduler_ms = st.total_scheduler_ms;
        telemetry.spec_ms_per_committed_token =
            (produced_n > 0 && telemetry.decode_ms > 0.0) ? (telemetry.decode_ms / (double) produced_n) : 0.0;
        telemetry.spec_estimated_normal_ms_per_token = st.estimated_normal_ms_per_token;
        telemetry.spec_estimated_speedup =
            (telemetry.spec_ms_per_committed_token > 1e-6)
                ? (st.estimated_normal_ms_per_token / telemetry.spec_ms_per_committed_token)
                : 1.0;
        telemetry.spec_avg_proposed_draft = st.avg_proposed_draft_len;
        telemetry.spec_avg_accepted_prefix = st.avg_accepted_prefix_len;
        telemetry.spec_mean_abs_cost_pred_error_ms = st.mean_abs_cost_pred_error_ms;
        telemetry.spec_mean_abs_acc_pred_error = st.mean_abs_acc_pred_error;
        telemetry.spec_effective_hysteresis_margin = st.effective_hysteresis_margin;
        telemetry.spec_measured_verify_cv = st.measured_verify_cv;
        telemetry.spec_strategy_switches = st.strategy_switches;
        telemetry.spec_fallback_count = st.fallback_to_normal_count;
        telemetry.spec_normal_windows = st.normal_iterations + 1; // +1 for initial TTFT single-token window
        telemetry.spec_mtp_windows = st.mtp_iterations;
        telemetry.spec_lookup_windows = st.lookup_iterations;
        telemetry.spec_retrieval_windows = st.retrieval_iterations;
    }
    telemetry.spec_drafter_memory_bytes =
        (impl_->sfx ? impl_->sfx->memory_bytes() : 0) + (impl_->retrieval ? impl_->retrieval->memory_bytes() : 0);
    telemetry.decode_verify_ms =
        std::max(0.0, telemetry.decode_ms - telemetry.decode_mtp_draft_ms - telemetry.decode_lookup_draft_ms -
                          telemetry.decode_retrieval_draft_ms - telemetry.decode_scheduler_ms);
    (void) capture_before;
    if (decode_timing) {
        const int64_t windows = m_impl->ver->windows - windows_before;
        const double denominator = std::max<int64_t>(1, windows);
        std::fprintf(
            stderr,
            "guild decode timing: %lld windows, %.2f tokens/window, %.2f ms/window; "
            "commit %.3f, CPU dispatch %.2f (finite checks %.2f), draft %.2f ms/window; "
            "PCIe %lld experts, drafts %d/%d; logit validation %.3f ms/window, %llu bytes read\n",
            static_cast<long long>(windows), produced_n / denominator, telemetry.decode_ms / denominator,
            (m_impl->ver->ms_commit - commit_before) / denominator, (m_impl->drive.cpu_ms - pool_before) / denominator,
            (m_impl->drive.validation_ms - validation_before) / denominator,
            (use_mtp ? m_impl->mtp->ms_draft - draft_before : 0) / denominator,
            static_cast<long long>(m_impl->drive.d.pcie_experts - pcie_before), telemetry.drafts_accepted,
            telemetry.drafts_offered, (m_impl->ver->ms_logit_validation - logits_validation_before) / denominator,
            static_cast<unsigned long long>(m_impl->ver->logits_readback_bytes - logits_bytes_before));
    }
    if (const std::string prof = m_impl->ver->profile_report(); !prof.empty()) {
        std::fprintf(stderr, "guild verify profile:%s\n", prof.c_str());
    }

    if (client_aborted) {
        telemetry.finish_reason = "cancel";
    } else if (eos) {
        telemetry.finish_reason = "stop";
    } else if (produced_n >= req.max_new_tokens) {
        telemetry.finish_reason = "length";
    } else {
        telemetry.finish_reason = "stop";
    }

    return true;
} catch (const std::exception& e) {
    telemetry.finish_reason = "error";
    error_msg = std::string("inference failed: ") + e.what();
    if (impl_ && impl_->model_impl && impl_->model_impl->ready)
        impl_->model_impl->ready->store(false);
    return false;
} catch (...) {
    telemetry.finish_reason = "error";
    error_msg = "inference failed with an unknown exception";
    if (impl_ && impl_->model_impl && impl_->model_impl->ready)
        impl_->model_impl->ready->store(false);
    return false;
}

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
