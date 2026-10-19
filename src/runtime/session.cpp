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

namespace guild::runtime {

struct GuildSession::Impl {
    std::shared_ptr<GuildModel::Impl> model_impl;
    GuildModel* model = nullptr;
    int64_t max_context = 262144;
    std::unique_ptr<spec::SuffixDrafter> sfx;
    std::unique_ptr<spec::DraftPolicy> policy;
    model::SessionRunnerLease runner_lease;
    kernels::SessionKernelLease kernel_lease;
    std::atomic<bool> cancelled{false};
};

GuildSession::GuildSession(GuildModel* model, int64_t max_context)
    : model_(model), max_context_(max_context), impl_(std::make_unique<Impl>()) {
    impl_->model = model;
    impl_->model_impl = model ? model->impl_ : nullptr;
    auto* m_impl = impl_->model_impl.get();
    impl_->max_context = max_context > 0 ? max_context : (m_impl ? m_impl->config.max_context : 262144);
    impl_->sfx = std::make_unique<spec::SuffixDrafter>(3, 64, (size_t) impl_->max_context + 4096);
    impl_->policy = std::make_unique<spec::DraftPolicy>(m_impl ? m_impl->config.spec : 2);
    impl_->runner_lease = model::SessionRunnerLease(
        &model::ModelRunnerRegistry::instance(),
        (m_impl && m_impl->runner && m_impl->runner->package_id()) ? m_impl->runner->package_id() : "");
    impl_->kernel_lease = kernels::SessionKernelLease(
        &kernels::KernelRegistry::instance(),
        m_impl ? m_impl->kernel_lease.package_ids() : std::vector<std::string>{});
}

GuildSession::~GuildSession() = default;

void GuildSession::reset() {
    if (!impl_ || !impl_->model_impl) return;
    auto* m_impl = impl_->model_impl.get();
    std::lock_guard<std::mutex> lock(m_impl->generation_mutex);
    if (!m_impl->ready || !m_impl->ready->load()) return;
    std::optional<kernels::ScopedCudaAttn> attn_scope;
    if (m_impl->cuda_attn) attn_scope.emplace(*m_impl->cuda_attn);
    std::optional<kernels::ScopedCudaMoE> moe_scope;
    if (m_impl->cuda_moe) moe_scope.emplace(*m_impl->cuda_moe);
    std::string error;
    if (!m_impl->ver->wait_commit(error) || (m_impl->mtp && !m_impl->mtp->idle(error))) {
        m_impl->ready->store(false);
        return;
    }
    core::session_zero(m_impl->ss, m_impl->g, nullptr, m_impl->main_stream);
    if (m_impl->mtp) core::qsa_state_zero(m_impl->mtp->kv_state(), m_impl->g, m_impl->main_stream);
    if (cudaStreamSynchronize(m_impl->main_stream) != cudaSuccess) m_impl->ready->store(false);
}

void GuildSession::cancel() {
    if (impl_) {
        impl_->cancelled.store(true);
    }
}

bool GuildSession::generate(
    const GenerationRequest& req,
    TokenStreamCallback on_token,
    RuntimeTelemetry& telemetry,
    std::string& error_msg,
    PrefillProgressCallback on_prefill
) try {
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
    auto runtime_failure = [&]() { m_impl->ready->store(false); return false; };
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
        error_msg = "prompt (" + std::to_string(n_prompt) + ") + max_tokens (" +
                    std::to_string(req.max_new_tokens) + ") exceeds max context (" +
                    std::to_string(impl_->max_context) + ")";
        return false;
    }
    for (int64_t id : ids) {
        if (id < 0 || id >= m_impl->ver->vocab()) {
            error_msg = "prompt token id " + std::to_string(id) + " is outside the model vocabulary";
            return false;
        }
    }

    if (m_impl->g.is_dense_attention() &&
        n_prompt + req.max_new_tokens > m_impl->ver->attention_capacity()) {
        error_msg = "dense attention request exceeds the allocated single-tile history capacity; multi-tile execution is unavailable";
        return false;
    }

    struct RequestAffinity {
        kernels::cpu::ThreadAffinity previous;
        ~RequestAffinity() { kernels::cpu::restore_thread_affinity(previous); }
    } affinity;
    const char* host_affinity=std::getenv("GUILD_HOST_AFFINITY");
    if (!host_affinity || std::atoi(host_affinity)!=0)
        affinity.previous=kernels::cpu::pin_current_thread_if_allowed(m_impl->pool->host_core());
    std::optional<kernels::ScopedCudaDense> dense_scope;
    if (m_impl->cuda_dense) dense_scope.emplace(*m_impl->cuda_dense);
    std::optional<kernels::ScopedCudaAttn> attn_scope;
    if (m_impl->cuda_attn) attn_scope.emplace(*m_impl->cuda_attn);
    std::optional<kernels::ScopedCudaMoE> moe_scope;
    if (m_impl->cuda_moe) moe_scope.emplace(*m_impl->cuda_moe);

    // Reset KV state for fresh generation
    if (!m_impl->ver->wait_commit(error_msg)) return runtime_failure();
    if (m_impl->mtp && !m_impl->mtp->idle(error_msg)) return runtime_failure();
    core::session_zero(m_impl->ss, m_impl->g, nullptr, m_impl->main_stream);
    if (m_impl->mtp) core::qsa_state_zero(m_impl->mtp->kv_state(), m_impl->g, m_impl->main_stream);
    if (cudaStreamSynchronize(m_impl->main_stream) != cudaSuccess) {
        error_msg = "failed to reset inference state";
        return runtime_failure();
    }

    // Reset prefill stats
    auto t_req_start = Clock::now();
    auto t_prefill_start = Clock::now();

    bool use_mtp = m_impl->mtp != nullptr;
    if (use_mtp) m_impl->mtp->set_prompt_len(n_prompt);
    if (m_impl->prefill) {
        m_impl->prefill->should_stop = [this, &req] {
            return impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load());
        };

        if (use_mtp) {
            m_impl->prefill->on_chunk = [this, m_impl, &ids](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                std::vector<int32_t> next_tokens(static_cast<size_t>(T));
                for (int64_t i = 0; i < T; ++i) {
                    if (p0 + i + 1 < static_cast<int64_t>(ids.size())) {
                        next_tokens[static_cast<size_t>(i)] = static_cast<int32_t>(ids[static_cast<size_t>(p0 + i + 1)]);
                    }
                }
                if (m_impl->prefill->draft_kv(*m_impl->mtp, R_rows, next_tokens.data(), T, p0, e)) return true;
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
                if (!m_impl->ver->run(1, &prompt_tok, pos, drive_pool_multi, &m_impl->drive, sink_out.data(), error_msg)) {
                    return runtime_failure();
                }
                if (!m_impl->ver->commit(1, error_msg)) {
                    return runtime_failure();
                }
                if (use_mtp) {
                    const int32_t next = int32_t(ids[size_t(pos + 1)]);
                    if (!m_impl->mtp->prefill(m_impl->ver->final_R_all(), &next, 1, pos, error_msg)) return runtime_failure();
                }
            }
        }
    }

    auto t_prefill_end = Clock::now();
    telemetry.prompt_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    telemetry.prompt_tokens = static_cast<int>(n_prompt);
    telemetry.prompt_tok_s = (n_prompt > 1 && telemetry.prompt_ms > 0)
                                 ? ((n_prompt - 1) / (telemetry.prompt_ms / 1000.0))
                                 : 0.0;
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

    const int S = std::max(1, m_impl->config.spec);
    std::vector<int32_t> window(static_cast<size_t>(S), 0);
    std::vector<int32_t> drafts(static_cast<size_t>(S), 0);
    std::vector<int32_t> outv(static_cast<size_t>(S), 0);
    std::vector<float> dprob(static_cast<size_t>(S), 1.0f);

    bool first_window = true;
    int produced_n = 0;
    bool eos = false;
    bool client_aborted = false;

    // Set sampling parameters
    kernels::SamplerParams sp_params;
    sp_params.temperature = req.temperature;
    sp_params.top_p = req.top_p;
    sp_params.top_k = req.top_k;
    sp_params.seed = req.seed;
    sp_params.greedy = req.temperature <= 0;
    m_impl->ver->set_sampling(sp_params);
    if (use_mtp) m_impl->mtp->set_draft_sampling(sp_params);

    std::string text_accum;
    auto decoder = m_impl->tokenizer->create_incremental_decoder(true);

    while (produced_n < req.max_new_tokens && !eos) {
        if (impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load())) {
            client_aborted = true;
            break;
        }

        int T = S;
        if (use_mtp && m_impl->config.spec_min_p > 0.0f) {
            T = 1;
            while (T < S && dprob[static_cast<size_t>(T - 1)] >= m_impl->config.spec_min_p) ++T;
        }
        // There are no valid draft IDs before the first round or without a drafter.
        if (first_window || !use_mtp) T = 1;
        T = std::min(T, req.max_new_tokens - produced_n);

        if (p + T > impl_->max_context) {
            telemetry.finish_reason = "length";
            break;
        }

        window[0] = x;
        for (int i = 1; i < T; ++i) {
            window[static_cast<size_t>(i)] = drafts[static_cast<size_t>(i - 1)];
        }

        m_impl->drive.d.layers = 0;
        m_impl->drive.d.experts = 0;
        const double cap_w0 = m_impl->ver->ms_capture;
        if (!m_impl->ver->run(T, window.data(), p, drive_pool_multi, &m_impl->drive, outv.data(), error_msg)) {
            return runtime_failure();
        }
        verified_entries += int64_t(T) * m_impl->K * m_impl->g.n_layers;

        int a = 0;
        while (a < T - 1 && window[static_cast<size_t>(a + 1)] == outv[static_cast<size_t>(a)]) ++a;

        if (!m_impl->ver->commit(a + 1, error_msg)) {
            return runtime_failure();
        }
        if (first_window) {
            telemetry.first_window_capture_ms = m_impl->ver->ms_capture - cap_w0;
            telemetry.ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_req_start).count();
        }

        telemetry.drafts_offered += (T - 1);
        telemetry.drafts_accepted += a;
        first_window = false;

        // Emit accepted tokens
        for (int i = 0; i <= a && produced_n < req.max_new_tokens && !eos; ++i) {
            int32_t tid = outv[static_cast<size_t>(i)];
            std::string tok_str = decoder->add(tid);
            text_accum += tok_str;

            // Check EOS
            eos = std::find(m_impl->config.eos_token_ids.begin(),
                            m_impl->config.eos_token_ids.end(),
                            static_cast<int64_t>(tid)) != m_impl->config.eos_token_ids.end();

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
            if (impl_->sfx) impl_->sfx->append(tid);
        }

        if (client_aborted || eos) break;

        // MTP Draft for next round
        if (use_mtp && produced_n < req.max_new_tokens) {
            if (!m_impl->mtp->draft(T, outv.data(), p, a, drafts.data(), error_msg, dprob.data(), m_impl->config.spec_min_p)) {
                return runtime_failure();
            }
        }

        p += (a + 1);
        x = outv[static_cast<size_t>(a)];
    }

    // A successful request must leave completed state before reset, another session, or teardown.
    if (!m_impl->ver->wait_commit(error_msg)) return runtime_failure();
    std::string flushed = decoder->flush();
    if (!flushed.empty() && on_token && !client_aborted) {
        on_token(TokenEvent{0, flushed, false});
    }

    auto t_decode_end = Clock::now();
    telemetry.completion_tokens = produced_n;
    telemetry.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    telemetry.decode_tok_s = (produced_n > 0 && telemetry.decode_ms > 0)
                                 ? (produced_n / (telemetry.decode_ms / 1000.0))
                                 : 0.0;
    telemetry.current_context_length = p;
    telemetry.prefill_chunk = m_impl->batched_prefill ? m_impl->batched_prefill->batch_size() : m_impl->config.prefill_chunk;
    telemetry.ram_expert_hits = m_impl->src.ram_reads() - ram_before;
    telemetry.file_expert_reads = m_impl->src.file_reads() - files_before;
    telemetry.file_read_bytes = m_impl->src.file_blob_bytes() - file_bytes_before;
    telemetry.gpu_cache_hits = m_impl->ver->all_experts_resident() ? verified_entries : m_impl->drive.d.cache_hits - cache_hits_before;
    telemetry.gpu_cache_lookups = m_impl->ver->all_experts_resident() ? verified_entries
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
    telemetry.decode_verify_ms = std::max(0.0, telemetry.decode_ms - telemetry.decode_mtp_draft_ms);
    (void) capture_before;
    if (decode_timing) {
        const int64_t windows = m_impl->ver->windows - windows_before;
        const double denominator = std::max<int64_t>(1, windows);
        std::fprintf(stderr, "guild decode timing: %lld windows, %.2f tokens/window, %.2f ms/window; "
                             "commit %.3f, CPU dispatch %.2f (finite checks %.2f), draft %.2f ms/window; "
                             "PCIe %lld experts, drafts %d/%d; logit validation %.3f ms/window, %llu bytes read\n",
                     static_cast<long long>(windows), produced_n / denominator, telemetry.decode_ms / denominator,
                     (m_impl->ver->ms_commit - commit_before) / denominator,
                     (m_impl->drive.cpu_ms - pool_before) / denominator,
                     (m_impl->drive.validation_ms - validation_before) / denominator,
                     (use_mtp ? m_impl->mtp->ms_draft - draft_before : 0) / denominator,
                     static_cast<long long>(m_impl->drive.d.pcie_experts - pcie_before),
                     telemetry.drafts_accepted, telemetry.drafts_offered,
                      (m_impl->ver->ms_logit_validation - logits_validation_before) / denominator,
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
    if (impl_ && impl_->model_impl && impl_->model_impl->ready) impl_->model_impl->ready->store(false);
    return false;
} catch (...) {
    telemetry.finish_reason = "error";
    error_msg = "inference failed with an unknown exception";
    if (impl_ && impl_->model_impl && impl_->model_impl->ready) impl_->model_impl->ready->store(false);
    return false;
}

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
