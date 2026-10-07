#include "guild/runtime/session.hpp"
#include "guild/runtime/model.hpp"

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "model_impl.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <vector>

namespace guild::runtime {

struct GuildSession::Impl {
    GuildModel* model = nullptr;
    int64_t max_context = 262144;
    std::unique_ptr<spec::SuffixDrafter> sfx;
    std::unique_ptr<spec::DraftPolicy> policy;
    std::atomic<bool> cancelled{false};
};

GuildSession::GuildSession(GuildModel* model, int64_t max_context)
    : model_(model), max_context_(max_context), impl_(std::make_unique<Impl>()) {
    impl_->model = model;
    impl_->max_context = max_context > 0 ? max_context : model->config().max_context;
    impl_->sfx = std::make_unique<spec::SuffixDrafter>(3, 64, (size_t) impl_->max_context + 4096);
    impl_->policy = std::make_unique<spec::DraftPolicy>(model->config().spec);
}

GuildSession::~GuildSession() = default;

void GuildSession::reset() {
    if (!impl_ || !model_) return;
    auto* m_impl = model_->impl();
    core::session_zero(m_impl->ss, m_impl->g, nullptr, m_impl->main_stream);
    cudaStreamSynchronize(m_impl->main_stream);
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
) {
    if (!model_ || !impl_) {
        error_msg = "session uninitialized";
        return false;
    }

    auto* m_impl = model_->impl();
    std::lock_guard<std::mutex> lock(m_impl->generation_mutex);

    impl_->cancelled.store(false);

    // Resolve prompt tokens
    std::vector<int64_t> ids;
    if (!req.prompt_tokens.empty()) {
        ids.assign(req.prompt_tokens.begin(), req.prompt_tokens.end());
    } else if (!req.prompt.empty()) {
        auto toks = model_->tokenizer().tokenize(req.prompt);
        ids.assign(toks.begin(), toks.end());
    }

    if (ids.empty()) {
        error_msg = "prompt cannot be empty";
        return false;
    }

    const int64_t n_prompt = static_cast<int64_t>(ids.size());
    if (n_prompt + req.max_new_tokens > impl_->max_context) {
        error_msg = "prompt (" + std::to_string(n_prompt) + ") + max_tokens (" +
                    std::to_string(req.max_new_tokens) + ") exceeds max context (" +
                    std::to_string(impl_->max_context) + ")";
        return false;
    }

    // Reset KV state for fresh generation
    core::session_zero(m_impl->ss, m_impl->g, nullptr, m_impl->main_stream);
    cudaStreamSynchronize(m_impl->main_stream);

    // Reset prefill stats
    auto t_req_start = Clock::now();
    auto t_prefill_start = Clock::now();

    m_impl->prefill->should_stop = [this, &req] {
        return impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load());
    };

    bool use_mtp = !model_->paths().mtp_dir.empty();
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
            if (impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load())) {
                telemetry.finish_reason = "cancel";
                return true;
            }
            return false;
        }
    }

    auto t_prefill_end = Clock::now();
    telemetry.prompt_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    telemetry.prompt_tokens = static_cast<int>(n_prompt);
    telemetry.prompt_tok_s = (n_prompt > 0 && telemetry.prompt_ms > 0)
                                 ? (n_prompt / (telemetry.prompt_ms / 1000.0))
                                 : 0.0;

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
    int32_t x = static_cast<int32_t>(ids.back());
    int64_t p = n_prompt - 1;

    const int S = std::max(1, model_->config().spec);
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
    m_impl->ver->set_sampling(sp_params);

    std::string text_accum;

    while (produced_n < req.max_new_tokens && !eos) {
        if (impl_->cancelled.load() || (req.cancel_flag && req.cancel_flag->load())) {
            client_aborted = true;
            break;
        }

        int T = S;
        if (req.temperature == 0.0f && req.top_p >= 1.0f && model_->config().spec_min_p > 0.0f) {
            T = 1;
            while (T < S && dprob[static_cast<size_t>(T - 1)] >= model_->config().spec_min_p) ++T;
        }
        if (first_window) T = 1;

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
        m_impl->drive.d.failed = false;

        if (!m_impl->ver->run(T, window.data(), p, drive_pool_multi, &m_impl->drive, outv.data(), error_msg)) {
            return false;
        }

        int a = 0;
        while (a < T - 1 && window[static_cast<size_t>(a + 1)] == outv[static_cast<size_t>(a)]) ++a;

        if (!m_impl->ver->commit(a + 1, error_msg)) {
            return false;
        }

        telemetry.drafts_offered += (T - 1);
        telemetry.drafts_accepted += a;
        first_window = false;

        // Emit accepted tokens
        for (int i = 0; i <= a && produced_n < req.max_new_tokens && !eos; ++i) {
            int32_t tid = outv[static_cast<size_t>(i)];
            std::string tok_str = model_->tokenizer().decode(tid);
            text_accum += tok_str;

            // Check EOS
            eos = std::find(model_->config().eos_token_ids.begin(),
                            model_->config().eos_token_ids.end(),
                            static_cast<int64_t>(tid)) != model_->config().eos_token_ids.end();

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
        if (use_mtp) {
            m_impl->mtp->draft(T, outv.data(), p, a, drafts.data(), error_msg, dprob.data(), model_->config().spec_min_p);
        }

        p += (a + 1);
        x = outv[static_cast<size_t>(a)];
    }

    auto t_decode_end = Clock::now();
    telemetry.completion_tokens = produced_n;
    telemetry.decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    telemetry.decode_tok_s = (produced_n > 0 && telemetry.decode_ms > 0)
                                 ? (produced_n / (telemetry.decode_ms / 1000.0))
                                 : 0.0;
    telemetry.current_context_length = p;
    telemetry.prefill_chunk = model_->config().prefill_chunk;
    telemetry.ram_expert_hits = m_impl->src.ram_reads();
    telemetry.file_expert_reads = m_impl->src.file_reads();
    telemetry.gpu_cache_hits = m_impl->drive.d.cache_hits;
    telemetry.request_latency_s = std::chrono::duration<double>(t_decode_end - t_req_start).count();

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
}

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
