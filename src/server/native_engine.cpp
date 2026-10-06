#include "guild/server/native_engine.hpp"

namespace guild::server {

NativeInferenceEngine::NativeInferenceEngine(NativeInferenceEngineOptions options)
    : options_(std::move(options)) {}

NativeInferenceEngine::~NativeInferenceEngine() {
    stop();
}

bool NativeInferenceEngine::init(std::string& error_msg) {
    model_ = guild::runtime::GuildModel::load(options_.paths, options_.desc, options_.plan, error_msg);
    if (!model_) {
        ready_ = false;
        return false;
    }

    session_ = model_->create_session(options_.plan.context_length);
    if (!session_) {
        error_msg = "failed to create in-process session";
        ready_ = false;
        return false;
    }

    ready_ = true;
    return true;
}

void NativeInferenceEngine::stop() {
    if (session_) {
        session_->cancel();
    }
}

bool NativeInferenceEngine::generate(const InferenceRequest& req, GenerationResult& result) {
    return generate_stream(req, nullptr, result);
}

bool NativeInferenceEngine::generate_stream(const InferenceRequest& req,
                                           StreamCallback on_token,
                                           GenerationResult& result) {
    if (!is_ready()) {
        return false;
    }

    result.text.clear();
    result.tokens.clear();
    result.finish_reason = "stop";

    guild::runtime::GenerationRequest g_req;
    g_req.request_id = req.request_id;
    g_req.prompt = req.prompt;
    g_req.prompt_tokens = req.prompt_tokens;
    g_req.max_new_tokens = req.max_tokens > 0 ? req.max_tokens : 4096;
    g_req.temperature = req.sampling.temperature;
    g_req.top_p = req.sampling.top_p;
    g_req.top_k = req.sampling.top_k;
    g_req.seed = req.sampling.seed;
    g_req.stop = req.sampling.stop;

    auto token_cb = [&](const guild::runtime::TokenEvent& ev) -> bool {
        TokenOutput tok;
        tok.token_id = ev.token_id;
        tok.text = ev.text;
        tok.is_special = ev.is_special;

        result.text += ev.text;
        result.tokens.push_back(tok);

        if (on_token) {
            return on_token(tok);
        }
        return true;
    };

    auto prefill_cb = [&](const guild::runtime::PrefillProgressEvent& pe) {
        result.prompt_tok_s = pe.tok_s;
        result.prompt_ms = pe.ms;
    };

    guild::runtime::RuntimeTelemetry telem;
    std::string gen_err;
    bool ok = session_->generate(g_req, token_cb, telem, gen_err, prefill_cb);

    result.prompt_tokens = telem.prompt_tokens;
    result.completion_tokens = telem.completion_tokens;
    result.prompt_ms = telem.prompt_ms;
    result.decode_ms = telem.decode_ms;
    result.prompt_tok_s = telem.prompt_tok_s;
    result.decode_tok_s = telem.decode_tok_s;
    result.finish_reason = telem.finish_reason;
    result.drafts_accepted = telem.drafts_accepted;
    result.drafts_offered = telem.drafts_offered;
    result.ram_blobs = telem.ram_expert_hits;
    result.file_blobs = telem.file_expert_reads;
    result.gpu_cache_hits = telem.gpu_cache_hits;

    return ok;
}

} // namespace guild::server
