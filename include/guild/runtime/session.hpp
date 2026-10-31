#pragma once

#include "guild/runtime/types.hpp"

#include <memory>
#include <string>

namespace guild::runtime {

class GuildModel;

class GuildSession {
public:
    explicit GuildSession(GuildModel* model, int64_t max_context);
    ~GuildSession();

    GuildSession(const GuildSession&) = delete;
    GuildSession& operator=(const GuildSession&) = delete;

    // Generates output tokens for request
    bool generate(const GenerationRequest& req, TokenStreamCallback on_token, RuntimeTelemetry& telemetry,
                  std::string& error_msg, PrefillProgressCallback on_prefill = nullptr);

    // Scores a prompt without generating: reads req.tokens (reusing the longest prefix the session still holds or
    // has a checkpoint of) and returns the next-token logits of req.candidates at the last position.  The state is
    // left after the prompt's last-but-one token, so a later prompt with the same beginning reads only the rest.
    bool score(const ScoreRequest& req, ScoreResult& result, std::string& error_msg);

    // Resets session conversation/KV history
    void reset();

    // Session-local retrieval datastore controls (strictly isolated to this session)
    void ingest_retrieval_tokens(const int32_t* ids, size_t n);
    void clear_retrieval_corpus();
    size_t retrieval_corpus_tokens() const;

    // Signals in-flight generation to cancel immediately
    void cancel();

    GuildModel* model() const {
        return model_;
    }
    int64_t max_context() const {
        return max_context_;
    }

    struct Impl;
    Impl* impl() {
        return impl_.get();
    }

private:
    GuildModel* model_;
    int64_t max_context_;
    std::unique_ptr<Impl> impl_;
};

} // namespace guild::runtime
