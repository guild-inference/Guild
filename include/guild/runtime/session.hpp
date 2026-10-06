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
    bool generate(
        const GenerationRequest& req,
        TokenStreamCallback on_token,
        RuntimeTelemetry& telemetry,
        std::string& error_msg,
        PrefillProgressCallback on_prefill = nullptr
    );

    // Resets session conversation/KV history
    void reset();

    // Signals in-flight generation to cancel immediately
    void cancel();

    GuildModel* model() const { return model_; }
    int64_t max_context() const { return max_context_; }

    struct Impl;
    Impl* impl() { return impl_.get(); }

private:
    GuildModel* model_;
    int64_t max_context_;
    std::unique_ptr<Impl> impl_;
};

} // namespace guild::runtime
