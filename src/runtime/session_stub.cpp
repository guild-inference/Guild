#include "guild/runtime/session.hpp"
#include "guild/runtime/model.hpp"

namespace guild::runtime {

struct GuildSession::Impl {};

GuildSession::GuildSession(GuildModel* model, int64_t max_context)
    : model_(model),
      max_context_(max_context),
      impl_(std::make_unique<Impl>()) {
}

GuildSession::~GuildSession() = default;

bool GuildSession::generate(const GenerationRequest& req, TokenStreamCallback on_token, RuntimeTelemetry& telemetry,
                            std::string& error_msg, PrefillProgressCallback on_prefill) {
    (void) req;
    (void) on_token;
    telemetry = RuntimeTelemetry{};
    telemetry.finish_reason = "error";
    (void) on_prefill;
    error_msg = "Guild GPU runtime is disabled in CPU-only build";
    return false;
}

void GuildSession::reset() {
}
void GuildSession::ingest_retrieval_tokens(const int32_t* ids, size_t n) {
    (void) ids;
    (void) n;
}
void GuildSession::clear_retrieval_corpus() {
}
size_t GuildSession::retrieval_corpus_tokens() const {
    return 0;
}
void GuildSession::cancel() {
}

} // namespace guild::runtime
