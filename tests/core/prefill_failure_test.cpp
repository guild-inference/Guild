#include "../check.hpp"
#include "runtime_fixture.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/server/native_engine.hpp"
#include <unistd.h>
#include <limits>
class MissingSource final : public guild::core::ExpertSource {
    const uint8_t* blob(int64_t, int64_t) override {
        return nullptr;
    }
};
int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("guild-prefill-failure-" + std::to_string(getpid()));
    fs::create_directories(root);
    guild::server::NativeInferenceEngineOptions opts;
    opts.paths = fixture::runtime_pack(root);
    opts.plan.context_length = 64;
    opts.plan.mtp_spec_tokens = 0;
    opts.plan.prefill_batch = 8;
    const auto before = guild::core::qsa_kv_host_bytes();
    {
        guild::server::NativeInferenceEngine engine(opts);
        std::string err;
        CHECK(engine.init(err));
        auto* m = engine.model()->impl();
        CHECK(m->batched_prefill);
        guild::server::InferenceRequest req;
        req.prompt_tokens = {1, 2, 3};
        req.max_tokens = 2;
        guild::server::GenerationResult result;
        CHECK(engine.generate(req, result));
        CHECK(result.completion_tokens == 2);
        guild::runtime::BatchedPrefill failing;
        MissingSource missing;
        CHECK(failing.init(m->wt, m->g, m->ss, missing, *m->pool, m->main_stream, err, 8));
        const int64_t ids[] = {1, 2, 3};
        CHECK(!failing.run(ids, 3, 0, err));
        CHECK(err.find("missing routed expert") != std::string::npos);
        failing.should_stop = [] { return true; };
        CHECK(!failing.run(ids, 3, 0, err) && err == "cancelled");
        CHECK(engine.generate(req, result)); // reset of a partially ingested prompt is complete
        const auto* norm = m->wt.find("blk.0.attn_norm.weight");
        CHECK(norm);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        CHECK(cudaMemcpy(const_cast<void*>(norm->data), &nan, 4, cudaMemcpyHostToDevice) == cudaSuccess);
        int emitted = 0;
        CHECK(!engine.generate_stream(
            req,
            [&](const auto&) {
                ++emitted;
                return true;
            },
            result));
        CHECK(emitted == 0 && result.finish_reason == "error" &&
              result.error_message.find("nonfinite") != std::string::npos);
        CHECK(!engine.is_ready());
    }
    CHECK(guild::core::qsa_kv_host_bytes() == before);
    fs::remove_all(root);
}
