#include "../check.hpp"
#include "runtime_fixture.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/server/native_engine.hpp"
#include <cuda_runtime.h>
#include <unistd.h>

class MissingSource final : public guild::core::ExpertSource {
    const uint8_t* blob(int64_t, int64_t) override { return nullptr; }
};

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("guild-inference-failure-" + std::to_string(getpid()));
    fs::create_directories(root);
    guild::server::NativeInferenceEngineOptions opts;
    opts.paths = fixture::runtime_pack(root);
    opts.plan.context_length = 64;
    opts.plan.mtp_spec_tokens = 2;
    opts.plan.kv_mode = guild::memory::KvMode::HostOnly;
    const uint64_t pinned_before = guild::core::qsa_kv_host_bytes();
    for (bool nonfinite : {false, true}) {
        guild::server::NativeInferenceEngine engine(opts);
        std::string err;
        if (!engine.init(err)) std::fprintf(stderr, "init: %s\n", err.c_str());
        CHECK(engine.is_ready());
        guild::server::InferenceRequest req;
        req.prompt_tokens = {1};
        req.max_tokens = 2;
        guild::server::GenerationResult healthy;
        CHECK(engine.generate(req, healthy));
        CHECK(healthy.completion_tokens == 2 && healthy.tokens[0].token_id == 0);
        CHECK(engine.is_ready());
        req.prompt_tokens = {-1};
        CHECK(!engine.generate(req, healthy));
        CHECK(healthy.error_message.find("outside") != std::string::npos);
        CHECK(engine.is_ready()); // invalid requests do not poison a healthy model
        req.prompt_tokens = {1};
        MissingSource missing;
        if (nonfinite) {
            // Poison a head scale on the device: the real CUDA head produces NaN,
            // which must be rejected even if its argmax would look like token 0.
            const uint16_t nan = 0x7e00;
            CHECK(cudaMemcpy(const_cast<void*>(engine.model()->impl()->native_head.weights()), &nan,
                             sizeof(nan), cudaMemcpyHostToDevice) == cudaSuccess);
        } else {
            engine.model()->impl()->drive.d.src = &missing;
        }
        int emitted = 0;
        guild::server::GenerationResult result;
        CHECK(!engine.generate_stream(req, [&](const auto&) { ++emitted; return true; }, result));
        CHECK(emitted == 0 && result.completion_tokens == 0 && result.text.empty());
        CHECK(result.finish_reason == "error");
        CHECK(result.error_message.find(nonfinite ? "nonfinite" : "could not produce a blob") != std::string::npos);
        CHECK(!engine.is_ready());
        CHECK(!engine.generate(req, result));
        CHECK(result.finish_reason == "error");
        CHECK(!engine.model()->impl()->ver->commit(1, err));
    }
    CHECK(guild::core::qsa_kv_host_bytes() == pinned_before);
    CHECK(guild::core::native_embed() == nullptr);
    fs::remove_all(root);
    std::puts("runtime_inference_failure_test: real CUDA failures emit no tokens, drain waits, refuse commit and release KV");
}
