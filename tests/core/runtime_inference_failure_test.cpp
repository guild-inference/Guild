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
    {
        auto automatic = opts;
        automatic.plan.expert_cache_auto = true;
        automatic.plan.pcie_expert_fraction = 0;
        automatic.plan.pool_workers = 2;
        automatic.plan.spec_min_p = 0.25f;
        guild::server::NativeInferenceEngine engine(automatic);
        std::string err;
        CHECK(engine.init(err));
        auto* m = engine.model()->impl();
        CHECK(m->xcache.resident() == 8 && m->xcache.slot_offsets());
        CHECK(m->pool->workers() == 2 && engine.model()->config().spec_min_p == 0.25f);
        CHECK(m->ver->async_commit_enabled());
        guild::core::Verifier independent;
        guild::core::VerifyHits hits;
        hits.d_res = m->d_res; hits.h_res = m->host_res.data(); hits.blob = int64_t(guild::kernels::cpu::expert_layout().max_blob);
        hits.cache_base = m->xcache.device_slot(0); hits.slot_off = m->xcache.slot_offsets(); hits.n_slots = m->xcache.slots();
        CHECK(independent.init(m->wt, m->g, m->ss, hits, &m->native_head, 2, err));
        independent.set_async_commit(false);
        CHECK(!independent.async_commit_enabled() && m->ver->async_commit_enabled());
        guild::server::InferenceRequest req;
        req.prompt_tokens = {1, 2}; req.max_tokens = 2;
        guild::server::GenerationResult result;
        CHECK(engine.generate(req, result) && result.completion_tokens == 2);
        CHECK(result.gpu_cache_hits > 0 && result.ram_blobs == 0 && result.file_blobs == 0);
        CHECK(!m->ver->configure_expert_cache(hits, err)); // captured pointers cannot be rebound
    }
    {
        const auto unsupported = root / "cpu-format";
        fs::create_directories(unsupported);
        auto fallback = opts;
        fallback.paths = fixture::runtime_pack(unsupported, 1); // F16 experts: CPU supported, GPU kernel unavailable
        fallback.plan.expert_cache_auto = true;
        guild::server::NativeInferenceEngine engine(fallback);
        std::string err;
        CHECK(engine.init(err));
        auto* m = engine.model()->impl();
        CHECK(!m->xcache.valid() && m->drive.d.pcie_num == 0);
        guild::server::InferenceRequest req;
        req.prompt_tokens = {1, 2}; req.max_tokens = 2;
        guild::server::GenerationResult result;
        CHECK(engine.generate(req, result) && result.completion_tokens == 2);
    }
    {
        // PCIe offload works with no persistent GPU cache. Exercise generation,
        // a pending asynchronous commit, reset, and another fresh request.
        auto offload_opts = opts;
        offload_opts.plan.pcie_expert_fraction = 1;
        guild::server::NativeInferenceEngine offload(offload_opts);
        std::string err;
        CHECK(offload.init(err));
        auto* m = offload.model()->impl();
        CHECK(m->xcache.slots() == 0 && m->drive.d.pcie_num == 256);
        guild::server::InferenceRequest req;
        req.prompt_tokens = {1}; req.max_tokens = 2;
        guild::server::GenerationResult result;
        CHECK(offload.generate(req, result));
        CHECK(result.completion_tokens == 2 && m->drive.d.pcie_experts > 0);
        auto session = offload.model()->create_session();
        int32_t input = 1, output = -1;
        CHECK(m->ver->run(1, &input, 2, guild::runtime::drive_pool_multi, &m->drive, &output, err));
        CHECK(m->ver->commit(1, err));
        session->reset();
        CHECK(offload.is_ready());
        CHECK(offload.generate(req, result));
        CHECK(result.completion_tokens == 2 && result.tokens[0].token_id == 0);
    }
    {
        auto cached_opts = opts;
        cached_opts.paths.expert_profile_path = (root / "profile.bin").string();
        cached_opts.plan.routed_experts_in_gpu = 4;
        const std::vector<std::pair<int32_t, int32_t>> ranking = {{0, 0}, {1, 0}, {2, 0}, {3, 0},
                                                                {0, 1}, {1, 1}, {2, 1}, {3, 1}};
        std::string err;
        CHECK(guild::core::write_expert_profile(cached_opts.paths.expert_profile_path, 4, 2, ranking, err));
        guild::server::NativeInferenceEngine cached(cached_opts);
        CHECK(cached.init(err));
        auto* m = cached.model()->impl();
        CHECK(m->xcache.resident() == 4 && m->xcache.fills() == 4);
        for (int l = 0; l < 4; ++l) {
            const int32_t slot = m->xcache.slot_of(l, 0);
            CHECK(slot >= 0 && m->host_res[size_t(l * 2)] == slot);
            CHECK(m->xcache.verify_slot(slot, m->src.blob(l, 0), err,
                                       int64_t(guild::kernels::cpu::expert_layout().blob_bytes(l))));
        }
        guild::server::InferenceRequest req;
        req.prompt_tokens = {1};
        req.max_tokens = 2;
        guild::server::GenerationResult result;
        CHECK(cached.generate(req, result));
        CHECK(result.completion_tokens == 2 && result.gpu_cache_hits > 0);
    }
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
