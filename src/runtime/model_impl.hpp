#pragma once

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

#include "guild/artifact/gguf_split.hpp"
#include "guild/core/device.hpp"
#include "guild/core/expert_cache.hpp"
#include "guild/core/expert_source.hpp"
#include "guild/core/layer.hpp"
#include "guild/core/mtp.hpp"
#include "guild/core/native_dense.hpp"
#include "guild/core/native_head.hpp"
#include "guild/core/on_device.hpp"
#include "guild/core/pinned.hpp"
#include "guild/core/session.hpp"
#include "guild/core/verify.hpp"
#include "guild/core/weights.hpp"
#include "guild/core/validation.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include "guild/kernels/cpu/pool.hpp"
#include "guild/kernels/mrope.hpp"
#include "guild/kernels/native_rope.hpp"
#include "guild/kernels/ngram.hpp"
#include "guild/kernels/ple.hpp"
#include "guild/kernels/s2_expert_grouped.hpp"
#include "guild/kernels/shared_expert.hpp"
#include "guild/kernels/native_moe.hpp"
#include "guild/kernels/native_gdn.hpp"
#include "guild/kernels/native_router.hpp"
#include "guild/kernels/native_qsa.hpp"
#include "guild/kernels/native_qsa_indexer.hpp"
#include "guild/prefill/prefill.hpp"
#include "guild/spec/draft_policy.hpp"
#include "guild/spec/suffix_drafter.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace guild::runtime {

using Clock = std::chrono::steady_clock;

struct Drive {
    guild::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    std::FILE* routing = nullptr;
};

inline void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer) {
    Drive* t = static_cast<Drive*>(user);
    std::string error;
    if (!core::validate_finite(x_f, size_t(n_tok * t->d.n_embd), "expert input", error)) {
        t->d.failed = true;
        throw std::runtime_error(error);
    }
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    guild::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    if (t->d.failed) {
        throw std::runtime_error(std::string(t->d.fail ? t->d.fail : "expert dispatch failed") +
                                 " (layer " + std::to_string(t->d.fail_layer) +
                                 ", expert " + std::to_string(t->d.fail_expert) + ")");
    }
    if (!core::validate_finite(out, size_t(n_tok * k * t->d.n_embd), "expert output", error)) {
        t->d.failed = true;
        throw std::runtime_error(error);
    }
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
}

struct GuildModel::Impl {
    ModelPaths paths;
    model::ModelDescriptor desc;
    memory::ExecutionPlan plan;
    RuntimeConfig config;
    core::ModelGeometry g;
    int64_t K = 10;

    cudaStream_t main_stream = nullptr;
    core::Doorbell db;
    core::NativeEmbed native_embed;
    core::WeightTable wt;
    core::NativeDense native_dense;
    core::NativeHead native_head;
    core::FileExpertSource src;
    core::ExpertSource* srcp = nullptr;
    core::ExpertCache xcache;
    std::vector<int32_t> host_res;
    int32_t* d_res = nullptr;
    std::unique_ptr<kernels::cpu::ExpertPool> pool;
    kernels::PleTable ple_table;
    bool ple_required = false;
    std::vector<float> ple_emb_host;
    float* ple_emb_dev = nullptr;
    void* ple_scratch = nullptr;
    std::unique_ptr<core::MtpDrafter> mtp;
    std::unique_ptr<core::Verifier> ver;
    Drive drive;

    // Primary session state & prefill engine
    core::SessionState ss;
    void* sbuf = nullptr;
    std::unique_ptr<prefill::Prefill> prefill;

    void* arena = nullptr;
    float* window_R = nullptr;
    int32_t* d_mrope = nullptr;

    std::mutex generation_mutex;

    ~Impl() {
        if (prefill) prefill.reset();
        if (ver) ver.reset();
        if (mtp) mtp.reset();
        if (pool) pool.reset();
        if (main_stream) cudaStreamSynchronize(main_stream);
        if (ple_emb_dev) cudaFree(ple_emb_dev);
        if (ple_scratch) cudaFree(ple_scratch);
        core::session_release(ss);
        for (int64_t j = 0; ss.qsa_states && j < ss.qsa_alloc; ++j) {
            core::qsa_state_release_host(ss.qsa_states[ss.qsa_ord0 + j]);
        }
        delete[] ss.qsa_states;
        ss.qsa_states = nullptr;
        if (core::native_embed() == &native_embed) core::set_native_embed(nullptr);
        if (sbuf) { cudaFree(sbuf); sbuf = nullptr; }
        if (d_res) { cudaFree(d_res); d_res = nullptr; }
        if (arena) { cudaFree(arena); arena = nullptr; }
        if (d_mrope) { cudaFree(d_mrope); d_mrope = nullptr; }
        if (window_R) { cudaFree(window_R); window_R = nullptr; }
        core::doorbell_free(db);
        if (main_stream) {
            cudaStreamSynchronize(main_stream);
            cudaStreamDestroy(main_stream);
            main_stream = nullptr;
        }
    }
};

bool bind_required_ple(GuildModel::Impl& model, const std::vector<std::string>& shards, std::string& err);

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
