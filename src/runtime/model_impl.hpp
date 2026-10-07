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
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    guild::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
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

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
