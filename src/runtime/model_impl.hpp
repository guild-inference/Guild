#pragma once

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

#include "guild/artifact/gguf_split.hpp"
#include "guild/artifact/gpack.hpp"
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
#include "guild/kernels/cuda_attn.hpp"
#include "guild/kernels/cuda_dense.hpp"
#include "guild/kernels/cuda_moe.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/model/runner.hpp"
#include "guild/prefill/prefill.hpp"
#include "guild/runtime/batched_prefill.hpp"
#include "guild/runtime/performance.hpp"
#include "guild/spec/draft_policy.hpp"
#include "guild/spec/suffix_drafter.hpp"

#include <chrono>
#include <array>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace guild::runtime {

using Clock = std::chrono::steady_clock;

struct Drive {
    guild::core::ExpertDispatch d;
    double cpu_ms = 0;
    double validation_ms = 0;
    int64_t calls = 0;
    std::FILE* routing = nullptr;
};

inline void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                             int64_t layer) {
    Drive* t = static_cast<Drive*>(user);
    const auto validation_begin = Clock::now();
    std::string error;
    if (!core::validate_finite(x_f, size_t(n_tok * t->d.n_embd), "expert input", error)) {
        t->d.failed = true;
        throw std::runtime_error(error);
    }
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    t->validation_ms += std::chrono::duration<double, std::milli>(a - validation_begin).count();
    guild::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    const auto validation_output_begin = Clock::now();
    if (t->d.failed) {
        throw std::runtime_error(std::string(t->d.fail ? t->d.fail : "expert dispatch failed") +
                                 " (layer " + std::to_string(t->d.fail_layer) +
                                 ", expert " + std::to_string(t->d.fail_expert) + ")");
    }
    // GPU-planned rows are not written into the CPU output buffer: the captured
    // copy/combine kernels zero those rows and add the GPU result separately.
    // Validate every CPU result, rather than scanning stale GPU-owned host rows.
    std::array<bool, 128> gpu_rows{};
    const int64_t rows = n_tok * k;
    if (t->d.plan && rows <= int64_t(gpu_rows.size()) && rows <= t->d.plan->cap) {
        const auto& plan = *t->d.plan;
        for (int32_t i = 0; i < plan.counts[1]; ++i) {
            const int32_t row = plan.dst[i];
            if (row >= 0 && row < rows) gpu_rows[size_t(row)] = true;
        }
    }
    for (int64_t row = 0; row < rows; ++row) {
        if (row < int64_t(gpu_rows.size()) && gpu_rows[size_t(row)]) continue;
        if (!core::validate_finite(out + row * t->d.n_embd, size_t(t->d.n_embd), "expert output", error)) {
            t->d.failed = true;
            throw std::runtime_error(error);
        }
    }
    const auto done = Clock::now();
    t->validation_ms += std::chrono::duration<double, std::milli>(done - validation_output_begin).count();
    t->cpu_ms += std::chrono::duration<double, std::milli>(done - a).count();
    ++t->calls;
}

struct GuildModel::Impl {
    ModelPaths paths;
    model::ModelDescriptor desc;
    memory::ExecutionPlan plan;
    RuntimeConfig config;
    StartupProfile startup_profile;
    std::unique_ptr<artifact::GPackReader> gpack;
    std::shared_ptr<Tokenizer> tokenizer;
    std::shared_ptr<std::atomic<bool>> ready;
    ExpertPerformance expert_performance;
    std::shared_ptr<const model::IModelRunner> runner;
    kernels::KernelBinding cpu_binding;
    model::RunnerExecutionPlan runner_exec;
    std::string native_cpu_package;
    std::optional<guild_cpu_backend_ext_v2_t> cpu_backend;
    std::optional<kernels::ResolvedCudaDense> cuda_dense;
    std::optional<kernels::ResolvedCudaAttn> cuda_attn;
    std::optional<kernels::ResolvedCudaMoE> cuda_moe;
    model::SessionRunnerLease runner_lease;
    kernels::SessionKernelLease kernel_lease;
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
    std::unique_ptr<BatchedPrefill> batched_prefill;

    void* arena = nullptr;
    float* window_R = nullptr;
    int32_t* d_mrope = nullptr;

    std::mutex generation_mutex;

    ~Impl() {
        std::optional<kernels::ScopedCudaDense> dense_scope;
        if (cuda_dense) dense_scope.emplace(*cuda_dense);
        std::optional<kernels::ScopedCudaAttn> attn_scope;
        if (cuda_attn) attn_scope.emplace(*cuda_attn);
        std::optional<kernels::ScopedCudaMoE> moe_scope;
        if (cuda_moe) moe_scope.emplace(*cuda_moe);
        if (prefill) prefill.reset();
        if (batched_prefill) batched_prefill.reset();
        if (ver) ver.reset();
        if (mtp) mtp.reset();
        if (pool) pool.reset();
        if (main_stream) cudaStreamSynchronize(main_stream);
        moe_scope.reset();
        attn_scope.reset();
        dense_scope.reset();
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
            cuda_moe.reset();
            cuda_attn.reset();
            cuda_dense.reset();
            cpu_binding.reset();
            runner_exec.bindings.clear();
            runner.reset();
            gpack.reset();
            kernel_lease = {};
            runner_lease = {};
            cudaStreamDestroy(main_stream);
            main_stream = nullptr;
        } else {
            cuda_moe.reset();
            cuda_attn.reset();
            cuda_dense.reset();
            cpu_binding.reset();
            runner_exec.bindings.clear();
            runner.reset();
            gpack.reset();
            kernel_lease = {};
            runner_lease = {};
        }
    }
};

bool bind_required_ple(GuildModel::Impl& model, const std::vector<std::string>& shards, std::string& err);
bool bind_required_ple_gpack(GuildModel::Impl& model, const artifact::GPackReader& gpack, std::string& err);

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
