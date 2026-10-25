#pragma once

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

#include "guild/artifact/gguf_split.hpp"
#include "guild/artifact/gpack.hpp"
#include "guild/gexec/gexec_optimizer.hpp"
#include "guild/kernels/cpu/gexec_expert.hpp"
#include "guild/core/device.hpp"
#include "guild/core/expert_cache.hpp"
#include "guild/core/expert_source.hpp"
#include "guild/core/layer.hpp"
#include "guild/core/drama3_codec.hpp"
#include "guild/core/drama3_tower.hpp"
#include "guild/core/glm5.hpp"
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
#include "guild/kernels/gdn_multi.hpp"
#include "guild/kernels/native_moe.hpp"
#include "guild/kernels/native_gdn.hpp"
#include "guild/kernels/native_router.hpp"
#include "guild/kernels/native_qsa.hpp"
#include "guild/kernels/native_qsa_indexer.hpp"
#include "guild/kernels/kv_stream.hpp"
#include "guild/kernels/cuda_attn.hpp"
#include "guild/kernels/cuda_dense.hpp"
#include "guild/kernels/cuda_moe.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/model/runner.hpp"
#include "guild/prefill/prefill.hpp"
#include "guild/runtime/batched_prefill.hpp"
#include "guild/runtime/performance.hpp"
#include "guild/spec/adaptive_scheduler.hpp"
#include "guild/spec/draft_policy.hpp"
#include "guild/spec/retrieval_drafter.hpp"
#include "guild/spec/suffix_drafter.hpp"

#include <chrono>
#include <array>
#include <deque>
#include <memory>
#include <new>
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
    if (t->routing) { // tools/make_profile.py's record: layer, k, k ids, k weights (not known here: zeros)
        static const float zeros[64] = {};
        for (int64_t i = 0; i < n_tok; ++i) {
            const int32_t head[2] = {int32_t(layer), int32_t(k)};
            std::fwrite(head, sizeof head, 1, t->routing);
            std::fwrite(ids + i * k, sizeof(int32_t), size_t(k), t->routing);
            for (int64_t w = 0; w < k; w += 64)
                std::fwrite(zeros, sizeof(float), size_t(std::min<int64_t>(k - w, 64)), t->routing);
        }
    }
    const Clock::time_point a = Clock::now();
    t->validation_ms += std::chrono::duration<double, std::milli>(a - validation_begin).count();
    guild::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    const auto validation_output_begin = Clock::now();
    if (t->d.failed) {
        throw std::runtime_error(std::string(t->d.fail ? t->d.fail : "expert dispatch failed") + " (layer " +
                                 std::to_string(t->d.fail_layer) + ", expert " + std::to_string(t->d.fail_expert) +
                                 ")");
    }
    // GPU-planned rows are not written into the CPU output buffer: the captured
    // copy/combine kernels zero those rows and add the GPU result separately.
    // Validate every CPU result, rather than scanning stale GPU-owned host rows.
    if (!t->d.pool || !t->d.pool->last_run_all_finite()) {
        std::array<bool, 128> gpu_rows{};
        const int64_t rows = n_tok * k;
        if (t->d.plan && rows <= int64_t(gpu_rows.size()) && rows <= t->d.plan->cap) {
            const auto& plan = *t->d.plan;
            for (int32_t i = 0; i < plan.counts[1]; ++i) {
                const int32_t row = plan.dst[i];
                if (row >= 0 && row < rows)
                    gpu_rows[size_t(row)] = true;
            }
        }
        for (int64_t row = 0; row < rows; ++row) {
            if (row < int64_t(gpu_rows.size()) && gpu_rows[size_t(row)])
                continue;
            if (!core::validate_finite(out + row * t->d.n_embd, size_t(t->d.n_embd), "expert output", error)) {
                t->d.failed = true;
                throw std::runtime_error(error);
            }
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
    std::unique_ptr<gexec::GExecReader> gexec;
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
    std::unique_ptr<core::Drama3Tower> tower; ///< Drama 3 (ModelPaths::drama3_dir): the codebook tower
    std::unique_ptr<core::Drama3Codec> codec; ///< ... and the codec that turns frames into audio
    /// GLM-5.3-Flash (a glm5-next GGUF) runs on its own engine (src/runtime/glm5.cpp); the rest stays empty
    std::unique_ptr<core::Glm5Engine> glm;
    std::vector<int32_t> glm_ctx, glm_snap;             ///< the tokens of the live context and of the snapshot
    std::vector<float> glm_ctx_logits, glm_snap_logits; ///< their last logits, when known
    std::unique_ptr<core::Verifier> ver;
    Drive drive;
    spec::GExecSpecCalibration spec_calibration;
    std::atomic<int> active_sessions{0};

    // Primary session state & prefill engine
    std::shared_ptr<core::ModelExecutionState> exec_state;
    core::SessionState ss;
    void* sbuf = nullptr;
    std::unique_ptr<prefill::Prefill> prefill;
    std::unique_ptr<BatchedPrefill> batched_prefill;

    void* arena = nullptr;
    float* window_R = nullptr;
    int32_t* d_mrope = nullptr;

    // A checkpoint's recurrent state: 118 MB on Qwen3.8-Flash-Next, saved twice per request.  Page-locked so the copy
    // is a DMA, and never zero-filled; buffers are recycled through `spare_gdn` rather than freed.  Falls back to
    // pageable memory when the driver will not lock more.
    struct HostBytes {
        uint8_t* p = nullptr;
        size_t n = 0;
        bool pinned = false;
        HostBytes() = default;
        HostBytes(const HostBytes&) = delete;
        HostBytes& operator=(const HostBytes&) = delete;
        HostBytes(HostBytes&& o) noexcept : p(o.p), n(o.n), pinned(o.pinned) {
            o.p = nullptr;
            o.n = 0;
        }
        HostBytes& operator=(HostBytes&& o) noexcept {
            if (this != &o) {
                release();
                p = o.p;
                n = o.n;
                pinned = o.pinned;
                o.p = nullptr;
                o.n = 0;
            }
            return *this;
        }
        ~HostBytes() {
            release();
        }
        bool allocate(size_t bytes) {
            if (bytes == n && p != nullptr)
                return true;
            release();
            if (cudaMallocHost(reinterpret_cast<void**>(&p), bytes) == cudaSuccess) {
                pinned = true;
            } else {
                (void) cudaGetLastError();
                p = new (std::nothrow) uint8_t[bytes];
                pinned = false;
            }
            n = p != nullptr ? bytes : 0;
            return p != nullptr;
        }
        void release() {
            if (p != nullptr) {
                if (pinned)
                    cudaFreeHost(p);
                else
                    delete[] p;
            }
            p = nullptr;
            n = 0;
        }
        bool empty() const {
            return n == 0;
        }
    };

    // KV + Recurrent State Prefix Cache for multi-turn & agentic continuations
    struct PrefixStateCheckpoint {
        std::vector<int32_t> tokens;
        HostBytes gdn;
        std::vector<uint8_t> ple;
        int32_t ple_prev[2] = {-1, -1};
        int32_t ple_token = -1;
        std::vector<uint8_t> qsa_tails;
        std::vector<uint8_t> qsa_dead;
        std::vector<uint8_t> qsa_block_pos;
    };
    std::vector<int32_t> live_committed_tokens;
    std::deque<PrefixStateCheckpoint> prefix_checkpoints;
    std::vector<HostBytes> spare_gdn; // recycled checkpoint buffers (locking 118 MB again costs more than the copy)

    void recycle_checkpoint(PrefixStateCheckpoint& cp) {
        if (!cp.gdn.empty() && spare_gdn.size() < 2)
            spare_gdn.push_back(std::move(cp.gdn));
    }
    void clear_prefix_cache() {
        live_committed_tokens.clear();
        for (auto& cp : prefix_checkpoints)
            recycle_checkpoint(cp);
        prefix_checkpoints.clear();
    }
    /// Drops the checkpoints longer than `len`: the cells after it are about to be overwritten.
    void prune_prefix_checkpoints(size_t len) {
        for (auto it = prefix_checkpoints.begin(); it != prefix_checkpoints.end();) {
            if (it->tokens.size() > len) {
                recycle_checkpoint(*it);
                it = prefix_checkpoints.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool save_prefix_checkpoint(const std::vector<int32_t>& committed) {
        if (committed.empty())
            return false;
        if (cudaStreamSynchronize(main_stream) != cudaSuccess)
            return false;

        PrefixStateCheckpoint cp;
        cp.tokens = committed;

        // The slot this checkpoint takes - the same length's (replaced), else the oldest-but-one when the ring is
        // full (the earliest, the system + tools prompt, is retained) - lends its buffer, so a steady stream of
        // requests locks no new memory.
        size_t same = prefix_checkpoints.size();
        for (size_t i = 0; i < prefix_checkpoints.size(); ++i)
            if (prefix_checkpoints[i].tokens.size() == cp.tokens.size())
                same = i;
        if (same == prefix_checkpoints.size() && prefix_checkpoints.size() >= 8) {
            recycle_checkpoint(prefix_checkpoints[1]);
            prefix_checkpoints.erase(prefix_checkpoints.begin() + 1);
            same = prefix_checkpoints.size();
        }
        if (same < prefix_checkpoints.size())
            cp.gdn = std::move(prefix_checkpoints[same].gdn);
        else if (!spare_gdn.empty()) {
            cp.gdn = std::move(spare_gdn.back());
            spare_gdn.pop_back();
        }
        auto fail = [&] {
            (void) cudaStreamSynchronize(main_stream); // no copy may still be writing the buffer being recycled
            if (same < prefix_checkpoints.size())      // its buffer was lent to this one: it holds no state now
                prefix_checkpoints.erase(prefix_checkpoints.begin() + static_cast<std::ptrdiff_t>(same));
            recycle_checkpoint(cp);
            return false;
        };

        const size_t gdn_per_layer =
            static_cast<size_t>(g.ssm_state_size) * static_cast<size_t>(g.ssm_v_heads) *
                static_cast<size_t>(g.ssm_state_size) +
            static_cast<size_t>(g.ssm_conv_channels) * static_cast<size_t>(std::max<int64_t>(0, g.ssm_d_conv - 1));
        const size_t gdn_bytes =
            static_cast<size_t>(std::max<int64_t>(0, ss.gdn_alloc)) * gdn_per_layer * sizeof(float);
        if (gdn_bytes > 0 && ss.gdn_state != nullptr) {
            if (!cp.gdn.allocate(gdn_bytes) ||
                cudaMemcpyAsync(cp.gdn.p, ss.gdn_state, gdn_bytes, cudaMemcpyDeviceToHost, main_stream) != cudaSuccess)
                return fail();
        } else {
            recycle_checkpoint(cp);
        }

        if (ss.ple_hist != nullptr) {
            const size_t ple_bytes =
                static_cast<size_t>(kernels::NG_HIST) * static_cast<size_t>(kernels::NG_HC_DIM) * sizeof(float);
            cp.ple.resize(ple_bytes);
            if (cudaMemcpyAsync(cp.ple.data(), ss.ple_hist, ple_bytes, cudaMemcpyDeviceToHost, main_stream) !=
                cudaSuccess)
                return fail();
        }
        cp.ple_prev[0] = ss.ple_prev[0];
        cp.ple_prev[1] = ss.ple_prev[1];
        cp.ple_token = ss.ple_token;

        if (g.idx_key_dim > 0 && ss.qsa_states != nullptr && ss.qsa_alloc > 0) {
            const size_t qsa_layers = static_cast<size_t>(ss.qsa_alloc);
            const size_t tail_bytes =
                static_cast<size_t>(kernels::qsa_real_shapes().idx_block - 1) * static_cast<size_t>(g.idx_key_dim) * 4;
            const size_t dead_bytes = static_cast<size_t>(g.idx_key_dim) * 4;
            const size_t pos_bytes = sizeof(int32_t);
            cp.qsa_tails.resize(qsa_layers * tail_bytes);
            cp.qsa_dead.resize(qsa_layers * dead_bytes);
            cp.qsa_block_pos.resize(qsa_layers * pos_bytes);

            for (size_t j = 0; j < qsa_layers; ++j) {
                const auto& st = ss.qsa_states[static_cast<size_t>(ss.qsa_ord0) + j];
                if (cudaMemcpyAsync(cp.qsa_tails.data() + j * tail_bytes, st.idx_tail, tail_bytes,
                                    cudaMemcpyDeviceToHost, main_stream) != cudaSuccess ||
                    cudaMemcpyAsync(cp.qsa_dead.data() + j * dead_bytes, st.idx_dead, dead_bytes,
                                    cudaMemcpyDeviceToHost, main_stream) != cudaSuccess ||
                    cudaMemcpyAsync(cp.qsa_block_pos.data() + j * pos_bytes, st.idx_block_pos, pos_bytes,
                                    cudaMemcpyDeviceToHost, main_stream) != cudaSuccess) {
                    return fail();
                }
            }
        }
        if (cudaStreamSynchronize(main_stream) != cudaSuccess)
            return fail();

        if (same < prefix_checkpoints.size())
            prefix_checkpoints[same] = std::move(cp);
        else
            prefix_checkpoints.push_back(std::move(cp));
        return true;
    }

    bool restore_prefix_checkpoint(const PrefixStateCheckpoint& cp) {
        if (cp.tokens.empty())
            return false;
        if (!cp.gdn.empty() && ss.gdn_state != nullptr) {
            if (cudaMemcpyAsync(ss.gdn_state, cp.gdn.p, cp.gdn.n, cudaMemcpyHostToDevice, main_stream) != cudaSuccess)
                return false;
        }
        if (!cp.ple.empty() && ss.ple_hist != nullptr) {
            if (cudaMemcpyAsync(ss.ple_hist, cp.ple.data(), cp.ple.size(), cudaMemcpyHostToDevice, main_stream) !=
                cudaSuccess)
                return false;
        }
        ss.ple_prev[0] = cp.ple_prev[0];
        ss.ple_prev[1] = cp.ple_prev[1];
        ss.ple_token = cp.ple_token;

        if (ss.qsa_states != nullptr && ss.qsa_alloc > 0) {
            const size_t qsa_layers = static_cast<size_t>(ss.qsa_alloc);
            const size_t tail_bytes =
                static_cast<size_t>(kernels::qsa_real_shapes().idx_block - 1) * static_cast<size_t>(g.idx_key_dim) * 4;
            const size_t dead_bytes = static_cast<size_t>(g.idx_key_dim) * 4;
            const size_t pos_bytes = sizeof(int32_t);
            const size_t pooled_row = cp.tokens.size() / static_cast<size_t>(kernels::qsa_real_shapes().idx_block);

            for (size_t j = 0; j < qsa_layers; ++j) {
                const auto& st = ss.qsa_states[static_cast<size_t>(ss.qsa_ord0) + j];
                if (st.kv_mode == 3 && st.shares_staging) {
                    kernels::kv_stream_invalidate(st.map, main_stream);
                }
                if (g.idx_key_dim > 0 && cp.qsa_tails.size() == qsa_layers * tail_bytes) {
                    if (cudaMemcpyAsync(st.idx_tail, cp.qsa_tails.data() + j * tail_bytes, tail_bytes,
                                        cudaMemcpyHostToDevice, main_stream) != cudaSuccess ||
                        cudaMemcpyAsync(st.idx_dead, cp.qsa_dead.data() + j * dead_bytes, dead_bytes,
                                        cudaMemcpyHostToDevice, main_stream) != cudaSuccess ||
                        cudaMemcpyAsync(st.idx_block_pos, cp.qsa_block_pos.data() + j * pos_bytes, pos_bytes,
                                        cudaMemcpyHostToDevice, main_stream) != cudaSuccess ||
                        cudaMemcpyAsync(st.idx_pooled + pooled_row * static_cast<size_t>(g.idx_key_dim),
                                        cp.qsa_dead.data() + j * dead_bytes, dead_bytes, cudaMemcpyHostToDevice,
                                        main_stream) != cudaSuccess) {
                        return false;
                    }
                }
            }
        }
        return cudaStreamSynchronize(main_stream) == cudaSuccess;
    }

    std::mutex generation_mutex;

    ~Impl() {
        if (drive.routing)
            std::fclose(drive.routing);
        core::ScopedModelExecution exec_scope(exec_state);
        std::optional<kernels::ScopedCudaDense> dense_scope;
        if (cuda_dense)
            dense_scope.emplace(*cuda_dense);
        std::optional<kernels::ScopedCudaAttn> attn_scope;
        if (cuda_attn)
            attn_scope.emplace(*cuda_attn);
        std::optional<kernels::ScopedCudaMoE> moe_scope;
        if (cuda_moe)
            moe_scope.emplace(*cuda_moe);
        if (prefill)
            prefill.reset();
        if (batched_prefill)
            batched_prefill.reset();
        if (ver)
            ver.reset();
        if (mtp)
            mtp.reset();
        if (pool)
            pool.reset();
        if (main_stream)
            cudaStreamSynchronize(main_stream);
        moe_scope.reset();
        attn_scope.reset();
        dense_scope.reset();
        if (ple_emb_dev)
            cudaFree(ple_emb_dev);
        if (ple_scratch)
            cudaFree(ple_scratch);
        core::session_release(ss);
        for (int64_t j = 0; ss.qsa_states && j < ss.qsa_alloc; ++j) {
            core::qsa_state_release_host(ss.qsa_states[ss.qsa_ord0 + j]);
        }
        delete[] ss.qsa_states;
        ss.qsa_states = nullptr;
        if (core::native_embed() == &native_embed)
            core::set_native_embed(nullptr);
        if (sbuf) {
            cudaFree(sbuf);
            sbuf = nullptr;
        }
        if (d_res) {
            cudaFree(d_res);
            d_res = nullptr;
        }
        if (arena) {
            cudaFree(arena);
            arena = nullptr;
        }
        if (d_mrope) {
            cudaFree(d_mrope);
            d_mrope = nullptr;
        }
        if (window_R) {
            cudaFree(window_R);
            window_R = nullptr;
        }
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
        (void) cudaGetLastError();
    }
};

bool bind_required_ple(GuildModel::Impl& model, const std::vector<std::string>& shards, std::string& err);
bool bind_required_ple_gpack(GuildModel::Impl& model, const artifact::GPackReader& gpack, std::string& err);

/// GLM-5.3-Flash (src/runtime/glm5.cpp)
bool glm5_is_model(const std::string& gguf_path);
bool glm5_load(GuildModel::Impl& m, const ModelPaths& paths, const memory::ExecutionPlan& plan, std::string& err);
bool glm5_generate(GuildModel::Impl& m, const GenerationRequest& req, const std::vector<int64_t>& prompt,
                   TokenStreamCallback on_token, RuntimeTelemetry& tel, std::string& err,
                   PrefillProgressCallback on_prefill, std::atomic<bool>& cancelled);

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
