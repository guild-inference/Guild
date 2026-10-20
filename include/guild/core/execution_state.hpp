#pragma once

#include "guild/kernels/cpu/expert_layout.hpp"
#include "guild/kernels/cuda_attn.hpp"
#include "guild/kernels/cuda_moe.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace guild::core {

class NativeEmbed;

/// Per-model execution configuration owned by a GuildModel instance and shared with its SessionState.
/// Isolates layer scheduling, KV residency/format/staging, RoPE scaling/tables, native token embedding,
/// expert pack layout, and CUDA attention/MoE bridge settings across multiple models in one process.
struct ModelExecutionState {
    // Layer execution settings (src/core/layer.cpp)
    bool shared_early = true;
    bool fused_gr = false;
    bool fast_attn = true;
    bool publish_kernel = true;
    bool fused_gdn = true;
    bool fast_select = true;
    bool native_bf16_projections = false;
    bool native_flash_attn_short = false;

    // KV residency, precision, and staging settings (src/core/layer.cpp, src/core/session.cpp)
    int64_t kv_resident = 0;
    bool kv_host_only = false;
    bool kv_int8 = false;
    bool kv_int8_rot = false;
    bool kv_q4 = false;
    bool kv_hybrid = false;
    int kv_share_staging_mode = -1;

    // Native token embedding (src/core/native_head.cpp)
    const NativeEmbed* native_embed = nullptr;

    // Expert pack layout (src/kernels/cpu/pack_layout.cpp)
    kernels::cpu::ExpertLayout expert_layout{};

    // CUDA bridge configuration states (src/kernels/cuda_attn_bridge.cpp, src/kernels/cuda_moe_bridge.cpp)
    kernels::CudaAttnConfigState cuda_attn{};
    kernels::CudaMoEConfigState cuda_moe{};

    /// Snapshots the currently active (or process-default) execution configuration.
    static ModelExecutionState snapshot_current();
};

/// Returns the active thread-local ModelExecutionState, or nullptr when using process defaults.
ModelExecutionState* active_model_execution_state() noexcept;
std::shared_ptr<ModelExecutionState> active_model_execution_state_shared() noexcept;

/// Binds a ModelExecutionState to the calling thread for the lifetime of the guard.
/// Re-entering with the already-active state or nullptr is a fast no-op.
class ScopedModelExecution {
public:
    explicit ScopedModelExecution(std::shared_ptr<ModelExecutionState> state);
    explicit ScopedModelExecution(ModelExecutionState* state);
    ~ScopedModelExecution() noexcept;

    ScopedModelExecution(const ScopedModelExecution&) = delete;
    ScopedModelExecution& operator=(const ScopedModelExecution&) = delete;

private:
    void enter(ModelExecutionState* raw, std::shared_ptr<ModelExecutionState> shared);

    bool active_ = false;
    ModelExecutionState* prev_raw_ = nullptr;
    std::shared_ptr<ModelExecutionState> prev_shared_;
    std::optional<kernels::cpu::ScopedExpertLayout> layout_scope_;
    std::optional<kernels::ScopedCudaAttnConfig> attn_scope_;
    std::optional<kernels::ScopedCudaMoEConfig> moe_scope_;
};

} // namespace guild::core
