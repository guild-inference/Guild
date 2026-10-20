#pragma once
#include "guild/abi/cuda_attn_abi.h"
#include "guild/kernels/rope_scaling.hpp"
#include <string>
#include <memory>
namespace guild::kernels {
struct ResolvedCudaAttn {
    std::shared_ptr<const guild_kernel_op_desc_v1_t> binding;
    guild_cuda_attn_ext_v1_t table{};
    std::string impl_id, package_id;
};

struct CudaAttnConfigState {
    static constexpr int kMaxDevices = 64;
    bool native_qsa = false;
    bool native_rope = false;
    bool native_gdn = false;
    bool native_qsa_indexer = false;
    bool native_qsa_score = false;
    RopeScaling rope_scaling{};
    const int32_t* mrope_tab[kMaxDevices]{};
    struct RopeDeviceReg {
        const float* cos = nullptr;
        const float* sin = nullptr;
        int max_pos = 0;
        RopeScaling scaling{};
    } rope_tab[kMaxDevices]{};

    static CudaAttnConfigState snapshot_current() noexcept;
};

class ScopedCudaAttnConfig {
public:
    explicit ScopedCudaAttnConfig(CudaAttnConfigState* state);
    ~ScopedCudaAttnConfig() noexcept;
    ScopedCudaAttnConfig(const ScopedCudaAttnConfig&) = delete;
    ScopedCudaAttnConfig& operator=(const ScopedCudaAttnConfig&) = delete;

private:
    CudaAttnConfigState* previous_ = nullptr;
};

// A copied model-time binding. Capture stores backend kernel nodes; the model's
// package lease must outlive those graphs and all outstanding stream execution.
class ScopedCudaAttn {
public:
    explicit ScopedCudaAttn(const ResolvedCudaAttn& binding);
    ~ScopedCudaAttn();
    ScopedCudaAttn(const ScopedCudaAttn&) = delete;
    ScopedCudaAttn& operator=(const ScopedCudaAttn&) = delete;

private:
    const guild_cuda_attn_ext_v1_t* previous_;
};
} // namespace guild::kernels
