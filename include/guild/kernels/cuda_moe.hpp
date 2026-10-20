#pragma once
#include "guild/abi/cuda_moe_abi.h"
#include <cstdint>
#include <string>
#include <memory>
namespace guild::kernels {
struct ResolvedCudaMoE {
    std::shared_ptr<const guild_kernel_op_desc_v1_t> binding;
    guild_cuda_moe_ext_v1_t table{};
    std::string impl_id, package_id;
};

struct CudaMoEConfigState {
    uint32_t flags = 0;
    int grouped_old = -1;
    int gr_fast = -1;
    int expert_mode = -1;
    int expert_phase = 0;

    static CudaMoEConfigState snapshot_current() noexcept;
};

class ScopedCudaMoEConfig {
public:
    explicit ScopedCudaMoEConfig(CudaMoEConfigState* state);
    ~ScopedCudaMoEConfig() noexcept;
    ScopedCudaMoEConfig(const ScopedCudaMoEConfig&) = delete;
    ScopedCudaMoEConfig& operator=(const ScopedCudaMoEConfig&) = delete;

private:
    CudaMoEConfigState* previous_ = nullptr;
};

// A copied model-time binding. Capture stores backend kernel nodes; the model's
// package lease must outlive those graphs and all outstanding stream execution.
class ScopedCudaMoE {
public:
    explicit ScopedCudaMoE(const ResolvedCudaMoE& binding);
    ~ScopedCudaMoE();
    ScopedCudaMoE(const ScopedCudaMoE&) = delete;
    ScopedCudaMoE& operator=(const ScopedCudaMoE&) = delete;

private:
    const guild_cuda_moe_ext_v1_t* previous_;
};
} // namespace guild::kernels
