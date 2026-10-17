#pragma once
#include "guild/abi/cuda_attn_abi.h"
#include <string>
namespace guild::kernels {
struct ResolvedCudaAttn {
    guild_cuda_attn_ext_v1_t table{};
    std::string impl_id, package_id;
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
}
