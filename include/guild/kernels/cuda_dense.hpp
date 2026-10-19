#pragma once
#include "guild/abi/cuda_dense_abi.h"
#include <string>
#include <memory>
namespace guild::kernels {
struct ResolvedCudaDense {
    std::shared_ptr<const guild_kernel_op_desc_v1_t> binding;
    guild_cuda_dense_ext_v1_t table{};
    std::string impl_id,package_id;
};
// A copied model-time binding. Capture stores backend kernel nodes; the model's
// package lease must outlive those graphs and all outstanding stream execution.
class ScopedCudaDense {
public:
    explicit ScopedCudaDense(const ResolvedCudaDense& binding);
    ~ScopedCudaDense();
    ScopedCudaDense(const ScopedCudaDense&)=delete;
    ScopedCudaDense& operator=(const ScopedCudaDense&)=delete;
private:
    const guild_cuda_dense_ext_v1_t* previous_;
};
}
