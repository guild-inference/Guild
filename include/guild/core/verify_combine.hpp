// include/guild/core/verify_combine.hpp - the verify window's host-combine add, for the CPU's sums in mapped memory.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace guild::core {

/// out[t][c] = sums[t][c] * weights[t] + shared[t][c] over n_tok rows of n_embd floats, with the operations
/// guild::kernels::native_moe_combine_multi compiles to at k = 1 (it is built with --use_fast_math: a flush-to-zero
/// multiply, then a flush-to-zero add), so its bits whatever the inputs: for a weight of 1.0 the product is the flushed
/// sum exactly, and one flush-to-zero add follows in both (a fused multiply-add gives the same).  float4 loads: `sums`
/// may be mapped host memory, read over PCIe in 512-byte warp requests (the combine kernel reads 128 bytes a warp).
/// False, with nothing launched, when n_embd is not a multiple of 4, a pointer is not 16-byte aligned, or the build has
/// no such kernel (HIP); false after the launch when a CUDA error is pending (the caller's combine kernel then writes
/// every row again, and the error stays for its check).
bool combine_sums_f4(const float* sums, const float* weights, const float* shared, float* out, int64_t n_embd,
                     int n_tok, cudaStream_t stream);

} // namespace guild::core
