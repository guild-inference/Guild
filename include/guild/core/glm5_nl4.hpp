// include/guild/core/glm5_nl4.hpp - IQ3_S rows in guild-kernels-cpu's NL4 form (repacked at load by
// guild::kernels::cpu::iq3s_nl4_repack), dotted against Q8_K activations without the SwiGLU fused in.
// AVX2 + FMA + F16C: call only when guild::kernels::cpu::cpu_avx2_ok().
#pragma once

#include <cstddef>
#include <cstdint>

namespace guild::core::glm5cpu {

/// out[t][r] = row r . act[t] for rows [r0, r1) of `w` (`row_bytes` apart, `n` values each), `nt` tokens (any
/// number; one token is bitwise ggml-cpu's IQ3_S dot product)
void nl4_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
              int r1);

} // namespace guild::core::glm5cpu
