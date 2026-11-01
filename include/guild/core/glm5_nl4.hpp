// include/guild/core/glm5_nl4.hpp - IQ3_S rows in guild-kernels-cpu's NL4 form (repacked at load by
// guild::kernels::cpu::iq3s_nl4_repack), dotted against Q8_K activations without the SwiGLU fused in.
// AVX2 + FMA + F16C: call only when guild::kernels::cpu::cpu_avx2_ok(); two rows to an AVX-512BW register where the
// CPU has it (nl4_rows512_ok).
#pragma once

#include <cstddef>
#include <cstdint>

namespace guild::core::glm5cpu {

/// out[t][r] = row r . act[t] for rows [r0, r1) of `w` (`row_bytes` apart, `n` values each), `nt` tokens (any
/// number; one token is bitwise ggml-cpu's IQ3_S dot product)
void nl4_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
              int r1);
/// The AVX-2 rows nl4_rows takes when nl4_rows512_ok() is false (and the reference of its AVX-512 rows)
void nl4_rows_avx2(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                   int r1);
/// Whether nl4_rows takes two rows to an AVX-512BW register (the CPU has AVX-512 F, BW, DQ and VL, the OS saves
/// its state, and GUILD_NL4_512 is not 0): the same bits as the AVX-2 rows, in ggml's order (GUILD_IQ_GGML_ORDER=0
/// keeps the AVX-2 rows)
bool nl4_rows512_ok();

} // namespace guild::core::glm5cpu
