// include/guild/core/kq_rows512.hpp - Q4_K and Q6_K expert rows two to an AVX-512BW pass, for any number of tokens,
// each (row, token) bitwise ggml-cpu's one-token AVX2 dot (ggml_vec_dot_q4_K_q8_K / ggml_vec_dot_q6_K_q8_K).
// pool_rows512 (pool_rows.hpp) routes Drama 3's and Ornith's Q4_K gate/up and Q4_K / Q6_K down rows here, where the
// library took ggml's dot one row at a time at one token and its AVX2-pair kernel from two (kq_rows512_test).
#pragma once

#include <cstddef>
#include <cstdint>

namespace guild::core::cpu_moe {

/// The CPU has AVX-512 F, BW, DQ and VL (and the OS saves the registers); GUILD_FORCE_AVX2=1 says no.
bool kq_rows512_ok();
/// Rows of `type` (12 = Q4_K, 14 = Q6_K) and `n` values that the kernels below take.
bool kq_rows512_supported(int type, int n);
/// Rows [r0, r1) of `w` (`row_bytes` apart) against `nt` tokens' Q8_K activations: out[t][r].
void kq_rows512(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1);
/// Q4_K gate row r and up row r (`up_off` bytes after it) together: ff[t][r] = silu(gate) * up, native_gu_rows'
/// expression.
void q4k_gu_rows512(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                    float* const* ff, int r0, int r1);

} // namespace guild::core::cpu_moe
