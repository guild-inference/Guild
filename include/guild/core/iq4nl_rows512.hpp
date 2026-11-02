// include/guild/core/iq4nl_rows512.hpp - IQ4_NL weight rows dotted against Q8_0 activations with AVX-512BW, two rows
// to a 512-bit register: every lane computed and summed as ggml-cpu's AVX2 ggml_vec_dot_iq4_nl_q8_0 computes it, so
// each (row, token) result is bitwise that one-token dot product, for any number of tokens.  Rows of an even number
// of 32-value blocks only (ggml's last block of an odd count is a scalar step).  Call only when iq4nl_rows512_ok().
#pragma once

#include <cstddef>
#include <cstdint>

namespace guild::core::cpu_moe {

/// AVX-512 F, BW, DQ and VL, and the OS saves their state (guild-kernels-cpu's cpu_avx512bw_ok; GUILD_FORCE_AVX2=1:
/// false)
bool iq4nl_rows512_ok();

/// out[t][r] = row r . act[t] for rows [r0, r1) of `w` (`row_bytes` apart, `n` values each, n % 64 == 0) and `nt`
/// tokens' block_q8_0 rows.  Each row pair asks for the pair after next before it is read.
void iq4nl_rows512(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                   int r1);

} // namespace guild::core::cpu_moe
