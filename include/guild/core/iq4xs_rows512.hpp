// include/guild/core/iq4xs_rows512.hpp - IQ4_XS weight rows dotted against Q8_K activations with AVX-512BW, two rows
// to a 512-bit register: every lane computed and summed as ggml-cpu's AVX2 ggml_vec_dot_iq4_xs_q8_K computes it, so
// each (row, token) result is bitwise that one-token dot product, for any number of tokens.  Call only when
// iq4xs_rows512_ok().
#pragma once

#include <cstddef>
#include <cstdint>

namespace guild::core::cpu_moe {

/// AVX-512 F, BW, DQ and VL, and the OS saves their state (guild-kernels-cpu's cpu_avx512bw_ok; GUILD_FORCE_AVX2=1:
/// false)
bool iq4xs_rows512_ok();

/// out[t][r] = row r . act[t] for rows [r0, r1) of `w` (`row_bytes` apart, `n` values each, n % 256 == 0) and `nt`
/// tokens' block_q8_K rows.  Each row pair asks for the next pair before it is read; the last asks for `next` (null:
/// nothing), the first two rows of the caller's next call, `next_row_bytes` apart.
void iq4xs_rows512(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                   int r1, const uint8_t* next = nullptr, size_t next_row_bytes = 0);

} // namespace guild::core::cpu_moe
