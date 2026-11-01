// include/guild/core/q8_0_rows512.hpp - Q8_0 weight rows dotted against Q8_0 activations of several tokens with
// AVX-512BW: two rows to a 512-bit register, every token's lanes computed as ggml-cpu's AVX2 ggml_vec_dot_q8_0_q8_0
// computes them and summed in its order, so each (row, token) result is bitwise that one-token dot product.
// Call only when q8_0_rows512_ok().
#pragma once

#include <cstddef>
#include <cstdint>

namespace guild::core::cpu_moe {

/// the most tokens one call takes
constexpr int kQ8Rows512MaxT = 16;

/// AVX-512 F, BW, DQ and VL, and the OS saves their state (guild-kernels-cpu's cpu_avx512bw_ok; GUILD_FORCE_AVX2=1:
/// false)
bool q8_0_rows512_ok();

/// The bytes q8_0_pack writes for `nt` tokens of `n` values (a multiple of 64)
size_t q8_0_packed_bytes(int n, int nt);

/// `nt` <= kQ8Rows512MaxT tokens' Q8_0 rows (act[t], `n` values) packed block by block, the form q8_0_rows512 reads:
/// every token's values of block 0, then of block 1, ..., then the blocks' scales as floats in the same order
void q8_0_pack(const void* const* act, int nt, int n, uint8_t* packed);

/// out[t][r] = row r . token t for rows [r0, r1) of `w` (`row_bytes` apart, `n` values each, n % 32 == 0) and the
/// `nt` tokens q8_0_pack packed.  Each row pair asks for the next pair before it is read; the last asks for `next`
/// (null: nothing), the first two rows of the caller's next call, `next_row_bytes` apart.
void q8_0_rows512(const uint8_t* w, size_t row_bytes, int n, const uint8_t* packed, int nt, float* const* out, int r0,
                  int r1, const uint8_t* next = nullptr, size_t next_row_bytes = 0);

} // namespace guild::core::cpu_moe
