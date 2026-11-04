// include/guild/prefill/hc_q8.hpp - the prompt path's bf16 view of a hyper-connection matrix held as the GGUF's Q8_0
// blocks (--hc q8_0, WeightRef::hc_q8).
#pragma once

#include <cstdint>

namespace guild::prefill {

/// out[i] = bf16(d * q) of the matrix's values in row-major order, rounded to nearest even: the rounding the pack's
/// --compat-bf16 conversion made, so the prompt path's bf16 GEMMs read the values they read without --hc q8_0.
/// `planes` is WeightRef::hc_q8 (rows * cols int8 values, then rows * cols / 32 f16 scales); `out` is device memory.
void hc_q8_to_bf16(const int8_t* planes, int64_t rows, int64_t cols, uint16_t* out, void* stream);

/// out[i] = the FP16 value in[i] as a float (exact), for the n values of device memory `in`.
void f16_to_f32(const uint16_t* in, float* out, int64_t n, void* stream);
/// out[i] = the BF16 value hi[i] (plus lo[i] when lo is not null) as a float, for n values.
void bf16_to_f32(const uint16_t* hi, const uint16_t* lo, float* out, int64_t n, void* stream);
/// The matrix as the GGUF's Q8_0 blocks again (an f16 scale, then 32 int8 values, 34 bytes a block), in `out`.
void hc_q8_to_blocks(const int8_t* planes, int64_t rows, int64_t cols, void* out, void* stream);

} // namespace guild::prefill
