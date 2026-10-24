// include/guild/core/drama3_tower_kernels.hpp - the GPU pieces of Drama3Tower (src/core/drama3_tower.cu).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace guild::core::drama3 {

struct EmbedAdd {
    int n = 0; ///< rows of the add table summed into the output
    int rows[9] = {};
};

struct QkvShape {
    int n_head = 0, n_kv = 0, head_dim = 0, rope_dim = 0;
    float rope_base = 0.0f, eps = 0.0f;
};

/// The backbone's input on an audio row (lo <= tok[t] <= hi): emb[t] = (emb[t] + rows[code1[t]]) / sqrt(2), T rows
/// of n; other rows are left as they are.  rows: BF16 [codebook][n].
void backbone_embed_add(float* emb, const int32_t* tok, const int32_t* code1, const uint16_t* rows, int32_t lo,
                        int32_t hi, int n, int T, cudaStream_t st);
/// out = x * rsqrt(mean(x^2) + eps) * w (one row of n floats)
void rms_norm(const float* x, const float* w, float* out, int n, float eps, cudaStream_t st);
/// out = table row `row` (BF16, zeros when row < 0) + the sum of add_table's rows in `add`
void embed(const uint16_t* table, int row, const EmbedAdd& add, const uint16_t* add_table, float* out, int n,
           cudaStream_t st);
/// x += y
void add(float* x, const float* y, int n, cudaStream_t st);
/// From one qkv row [Q n_head*hd | gate n_head*hd | K n_kv*hd | V n_kv*hd]: per-head RMS norm of q and k (scaled by
/// q_w / k_w), pairwise RoPE on the first rope_dim channels at `pos`; q to q_out, k and v (FP16) to row `pos` of the
/// caches ([pos][n_kv][hd]).
void qk_norm_rope(const float* qkv, const float* q_w, const float* k_w, float* q_out, uint16_t* k_cache,
                  uint16_t* v_cache, const QkvShape& s, int64_t pos, cudaStream_t st);
/// floats of the split attention's scratch for up to max_pos positions
size_t attn_part_floats(const QkvShape& s, int64_t max_pos);
/// out = softmax(q k^T / sqrt(hd)) v over positions 0..n_pos-1, times sigmoid(gate); false for an unsupported shape
bool attention(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, const float* gate, float* part,
               float* out, const QkvShape& s, int64_t n_pos, cudaStream_t st);

} // namespace guild::core::drama3
