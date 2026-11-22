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
/// out = x * rsqrt(mean(x^2) + eps) * w, for `rows` rows of n floats (x's rows xs floats apart, out's os; 0 = n)
void rms_norm(const float* x, const float* w, float* out, int n, float eps, cudaStream_t st, int rows = 1, int xs = 0,
              int os = 0);
/// Row r of out (n floats) = table row rows[r] (zeros when < 0) + the sum of add_table's rows add[9 r ..] (add_n[r]
/// of them); rows / add / add_n are device arrays.
void embed_rows(const uint16_t* table, const int32_t* rows, const int32_t* add, const int32_t* add_n,
                const uint16_t* add_table, float* out, int n, int n_rows, cudaStream_t st);
/// out = table row `row` (BF16, zeros when row < 0) + the sum of add_table's rows in `add`
void embed(const uint16_t* table, int row, const EmbedAdd& add, const uint16_t* add_table, float* out, int n,
           cudaStream_t st);
/// x += y
void add(float* x, const float* y, int n, cudaStream_t st);
/// From one qkv row [Q n_head*hd | gate n_head*hd | K n_kv*hd | V n_kv*hd]: per-head RMS norm of q and k (scaled by
/// q_w / k_w), pairwise RoPE on the first rope_dim channels at `pos`; q to q_out, k and v (FP16) to row `pos` of the
/// caches ([pos][n_kv][hd]).  `rows` > 1: consecutive qkv rows at positions pos, pos + 1, ... (q rows n_head * hd).
void qk_norm_rope(const float* qkv, const float* q_w, const float* k_w, float* q_out, uint16_t* k_cache,
                  uint16_t* v_cache, const QkvShape& s, int64_t pos, cudaStream_t st, int rows = 1);
/// floats of the split attention's scratch for up to max_pos positions
size_t attn_part_floats(const QkvShape& s, int64_t max_pos);
/// out = softmax(q k^T / sqrt(hd)) v over positions 0..n_pos-1, times sigmoid(gate); false for an unsupported shape
bool attention(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, const float* gate, float* part,
               float* out, const QkvShape& s, int64_t n_pos, cudaStream_t st);

// ---- fused small kernels (one row of kFusedDim floats, 1024 threads).  Each writes what the kernels it replaces
// write, bit for bit: rms_norm's reduction by its 256 threads and the same trees, and the Q8_1 blocks of the native
// MMVQ's own quantize kernel (guild-kernels-cuda's native_quantize_q8_1, 36-byte blocks: half2 d and sum, 32 int8).
// The tower checks them at load against those kernels (fused_self_check) and keeps the separate kernels on any
// difference.

/// The only row width the fused kernels take
constexpr int kFusedDim = 2048;
/// Bytes of the Q8_1 block the fused kernels write (native_q8_1_bytes(32, 1) must equal it)
constexpr int kFusedQ81Bytes = 36;
/// The layer's input: xq's first kFusedDim / 32 Q8_1 blocks = rms_norm(e, w0), the next = rms_norm(h, w1), where e is
/// embed(table, row, add, add_table) (zeros for row < 0).  Replaces embed, two rms_norm and a quantize of the two
/// rows; `cat` (optional) receives the two normed rows as rms_norm writes them.
void embed_norm2_q8(const uint16_t* table, int row, const EmbedAdd& add, const uint16_t* add_table, const float* h,
                    const float* w0, const float* w1, float eps, void* xq, cudaStream_t st, float* cat = nullptr);
/// x += y (when y), then out = rms_norm(x, w) (when out) and xq = its Q8_1 blocks (when xq).  Replaces add, rms_norm
/// and a quantize.
void add_norm_q8(float* x, const float* y, const float* w, float eps, float* out, void* xq, cudaStream_t st);

} // namespace guild::core::drama3
