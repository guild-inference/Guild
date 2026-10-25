// include/guild/core/glm5_kernels.hpp - the GPU kernels of GLM-5.3-Flash (glm5-next) that Guild's shared kernels
// do not have: mHC hyper-connections, the KDA linear-attention scan, absorbed MLA attention and the sigmoid router.
//
// Every pointer is a device pointer, every call enqueues on `st` (a cudaStream_t).  `T` is the number of tokens
// in the window (1..kMaxT); per-token rows are contiguous: token t's row of width n starts at t * n.
#pragma once

#include <cstdint>

namespace guild::core::glm5k {

inline constexpr int kMaxT = 128; ///< tokens per window (prompt windows; decode runs 1 or 2)
inline constexpr int kKdaDim = 128;
inline constexpr int kHc = 4;

/// y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w (w may be null), rows of n
void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* st);

/// The mHC mix before a sublayer.  mixes [T][24] = hc_fn . rms_norm(flat R); from them pre/post/comb (comb by
/// Sinkhorn, `iters` iterations), xn[t] = rms_norm(sum_h pre[h] R[t][h]) * norm_w.  post [T][4], comb [T][16].
void hc_pre(const float* mixes, const float* scale, const float* base, const float* R, const float* norm_w, float* post,
            float* comb, float* xn, int N, int T, int iters, float hc_eps, float rms_eps, void* st);
/// R[t][dst] = y[t] * post[t][dst] + sum_src R[t][src] * comb[t][dst + 4 * src], in place
void hc_post(const float* y, float* R, const float* post, const float* comb, int N, int T, void* st);
/// out[t] = rms_norm(mean_h R[t][h]) * w
void hc_mean_norm(const float* R, const float* w, float* out, int N, int T, float eps, void* st);
/// R[t][h] = e[t] for every stream h
void hc_init(const float* e, float* R, int N, int T, void* st);

/// KDA short convolution over the q|k|v channels (C = 3 * D; xq, xk, xv are [T][D] each): out [T][C] =
/// silu(conv), state = the last 3 inputs per channel ([C][3], oldest first).  w: q, k, v taps, each [D][4].  Then q
/// and k are L2-normalised per head of 128.
/// `mid` (may be null): where the state after the window's first token goes (a verify window's rollback point).
void kda_conv(const float* xq, const float* xk, const float* xv, float* state, const float* wq, const float* wk,
              const float* wv, float* out, int D, int T, float* mid, void* st);
/// g = lower_bound * sigmoid(-(fpre + dt_b) * ssm_a[head]) (per channel), beta = sigmoid(bpre) (per head)
void kda_gates(const float* fpre, const float* dt_b, const float* ssm_a, float* g, const float* bpre, float* beta,
               int D, int H, int T, float lower_bound, void* st);
/// The KDA recurrence for T tokens: S [H][128 key][128 value]; qkv = the convolved, normalised [T][3D].  `mid` (may
/// be null): the state after the first token, in S's layout.
void kda_step(float* S, const float* qkv, const float* g, const float* beta, float* o, int H, int T, float* mid,
              void* st);
/// y = rms_norm(o per head) * norm_w * sigmoid(gate)
void kda_out(const float* o, const float* gate, const float* norm_w, float* y, int H, int T, float eps, void* st);

/// Per-head Q8_0 products: y[t][h][r] = W[h][r] . x[t][h], W [H][n_out][n_in] in q8_0_split's layout; n_in 64,
/// 128, 256, 512 or 1024 (q8_0_heads_supported)
void q8_0_heads(const void* W, const float* x, float* y, int H, int n_in, int n_out, int T, void* st);
bool q8_0_heads_supported(int n_in);
/// Host: `rows` GGUF Q8_0 rows of n_in values -> all the int8 values (rows x n_in, row-major), then all the FP16
/// scales (rows x n_in / 32): the same bytes, aligned for wide loads
void q8_0_split(const uint8_t* q8, uint8_t* out, int64_t rows, int n_in);

/// Q6_K rows (ggml's layout) times one or two tokens of Q8_1 activations (q6k_mmvq_supported): y[t][r], bitwise
/// guild::kernels::native_mmvq's one-token result, with fewer load instructions (Pascal is bound by them on Q6_K)
bool q6k_mmvq_supported(int n_in, int T);
void q6k_mmvq(const void* W, const void* xq8_1, float* y, int n_in, int n_out, int T, void* st);
/// The same for Q8_0 rows (bitwise native_mmvq's one-token result)
bool q8_0_mmvq_supported(int n_in, int T);
void q8_0_mmvq(const void* W, const void* xq8_1, float* y, int n_in, int n_out, int T, void* st);
/// y[t][r] = W[r] . x[t], W [n_out][n_in] F32
void f32_matvec(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* st);

/// The latent K/V rows of a DSA layer: cache [cells][L] FP16, rows pos0 .. pos0 + T - 1
void kv_store(const float* kv, void* cache, int64_t pos0, int L, int T, void* st);
/// Absorbed MLA attention (one latent head shared by all H query heads): for query t, the cells
/// [0, pos0 + t], or when sel is not null the n_sel[t] cells of sel[t] (rows of sel_stride); lat[t][h] =
/// softmax(scale q.kv) kv.  `part` is scratch of attend_scratch_floats(...) floats.
void mla_attend(const float* q, const void* cache, int64_t pos0, int H, int L, int T, float scale, const int32_t* sel,
                const int32_t* n_sel, int sel_stride, float* lat, float* part, void* st);

// ---- the DSA indexer (pools of kKpool cells; cell = position) ----
inline constexpr int kKpool = 4, kIdxDim = 128;
/// LayerNorm(kraw) * w + b and the compressor gate, stored per cell as FP16 ([cells][128] each)
void idx_store(const float* kraw, const float* ln_w, const float* ln_b, const float* gate, void* kcache, void* gcache,
               int64_t pos0, int T, float eps, void* st);
/// The pools that positions pos0 .. pos0 + T - 1 complete: pooled[p][d] = sum_s softmax_s(gate + ape)[s][d] k[s][d]
/// (ape [kKpool][128]); pooled [cells / kKpool][128] FP32
void idx_pool(const void* kcache, const void* gcache, const float* ape, float* pooled, int64_t pos0, int T, void* st);
/// scores[t][p] = sum_h w[t][h] * relu(iq[t][h] . pooled[p]) * wscale for the pools visible to query t
/// ((pos0 + t + 1) / kKpool of them); rows of `stride`
void idx_score(const float* iq, const float* w, const float* pooled, int64_t pos0, int T, int heads, float wscale,
               float* scores, int stride, void* st);
/// The cells query t attends: the members of its top `k_pools` pools (all of them when fewer are visible, ties to
/// the lower pool) and its tail (the incomplete pool's cells up to itself); sel [T][sel_stride], n_sel [T]
void idx_select(const float* scores, int stride, int64_t pos0, int T, int k_pools, int32_t* sel, int32_t* n_sel,
                int sel_stride, void* st);
int64_t attend_scratch_floats(int64_t max_cells, int H, int L, int T);

/// Sigmoid routing with a selection bias (noaux_tc): ids/w [T][k], w normalised (when norm) and scaled
void route(const float* logits, const float* bias, int E, int k, bool norm, float scale, int32_t* ids, float* w, int T,
           void* st);
/// y = silu(min(gate, lim)) * clamp(up, -lim, lim), [T][n] each
void swiglu_clamp(const float* gate, const float* up, float* y, int n, int T, float lim, void* st);
/// y += x (n floats)
void add(float* y, const float* x, int64_t n, void* st);
/// out[0] = the index of the largest of x[0 .. n) (the lowest on ties)
void argmax(const float* x, int n, int32_t* out, void* st);

} // namespace guild::core::glm5k
