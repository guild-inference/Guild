// include/guild/core/nemotron_h_kernels.hpp - the GPU kernels of the Nemotron-H engine (nemotron_h.hpp) that the
// GLM-5.3-Flash engine's (glm5_kernels.hpp: RMSNorm, router, products) do not cover: the Mamba-2 mixer's
// convolution, scan and gated norm, the ReLU^2 FFN activation and grouped-query attention without positions.
//
// Every pointer is a device pointer, every call enqueues on `st` (a cudaStream_t).  `T` is the window's tokens; token
// t's row of a [T][n] buffer starts at t * n unless a row stride is given.
#pragma once

#include <cstddef>
#include <cstdint>

namespace guild::core::nhk {

/// The causal convolution of the Mamba-2 mixer's x|B|C channels and SiLU: out[t][c] = silu(sum_k w[c][k] x[t-3+k][c]
/// + b[c]), x[t] = in[t * ld_in + c], the 3 inputs before the window from `state` ([C][3], oldest first), which ends
/// as the window's last 3.  `snaps` (may be null): the state after each of the window's first n_snaps tokens
/// ([n_snaps][C][3]), a verify window's rollback points.
void ssm_conv(const float* in, int ld_in, float* state, const float* w, const float* b, float* out, int C, int T,
              float* snaps, int n_snaps, void* st);

/// The Mamba-2 selective scan, one token after another: per head h (group h / (H / G)),
///   dt = max(softplus(dt_raw[t][h] + dt_bias[h]), dt_min), s[h][p][n] = s * exp(dt A[h]) + dt B[t][g][n] x[t][h][p],
///   y[t][h][p] = sum_n s[h][p][n] C[t][g][n] + D[h] x[t][h][p]
/// xbc [T][ld]: x (H * P) | B (G * N) | C (G * N); dt_raw rows of ld_dt; state [H][P][N] (float32, kept);
/// y [T][H * P].  P = 64 and N = 128 (ssm_scan_supported).  `snaps` (may be null): the state after each of the
/// window's first n_snaps tokens ([n_snaps][H][P][N]).
bool ssm_scan_supported(int P, int N);
void ssm_scan(const float* xbc, int ld, const float* dt_raw, int ld_dt, const float* dt_bias, const float* A,
              const float* D, float* state, float* y, int H, int P, int N, int G, int T, float dt_min, float* snaps,
              int n_snaps, void* st);

/// The mixer's output norm: v = y * silu(z), then RMSNorm over groups of `group` channels, times w: out [T][I];
/// z rows of ld_z
void gated_group_norm(const float* y, const float* z, int ld_z, const float* w, float* out, int I, int group, float eps,
                      int T, void* st);

/// y = relu(x)^2, n values
void relu2(const float* x, float* y, int64_t n, void* st);

/// LayerNorm without a bias: y[r] = (x[r] - mean) / sqrt(var + eps) * w, rows of n
void layer_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* st);

/// K/V rows into the cache: kc / vc [cells][kv_dim] FP16, rows pos0 .. pos0 + T - 1 from k / v [T][kv_dim]
void kv_store(const float* k, const float* v, void* kc, void* vc, int64_t pos0, int kv_dim, int T, void* st);

/// Causal grouped-query attention without positional encoding: query t (position pos0 + t) over the cells
/// [0, pos0 + t]; q / out [T][Hq * D], head h reads key/value head h / (Hq / Hk); scores scaled by `scale`, softmax in
/// float32.  D = 128; up to attend_max_cells() cells.
int64_t attend_max_cells();
void gqa_attend(const float* q, const void* kc, const void* vc, float* out, int Hq, int Hk, int D, int64_t pos0, int T,
                float scale, void* st);

} // namespace guild::core::nhk
