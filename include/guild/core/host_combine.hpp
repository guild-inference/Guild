// include/guild/core/host_combine.hpp - the MoE combine's weighted sum on the CPU, bit for bit the GPU kernel's.
//
// When the CPU serves every routed expert of a verify group, the GPU used to copy the group's K expert rows per token
// back over PCIe (K * n_embd floats a token: 205 KB for two Qwen tokens, 23.6 us on a GTX 1070) and weight them
// there.  The CPU has the rows and the routing weights already, so it sums them itself and only n_embd floats a
// token cross; the GPU adds the shared expert as before.
//
// The sum must be the GPU combine's to the bit (verify == decode): native_moe_combine computes
// `s = p[0] * w[0]`, then `s = fma(p[e], w[e], s)` for e = 1..k-1 in that order, in a translation unit built with
// -ftz=true (FMUL.FTZ / FFMA.FTZ), so this one runs the same operations with the x86 FTZ and DAZ modes on.  The GPU
// then adds `shared` through the same kernel with k = 1 and a weight of 1.0 (an exact product), so the final add is
// the kernel's FADD.FTZ as before.  host_combine_parity checks the whole path against the kernel.
#pragma once

namespace guild::core {

/// True when this CPU can run host_moe_combine (AVX2 and FMA: the fused multiply-add is required, not emulated).
bool host_combine_supported();

/// out[t][c] = the GPU combine's weighted sum of parts[t][0..k-1][c] with weights[t][0..k-1], t < n_tok, c < n_embd.
/// parts is [n_tok][k][n_embd], weights [n_tok][k], out [n_tok][n_embd].  Call only when host_combine_supported().
void host_moe_combine(const float* parts, const float* weights, float* out, int n_tok, int k, int n_embd);

} // namespace guild::core
