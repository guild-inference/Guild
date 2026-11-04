// include/guild/core/nemotron_h.hpp - Nemotron-H MoE language models (GGUF architecture `nemotron_h_moe`, e.g. the
// language model of Nemotron 3.5 Transcribe): pre-norm blocks of three kinds,
//   Mamba-2 (M):    in_proj -> z | xBC | dt; causal conv (k 4) + SiLU on xBC; the selective scan over 64 heads of 64
//                   with a 128-wide state per (head, channel) and 8 groups of B / C; + D x; y * silu(z); RMSNorm over
//                   groups of 512; out_proj
//   MoE (E):        sigmoid router with a selection bias, top 6 of 128 experts, weights normalized and scaled 2.5;
//                   experts and a shared expert down(relu(up x)^2)
//   attention (*):  grouped-query (32 query heads, 2 key/value heads of 128), causal, no positional encoding
// and an MTP block (attention + MoE after [enorm(embedding) | hnorm(hidden)] -> eh_proj) for speculative decoding.
//
// Placement as the GLM-5.3-Flash engine's (glm5.hpp): the dense weights, the recurrent states and the K/V cache on
// the GPU in their GGUF formats; the routed experts on the CPU (cpu_moe.hpp), split across NUMA nodes.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace guild::core {

struct NemotronHGeometry {
    int64_t n_embd = 2688, n_layer = 52, n_vocab = 131072;
    int64_t n_head = 32, n_head_kv = 2, head_dim = 128;
    int64_t ssm_inner = 4096, ssm_state = 128, ssm_heads = 64, ssm_groups = 8, ssm_conv = 4;
    int64_t n_expert = 128, n_used = 6, n_ff_exp = 1856, n_ff_shexp = 3712;
    float rms_eps = 1e-5f, ln_eps = 1e-5f, expert_scale = 2.5f;
    bool expert_norm = true;
    std::string kinds; ///< per layer: 'M' Mamba-2, 'E' MoE, '*' attention
};

struct NemotronHOptions {
    int64_t max_context = 4096;
    int threads = 0;                      ///< CPU expert threads (0: the physical cores)
    uint64_t vram_reserve = 384ull << 20; ///< left free on the GPU for the CUDA context and other users
    bool numa_split = true; ///< on two NUMA nodes, copy half of every expert's rows into each node's memory
    bool mtp = true;        ///< load the MTP block for speculative decoding (draft / verify / accept)
    /// the MTP block's last norm: LayerNorm (llama.cpp's nemotron-h-moe graph) or RMSNorm (vLLM's NemotronHMTP); only
    /// the drafts depend on it
    bool mtp_layer_norm = true;
};

class NemotronHEngine {
public:
    NemotronHEngine();
    ~NemotronHEngine();
    NemotronHEngine(const NemotronHEngine&) = delete;
    NemotronHEngine& operator=(const NemotronHEngine&) = delete;

    bool load(const std::string& gguf, const NemotronHOptions& opt, std::string& err);
    const NemotronHGeometry& geometry() const;
    int64_t max_context() const;
    uint64_t vram_bytes() const;
    /// one line per placement decision, for the startup log
    const std::string& placement() const;

    /// Forgets the context (K/V and recurrent states, the MTP block's too).
    void reset();
    /// Runs rows [0, n) at positions pos0 .. pos0 + n - 1: row i is token tokens[i]'s embedding, or, where tokens[i] is
    /// negative, `embeds` row i (n_embd floats, host: an audio embedding).  A prompt: windows of up to the prompt
    /// window (llama.cpp's MMQ for the wide products); the Mamba-2 scan clamps the time step at 0.001 there, as the
    /// reference's chunked scan does.  `logits` (n_vocab floats, host): the last row's when not null.
    bool forward(const int32_t* tokens, const float* embeds, int n, int64_t pos0, float* logits, std::string& err);
    /// One generated token at `pos`, as the reference's one-token decode step computes it (no time-step clamp);
    /// `logits` (n_vocab floats, host) when not null.
    bool step(int32_t token, int64_t pos, float* logits, std::string& err);

    /// Speculative decoding with the MTP block (one layer used repeatedly).  draft: up to `k` guesses for the tokens
    /// after `last` (which sits at `pos`, not yet fed), each the argmax of the block's head.  verify: runs tokens[0 ..
    /// n) (the last token and the drafts, n <= 1 + max drafts) at pos0 and gives every row's logits ([n][n_vocab],
    /// host).  accept: keeps the first n_keep tokens of that window (the recurrent states roll back to after them).
    /// A verify window's rows are computed as one-token steps would be, so greedy output does not depend on the drafts.
    bool has_mtp() const;
    int max_drafts() const;
    bool draft(int32_t last, int64_t pos, int k, int32_t* drafts, std::string& err);
    bool verify(const int32_t* tokens, int n, int64_t pos0, float* logits_rows, std::string& err);
    bool accept(int n_keep, std::string& err);

    /// Milliseconds of the last forward / verify spent in the CPU experts
    double last_cpu_ms() const;
    /// GUILD_MOE_TIMING=1: the CPU experts' phases so far, to stderr
    void print_timing(const char* what) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace guild::core
