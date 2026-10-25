// include/guild/core/glm5.hpp - GLM-5.3-Flash (GGUF architecture `glm5-next`): 45 layers of mHC hyper-connected
// blocks, 34 of them KDA linear attention and 11 DSA (absorbed MLA) attention, 3 dense FFN layers then 42 MoE
// layers (288 experts, 8 used, plus a shared expert).
//
// Placement: the attention, the mixers, the router and the output head on the GPU in their GGUF formats; the
// routed experts on the CPU (ggml-cpu's dot products, as llama.cpp's CPU backend computes them); the shared
// experts and the dense FFN layers on the GPU while VRAM lasts, otherwise on the CPU with the routed ones.
//
// Not yet: the vision tower.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace guild::core {

struct Glm5Geometry {
    int64_t n_embd = 4096, n_layer = 45, n_vocab = 154880;
    int64_t n_head = 64, kda_dim = 128;
    int64_t q_lora = 1536, kv_lora = 512, qk_dim = 256, v_dim = 256;
    int64_t idx_heads = 32, idx_dim = 128, idx_topk = 2048, idx_kpool = 4;
    int64_t hc = 4, sinkhorn_iters = 20;
    int64_t n_expert = 288, n_used = 8, n_ff = 12288, n_ff_exp = 2048, n_ff_shexp = 2048, n_dense_lead = 3;
    float rms_eps = 1e-5f, ln_eps = 1e-6f, hc_eps = 1e-6f, kda_lower_bound = -5.f, expert_scale = 2.5f;
    bool expert_norm = true;
    std::vector<bool> is_kda;                  ///< per layer: KDA (true) or DSA attention
    std::vector<float> clamp_exp, clamp_shexp; ///< per layer SwiGLU limits (0: none)
    std::vector<int32_t> eos_ids;
};

struct Glm5Options {
    int64_t max_context = 32768;
    int threads = 0;                      ///< CPU expert threads (0: the physical cores)
    uint64_t vram_reserve = 384ull << 20; ///< left free on the GPU for the CUDA context and other users
    bool prefault = true;                 ///< read the routed experts into memory at load
    bool numa_split = true; ///< on two NUMA nodes, copy half of every expert's rows into each node's memory
    bool mtp = true;        ///< load the NextN block for speculative decoding (draft / verify / accept)
    bool nl4 = true;        ///< IQ3_S expert gate/up rows repacked for the CPU's faster decode (AVX2)
    bool dense_q6k = false; ///< re-quantize the attention's largest Q8_0 matrices to Q6_K (less VRAM and GPU time)
};

class Glm5Engine {
public:
    Glm5Engine();
    ~Glm5Engine();
    Glm5Engine(const Glm5Engine&) = delete;
    Glm5Engine& operator=(const Glm5Engine&) = delete;

    /// `gguf` is any shard of the model
    bool load(const std::string& gguf, const Glm5Options& opt, std::string& err);
    const Glm5Geometry& geometry() const;
    int64_t max_context() const;
    uint64_t vram_bytes() const;
    /// one line per placement decision, for the startup log
    const std::string& placement() const;

    /// Forgets the context (K/V, KDA and convolution states).
    void reset();
    /// Saves / restores the recurrent (KDA and convolution) states.  The K/V and indexer rows stay where they are:
    /// a restored context of n tokens reads cells 0 .. n - 1 and overwrites the rest as it goes.
    bool snapshot(std::string& err);
    bool restore(std::string& err);
    /// Runs tokens[0 .. n) at positions pos0 .. pos0 + n - 1; `logits` (n_vocab floats, host) gets the last token's
    /// logits when it is not null.
    bool forward(const int32_t* tokens, int n, int64_t pos0, float* logits, std::string& err);

    /// Speculative decoding with the NextN block.  draft: the block's guess for the token after `last` (which sits at
    /// `pos`, not yet fed).  verify: runs tokens[0 .. n) (n <= 2: the last token and the draft) at pos0 and gives
    /// every row's logits ([n][n_vocab], host).  accept: keeps the first n_keep tokens of that window (the state
    /// rolls back to after the first when the draft is refused).  forward() needs no accept.
    bool has_mtp() const;
    bool draft(int32_t last, int64_t pos, int32_t& draft, std::string& err);
    bool verify(const int32_t* tokens, int n, int64_t pos0, float* logits_rows, std::string& err);
    bool accept(int n_keep, std::string& err);

    /// Time spent in the last forward: GPU+orchestration and CPU experts, milliseconds
    double last_cpu_ms() const;
    /// GUILD_GLM_TIMING=1: the CPU experts' phases so far (activations, gate/up, hidden, down), to stderr
    void print_timing(const char* what) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace guild::core
