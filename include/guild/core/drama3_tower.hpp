// include/guild/core/drama3_tower.hpp - Fish Audio Drama 3's codebook tower on the GPU.
//
// Drama 3 (fishaudio/drama-3) is a qwen35moe backbone that emits one audio frame per step.  A frame is
// [main, c0, ..., c9]: `main` is end-of-speech or <|semantic:c0|>, and c1..c9 are the residual codebooks.  The
// backbone's head picks `main` (over eos + the semantic band only); this tower then predicts c1..c9.  It is 13
// pre-norm blocks over time (4 "shared", 9 "residual"), each with its own causal K/V over the frames, in
// fish_speech/models/text2semantic/fish_omni/tower.py's order:
//
//   h = the backbone's final-norm hidden state at position p (it predicts frame p + 1)
//   shared i:    e = semantic ? semantic_embedding_i[c0] : 0;  e += unseen when i == 0
//   residual j:  e = semantic ? codebook_embedding_j[c_j] : 0;  then head_j(h) samples c_{j+1}
//   each layer:  h = final_norm(block(eh_proj([enorm(e), hnorm(h)])))   (norms scale by 1 + w)
//   block:       x += wo(attn(x) * sigmoid(gate));  x += w2(silu(w1 x) * w3 x)
//
// `unseen` is the sum of frame p's codebook 2..9 embeddings (codebook_embeddings[k * 4096 + c_k]), zero when frame
// p is not audio.  The attention: 16 query heads, 2 K/V heads, head dim 256, q/k RMS-normed per head, pairwise
// (interleaved) RoPE on the first 64 channels, base 1e7, causal over positions 0..p.
//
// Runtime choices: the projections and heads are Q8_0 through the native MMVQ (tools/drama3_tower_rt.py writes
// them), the K/V cache is FP16, and sampling draws with the engine's own random stream, so a sampled frame equals the
// reference's only in distribution; a teacher-forced frame (`step` with `forced`) reproduces the reference's logits
// to the precision of those choices (tests/core/drama3_tower_test.cpp checks it against ref_dump vectors).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace guild::core {

struct Drama3Sampling {
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 30;
    uint64_t seed = 0;
};

struct Drama3TowerGeometry {
    int dim = 0, n_head = 0, n_kv_head = 0, head_dim = 0, ffn = 0, rope_dim = 0;
    float rope_base = 0.0f, norm_eps = 0.0f, backbone_eps = 0.0f;
    int n_shared = 0, n_codebooks = 0, codebook_size = 0;
    int32_t semantic_start = 0, semantic_end = 0, eos = 0;
    int width() const {
        return 1 + n_codebooks; // a frame row: main token, then the codes
    }
    bool is_semantic(int32_t token) const {
        return token >= semantic_start && token <= semantic_end;
    }
};

class Drama3Tower {
public:
    Drama3Tower() = default;
    ~Drama3Tower();
    Drama3Tower(const Drama3Tower&) = delete;
    Drama3Tower& operator=(const Drama3Tower&) = delete;

    /// Loads `dir` (tools/drama3_tower_rt.py) onto the current device and sizes the K/V for `max_frames` positions.
    bool load(const std::string& dir, int64_t max_frames, std::string& err);
    const Drama3TowerGeometry& geometry() const {
        return g_;
    }
    int64_t max_frames() const {
        return max_frames_;
    }
    uint64_t vram_bytes() const {
        return vram_;
    }
    /// codebook_embeddings' codebook-1 rows ([codebook_size][dim] BF16, device): the backbone's input on an audio
    /// row adds row c1 and scales by 1/sqrt(2).
    const uint16_t* code1_rows() const;

    /// One position.  `hidden` is the backbone's final-norm state at `pos` (device, dim floats); `frame` is frame
    /// `pos`'s row (host, width ints).  With `forced` (the prompt's next frame, host) the tower is teacher-forced and
    /// `out` = forced; otherwise it samples the next frame into `out` (host), drawing `main` over eos + the semantic
    /// band with the reference's repetition-aware resample against `history` (the last `n_history` main tokens).
    /// Optional `cand_logits` / `code_logits` / `layer_out` (host) receive the step's 1 + codebook_size candidate
    /// logits, the 9 heads' logits and the 13 layers' outputs, for tests.
    bool step(const float* hidden, int64_t pos, const int32_t* frame, const int32_t* forced, const int32_t* history,
              int n_history, const Drama3Sampling& sp, int32_t* out, std::string& err, float* cand_logits = nullptr,
              float* code_logits = nullptr, float* layer_out = nullptr);
    /// `step` in two parts.  step_begin runs the shared layers and the first residual one, so `out` holds main, c0
    /// and c1: all the backbone's next row reads.  step_finish runs the other residual layers into the same `out`
    /// (c2..), and may run on another thread while the backbone takes the next row; nothing else may use the tower
    /// between the two.  `frame`, `forced`, `out` and the test outputs must stay valid until step_finish returns.
    bool step_begin(const float* hidden, int64_t pos, const int32_t* frame, const int32_t* forced,
                    const int32_t* history, int n_history, const Drama3Sampling& sp, int32_t* out, std::string& err,
                    float* cand_logits = nullptr, float* code_logits = nullptr, float* layer_out = nullptr);
    bool step_finish(std::string& err);

    /// The prompt: teacher-forces positions pos0 .. pos0 + n - 1 at once.  Row i of `hidden` (device, [n][dim]) is the
    /// backbone's state at pos0 + i; `rows` (host, [n + 1][width]) are frames pos0 .. pos0 + n.  Each row gets
    /// `step`'s arithmetic with `forced` = the next row (the projections are the multi-column MMVQ, equal to a
    /// one-column call); no logits are computed.
    bool prefill(const float* hidden, int64_t pos0, int n, const int32_t* rows, std::string& err,
                 float* layer_out = nullptr);

    /// Tests: layer `l`'s K and V rows at `pos` (FP16 bits, n_kv_head * head_dim each)
    bool read_kv(int l, int64_t pos, uint16_t* k, uint16_t* v) const;

    void* stream() const {
        return stream_;
    }
    /// Restarts the sampler's draw counter (a request with a seed replays its draws)
    void reset_sampling() {
        draws_ = 0;
    }

private:
    struct Mat {
        const void* w = nullptr;
        int rows = 0, cols = 0;
    };
    struct Layer {
        const uint16_t* embedding = nullptr;
        const float *enorm = nullptr, *hnorm = nullptr, *final_norm = nullptr, *attn_norm = nullptr,
                    *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        Mat eh_proj, wqkv, wo, w1, w2, w3;
        uint16_t *k_cache = nullptr, *v_cache = nullptr; // [max_frames][n_kv_head][head_dim] FP16
    };
    bool layer_forward(Layer& L, const float* e_dev, int64_t pos, std::string& err);
    bool matvec(const Mat& m, const float* x, float* y, std::string& err, int ncols = 1);
    bool prefill_chunk(const float* hidden, int64_t pos0, int n, const int32_t* rows, std::string& err,
                       float* layer_out);
    static constexpr int kPrefillCols = 64;
    int sample(const float* logits, int n, const Drama3Sampling& sp, std::string& err);
    bool step_layers(int l0, int l1, std::string& err);     // layers [l0, l1) of the step in `cur_`
    bool read_logits(int n, float* also, std::string& err); // logits_ -> host_logits_ (and `also`), synchronized

    // the step between step_begin and step_finish
    struct Step {
        int64_t pos = 0;
        const int32_t *frame = nullptr, *forced = nullptr;
        Drama3Sampling sp;
        int32_t* out = nullptr;
        float *code_logits = nullptr, *layer_out = nullptr;
        bool semantic = false;
        int32_t codes[10] = {};
        int next_layer = 0;
    } cur_;

    Drama3TowerGeometry g_;
    int64_t max_frames_ = 0;
    uint64_t vram_ = 0;
    void* dense_ = nullptr; // device copy of dense.bin
    std::vector<Layer> layers_;
    std::vector<Mat> heads_;
    Mat cand_;
    const uint16_t* codebook_embeddings_ = nullptr;
    void* stream_ = nullptr;
    // device scratch
    float *h_ = nullptr, *e_ = nullptr, *cat_ = nullptr, *x_ = nullptr, *xn_ = nullptr, *qkv_ = nullptr, *q_ = nullptr,
          *att_ = nullptr, *proj_ = nullptr, *ff1_ = nullptr, *ff3_ = nullptr, *logits_ = nullptr, *part_ = nullptr;
    void* xq_ = nullptr;
    // the prompt's batch (kPrefillCols rows)
    float *bh_ = nullptr, *be_ = nullptr, *bcat_ = nullptr, *bx_ = nullptr, *bxn_ = nullptr, *bqkv_ = nullptr,
          *bq_ = nullptr, *batt_ = nullptr, *bproj_ = nullptr, *bff1_ = nullptr, *bff3_ = nullptr;
    void* bxq_ = nullptr;
    int32_t* bidx_ =
        nullptr; // [layers][kPrefillCols] embedding rows, then [kPrefillCols][9] adds, [kPrefillCols] counts
    std::vector<int32_t> bidx_host_;
    float* host_logits_ = nullptr; // pinned
    uint64_t draws_ = 0;
};

} // namespace guild::core
