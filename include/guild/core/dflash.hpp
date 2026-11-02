// include/guild/core/dflash.hpp - a DFlash 2 block-diffusion drafter (z-lab/dflash's DFlash2DraftModel, the
// incoai/*-DFlash2 checkpoints): one pass of a small non-causal transformer drafts a whole block of tokens after an
// anchor token, reading the target model's own layer outputs (`target_layers`, the hyper-connection streams
// averaged) of every committed position as its context, then a learned selector walks one path through each
// position's top candidates.  The drafts never change the target's output: the verifier accepts the prefix that
// equals its own greedy choice.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace guild::core {

class NativeHead;
class NativeEmbed;

/// Kernels (src/core/dflash.cu); `st` is a cudaStream_t.
namespace dflashk {
/// out[r][d] = the mean of R[r]'s `hc` streams of n values each (rows `row_stride` floats apart in R)
void stream_mean(const float* R, int hc, int n, int rows, size_t row_stride, float* out, size_t out_stride, void* st);
/// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, rows of n
void rms_rows(const float* x, const float* w, float* y, int n, int rows, float eps, void* st);
/// in place, per head of d values (rows `row_stride` floats apart): x * rsqrt(mean(x^2) + eps) * w, then NeoX-style
/// RoPE at pos[row] (theta)
void head_norm_rope(float* x, size_t row_stride, const float* w, int rows, int heads, int d, float eps,
                    const int32_t* pos, float theta, void* st);
/// pos[i] = pos0 + i, i < n
void positions(int32_t* pos, int64_t pos0, int n, void* st);
/// y[t][c] = sum over taps k of (base[k][c] + dyn[t][k][c / group]) * x[t - k][c] (x[-1] = 0), rows of n; dyn rows
/// `dyn_stride` floats apart
void block_conv(const float* x, const float* dyn, int dyn_stride, const float* base, float* y, int n, int rows,
                int taps, int group, void* st);
/// y[t] = x[t] + y[t] (rows of n), in place in y
void add_rows(const float* x, float* y, int n, int rows, void* st);
/// h[t][i] = silu(gu[t][i]) * gu[t][f + i] (gu rows of 2f), rows of f
void silu_mul(const float* gu, float* h, int f, int rows, void* st);
/// K/V of `rows` new positions (k, v: rows of kv_dim floats, `row_stride` floats apart) into the fp16 rings at slots
/// pos % ring
void ring_store(const float* k, const float* v, size_t row_stride, uint16_t* kring, uint16_t* vring, const int32_t* pos,
                int rows, int kv_dim, int ring, void* st);
/// Block attention: q (B rows of heads x d, normed and rotated, `q_stride` floats apart), the context ring
/// (positions ctx_lo .. ctx_hi - 1, slots pos % ring) and the block's own k, v (B rows of kv_heads x d, `kv_stride`
/// floats apart); every block query sees the whole block and the context within `window` positions; out B x heads x d.
/// scratch: attn_scratch_floats(...)
size_t attn_scratch_floats(int B, int kv_heads, int ring);
void block_attention(const float* q, size_t q_stride, const uint16_t* kring, const uint16_t* vring, const float* kb,
                     const float* vb, size_t kv_stride, int B, int heads, int kv_heads, int d, int ring, int64_t ctx_lo,
                     int64_t ctx_hi, int64_t start, int window, float* scratch, float* out, void* st);
/// per row of n logits: the top k (k <= 16) indices and values (best first; equal values, the lower index first) and
/// the log-sum-exp.  scratch: topk_scratch_floats(rows, n), its counts zero before the first call (topk_lse leaves
/// them zero)
size_t topk_scratch_floats(int rows, int n);
void topk_lse(const float* logits, int n, int rows, int k, int32_t* idx, float* val, float* lse, float* scratch,
              void* st);
/// the drafter's products (its "D8" weights): int8 rows of K values with an fp16 scale a 32 (`ws`, K / 32 a row) times
/// `cols` (<= 8) columns of int8 with a float scale a 32 (q8_act's), y[c][n_out]; K a multiple of 32, n_out of 4
/// when it is 4096 or more
void q8_act(const float* x, int K, int cols, int8_t* xq, float* xs, void* st, float* xsum = nullptr);
bool d8_mm(const int8_t* wq, const uint16_t* ws, const int8_t* xq, const float* xs, float* y, int K, int n_out,
           int cols, void* st);
/// ggml Q4_K rows of K (<= 2560, a multiple of 256) values times `cols` (<= 8) columns from q8_act with `xsum` (the
/// scale times the sum of a 32's int8 values), y[c][n_out]; false: a shape it does not take
bool k4_mm(const void* w, const int8_t* xq, const float* xs, const float* xsum, float* y, int K, int n_out, int cols,
           void* st);
} // namespace dflashk

class DFlashDrafter {
public:
    DFlashDrafter();
    ~DFlashDrafter();
    DFlashDrafter(const DFlashDrafter&) = delete;
    DFlashDrafter& operator=(const DFlashDrafter&) = delete;

    /// `dir`: a DFlash 2 checkpoint (config.json, model.safetensors); `head` / `embed`: the target's output head
    /// and token embedding; `vocab_path`: the draft vocabulary (ascending int32 token ids; empty: the whole head),
    /// whose rows of the head are copied as Q4_K for the drafts
    bool load(const std::string& dir, const NativeHead& head, const NativeEmbed& embed, const std::string& vocab_path,
              std::string& err);
    int block() const {
        return block_;
    }
    int hidden() const {
        return H_;
    }
    const std::vector<int>& target_layers() const {
        return target_layers_;
    }
    /// no context: a new sequence whose context starts at position `pos0` (the positions before it are never read:
    /// a prefix reused from a cache, or a prompt's start beyond the window)
    void reset(int64_t pos0 = 0);
    /// keeps the context's positions before `pos` for a sequence that continues at `pos` with the same tokens before
    /// it (a reused prefix); false, and nothing changed, when the ring holds no position before it or `pos` is past
    /// the context's end
    bool rewind(int64_t pos);
    /// the next position the context takes (add_context's pos0, draft's start)
    int64_t context_end() const {
        return ctx_hi_;
    }
    int window() const {
        return window_;
    }
    /// the committed positions pos0 .. pos0 + n - 1 (any n): their features (device, n rows of
    /// target_layers().size() x hidden floats, `feat_stride` floats apart), read in stream order (no host sync)
    bool add_context(const float* feats, size_t feat_stride, int64_t n, int64_t pos0, void* stream, std::string& err);
    /// the B - 1 tokens after `anchor` at position `start` (B <= block()), and each one's probability under the
    /// drafter's head, to the host
    bool draft(int32_t anchor, int64_t start, int B, int32_t* drafts, float* probs, void* stream, std::string& err);
    uint64_t vram_bytes() const {
        return vram_;
    }
    double ms_draft = 0, ms_context = 0;
    int64_t rounds = 0;

private:
    struct Mat {
        void* w = nullptr;      // D8: int8 rows (dflashk::d8_mm); else the GGUF type's blocks
        uint16_t* ws = nullptr; // D8: the rows' fp16 scales, one a 32 values
        int n_in = 0, n_out = 0, type = 8;
        bool d8 = false;
    };
    struct Layer {
        float *in_ln = nullptr, *post_ln = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        float *attn_base = nullptr, *mlp_base = nullptr; // [2][taps][H]
        Mat attn_kp, mlp_kp, qkv, o, gu, down;           // qkv: q rows, then k, then v
        uint16_t *kring = nullptr, *vring = nullptr;
    };
    bool upload_mat(const std::vector<float>& f, int n_in, int n_out, Mat& m, std::string& err);
    /// y = m x over `cols` columns: prep quantizes x for m's kind of product, run multiplies what prep left, so
    /// products of one input and one kind share a prep (mm: both)
    bool use_mmq(const Mat& m, int cols) const;
    void prep(const Mat& m, const float* x, int cols, void* st);
    void run(const Mat& m, float* y, int cols, void* st);
    void mm(const Mat& m, const float* x, float* y, int cols, void* st) {
        prep(m, x, cols, st);
        run(m, y, cols, st);
    }

    int H_ = 0, layers_ = 0, heads_ = 0, kv_heads_ = 0, hd_ = 0, ffn_ = 0, block_ = 0, window_ = 0, ring_ = 0;
    int taps_ = 2, group_ = 16, rank_ = 256, topk_ = 16, mask_id_ = 0, n_feat_ = 0;
    float eps_ = 1e-6f, theta_ = 10000.f;
    std::vector<int> target_layers_;
    std::vector<Layer> L_;
    Mat fc_, hp_, ctx_kv_; // ctx_kv_: every layer's k rows, then its v rows (the context's one product)
    float *hidden_norm_ = nullptr, *norm_ = nullptr;
    // the head's rows of the draft vocabulary as Q4_K, and their token ids
    void* head_ = nullptr;
    int head_type_ = 12, n_sub_ = 0;
    std::vector<int32_t> vocab_;
    // the selector's codebooks (host, bf16 bits) and the target's mask-token embedding (device)
    std::vector<uint16_t> pred_cb_, succ_cb_;
    float* mask_emb_ = nullptr;
    const NativeEmbed* embed_ = nullptr;
    // scratch
    float *x_ = nullptr, *h_ = nullptr, *a_ = nullptr, *b_ = nullptr, *c_ = nullptr, *dyn_ = nullptr, *att_ = nullptr;
    float* tscratch_ = nullptr;
    float *qkv_ = nullptr, *ckv_ = nullptr, *ctx_ = nullptr, *logits_ = nullptr, *ascratch_ = nullptr;
    void* xq_ = nullptr;
    int8_t* xq8_ = nullptr;                 // D8 products' columns (q8_act): int8, and a float scale a 32
    float *xs8_ = nullptr, *xm8_ = nullptr; // ... and the scaled sums (the Q4_K head's)
    int32_t* pos_ = nullptr;                // device positions
    int32_t* h_idx_ = nullptr;              // pinned: top-k indices, values, lse, hp
    float *h_val_ = nullptr, *h_lse_ = nullptr, *h_hp_ = nullptr;
    int32_t* d_idx_ = nullptr;
    float *d_val_ = nullptr, *d_lse_ = nullptr, *d_hp_ = nullptr;
    int64_t ctx_lo_ = 0, ctx_hi_ = 0; ///< positions [ctx_lo_, ctx_hi_) are in the rings (the last `ring_` of them)
    int R_ = 0;                       ///< scratch columns: the block, or a batch of context rows
    uint64_t vram_ = 0;
    std::vector<void*> allocs_;
    // llama.cpp's MMQ, for a head k4_mm does not take (GUILD_DFLASH_K4=0 forces it: the A/B switch)
    struct Mmq;
    std::unique_ptr<Mmq> mmq_;
    int mmq_min_ = 5; ///< columns from which a non-D8 product takes MMQ (GUILD_DFLASH_MMQ_MIN; 0: never)
};

} // namespace guild::core
