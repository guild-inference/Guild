// src/core/dflash.cpp - the DFlash 2 drafter's host side (include/guild/core/dflash.hpp).
//
// The checkpoint (z-lab/dflash's DFlash2DraftModel, bf16 safetensors) is read once: every matrix is quantized as
// Q8_0 at load (the drafts measured the same with Q8_0 weights as with bf16 ones, offline on Qwen3.8-Flash-Next's
// greedy texts) and kept as int8 rows with a scale a 32 for dflashk::d8_mm, which reads them at 200-215 GB/s at every
// column count on a GTX 1070 (MMQ: 115-153 at eight columns); the norms and the convolutions' base kernels stay f32,
// and the candidate selector's two codebooks (2 x vocab x 256 bf16) stay in host memory, where the selector runs.  The
// output head is the target's own, its rows of the draft vocabulary copied as Q4_K (Q4_0 drafted 1.747 against 1.794
// accepted drafts of 7 on a prose text; Q5_0 1.825 at 22% more bytes) for dflashk::k4_mm.
//
// The forward follows dflash/model.py: the context features of a committed position go through fc and hidden_norm
// and each layer's k_proj (k_norm, RoPE) and v_proj into that layer's ring of K/V rows (fp16, `window` slots); a
// draft embeds [anchor, mask, ..., mask] with the target's token embedding and runs every layer non-causally over
// the context within the window and the block itself, with a two-tap dynamic convolution over the block before and
// after the attention and the MLP; the final norm's rows after the anchor give the head's logits, whose top 16 a
// position the selector scores pairwise along one greedy path.
#include "guild/core/dflash.hpp"
#include "guild/core/native_head.hpp"
#include "guild/kernels/native_mmvq.hpp"
#include "guild/kernels/verify_kernels.hpp"
#include "guild/server/json.hpp"

#include "guild/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <thread>

namespace guild::core {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

float bf16f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct StTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;
};

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    out.assign((std::istreambuf_iterator<char>(f)), {});
    return true;
}

} // namespace

struct DFlashDrafter::Mmq {
    guild::prefill::mmq::Context ctx;
    void *xq8 = nullptr, *xqk = nullptr; // MMQ activations: Q8_0 / Q6_K's layout and Q4_K's (DS4)
    int32_t *bounds = nullptr, *ids = nullptr;
};

DFlashDrafter::DFlashDrafter() = default;

DFlashDrafter::~DFlashDrafter() {
    for (void* p : allocs_)
        cudaFree(p);
    for (void* p : {(void*) h_idx_, (void*) h_val_, (void*) h_lse_, (void*) h_hp_})
        if (p)
            cudaFreeHost(p);
}

bool DFlashDrafter::upload_mat(const std::vector<float>& f, int n_in, int n_out, Mat& m, std::string& err) {
    // Q8_0's rounding (ggml's), then its blocks split into the int8 rows and the scales d8_mm reads
    const size_t row = ggml_row_size(GGML_TYPE_Q8_0, n_in);
    std::vector<uint8_t> q((size_t) n_out * row);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, n_out, n_in, nullptr);
    const int nb = n_in / 32;
    std::vector<int8_t> wq((size_t) n_out * n_in);
    std::vector<uint16_t> ws((size_t) n_out * nb);
    for (int r = 0; r < n_out; ++r)
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = q.data() + (size_t) r * row + (size_t) b * 34;
            std::memcpy(&ws[(size_t) r * nb + b], blk, 2);
            std::memcpy(&wq[(size_t) r * n_in + (size_t) b * 32], blk + 2, 32);
        }
    if (cudaMalloc(&m.w, wq.size()) != cudaSuccess || cudaMalloc((void**) &m.ws, ws.size() * 2) != cudaSuccess) {
        err = "dflash: no VRAM for a matrix";
        return false;
    }
    allocs_.push_back(m.w);
    allocs_.push_back(m.ws);
    cudaMemcpy(m.w, wq.data(), wq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m.ws, ws.data(), ws.size() * 2, cudaMemcpyHostToDevice);
    m.n_in = n_in;
    m.n_out = n_out;
    m.type = (int) GGML_TYPE_Q8_0;
    m.d8 = true;
    vram_ += wq.size() + ws.size() * 2;
    return true;
}

bool DFlashDrafter::use_mmq(const Mat& m, int cols) const {
    return mmq_ && mmq_min_ > 0 && cols >= mmq_min_ && m.n_in % 256 == 0 && guild::prefill::mmq::fits(m.type, m.n_out);
}

void DFlashDrafter::prep(const Mat& m, const float* x, int cols, void* st) {
    namespace mmq = guild::prefill::mmq;
    if (m.d8) {
        dflashk::q8_act(x, m.n_in, cols, xq8_, xs8_, st);
        return;
    }
    if (use_mmq(m, cols)) {
        const bool ds4 = m.type == (int) GGML_TYPE_Q4_K || m.type == (int) GGML_TYPE_Q5_K;
        mmq::quantize(x, nullptr, ds4 ? mmq_->xqk : mmq_->xq8, m.type, m.n_in, m.n_in, cols, st);
    } else {
        guild::kernels::native_quantize_q8_1(x, xq_, m.n_in, cols, st);
    }
}

void DFlashDrafter::run(const Mat& m, float* y, int cols, void* st) {
    namespace mmq = guild::prefill::mmq;
    if (m.d8) { // eight columns at a time
        const int nb = m.n_in / 32;
        for (int c0 = 0; c0 < cols; c0 += 8)
            dflashk::d8_mm((const int8_t*) m.w, m.ws, xq8_ + (size_t) c0 * m.n_in, xs8_ + (size_t) c0 * nb,
                           y + (size_t) c0 * m.n_out, m.n_in, m.n_out, std::min(8, cols - c0), st);
        return;
    }
    if (use_mmq(m, cols)) {
        const bool ds4 = m.type == (int) GGML_TYPE_Q4_K || m.type == (int) GGML_TYPE_Q5_K;
        mmq::Product p;
        p.w = m.w;
        p.type = m.type;
        p.w_rows = m.n_out;
        p.w_cols = m.n_in;
        p.expert_bytes = mmq::matrix_bytes(m.type, m.n_out, m.n_in);
        p.n = 1;
        p.xq = ds4 ? mmq_->xqk : mmq_->xq8;
        p.bounds = mmq_->bounds + 2 * cols;
        p.ids = mmq_->ids;
        p.total_rows = cols;
        p.max_rows = cols;
        p.dst = y;
        p.ld_dst = m.n_out;
        mmq_->ctx.run(p, st);
        return;
    }
    guild::kernels::native_mmvq(m.type, m.w, xq_, y, m.n_in, m.n_out, cols, st);
}

bool DFlashDrafter::load(const std::string& dir, const NativeHead& head, const NativeEmbed& embed,
                         const std::string& vocab_path, std::string& err) {
    using guild::server::json::JsonValue;
    std::string text;
    JsonValue cfg;
    if (!read_file(dir + "/config.json", text) || !JsonValue::parse(text, cfg, err)) {
        err = "dflash: cannot read " + dir + "/config.json" + (err.empty() ? "" : ": " + err);
        return false;
    }
    const JsonValue& dc = cfg["dflash_config"];
    H_ = (int) cfg["hidden_size"].as_int();
    layers_ = (int) cfg["num_hidden_layers"].as_int();
    heads_ = (int) cfg["num_attention_heads"].as_int();
    kv_heads_ = (int) cfg["num_key_value_heads"].as_int();
    hd_ = (int) cfg["head_dim"].as_int();
    ffn_ = (int) cfg["intermediate_size"].as_int();
    window_ = (int) cfg["sliding_window"].as_int(2048);
    eps_ = (float) cfg["rms_norm_eps"].as_double(1e-6);
    theta_ = (float) cfg["rope_parameters"]["rope_theta"].as_double(10000.0);
    block_ = (int) dc["block_size"].as_int(8);
    taps_ = (int) dc["conv_kernel_size"].as_int(2);
    group_ = (int) dc["conv_group_size"].as_int(16);
    rank_ = (int) dc["selector_rank"].as_int(256);
    topk_ = (int) dc["selector_top_k"].as_int(16);
    mask_id_ = (int) dc["mask_token_id"].as_int();
    target_layers_.clear();
    for (const JsonValue& v : dc["target_layer_ids"].arr_val)
        target_layers_.push_back((int) v.as_int());
    n_feat_ = (int) target_layers_.size();
    if (H_ <= 0 || hd_ != 128 || heads_ % kv_heads_ != 0 || heads_ / kv_heads_ * block_ > 32 || block_ > 8 ||
        topk_ > 16 || H_ % (32 * group_) != 0 || n_feat_ <= 0 || taps_ != 2) {
        err = "dflash: a configuration this drafter does not take (head_dim 128, block <= 8, top-k <= 16, ...)";
        return false;
    }
    ring_ = window_;

    // the safetensors: an 8-byte header length, the JSON header, the data
    std::ifstream f(dir + "/model.safetensors", std::ios::binary);
    if (!f) {
        err = "dflash: cannot open " + dir + "/model.safetensors";
        return false;
    }
    uint64_t hlen = 0;
    f.read((char*) &hlen, 8);
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), (std::streamsize) hlen);
    JsonValue jh;
    if (!f || !JsonValue::parse(hdr, jh, err)) {
        err = "dflash: a malformed safetensors header";
        return false;
    }
    const uint64_t base = 8 + hlen;
    std::map<std::string, StTensor> tensors;
    for (const auto& [name, v] : jh.obj_val) {
        if (name == "__metadata__")
            continue;
        StTensor t;
        t.dtype = v["dtype"].str_val;
        for (const JsonValue& d : v["shape"].arr_val)
            t.shape.push_back(d.as_int());
        t.begin = (uint64_t) v["data_offsets"][0].as_int();
        t.end = (uint64_t) v["data_offsets"][1].as_int();
        tensors[name] = t;
    }
    auto get = [&](const std::string& name, std::vector<float>& out, int64_t expect) -> bool {
        auto it = tensors.find(name);
        if (it == tensors.end() || it->second.dtype != "BF16") {
            err = "dflash: missing bf16 tensor " + name;
            return false;
        }
        const StTensor& t = it->second;
        int64_t n = 1;
        for (int64_t d : t.shape)
            n *= d;
        if (n != expect || (uint64_t) n * 2 != t.end - t.begin) {
            err = "dflash: " + name + " has an unexpected shape";
            return false;
        }
        std::vector<uint16_t> raw((size_t) n);
        f.seekg((std::streamoff)(base + t.begin));
        f.read((char*) raw.data(), (std::streamsize)(n * 2));
        out.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i)
            out[(size_t) i] = bf16f(raw[(size_t) i]);
        return (bool) f;
    };
    auto dev_f32 = [&](const std::vector<float>& v, float*& p) -> bool {
        if (cudaMalloc((void**) &p, v.size() * 4) != cudaSuccess) {
            err = "dflash: no VRAM";
            return false;
        }
        allocs_.push_back(p);
        cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
        vram_ += v.size() * 4;
        return true;
    };
    std::vector<float> a, b, c, ckv;
    const int groups = H_ / group_, kp_rows = 2 * taps_ * groups, qd = heads_ * hd_, kvd = kv_heads_ * hd_;
    L_.assign((size_t) layers_, Layer{});
    ckv.reserve((size_t) layers_ * 2 * kvd * H_);
    for (int l = 0; l < layers_; ++l) {
        Layer& y = L_[(size_t) l];
        const std::string p = "layers." + std::to_string(l) + ".";
        if (!get(p + "input_layernorm.weight", a, H_) || !dev_f32(a, y.in_ln) ||
            !get(p + "post_attention_layernorm.weight", a, H_) || !dev_f32(a, y.post_ln) ||
            !get(p + "self_attn.q_norm.weight", a, hd_) || !dev_f32(a, y.q_norm) ||
            !get(p + "self_attn.k_norm.weight", a, hd_) || !dev_f32(a, y.k_norm) ||
            !get(p + "attention_conv.base_kernel", a, (int64_t) 2 * taps_ * H_) || !dev_f32(a, y.attn_base) ||
            !get(p + "mlp_conv.base_kernel", a, (int64_t) 2 * taps_ * H_) || !dev_f32(a, y.mlp_base) ||
            !get(p + "attention_conv.kernel_projection.weight", a, (int64_t) kp_rows * H_) ||
            !upload_mat(a, H_, kp_rows, y.attn_kp, err) ||
            !get(p + "mlp_conv.kernel_projection.weight", a, (int64_t) kp_rows * H_) ||
            !upload_mat(a, H_, kp_rows, y.mlp_kp, err) || !get(p + "self_attn.q_proj.weight", a, (int64_t) qd * H_) ||
            !get(p + "self_attn.k_proj.weight", b, (int64_t) kvd * H_) ||
            !get(p + "self_attn.v_proj.weight", c, (int64_t) kvd * H_))
            return false;
        ckv.insert(ckv.end(), b.begin(), b.end()); // the context's product: this layer's k rows, then its v rows
        ckv.insert(ckv.end(), c.begin(), c.end());
        a.insert(a.end(), b.begin(), b.end()); // the block's: q rows, k rows, v rows
        a.insert(a.end(), c.begin(), c.end());
        if (!upload_mat(a, H_, qd + 2 * kvd, y.qkv, err) || !get(p + "self_attn.o_proj.weight", a, (int64_t) H_ * qd) ||
            !upload_mat(a, qd, H_, y.o, err) || !get(p + "mlp.gate_proj.weight", a, (int64_t) ffn_ * H_) ||
            !get(p + "mlp.up_proj.weight", b, (int64_t) ffn_ * H_))
            return false;
        a.insert(a.end(), b.begin(), b.end()); // gate rows, then up rows: one product
        if (!upload_mat(a, H_, 2 * ffn_, y.gu, err) || !get(p + "mlp.down_proj.weight", a, (int64_t) H_ * ffn_) ||
            !upload_mat(a, ffn_, H_, y.down, err))
            return false;
        for (uint16_t** r : {&y.kring, &y.vring}) {
            if (cudaMalloc((void**) r, (size_t) ring_ * kvd * 2) != cudaSuccess) {
                err = "dflash: no VRAM for the K/V rings";
                return false;
            }
            allocs_.push_back(*r);
            vram_ += (uint64_t) ring_ * kvd * 2;
        }
    }
    if (!upload_mat(ckv, H_, layers_ * 2 * kvd, ctx_kv_, err))
        return false;
    ckv = {};
    if (!get("fc.weight", a, (int64_t) H_ * n_feat_ * H_) || !upload_mat(a, n_feat_ * H_, H_, fc_, err) ||
        !get("hidden_norm.weight", a, H_) || !dev_f32(a, hidden_norm_) || !get("norm.weight", a, H_) ||
        !dev_f32(a, norm_) || !get("candidate_selector.hidden_projection.weight", a, (int64_t) rank_ * H_) ||
        !upload_mat(a, H_, rank_, hp_, err))
        return false;
    // the codebooks, host bf16 bits
    const int64_t V = tensors["candidate_selector.predecessor_codebook"].shape.empty()
                          ? 0
                          : tensors["candidate_selector.predecessor_codebook"].shape[0];
    for (auto [name, cb] :
         {std::pair<const char*, std::vector<uint16_t>*>{"candidate_selector.predecessor_codebook", &pred_cb_},
          {"candidate_selector.successor_codebook", &succ_cb_}}) {
        const StTensor& t = tensors[name];
        if (t.dtype != "BF16" || t.shape.size() != 2 || t.shape[0] != V || t.shape[1] != rank_) {
            err = std::string("dflash: ") + name + " is not vocab x rank bf16";
            return false;
        }
        cb->resize((size_t) V * rank_);
        f.seekg((std::streamoff)(base + t.begin));
        f.read((char*) cb->data(), (std::streamsize)(cb->size() * 2));
        if (!f) {
            err = std::string("dflash: cannot read ") + name;
            return false;
        }
    }

    // the draft head: the target head's rows of the draft vocabulary (or all), as Q4_K
    vocab_.clear();
    if (!vocab_path.empty()) {
        std::string raw;
        if (!read_file(vocab_path, raw) || raw.size() < 4 || raw.size() % 4 != 0) {
            err = "dflash: cannot read the draft vocabulary " + vocab_path;
            return false;
        }
        vocab_.resize(raw.size() / 4);
        std::memcpy(vocab_.data(), raw.data(), raw.size());
    } else {
        vocab_.resize((size_t) V);
        for (int64_t i = 0; i < V; ++i)
            vocab_[(size_t) i] = (int32_t) i;
    }
    n_sub_ = (int) vocab_.size();
    {
        const int64_t row_bytes = (int64_t) head.row_bytes();
        int32_t* d_ids = nullptr;
        uint8_t* rows = nullptr;
        if (cudaMalloc((void**) &d_ids, vocab_.size() * 4) != cudaSuccess ||
            cudaMalloc((void**) &rows, (size_t) n_sub_ * row_bytes) != cudaSuccess) {
            err = "dflash: no VRAM to gather the head's rows";
            return false;
        }
        cudaMemcpy(d_ids, vocab_.data(), vocab_.size() * 4, cudaMemcpyHostToDevice);
        guild::kernels::gather_rows((const uint8_t*) head.weights(), row_bytes, d_ids, n_sub_, rows, nullptr);
        std::vector<uint8_t> src((size_t) n_sub_ * row_bytes);
        cudaMemcpy(src.data(), rows, src.size(), cudaMemcpyDeviceToHost);
        cudaFree(rows);
        cudaFree(d_ids);
        const auto* tr = ggml_get_type_traits((ggml_type) head.type());
        const ggml_type ht = (ggml_type) head_type_;
        const size_t out_row = ggml_row_size(ht, H_);
        std::vector<uint8_t> out((size_t) n_sub_ * out_row);
        std::atomic<int64_t> next{0};
        auto work = [&] {
            std::vector<float> fl((size_t) 64 * H_);
            for (int64_t r0; (r0 = next.fetch_add(64)) < n_sub_;) {
                const int64_t nr = std::min<int64_t>(64, n_sub_ - r0);
                for (int64_t r = 0; r < nr; ++r)
                    tr->to_float(src.data() + (size_t) (r0 + r) * row_bytes, fl.data() + (size_t) r * H_, H_);
                ggml_quantize_chunk(ht, fl.data(), out.data() + (size_t) r0 * out_row, 0, nr, H_, nullptr);
            }
        };
        std::vector<std::thread> th;
        for (unsigned i = 1; i < std::max(1u, std::min(32u, std::thread::hardware_concurrency())); ++i)
            th.emplace_back(work);
        work();
        for (auto& t : th)
            t.join();
        if (cudaMalloc(&head_, out.size()) != cudaSuccess) {
            err = "dflash: no VRAM for the draft head";
            return false;
        }
        allocs_.push_back(head_);
        cudaMemcpy(head_, out.data(), out.size(), cudaMemcpyHostToDevice);
        vram_ += out.size();
    }

    // scratch: the block's rows, or a batch of up to 64 context rows
    R_ = std::max(block_, 64);
    const int R = R_;
    auto take = [&](float*& p, size_t floats) -> bool {
        if (cudaMalloc((void**) &p, floats * 4) != cudaSuccess) {
            err = "dflash: no VRAM for scratch";
            return false;
        }
        allocs_.push_back(p);
        vram_ += floats * 4;
        return true;
    };
    const size_t widest = (size_t) std::max({2 * ffn_, qd + 2 * kvd, n_feat_ * H_, layers_ * 2 * kvd, H_});
    if (!take(x_, (size_t) R * H_) || !take(h_, (size_t) block_ * H_) || !take(a_, (size_t) R * widest) ||
        !take(b_, (size_t) block_ * widest) || !take(c_, (size_t) block_ * widest) ||
        !take(dyn_, (size_t) block_ * kp_rows) || !take(att_, (size_t) block_ * qd) ||
        !take(qkv_, (size_t) block_ * (qd + 2 * kvd)) || !take(ckv_, (size_t) R * layers_ * 2 * kvd) ||
        !take(ctx_, (size_t) R * H_) || !take(logits_, (size_t) block_ * n_sub_) ||
        !take(ascratch_, dflashk::attn_scratch_floats(block_, kv_heads_, ring_)) ||
        !take(tscratch_, dflashk::topk_scratch_floats(block_, n_sub_)) || !take(mask_emb_, (size_t) H_) ||
        !take(d_val_, (size_t) block_ * 16) || !take(d_lse_, (size_t) block_) || !take(d_hp_, (size_t) block_ * rank_))
        return false;
    cudaMemset(tscratch_, 0, dflashk::topk_scratch_floats(block_, n_sub_) * 4); // topk_lse's counts start at zero
    const size_t xq_bytes = (size_t) R * (size_t) (widest / 32 + 1) * 36 + 64;
    if (cudaMalloc(&xq_, xq_bytes) != cudaSuccess || cudaMalloc((void**) &pos_, (size_t) R * 4) != cudaSuccess ||
        cudaMalloc((void**) &d_idx_, (size_t) block_ * 16 * 4) != cudaSuccess) {
        err = "dflash: no VRAM for scratch";
        return false;
    }
    allocs_.push_back(xq_);
    allocs_.push_back(pos_);
    allocs_.push_back(d_idx_);
    if (cudaMalloc((void**) &xq8_, (size_t) R * widest) != cudaSuccess ||
        cudaMalloc((void**) &xs8_, (size_t) R * (widest / 32) * 4) != cudaSuccess ||
        cudaMalloc((void**) &xm8_, (size_t) R * (widest / 32) * 4) != cudaSuccess) {
        err = "dflash: no VRAM for scratch";
        return false;
    }
    allocs_.push_back(xq8_);
    allocs_.push_back(xs8_);
    allocs_.push_back(xm8_);
    if (cudaMallocHost((void**) &h_idx_, (size_t) block_ * 16 * 4) != cudaSuccess ||
        cudaMallocHost((void**) &h_val_, (size_t) block_ * 16 * 4) != cudaSuccess ||
        cudaMallocHost((void**) &h_lse_, (size_t) block_ * 4) != cudaSuccess ||
        cudaMallocHost((void**) &h_hp_, (size_t) block_ * rank_ * 4) != cudaSuccess) {
        err = "dflash: no pinned memory";
        return false;
    }
    if (const char* e = std::getenv("GUILD_DFLASH_MMQ_MIN"))
        mmq_min_ = std::atoi(e);
    if (guild::prefill::mmq::built() && mmq_min_ > 0) {
        namespace mmq = guild::prefill::mmq;
        mmq_ = std::make_unique<Mmq>();
        const size_t mb = mmq::q8_bytes(R, (int64_t) widest);
        std::vector<int32_t> bounds((size_t) 2 * (R + 1));
        for (int t = 0; t <= R; ++t) {
            bounds[(size_t) 2 * t] = 0;
            bounds[(size_t) 2 * t + 1] = t;
        }
        if (cudaMalloc(&mmq_->xq8, mb) != cudaSuccess || cudaMalloc(&mmq_->xqk, mb) != cudaSuccess ||
            cudaMalloc((void**) &mmq_->bounds, bounds.size() * 4) != cudaSuccess ||
            cudaMalloc((void**) &mmq_->ids, (size_t) R * 4) != cudaSuccess) {
            err = "dflash: no VRAM for MMQ";
            return false;
        }
        allocs_.insert(allocs_.end(), {mmq_->xq8, mmq_->xqk, (void*) mmq_->bounds, (void*) mmq_->ids});
        cudaMemcpy(mmq_->bounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice);
        mmq::iota(mmq_->ids, R, nullptr);
    }
    embed_ = &embed;
    embed.gather_one(mask_id_, mask_emb_, nullptr);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        err = "dflash: a CUDA error while loading";
        return false;
    }
    std::fprintf(stderr,
                 "guild dflash: %d layers, block %d, window %d, features of target layers %d..%d; draft head over %d "
                 "tokens as Q4_K; %.0f MiB of VRAM\n",
                 layers_, block_, window_, target_layers_.front(), target_layers_.back(), n_sub_,
                 (double) vram_ / 1048576.0);
    reset();
    return true;
}

void DFlashDrafter::reset(int64_t pos0) {
    ctx_lo_ = ctx_hi_ = std::max<int64_t>(0, pos0);
}

bool DFlashDrafter::rewind(int64_t pos) {
    const int64_t lo = std::max<int64_t>(ctx_lo_, ctx_hi_ - ring_); // the positions the ring still holds
    if (pos < lo || pos > ctx_hi_ || (pos == lo && pos > ctx_lo_))
        return false;
    ctx_lo_ = lo;
    ctx_hi_ = pos;
    return true;
}

bool DFlashDrafter::add_context(const float* feats, size_t feat_stride, int64_t n, int64_t pos0, void* st,
                                std::string& err) {
    if (pos0 != ctx_hi_ || n < 0) {
        err = "dflash: the context's positions must follow each other";
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    cudaStream_t cs = (cudaStream_t) st;
    const int kvd = kv_heads_ * hd_;
    const size_t ckv_stride = (size_t) layers_ * 2 * kvd;
    for (int64_t c0 = 0; c0 < n; c0 += R_) {
        const int m = (int) std::min<int64_t>(R_, n - c0);
        const float* fx = feats + (size_t) c0 * feat_stride;
        const float* in = fx;
        if (feat_stride != (size_t) n_feat_ * H_) { // rows to contiguous
            cudaMemcpy2DAsync(a_, (size_t) n_feat_ * H_ * 4, fx, feat_stride * 4, (size_t) n_feat_ * H_ * 4, m,
                              cudaMemcpyDeviceToDevice, cs);
            in = a_;
        }
        dflashk::positions(pos_, pos0 + c0, m, cs);
        mm(fc_, in, ctx_, m, cs);
        dflashk::rms_rows(ctx_, hidden_norm_, x_, H_, m, eps_, cs);
        mm(ctx_kv_, x_, ckv_, m, cs); // every layer's k and v at once
        for (int l = 0; l < layers_; ++l) {
            Layer& y = L_[(size_t) l];
            float* k = ckv_ + (size_t) l * 2 * kvd;
            dflashk::head_norm_rope(k, ckv_stride, y.k_norm, m, kv_heads_, hd_, eps_, pos_, theta_, cs);
            dflashk::ring_store(k, k + kvd, ckv_stride, y.kring, y.vring, pos_, m, kvd, ring_, cs);
        }
    }
    ctx_hi_ = pos0 + n;
    ms_context += ms_since(t0);
    return cudaGetLastError() == cudaSuccess || (err = "dflash: a CUDA error in add_context", false);
}

bool DFlashDrafter::draft(int32_t anchor, int64_t start, int B, int32_t* drafts, float* probs, void* st,
                          std::string& err) {
    if (B < 2 || B > block_ || start != ctx_hi_) {
        err = "dflash: draft arguments out of range";
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    cudaStream_t cs = (cudaStream_t) st;
    const int groups = H_ / group_, kp_rows = 2 * taps_ * groups, qd = heads_ * hd_, kvd = kv_heads_ * hd_;
    const int half = taps_ * groups; // a dynamic row's prepare part, then its finish part
    const size_t qkvd = (size_t) qd + 2 * kvd;
    dflashk::positions(pos_, start, B, cs);
    embed_->gather_one(anchor, h_, cs);
    for (int i = 1; i < B; ++i)
        cudaMemcpyAsync(h_ + (size_t) i * H_, mask_emb_, (size_t) H_ * 4, cudaMemcpyDeviceToDevice, cs);
    const int64_t ctx_lo = std::max<int64_t>(ctx_lo_, ctx_hi_ - ring_);
    for (Layer& y : L_) {
        // attention
        dflashk::rms_rows(h_, y.in_ln, x_, H_, B, eps_, cs);
        mm(y.attn_kp, x_, dyn_, B, cs);
        dflashk::block_conv(x_, dyn_, kp_rows, y.attn_base, a_, H_, B, taps_, group_, cs);
        mm(y.qkv, a_, qkv_, B, cs);
        float *q = qkv_, *kb = qkv_ + qd, *vb = qkv_ + qd + kvd;
        dflashk::head_norm_rope(q, qkvd, y.q_norm, B, heads_, hd_, eps_, pos_, theta_, cs);
        dflashk::head_norm_rope(kb, qkvd, y.k_norm, B, kv_heads_, hd_, eps_, pos_, theta_, cs);
        dflashk::block_attention(q, qkvd, y.kring, y.vring, kb, vb, qkvd, B, heads_, kv_heads_, hd_, ring_, ctx_lo,
                                 ctx_hi_, start, window_, ascratch_, att_, cs);
        mm(y.o, att_, b_, B, cs);
        dflashk::block_conv(b_, dyn_ + half, kp_rows, y.attn_base + (size_t) taps_ * H_, c_, H_, B, taps_, group_, cs);
        dflashk::add_rows(c_, h_, H_, B, cs);
        // MLP
        dflashk::rms_rows(h_, y.post_ln, x_, H_, B, eps_, cs);
        mm(y.mlp_kp, x_, dyn_, B, cs);
        dflashk::block_conv(x_, dyn_, kp_rows, y.mlp_base, a_, H_, B, taps_, group_, cs);
        mm(y.gu, a_, b_, B, cs);
        dflashk::silu_mul(b_, c_, ffn_, B, cs);
        mm(y.down, c_, a_, B, cs);
        dflashk::block_conv(a_, dyn_ + half, kp_rows, y.mlp_base + (size_t) taps_ * H_, c_, H_, B, taps_, group_, cs);
        dflashk::add_rows(c_, h_, H_, B, cs);
    }
    const int D = B - 1;
    dflashk::rms_rows(h_ + H_, norm_, x_, H_, D, eps_, cs);
    mm(hp_, x_, d_hp_, D, cs);
    Mat hm;
    hm.w = head_;
    hm.n_in = H_;
    hm.n_out = n_sub_;
    hm.type = head_type_;
    static const bool k4_off = std::getenv("GUILD_DFLASH_K4") && std::atoi(std::getenv("GUILD_DFLASH_K4")) == 0;
    bool head_done = false;
    if (head_type_ == (int) GGML_TYPE_Q4_K && !k4_off) { // the Q4_K head kernel, else MMQ / MMVQ
        dflashk::q8_act(x_, H_, D, xq8_, xs8_, cs, xm8_);
        head_done = dflashk::k4_mm(head_, xq8_, xs8_, xm8_, logits_, H_, n_sub_, D, cs);
    }
    if (!head_done)
        mm(hm, x_, logits_, D, cs);
    dflashk::topk_lse(logits_, n_sub_, D, topk_, d_idx_, d_val_, d_lse_, tscratch_, cs);
    cudaMemcpyAsync(h_idx_, d_idx_, (size_t) D * topk_ * 4, cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(h_val_, d_val_, (size_t) D * topk_ * 4, cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(h_lse_, d_lse_, (size_t) D * 4, cudaMemcpyDeviceToHost, cs);
    cudaMemcpyAsync(h_hp_, d_hp_, (size_t) D * rank_ * 4, cudaMemcpyDeviceToHost, cs);
    if (cudaStreamSynchronize(cs) != cudaSuccess) {
        err = std::string("dflash draft: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // the selector: from the anchor, each position's candidate with the best unary + pairwise score
    int32_t pred = anchor;
    for (int i = 0; i < D; ++i) {
        const float* hp = h_hp_ + (size_t) i * rank_;
        const uint16_t* pc = pred_cb_.data() + (size_t) pred * rank_;
        float pw[256];
        for (int r = 0; r < rank_; ++r)
            pw[r] = bf16f(pc[r]) * hp[r];
        int best = 0;
        float best_s = -INFINITY;
        for (int j = 0; j < topk_; ++j) {
            const int32_t tok = vocab_[(size_t) h_idx_[i * topk_ + j]];
            const uint16_t* sc = succ_cb_.data() + (size_t) tok * rank_;
            float s = 0.f;
            for (int r = 0; r < rank_; ++r)
                s += pw[r] * bf16f(sc[r]);
            s += h_val_[i * topk_ + j];
            if (s > best_s) {
                best_s = s;
                best = j;
            }
        }
        pred = vocab_[(size_t) h_idx_[i * topk_ + best]];
        drafts[i] = pred;
        if (probs)
            probs[i] = std::exp(h_val_[i * topk_ + best] - h_lse_[i]);
    }
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

} // namespace guild::core
