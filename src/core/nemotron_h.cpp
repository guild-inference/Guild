// src/core/nemotron_h.cpp - Nemotron-H MoE on one GPU and the CPU (see include/guild/core/nemotron_h.hpp).
//
// One forward pass over a window of T tokens, layer by layer, in the order of transformers' NemotronHBlock (and
// llama.cpp's nemotron-h graph): x += mixer(rms_norm(x)).  A MoE layer hands the normed activations and the router's
// choice to the CPU, which computes the routed experts while the GPU computes the shared one.  Prompt windows of up to
// kPromptT tokens take llama.cpp's MMQ for the products (rows padded so that MMQ's 256-value steps read zeros);
// decode and verify windows take MMVQ, whose columns are bitwise one-column products.
#include "guild/core/nemotron_h.hpp"

#include "guild/artifact/gguf_reader.hpp"
#include "guild/core/cpu_moe.hpp"
#include "guild/core/dense_mm.hpp"
#include "guild/core/glm5_kernels.hpp"
#include "guild/core/nemotron_h_kernels.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace guild::core {

namespace {

using cpu_moe::CpuFfn;
constexpr int kPromptT = 256; ///< the tokens of a prompt window
constexpr int kMaxVerify = 4; ///< a verify window: the last token and up to 3 drafts
constexpr int kF32 = 0, kF16 = 1, kBF16 = 30;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Layer {
    char kind = 'M';
    float* norm = nullptr; ///< the block's pre-norm (MTP: attn_norm)
    // Mamba-2
    GgufMat in_proj, out_proj;
    float *conv_w = nullptr, *conv_b = nullptr, *dt_b = nullptr, *A = nullptr, *D = nullptr, *ssm_norm = nullptr;
    float *ssm_state = nullptr, *conv_state = nullptr;
    float *ssm_snap = nullptr, *conv_snap = nullptr; ///< states after each of a verify window's first tokens
    // attention
    GgufMat wq, wk, wv, wo;
    void *kc = nullptr, *vc = nullptr;
    // MoE
    float *router = nullptr, *bias = nullptr;
    GgufMat up_sh, down_sh;
    std::vector<CpuFfn> experts;
    // the MTP block: [enorm(embedding) | hnorm(hidden)] -> eh_proj, the MoE sublayer's norm, the head norm
    GgufMat eh_proj;
    float *enorm = nullptr, *hnorm = nullptr, *post_norm = nullptr, *head_norm = nullptr;
};

} // namespace

struct NemotronHEngine::Impl {
    std::unique_ptr<GgufModel> gguf;
    NemotronHGeometry g;
    NemotronHOptions opt;
    std::vector<Layer> L;
    Layer mtp;
    bool has_mtp = false;
    GgufMat head;
    float* out_norm = nullptr;
    // the token embedding, read on the host: a row's float32 values (BF16 and F16 exactly)
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr;
    uint64_t vram = 0;
    std::string placement;
    cpu_moe::Workers wk;
    cpu_moe::Experts xp;
    uint8_t* arena[2] = {nullptr, nullptr};
    size_t arena_bytes[2] = {0, 0};
    DenseProducts dp;
    std::vector<void*> allocs;
    double cpu_ms = 0.0;
    double gpu_wait_ms = 0.0; ///< the CPU waiting for a layer's rows before its experts (GUILD_MOE_TIMING)

    // device scratch, for kPromptT rows
    float *x = nullptr, *xn = nullptr, *a = nullptr, *cv = nullptr, *ys = nullptr, *yn = nullptr, *q = nullptr;
    float *k = nullptr, *v = nullptr, *att = nullptr, *sa = nullptr, *sb = nullptr, *y = nullptr, *ycpu = nullptr;
    float *rlog = nullptr, *rw = nullptr, *logits = nullptr, *hn = nullptr, *cat = nullptr;
    int32_t *rids = nullptr, *d_arg = nullptr;
    // MTP: the hidden row waiting for its next token, and the block's own rows
    float *pending = nullptr, *mh = nullptr;
    bool has_pending = false;
    // pinned host
    float *h_x = nullptr, *h_y = nullptr, *h_w = nullptr, *h_emb = nullptr;
    int32_t *h_ids = nullptr, *h_arg = nullptr;
    // A window's own buffers (the rest of the scratch is used up within a layer): x, y, ycpu, hn, h_x, h_y, h_w,
    // h_emb, h_ids and ev above are the bound window's (bind), window 0's but inside a window pair.  A verify window
    // split in two: the second part's snapshots start at slot snap0, and the first part keeps its last token's too.
    struct Win {
        float *x = nullptr, *y = nullptr, *ycpu = nullptr, *hn = nullptr;
        float *h_x = nullptr, *h_y = nullptr, *h_w = nullptr, *h_emb = nullptr;
        int32_t* h_ids = nullptr;
        cudaEvent_t ev = nullptr;
        int snap0 = 0;
        bool snap_last = false;
    };
    Win win[2];
    int snap0 = 0;
    bool snap_last = false;
    void bind(const Win& w) {
        x = w.x, y = w.y, ycpu = w.ycpu, hn = w.hn, h_x = w.h_x, h_y = w.h_y, h_w = w.h_w, h_emb = w.h_emb;
        h_ids = w.h_ids, ev = w.ev, snap0 = w.snap0, snap_last = w.snap_last;
    }
    Win bound() const {
        return {x, y, ycpu, hn, h_x, h_y, h_w, h_emb, h_ids, ev, snap0, snap_last};
    }
    int64_t ctx_end = 0;
    // the last verify window, until accept()
    std::vector<int32_t> win_tok;
    int64_t win_pos = 0;
    int win_n = 0;

    ~Impl() {
        if (st)
            cudaStreamSynchronize(st);
        for (void* p : allocs)
            cudaFree(p);
        if (win[0].x)
            bind(win[0]);
        for (void* p :
             {(void*) h_x, (void*) h_y, (void*) h_w, (void*) h_emb, (void*) h_ids, (void*) h_arg, (void*) win[1].h_x,
              (void*) win[1].h_y, (void*) win[1].h_w, (void*) win[1].h_emb, (void*) win[1].h_ids})
            if (p)
                cudaFreeHost(p);
        if (win[1].ev)
            cudaEventDestroy(win[1].ev);
        wk.pool.reset(); // before the memory its workers might still touch
        wk.launcher.reset();
#if defined(__linux__)
        for (int n = 0; n < 2; ++n)
            if (arena[n])
                munmap(arena[n], arena_bytes[n]);
#endif
        if (ev)
            cudaEventDestroy(ev);
        if (st)
            cudaStreamDestroy(st);
    }

    template <class T> T* dalloc(size_t n, std::string& err) {
        void* p = nullptr;
        const size_t bytes = std::max<size_t>(n * sizeof(T), 256);
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            cudaGetLastError();
            err = "nemotron_h: out of GPU memory (" + std::to_string(bytes >> 20) + " MiB more, " +
                  std::to_string(vram >> 20) + " MiB placed)";
            return nullptr;
        }
        cudaMemset(p, 0, bytes);
        allocs.push_back(p);
        vram += bytes;
        return (T*) p;
    }

    // ---- tensors ----
    const uint8_t* data(const std::string& name, const TensorInfo** ti = nullptr) const {
        size_t s = 0;
        const TensorInfo* t = gguf->find(name, &s);
        if (!t)
            return nullptr;
        if (ti)
            *ti = t;
        return gguf->shard(s).tensor_data(*t);
    }
    /// a matrix on the GPU, followed by zeros enough for MMQ's 256-value steps past a row's end (llama.cpp pads its
    /// allocations the same way: MATRIX_ROW_PADDING)
    bool mat(const std::string& name, GgufMat& m, std::string& err) {
        const TensorInfo* t = nullptr;
        m.host = data(name, &t);
        if (!m.host) {
            err = "nemotron_h: missing tensor " + name;
            return false;
        }
        m.type = (int) t->type;
        m.n_in = (int64_t) t->shape[0];
        m.n_out = 1;
        for (size_t i = 1; i < t->shape.size(); ++i)
            m.n_out *= (int64_t) t->shape[i];
        m.bytes = m.row_bytes() * (size_t) m.n_out;
        const size_t pad = ggml_row_size((ggml_type) m.type, 512);
        uint8_t* p = dalloc<uint8_t>(m.bytes + pad, err);
        if (!p)
            return false;
        cudaMemcpy(p, m.host, m.bytes, cudaMemcpyHostToDevice);
        m.dev = p;
        return true;
    }
    /// a vector or matrix as float32 on the GPU (F32, F16 and BF16 exactly; other types through ggml's to_float)
    float* f32(const std::string& name, int64_t n, std::string& err) {
        const TensorInfo* t = nullptr;
        const uint8_t* h = data(name, &t);
        if (!h) {
            err = "nemotron_h: missing tensor " + name;
            return nullptr;
        }
        if ((int64_t) t->elements() != n) {
            err = "nemotron_h: " + name + " has " + std::to_string(t->elements()) + " values, not " + std::to_string(n);
            return nullptr;
        }
        std::vector<float> v((size_t) n);
        to_float((int) t->type, h, v.data(), n);
        float* p = dalloc<float>((size_t) n, err);
        if (p)
            cudaMemcpy(p, v.data(), (size_t) n * 4, cudaMemcpyHostToDevice);
        return p;
    }
    static void to_float(int type, const uint8_t* src, float* dst, int64_t n) {
        if (type == kF32) {
            std::memcpy(dst, src, (size_t) n * 4);
        } else if (type == kBF16) {
            for (int64_t i = 0; i < n; ++i) {
                uint16_t b;
                std::memcpy(&b, src + 2 * i, 2);
                const uint32_t u = (uint32_t) b << 16;
                std::memcpy(dst + i, &u, 4);
            }
        } else {
            ggml_get_type_traits((ggml_type) type)->to_float(src, dst, n);
        }
    }

    // ---- the GPU products ----
    void quant(const float* xin, int64_t n, int T, int slot) {
        dp.quant(xin, n, T, dp.xq[slot]);
    }
    void mm(const GgufMat& m, int slot, float* out, int T) {
        dp.mm(m, dp.xq[slot], out, T);
    }

    bool load(const std::string& path, const NemotronHOptions& o, std::string& err);
    bool embed_rows(const int32_t* tokens, const float* embeds, int T, float* dst, std::string& err);
    void mamba(Layer& Y, int T, bool prompt, bool snaps);
    void attention(Layer& Y, int T, int64_t pos0);
    void moe_begin(const Layer& Y, int T);
    bool moe_end(const Layer& Y, int T, const Win& w, std::string& err);
    void dump_row(const char* what, int l, const float* buf, int T);
    size_t segment(size_t l, int T, int64_t pos0, bool prompt, bool snaps);
    bool moe_finish(size_t l, int T, std::string& err);
    bool window_pair(const int32_t* tokens, const float* embeds, int T0, int T1, int64_t pos0, bool prompt, bool snaps,
                     std::string& err);
    bool window(const int32_t* tokens, const float* embeds, int T, int64_t pos0, bool prompt, bool snaps,
                std::string& err);
    bool head_rows(const float* rows, int n, float* host, std::string& err);
    bool mtp_rows(const float* h_rows, const int32_t* tokens, const float* embeds, int T, int64_t pos0, float* out,
                  std::string& err);
    bool mtp_catch_up(const int32_t* tokens, const float* embeds, int T, int64_t pos0, std::string& err);
};

// ---- loading --------------------------------------------------------------------------------------------------------

bool NemotronHEngine::Impl::load(const std::string& path, const NemotronHOptions& o, std::string& err) {
    opt = o;
    try {
        gguf = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception& ex) {
        err = std::string("nemotron_h: ") + ex.what();
        return false;
    }
    const GgufFile& meta = gguf->meta();
    const MetaValue* arch = meta.get("general.architecture");
    if (!arch || arch->s != "nemotron_h_moe") {
        err = "nemotron_h: not a nemotron_h_moe model";
        return false;
    }
    const std::string Ar = arch->s + ".";
    auto num = [&](const char* k, double def) {
        const MetaValue* v = meta.get(Ar + k);
        return v && v->is_num() ? v->num() : def;
    };
    auto per_layer = [&](const char* k, int64_t l, double def) {
        const MetaValue* v = meta.get(Ar + k);
        if (!v)
            return def;
        if (v->is_num())
            return v->num();
        return l < (int64_t) v->items.size() ? v->items[(size_t) l].num() : def;
    };
    g.n_embd = (int64_t) num("embedding_length", 2688);
    const int64_t blocks = (int64_t) num("block_count", 53);
    const int64_t nextn = (int64_t) num("nextn_predict_layers", 0);
    g.n_layer = blocks - nextn;
    g.ssm_conv = (int64_t) num("ssm.conv_kernel", 4);
    g.ssm_inner = (int64_t) num("ssm.inner_size", 4096);
    g.ssm_state = (int64_t) num("ssm.state_size", 128);
    g.ssm_heads = (int64_t) num("ssm.time_step_rank", 64);
    g.ssm_groups = (int64_t) num("ssm.group_count", 8);
    g.n_expert = (int64_t) num("expert_count", 128);
    g.n_used = (int64_t) num("expert_used_count", 6);
    g.n_ff_exp = (int64_t) num("expert_feed_forward_length", 1856);
    g.n_ff_shexp = (int64_t) num("expert_shared_feed_forward_length", 3712);
    g.expert_norm = num("expert_weights_norm", 1) != 0;
    g.expert_scale = (float) num("expert_weights_scale", 2.5);
    g.rms_eps = (float) num("attention.layer_norm_rms_epsilon", 1e-5);
    g.ln_eps = (float) num("attention.layer_norm_epsilon", 1e-5);
    g.head_dim = (int64_t) num("attention.key_length", 128);
    g.kinds.assign((size_t) g.n_layer, 'M');
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const double hkv = per_layer("attention.head_count_kv", l, 0), ff = per_layer("feed_forward_length", l, 0);
        g.kinds[(size_t) l] = hkv == 0 && ff == 0 ? 'M' : ff == 0 ? '*' : 'E';
        if (g.kinds[(size_t) l] == '*') {
            g.n_head = (int64_t) per_layer("attention.head_count", l, 32);
            g.n_head_kv = (int64_t) hkv;
        }
    }
    const int64_t N = g.n_embd, P = g.ssm_inner / g.ssm_heads;
    if (!nhk::ssm_scan_supported((int) P, (int) g.ssm_state) || g.ssm_conv != 4 || g.head_dim != 128 ||
        g.n_expert > glm5k::kMaxExperts || g.n_used > 32 || g.n_head % g.n_head_kv != 0 ||
        g.ssm_heads % g.ssm_groups != 0 || g.ssm_inner % g.ssm_groups != 0) {
        err = "nemotron_h: unexpected geometry (Mamba head " + std::to_string(P) + ", state " +
              std::to_string(g.ssm_state) + ", attention head " + std::to_string(g.head_dim) + ")";
        return false;
    }
    if (opt.max_context > nhk::attend_max_cells())
        opt.max_context = nhk::attend_max_cells();
    const int64_t ctx = opt.max_context;

    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        err = "nemotron_h: cannot create a CUDA stream";
        return false;
    }
    wk = cpu_moe::make_workers(opt.numa_split, opt.threads);

    // the token embedding stays on the host (the audio rows replace some of it)
    {
        const TensorInfo* t = nullptr;
        embd = data("token_embd.weight", &t);
        if (!embd || t->shape.size() != 2 || (int64_t) t->shape[0] != N) {
            err = "nemotron_h: no token_embd.weight of " + std::to_string(N) + " columns";
            return false;
        }
        embd_type = (int) t->type;
        embd_row = ggml_row_size((ggml_type) embd_type, N);
    }
    if (!mat("output.weight", head, err) || !(out_norm = f32("output_norm.weight", N, err)))
        return false;
    g.n_vocab = head.n_out;

    const int64_t C = g.ssm_inner + 2 * g.ssm_groups * g.ssm_state, H = g.ssm_heads;
    const int64_t kv_dim = g.n_head_kv * g.head_dim;
    auto moe = [&](Layer& y, const std::string& p, std::string& e) -> bool {
        if (!(y.router = f32(p + "ffn_gate_inp.weight", N * g.n_expert, e)) ||
            !(y.bias = f32(p + "exp_probs_b.bias", g.n_expert, e)) || !mat(p + "ffn_up_shexp.weight", y.up_sh, e) ||
            !mat(p + "ffn_down_shexp.weight", y.down_sh, e))
            return false;
        const TensorInfo *tu = nullptr, *td = nullptr;
        const uint8_t* up = data(p + "ffn_up_exps.weight", &tu);
        const uint8_t* down = data(p + "ffn_down_exps.weight", &td);
        if (!up || !down) {
            e = "nemotron_h: missing the routed experts of " + p;
            return false;
        }
        const size_t gu_row = ggml_row_size((ggml_type) tu->type, N);
        const size_t d_row = ggml_row_size((ggml_type) td->type, g.n_ff_exp);
        y.experts.resize((size_t) g.n_expert);
        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
            CpuFfn& f = y.experts[(size_t) ex];
            f.act = CpuFfn::ReLU2;
            f.gu_type = (int) tu->type;
            f.d_type = (int) td->type;
            f.n_ff = g.n_ff_exp;
            f.gu_row = gu_row;
            f.d_row = d_row;
            f.up[0] = f.up[1] = up + (size_t) ex * g.n_ff_exp * gu_row;
            f.down[0] = f.down[1] = down + (size_t) ex * N * d_row;
            f.gu_split = g.n_ff_exp;
            f.d_split = N;
        }
        return true;
    };
    auto attn = [&](Layer& y, const std::string& p, std::string& e) -> bool {
        if (!mat(p + "attn_q.weight", y.wq, e) || !mat(p + "attn_k.weight", y.wk, e) ||
            !mat(p + "attn_v.weight", y.wv, e) || !mat(p + "attn_output.weight", y.wo, e))
            return false;
        return (y.kc = dalloc<uint16_t>((size_t) ctx * kv_dim, e)) &&
               (y.vc = dalloc<uint16_t>((size_t) ctx * kv_dim, e));
    };
    L.resize((size_t) g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& y = L[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.kind = g.kinds[(size_t) l];
        if (!(y.norm = f32(p + "attn_norm.weight", N, err)))
            return false;
        if (y.kind == 'M') {
            if (!mat(p + "ssm_in.weight", y.in_proj, err) || !mat(p + "ssm_out.weight", y.out_proj, err) ||
                !(y.conv_w = f32(p + "ssm_conv1d.weight", C * 4, err)) ||
                !(y.conv_b = f32(p + "ssm_conv1d.bias", C, err)) || !(y.dt_b = f32(p + "ssm_dt.bias", H, err)) ||
                !(y.A = f32(p + "ssm_a", H, err)) || !(y.D = f32(p + "ssm_d", H, err)) ||
                !(y.ssm_norm = f32(p + "ssm_norm.weight", g.ssm_inner, err)) ||
                !(y.ssm_state = dalloc<float>((size_t) H * P * g.ssm_state, err)) ||
                !(y.conv_state = dalloc<float>((size_t) C * 3, err)) ||
                !(y.ssm_snap = dalloc<float>((size_t) (kMaxVerify - 1) * H * P * g.ssm_state, err)) ||
                !(y.conv_snap = dalloc<float>((size_t) (kMaxVerify - 1) * C * 3, err)))
                return false;
            if (y.in_proj.n_out != 2 * g.ssm_inner + 2 * g.ssm_groups * g.ssm_state + H) {
                err = "nemotron_h: " + p + "ssm_in has " + std::to_string(y.in_proj.n_out) + " rows";
                return false;
            }
        } else if (y.kind == '*') {
            if (!attn(y, p, err))
                return false;
        } else if (!moe(y, p, err)) {
            return false;
        }
    }
    has_mtp = false;
    if (opt.mtp && nextn > 0) {
        const std::string p = "blk." + std::to_string(g.n_layer) + ".";
        std::string why;
        mtp.kind = 'T';
        if (data(p + "nextn.eh_proj.weight") && mat(p + "nextn.eh_proj.weight", mtp.eh_proj, why) &&
            (mtp.enorm = f32(p + "nextn.enorm.weight", N, why)) &&
            (mtp.hnorm = f32(p + "nextn.hnorm.weight", N, why)) &&
            (mtp.head_norm = f32(p + "nextn.shared_head_norm.weight", N, why)) &&
            (mtp.norm = f32(p + "attn_norm.weight", N, why)) &&
            (mtp.post_norm = f32(p + "post_attention_norm.weight", N, why)) && attn(mtp, p, why) && moe(mtp, p, why))
            has_mtp = true;
        else
            std::fprintf(stderr, "guild nemotron_h: no usable MTP block (%s); decoding without MTP\n", why.c_str());
    }

    // scratch for kPromptT rows
    const int W = kPromptT;
    const int64_t widest = std::max<int64_t>({2 * N, g.ssm_inner, g.n_ff_shexp, g.n_head * g.head_dim});
    if (!(x = dalloc<float>((size_t) W * N, err)) || !(xn = dalloc<float>((size_t) W * N, err)) ||
        !(a = dalloc<float>((size_t) W * std::max<int64_t>(L.empty() ? 0 : L[0].in_proj.n_out, 2 * N), err)) ||
        !(cv = dalloc<float>((size_t) W * C, err)) || !(ys = dalloc<float>((size_t) W * g.ssm_inner, err)) ||
        !(yn = dalloc<float>((size_t) W * g.ssm_inner, err)) ||
        !(q = dalloc<float>((size_t) W * g.n_head * g.head_dim, err)) ||
        !(k = dalloc<float>((size_t) W * kv_dim, err)) || !(v = dalloc<float>((size_t) W * kv_dim, err)) ||
        !(att = dalloc<float>((size_t) W * g.n_head * g.head_dim, err)) ||
        !(sa = dalloc<float>((size_t) W * g.n_ff_shexp, err)) ||
        !(sb = dalloc<float>((size_t) W * g.n_ff_shexp, err)) || !(y = dalloc<float>((size_t) W * N, err)) ||
        !(ycpu = dalloc<float>((size_t) W * N, err)) || !(rlog = dalloc<float>((size_t) W * g.n_expert, err)) ||
        !(rw = dalloc<float>((size_t) W * g.n_used, err)) || !(rids = dalloc<int32_t>((size_t) W * g.n_used, err)) ||
        !(logits = dalloc<float>((size_t) kMaxVerify * g.n_vocab, err)) || !(hn = dalloc<float>((size_t) W * N, err)) ||
        !(cat = dalloc<float>((size_t) W * 2 * N, err)) || !(pending = dalloc<float>((size_t) N, err)) ||
        !(mh = dalloc<float>((size_t) W * N, err)) || !(d_arg = dalloc<int32_t>(1, err)))
        return false;
    if (!dp.init(
            st, widest, W, std::getenv("GUILD_NEMOTRON_MMQ") == nullptr,
            [&](size_t bytes) { return (void*) dalloc<uint8_t>(bytes, err); }, err))
        return false;
    int n_mmq = 0;
    if (dp.mmq_on()) {
        // the padding after each matrix lets MMQ step past a row of 2,688 or 3,712 values into zeros
        auto mark = [&](GgufMat& m) {
            if (!m.dev)
                return;
            const int64_t n_in = m.n_in;
            m.n_in = (n_in + 255) / 256 * 256; // fits() and the tile sizes look at the padded width only
            const bool ok = dp.mark(m);
            m.n_in = n_in;
            n_mmq += ok;
        };
        for (Layer& yl : L)
            for (GgufMat* m : {&yl.in_proj, &yl.out_proj, &yl.wq, &yl.wk, &yl.wv, &yl.wo, &yl.up_sh, &yl.down_sh})
                mark(*m);
        for (GgufMat* m : {&mtp.eh_proj, &mtp.wq, &mtp.wk, &mtp.wv, &mtp.wo, &mtp.up_sh, &mtp.down_sh})
            mark(*m);
    }
    if (cudaMallocHost(&h_x, (size_t) W * N * 4) != cudaSuccess || cudaMallocHost(&h_y, (size_t) W * N * 4) ||
        cudaMallocHost(&h_w, (size_t) W * g.n_used * 4) || cudaMallocHost(&h_ids, (size_t) W * g.n_used * 4) ||
        cudaMallocHost(&h_emb, (size_t) W * N * 4) || cudaMallocHost(&h_arg, 64)) {
        err = "nemotron_h: no pinned host memory";
        return false;
    }
    win[0] = {x, y, ycpu, hn, h_x, h_y, h_w, h_emb, h_ids, ev};
    Win& w1 = win[1];
    if (!(w1.x = dalloc<float>((size_t) W * N, err)) || !(w1.y = dalloc<float>((size_t) W * N, err)) ||
        !(w1.ycpu = dalloc<float>((size_t) W * N, err)) || !(w1.hn = dalloc<float>((size_t) W * N, err)))
        return false;
    if (cudaMallocHost(&w1.h_x, (size_t) W * N * 4) != cudaSuccess || cudaMallocHost(&w1.h_y, (size_t) W * N * 4) ||
        cudaMallocHost(&w1.h_w, (size_t) W * g.n_used * 4) || cudaMallocHost(&w1.h_ids, (size_t) W * g.n_used * 4) ||
        cudaMallocHost(&w1.h_emb, (size_t) W * N * 4) ||
        cudaEventCreateWithFlags(&w1.ev, cudaEventDisableTiming) != cudaSuccess) {
        err = "nemotron_h: no pinned host memory";
        return false;
    }

    // the routed experts: halves of every expert's rows in memory bound to each node (or one copy: no split)
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<CpuFfn*> all;
    for (Layer& yl : L)
        for (CpuFfn& f : yl.experts)
            all.push_back(&f);
    if (has_mtp)
        for (CpuFfn& f : mtp.experts)
            all.push_back(&f);
    if (!cpu_moe::place_ffns(*wk.pool, all, N, wk.split, false, arena, arena_bytes, err)) {
        err = "nemotron_h: " + err;
        return false;
    }
    std::fprintf(stderr, "guild nemotron_h: experts%s: %.1f + %.1f GiB in %.1f s\n",
                 wk.split ? " split across 2 NUMA nodes" : "", arena_bytes[0] / 1073741824.0,
                 arena_bytes[1] / 1073741824.0, ms_since(t0) / 1000.0);

    char line[320];
    std::snprintf(line, sizeof line,
                  "nemotron_h: %lld layers (%d Mamba-2, %d MoE, %d attention)%s, %d CPU threads%s, %llu MiB VRAM, "
                  "context %lld, %d matrices take MMQ in prompt windows\n",
                  (long long) g.n_layer, (int) std::count(g.kinds.begin(), g.kinds.end(), 'M'),
                  (int) std::count(g.kinds.begin(), g.kinds.end(), 'E'),
                  (int) std::count(g.kinds.begin(), g.kinds.end(), '*'), has_mtp ? " + MTP" : "", wk.pool->size(),
                  wk.split ? " on 2 NUMA nodes" : "", (unsigned long long) (vram >> 20), (long long) ctx, n_mmq);
    placement = line;
    return true;
}

// ---- the forward pass
// -------------------------------------------------------------------------------------------------

bool NemotronHEngine::Impl::embed_rows(const int32_t* tokens, const float* embeds, int T, float* dst,
                                       std::string& err) {
    const int64_t N = g.n_embd;
    for (int t = 0; t < T; ++t) {
        float* row = h_emb + (size_t) t * N;
        if (tokens[t] < 0) {
            if (!embeds) {
                err = "nemotron_h: an embedding row without embeddings";
                return false;
            }
            std::memcpy(row, embeds + (size_t) t * N, (size_t) N * 4);
        } else if (tokens[t] < g.n_vocab) {
            to_float(embd_type, embd + (size_t) tokens[t] * embd_row, row, N);
        } else {
            err = "nemotron_h: token " + std::to_string(tokens[t]) + " is past the vocabulary";
            return false;
        }
    }
    cudaMemcpyAsync(dst, h_emb, (size_t) T * N * 4, cudaMemcpyHostToDevice, st);
    return true;
}

/// xn -> y: the Mamba-2 mixer.  `prompt`: the time step clamped at 0.001 (the reference's chunked scan clamps,
/// its one-token step does not); `snaps`: the states after each token but the last kept (a verify window)
void NemotronHEngine::Impl::mamba(Layer& Y, int T, bool prompt, bool snaps) {
    const int64_t I = g.ssm_inner, C = I + 2 * g.ssm_groups * g.ssm_state, H = g.ssm_heads;
    const int ld = (int) Y.in_proj.n_out; // z | xBC | dt
    quant(xn, g.n_embd, T, 0);
    mm(Y.in_proj, 0, a, T);
    const int n_snaps = snaps ? (snap_last ? T : T - 1) : 0;
    const size_t ssm_floats = (size_t) H * (I / H) * g.ssm_state;
    nhk::ssm_conv(a + I, ld, Y.conv_state, Y.conv_w, Y.conv_b, cv, (int) C, T,
                  snaps ? Y.conv_snap + (size_t) snap0 * C * 3 : nullptr, n_snaps, st);
    nhk::ssm_scan(cv, (int) C, a + I + C, ld, Y.dt_b, Y.A, Y.D, Y.ssm_state, ys, (int) H, (int) (I / H),
                  (int) g.ssm_state, (int) g.ssm_groups, T, prompt ? 0.001f : 0.f,
                  snaps ? Y.ssm_snap + (size_t) snap0 * ssm_floats : nullptr, n_snaps, st);
    nhk::gated_group_norm(ys, a, ld, Y.ssm_norm, yn, (int) I, (int) (I / g.ssm_groups), g.rms_eps, T, st);
    quant(yn, I, T, 1);
    mm(Y.out_proj, 1, y, T);
}

/// xn -> y: grouped-query attention over the cells [0, pos0 + t]
void NemotronHEngine::Impl::attention(Layer& Y, int T, int64_t pos0) {
    const int64_t kv_dim = g.n_head_kv * g.head_dim;
    quant(xn, g.n_embd, T, 0);
    mm(Y.wq, 0, q, T);
    mm(Y.wk, 0, k, T);
    mm(Y.wv, 0, v, T);
    nhk::kv_store(k, v, Y.kc, Y.vc, pos0, (int) kv_dim, T, st);
    nhk::gqa_attend(q, Y.kc, Y.vc, att, (int) g.n_head, (int) g.n_head_kv, (int) g.head_dim, pos0, T,
                    1.0f / std::sqrt((float) g.head_dim), st);
    quant(att, g.n_head * g.head_dim, T, 1);
    mm(Y.wo, 1, y, T);
}

/// xn -> the router's choice and the normed rows to the host; the shared expert into y on the GPU meanwhile
void NemotronHEngine::Impl::moe_begin(const Layer& Y, int T) {
    const int64_t N = g.n_embd;
    // the router's logits of a decode or verify window one token at a time: f32_matvec's one-token kernel, so that a
    // verify row's routing is bitwise a decode step's (its wider kernels sum in another order, and this router's
    // choices are often within float rounding of each other: the selection bias is ~41, the sigmoid ~0.01 apart)
    if (T <= kMaxVerify)
        for (int t = 0; t < T; ++t)
            glm5k::f32_matvec(Y.router, xn + (size_t) t * N, rlog + (size_t) t * g.n_expert, (int) N, (int) g.n_expert,
                              1, st);
    else
        glm5k::f32_matvec(Y.router, xn, rlog, (int) N, (int) g.n_expert, T, st);
    glm5k::route(rlog, Y.bias, (int) g.n_expert, (int) g.n_used, g.expert_norm, g.expert_scale, rids, rw, T, st);
    cudaMemcpyAsync(h_ids, rids, (size_t) T * g.n_used * 4, cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(h_w, rw, (size_t) T * g.n_used * 4, cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(h_x, xn, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st);
    cudaEventRecord(ev, st);
    const int64_t F = Y.up_sh.n_out;
    quant(xn, N, T, 0);
    mm(Y.up_sh, 0, sa, T);
    nhk::relu2(sa, sb, (int64_t) T * F, st);
    quant(sb, F, T, 1);
    mm(Y.down_sh, 1, y, T);
}

/// window w's routed experts on the CPU, added to the shared expert's rows in w.y.  Its buffers only, not the bound
/// ones: a window pair runs it beside the launcher, which binds the other window meanwhile.
bool NemotronHEngine::Impl::moe_end(const Layer& Y, int T, const Win& w, std::string& err) {
    const int64_t N = g.n_embd;
    const auto w0 = std::chrono::steady_clock::now();
    if (cudaEventSynchronize(w.ev) != cudaSuccess) {
        err = "nemotron_h: CUDA error before the CPU experts: " + std::string(cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    gpu_wait_ms += std::chrono::duration<double, std::milli>(t0 - w0).count();
    xp.run(*wk.pool, N, T, kPromptT, w.h_ids, w.h_w, (int) g.n_used, Y.experts, nullptr, w.h_x, w.h_y, T <= kMaxVerify);
    cpu_ms += ms_since(t0);

    cudaMemcpyAsync(w.ycpu, w.h_y, (size_t) T * N * 4, cudaMemcpyHostToDevice, st);
    glm5k::add(w.y, w.ycpu, (int64_t) T * N, st);
    return true;
}

/// GUILD_NEMOTRON_DUMP=FILE (debugging): every layer's mixer output and block output for the window's last row, in
/// the record form of llama.cpp's eval callback names (name, n, floats)
void NemotronHEngine::Impl::dump_row(const char* what, int l, const float* buf, int T) {
    static std::FILE* dump = [] {
        const char* p = std::getenv("GUILD_NEMOTRON_DUMP");
        return p && *p ? std::fopen(p, "wb") : nullptr;
    }();
    if (!dump)
        return;
    const int64_t N = g.n_embd;
    static const bool all = std::getenv("GUILD_NEMOTRON_DUMP_ALL") != nullptr; // every row of the window
    const int rows = all ? T : 1;
    std::vector<float> h((size_t) rows * N);
    cudaMemcpyAsync(h.data(), buf + (size_t) (T - rows) * N, h.size() * 4, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    const std::string name = std::string(what) + "-" + std::to_string(l);
    const int32_t len = (int32_t) name.size(), n = (int32_t) h.size();
    std::fwrite(&len, 4, 1, dump);
    std::fwrite(name.data(), 1, (size_t) len, dump);
    std::fwrite(&n, 4, 1, dump);
    std::fwrite(h.data(), 4, h.size(), dump);
    std::fflush(dump);
}

/// The bound window's rows [0, T) at pos0 from layer l through the next MoE layer's moe_begin: that layer's index;
/// or past the last layer, the final norm (x -> hn) and L.size()
size_t NemotronHEngine::Impl::segment(size_t l, int T, int64_t pos0, bool prompt, bool snaps) {
    const int64_t N = g.n_embd;
    for (; l < L.size(); ++l) {
        Layer& Y = L[l];
        glm5k::rms_norm(x, Y.norm, xn, (int) N, T, g.rms_eps, st);
        if (Y.kind == 'M') {
            mamba(Y, T, prompt, snaps);
            dump_row("mamba_out", (int) l, y, T);
        } else if (Y.kind == '*') {
            attention(Y, T, pos0);
            dump_row("attn_out", (int) l, y, T);
        } else {
            moe_begin(Y, T);
            dump_row("shexp_out", (int) l, y, T);
            return l;
        }
        glm5k::add(x, y, (int64_t) T * N, st);
        dump_row("nemotron_h_block_out", (int) l, x, T);
    }
    glm5k::rms_norm(x, out_norm, hn, (int) N, T, g.rms_eps, st);
    return L.size();
}

/// the bound window's MoE layer l after segment(): its CPU experts, and the layer's output added to its rows
bool NemotronHEngine::Impl::moe_finish(size_t l, int T, std::string& err) {
    if (!moe_end(L[l], T, bound(), err))
        return false;
    dump_row("ffn_out", (int) l, y, T);
    glm5k::add(x, y, (int64_t) T * g.n_embd, st);
    dump_row("nemotron_h_block_out", (int) l, x, T);
    return true;
}

/// rows [0, T) at pos0 through every layer: x -> hn (the final norm's output, the head's input)
bool NemotronHEngine::Impl::window(const int32_t* tokens, const float* embeds, int T, int64_t pos0, bool prompt,
                                   bool snaps, std::string& err) {
    if (!embed_rows(tokens, embeds, T, x, err))
        return false;
    dump_row("block_in", 0, x, T);
    for (size_t l = 0; (l = segment(l, T, pos0, prompt, snaps)) < L.size(); ++l)
        if (!moe_finish(l, T, err))
            return false;
    return true;
}

/// Two windows, T0 rows at pos0 and T1 rows right after (win[0] and win[1]): at each MoE layer the CPU experts of
/// one window here while the launcher enqueues the other's layers up to its next MoE layer (bound to that window
/// meanwhile; moe_end takes its window's own buffers).  The stream keeps every layer's order (window 0 before window
/// 1: the Mamba states and the attention cells pass from one to the next), so each window computes what window()
/// computes for it.  `snaps`: the two halves of a verify window, whose snapshots are the whole window's (each verify
/// row is a decode step's: the products, the router and the CPU experts compute a row alike in a window of 1 to
/// kMaxVerify rows).
bool NemotronHEngine::Impl::window_pair(const int32_t* tokens, const float* embeds, int T0, int T1, int64_t pos0,
                                        bool prompt, bool snaps, std::string& err) {
    const int64_t N = g.n_embd;
    cpu_moe::Launcher& lc = *wk.launcher;
    win[0].snap0 = 0, win[0].snap_last = true;
    win[1].snap0 = T0, win[1].snap_last = false;
    bind(win[0]);
    bool ok = embed_rows(tokens, embeds, T0, x, err);
    size_t l0 = ok ? segment(0, T0, pos0, prompt, snaps) : L.size(), l1 = 0;
    bool ok1 = true;
    std::string err1;
    auto launch0 = [&](size_t from) {
        bind(win[0]);
        l0 = segment(from, T0, pos0, prompt, snaps);
    };
    auto launch1 = [&](size_t from) {
        bind(win[1]);
        if (from == 0 && !embed_rows(tokens + T0, embeds ? embeds + (size_t) T0 * N : nullptr, T1, x, err1))
            ok1 = false;
        else
            l1 = segment(from, T1, pos0 + T0, prompt, snaps);
    };
    if (ok)
        lc.post([&] { launch1(0); });
    while (ok && l0 < L.size()) { // both windows meet the same MoE layers
        const size_t m = l0;
        ok = moe_end(L[m], T0, win[0], err); // window 0's experts here, window 1's layers to MoE layer m meanwhile
        glm5k::add(win[0].x, win[0].y, (int64_t) T0 * N, st);
        lc.wait();
        if (!ok1) {
            err = err1;
            ok = false;
        }
        if (!ok)
            break;
        lc.post([&, m] { launch0(m + 1); }); // window 0's next layers, window 1's experts meanwhile
        ok = moe_end(L[m], T1, win[1], err);
        glm5k::add(win[1].x, win[1].y, (int64_t) T1 * N, st);
        lc.wait();
        if (ok && l0 < L.size())
            lc.post([&, m] { launch1(m + 1); });
        else if (ok)
            launch1(m + 1); // past the last MoE layer: window 1's last layers
    }
    win[0].snap_last = false;
    bind(win[0]);
    return ok;
}

/// logits of n rows of normed hidden states (device), to the host
bool NemotronHEngine::Impl::head_rows(const float* rows, int n, float* host, std::string& err) {
    quant(rows, g.n_embd, n, 0);
    mm(head, 0, logits, n);
    if (cudaMemcpyAsync(host, logits, (size_t) n * g.n_vocab * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess) {
        err = "nemotron_h: CUDA error: " + std::string(cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return true;
}

/// The MTP block over T rows: hidden rows h_rows (device) paired with the next tokens (embedding rows where negative)
/// at cells pos0 ..; out: the block's normed outputs (device, the head's input)
bool NemotronHEngine::Impl::mtp_rows(const float* h_rows, const int32_t* tokens, const float* embeds, int T,
                                     int64_t pos0, float* out, std::string& err) {
    const int64_t N = g.n_embd;
    Layer& Y = mtp;
    if (!embed_rows(tokens, embeds, T, xn, err))
        return false;
    // cat [T][2N] = [enorm(embedding) | hnorm(hidden)]
    glm5k::rms_norm(xn, Y.enorm, x, (int) N, T, g.rms_eps, st);
    glm5k::rms_norm(h_rows, Y.hnorm, xn, (int) N, T, g.rms_eps, st);
    cudaMemcpy2DAsync(cat, 2 * N * 4, x, N * 4, N * 4, T, cudaMemcpyDeviceToDevice, st);
    cudaMemcpy2DAsync(cat + N, 2 * N * 4, xn, N * 4, N * 4, T, cudaMemcpyDeviceToDevice, st);
    quant(cat, 2 * N, T, 0);
    mm(Y.eh_proj, 0, x, T);
    // attention sublayer, then the MoE sublayer, each with its residual
    glm5k::rms_norm(x, Y.norm, xn, (int) N, T, g.rms_eps, st);
    attention(Y, T, pos0);
    glm5k::add(x, y, (int64_t) T * N, st);
    glm5k::rms_norm(x, Y.post_norm, xn, (int) N, T, g.rms_eps, st);
    moe_begin(Y, T);
    if (!moe_end(Y, T, bound(), err))
        return false;
    glm5k::add(x, y, (int64_t) T * N, st);
    if (opt.mtp_layer_norm)
        nhk::layer_norm(x, Y.head_norm, out, (int) N, T, g.ln_eps, st);
    else
        glm5k::rms_norm(x, Y.head_norm, out, (int) N, T, g.rms_eps, st);
    return true;
}

/// After a main window of T rows at pos0 (hidden rows in hn): the MTP block over (the row waiting for its next token
/// and every row but the window's last) paired with the window's tokens, so that its cells follow the context; the
/// window's last row waits.  The main window's hn is read before it is overwritten.
bool NemotronHEngine::Impl::mtp_catch_up(const int32_t* tokens, const float* embeds, int T, int64_t pos0,
                                         std::string& err) {
    const int64_t N = g.n_embd;
    // rows: [pending, hn[0 .. T - 2]] with tokens [0 .. T - 1] at cells pos0 - 1 ..; without pending: hn[0 .. T - 2]
    // with tokens [1 .. T - 1] at cells pos0 ..
    const int off = has_pending ? 0 : 1, n = T - off;
    if (n > 0) {
        if (has_pending)
            cudaMemcpyAsync(mh, pending, (size_t) N * 4, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(mh + (size_t) (1 - off) * N, hn, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(pending, hn + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st);
        if (!mtp_rows(mh, tokens + off, embeds ? embeds + (size_t) off * N : nullptr, n, pos0 - 1 + off, hn, err))
            return false;
    } else {
        cudaMemcpyAsync(pending, hn + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st);
    }
    has_pending = true;
    return true;
}

// ---- the public face ------------------------------------------------------------------------------------------------

NemotronHEngine::NemotronHEngine() : p_(std::make_unique<Impl>()) {
}
NemotronHEngine::~NemotronHEngine() = default;

bool NemotronHEngine::load(const std::string& gguf, const NemotronHOptions& opt, std::string& err) {
    return p_->load(gguf, opt, err);
}
const NemotronHGeometry& NemotronHEngine::geometry() const {
    return p_->g;
}
int64_t NemotronHEngine::max_context() const {
    return p_->opt.max_context;
}
uint64_t NemotronHEngine::vram_bytes() const {
    return p_->vram;
}
const std::string& NemotronHEngine::placement() const {
    return p_->placement;
}
bool NemotronHEngine::has_mtp() const {
    return p_->has_mtp;
}
int NemotronHEngine::max_drafts() const {
    return p_->has_mtp ? kMaxVerify - 1 : 0;
}
double NemotronHEngine::last_cpu_ms() const {
    return p_->cpu_ms;
}
void NemotronHEngine::print_timing(const char* what) const {
    if (std::getenv("GUILD_MOE_TIMING"))
        std::fprintf(stderr,
                     "nemotron_h timing (%s): activations %.0f ms, up %.0f, hidden %.0f, down %.0f; %.2f GB of "
                     "expert rows; %.0f ms waiting for the GPU before them\n",
                     what, p_->xp.phase_ms[0], p_->xp.phase_ms[1], p_->xp.phase_ms[2], p_->xp.phase_ms[3],
                     p_->xp.weight_bytes * 1e-9, p_->gpu_wait_ms);
}

void NemotronHEngine::reset() {
    Impl& m = *p_;
    const int64_t C = m.g.ssm_inner + 2 * m.g.ssm_groups * m.g.ssm_state;
    for (Layer& y : m.L)
        if (y.kind == 'M') {
            cudaMemsetAsync(y.ssm_state, 0,
                            (size_t) m.g.ssm_heads * (m.g.ssm_inner / m.g.ssm_heads) * m.g.ssm_state * 4, m.st);
            cudaMemsetAsync(y.conv_state, 0, (size_t) C * 3 * 4, m.st);
        }
    cudaStreamSynchronize(m.st);
    m.ctx_end = 0;
    m.has_pending = false;
    m.win_tok.clear();
}

bool NemotronHEngine::forward(const int32_t* tokens, const float* embeds, int n, int64_t pos0, float* logits,
                              std::string& err) {
    Impl& m = *p_;
    const cpu_moe::PinCaller pin(m.wk.caller_cpu);
    m.cpu_ms = 0;
    if (pos0 + n > m.opt.max_context) {
        err = "nemotron_h: the context holds " + std::to_string(m.opt.max_context) + " tokens";
        return false;
    }
    const int64_t N = m.g.n_embd;
    // two windows at a time, each one's CPU experts beside the other's GPU layers (window_pair); one at a time with
    // GUILD_NEMOTRON_DUMP, whose rows follow the layers of one window
    static const bool pairs = std::getenv("GUILD_NEMOTRON_DUMP") == nullptr;
    for (int done = 0; done < n;) {
        const int T = std::min(kPromptT, n - done), T1 = pairs ? std::min(kPromptT, n - done - T) : 0;
        const float* e = embeds ? embeds + (size_t) done * N : nullptr;
        if (T1 > 0) {
            if (!m.window_pair(tokens + done, e, T, T1, pos0 + done, true, false, err))
                return false;
            if (done + T + T1 == n && logits && !m.head_rows(m.win[1].hn + (size_t) (T1 - 1) * N, 1, logits, err))
                return false;
            if (m.has_mtp) {
                if (!m.mtp_catch_up(tokens + done, e, T, pos0 + done, err))
                    return false;
                m.bind(m.win[1]); // the second window's rows in its own hn
                const bool ok =
                    m.mtp_catch_up(tokens + done + T, e ? e + (size_t) T * N : nullptr, T1, pos0 + done + T, err);
                m.bind(m.win[0]);
                if (!ok)
                    return false;
            }
            done += T + T1;
            continue;
        }
        // a prompt: the reference reads it with its chunked scan, which clamps the time step
        if (!m.window(tokens + done, e, T, pos0 + done, true, false, err))
            return false;
        if (done + T == n && logits &&
            !m.head_rows(m.hn + (size_t) (T - 1) * N, 1, logits, err)) // before mtp_catch_up reuses hn
            return false;
        if (m.has_mtp && !m.mtp_catch_up(tokens + done, e, T, pos0 + done, err))
            return false;
        done += T;
    }
    m.ctx_end = pos0 + n;
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = "nemotron_h: CUDA error: " + std::string(cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return true;
}

bool NemotronHEngine::step(int32_t token, int64_t pos, float* logits, std::string& err) {
    Impl& m = *p_;
    if (pos != m.ctx_end || pos + 1 > m.opt.max_context) {
        err = "nemotron_h: a step continues the context at its end";
        return false;
    }
    const cpu_moe::PinCaller pin(m.wk.caller_cpu);
    m.cpu_ms = 0;
    if (!m.window(&token, nullptr, 1, pos, false, false, err) || (logits && !m.head_rows(m.hn, 1, logits, err)) ||
        (m.has_mtp && !m.mtp_catch_up(&token, nullptr, 1, pos, err)))
        return false;
    m.ctx_end = pos + 1;
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = "nemotron_h: CUDA error: " + std::string(cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return true;
}

bool NemotronHEngine::draft(int32_t last, int64_t pos, int kd, int32_t* drafts, std::string& err) {
    Impl& m = *p_;
    if (!m.has_mtp || !m.has_pending || pos != m.ctx_end || kd < 1 || kd > kMaxVerify - 1) {
        err = "nemotron_h: draft needs the MTP block, a context ending at its position and 1.." +
              std::to_string(kMaxVerify - 1) + " drafts";
        return false;
    }
    const cpu_moe::PinCaller pin(m.wk.caller_cpu);
    const int64_t N = m.g.n_embd;
    // step i: (hidden, token) at cell pos - 1 + i; the first pairs the main model's waiting row with `last`, the next
    // ones the block's own output with the previous draft (the repeated layer)
    int32_t tok = last;
    const float* h = m.pending;
    for (int i = 0; i < kd; ++i) {
        float* out = m.mh + (size_t) (i % 2) * N;
        if (!m.mtp_rows(h, &tok, nullptr, 1, pos - 1 + i, out, err))
            return false;
        m.quant(out, N, 1, 0);
        m.mm(m.head, 0, m.logits, 1);
        glm5k::argmax(m.logits, (int) m.g.n_vocab, m.d_arg, m.st);
        cudaMemcpyAsync(m.h_arg, m.d_arg, 4, cudaMemcpyDeviceToHost, m.st);
        if (cudaStreamSynchronize(m.st) != cudaSuccess) {
            err = "nemotron_h: CUDA error: " + std::string(cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        drafts[i] = tok = m.h_arg[0];
        h = out;
    }
    return true;
}

bool NemotronHEngine::verify(const int32_t* tokens, int n, int64_t pos0, float* logits_rows, std::string& err) {
    Impl& m = *p_;
    if (n < 1 || n > kMaxVerify || pos0 != m.ctx_end || pos0 + n > m.opt.max_context) {
        err = "nemotron_h: a verify window is 1.." + std::to_string(kMaxVerify) + " tokens at the context's end";
        return false;
    }
    const cpu_moe::PinCaller pin(m.wk.caller_cpu);
    m.cpu_ms = 0;
    // two halves, the first the larger, each one's CPU experts beside the other's GPU layers (window_pair); one
    // window with GUILD_NEMOTRON_DUMP, and GUILD_SPEC_SPLIT=0 is the one-window control (as the Qwen runtime's)
    static const bool split = std::getenv("GUILD_NEMOTRON_DUMP") == nullptr && [] {
        const char* e = std::getenv("GUILD_SPEC_SPLIT");
        return e == nullptr || std::atoi(e) != 0;
    }();
    const int n0 = split && n > 1 ? (n + 1) / 2 : n;
    if (n0 < n) {
        if (!m.window_pair(tokens, nullptr, n0, n - n0, pos0, false, true, err))
            return false;
        cudaMemcpyAsync(m.hn + (size_t) n0 * m.g.n_embd, m.win[1].hn, (size_t) (n - n0) * m.g.n_embd * 4,
                        cudaMemcpyDeviceToDevice, m.st); // the second half's rows after the first's
    } else if (!m.window(tokens, nullptr, n, pos0, false, n > 1, err)) {
        return false;
    }
    if (!m.head_rows(m.hn, n, logits_rows, err))
        return false;
    m.win_tok.assign(tokens, tokens + n);
    m.win_pos = pos0;
    m.win_n = n;
    return true;
}

bool NemotronHEngine::accept(int n_keep, std::string& err) {
    Impl& m = *p_;
    if (m.win_tok.empty() || n_keep < 1 || n_keep > m.win_n) {
        err = "nemotron_h: accept without a verify window";
        return false;
    }
    const cpu_moe::PinCaller pin(m.wk.caller_cpu);
    const int64_t N = m.g.n_embd, C = m.g.ssm_inner + 2 * m.g.ssm_groups * m.g.ssm_state;
    const size_t ssm_floats = (size_t) m.g.ssm_heads * (m.g.ssm_inner / m.g.ssm_heads) * m.g.ssm_state;
    if (n_keep < m.win_n)
        for (Layer& y : m.L)
            if (y.kind == 'M') { // the states after the last kept token
                cudaMemcpyAsync(y.ssm_state, y.ssm_snap + (size_t) (n_keep - 1) * ssm_floats, ssm_floats * 4,
                                cudaMemcpyDeviceToDevice, m.st);
                cudaMemcpyAsync(y.conv_state, y.conv_snap + (size_t) (n_keep - 1) * C * 3, (size_t) C * 3 * 4,
                                cudaMemcpyDeviceToDevice, m.st);
            }
    m.ctx_end = m.win_pos + n_keep;
    if (m.has_mtp) {
        // the MTP block over the kept rows: cell win_pos - 1 (the draft's first step) already pairs the waiting row
        // with the window's first token; rows 0 .. n_keep - 2 pair with the tokens after them, and row n_keep - 1 waits
        const int rows = n_keep - 1;
        cudaMemcpyAsync(m.pending, m.hn + (size_t) (n_keep - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, m.st);
        if (rows > 0) {
            cudaMemcpyAsync(m.mh, m.hn, (size_t) rows * N * 4, cudaMemcpyDeviceToDevice, m.st);
            if (!m.mtp_rows(m.mh, m.win_tok.data() + 1, nullptr, rows, m.win_pos, m.hn, err))
                return false;
        }
        m.has_pending = true;
    }
    m.win_tok.clear();
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = "nemotron_h: CUDA error: " + std::string(cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return true;
}

} // namespace guild::core
