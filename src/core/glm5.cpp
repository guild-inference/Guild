// src/core/glm5.cpp - GLM-5.3-Flash on one GPU and the CPU (see include/guild/core/glm5.hpp).
//
// One forward pass over a window of up to two groups of kMaxT tokens (pipelined: see main_window), layer by layer, in
// the order of llama.cpp's glm5next graph: mHC pre-mix -> attention (KDA or DSA) -> mHC post-mix -> mHC pre-mix -> FFN
// (dense or MoE) -> mHC post-mix. The MoE layer hands the normed activations and the router's choice to the CPU, which
// computes the routed experts while the GPU computes the shared one.
#include "guild/core/glm5.hpp"
#include "guild/core/cpu_moe.hpp"
#include "guild/core/dense_mm.hpp"
#include "guild/core/glm5_kernels.hpp"
#include "guild/core/glm5_nl4.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include "guild/kernels/native_mmvq.hpp"
#if defined(GUILD_GLM_MMQ)
#include "guild/prefill/moe_mmq.hpp"
#endif
#include "guild/kernels/cpu/expert.hpp"
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "guild/kernels/cpu/iq_avx512.hpp"
#include "guild/kernels/cpu/kq_avx2.hpp"
#include "guild/kernels/cpu/numa.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace guild::core {
namespace {

using glm5k::kMaxT;
constexpr int kMaxVerify = 4; ///< tokens of a verify window (logits for each)
namespace kcpu = guild::kernels::cpu;

constexpr int kQ8_0 = 8, kQ6_K = 14, kF32 = 0;

/// A weight matrix in its GGUF form (dense_mm.hpp)
using Mat = GgufMat;

/// The CPU's FFNs, its pool and its launcher: cpu_moe.hpp (shared with the Nemotron-H engine)
using cpu_moe::CpuFfn;
using cpu_moe::Launcher;
using cpu_moe::PinCaller;
using cpu_moe::Pool;

struct Layer {
    bool kda = true, moe = false;
    float *attn_norm = nullptr, *ffn_norm = nullptr;
    Mat hc_attn_fn, hc_ffn_fn;
    float *hc_attn_scale = nullptr, *hc_attn_base = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
    // KDA
    Mat wq, wk, wv, f_a, f_b, g_a, g_b, beta;
    Mat fgb;  ///< f_a, g_a and beta's rows, one after the other in VRAM: one product for a token
    Mat wqkv; ///< q, k and v's rows, one after the other in VRAM: one product for a token
    float *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr, *dt_b = nullptr, *ssm_a = nullptr,
          *ssm_norm = nullptr;
    float *kda_state = nullptr, *conv_state = nullptr;
    float *kda_mid = nullptr, *conv_mid = nullptr; ///< the states after a verify window's first token
    // DSA
    Mat q_a, q_b, kv_a, k_b, v_b;
    Mat qkvi; ///< q_a, kv_a and the indexer's key and gate rows, one after the other: one product for a token
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    void* kv_cache = nullptr; ///< the latent rows of cells [0, kv_near): [cells][kv_lora] FP16
    void* kv_far = nullptr;   ///< the cells' from kv_near on, in mapped host memory (a context past what VRAM holds)
    /// with kv_far: the rows a one-token window last read (glm5k::kv_cache_rows: sel_stride rows, their cells, each
    /// cell's slot), so that the next token copies only the cells it selects anew
    void* rc_rows = nullptr;
    int32_t *rc_slot_cell = nullptr, *rc_cell_slot = nullptr;
    // its indexer
    Mat idx_k, idx_gate, idx_qb;
    float *idx_proj = nullptr, *idx_ln_w = nullptr, *idx_ln_b = nullptr, *idx_ape = nullptr;
    void* pooled = nullptr;                    ///< [cells / kKpool][128] FP16
    void *idx_kc = nullptr, *idx_gc = nullptr; ///< rings of kIdxRing rows (see glm5k::idx_store)
    Mat wo;
    // FFN: the dense one (lead layers) or the shared expert, on the GPU when `ffn_gpu`
    Mat gate, up, down;
    Mat gu; ///< gate and up's rows, one after the other in VRAM: one product for a token
    bool ffn_gpu = false;
    CpuFfn ffn_cpu; ///< on the CPU, or (a MoE layer's shared expert on the GPU) its twin for a pipelined group
    bool ffn_twin = false;
    float lim_ffn = 0.f, lim_exp = 0.f;
    // routing
    float *router = nullptr, *exp_bias = nullptr;
    uint16_t* router16 = nullptr; ///< the router as bf16 when all its F32 values are bf16 values (then router is null)
    // the NextN (MTP) block: [enorm(embedding) | hnorm(hidden)] -> eh_proj, and the norm before the shared head
    Mat eh_proj;
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    const uint8_t *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
    int gate_exps_type = -1, down_exps_type = -1;
    std::vector<CpuFfn> experts; ///< where each routed expert's rows are (the GGUF mapping, or the NUMA halves)
};

const ggml_type_traits_cpu* cpu_traits(int type) {
    return ggml_get_type_traits_cpu((ggml_type) type);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

struct Glm5Engine::Impl {
    std::unique_ptr<GgufModel> gguf;
    Glm5Geometry g;
    Glm5Options opt;
    std::vector<Layer> L;
    Layer mtp; ///< the NextN block (when the GGUF has it and Glm5Options::mtp)
    bool has_mtp = false;
    Mat head, embd;
    float* out_norm = nullptr;
    cudaStream_t st = nullptr;
    cudaEvent_t ev = nullptr, ev2 = nullptr;
    uint64_t vram = 0;
    std::string placement;
    std::unique_ptr<Pool> pool;
    int caller_cpu = -1;    // the core the pool leaves to the calling thread
    int last_head_row = -1; // the row of hn that forward's head read last (last_hidden)
    double cpu_ms = 0.0;
    double enqueue_ms = 0; // GUILD_GLM_TIMING: the pipelined window's GPU launches
    // GUILD_GLM_TIMING: the calling thread's waits - for a group's router and copies before its CPU experts (by the
    // layer's attention, KDA or DSA: the GPU's half was the longer one), and for the launcher's enqueues
    double router_wait_ms[2] = {0, 0}, launcher_wait_ms = 0;
    // device memory: cudaMalloc rounds each allocation up to 2 MiB, which over a thousand weights and vectors of odd
    // sizes (Q6_K rows of 210-byte blocks) wasted ~560 MiB the placement did not count; dalloc carves 256-byte
    // aligned pieces out of the chunk the placement reserves (and, past it, chunks of their own)
    struct Chunk {
        uint8_t* base;
        size_t size, used;
    };
    std::vector<Chunk> chunks;
    cpu_moe::Experts xp;                    // the CPU experts' phases (GUILD_GLM_TIMING: xp.phase_ms)
    uint8_t* snap = nullptr;                // pinned host copy of the recurrent states (snapshot / restore)
    uint8_t* arena[2] = {nullptr, nullptr}; // the experts' NUMA halves
    size_t arena_bytes[2] = {0, 0};
    // token_embd in anonymous memory on the calling thread's node: embed_rows reads a row per token, and read from the
    // GGUF mapping, the first use of a token whose page the page cache had dropped waited on the disk (a draft or
    // verify step took 11-68 ms longer).  A thread copies it while the rest loads (embd_copy_start / _finish).
    uint8_t* embd_copy = nullptr;
    size_t embd_copy_bytes = 0;
    std::thread embd_thread;
    double embd_copy_ms = 0;
    void embd_copy_start();
    void embd_copy_finish();

    // device scratch, sized for kMaxT tokens
    float *R = nullptr, *xn = nullptr, *mix = nullptr, *post = nullptr, *comb = nullptr, *y = nullptr;
    float *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *e = nullptr, *f = nullptr, *qkv = nullptr;
    int64_t a_floats = 0; ///< a token's row of `a`
    float *small1 = nullptr, *small2 = nullptr, *gk = nullptr, *beta = nullptr, *part = nullptr, *logits = nullptr;
    float *rlog = nullptr, *rw = nullptr, *ycpu = nullptr;
    float* hcpart = nullptr; // glm5k::hc_mix's partial sums, for windows of up to kMmvqCols tokens
    int32_t* rids = nullptr;
    float *ikr = nullptr, *igr = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr;
    // MTP: the main model's final hidden rows of the last window, the hidden row waiting for the next token, and the
    // NextN block's own rows
    float *hn = nullptr, *pending = nullptr, *hin = nullptr, *mx = nullptr, *me = nullptr, *mh = nullptr;
    // a kept draft's NextN catch-up, deferred to the next draft (one 2-token pass of the block instead of two of one):
    // the token, its position and the main model's hidden row before it (defer_h); -1: none (GUILD_GLM_ACCEPT_FOLD=0,
    // read at each accept: run at accept)
    int32_t defer_tok = -1;
    int64_t defer_pos = 0;
    float* defer_h = nullptr;
    bool flush_defer(std::string& err) {
        if (defer_tok < 0)
            return true;
        const int32_t t = defer_tok;
        defer_tok = -1;
        return mtp_window(&t, 1, defer_pos, defer_h, false, err);
    }
    int32_t *d_arg = nullptr, *h_arg = nullptr;
    // the draft head's token subset (read_draft_vocab): ascending token ids until they are on the GPU, where the draft
    // maps its argmax to a token id
    std::vector<int32_t> dvocab;
    int32_t* d_dvocab = nullptr;
    int n_dvocab = 0;
    bool dvocab_on = true;        ///< Glm5Engine::use_draft_vocab
    std::vector<int32_t> win_tok; ///< the last verify window, until accept()
    int64_t win_pos = 0;
    // The indexer's key and gate rows: rings of kIdxRing rows a DSA layer, since only the pools a window completes
    // read them (its cells and the kKpool - 1 before it; a verify window rolls back at most one).  A snapshot keeps
    // the cells of its incomplete pool, which a restored context reads again.
    static constexpr int kIdxRing = 512;
    static_assert(kIdxRing % glm5k::kKpool == 0 && kIdxRing >= 2 * kMaxT + glm5k::kKpool, "a window and a pool");
    int64_t ctx_end = 0;  ///< the cells the context holds (forward, accept, restore and reset move it)
    int64_t snap_end = 0; ///< the snapshot's
    size_t snap_tail = 0; ///< where the snapshot keeps its incomplete pool's indexer rows
    int32_t *sel = nullptr, *nsel = nullptr;
    int score_stride = 0, sel_stride = 0;
    // a context past what the VRAM holds: the latent rows of cells from kv_near on are in kv_far_mem (pinned, mapped),
    // and a window's selected rows are gathered into kv_stage kStageQ queries at a time: a token's (stage_sel: q *
    // sel_stride + i, their rows there), or the union of several tokens' (glm5k::kv_gather_union: u_sel their rows,
    // u_mask, u_base and u_list its scratch).  kv_near is the context when it all is in VRAM.
    static constexpr int kStageQ = 8;
    int64_t kv_near = 0;
    void* kv_far_mem = nullptr;
    void* kv_stage = nullptr;
    int32_t *stage_sel = nullptr, *u_sel = nullptr, *u_base = nullptr, *u_list = nullptr;
    uint32_t* u_mask = nullptr;
    // the row caches' scratch (a one-token window's slots and misses; Layer::rc_rows); GUILD_GLM_ROW_CACHE=0 at load:
    // none (the VRAM goes to the context)
    int32_t *rc_sel = nullptr, *rc_misses = nullptr;
    const bool row_cache = [] {
        const char* e = std::getenv("GUILD_GLM_ROW_CACHE");
        return !(e && e[0] == '0');
    }();
    void *xq = nullptr, *xq2 = nullptr;
    DenseProducts dp; ///< the GPU products (dense_mm.hpp): xq / xq2 are its two activation buffers
    // pinned host
    float *h_x = nullptr, *h_y = nullptr, *h_w = nullptr, *h_emb = nullptr;
    int32_t* h_ids = nullptr;

    // A window in two groups of rows (main_window): one group's attention runs on the GPU while the CPU computes the
    // other's experts.  What a group keeps across that overlap - its streams, its FFN half's mix and output, its rows
    // on the host, its event - is taken at its row offset; the rest is scratch a group uses and frees in stream order.
    struct Rows {
        float *R, *post, *comb, *y, *ycpu, *h_x, *h_y, *h_w;
        int32_t* h_ids;
        cudaEvent_t ev;
        bool twin;  ///< the CPU computes the shared expert (its twin) for these rows
        bool dtwin; ///< and a dense layer's FFN, whose other rows the GPU computes
    } rows0{};
    bool twin = false, dtwin = false;
    /// whether these rows' FFN half of layer Y runs on its CPU twin
    static bool on_twin(const Layer& Y, bool tw, bool dtw) {
        return Y.ffn_twin && (Y.moe ? tw : dtw);
    }
    void group_rows(int t0, cudaEvent_t e, bool tw = false, bool dtw = false) {
        const int64_t N = g.n_embd, U = g.n_used;
        R = rows0.R + (size_t) t0 * 4 * N;
        post = rows0.post + (size_t) t0 * 4;
        comb = rows0.comb + (size_t) t0 * 16;
        y = rows0.y + (size_t) t0 * N;
        ycpu = rows0.ycpu + (size_t) t0 * N;
        h_x = rows0.h_x + (size_t) t0 * N;
        h_y = rows0.h_y + (size_t) t0 * N;
        h_w = rows0.h_w + (size_t) t0 * U;
        h_ids = rows0.h_ids + (size_t) t0 * U;
        ev = e;
        twin = tw;
        dtwin = dtw;
    }
    Rows rows() const {
        return {R, post, comb, y, ycpu, h_x, h_y, h_w, h_ids, ev, twin, dtwin};
    }
    std::unique_ptr<Launcher> launcher; // the pipelined window's GPU launches, beside the calling thread's CPU work

    ~Impl() {
        if (embd_thread.joinable()) // a load that failed after embd_copy_start
            embd_thread.join();
        if (embd_copy)
            munmap(embd_copy, embd_copy_bytes);
        if (st)
            cudaStreamSynchronize(st);
        for (const Chunk& c : chunks)
            cudaFree(c.base);
        for (void* p : {(void*) h_x, (void*) h_y, (void*) h_w, (void*) h_emb, (void*) h_ids, (void*) snap,
                        (void*) h_arg, kv_far_mem})
            if (p)
                cudaFreeHost(p);
        pool.reset(); // before the memory its workers might still touch
        for (int n = 0; n < 2; ++n)
            if (arena[n])
                munmap(arena[n], arena_bytes[n]);
        if (ev)
            cudaEventDestroy(ev);
        if (ev2)
            cudaEventDestroy(ev2);
        if (st)
            cudaStreamDestroy(st);
    }

    bool add_chunk(size_t bytes, std::string& err) {
        constexpr size_t kPage = 2ull << 20;
        const size_t size = (bytes + kPage - 1) / kPage * kPage;
        void* p = nullptr;
        if (cudaMalloc(&p, size) != cudaSuccess) {
            cudaGetLastError();
            err = "glm5: out of GPU memory (" + std::to_string(bytes >> 20) + " MiB more)";
            return false;
        }
        chunks.push_back({(uint8_t*) p, size, 0});
        return true;
    }
    template <class T> T* dalloc(size_t n, std::string& err) {
        const size_t bytes = (n * sizeof(T) + 255) & ~(size_t) 255;
        Chunk* c = nullptr;
        for (Chunk& k : chunks)
            if (k.size - k.used >= bytes) {
                c = &k;
                break;
            }
        if (!c) {
            if (!add_chunk(bytes, err))
                return nullptr;
            c = &chunks.back();
        }
        T* p = (T*) (c->base + c->used);
        c->used += bytes;
        vram += bytes;
        return p;
    }

    // ---- tensors ----
    const TensorInfo* find(const std::string& name, size_t* shard = nullptr) const {
        return gguf->find(name, shard);
    }
    const uint8_t* data(const std::string& name, const TensorInfo** ti = nullptr) const {
        size_t s = 0;
        const TensorInfo* t = gguf->find(name, &s);
        if (!t)
            return nullptr;
        if (ti)
            *ti = t;
        return gguf->shard(s).tensor_data(*t);
    }
    bool mat(const std::string& name, Mat& m, std::string& err) {
        const TensorInfo* t = nullptr;
        m.host = data(name, &t);
        if (!m.host) {
            err = "glm5: missing tensor " + name;
            return false;
        }
        m.type = (int) t->type;
        m.n_in = (int64_t) t->shape[0];
        m.n_out = 1;
        for (size_t i = 1; i < t->shape.size(); ++i)
            m.n_out *= (int64_t) t->shape[i];
        m.bytes = m.row_bytes() * (size_t) m.n_out;
        return true;
    }
    /// the per-head Q8_0 matrices (attn_k_b, attn_v_b) in q8_0_split's layout, for glm5k::q8_0_heads
    void split_heads(Mat& m) {
        m.owned.resize(m.bytes);
        glm5k::q8_0_split(m.host, m.owned.data(), m.n_out, (int) m.n_in);
        m.host = m.owned.data();
    }
    /// Q8_0 -> Q6_K on the host (Glm5Options::dense): 6.56 instead of 8.5 bits a weight for the GPU's largest
    /// matrices (Q5_K, Q4_K: 5.5, 4.5).  The rows are dequantized exactly, then quantized as llama-quantize would (no
    /// importance matrix).
    void requant(Mat& m, ggml_type to) {
        if (m.type != kQ8_0 || m.n_in % 256 != 0)
            return;
        const size_t q6_row = ggml_row_size(to, m.n_in), q8_row = m.row_bytes();
        m.owned.resize(q6_row * (size_t) m.n_out);
        const int64_t rows_per = 64;
        const int n_items = (int) ((m.n_out + rows_per - 1) / rows_per);
        auto body = [&](int i) {
            std::vector<float> buf((size_t) rows_per * m.n_in);
            const int64_t r0 = i * rows_per, nr = std::min<int64_t>(rows_per, m.n_out - r0);
            for (int64_t r = 0; r < nr; ++r) {
                const uint8_t* q = m.host + (size_t) (r0 + r) * q8_row;
                float* out = buf.data() + (size_t) r * m.n_in;
                for (int64_t b = 0; b < m.n_in / 32; ++b) {
                    uint16_t hb;
                    std::memcpy(&hb, q + b * 34, 2);
                    const float d = __half2float(*reinterpret_cast<const __half*>(&hb));
                    for (int k = 0; k < 32; ++k)
                        out[b * 32 + k] = d * (float) (int8_t) q[b * 34 + 2 + k];
                }
            }
            ggml_quantize_chunk(to, buf.data(), m.owned.data() + (size_t) r0 * q6_row, 0, nr, m.n_in, nullptr);
        };
        if (pool)
            pool->run(n_items, body);
        else
            for (int i = 0; i < n_items; ++i)
                body(i);
        m.type = to;
        m.host = m.owned.data();
        m.bytes = m.owned.size();
    }
    /// matrices of one type and width that read the same input, uploaded one after the other, and `all` the product of
    /// their rows together (a single launch for a token); each keeps its own view for windows
    bool upload_together(std::initializer_list<Mat*> ms, Mat& all, std::string& err) {
        const Mat& m0 = **ms.begin();
        size_t bytes = 0;
        for (const Mat* m : ms) {
            if (!m->host || m->dev || m->type != m0.type || m->n_in != m0.n_in || m->bytes != m->row_bytes() * m->n_out)
                return true; // not mergeable: each is uploaded on its own
            bytes += m->bytes;
        }
        uint8_t* p = dalloc<uint8_t>(bytes, err);
        if (!p)
            return false;
        all = Mat();
        all.type = m0.type;
        all.n_in = m0.n_in;
        all.dev = p;
        all.bytes = bytes;
        for (Mat* m : ms) {
            cudaMemcpy(p, m->host, m->bytes, cudaMemcpyHostToDevice);
            m->dev = p;
            p += m->bytes;
            all.n_out += m->n_out;
            std::vector<uint8_t>().swap(m->owned);
        }
        return true;
    }
    bool upload(Mat& m, std::string& err) {
        if (m.dev)
            return true;
        void* p = dalloc<uint8_t>(m.bytes, err);
        if (!p)
            return false;
        cudaMemcpy(p, m.host, m.bytes, cudaMemcpyHostToDevice);
        m.dev = p;
        std::vector<uint8_t>().swap(m.owned); // a re-quantized copy lives on the GPU now
        return true;
    }
    float* f32(const std::string& name, int64_t n, std::string& err) {
        const TensorInfo* t = nullptr;
        const uint8_t* h = data(name, &t);
        if (!h) {
            err = "glm5: missing tensor " + name;
            return nullptr;
        }
        if (t->type != kF32 || (int64_t) t->elements() != n) {
            err = "glm5: " + name + " is not " + std::to_string(n) + " F32 values";
            return nullptr;
        }
        float* p = dalloc<float>((size_t) n, err);
        if (p)
            cudaMemcpy(p, h, (size_t) n * 4, cudaMemcpyHostToDevice);
        return p;
    }

    // Whether an F32 tensor of n values holds only bf16 values (the low 16 bits of every one zero), GUILD_GLM_ROUTER16
    // not 0; read once a tensor.
    std::map<std::string, bool> bf16_seen;
    bool all_bf16(const std::string& name, int64_t n) {
        static const bool on = [] {
            const char* e = std::getenv("GUILD_GLM_ROUTER16");
            return !(e && e[0] == '0');
        }();
        if (!on)
            return false;
        const auto it = bf16_seen.find(name);
        if (it != bf16_seen.end())
            return it->second;
        const TensorInfo* t = nullptr;
        const uint8_t* h = data(name, &t);
        bool ok = h && t->type == kF32 && (int64_t) t->elements() == n;
        for (int64_t i = 0; ok && i < n; ++i) {
            uint32_t u;
            std::memcpy(&u, h + 4 * i, 4);
            ok = (u & 0xffffu) == 0;
        }
        bf16_seen[name] = ok;
        return ok;
    }
    // a MoE layer's router: its high halves when all_bf16 (half the bytes, the same numbers), else F32
    bool router_w(Layer& y, const std::string& name, std::string& err) {
        const int64_t n = g.n_expert * g.n_embd;
        if (!all_bf16(name, n))
            return (y.router = f32(name, n, err)) != nullptr;
        const uint8_t* h = data(name, nullptr);
        std::vector<uint16_t> hi((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            uint32_t u;
            std::memcpy(&u, h + 4 * i, 4);
            hi[(size_t) i] = (uint16_t) (u >> 16);
        }
        if (!(y.router16 = dalloc<uint16_t>((size_t) n, err)))
            return false;
        cudaMemcpy(y.router16, hi.data(), (size_t) n * 2, cudaMemcpyHostToDevice);
        return true;
    }
    uint64_t router_bytes(const std::string& name) {
        const int64_t n = g.n_expert * g.n_embd;
        return (uint64_t) n * (all_bf16(name, n) ? 2 : 4);
    }

    // ---- the GPU products (dense_mm.hpp) ----
    static constexpr int kMmvqCols = DenseProducts::kMmvqCols;
    static constexpr int kMaxWindow = 2 * kMaxT; ///< a window's tokens: two pipelined groups of up to kMaxT
    const bool pipe = [] {                       // GUILD_GLM_PIPE=0 (and GUILD_GLM_DEBUG): one group, windows of kMaxT
        const char* e = std::getenv("GUILD_GLM_PIPE");
        return !(e && e[0] == '0') && std::getenv("GUILD_GLM_DEBUG") == nullptr;
    }();
    void quant(const float* x, int64_t n, int T, void* dst) {
        dp.quant(x, n, T, dst);
    }
    void mm(const Mat& m, const void* xq_, float* out, int T) {
        const char* path = dp.mm(m, xq_, out, T);
        debug_check(out, (size_t) T * m.n_out, m, T, path);
    }
    // GUILD_GLM_DEBUG=2: every product's output checked; the first non-finite one is reported
    void debug_check(const float* out, size_t n, const Mat& m, int T, const char* path) {
        static const bool on = [] {
            const char* d = std::getenv("GUILD_GLM_DEBUG");
            return d && d[0] == '2';
        }();
        static bool said = false;
        if (!on || said)
            return;
        std::vector<float> h(n);
        cudaMemcpyAsync(h.data(), out, n * 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        float mx = 0.f;
        size_t bad = 0;
        for (float v : h)
            if (!std::isfinite(v))
                ++bad;
            else
                mx = std::max(mx, std::fabs(v));
        if (bad) {
            said = true;
            std::fprintf(stderr, "glm5 debug: %s product %lld x %lld (type %d, T %d): %zu non-finite, max |y| %.4g\n",
                         path, (long long) m.n_in, (long long) m.n_out, m.type, T, bad, mx);
        }
    }

    bool load(const std::string& path, const Glm5Options& o, std::string& err);
    void attn_kda(Layer& Y, int T, float* mid_state, float* mid_conv);
    void attn_dsa(Layer& Y, int T, int64_t pos0);
    /// the mHC pre-mix of one half layer: the fused hc_mix for a few tokens, the norm + MMQ product for a window
    // (xh: the normed rows also stored to that mapped host memory, for the CPU's FFN)
    void hc_pre_half(const Mat& fn, const float* scale, const float* base, const float* norm_w, int T,
                     float* xh = nullptr) {
        const int64_t N = g.n_embd;
        if (T <= kMmvqCols && fn.type == kQ8_0) {
            glm5k::hc_mix(fn.dev, R, hcpart, (int) N, T, st);
            glm5k::hc_pre_mixed(hcpart, scale, base, R, norm_w, post, comb, xn, (int) N, T, (int) g.sinkhorn_iters,
                                g.hc_eps, g.rms_eps, st, xh);
            return;
        }
        glm5k::rms_norm(R, nullptr, a, (int) (4 * N), T, g.rms_eps, st);
        quant(a, 4 * N, T, xq);
        mm(fn, xq, mix, T);
        glm5k::hc_pre(mix, scale, base, R, norm_w, post, comb, xn, (int) N, T, (int) g.sinkhorn_iters, g.hc_eps,
                      g.rms_eps, st, xh);
    }
    // The router's choice, an FFN's input rows and the CPU's output rows cross through the mapped host rows (h_ids,
    // h_w, h_x, h_y): the kernels store and read them there, instead of copies queued on the stream between kernels
    // (GUILD_GLM_ZC=0, read each window: the copies).  x_on_host: layer_ffn_begin's pre-mix wrote h_x.
    bool zc = true;
    bool x_on_host = false;
    void ffn_begin(const Layer& Y, int T);
    /// y_used (when given): where the FFN's output rows are, io.y or (zero copy) the CPU's mapped io.h_y
    bool ffn_end(const Layer& Y, int T, const Rows& io, std::string& err, const float** y_used = nullptr);
    bool ffn(const Layer& Y, int T, std::string& err) {
        ffn_begin(Y, T);
        return ffn_end(Y, T, rows(), err);
    }
    bool layer_attention(int l, int T, int64_t pos0, bool save_mid, std::string& err);
    // a pipelined window of one token a group that saves the state after its first token: group A's KDA state goes
    // to kda_mid alone and group B starts from there (GUILD_GLM_KDA_MID1=0, read each window: written twice)
    bool kda_mid1 = false;
    // the attention's matrices as Q5_K or Q4_K (Glm5Options::dense): the GPU's half of a layer is the shorter one
    bool narrow_attention = false;
    // a dense layer's FFN on the GPU: the second group of a pipelined window computes it on the CPU (load)
    bool dense_twin = true;
    // the GPU's copies of the shared experts re-quantized at load to this ggml type (0: the GGUF's Q8_0): the
    // attention's type when that is Q5_K or Q4_K; GUILD_GLM_SHEXP=q8_0 / q6_k / q5_k / q4_k picks one
    int shexp_type = 0;
    /// a FFN's bytes on the GPU: a shared expert's as shexp_type re-quantizes them
    uint64_t ffn_gpu_bytes(const Layer& y) const {
        uint64_t b = 0;
        for (const Mat* m : {&y.gate, &y.up, &y.down})
            b += y.moe && shexp_type && m->type == kQ8_0 && m->n_in % 256 == 0
                     ? ggml_row_size((ggml_type) shexp_type, m->n_in) * (uint64_t) m->n_out
                     : m->bytes;
        return b;
    }
    // read each window (switches for A/Bs in one process; the same bits either way): GUILD_GLM_MERGED=0 multiplies a
    // token's q, k and v as three products and its FFN's gate and up as two, GUILD_GLM_Q6K_ROWS=0 takes the
    // block-per-8-rows Q6_K kernel for every shape, GUILD_GLM_ONE_WAVE=0 the launches before glm5k::set_one_wave
    bool merged = true;
    void read_switches() {
        const char* e = std::getenv("GUILD_GLM_MERGED");
        merged = !(e && e[0] == '0');
        e = std::getenv("GUILD_GLM_Q6K_ROWS");
        glm5k::set_q6k_rows(!(e && e[0] == '0'));
        e = std::getenv("GUILD_GLM_ONE_WAVE");
        glm5k::set_one_wave(!(e && e[0] == '0'));
        e = std::getenv("GUILD_GLM_ZC");
        zc = !(e && e[0] == '0');
    }
    bool embed_rows(const int32_t* tokens, int T, std::string& err);
    const char* Y_kind(int l) const;
    bool main_window(const int32_t* tokens, int T, int64_t pos0, bool save_mid, std::string& err);
    bool mtp_window(const int32_t* tokens, int T, int64_t pos0, const float* h_rows, bool draft, std::string& err);
    bool mtp_catch_up(const int32_t* tokens, int T, int64_t pos0, std::string& err);
    bool head_rows(int first, int n, float* host, std::string& err);
    bool read_draft_vocab(std::string& err);
    void snap_idx_tail(bool save);
    void layer_ffn_begin(int l, int T);
    bool layer_ffn_end(int l, int T, const Rows& io, std::string& err);
    bool layer_ffn(int l, int T, std::string& err) {
        layer_ffn_begin(l, T);
        return layer_ffn_end(l, T, rows(), err);
    }
    void cpu_ffn(const Layer& Y, int T, const Rows& io);
};

// ---- loading ----------------------------------------------------------------------------------------------------

bool Glm5Engine::Impl::load(const std::string& path, const Glm5Options& o, std::string& err) {
    opt = o;
    try {
        gguf = std::make_unique<GgufModel>(GgufModel::open(path));
    } catch (const std::exception& ex) {
        err = std::string("glm5: ") + ex.what();
        return false;
    }
    const GgufFile& meta = gguf->meta();
    const MetaValue* arch = meta.get("general.architecture");
    if (!arch || (arch->s != "glm5-next" && arch->s != "glm5next")) {
        err = "glm5: not a glm5-next model";
        return false;
    }
    const std::string A = arch->s + ".";
    auto num = [&](const char* k, double def) {
        const MetaValue* v = meta.get(A + k);
        return v && v->is_num() ? v->num() : def;
    };
    g.n_embd = (int64_t) num("embedding_length", 4096);
    const int64_t blocks = (int64_t) num("block_count", 46);
    const int64_t nextn = (int64_t) num("nextn_predict_layers", 0);
    g.n_layer = blocks - nextn;
    g.n_head = (int64_t) num("attention.head_count", 64);
    g.kda_dim = (int64_t) num("kda.head_dim", 128);
    g.q_lora = (int64_t) num("attention.q_lora_rank", 1536);
    g.kv_lora = (int64_t) num("attention.kv_lora_rank", 512);
    g.qk_dim = (int64_t) num("attention.key_length_mla", 256);
    g.v_dim = (int64_t) num("attention.value_length_mla", 256);
    g.idx_heads = (int64_t) num("attention.indexer.head_count", 32);
    g.idx_dim = (int64_t) num("attention.indexer.key_length", 128);
    g.idx_topk = (int64_t) num("attention.indexer.top_k", 2048);
    g.idx_kpool = (int64_t) num("attention.indexer.kpool", 4);
    g.hc = (int64_t) num("hyper_connection.count", 4);
    g.sinkhorn_iters = (int64_t) num("hyper_connection.sinkhorn_iterations", 20);
    g.hc_eps = (float) num("hyper_connection.epsilon", 1e-6);
    g.n_expert = (int64_t) num("expert_count", 288);
    g.n_used = (int64_t) num("expert_used_count", 8);
    g.n_ff = (int64_t) num("feed_forward_length", 12288);
    g.n_ff_exp = (int64_t) num("expert_feed_forward_length", 2048);
    g.n_ff_shexp = (int64_t) num("expert_shared_feed_forward_length", (double) g.n_ff_exp);
    g.n_dense_lead = (int64_t) num("leading_dense_block_count", 3);
    g.rms_eps = (float) num("attention.layer_norm_rms_epsilon", 1e-5);
    g.ln_eps = (float) num("attention.layer_norm_epsilon", 1e-6);
    g.kda_lower_bound = (float) num("kda.gate_lower_bound", -5.0);
    g.expert_scale = (float) num("expert_weights_scale", 2.5);
    g.expert_norm = num("expert_weights_norm", 1) != 0;
    if (num("expert_gating_func", 2) != 2) {
        err = "glm5: only sigmoid expert gating is implemented";
        return false;
    }
    if (g.hc != glm5k::kHc || g.kda_dim != glm5k::kKdaDim || g.kv_lora > 512 || g.kv_lora % 32 != 0 ||
        g.n_head % 8 != 0 || g.n_expert > glm5k::kMaxExperts || g.n_used > 32 || num("rope.dimension_count", 0) != 0) {
        err = "glm5: unexpected geometry (hc " + std::to_string(g.hc) + ", KDA head " + std::to_string(g.kda_dim) +
              ", kv_lora " + std::to_string(g.kv_lora) + ")";
        return false;
    }
    const MetaValue* hkv = meta.get(A + "attention.head_count_kv");
    const MetaValue* ce = meta.get(A + "swiglu_clamp_exp");
    const MetaValue* cs = meta.get(A + "swiglu_clamp_shexp");
    if (!hkv || hkv->items.size() < (size_t) g.n_layer) {
        err = "glm5: no per-layer attention.head_count_kv";
        return false;
    }
    g.is_kda.assign((size_t) g.n_layer, true);
    g.clamp_exp.assign((size_t) g.n_layer, 0.f);
    g.clamp_shexp.assign((size_t) g.n_layer, 0.f);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        g.is_kda[(size_t) l] = hkv->items[(size_t) l].u == 0;
        if (ce && ce->items.size() > (size_t) l)
            g.clamp_exp[(size_t) l] = (float) ce->items[(size_t) l].num();
        if (cs && cs->items.size() > (size_t) l)
            g.clamp_shexp[(size_t) l] = (float) cs->items[(size_t) l].num();
    }
    for (const char* k : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"})
        if (const MetaValue* v = meta.get(k))
            g.eos_ids.push_back((int32_t) v->u);

    if (g.idx_kpool != glm5k::kKpool || g.idx_dim != glm5k::kIdxDim) {
        err = "glm5: unexpected indexer geometry (kpool " + std::to_string(g.idx_kpool) + ", key " +
              std::to_string(g.idx_dim) + ")";
        return false;
    }
    opt.max_context = (opt.max_context + glm5k::kKpool - 1) / glm5k::kKpool * glm5k::kKpool;

    ggml_cpu_init(); // the i-quant grids the CPU dot products read
    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming) != cudaSuccess) {
        err = "glm5: no CUDA stream";
        return false;
    }

    const int64_t N = g.n_embd, H = g.n_head, D = H * g.kda_dim, T = kMaxT;
    const std::string bp = "blk.";
    L.resize((size_t) g.n_layer);
    if (!mat("token_embd.weight", embd, err) || !mat("output.weight", head, err))
        return false;
    g.n_vocab = embd.n_out; // token_embd is [n_embd][n_vocab]
    if (head.n_out != g.n_vocab) {
        err = "glm5: output.weight has " + std::to_string(head.n_out) + " rows, the vocabulary " +
              std::to_string(g.n_vocab);
        return false;
    }
    if (embd.type != kQ8_0) {
        err = "glm5: token_embd must be Q8_0 (it is type " + std::to_string(embd.type) + ")";
        return false;
    }
    // collect every matrix first, then place
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& y = L[(size_t) l];
        const std::string p = bp + std::to_string(l) + ".";
        y.kda = g.is_kda[(size_t) l];
        y.moe = l >= g.n_dense_lead;
        y.lim_ffn = g.clamp_shexp[(size_t) l];
        y.lim_exp = g.clamp_exp[(size_t) l];
        if (!mat(p + "hc_attn_fn.weight", y.hc_attn_fn, err) || !mat(p + "hc_ffn_fn.weight", y.hc_ffn_fn, err) ||
            !mat(p + "attn_output.weight", y.wo, err))
            return false;
        if (y.kda) {
            if (!mat(p + "attn_q.weight", y.wq, err) || !mat(p + "attn_k.weight", y.wk, err) ||
                !mat(p + "attn_v.weight", y.wv, err) || !mat(p + "ssm_f_a.weight", y.f_a, err) ||
                !mat(p + "ssm_f_b.weight", y.f_b, err) || !mat(p + "ssm_g_a.weight", y.g_a, err) ||
                !mat(p + "ssm_g_b.weight", y.g_b, err) || !mat(p + "ssm_beta.weight", y.beta, err))
                return false;
        } else {
            if (!mat(p + "attn_q_a.weight", y.q_a, err) || !mat(p + "attn_q_b.weight", y.q_b, err) ||
                !mat(p + "attn_kv_a_mqa.weight", y.kv_a, err) || !mat(p + "attn_k_b.weight", y.k_b, err) ||
                !mat(p + "attn_v_b.weight", y.v_b, err))
                return false;
            if (y.k_b.type != kQ8_0 || y.v_b.type != kQ8_0 ||
                !glm5k::q8_0_heads_supported((int) y.k_b.n_in, (int) (y.k_b.n_out / g.n_head)) ||
                !glm5k::q8_0_heads_supported((int) y.v_b.n_in, (int) (y.v_b.n_out / g.n_head))) {
                err = "glm5: attn_k_b / attn_v_b must be Q8_0 with rows of 32 .. 1024 values, a multiple of 32 a head";
                return false;
            }
            split_heads(y.k_b);
            split_heads(y.v_b);
            if (!mat(p + "indexer.attn_k.weight", y.idx_k, err) ||
                !mat(p + "indexer_compressor_gate.weight", y.idx_gate, err) ||
                !mat(p + "indexer.attn_q_b.weight", y.idx_qb, err))
                return false;
        }
        const std::string ffn = y.moe ? "_shexp" : "";
        if (!mat(p + "ffn_gate" + ffn + ".weight", y.gate, err) || !mat(p + "ffn_up" + ffn + ".weight", y.up, err) ||
            !mat(p + "ffn_down" + ffn + ".weight", y.down, err))
            return false;
        if (y.moe) {
            const TensorInfo *tg = nullptr, *tu = nullptr, *td = nullptr;
            y.gate_exps = data(p + "ffn_gate_exps.weight", &tg);
            y.up_exps = data(p + "ffn_up_exps.weight", &tu);
            y.down_exps = data(p + "ffn_down_exps.weight", &td);
            if (!y.gate_exps || !y.up_exps || !y.down_exps || tg->type != tu->type) {
                err = "glm5: layer " + std::to_string(l) + " has no routed experts (or gate/up types differ)";
                return false;
            }
            y.gate_exps_type = (int) tg->type;
            y.down_exps_type = (int) td->type;
            for (int t : {y.gate_exps_type, y.down_exps_type}) {
                const ggml_type_traits_cpu* tr = cpu_traits(t);
                if (!tr || !tr->vec_dot) {
                    err = "glm5: no CPU dot product for expert type " + std::to_string(t);
                    return false;
                }
            }
        }
    }

    // the NextN block (layer n_layer): a DSA layer and a MoE FFN without hyper-connections, behind eh_proj
    if (nextn > 0 && opt.mtp) {
        Layer& y = mtp;
        const std::string p = bp + std::to_string(g.n_layer) + ".";
        y.kda = false;
        y.moe = true;
        if (ce && ce->items.size() > (size_t) g.n_layer)
            y.lim_exp = (float) ce->items[(size_t) g.n_layer].num();
        if (cs && cs->items.size() > (size_t) g.n_layer)
            y.lim_ffn = (float) cs->items[(size_t) g.n_layer].num();
        std::string why;
        const TensorInfo *tg = nullptr, *tu = nullptr, *td = nullptr;
        has_mtp = mat(p + "attn_output.weight", y.wo, why) && mat(p + "attn_q_a.weight", y.q_a, why) &&
                  mat(p + "attn_q_b.weight", y.q_b, why) && mat(p + "attn_kv_a_mqa.weight", y.kv_a, why) &&
                  mat(p + "attn_k_b.weight", y.k_b, why) && mat(p + "attn_v_b.weight", y.v_b, why) &&
                  mat(p + "indexer.attn_k.weight", y.idx_k, why) &&
                  mat(p + "indexer_compressor_gate.weight", y.idx_gate, why) &&
                  mat(p + "indexer.attn_q_b.weight", y.idx_qb, why) && mat(p + "ffn_gate_shexp.weight", y.gate, why) &&
                  mat(p + "ffn_up_shexp.weight", y.up, why) && mat(p + "ffn_down_shexp.weight", y.down, why) &&
                  mat(p + "nextn.eh_proj.weight", y.eh_proj, why) && y.k_b.type == kQ8_0 && y.v_b.type == kQ8_0;
        y.gate_exps = data(p + "ffn_gate_exps.weight", &tg);
        y.up_exps = data(p + "ffn_up_exps.weight", &tu);
        y.down_exps = data(p + "ffn_down_exps.weight", &td);
        has_mtp = has_mtp && y.gate_exps && y.up_exps && y.down_exps && tg->type == tu->type;
        if (has_mtp) {
            split_heads(y.k_b);
            split_heads(y.v_b);
            y.gate_exps_type = (int) tg->type;
            y.down_exps_type = (int) td->type;
            if (!read_draft_vocab(err))
                return false;
        } else {
            std::fprintf(stderr, "guild glm5: no usable NextN block (%s); decoding without MTP\n", why.c_str());
        }
    }

    // the CPU pool: a worker per physical core (the calling thread joins in), grouped by NUMA node, and the launcher
    cpu_moe::Workers wk = cpu_moe::make_workers(opt.numa_split, opt.threads);
    pool = std::move(wk.pool);
    launcher = std::move(wk.launcher);
    // the down phase in items of 64 rows of every expert (cpu_moe::Experts::down_rows): its IQ4_XS rows go through
    // ggml-cpu's one-token dot in a decode step, and longer runs of each expert's rows paid there (2x Xeon Gold 6130,
    // one process alternated six times: CPU experts 71.8-73.1 -> 68.1-70.7 ms a step, the step 85.7-86.5 -> 84.5-85.6
    // ms; a 2,051-token prompt 31.2 -> 30.9 s; the same ids)
    xp.down_rows = 64;
    caller_cpu = wk.caller_cpu;
    const bool split = wk.split;
    const int threads = pool->size();
    embd_copy_start();

    // ---- the VRAM everything but the context and the FFNs takes (the attention weights, the routers, the recurrent
    // states and their rollback copies, the NextN block, the scratch, the head) and what a token of context adds
    // (each DSA layer's latent K/V row (FP16) and a quarter of a pooled key (FP16), and a row of a prompt window's
    // indexer scores: 1088 B a DSA layer, 13,184 B over 11 and the NextN block).  The indexer's key and gate rows are
    // rings of a fixed size (kIdxRing).
    uint64_t per_token = 0;
    const uint64_t dsa_token = (uint64_t) g.kv_lora * 2 + (uint64_t) g.idx_dim * 2 / glm5k::kKpool;
    const uint64_t dsa_fixed = (uint64_t) kIdxRing * g.idx_dim * 2 * 2;
    auto fixed_need = [&] {
        uint64_t state_bytes = 0;
        per_token = (uint64_t) kMaxT * 4 / glm5k::kKpool;
        for (const Layer& y : L) {
            if (y.kda)
                state_bytes += (uint64_t) H * 128 * 128 * 4 + (uint64_t) 3 * D * 3 * 4;
            else {
                per_token += dsa_token;
                state_bytes += dsa_fixed;
            }
        }
        if (has_mtp) {
            // its K/V and indexer rows, its weights, and every KDA layer's rollback state for a verify window
            per_token += dsa_token;
            state_bytes += state_bytes + dsa_fixed; // the mid states equal the states
            for (const Mat* m : {&mtp.wo, &mtp.q_a, &mtp.q_b, &mtp.kv_a, &mtp.k_b, &mtp.v_b, &mtp.idx_k, &mtp.idx_gate,
                                 &mtp.idx_qb, &mtp.eh_proj})
                state_bytes += m->bytes;
            state_bytes += router_bytes(bp + std::to_string(g.n_layer) + ".ffn_gate_inp.weight") + 64 * 1024;
        }
        // a group of kMaxT tokens' scratch (~1.1 MB a token) and a window of kMaxWindow tokens' rows (~64 KB a token)
        const uint64_t scratch_bytes = (176ull << 20) + (uint64_t) (kMaxWindow - kMaxT) * 4 * N * 4 * 2;
        uint64_t need = state_bytes + scratch_bytes + head.bytes + dvocab.size() * 4 +
                        (1ull << 20); // + every piece's 256-byte alignment
        for (const Layer& y : L)
            for (const Mat* m :
                 {&y.hc_attn_fn, &y.hc_ffn_fn, &y.wo, &y.wq, &y.wk, &y.wv, &y.f_a, &y.f_b, &y.g_a, &y.g_b, &y.beta,
                  &y.q_a, &y.q_b, &y.kv_a, &y.k_b, &y.v_b, &y.idx_k, &y.idx_gate, &y.idx_qb})
                need += m->host ? m->bytes : 0;
        for (size_t l = 0; l < L.size(); ++l) // routers and the small vectors
            need += (L[l].moe ? router_bytes(bp + std::to_string(l) + ".ffn_gate_inp.weight") : 0) + 64 * 1024;
        return need;
    };
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    constexpr int64_t kMinContext = 2048;
    const int64_t want = std::max<int64_t>(kMinContext, opt.max_context);
    const int64_t first = std::min<int64_t>(want, 16384);

    // Glm5Dense::Auto: the attention's largest Q8_0 matrices as Q6_K when the GGUF's Q8_0 leave less context than asked
    // for, up to 16k tokens, beside every MoE layer's shared expert: one on the GPU runs beside its layer's CPU experts
    // instead of after them.  An 8 GB card at 32k tokens: Q8_0 attention leaves the 43 shared experts on the CPU, Q6_K
    // holds all of them; 22.8 / 23.0 against 23.9 / 23.6 tok/s (glm5_run --spec, 768 tokens, GTX 1070 + 2x Xeon Gold
    // 6130).  Counting only the context, Q8_0 still fitted once the routers went bf16 (110 MiB less), and kept them
    // all on the CPU.
    {
        // GUILD_GLM_SHEXP: the shared experts' GPU format, whatever --dense says
        const char* e = std::getenv("GUILD_GLM_SHEXP");
        const std::string v = e ? e : "";
        shexp_type = v == "q6_k"                    ? 14
                     : v == "q5_k"                  ? 13
                     : v == "q4_k"                  ? 12
                     : v == "q8_0"                  ? 0
                     : opt.dense == Glm5Dense::Q5_K ? 13
                     : opt.dense == Glm5Dense::Q4_K ? 12
                                                    : 0;
        if (shexp_type)
            std::fprintf(stderr, "guild glm5: the shared experts on the GPU as %s (GUILD_GLM_SHEXP=q8_0 keeps Q8_0)\n",
                         shexp_type == 14   ? "Q6_K"
                         : shexp_type == 13 ? "Q5_K"
                                            : "Q4_K");
    }
    bool q6 = opt.dense == Glm5Dense::Q6_K || opt.dense == Glm5Dense::Q5_K || opt.dense == Glm5Dense::Q4_K;
    if (opt.dense == Glm5Dense::Auto) {
        uint64_t ffn = 0; // the shared experts' bytes (and the NextN block's)
        for (const Layer& y : L)
            ffn += y.moe ? ffn_gpu_bytes(y) : 0;
        if (has_mtp)
            ffn += ffn_gpu_bytes(mtp);
        const uint64_t need = fixed_need() + ffn, budget = free_b > opt.vram_reserve ? free_b - opt.vram_reserve : 0;
        q6 = budget < need || (int64_t) ((budget - need) / per_token) < first;
    }
    if (q6) {
        const ggml_type to = opt.dense == Glm5Dense::Q5_K   ? GGML_TYPE_Q5_K
                             : opt.dense == Glm5Dense::Q4_K ? GGML_TYPE_Q4_K
                                                            : GGML_TYPE_Q6_K;
        narrow_attention = to != GGML_TYPE_Q6_K;
        const auto tq = std::chrono::steady_clock::now();
        uint64_t before = 0, after = 0;
        std::vector<Layer*> all_layers;
        for (Layer& y : L)
            all_layers.push_back(&y);
        if (has_mtp)
            all_layers.push_back(&mtp);
        for (Layer* yp : all_layers)
            for (Mat* m : {&yp->wq, &yp->wk, &yp->wv, &yp->wo, &yp->q_b}) {
                if (!m->host)
                    continue;
                before += m->bytes;
                requant(*m, to);
                after += m->bytes;
            }
        std::fprintf(stderr, "guild glm5: attention matrices Q8_0 -> %s%s: %llu -> %llu MiB in %.1f s\n",
                     to == GGML_TYPE_Q5_K   ? "Q5_K"
                     : to == GGML_TYPE_Q4_K ? "Q4_K"
                                            : "Q6_K",
                     opt.dense == Glm5Dense::Auto ? " (for the context; GUILD_GLM_DENSE=q8_0 keeps Q8_0)" : "",
                     (unsigned long long) (before >> 20), (unsigned long long) (after >> 20), ms_since(tq) / 1000.0);
    }

    // ---- placement: everything but the FFNs must be on the GPU; then the FFNs, layer by layer --------------------
    const uint64_t need = fixed_need();
    if (need + opt.vram_reserve + kMinContext * per_token > free_b) {
        err = "glm5: the attention weights need " + std::to_string((need + kMinContext * per_token) >> 20) +
              " MiB of VRAM with a 2048-token context, " + std::to_string((free_b - opt.vram_reserve) >> 20) +
              " MiB are free";
        return false;
    }
    // up to 32k tokens of context first (GUILD_GLM_CTX_FIRST=<tokens>), then the shared experts while they fit, then
    // the rest of the context asked for.  A shared expert runs on the GPU while the CPU computes its layer's routed
    // experts, so it takes its time off the CPU's (a dense layer's FFN would move its whole time from the CPU to the
    // GPU instead).  A context past what fits beside all of them (an 8 GB card: ~25k tokens) takes the place of
    // the last ones, which their CPU twins then compute for every token (32k tokens: 4 of 43).
    const int64_t ctx_first = [&] {
        const char* e = std::getenv("GUILD_GLM_CTX_FIRST");
        return std::min<int64_t>(want, e && std::atoll(e) > 0 ? std::atoll(e) : 32768);
    }();
    const uint64_t avail = free_b - opt.vram_reserve - need;
    int64_t ctx = std::min<int64_t>(ctx_first, (int64_t) (avail / per_token)) / 256 * 256;
    uint64_t left = avail - (uint64_t) ctx * per_token;
    int ffn_on_gpu = 0;
    std::vector<Layer*> ffn_order;
    for (Layer& y : L)
        if (y.moe)
            ffn_order.push_back(&y);
    if (has_mtp)
        ffn_order.push_back(&mtp);
    for (Layer& y : L)
        if (!y.moe)
            ffn_order.push_back(&y);
    // A dense layer's FFN (layers 0-2) on the GPU for both groups of a pipelined window leaves the CPU nothing to do
    // while the layer's attention and FFN wait on the GPU: the attention as Q5_K at 262,144 tokens, which leaves the
    // room, took a median step of 80.9 / 82.3 ms so, 79.3 / 78.3 with the FFNs on the CPU.  So when it fits it goes to
    // the GPU only as a twin, the second group computing it on the CPU as a shared expert's split does, and only where
    // that split is the default (Q8_0 / Q6_K attention); otherwise it stays on the CPU, which computes every token's
    // as one-token decode does.  GUILD_GLM_DENSE_FFN_GPU=0: on the CPU; =1: on the GPU for both groups.
    const char* dfg = std::getenv("GUILD_GLM_DENSE_FFN_GPU");
    const bool dense_ffn_gpu = dfg && dfg[0] ? dfg[0] == '1' : !narrow_attention;
    dense_twin = !(dfg && dfg[0] == '1');
    for (Layer* yp : ffn_order) {
        Layer& y = *yp;
        const uint64_t b = ffn_gpu_bytes(y);
        if (!y.moe && yp != &mtp && !dense_ffn_gpu)
            continue;
        if (b <= left) {
            y.ffn_gpu = true;
            left -= b;
            ++ffn_on_gpu;
        }
    }
    if (want > ctx) {
        const int64_t more = std::min<int64_t>(want - ctx, (int64_t) (left / per_token)) / 256 * 256;
        ctx += more;
        left -= (uint64_t) more * per_token;
    }
    // A context past that (GUILD_GLM_KV_HOST=0: none) keeps the latent rows of its cells from kv_near on in pinned
    // host memory.  The VRAM keeps every cell's pooled key and score row, and a stage for kStageQ queries' selected
    // rows; the budget of the cells placed so far pays for them, so fewer cells keep their latents in VRAM (an 8 GB
    // card asked for 262,144 tokens: the first 11,776, and the FFNs where they are at 32k).  kv_near is at least
    // min_near: a window the indexer does not select for (its cells all before idx_topk + kKpool) reads VRAM only.
    // GUILD_GLM_KV_NEAR=<cells>: at most that many in VRAM (tests).
    kv_near = ctx;
    uint64_t per_far = 0, stage = 0; // in VRAM, a cell from kv_near on, and the stage
    {
        const char* e = std::getenv("GUILD_GLM_KV_HOST");
        const char* en = std::getenv("GUILD_GLM_KV_NEAR");
        const int64_t min_near = std::max<int64_t>(4096, (g.idx_topk + glm5k::kKpool + 255) / 256 * 256);
        const int64_t near_cap =
            en && std::atoll(en) > 0 ? std::max<int64_t>(min_near, std::atoll(en) / 256 * 256) : INT64_MAX;
        int64_t n_dsa = has_mtp;
        for (const Layer& y : L)
            n_dsa += !y.kda;
        const uint64_t lat = (uint64_t) n_dsa * g.kv_lora * 2;
        per_far = per_token - lat;
        const uint64_t ss = (uint64_t) (g.idx_topk + glm5k::kKpool + 4); // sel_stride
        stage = (uint64_t) kStageQ * ss * (g.kv_lora * 2 + 12) + (uint64_t) glm5k::union_mask_words(want) * 8 + 4096;
        if (row_cache) // each DSA layer's rows, their cells and every cell's slot; the scratch
            stage += (uint64_t) n_dsa * (ss * (g.kv_lora * 2 + 4) + (uint64_t) want * 4 + 1024) + ss * 12 + 1024;
        const uint64_t budget = (uint64_t) ctx * per_token + left;
        if (!(e && e[0] == '0') && (want > ctx || near_cap < ctx) && budget >= min_near * per_token + stage) {
            const int64_t total =
                std::min<int64_t>(want, min_near + (int64_t) ((budget - min_near * per_token - stage) / per_far)) /
                256 * 256;
            const int64_t near =
                std::min<int64_t>({near_cap, total, (int64_t) ((budget - total * per_far - stage) / lat)}) / 256 * 256;
            if (near < total) {
                kv_near = near;
                ctx = total;
                left = budget - total * per_far - near * lat - stage;
                std::fprintf(stderr,
                             "guild glm5: context %lld tokens, the latent rows of cells %lld on in host memory "
                             "(%llu MiB pinned)\n",
                             (long long) ctx, (long long) near, (unsigned long long) ((ctx - near) * lat >> 20));
            }
        }
    }
    if (ctx < opt.max_context)
        std::fprintf(stderr, "guild glm5: context %lld -> %lld tokens (what the VRAM holds next to the weights)\n",
                     (long long) opt.max_context, (long long) ctx);
    {
        int shexp_cpu = 0;
        for (const Layer* yp : ffn_order)
            shexp_cpu += (yp->moe || yp == &mtp) && !yp->ffn_gpu;
        char cells[128];
        if (kv_near < ctx)
            std::snprintf(cells, sizeof cells, "%lld x %llu B + %lld x %llu B + a %llu MiB stage", (long long) kv_near,
                          (unsigned long long) per_token, (long long) (ctx - kv_near), (unsigned long long) per_far,
                          (unsigned long long) (stage >> 20));
        else
            std::snprintf(cells, sizeof cells, "x %llu B", (unsigned long long) per_token);
        std::fprintf(
            stderr,
            "guild glm5: VRAM: %llu MiB free, %llu MiB fixed, context %lld tokens %s, %llu MiB left; %d shared "
            "experts on the CPU\n",
            (unsigned long long) (free_b >> 20), (unsigned long long) (need >> 20), (long long) ctx, cells,
            (unsigned long long) (left >> 20), shexp_cpu);
    }
    opt.max_context = ctx;
    // everything the placement counted, in one chunk
    if (!add_chunk(free_b - opt.vram_reserve - left, err))
        return false;

    // the host part of each DSA layer's latent rows, one after the other
    uint8_t* far_next = nullptr;
    if (kv_near < ctx) {
        size_t bytes = (size_t) has_mtp * (ctx - kv_near) * g.kv_lora * 2;
        for (const Layer& y : L)
            bytes += y.kda ? 0 : (size_t) (ctx - kv_near) * g.kv_lora * 2;
        void* dev = nullptr;
        if (cudaHostAlloc(&kv_far_mem, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer(&dev, kv_far_mem, 0) != cudaSuccess) {
            cudaGetLastError();
            err =
                "glm5: no " + std::to_string(bytes >> 20) + " MiB of pinned host memory for the context's latent rows";
            return false;
        }
        far_next = (uint8_t*) dev;
    }
    const int64_t ss_cells = g.idx_topk + glm5k::kKpool + 4; // sel_stride (set below)
    auto far_rows = [&](Layer& y) {
        if (far_next) {
            y.kv_far = far_next;
            far_next += (size_t) (ctx - kv_near) * g.kv_lora * 2;
            if (row_cache) {
                if (!(y.rc_rows = dalloc<__half>((size_t) ss_cells * g.kv_lora, err)) ||
                    !(y.rc_slot_cell = dalloc<int32_t>((size_t) ss_cells, err)) ||
                    !(y.rc_cell_slot = dalloc<int32_t>((size_t) ctx, err)))
                    return false;
                cudaMemset(y.rc_slot_cell, 0xff, (size_t) ss_cells * 4);
                cudaMemset(y.rc_cell_slot, 0xff, (size_t) ctx * 4);
            }
        }
        return true;
    };
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& y = L[(size_t) l];
        const std::string p = bp + std::to_string(l) + ".";
        if (!upload(y.hc_attn_fn, err) || !upload(y.hc_ffn_fn, err) || !upload(y.wo, err))
            return false;
        if (!(y.attn_norm = f32(p + "attn_norm.weight", N, err)) ||
            !(y.ffn_norm = f32(p + "ffn_norm.weight", N, err)) ||
            !(y.hc_attn_scale = f32(p + "hc_attn_scale.weight", 3, err)) ||
            !(y.hc_attn_base = f32(p + "hc_attn_base.weight", 24, err)) ||
            !(y.hc_ffn_scale = f32(p + "hc_ffn_scale.weight", 3, err)) ||
            !(y.hc_ffn_base = f32(p + "hc_ffn_base.weight", 24, err)))
            return false;
        if (y.kda) {
            if (!upload_together({&y.f_a, &y.g_a, &y.beta}, y.fgb, err))
                return false;
            if (!upload_together({&y.wq, &y.wk, &y.wv}, y.wqkv, err))
                return false;
            for (Mat* m : {&y.wq, &y.wk, &y.wv, &y.f_a, &y.f_b, &y.g_a, &y.g_b, &y.beta})
                if (!upload(*m, err))
                    return false;
            if (!(y.conv_q = f32(p + "ssm_conv1d_q.weight", D * 4, err)) ||
                !(y.conv_k = f32(p + "ssm_conv1d_k.weight", D * 4, err)) ||
                !(y.conv_v = f32(p + "ssm_conv1d_v.weight", D * 4, err)) ||
                !(y.dt_b = f32(p + "ssm_dt.bias", D, err)) || !(y.ssm_a = f32(p + "ssm_a", H, err)) ||
                !(y.ssm_norm = f32(p + "ssm_norm.weight", g.kda_dim, err)))
                return false;
            if (!(y.kda_state = dalloc<float>((size_t) H * 128 * 128, err)) ||
                !(y.conv_state = dalloc<float>((size_t) 3 * D * 3, err)))
                return false;
        } else {
            if (!upload_together({&y.q_a, &y.kv_a, &y.idx_k, &y.idx_gate}, y.qkvi, err))
                return false;
            for (Mat* m : {&y.q_a, &y.q_b, &y.kv_a, &y.k_b, &y.v_b, &y.idx_k, &y.idx_gate, &y.idx_qb})
                if (!upload(*m, err))
                    return false;
            if (!(y.idx_proj = f32(p + "indexer.proj.weight", g.idx_heads * N, err)) ||
                !(y.idx_ln_w = f32(p + "indexer.k_norm.weight", g.idx_dim, err)) ||
                !(y.idx_ln_b = f32(p + "indexer.k_norm.bias", g.idx_dim, err)) ||
                !(y.idx_ape = f32(p + "indexer_compressor_ape.weight", g.idx_dim * glm5k::kKpool, err)))
                return false;
            if (!(y.idx_kc = dalloc<__half>((size_t) kIdxRing * g.idx_dim, err)) ||
                !(y.idx_gc = dalloc<__half>((size_t) kIdxRing * g.idx_dim, err)) ||
                !(y.pooled = dalloc<__half>((size_t) ctx / glm5k::kKpool * g.idx_dim, err)))
                return false;
            if (!(y.q_a_norm = f32(p + "attn_q_a_norm.weight", g.q_lora, err)) ||
                !(y.kv_a_norm = f32(p + "attn_kv_a_norm.weight", g.kv_lora, err)))
                return false;
            if (!(y.kv_cache = dalloc<__half>((size_t) kv_near * g.kv_lora, err)) || !far_rows(y))
                return false;
        }
        if (y.moe) {
            if (!router_w(y, p + "ffn_gate_inp.weight", err) ||
                !(y.exp_bias = f32(p + "exp_probs_b.bias", g.n_expert, err)))
                return false;
        }
        // the CPU's copy: the GGUF's rows (a GPU copy may be re-quantized below)
        const Mat cg = y.gate, cu = y.up, cd = y.down;
        if (y.ffn_gpu) {
            if (y.moe && shexp_type != 0) // the GPU's shared expert re-quantized (shexp_type)
                for (Mat* m : {&y.gate, &y.up, &y.down})
                    requant(*m, (ggml_type) shexp_type);
            if (!upload_together({&y.gate, &y.up}, y.gu, err) || !upload(y.gate, err) || !upload(y.up, err) ||
                !upload(y.down, err))
                return false;
        }
        // a shared expert or a dense layer's FFN on the GPU has a twin on the CPU: one group of a pipelined window
        // computes it there, which evens the GPU's and the CPU's halves of a layer
        {
            y.ffn_twin = y.ffn_gpu;
            CpuFfn& f = y.ffn_cpu;
            f.gu_type = cg.type;
            f.d_type = cd.type;
            f.n_ff = cg.n_out;
            f.gate[0] = f.gate[1] = cg.host;
            f.up[0] = f.up[1] = cu.host;
            f.down[0] = f.down[1] = cd.host;
            f.gu_split = cg.n_out;
            f.d_split = cd.n_out;
            f.gu_row = cg.row_bytes();
            f.d_row = cd.row_bytes();
            f.lim = y.lim_ffn;
        }
    }
    if (has_mtp) {
        Layer& y = mtp;
        const std::string p = bp + std::to_string(g.n_layer) + ".";
        for (Mat* m : {&y.wo, &y.q_a, &y.q_b, &y.kv_a, &y.k_b, &y.v_b, &y.idx_k, &y.idx_gate, &y.idx_qb, &y.eh_proj})
            if (!upload(*m, err))
                return false;
        if (!(y.attn_norm = f32(p + "attn_norm.weight", N, err)) ||
            !(y.ffn_norm = f32(p + "ffn_norm.weight", N, err)) ||
            !(y.q_a_norm = f32(p + "attn_q_a_norm.weight", g.q_lora, err)) ||
            !(y.kv_a_norm = f32(p + "attn_kv_a_norm.weight", g.kv_lora, err)) ||
            !(y.idx_proj = f32(p + "indexer.proj.weight", g.idx_heads * N, err)) ||
            !(y.idx_ln_w = f32(p + "indexer.k_norm.weight", g.idx_dim, err)) ||
            !(y.idx_ln_b = f32(p + "indexer.k_norm.bias", g.idx_dim, err)) ||
            !(y.idx_ape = f32(p + "indexer_compressor_ape.weight", g.idx_dim * glm5k::kKpool, err)) ||
            !router_w(y, p + "ffn_gate_inp.weight", err) ||
            !(y.exp_bias = f32(p + "exp_probs_b.bias", g.n_expert, err)) ||
            !(y.enorm = f32(p + "nextn.enorm.weight", N, err)) || !(y.hnorm = f32(p + "nextn.hnorm.weight", N, err)) ||
            !(y.head_norm = find(p + "nextn.shared_head_norm.weight") ? f32(p + "nextn.shared_head_norm.weight", N, err)
                                                                      : f32("output_norm.weight", N, err)))
            return false;
        if (!(y.kv_cache = dalloc<__half>((size_t) kv_near * g.kv_lora, err)) || !far_rows(y) ||
            !(y.idx_kc = dalloc<__half>((size_t) kIdxRing * g.idx_dim, err)) ||
            !(y.idx_gc = dalloc<__half>((size_t) kIdxRing * g.idx_dim, err)) ||
            !(y.pooled = dalloc<__half>((size_t) ctx / glm5k::kKpool * g.idx_dim, err)))
            return false;
        if (y.ffn_gpu) {
            if (shexp_type != 0)
                for (Mat* m : {&y.gate, &y.up, &y.down})
                    requant(*m, (ggml_type) shexp_type);
            if (!upload_together({&y.gate, &y.up}, y.gu, err) || !upload(y.gate, err) || !upload(y.up, err) ||
                !upload(y.down, err))
                return false;
        } else {
            CpuFfn& f = y.ffn_cpu;
            f.gu_type = y.gate.type;
            f.d_type = y.down.type;
            f.n_ff = y.gate.n_out;
            f.gate[0] = f.gate[1] = y.gate.host;
            f.up[0] = f.up[1] = y.up.host;
            f.down[0] = f.down[1] = y.down.host;
            f.gu_split = y.gate.n_out;
            f.d_split = y.down.n_out;
            f.gu_row = y.gate.row_bytes();
            f.d_row = y.down.row_bytes();
            f.lim = y.lim_ffn;
        }
        for (Layer& l2 : L)
            if (l2.kda && (!(l2.kda_mid = dalloc<float>((size_t) H * 128 * 128, err)) ||
                           !(l2.conv_mid = dalloc<float>((size_t) 3 * D * 3, err))))
                return false;
    }
    if (!upload(head, err) || !(out_norm = f32("output_norm.weight", N, err)))
        return false;
    if (!dvocab.empty()) {
        if (!(d_dvocab = dalloc<int32_t>(dvocab.size(), err)))
            return false;
        cudaMemcpy(d_dvocab, dvocab.data(), dvocab.size() * 4, cudaMemcpyHostToDevice);
        n_dvocab = (int) dvocab.size();
        std::vector<int32_t>().swap(dvocab);
    }

    // scratch
    const int64_t QB = (int64_t) H * g.qk_dim, VB = (int64_t) H * g.v_dim, LB = (int64_t) H * g.kv_lora;
    // each scratch row as wide as its widest use (a: the flat streams, q, the FFN gate; b: k, the latent, the FFN up;
    // c: v, the KDA scan, v_b, the SwiGLU; d: beta, the KDA output; the quantized inputs: any product's input)
    const int64_t wa = std::max<int64_t>({N * 4, D, QB, g.n_ff}), wb = std::max<int64_t>({D, LB, g.n_ff});
    a_floats = wa;
    const int64_t wc = std::max<int64_t>({D, VB, g.n_ff}), wd = std::max<int64_t>(D, H);
    const int64_t widest = std::max<int64_t>({N * 4, D, VB, g.n_ff, 2 * N, g.q_lora});
    // the rows a window keeps across its two pipelined groups (streams, embeddings, mixes, FFN output, final rows)
    // hold kMaxWindow tokens; the scratch a group uses and frees, kMaxT
    const int64_t TW = kMaxWindow;
    if (!(R = dalloc<float>((size_t) TW * 4 * N, err)) || !(xn = dalloc<float>((size_t) T * N, err)) ||
        !(mix = dalloc<float>((size_t) T * 24, err)) || !(post = dalloc<float>((size_t) TW * 4, err)) ||
        !(comb = dalloc<float>((size_t) TW * 16, err)) || !(y = dalloc<float>((size_t) TW * N, err)) ||
        !(a = dalloc<float>((size_t) T * wa, err)) || !(b = dalloc<float>((size_t) T * wb, err)) ||
        !(c = dalloc<float>((size_t) T * wc, err)) || !(d = dalloc<float>((size_t) T * wd, err)) ||
        !(e = dalloc<float>((size_t) T * std::max(LB, VB), err)) || !(f = dalloc<float>((size_t) TW * N, err)) ||
        !(qkv = dalloc<float>((size_t) T * 3 * D, err)) || !(small1 = dalloc<float>((size_t) T * 4096, err)) ||
        !(small2 = dalloc<float>((size_t) T * 4096, err)) || !(gk = dalloc<float>((size_t) T * D, err)) ||
        !(beta = dalloc<float>((size_t) T * H, err)) ||
        !(part = dalloc<float>((size_t) glm5k::attend_scratch_floats(ctx, (int) H, (int) g.kv_lora, T), err)) ||
        !(logits = dalloc<float>((size_t) kMaxVerify * g.n_vocab, err)) ||
        !(rlog = dalloc<float>((size_t) T * g.n_expert, err)) || !(rw = dalloc<float>((size_t) T * g.n_used, err)) ||
        !(rids = dalloc<int32_t>((size_t) T * g.n_used, err)) || !(ycpu = dalloc<float>((size_t) TW * N, err)) ||
        !(hcpart = dalloc<float>(glm5k::hc_mix_floats(kMmvqCols), err)))
        return false;
    if (!(hn = dalloc<float>((size_t) TW * N, err)) || !(pending = dalloc<float>((size_t) N, err)) ||
        !(defer_h = dalloc<float>((size_t) N, err)) || !(hin = dalloc<float>((size_t) T * N, err)) ||
        !(mx = dalloc<float>((size_t) T * N, err)) || !(me = dalloc<float>((size_t) T * N, err)) ||
        !(mh = dalloc<float>((size_t) T * N, err)) || !(d_arg = dalloc<int32_t>(4, err)))
        return false;
    cudaMemsetAsync(pending, 0, (size_t) N * 4, st);
    if (cudaMallocHost(&h_arg, 16) != cudaSuccess) {
        err = "glm5: no pinned host memory";
        return false;
    }
    score_stride = (int) (ctx / glm5k::kKpool + 8);
    sel_stride = (int) (g.idx_topk + glm5k::kKpool + 4);
    if (!(ikr = dalloc<float>((size_t) T * g.idx_dim, err)) || !(igr = dalloc<float>((size_t) T * g.idx_dim, err)) ||
        !(iq = dalloc<float>((size_t) T * g.idx_heads * g.idx_dim, err)) ||
        !(iw = dalloc<float>((size_t) T * g.idx_heads, err)) ||
        !(scores = dalloc<float>((size_t) T * score_stride, err)) ||
        !(sel = dalloc<int32_t>((size_t) T * sel_stride, err)) || !(nsel = dalloc<int32_t>((size_t) T, err)))
        return false;
    if (kv_near < ctx) {
        const size_t words = (size_t) glm5k::union_mask_words(ctx);
        if (!(kv_stage = dalloc<__half>((size_t) kStageQ * sel_stride * g.kv_lora, err)) ||
            !(stage_sel = dalloc<int32_t>((size_t) kStageQ * sel_stride, err)) ||
            !(u_sel = dalloc<int32_t>((size_t) kStageQ * sel_stride, err)) ||
            !(u_list = dalloc<int32_t>((size_t) kStageQ * sel_stride, err)) ||
            !(u_base = dalloc<int32_t>(words + 1, err)) || !(u_mask = dalloc<uint32_t>(words, err)))
            return false;
        cudaMemset(u_mask, 0, words * 4);
        if (row_cache && (!(rc_sel = dalloc<int32_t>((size_t) sel_stride, err)) ||
                          !(rc_misses = dalloc<int32_t>((size_t) 2 * sel_stride + 1, err))))
            return false;
        std::vector<int32_t> rows((size_t) kStageQ * sel_stride);
        for (size_t i = 0; i < rows.size(); ++i)
            rows[i] = (int32_t) i;
        cudaMemcpy(stage_sel, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice);
    }
    // GUILD_GLM_MMQ=0: MMVQ for every window
    if (!dp.init(
            st, widest, T, std::getenv("GUILD_GLM_MMQ") == nullptr,
            [&](size_t bytes) { return (void*) dalloc<uint8_t>(bytes, err); }, err))
        return false;
    xq = dp.xq[0];
    xq2 = dp.xq[1];
    if (dp.mmq_on()) {
        int n_mmq = 0;
        std::vector<Layer*> all_layers;
        for (Layer& y : L)
            all_layers.push_back(&y);
        if (has_mtp)
            all_layers.push_back(&mtp);
        for (Layer* yp : all_layers)
            for (Mat* m : {&yp->hc_attn_fn, &yp->hc_ffn_fn, &yp->wo,     &yp->wq,   &yp->wk,  &yp->wv,   &yp->f_a,
                           &yp->f_b,        &yp->g_a,       &yp->g_b,    &yp->beta, &yp->q_a, &yp->q_b,  &yp->kv_a,
                           &yp->idx_k,      &yp->idx_gate,  &yp->idx_qb, &yp->gate, &yp->up,  &yp->down, &yp->eh_proj})
                n_mmq += dp.mark(*m);
        std::fprintf(stderr, "guild glm5: %d matrices take MMQ in prompt windows\n", n_mmq);
    }
    if (cudaMallocHost(&h_x, (size_t) kMaxWindow * N * 4) != cudaSuccess ||
        cudaMallocHost(&h_y, (size_t) kMaxWindow * N * 4) != cudaSuccess ||
        cudaMallocHost(&h_w, (size_t) kMaxWindow * g.n_used * 4) != cudaSuccess ||
        cudaMallocHost(&h_ids, (size_t) kMaxWindow * g.n_used * 4) != cudaSuccess ||
        cudaMallocHost(&h_emb, (size_t) kMaxWindow * N * 4) != cudaSuccess) {
        err = "glm5: no pinned host memory";
        return false;
    }
    rows0 = {R, post, comb, y, ycpu, h_x, h_y, h_w, h_ids, ev, false};

    // the routed experts: halves of every expert's rows in memory bound to each node, or the GGUF mapping itself
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t expert_bytes = 0;
    std::vector<Layer*> moe_layers;
    for (Layer& y : L)
        if (y.moe)
            moe_layers.push_back(&y);
    if (has_mtp)
        moe_layers.push_back(&mtp);
    for (Layer* yp : moe_layers) {
        Layer& y = *yp;
        const size_t gu_row = ggml_row_size((ggml_type) y.gate_exps_type, N);
        const size_t d_row = ggml_row_size((ggml_type) y.down_exps_type, g.n_ff_exp);
        y.experts.resize((size_t) g.n_expert);
        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
            CpuFfn& f = y.experts[(size_t) ex];
            f.gu_type = y.gate_exps_type;
            f.d_type = y.down_exps_type;
            f.n_ff = g.n_ff_exp;
            f.gu_row = gu_row;
            f.d_row = d_row;
            f.lim = y.lim_exp;
            f.gate[0] = f.gate[1] = y.gate_exps + (size_t) ex * g.n_ff_exp * gu_row;
            f.up[0] = f.up[1] = y.up_exps + (size_t) ex * g.n_ff_exp * gu_row;
            f.down[0] = f.down[1] = y.down_exps + (size_t) ex * N * d_row;
            f.gu_split = g.n_ff_exp;
            f.d_split = N;
            expert_bytes += 2 * g.n_ff_exp * gu_row + N * d_row;
        }
    }
    // IQ3_S gate/up rows repacked as NL4 (glm5_nl4.hpp): the grid lookup done once here, decode 3.9x faster
    // (one Xeon Gold 6130 core: 9.1 GB/s of IQ3_S rows against ggml-cpu's 2.3)
    bool nl4 = false;
    if (opt.nl4 && kcpu::cpu_avx2_ok()) {
        const char* e = std::getenv("GUILD_GLM_NL4");
        nl4 = !(e && e[0] == '0');
    }
    bool any_iq3s = false;
    for (Layer* yp : moe_layers)
        any_iq3s |= yp->gate_exps_type == GGML_TYPE_IQ3_S;
    nl4 = nl4 && any_iq3s;
    if (split || nl4) {
        // node n holds gate/up rows [n h, (n + 1) h) and down rows [n dh, (n + 1) dh) of every routed expert and of
        // the FFNs the CPU computes (shared experts, dense layers) (one node: all of them)
        std::vector<CpuFfn*> all;
        for (Layer* yp : moe_layers)
            for (CpuFfn& f : yp->experts)
                all.push_back(&f);
        std::vector<Layer*> ffn_layers;
        for (Layer& y : L)
            ffn_layers.push_back(&y);
        if (has_mtp)
            ffn_layers.push_back(&mtp);
        for (Layer* yp : ffn_layers)
            if (!yp->ffn_gpu || yp->ffn_twin)
                all.push_back(&yp->ffn_cpu);
        if (!cpu_moe::place_ffns(*pool, all, N, split, nl4, arena, arena_bytes, err)) {
            err = "glm5: " + err;
            return false;
        }
        const size_t node_bytes[2] = {arena_bytes[0], arena_bytes[1]};
        // the routed experts' pages of the shards are no longer read: let the kernel drop them (only theirs: the
        // embedding copy may still be reading token_embd's)
        for (size_t sh = 0; sh < gguf->size(); ++sh) {
            const int fd = ::open(gguf->shard(sh).path().c_str(), O_RDONLY);
            if (fd < 0)
                continue;
            const GgufFile& gf = gguf->shard(sh);
            for (const TensorInfo& t : gf.tensors())
                if (t.name.find("_exps.weight") != std::string::npos)
                    posix_fadvise(fd, (off_t) (gf.data_start() + t.offset), (off_t) tensor_payload_bytes(t),
                                  POSIX_FADV_DONTNEED);
            ::close(fd);
        }
        char second[32] = "";
        if (split)
            std::snprintf(second, sizeof second, " + %.1f", node_bytes[1] / 1073741824.0);
        std::fprintf(stderr, "guild glm5: experts%s%s: %.1f%s GiB in %.1f s\n",
                     split ? " split across 2 NUMA nodes" : "", nl4 ? ", IQ3_S gate/up as NL4" : "",
                     node_bytes[0] / 1073741824.0, second, ms_since(t0) / 1000.0);
    } else if (opt.prefault) {
        // read the routed experts once, so the first tokens do not wait on the disk
        std::vector<std::pair<const uint8_t*, size_t>> spans;
        for (int64_t l = 0; l < g.n_layer + (has_mtp ? 1 : 0); ++l) {
            if (l < g.n_layer && !L[(size_t) l].moe)
                continue;
            for (const char* nm : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const TensorInfo* t = nullptr;
                const uint8_t* p = data(bp + std::to_string(l) + "." + nm, &t);
                if (p)
                    spans.emplace_back(p, (size_t) tensor_payload_bytes(*t));
            }
        }
        std::atomic<uint64_t> sink{0};
        pool->run((int) spans.size(), [&](int i) {
            const uint8_t* p = spans[(size_t) i].first;
            const size_t n = spans[(size_t) i].second;
            madvise((void*) ((uintptr_t) p & ~(uintptr_t) 4095), n + ((uintptr_t) p & 4095), MADV_WILLNEED);
            uint64_t s = 0;
            for (size_t k = 0; k < n; k += 4096)
                s += p[k];
            sink += s;
        });
        std::fprintf(stderr, "guild glm5: experts read into memory: %.1f GiB in %.1f s\n", expert_bytes / 1073741824.0,
                     ms_since(t0) / 1000.0);
    }

    embd_copy_finish();

    char line[256];
    std::snprintf(line, sizeof line,
                  "glm5: %lld layers (%d KDA, %d DSA)%s, %d of %lld FFNs on the GPU, %d CPU threads%s, %llu MiB VRAM, "
                  "context %lld\n",
                  (long long) g.n_layer, (int) std::count(g.is_kda.begin(), g.is_kda.end(), true),
                  (int) std::count(g.is_kda.begin(), g.is_kda.end(), false), has_mtp ? " + MTP" : "", ffn_on_gpu,
                  (long long) g.n_layer + (has_mtp ? 1 : 0), threads, split ? " on 2 NUMA nodes" : "",
                  (unsigned long long) (vram >> 20), (long long) ctx);
    placement = line;
    return true;
}

// ---- the forward pass -------------------------------------------------------------------------------------------

void Glm5Engine::Impl::attn_kda(Layer& Y, int T, float* mid_state, float* mid_conv) {
    const int64_t H = g.n_head, D = H * g.kda_dim;
    quant(xn, g.n_embd, T, xq);
    // a token's q, k and v in one product (their rows are consecutive), into the convolution's output, which it
    // computes in place (a thread reads channel c and then writes it); `merged` off: three products (the same bits)
    const bool one = T == 1 && Y.wqkv.dev && merged;
    if (one) {
        mm(Y.wqkv, xq, qkv, 1);
    } else {
        mm(Y.wq, xq, a, T);
        mm(Y.wk, xq, b, T);
        mm(Y.wv, xq, c, T);
    }
    float *fa = small1, *ga = small2, *bl = d; // f_a, g_a and beta outputs: [T][128], [T][128], [T][H]
    if (T == 1 && Y.fgb.dev) {
        mm(Y.fgb, xq, small1, 1);
        ga = small1 + Y.f_a.n_out;
        bl = ga + Y.g_a.n_out;
    } else {
        mm(Y.f_a, xq, small1, T);
        mm(Y.g_a, xq, small2, T);
        mm(Y.beta, xq, d, T);
    }
    if (one)
        glm5k::kda_conv(qkv, qkv + D, qkv + 2 * D, Y.conv_state, Y.conv_q, Y.conv_k, Y.conv_v, qkv, (int) D, 1,
                        mid_conv, st);
    else
        glm5k::kda_conv(a, b, c, Y.conv_state, Y.conv_q, Y.conv_k, Y.conv_v, qkv, (int) D, T, mid_conv, st);
    quant(fa, Y.f_a.n_out, T, xq2);
    mm(Y.f_b, xq2, a, T); // fpre [T][D]
    glm5k::kda_gates(a, Y.dt_b, Y.ssm_a, gk, bl, beta, (int) D, (int) H, T, g.kda_lower_bound, st);
    quant(ga, Y.g_a.n_out, T, xq2);
    mm(Y.g_b, xq2, b, T); // output gate [T][D]
    // kda_mid1: group A of a 2-token verify window writes the state after its token to kda_mid only (it is the same
    // state), and group B reads it from there
    const bool a_only = kda_mid1 && mid_state && T == 1, b_from = kda_mid1 && !mid_state;
    glm5k::kda_step(b_from ? Y.kda_mid : Y.kda_state, a_only ? nullptr : Y.kda_state, qkv, gk, beta, c, (int) H, T,
                    mid_state, st);
    glm5k::kda_out(c, b, Y.ssm_norm, d, (int) H, T, g.rms_eps, st);
    quant(d, D, T, xq2);
    mm(Y.wo, xq2, y, T);
}

void Glm5Engine::Impl::attn_dsa(Layer& Y, int T, int64_t pos0) {
    const int64_t N = g.n_embd, H = g.n_head, QL = g.q_lora, KL = g.kv_lora;
    quant(xn, N, T, xq);
    float *qa = small1, *kva = small2, *ik = ikr, *ig = igr; // [T][QL], [T][KL], [T][128], [T][128]
    if (T == 1 && Y.qkvi.dev) {
        mm(Y.qkvi, xq, small1, 1);
        kva = qa + QL;
        ik = kva + KL;
        ig = ik + Y.idx_k.n_out;
    } else {
        mm(Y.q_a, xq, small1, T);
        mm(Y.kv_a, xq, small2, T);
        mm(Y.idx_k, xq, ikr, T);
        mm(Y.idx_gate, xq, igr, T);
    }
    glm5k::f32_matvec(Y.idx_proj, xn, iw, (int) N, (int) g.idx_heads, T, st);
    glm5k::rms_norm(qa, Y.q_a_norm, qa, (int) QL, T, g.rms_eps, st);
    glm5k::rms_norm(kva, Y.kv_a_norm, kva, (int) KL, T, g.rms_eps, st);
    quant(qa, QL, T, xq2);
    mm(Y.q_b, xq2, a, T); // q [T][H][qk_dim]
    const int t_near = (int) std::clamp<int64_t>(kv_near - pos0, 0, T);
    if (t_near > 0)
        glm5k::kv_store(kva, Y.kv_cache, pos0, (int) KL, t_near, st);
    if (t_near < T)
        glm5k::kv_store(kva + (size_t) t_near * KL, Y.kv_far, pos0 + t_near - kv_near, (int) KL, T - t_near, st);
    if (Y.rc_cell_slot) // a cached row of these cells is an earlier one's
        glm5k::kv_cache_forget(Y.rc_cell_slot, pos0, T, st);
    // the indexer's key and gate are stored for every cell, dense or not, and pooled as pools complete
    glm5k::idx_store(ik, Y.idx_ln_w, Y.idx_ln_b, ig, Y.idx_kc, Y.idx_gc, pos0, T, kIdxRing, g.ln_eps, st);
    glm5k::idx_pool(Y.idx_kc, Y.idx_gc, kIdxRing, Y.idx_ape, Y.pooled, pos0, T, st);
    glm5k::q8_0_heads(Y.k_b.dev, a, e, (int) H, (int) g.qk_dim, (int) KL, T, st); // absorbed q [T][H][KL]
    const float scale = 1.f / std::sqrt((float) g.qk_dim);
    // GUILD_GLM_ATTEND_LEGACY=1: the slower attention kernel that gives the same bits (read each window, for A/Bs)
    const char* al = std::getenv("GUILD_GLM_ATTEND_LEGACY");
    const bool a_legacy = al && al[0] == '1';
    // GUILD_GLM_ROW_CACHE=0 (read each window too): a token's rows copied whole, the row cache unused (it is kept
    // current by every write all the same)
    const char* rc = std::getenv("GUILD_GLM_ROW_CACHE");
    const bool rc_use = Y.rc_rows && !(rc && rc[0] == '0');
    // a query sees at most idx_topk / kpool pools before the selection is anything but all of them
    const int k_pools = (int) (g.idx_topk / glm5k::kKpool);
    if ((pos0 + T) / glm5k::kKpool > k_pools) {
        // GUILD_GLM_IDX_LEGACY=1: the slower kernels that give the same bits (read each window, for A/Bs in one run)
        const char* lg = std::getenv("GUILD_GLM_IDX_LEGACY");
        const bool legacy = lg && lg[0] == '1';
        mm(Y.idx_qb, xq2, iq, T);
        glm5k::idx_score(iq, iw, Y.pooled, pos0, T, (int) g.idx_heads,
                         1.f / std::sqrt((float) (g.idx_dim * g.idx_heads)), scores, score_stride, legacy, st);
        glm5k::idx_select(scores, score_stride, pos0, T, k_pools, sel, nsel, sel_stride, legacy, st);
        if (pos0 + T <= kv_near)
            glm5k::mla_attend(e, Y.kv_cache, pos0, (int) H, (int) KL, T, scale, sel, nsel, sel_stride, b, part, st, 0,
                              a_legacy);
        else // some rows in host memory: kStageQ queries' rows gathered at a time, each summed as in the whole window
            for (int q0 = 0; q0 < T; q0 += kStageQ) {
                const int nq = std::min(kStageQ, T - q0);
                const int32_t* qsel = sel + (size_t) q0 * sel_stride;
                if (nq == 1 && rc_use) { // a token: only the cells the layer's last token did not select
                    glm5k::kv_cache_rows(Y.kv_cache, kv_near, Y.kv_far, qsel, nsel + q0, sel_stride, Y.rc_rows,
                                         Y.rc_slot_cell, Y.rc_cell_slot, rc_sel, rc_misses, (int) KL, st);
                    glm5k::mla_attend(e + (size_t) q0 * H * KL, Y.rc_rows, pos0 + q0, (int) H, (int) KL, 1, scale,
                                      rc_sel, nsel + q0, sel_stride, b + (size_t) q0 * H * KL, part, st, T, a_legacy);
                    continue;
                }
                if (nq == 1)
                    glm5k::kv_gather(Y.kv_cache, kv_near, Y.kv_far, qsel, nsel + q0, sel_stride, 1, kv_stage, (int) KL,
                                     st);
                else // a prompt window's tokens select mostly the same cells: each row read once
                    glm5k::kv_gather_union(Y.kv_cache, kv_near, Y.kv_far, qsel, nsel + q0, sel_stride, nq, pos0 + T,
                                           u_mask, u_base, u_list, u_sel, kv_stage, kStageQ * sel_stride, (int) KL, st);
                glm5k::mla_attend(e + (size_t) q0 * H * KL, kv_stage, pos0 + q0, (int) H, (int) KL, nq, scale,
                                  nq == 1 ? stage_sel : u_sel, nsel + q0, sel_stride, b + (size_t) q0 * H * KL, part,
                                  st, T, a_legacy);
            }
    } else { // cells [0, pos0 + T) < kv_near
        glm5k::mla_attend(e, Y.kv_cache, pos0, (int) H, (int) KL, T, scale, nullptr, nullptr, 0, b, part, st, 0,
                          a_legacy);
    }
    glm5k::q8_0_heads(Y.v_b.dev, b, c, (int) H, (int) KL, (int) g.v_dim, T, st); // [T][H][v_dim]
    quant(c, H * g.v_dim, T, xq2);
    mm(Y.wo, xq2, y, T);
}

bool Glm5Engine::Impl::layer_attention(int l, int T, int64_t pos0, bool save_mid, std::string& err) {
    Layer& Y = L[(size_t) l];
    const int64_t N = g.n_embd;
    // mHC pre-mix: mixes = hc_fn . rms_norm(R flat), then the stream sum, normed by attn_norm
    hc_pre_half(Y.hc_attn_fn, Y.hc_attn_scale, Y.hc_attn_base, Y.attn_norm, T);
    if (Y.kda)
        attn_kda(Y, T, save_mid ? Y.kda_mid : nullptr, save_mid ? Y.conv_mid : nullptr);
    else
        attn_dsa(Y, T, pos0);
    glm5k::hc_post(y, R, post, comb, (int) N, T, st);
    (void) err;
    return true;
}

void Glm5Engine::Impl::cpu_ffn(const Layer& Y, int T, const Rows& io) {
    // the routed experts of a MoE layer, and the layer's own FFN for every token when it lives on the CPU (or the
    // shared expert's twin for a pipelined group)
    const bool own = !Y.ffn_gpu || on_twin(Y, io.twin, io.dtwin);
    xp.run(*pool, g.n_embd, T, kMaxT, Y.moe ? io.h_ids : nullptr, io.h_w, (int) g.n_used, Y.experts,
           own ? &Y.ffn_cpu : nullptr, io.h_x, io.h_y);
}

void Glm5Engine::Impl::ffn_begin(const Layer& Y, int T) {
    const int64_t N = g.n_embd;
    const bool tw = on_twin(Y, twin, dtwin);
    const bool cpu = Y.moe || !Y.ffn_gpu || tw, gpu_ffn = Y.ffn_gpu && !tw;
    if (Y.moe) {
        if (Y.router16)
            glm5k::bf16_matvec(Y.router16, xn, rlog, (int) N, (int) g.n_expert, T, st);
        else
            glm5k::f32_matvec(Y.router, xn, rlog, (int) N, (int) g.n_expert, T, st);
        glm5k::route(rlog, Y.exp_bias, (int) g.n_expert, (int) g.n_used, g.expert_norm, g.expert_scale, rids, rw, T, st,
                     zc ? h_ids : nullptr, zc ? h_w : nullptr);
        if (!zc) {
            cudaMemcpyAsync(h_ids, rids, (size_t) T * g.n_used * 4, cudaMemcpyDeviceToHost, st);
            cudaMemcpyAsync(h_w, rw, (size_t) T * g.n_used * 4, cudaMemcpyDeviceToHost, st);
        }
    }
    if (cpu) {
        if (!x_on_host)
            cudaMemcpyAsync(h_x, xn, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st);
        cudaEventRecord(ev, st);
    }
    if (gpu_ffn) { // the shared expert or the dense FFN, while the CPU computes the routed experts
        const int64_t F = Y.gate.n_out;
        quant(xn, N, T, xq);
        // a token's gate and up rows in one product, when both fit the row of `a`
        if (T == 1 && Y.gu.dev && merged && 2 * F <= a_floats) {
            mm(Y.gu, xq, a, 1);
            glm5k::swiglu_clamp(a, a + F, c, (int) F, 1, Y.lim_ffn > 0.f ? Y.lim_ffn : 3.0e38f, st);
        } else {
            mm(Y.gate, xq, a, T);
            mm(Y.up, xq, b, T);
            glm5k::swiglu_clamp(a, b, c, (int) F, T, Y.lim_ffn > 0.f ? Y.lim_ffn : 3.0e38f, st);
        }
        quant(c, F, T, xq2);
        mm(Y.down, xq2, y, T);
    }
}

bool Glm5Engine::Impl::ffn_end(const Layer& Y, int T, const Rows& io, std::string& err, const float** y_used) {
    const int64_t N = g.n_embd;
    const bool on_cpu = on_twin(Y, io.twin, io.dtwin);
    if (Y.moe || !Y.ffn_gpu || on_cpu) {
        const auto tw = std::chrono::steady_clock::now();
        const cudaError_t ws = cudaEventSynchronize(io.ev);
        router_wait_ms[Y.kda ? 0 : 1] += ms_since(tw);
        if (ws != cudaSuccess) {
            err = "glm5: CUDA error before the CPU experts: " + std::string(cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        const auto t0 = std::chrono::steady_clock::now();
        cpu_ffn(Y, T, io);
        cpu_ms += ms_since(t0);
        if (Y.ffn_gpu && !on_cpu) {
            if (zc) { // the add reads the CPU's rows in mapped host memory
                glm5k::add(io.y, io.h_y, T * N, st);
            } else {
                cudaMemcpyAsync(io.ycpu, io.h_y, (size_t) T * N * 4, cudaMemcpyHostToDevice, st);
                glm5k::add(io.y, io.ycpu, T * N, st);
            }
        } else if (zc && y_used) {
            *y_used = io.h_y; // the caller's next kernel reads them there
        } else {
            cudaMemcpyAsync(io.y, io.h_y, (size_t) T * N * 4, cudaMemcpyHostToDevice, st);
        }
    }
    return true;
}

void Glm5Engine::Impl::layer_ffn_begin(int l, int T) {
    Layer& Y = L[(size_t) l];
    // the CPU reads the pre-mix's rows: it stores them to h_x too
    x_on_host = zc && (Y.moe || !Y.ffn_gpu || on_twin(Y, twin, dtwin));
    hc_pre_half(Y.hc_ffn_fn, Y.hc_ffn_scale, Y.hc_ffn_base, Y.ffn_norm, T, x_on_host ? h_x : nullptr);
    ffn_begin(Y, T);
    x_on_host = false;
}

bool Glm5Engine::Impl::layer_ffn_end(int l, int T, const Rows& io, std::string& err) {
    const float* y_rows = io.y;
    if (!ffn_end(L[(size_t) l], T, io, err, &y_rows))
        return false;
    glm5k::hc_post(y_rows, io.R, io.post, io.comb, (int) g.n_embd, T, st);
    return true;
}

// ---- the public face --------------------------------------------------------------------------------------------

Glm5Engine::Glm5Engine() : p_(std::make_unique<Impl>()) {
}
Glm5Engine::~Glm5Engine() = default;

bool Glm5Engine::load(const std::string& gguf, const Glm5Options& opt, std::string& err) {
    if (!p_->load(gguf, opt, err))
        return false;
    reset();
    return true;
}

const Glm5Geometry& Glm5Engine::geometry() const {
    return p_->g;
}
int64_t Glm5Engine::max_context() const {
    return p_->opt.max_context;
}
int Glm5Engine::max_window() const {
    return p_->pipe ? Impl::kMaxWindow : kMaxT;
}
uint64_t Glm5Engine::vram_bytes() const {
    return p_->vram;
}
const std::string& Glm5Engine::placement() const {
    return p_->placement;
}
void Glm5Engine::print_timing(const char* what) const {
    if (std::getenv("GUILD_GLM_TIMING"))
        std::fprintf(
            stderr,
            "glm5 timing (%s): activations %.0f ms, gate/up %.0f, hidden %.0f, down %.0f, both in one phase %.0f; "
            "launches %.0f; waits for a group's router %.0f (KDA layers) / %.0f (DSA), for the launcher %.0f; threads "
            "busy %.0f%% / %.0f%% / %.0f%%; %.2f GB of expert rows (gate/up %.2f GB, down %.2f) at %.1f GB/s\n",
            what, p_->xp.phase_ms[0], p_->xp.phase_ms[1], p_->xp.phase_ms[2], p_->xp.phase_ms[3], p_->xp.phase_ms[4],
            p_->enqueue_ms, p_->router_wait_ms[0], p_->router_wait_ms[1], p_->launcher_wait_ms,
            100.0 * p_->xp.busy_ms[1] / std::max(1.0, p_->xp.phase_ms[1] * p_->pool->size()),
            100.0 * p_->xp.busy_ms[3] / std::max(1.0, p_->xp.phase_ms[3] * p_->pool->size()),
            100.0 * p_->xp.busy_ms[4] / std::max(1.0, p_->xp.phase_ms[4] * p_->pool->size()),
            p_->xp.weight_bytes * 1e-9, p_->xp.gu_bytes * 1e-9, p_->xp.down_bytes * 1e-9,
            p_->xp.weight_bytes * 1e-6 / std::max(1e-3, p_->xp.phase_ms[1] + p_->xp.phase_ms[3] + p_->xp.phase_ms[4]));
}

double Glm5Engine::last_cpu_ms() const {
    return p_->cpu_ms;
}

void Glm5Engine::reset() {
    Impl& m = *p_;
    const int64_t D = m.g.n_head * m.g.kda_dim;
    for (Layer& y : m.L) {
        if (y.kda) {
            cudaMemsetAsync(y.kda_state, 0, (size_t) m.g.n_head * 128 * 128 * 4, m.st);
            cudaMemsetAsync(y.conv_state, 0, (size_t) 3 * D * 3 * 4, m.st);
        }
    }
    if (m.pending)
        cudaMemsetAsync(m.pending, 0, (size_t) m.g.n_embd * 4, m.st);
    m.win_tok.clear();
    m.defer_tok = -1;
    m.ctx_end = 0;
    cudaStreamSynchronize(m.st);
}

// the snapshot's incomplete pool: its cells' indexer rows in every DSA layer's rings, to (or from) the host copy
void Glm5Engine::Impl::snap_idx_tail(bool save) {
    const int64_t c0 = snap_end / glm5k::kKpool * glm5k::kKpool;
    const size_t rb = (size_t) (snap_end - c0) * g.idx_dim * 2, off = (size_t) (c0 % kIdxRing) * g.idx_dim * 2;
    if (rb == 0)
        return;
    size_t at = snap_tail;
    std::vector<Layer*> dsa;
    for (Layer& y : L)
        if (!y.kda)
            dsa.push_back(&y);
    if (has_mtp)
        dsa.push_back(&mtp);
    for (Layer* y : dsa)
        for (void* ring : {y->idx_kc, y->idx_gc}) {
            uint8_t* dev = (uint8_t*) ring + off;
            if (save)
                cudaMemcpyAsync(snap + at, dev, rb, cudaMemcpyDeviceToHost, st);
            else
                cudaMemcpyAsync(dev, snap + at, rb, cudaMemcpyHostToDevice, st);
            at += (size_t) (glm5k::kKpool - 1) * g.idx_dim * 2;
        }
}

bool Glm5Engine::snapshot(std::string& err) {
    Impl& m = *p_;
    if (!m.flush_defer(err)) // the block's state as of ctx_end
        return false;
    const size_t sb = (size_t) m.g.n_head * 128 * 128 * 4, cb = (size_t) 3 * m.g.n_head * m.g.kda_dim * 3 * 4;
    const size_t n_kda = (size_t) std::count(m.g.is_kda.begin(), m.g.is_kda.end(), true);
    const size_t pb = (size_t) m.g.n_embd * 4; // the NextN block's pending row
    // and the indexer rows of the incomplete pool (kKpool - 1 cells at most) of every DSA layer
    const size_t n_dsa = m.g.is_kda.size() - n_kda + (m.has_mtp ? 1 : 0);
    const size_t tb = n_dsa * 2 * (glm5k::kKpool - 1) * (size_t) m.g.idx_dim * 2;
    if (!m.snap && cudaMallocHost(&m.snap, n_kda * (sb + cb) + pb + tb) != cudaSuccess) {
        m.snap = nullptr;
        err = "glm5: no pinned memory for the state snapshot";
        return false;
    }
    size_t at = 0;
    for (Layer& y : m.L)
        if (y.kda) {
            cudaMemcpyAsync(m.snap + at, y.kda_state, sb, cudaMemcpyDeviceToHost, m.st);
            cudaMemcpyAsync(m.snap + at + sb, y.conv_state, cb, cudaMemcpyDeviceToHost, m.st);
            at += sb + cb;
        }
    cudaMemcpyAsync(m.snap + at, m.pending, pb, cudaMemcpyDeviceToHost, m.st);
    m.snap_tail = at + pb;
    m.snap_end = m.ctx_end;
    m.snap_idx_tail(true);
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: snapshot: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool Glm5Engine::restore(std::string& err) {
    Impl& m = *p_;
    m.defer_tok = -1; // past the snapshot (snapshot ran any before it)
    if (!m.snap) {
        err = "glm5: restore without a snapshot";
        return false;
    }
    const size_t sb = (size_t) m.g.n_head * 128 * 128 * 4, cb = (size_t) 3 * m.g.n_head * m.g.kda_dim * 3 * 4;
    size_t at = 0;
    for (Layer& y : m.L)
        if (y.kda) {
            cudaMemcpyAsync(y.kda_state, m.snap + at, sb, cudaMemcpyHostToDevice, m.st);
            cudaMemcpyAsync(y.conv_state, m.snap + at + sb, cb, cudaMemcpyHostToDevice, m.st);
            at += sb + cb;
        }
    cudaMemcpyAsync(m.pending, m.snap + at, (size_t) m.g.n_embd * 4, cudaMemcpyHostToDevice, m.st);
    m.snap_idx_tail(false);
    m.ctx_end = m.snap_end;
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: restore: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

void Glm5Engine::Impl::embd_copy_start() {
    // GUILD_GLM_EMBED_COPY=0: embed_rows reads token_embd from the mapping, as before
    if (const char* e = std::getenv("GUILD_GLM_EMBED_COPY"); e && e[0] == '0')
        return;
    size_t shard = 0;
    const TensorInfo* t = gguf->find("token_embd.weight", &shard);
    if (!t || !embd.host || embd.bytes == 0)
        return;
    constexpr size_t kHuge = 2ull << 20;
    const size_t bytes = (embd.bytes + kHuge - 1) / kHuge * kHuge;
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        return; // the mapping still works
    // preferred, not bound: a full node falls back to the other; a policy of its own also keeps the region out of NUMA
    // balancing's sampling faults
    const kcpu::NumaTopology& topo = kcpu::numa_topology();
    if (topo.nodes > 1 && caller_cpu >= 0 && caller_cpu < (int) topo.cpu_node.size() &&
        topo.cpu_node[(size_t) caller_cpu] >= 0) {
        unsigned long mask = 1ul << topo.cpu_node[(size_t) caller_cpu];
        syscall(SYS_mbind, p, bytes, 1 /* MPOL_PREFERRED */, &mask, 64ul, 0u);
    }
    madvise(p, bytes, MADV_HUGEPAGE);
    embd_copy = (uint8_t*) p;
    embd_copy_bytes = bytes;
    const GgufFile& gf = gguf->shard(shard);
    const std::string file = gf.path();
    const off_t off = (off_t) (gf.data_start() + t->offset);
    const size_t n = embd.bytes;
    const uint8_t* mapped = embd.host;
    // pread rather than through the mapping: large sequential reads, and no page of the mapping faulted in
    embd_thread = std::thread([this, file, off, n, mapped] {
        const auto t0 = std::chrono::steady_clock::now();
        size_t done = 0;
        const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd >= 0) {
            posix_fadvise(fd, off, (off_t) n, POSIX_FADV_SEQUENTIAL);
            while (done < n) {
                const ssize_t r = pread(fd, embd_copy + done, std::min<size_t>(n - done, 8u << 20), off + (off_t) done);
                if (r <= 0)
                    break;
                done += (size_t) r;
            }
            ::close(fd);
        }
        if (done < n)
            std::memcpy(embd_copy + done, mapped + done, n - done); // the same bytes, through the mapping
        embd_copy_ms = ms_since(t0);
    });
}

void Glm5Engine::Impl::embd_copy_finish() {
    if (!embd_thread.joinable())
        return;
    const auto t0 = std::chrono::steady_clock::now();
    embd_thread.join();
    embd.host = embd_copy;
    std::fprintf(stderr,
                 "guild glm5: token_embd copied to memory: %.0f MiB in %.2f s (the load waited %.2f s for it)\n",
                 embd.bytes / 1048576.0, embd_copy_ms / 1000.0, ms_since(t0) / 1000.0);
}

bool Glm5Engine::Impl::embed_rows(const int32_t* tokens, int T, std::string& err) {
    // the embedding rows, dequantized on the host into `f`
    const int64_t N = g.n_embd;
    const size_t erow = embd.row_bytes();
    for (int t = 0; t < T; ++t) {
        const int32_t id = tokens[t];
        if (id < 0 || id >= g.n_vocab) {
            err = "glm5: token id " + std::to_string(id) + " is outside the vocabulary";
            return false;
        }
        const uint8_t* row = embd.host + (size_t) id * erow;
        float* out = h_emb + (size_t) t * N;
        for (int64_t blk = 0; blk < N / 32; ++blk) {
            const uint8_t* q = row + blk * 34;
            uint16_t hb;
            std::memcpy(&hb, q, 2);
            const float dsc = __half2float(*reinterpret_cast<const __half*>(&hb));
            for (int k = 0; k < 32; ++k)
                out[blk * 32 + k] = dsc * (float) (int8_t) q[2 + k];
        }
    }
    // h_emb is rewritten only after the stream has passed this copy: every window syncs before it returns
    cudaMemcpyAsync(f, h_emb, (size_t) T * N * 4, cudaMemcpyHostToDevice, st);
    return true;
}

const char* Glm5Engine::Impl::Y_kind(int l) const {
    return L[(size_t) l].kda ? "kda" : "dsa";
}

bool Glm5Engine::Impl::main_window(const int32_t* tokens, int T, int64_t pos0, bool save_mid, std::string& err) {
    const int64_t N = g.n_embd;
    read_switches();
    if (!embed_rows(tokens, T, err))
        return false;
    glm5k::hc_init(f, R, (int) N, T, st);
    // GUILD_GLM_DEBUG=1: the streams' largest magnitude after every half layer, and where a non-finite value appears
    static const bool debug = std::getenv("GUILD_GLM_DEBUG") != nullptr;
    std::vector<float> dbg;
    auto check = [&](int l, const char* half) {
        if (!debug)
            return;
        dbg.resize((size_t) T * 4 * N);
        cudaMemcpyAsync(dbg.data(), R, dbg.size() * 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        float mx = 0.f;
        size_t bad = 0;
        for (float v : dbg) {
            if (!std::isfinite(v))
                ++bad;
            else
                mx = std::max(mx, std::fabs(v));
        }
        std::fprintf(stderr, "glm5 debug: layer %2d %-4s max |R| %.4g%s\n", l, half, mx,
                     bad ? (" NON-FINITE x" + std::to_string(bad)).c_str() : "");
    };
    // Two groups (GUILD_GLM_PIPE=0: one): a layer's attention for group B runs on the GPU while the CPU computes
    // group A's experts, and A's next attention while it computes B's.  B's attention needs A's keys and recurrent
    // state at the same layer, both written before it in stream order; neither group waits for the other's FFN.
    if (pipe && T >= 2) {
        const int TA = T / 2, TB = T - TA;
        const char* m1 = std::getenv("GUILD_GLM_KDA_MID1");
        kda_mid1 = save_mid && TA == 1 && !(m1 && m1[0] == '0');
        // group B's shared experts on the CPU when the attention is Q8_0 or Q6_K (GUILD_GLM_SHEXP_SPLIT=0 / 1: both on
        // the GPU / split, whatever the format): the GPU's half of a layer was the longer one, by about a shared
        // expert.  The CPU twin rounds differently from the GPU's shared expert (Q8_1 activations, MMVQ, the GPU's
        // expf), so a 2-token verify window's second row is not what one-token decode computes and --spec leaves plain
        // greedy decode.  With both on the GPU every row of 128 2-token windows equals greedy decode's logits bit for
        // bit (5-token prompt, 256 tokens; GTX 1070 + 2x Xeon Gold 6130), at 20.8 against 21.5 tok/s (verify 83.7
        // against 79.6 ms a step).  With the attention as Q4_K (--dense q4_k) the GPU has the time: the median step
        // 78.4 -> 74.5 ms with both on the GPU; Q5_K 80.2 -> 79.5 (glm5_run --spec, 768 tokens, --ctx 262144).
        const char* sse = std::getenv("GUILD_GLM_SHEXP_SPLIT"); // read each window
        const bool split_shexp = sse && sse[0] ? sse[0] != '0' : !narrow_attention;
        group_rows(TA, ev2, split_shexp, split_shexp && dense_twin);
        const Rows rB = rows();
        group_rows(0, rows0.ev);
        const Rows rA = rows();
        // a group's attention and the GPU half of its FFN, on the launcher (which owns the row pointers meanwhile);
        // the calling thread runs the other group's CPU experts and enqueues its FFN's end with that group's rows
        bool launch_ok = true;
        std::string launch_err;
        auto launch = [&](int l, bool a) {
            const auto tq = std::chrono::steady_clock::now();
            group_rows(a ? 0 : TA, a ? rA.ev : rB.ev, a ? rA.twin : rB.twin, a ? rA.dtwin : rB.dtwin);
            if (!layer_attention(l, a ? TA : TB, a ? pos0 : pos0 + TA, a && save_mid, launch_err))
                launch_ok = false;
            layer_ffn_begin(l, a ? TA : TB);
            enqueue_ms += ms_since(tq);
        };
        launch(0, true);
        bool ok = true;
        for (int l = 0; l < (int) g.n_layer && ok; ++l) {
            launcher->post([&, l] { launch(l, false); }); // B's attention on the GPU ...
            ok = layer_ffn_end(l, TA, rA, err);           // ... while A's experts run here
            auto tl = std::chrono::steady_clock::now();
            launcher->wait();
            launcher_wait_ms += ms_since(tl);
            if (ok && l + 1 < (int) g.n_layer)
                launcher->post([&, l] { launch(l + 1, true); }); // A's next attention after its FFN
            ok = ok && layer_ffn_end(l, TB, rB, err);
            tl = std::chrono::steady_clock::now();
            launcher->wait();
            launcher_wait_ms += ms_since(tl);
            ok = ok && launch_ok;
        }
        group_rows(0, rows0.ev);
        kda_mid1 = false;
        if (!ok) {
            if (!launch_ok && err.empty())
                err = launch_err;
            return false;
        }
    } else {
        for (int l = 0; l < (int) g.n_layer; ++l) {
            if (!layer_attention(l, T, pos0, save_mid, err))
                return false;
            check(l, Y_kind(l));
            if (!layer_ffn(l, T, err))
                return false;
            check(l, "ffn");
        }
    }
    // the final hidden rows (the head's input, and the NextN block's)
    glm5k::hc_mean_norm(R, out_norm, hn, (int) N, T, g.rms_eps, st);
    return true;
}

bool Glm5Engine::Impl::head_rows(int first, int n, float* host, std::string& err) {
    const int64_t N = g.n_embd;
    quant(hn + (size_t) first * N, N, n, xq);
    mm(head, xq, logits, n);
    cudaMemcpyAsync(host, logits, (size_t) n * g.n_vocab * 4, cudaMemcpyDeviceToHost, st);
    (void) err;
    return true;
}

// The draft head's token subset (tools/draft_vocab.py: int32 token ids): Glm5Options::draft_vocab, or
// GUILD_GLM_DRAFT_VOCAB (a file, or 0 for the whole head).  A draft is the argmax over these rows of the head only;
// verify multiplies the whole head, so the subset changes which drafts are offered and how many are accepted.  With
// both groups' shared experts on the GPU (GUILD_GLM_SHEXP_SPLIT=0, or attention as Q5_K / Q4_K) verify equals greedy
// decode and the output does not change; with Q8_0 or Q6_K attention a window's second row takes its shared expert on
// the CPU, so the output depends on the window shapes and any change in the drafts can change it.  The rows are read
// where the head has them (no copy: VRAM is what limits the context); the ids are kept ascending so a tie goes to the
// lower token id, as the whole head's argmax does.
bool Glm5Engine::Impl::read_draft_vocab(std::string& err) {
    std::string path = opt.draft_vocab;
    if (const char* e = std::getenv("GUILD_GLM_DRAFT_VOCAB"))
        path = std::string(e) == "0" ? std::string() : std::string(e);
    if (path.empty())
        return true;
    if (head.type != kQ6_K || !glm5k::q6k_mmvq_supported((int) head.n_in, (int) head.n_out, 1)) {
        std::fprintf(stderr, "guild glm5: the draft head's token subset needs a Q6_K head; drafting over all of it\n");
        return true;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const std::streamoff size = f ? (std::streamoff) f.tellg() : -1;
    if (size > 0 && size % 4 == 0) {
        dvocab.resize((size_t) size / 4);
        f.seekg(0);
        f.read((char*) dvocab.data(), size);
    }
    if (size <= 0 || size % 4 != 0 || !f) {
        err = "glm5: the draft vocabulary " + path + " is missing or not a list of int32 token ids";
        return false;
    }
    std::sort(dvocab.begin(), dvocab.end());
    dvocab.erase(std::unique(dvocab.begin(), dvocab.end()), dvocab.end());
    if (dvocab.front() < 0 || dvocab.back() >= g.n_vocab) {
        err = "glm5: the draft vocabulary " + path + " has ids outside the vocabulary (" + std::to_string(g.n_vocab) +
              ")";
        return false;
    }
    std::fprintf(stderr, "guild glm5: draft head over %zu of %lld tokens (%s; %.0f of %.0f MiB of rows)\n",
                 dvocab.size(), (long long) g.n_vocab, path.c_str(),
                 (double) dvocab.size() * (double) head.row_bytes() / 1048576.0, (double) head.bytes / 1048576.0);
    return true;
}

bool Glm5Engine::Impl::mtp_window(const int32_t* tokens, int T, int64_t pos0, const float* h_rows, bool draft,
                                  std::string& err) {
    // x = eh_proj [enorm(embed(token)) | hnorm(h)]; x += attn(norm x); x += ffn(norm x); the draft from the shared
    // head over shared_head_norm(x)
    Layer& Y = mtp;
    const int64_t N = g.n_embd;
    if (!embed_rows(tokens, T, err))
        return false;
    glm5k::rms_norm(f, Y.enorm, me, (int) N, T, g.rms_eps, st);
    glm5k::rms_norm(h_rows, Y.hnorm, mh, (int) N, T, g.rms_eps, st);
    float* cat = qkv; // [T][2N] (the KDA scratch, idle here)
    cudaMemcpy2DAsync(cat, 2 * N * 4, me, N * 4, N * 4, T, cudaMemcpyDeviceToDevice, st);
    cudaMemcpy2DAsync(cat + N, 2 * N * 4, mh, N * 4, N * 4, T, cudaMemcpyDeviceToDevice, st);
    quant(cat, 2 * N, T, xq);
    mm(Y.eh_proj, xq, mx, T);
    glm5k::rms_norm(mx, Y.attn_norm, xn, (int) N, T, g.rms_eps, st);
    attn_dsa(Y, T, pos0);
    glm5k::add(mx, y, (int64_t) T * N, st);
    glm5k::rms_norm(mx, Y.ffn_norm, xn, (int) N, T, g.rms_eps, st);
    if (!ffn(Y, T, err))
        return false;
    glm5k::add(mx, y, (int64_t) T * N, st);
    if (draft) {
        glm5k::rms_norm(mx + (size_t) (T - 1) * N, Y.head_norm, xn, (int) N, 1, g.rms_eps, st);
        quant(xn, N, 1, xq);
        if (d_dvocab && dvocab_on) { // the subset's rows, the argmax mapped to its token id on the GPU
            glm5k::q6k_mmvq_rows(head.dev, d_dvocab, n_dvocab, xq, logits, (int) N, st);
            glm5k::argmax(logits, n_dvocab, d_arg, st, d_dvocab);
        } else {
            mm(head, xq, logits, 1);
            glm5k::argmax(logits, (int) g.n_vocab, d_arg, st);
        }
        cudaMemcpyAsync(h_arg, d_arg, 4, cudaMemcpyDeviceToHost, st);
    }
    return true;
}

bool Glm5Engine::Impl::mtp_catch_up(const int32_t* tokens, int T, int64_t pos0, std::string& err) {
    // the NextN block reads (token p, hidden p - 1): the window's rows shifted by one, the pending row first; in
    // pieces of kMaxT (its scratch's size)
    const int64_t N = g.n_embd;
    for (int b = 0; b < T; b += kMaxT) {
        const int n = std::min(kMaxT, T - b);
        cudaMemcpyAsync(hin, b ? hn + (size_t) (b - 1) * N : pending, (size_t) N * 4, cudaMemcpyDeviceToDevice, st);
        if (n > 1)
            cudaMemcpyAsync(hin + N, hn + (size_t) b * N, (size_t) (n - 1) * N * 4, cudaMemcpyDeviceToDevice, st);
        if (!mtp_window(tokens + b, n, pos0 + b, hin, false, err))
            return false;
    }
    cudaMemcpyAsync(pending, hn + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st);
    return true;
}

bool Glm5Engine::forward(const int32_t* tokens, int n, int64_t pos0, float* logits, std::string& err) {
    Impl& m = *p_;
    const PinCaller pin(m.caller_cpu);
    const int64_t N = m.g.n_embd;
    m.cpu_ms = 0.0;
    m.win_tok.clear();
    if (n <= 0)
        return true;
    if (pos0 < 0 || pos0 + n > m.opt.max_context) {
        err = "glm5: the context is full (" + std::to_string(m.opt.max_context) + " tokens)";
        return false;
    }
    // a deferred catch-up before the block reads new rows (dropped when this window rewrites its cell)
    if (m.defer_pos >= pos0)
        m.defer_tok = -1;
    if (!m.flush_defer(err))
        return false;
    const int W = m.pipe ? Impl::kMaxWindow : kMaxT;
    for (int base = 0; base < n; base += W) {
        const int T = std::min(W, n - base);
        if (!m.main_window(tokens + base, T, pos0 + base, false, err))
            return false;
        if (logits && base + T == n && !m.head_rows(T - 1, 1, logits, err))
            return false;
        if (logits && base + T == n)
            m.last_head_row = T - 1;
        if (m.has_mtp && !m.mtp_catch_up(tokens + base, T, pos0 + base, err))
            return false;
        if (cudaStreamSynchronize(m.st) != cudaSuccess) {
            err = std::string("glm5: CUDA error: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    m.ctx_end = pos0 + n;
    (void) N;
    return true;
}

bool Glm5Engine::has_mtp() const {
    return p_->has_mtp;
}

bool Glm5Engine::use_draft_vocab(bool on) {
    p_->dvocab_on = on;
    return p_->d_dvocab != nullptr;
}

bool Glm5Engine::draft(int32_t last, int64_t pos, int32_t& draft, std::string& err) {
    Impl& m = *p_;
    const PinCaller pin(m.caller_cpu);
    if (!m.has_mtp) {
        err = "glm5: no NextN block";
        return false;
    }
    if (pos + 1 >= m.opt.max_context) {
        err = "glm5: the context is full";
        return false;
    }
    if (m.defer_tok >= 0 && m.defer_pos + 1 == pos) { // the kept draft and this token: one 2-token pass
        const int64_t N = m.g.n_embd;
        const int32_t two[2] = {m.defer_tok, last};
        m.defer_tok = -1;
        cudaMemcpyAsync(m.hin, m.defer_h, (size_t) N * 4, cudaMemcpyDeviceToDevice, m.st);
        cudaMemcpyAsync(m.hin + N, m.pending, (size_t) N * 4, cudaMemcpyDeviceToDevice, m.st);
        if (!m.mtp_window(two, 2, pos - 1, m.hin, true, err))
            return false;
    } else {
        if (m.defer_pos >= pos)
            m.defer_tok = -1;
        if (!m.flush_defer(err) || !m.mtp_window(&last, 1, pos, m.pending, true, err))
            return false;
    }
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: CUDA error: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    draft = m.h_arg[0];
    return true;
}

bool Glm5Engine::last_hidden(float* host, std::string& err) {
    Impl& m = *p_;
    if (m.last_head_row < 0) {
        err = "glm5: no head was read";
        return false;
    }
    const size_t N = (size_t) m.g.n_embd;
    if (cudaMemcpy(host, m.hn + (size_t) m.last_head_row * N, N * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        err = "glm5: the hidden state could not be read";
        return false;
    }
    return true;
}

bool Glm5Engine::verify(const int32_t* tokens, int n, int64_t pos0, float* logits_rows, std::string& err) {
    Impl& m = *p_;
    const PinCaller pin(m.caller_cpu);
    m.cpu_ms = 0.0;
    if (n < 1 || n > kMaxVerify || pos0 + n > m.opt.max_context) {
        err = "glm5: a verify window of " + std::to_string(n) + " tokens at " + std::to_string(pos0) + " does not fit";
        return false;
    }
    if (n > 2 || (n > 1 && !m.has_mtp)) { // one rollback point, allocated with the NextN block
        err = "glm5: a verify window of " + std::to_string(n) + " tokens needs the NextN block and at most 2 tokens";
        return false;
    }
    if (!m.main_window(tokens, n, pos0, n > 1, err) || !m.head_rows(0, n, logits_rows, err))
        return false;
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: CUDA error: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    m.win_tok.assign(tokens, tokens + n);
    m.win_pos = pos0;
    return true;
}

bool Glm5Engine::accept(int n_keep, std::string& err) {
    Impl& m = *p_;
    const PinCaller pin(m.caller_cpu);
    const int n = (int) m.win_tok.size();
    if (n == 0 || n_keep < 1 || n_keep > n) {
        err = "glm5: accept without a verify window";
        return false;
    }
    const int64_t N = m.g.n_embd, D = m.g.n_head * m.g.kda_dim;
    if (n_keep < n) { // back to the state after the window's first token
        for (Layer& y : m.L)
            if (y.kda) {
                cudaMemcpyAsync(y.kda_state, y.kda_mid, (size_t) m.g.n_head * 128 * 128 * 4, cudaMemcpyDeviceToDevice,
                                m.st);
                cudaMemcpyAsync(y.conv_state, y.conv_mid, (size_t) 3 * D * 3 * 4, cudaMemcpyDeviceToDevice, m.st);
            }
    }
    if (m.has_mtp) {
        // the draft step already ran the block at win_pos with the pending row; it catches up the accepted rest
        const char* fold = std::getenv("GUILD_GLM_ACCEPT_FOLD");
        if (n_keep == 2 && !(fold && fold[0] == '0')) { // to the next draft (flush_defer runs it if none comes)
            m.defer_tok = m.win_tok[1];
            m.defer_pos = m.win_pos + 1;
            cudaMemcpyAsync(m.defer_h, m.hn, (size_t) N * 4, cudaMemcpyDeviceToDevice, m.st);
        } else if (n_keep > 1) {
            cudaMemcpyAsync(m.hin, m.hn, (size_t) (n_keep - 1) * N * 4, cudaMemcpyDeviceToDevice, m.st);
            if (!m.mtp_window(m.win_tok.data() + 1, n_keep - 1, m.win_pos + 1, m.hin, false, err))
                return false;
        }
        cudaMemcpyAsync(m.pending, m.hn + (size_t) (n_keep - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, m.st);
    }
    m.win_tok.clear();
    m.ctx_end = m.win_pos + n_keep;
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: CUDA error: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

} // namespace guild::core
