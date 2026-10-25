// src/core/glm5.cpp - GLM-5.3-Flash on one GPU and the CPU (see include/guild/core/glm5.hpp).
//
// One forward pass over a window of T <= kMaxT tokens, layer by layer, in the order of llama.cpp's glm5next graph:
// mHC pre-mix -> attention (KDA or DSA) -> mHC post-mix -> mHC pre-mix -> FFN (dense or MoE) -> mHC post-mix.
// The MoE layer hands the normed activations and the router's choice to the CPU, which computes the routed experts
// while the GPU computes the shared one.
#include "guild/core/glm5.hpp"
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

/// A weight matrix in its GGUF form: n_out rows of n_in values
struct Mat {
    int type = -1;
    int64_t n_in = 0, n_out = 0;
    const uint8_t* host = nullptr;
    void* dev = nullptr;
    size_t bytes = 0;
    std::vector<uint8_t> owned; ///< host bytes of a re-quantized copy (until uploaded)
    bool mmq = false;           ///< prompt windows multiply it with llama.cpp's MMQ (int8 tiles) instead of MMVQ
    size_t row_bytes() const {
        return ggml_row_size((ggml_type) type, n_in);
    }
};

/// A SwiGLU FFN on the CPU: gate/up rows of n_embd, down rows of n_ff (one expert's slices, or a whole tensor)
struct CpuFfn {
    int gu_type = -1, d_type = -1;
    int64_t n_ff = 0;
    // Row r of a matrix is at base[node] + r * row bytes, node = r >= split: two NUMA nodes each hold half the rows
    // (node 1's base is offset back by its first row so a row keeps its global index); unsplit, both bases are one
    const uint8_t* gate[2] = {nullptr, nullptr};
    const uint8_t* up[2] = {nullptr, nullptr};
    const uint8_t* down[2] = {nullptr, nullptr};
    int64_t gu_split = 0, d_split = 0; ///< the first gate/up row and the first down row of node 1
    size_t gu_row = 0, d_row = 0;
    float lim = 0.f;
};

/// `p` moved back by `rows` rows of `row` bytes (the base that puts row `rows` at `p`)
inline const uint8_t* rebase(const uint8_t* p, int64_t rows, size_t row) {
    return (const uint8_t*) ((uintptr_t) p - (uintptr_t) rows * row);
}

struct Layer {
    bool kda = true, moe = false;
    float *attn_norm = nullptr, *ffn_norm = nullptr;
    Mat hc_attn_fn, hc_ffn_fn;
    float *hc_attn_scale = nullptr, *hc_attn_base = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
    // KDA
    Mat wq, wk, wv, f_a, f_b, g_a, g_b, beta;
    float *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr, *dt_b = nullptr, *ssm_a = nullptr,
          *ssm_norm = nullptr;
    float *kda_state = nullptr, *conv_state = nullptr;
    float *kda_mid = nullptr, *conv_mid = nullptr; ///< the states after a verify window's first token
    // DSA
    Mat q_a, q_b, kv_a, k_b, v_b;
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    void* kv_cache = nullptr;
    // its indexer
    Mat idx_k, idx_gate, idx_qb;
    float *idx_proj = nullptr, *idx_ln_w = nullptr, *idx_ln_b = nullptr, *idx_ape = nullptr, *pooled = nullptr;
    void *idx_kc = nullptr, *idx_gc = nullptr;
    Mat wo;
    // FFN: the dense one (lead layers) or the shared expert, on the GPU when `ffn_gpu`
    Mat gate, up, down;
    bool ffn_gpu = false;
    CpuFfn ffn_cpu;
    float lim_ffn = 0.f, lim_exp = 0.f;
    // routing
    float *router = nullptr, *exp_bias = nullptr;
    // the NextN (MTP) block: [enorm(embedding) | hnorm(hidden)] -> eh_proj, and the norm before the shared head
    Mat eh_proj;
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
    const uint8_t *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
    int gate_exps_type = -1, down_exps_type = -1;
    std::vector<CpuFfn> experts; ///< where each routed expert's rows are (the GGUF mapping, or the NUMA halves)
};

// ---- the CPU pool: persistent workers, one queue per NUMA node, a parallel_for that the calling thread joins ------

/// Pins the calling thread to `cpu` (the core the pool leaves it) for one call into the engine and gives it back its
/// affinity afterwards.  It runs a share of every phase; unpinned, the scheduler sometimes kept it on a hyper-thread
/// sibling of a worker or on the other node for a whole run (2x Xeon Gold 6130: CPU experts 33 -> 41 ms a token).
class PinCaller {
public:
    explicit PinCaller(int cpu) {
#if defined(__linux__)
        if (cpu < 0 || pthread_getaffinity_np(pthread_self(), sizeof(saved_), &saved_) != 0)
            return;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        pinned_ = pthread_setaffinity_np(pthread_self(), sizeof(one), &one) == 0;
#else
        (void) cpu;
#endif
    }
    ~PinCaller() {
#if defined(__linux__)
        if (pinned_)
            pthread_setaffinity_np(pthread_self(), sizeof(saved_), &saved_);
#endif
    }
    PinCaller(const PinCaller&) = delete;
    PinCaller& operator=(const PinCaller&) = delete;

private:
#if defined(__linux__)
    cpu_set_t saved_;
    bool pinned_ = false;
#endif
};

class Pool {
public:
    /// cpus[n]: the logical CPUs of node n's workers (a worker per CPU)
    explicit Pool(const std::vector<std::vector<int>>& cpus) {
        for (int node = 0; node < (int) cpus.size(); ++node)
            for (int cpu : cpus[(size_t) node])
                workers_.emplace_back([this, node, cpu] { loop(node, cpu); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& w : workers_)
            w.join();
    }
    int size() const {
        return (int) workers_.size() + 1;
    }
    /// fn(i) for i in [0, n), any node
    void run(int n, const std::function<void(int)>& fn) {
        run_nodes(n, 0, [&](int, int i) { fn(i); });
    }
    /// fn(node, i) for i in [0, n0) on node 0's queue and [0, n1) on node 1's; a worker drains its node's queue
    /// first, then helps the other
    void run_nodes(int n0, int n1, const std::function<void(int, int)>& fn) {
        if (n0 + n1 <= 0)
            return;
        fn_ = &fn;
        n_[0] = n0;
        n_[1] = n1;
        next_[0].store(0);
        next_[1].store(0);
        done_.store(0);
        {
            std::lock_guard<std::mutex> lk(mu_);
            ++gen_;
        }
        cv_.notify_all();
        work(0);
        while (done_.load(std::memory_order_acquire) < (int) workers_.size())
            std::this_thread::yield();
    }

private:
#if defined(__linux__)
    static void pin(int cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
    void work(int node) {
        for (int q : {node, 1 - node})
            for (;;) {
                const int i = next_[q].fetch_add(1, std::memory_order_relaxed);
                if (i >= n_[q])
                    break;
                (*fn_)(q, i);
            }
    }
    void loop(int node, int cpu) {
#if defined(__linux__)
        pin(cpu);
#endif
        (void) cpu;
        uint64_t seen = 0;
        for (;;) {
            // spin a while for the next phase (a layer's phases follow each other within microseconds), then sleep
            bool got = false;
            for (int s = 0; s < 200000; ++s) {
                if (gen_.load(std::memory_order_acquire) != seen) {
                    got = true;
                    break;
                }
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
            if (!got) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return gen_.load() != seen; });
            }
            seen = gen_.load(std::memory_order_acquire);
            if (stop_)
                return;
            work(node);
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<uint64_t> gen_{0};
    bool stop_ = false;
    const std::function<void(int, int)>* fn_ = nullptr;
    int n_[2] = {0, 0};
    std::atomic<int> next_[2];
    std::atomic<int> done_{0};
};

const ggml_type_traits_cpu* cpu_traits(int type) {
    return ggml_get_type_traits_cpu((ggml_type) type);
}

/// the activation format a gate/up weight type multiplies (NL4 rows are IQ3_S's: Q8_K)
int act_type(int gu_type) {
    return gu_type == kcpu::kIq3sNl4Type ? (int) GGML_TYPE_Q8_K : (int) cpu_traits(gu_type)->vec_dot_type;
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
    cudaEvent_t ev = nullptr;
    uint64_t vram = 0;
    std::string placement;
    std::unique_ptr<Pool> pool;
    int caller_cpu = -1; // the core the pool leaves to the calling thread
    double cpu_ms = 0.0;
    double phase_ms[4] = {0, 0, 0, 0}; // GUILD_GLM_TIMING: activations, gate/up, hidden, down
    // device memory: cudaMalloc rounds each allocation up to 2 MiB, which over a thousand weights and vectors of odd
    // sizes (Q6_K rows of 210-byte blocks) wasted ~560 MiB the placement did not count; dalloc carves 256-byte
    // aligned pieces out of the chunk the placement reserves (and, past it, chunks of their own)
    struct Chunk {
        uint8_t* base;
        size_t size, used;
    };
    std::vector<Chunk> chunks;
    std::vector<uint8_t> actbuf, hbuf, hqbuf; // the CPU experts' activations
    uint8_t* snap = nullptr;                  // pinned host copy of the recurrent states (snapshot / restore)
    uint8_t* arena[2] = {nullptr, nullptr};   // the experts' NUMA halves
    size_t arena_bytes[2] = {0, 0};

    // device scratch, sized for kMaxT tokens
    float *R = nullptr, *xn = nullptr, *mix = nullptr, *post = nullptr, *comb = nullptr, *y = nullptr;
    float *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *e = nullptr, *f = nullptr, *qkv = nullptr;
    float *small1 = nullptr, *small2 = nullptr, *gk = nullptr, *beta = nullptr, *part = nullptr, *logits = nullptr;
    float *rlog = nullptr, *rw = nullptr, *ycpu = nullptr;
    int32_t* rids = nullptr;
    float *ikr = nullptr, *igr = nullptr, *iq = nullptr, *iw = nullptr, *scores = nullptr;
    // MTP: the main model's final hidden rows of the last window, the hidden row waiting for the next token, and the
    // NextN block's own rows
    float *hn = nullptr, *pending = nullptr, *hin = nullptr, *mx = nullptr, *me = nullptr, *mh = nullptr;
    int32_t *d_arg = nullptr, *h_arg = nullptr;
    std::vector<int32_t> win_tok; ///< the last verify window, until accept()
    int64_t win_pos = 0;
    int32_t *sel = nullptr, *nsel = nullptr;
    int score_stride = 0, sel_stride = 0;
    void *xq = nullptr, *xq2 = nullptr;
    // prompt windows (more than kMmvqCols tokens): the activations in MMQ's layout too, and its launch tables
#if defined(GUILD_GLM_MMQ)
    std::unique_ptr<guild::prefill::mmq::Context> mmq_ctx;
#endif
    void *xqm = nullptr, *xqm2 = nullptr;
    int32_t *mmq_bounds = nullptr, *mmq_ids = nullptr;
    // pinned host
    float *h_x = nullptr, *h_y = nullptr, *h_w = nullptr, *h_emb = nullptr;
    int32_t* h_ids = nullptr;

    ~Impl() {
        if (st)
            cudaStreamSynchronize(st);
        for (const Chunk& c : chunks)
            cudaFree(c.base);
        for (void* p :
             {(void*) h_x, (void*) h_y, (void*) h_w, (void*) h_emb, (void*) h_ids, (void*) snap, (void*) h_arg})
            if (p)
                cudaFreeHost(p);
        pool.reset(); // before the memory its workers might still touch
        for (int n = 0; n < 2; ++n)
            if (arena[n])
                munmap(arena[n], arena_bytes[n]);
        if (ev)
            cudaEventDestroy(ev);
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
    /// Q8_0 -> Q6_K on the host (Glm5Options::dense_q6k): 6.56 instead of 8.5 bits a weight for the GPU's largest
    /// matrices.  The rows are dequantized exactly, then quantized as llama-quantize would (no importance matrix).
    void requant_q6k(Mat& m) {
        if (m.type != kQ8_0 || m.n_in % 256 != 0)
            return;
        const size_t q6_row = ggml_row_size(GGML_TYPE_Q6_K, m.n_in), q8_row = m.row_bytes();
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
            ggml_quantize_chunk(GGML_TYPE_Q6_K, buf.data(), m.owned.data() + (size_t) r0 * q6_row, 0, nr, m.n_in,
                                nullptr);
        };
        if (pool)
            pool->run(n_items, body);
        else
            for (int i = 0; i < n_items; ++i)
                body(i);
        m.type = GGML_TYPE_Q6_K;
        m.host = m.owned.data();
        m.bytes = m.owned.size();
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

    // ---- the GPU products ----
    // MMVQ reads the weights once for up to 8 columns; beyond that each column costs about as much again (GTX 1070:
    // 0.20 ms for one column of a 4096 x 8192 Q8_0 matrix, 0.31 for 8, 2.75 for 64), so wider windows use MMQ
    static constexpr int kMmvqCols = 8;
    void quant(const float* x, int64_t n, int T, void* dst) {
        guild::kernels::native_quantize_q8_1(x, dst, (int) n, T, st);
#if defined(GUILD_GLM_MMQ)
        if (T > kMmvqCols && mmq_ctx)
            guild::prefill::mmq::quantize(x, nullptr, dst == xq ? xqm : xqm2, kQ8_0, n, n, T, st);
#endif
    }
    void mm(const Mat& m, const void* xq_, float* out, int T) {
#if defined(GUILD_GLM_MMQ)
        if (T > kMmvqCols && m.mmq && mmq_ctx && (xq_ == xq || xq_ == xq2)) {
            namespace mmq = guild::prefill::mmq;
            mmq::Product p;
            p.w = m.dev;
            p.type = m.type;
            p.w_rows = m.n_out;
            p.w_cols = m.n_in;
            p.expert_bytes = mmq::matrix_bytes(m.type, m.n_out, m.n_in);
            p.n = 1;
            p.xq = xq_ == xq ? xqm : xqm2;
            p.bounds = mmq_bounds + 2 * T;
            p.ids = mmq_ids;
            p.total_rows = T;
            p.max_rows = T;
            p.dst = out;
            p.ld_dst = m.n_out;
            mmq_ctx->run(p, st);
            debug_check(out, (size_t) T * m.n_out, m, T, "mmq");
            return;
        }
#endif
        // one or two tokens of Q6_K / Q8_0: the staged kernels (bitwise native_mmvq's, fewer load instructions)
        if (m.type == kQ6_K && glm5k::q6k_mmvq_supported((int) m.n_in, T))
            glm5k::q6k_mmvq(m.dev, xq_, out, (int) m.n_in, (int) m.n_out, T, st);
        else if (m.type == kQ8_0 && glm5k::q8_0_mmvq_supported((int) m.n_in, T))
            glm5k::q8_0_mmvq(m.dev, xq_, out, (int) m.n_in, (int) m.n_out, T, st);
        else
            guild::kernels::native_mmvq(m.type, m.dev, xq_, out, (int) m.n_in, (int) m.n_out, T, st);
        debug_check(out, (size_t) T * m.n_out, m, T, "mmvq");
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
    bool ffn(const Layer& Y, int T, std::string& err);
    bool layer_attention(int l, int T, int64_t pos0, bool save_mid, std::string& err);
    bool embed_rows(const int32_t* tokens, int T, std::string& err);
    const char* Y_kind(int l) const;
    bool main_window(const int32_t* tokens, int T, int64_t pos0, bool save_mid, std::string& err);
    bool mtp_window(const int32_t* tokens, int T, int64_t pos0, const float* h_rows, bool draft, std::string& err);
    bool mtp_catch_up(const int32_t* tokens, int T, int64_t pos0, std::string& err);
    bool head_rows(int first, int n, float* host, std::string& err);
    bool layer_ffn(int l, int T, std::string& err);
    void cpu_ffn(const Layer& Y, int T);
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
    if (g.hc != glm5k::kHc || g.kda_dim != glm5k::kKdaDim || g.kv_lora > 512 || g.n_head % 8 != 0 ||
        num("rope.dimension_count", 0) != 0) {
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
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
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
            if (y.k_b.type != kQ8_0 || y.v_b.type != kQ8_0 || !glm5k::q8_0_heads_supported((int) y.k_b.n_in) ||
                !glm5k::q8_0_heads_supported((int) y.v_b.n_in)) {
                err = "glm5: attn_k_b / attn_v_b must be Q8_0 with rows of 64 .. 1024 (a power of two)";
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
        } else {
            std::fprintf(stderr, "guild glm5: no usable NextN block (%s); decoding without MTP\n", why.c_str());
        }
    }

    // the CPU pool: a worker per physical core (the calling thread joins in), grouped by NUMA node
    const kcpu::NumaTopology& topo = kcpu::numa_topology();
    const bool split = topo.nodes == 2 && opt.numa_split && kcpu::numa_split_requested(); // GUILD_EXPERT_NUMA
    std::vector<std::vector<int>> cpus(split ? 2 : 1);
    {
        const int ncpu = (int) std::thread::hardware_concurrency();
        std::vector<int> core_of((size_t) ncpu, -1);
        auto first_sibling = [](int cpu) { // the lowest CPU sharing this one's core
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
            int first = cpu;
            if (f >> first)
                return first;
            return cpu;
        };
        for (int cpu = 0; cpu < ncpu; ++cpu) {
            if (first_sibling(cpu) != cpu)
                continue; // a hyper-thread sibling
            const int node = split && cpu < (int) topo.cpu_node.size() && topo.cpu_node[(size_t) cpu] == 1 ? 1 : 0;
            cpus[(size_t) node].push_back(cpu);
        }
        // the calling thread runs on the first core (PinCaller); a thread budget trims both nodes alike
        if (!cpus[0].empty()) {
            caller_cpu = cpus[0].front();
            cpus[0].erase(cpus[0].begin());
        }
        if (opt.threads > 0)
            for (auto& c : cpus)
                c.resize(std::min(c.size(), (size_t) std::max(1, opt.threads / (int) cpus.size())));
    }
    pool = std::make_unique<Pool>(cpus);
    const int threads = pool->size();

    // GUILD_GLM_DENSE=q6_k (or Glm5Options::dense_q6k): the attention's largest Q8_0 matrices as Q6_K
    if (opt.dense_q6k) {
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
                requant_q6k(*m);
                after += m->bytes;
            }
        std::fprintf(stderr, "guild glm5: attention matrices Q8_0 -> Q6_K: %llu -> %llu MiB in %.1f s\n",
                     (unsigned long long) (before >> 20), (unsigned long long) (after >> 20), ms_since(tq) / 1000.0);
    }

    // ---- placement: everything but the FFNs must be on the GPU; then the FFNs, layer by layer --------------------
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    // per token of context, each DSA layer keeps a latent K/V row (FP16), the indexer's key and gate (FP16) and a
    // quarter of a pooled key (FP32); the context is what the VRAM left after the weights holds
    uint64_t per_token = 0, state_bytes = 0;
    for (const Layer& y : L) {
        if (y.kda)
            state_bytes += (uint64_t) H * 128 * 128 * 4 + (uint64_t) 3 * D * 3 * 4;
        else
            per_token +=
                (uint64_t) g.kv_lora * 2 + (uint64_t) g.idx_dim * 2 * 2 + (uint64_t) g.idx_dim * 4 / glm5k::kKpool;
    }
    if (has_mtp) {
        // its K/V and indexer rows, its weights, and every KDA layer's rollback state for a verify window
        per_token += (uint64_t) g.kv_lora * 2 + (uint64_t) g.idx_dim * 2 * 2 + (uint64_t) g.idx_dim * 4 / glm5k::kKpool;
        state_bytes += state_bytes; // the mid states equal the states
        for (const Mat* m : {&mtp.wo, &mtp.q_a, &mtp.q_b, &mtp.kv_a, &mtp.k_b, &mtp.v_b, &mtp.idx_k, &mtp.idx_gate,
                             &mtp.idx_qb, &mtp.eh_proj})
            state_bytes += m->bytes;
        state_bytes += (uint64_t) g.n_expert * N * 4 + 64 * 1024;
    }
    const uint64_t scratch_bytes = 176ull << 20; // a window of kMaxT tokens: ~1.1 MB of scratch a token
    uint64_t need = state_bytes + scratch_bytes + head.bytes + (1ull << 20); // + every piece's 256-byte alignment
    for (const Layer& y : L)
        for (const Mat* m : {&y.hc_attn_fn, &y.hc_ffn_fn, &y.wo, &y.wq, &y.wk, &y.wv, &y.f_a, &y.f_b, &y.g_a, &y.g_b,
                             &y.beta, &y.q_a, &y.q_b, &y.kv_a, &y.k_b, &y.v_b, &y.idx_k, &y.idx_gate, &y.idx_qb})
            need += m->host ? m->bytes : 0;
    need += (uint64_t) L.size() * (g.n_expert * N * 4 + 64 * 1024); // routers and the small vectors
    constexpr int64_t kMinContext = 2048;
    if (need + opt.vram_reserve + kMinContext * per_token > free_b) {
        err = "glm5: the attention weights need " + std::to_string((need + kMinContext * per_token) >> 20) +
              " MiB of VRAM with a 2048-token context, " + std::to_string((free_b - opt.vram_reserve) >> 20) +
              " MiB are free";
        return false;
    }
    // up to 16k tokens of context first, then the FFNs while they fit, then the rest of the context asked for.  The
    // shared experts go first: one runs on the GPU while the CPU computes its layer's routed experts, so it takes
    // its time off the CPU's (a dense layer's FFN moves its whole time from the CPU to the GPU instead).

    const int64_t want = std::max<int64_t>(kMinContext, opt.max_context);
    const int64_t first = std::min<int64_t>(want, 16384);
    const uint64_t avail = free_b - opt.vram_reserve - need;
    int64_t ctx = std::min<int64_t>(first, (int64_t) (avail / per_token)) / 256 * 256;
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
    for (Layer* yp : ffn_order) {
        Layer& y = *yp;
        const uint64_t b = y.gate.bytes + y.up.bytes + y.down.bytes;
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
    if (ctx < opt.max_context)
        std::fprintf(stderr, "guild glm5: context %lld -> %lld tokens (what the VRAM holds next to the weights)\n",
                     (long long) opt.max_context, (long long) ctx);
    opt.max_context = ctx;
    // everything the placement counted, in one chunk
    if (!add_chunk(free_b - opt.vram_reserve - left, err))
        return false;

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
            for (Mat* m : {&y.q_a, &y.q_b, &y.kv_a, &y.k_b, &y.v_b, &y.idx_k, &y.idx_gate, &y.idx_qb})
                if (!upload(*m, err))
                    return false;
            if (!(y.idx_proj = f32(p + "indexer.proj.weight", g.idx_heads * N, err)) ||
                !(y.idx_ln_w = f32(p + "indexer.k_norm.weight", g.idx_dim, err)) ||
                !(y.idx_ln_b = f32(p + "indexer.k_norm.bias", g.idx_dim, err)) ||
                !(y.idx_ape = f32(p + "indexer_compressor_ape.weight", g.idx_dim * glm5k::kKpool, err)))
                return false;
            if (!(y.idx_kc = dalloc<__half>((size_t) ctx * g.idx_dim, err)) ||
                !(y.idx_gc = dalloc<__half>((size_t) ctx * g.idx_dim, err)) ||
                !(y.pooled = dalloc<float>((size_t) ctx / glm5k::kKpool * g.idx_dim, err)))
                return false;
            if (!(y.q_a_norm = f32(p + "attn_q_a_norm.weight", g.q_lora, err)) ||
                !(y.kv_a_norm = f32(p + "attn_kv_a_norm.weight", g.kv_lora, err)))
                return false;
            if (!(y.kv_cache = dalloc<__half>((size_t) ctx * g.kv_lora, err)))
                return false;
        }
        if (y.moe) {
            if (!(y.router = f32(p + "ffn_gate_inp.weight", g.n_expert * N, err)) ||
                !(y.exp_bias = f32(p + "exp_probs_b.bias", g.n_expert, err)))
                return false;
        }
        if (y.ffn_gpu) {
            if (!upload(y.gate, err) || !upload(y.up, err) || !upload(y.down, err))
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
            !(y.router = f32(p + "ffn_gate_inp.weight", g.n_expert * N, err)) ||
            !(y.exp_bias = f32(p + "exp_probs_b.bias", g.n_expert, err)) ||
            !(y.enorm = f32(p + "nextn.enorm.weight", N, err)) || !(y.hnorm = f32(p + "nextn.hnorm.weight", N, err)) ||
            !(y.head_norm = find(p + "nextn.shared_head_norm.weight") ? f32(p + "nextn.shared_head_norm.weight", N, err)
                                                                      : f32("output_norm.weight", N, err)))
            return false;
        if (!(y.kv_cache = dalloc<__half>((size_t) ctx * g.kv_lora, err)) ||
            !(y.idx_kc = dalloc<__half>((size_t) ctx * g.idx_dim, err)) ||
            !(y.idx_gc = dalloc<__half>((size_t) ctx * g.idx_dim, err)) ||
            !(y.pooled = dalloc<float>((size_t) ctx / glm5k::kKpool * g.idx_dim, err)))
            return false;
        if (y.ffn_gpu) {
            if (!upload(y.gate, err) || !upload(y.up, err) || !upload(y.down, err))
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

    // scratch
    const int64_t QB = (int64_t) H * g.qk_dim, VB = (int64_t) H * g.v_dim, LB = (int64_t) H * g.kv_lora;
    // each scratch row as wide as its widest use (a: the flat streams, q, the FFN gate; b: k, the latent, the FFN up;
    // c: v, the KDA scan, v_b, the SwiGLU; d: beta, the KDA output; the quantized inputs: any product's input)
    const int64_t wa = std::max<int64_t>({N * 4, D, QB, g.n_ff}), wb = std::max<int64_t>({D, LB, g.n_ff});
    const int64_t wc = std::max<int64_t>({D, VB, g.n_ff}), wd = std::max<int64_t>(D, H);
    const int64_t widest = std::max<int64_t>({N * 4, D, VB, g.n_ff, 2 * N, g.q_lora});
    if (!(R = dalloc<float>((size_t) T * 4 * N, err)) || !(xn = dalloc<float>((size_t) T * N, err)) ||
        !(mix = dalloc<float>((size_t) T * 24, err)) || !(post = dalloc<float>((size_t) T * 4, err)) ||
        !(comb = dalloc<float>((size_t) T * 16, err)) || !(y = dalloc<float>((size_t) T * N, err)) ||
        !(a = dalloc<float>((size_t) T * wa, err)) || !(b = dalloc<float>((size_t) T * wb, err)) ||
        !(c = dalloc<float>((size_t) T * wc, err)) || !(d = dalloc<float>((size_t) T * wd, err)) ||
        !(e = dalloc<float>((size_t) T * std::max(LB, VB), err)) || !(f = dalloc<float>((size_t) T * N, err)) ||
        !(qkv = dalloc<float>((size_t) T * 3 * D, err)) || !(small1 = dalloc<float>((size_t) T * 4096, err)) ||
        !(small2 = dalloc<float>((size_t) T * 4096, err)) || !(gk = dalloc<float>((size_t) T * D, err)) ||
        !(beta = dalloc<float>((size_t) T * H, err)) ||
        !(part = dalloc<float>((size_t) glm5k::attend_scratch_floats(ctx, (int) H, (int) g.kv_lora, T), err)) ||
        !(logits = dalloc<float>((size_t) kMaxVerify * g.n_vocab, err)) ||
        !(rlog = dalloc<float>((size_t) T * g.n_expert, err)) || !(rw = dalloc<float>((size_t) T * g.n_used, err)) ||
        !(rids = dalloc<int32_t>((size_t) T * g.n_used, err)) || !(ycpu = dalloc<float>((size_t) T * N, err)))
        return false;
    if (!(hn = dalloc<float>((size_t) T * N, err)) || !(pending = dalloc<float>((size_t) N, err)) ||
        !(hin = dalloc<float>((size_t) T * N, err)) || !(mx = dalloc<float>((size_t) T * N, err)) ||
        !(me = dalloc<float>((size_t) T * N, err)) || !(mh = dalloc<float>((size_t) T * N, err)) ||
        !(d_arg = dalloc<int32_t>(4, err)))
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
    const size_t qbytes = guild::kernels::native_q8_1_bytes((int) widest, T);
    if (!(xq = dalloc<uint8_t>(qbytes, err)) || !(xq2 = dalloc<uint8_t>(qbytes, err)))
        return false;
#if defined(GUILD_GLM_MMQ)
    {
        namespace mmq = guild::prefill::mmq;
        if (mmq::built() && std::getenv("GUILD_GLM_MMQ") == nullptr) { // GUILD_GLM_MMQ=0: MMVQ for every window
            const size_t mb = mmq::q8_bytes(T, widest);
            std::vector<int32_t> bounds((size_t) 2 * (T + 1));
            for (int t = 0; t <= T; ++t) {
                bounds[(size_t) 2 * t] = 0;
                bounds[(size_t) 2 * t + 1] = t;
            }
            if (!(xqm = dalloc<uint8_t>(mb, err)) || !(xqm2 = dalloc<uint8_t>(mb, err)) ||
                !(mmq_bounds = dalloc<int32_t>(bounds.size(), err)) || !(mmq_ids = dalloc<int32_t>((size_t) T, err)))
                return false;
            cudaMemcpy(mmq_bounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice);
            mmq::iota(mmq_ids, T, st);
            mmq_ctx = std::make_unique<mmq::Context>();
            int n_mmq = 0;
            auto mark = [&](Mat& m) {
                // MMQ steps through a row 256 values at a time: a narrower row (the KDA gates' 128) reads past it.
                // Q6_K (GLM's re-quantized matrices, when the build has its MMQ) takes the activations Q8_0 does:
                // llama.cpp quantizes both to q8_1 in the same D4 layout
                const bool type_ok = m.type == kQ8_0 || (m.type == kQ6_K && mmq::supported(kQ6_K));
                m.mmq = m.dev && type_ok && m.n_in % 256 == 0 && mmq::fits(m.type, m.n_out);
                n_mmq += m.mmq;
            };
            std::vector<Layer*> all_layers;
            for (Layer& y : L)
                all_layers.push_back(&y);
            if (has_mtp)
                all_layers.push_back(&mtp);
            for (Layer* yp : all_layers)
                for (Mat* m :
                     {&yp->hc_attn_fn, &yp->hc_ffn_fn, &yp->wo,     &yp->wq,   &yp->wk,  &yp->wv,   &yp->f_a,
                      &yp->f_b,        &yp->g_a,       &yp->g_b,    &yp->beta, &yp->q_a, &yp->q_b,  &yp->kv_a,
                      &yp->idx_k,      &yp->idx_gate,  &yp->idx_qb, &yp->gate, &yp->up,  &yp->down, &yp->eh_proj})
                    mark(*m);
            std::fprintf(stderr, "guild glm5: %d matrices take MMQ in prompt windows\n", n_mmq);
        }
    }
#endif
    if (cudaMallocHost(&h_x, (size_t) T * N * 4) != cudaSuccess ||
        cudaMallocHost(&h_y, (size_t) T * N * 4) != cudaSuccess ||
        cudaMallocHost(&h_w, (size_t) T * g.n_used * 4) != cudaSuccess ||
        cudaMallocHost(&h_ids, (size_t) T * g.n_used * 4) != cudaSuccess ||
        cudaMallocHost(&h_emb, (size_t) T * N * 4) != cudaSuccess) {
        err = "glm5: no pinned host memory";
        return false;
    }

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
        const int nn = split ? 2 : 1;
        const int64_t dh = split ? N / 2 : N;
        auto h_of = [&](const CpuFfn& f) { return split ? f.n_ff / 2 : f.n_ff; };
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
            if (!yp->ffn_gpu)
                all.push_back(&yp->ffn_cpu);
        auto repacked = [&](const CpuFfn& f) { return nl4 && f.gu_type == GGML_TYPE_IQ3_S; };
        auto gu_row_of = [&](const CpuFfn& f) {
            return repacked(f) ? (size_t) (N / 256) * kcpu::kIq3sNl4Block : f.gu_row;
        };
        size_t node_bytes[2] = {0, 0};
        std::vector<std::array<size_t, 2>> at; // per FFN in `all`: the offset of its part in each arena
        for (const CpuFfn* fp : all) {
            const CpuFfn& f = *fp;
            const size_t gr = gu_row_of(f);
            const int64_t h = h_of(f);
            at.push_back({node_bytes[0], node_bytes[1]});
            node_bytes[0] += 2 * (size_t) h * gr + (size_t) dh * f.d_row;
            if (nn == 2)
                node_bytes[1] += 2 * (size_t) (f.n_ff - h) * gr + (size_t) (N - dh) * f.d_row;
        }
        for (int node = 0; node < nn; ++node) {
            void* p = mmap(nullptr, node_bytes[node], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) {
                err = "glm5: no memory for the experts (" + std::to_string(node_bytes[node] >> 30) + " GiB on node " +
                      std::to_string(node) + ")";
                return false;
            }
            unsigned long mask = 1ul << node;
            if (split && syscall(SYS_mbind, p, node_bytes[node], 2 /* MPOL_BIND */, &mask, 64ul, 0u) != 0)
                std::fprintf(stderr, "guild glm5: mbind to node %d failed; the experts' halves go where they land\n",
                             node);
            madvise(p, node_bytes[node], MADV_HUGEPAGE);
            arena[node] = (uint8_t*) p;
            arena_bytes[node] = node_bytes[node];
        }
        const int n_all = (int) all.size();
        std::atomic<bool> repack_ok{true};
        pool->run_nodes(n_all, nn == 2 ? n_all : 0, [&](int node, int i) {
            CpuFfn& f = *all[(size_t) i];
            const int64_t h = h_of(f);
            const int64_t r0 = node ? h : 0, rg = node ? f.n_ff - h : h;
            const int64_t d0 = node ? dh : 0, rd = node ? N - dh : dh;
            const size_t gr = gu_row_of(f);
            uint8_t* dst = arena[node] + at[(size_t) i][(size_t) node];
            if (repacked(f)) {
                for (int64_t r = 0; r < rg; ++r) {
                    bool ok = kcpu::iq3s_nl4_repack(f.gate[0] + (size_t) (r0 + r) * f.gu_row, (int) (N / 256),
                                                    dst + (size_t) r * gr);
                    ok &= kcpu::iq3s_nl4_repack(f.up[0] + (size_t) (r0 + r) * f.gu_row, (int) (N / 256),
                                                dst + (size_t) (rg + r) * gr);
                    if (!ok)
                        repack_ok = false;
                }
            } else {
                std::memcpy(dst, f.gate[0] + (size_t) r0 * f.gu_row, (size_t) rg * f.gu_row);
                std::memcpy(dst + (size_t) rg * gr, f.up[0] + (size_t) r0 * f.gu_row, (size_t) rg * f.gu_row);
            }
            std::memcpy(dst + 2 * (size_t) rg * gr, f.down[0] + (size_t) d0 * f.d_row, (size_t) rd * f.d_row);
        });
        if (!repack_ok) {
            err = "glm5: an IQ3_S expert row does not repack (a grid value outside the NL4 codes); run with "
                  "GUILD_GLM_NL4=0";
            return false;
        }
        // the pointers, once every part is copied (the copies read the mapping through gate[0] and friends)
        for (int i = 0; i < n_all; ++i) {
            CpuFfn& f = *all[(size_t) i];
            const size_t gr = gu_row_of(f);
            const int64_t h = h_of(f);
            for (int node = 0; node < 2; ++node) {
                const int src = node < nn ? node : 0;
                const int64_t r0 = src ? h : 0, rg = src ? f.n_ff - h : h, d0 = src ? dh : 0;
                const uint8_t* base = arena[src] + at[(size_t) i][(size_t) src];
                f.gate[node] = rebase(base, r0, gr);
                f.up[node] = rebase(base + (size_t) rg * gr, r0, gr);
                f.down[node] = rebase(base + 2 * (size_t) rg * gr, d0, f.d_row);
            }
            if (repacked(f))
                f.gu_type = kcpu::kIq3sNl4Type;
            f.gu_row = gr;
            f.gu_split = h;
            f.d_split = dh;
        }
        // the routed experts' pages of the shards are no longer read: let the kernel drop them (only theirs: the
        // embeddings stay read from the mapping)
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
    mm(Y.wq, xq, a, T);
    mm(Y.wk, xq, b, T);
    mm(Y.wv, xq, c, T);
    mm(Y.f_a, xq, small1, T);
    mm(Y.g_a, xq, small2, T);
    mm(Y.beta, xq, d, T); // [T][H] beta logits
    glm5k::kda_conv(a, b, c, Y.conv_state, Y.conv_q, Y.conv_k, Y.conv_v, qkv, (int) D, T, mid_conv, st);
    quant(small1, Y.f_a.n_out, T, xq2);
    mm(Y.f_b, xq2, a, T); // fpre [T][D]
    glm5k::kda_gates(a, Y.dt_b, Y.ssm_a, gk, d, beta, (int) D, (int) H, T, g.kda_lower_bound, st);
    quant(small2, Y.g_a.n_out, T, xq2);
    mm(Y.g_b, xq2, b, T); // output gate [T][D]
    glm5k::kda_step(Y.kda_state, qkv, gk, beta, c, (int) H, T, mid_state, st);
    glm5k::kda_out(c, b, Y.ssm_norm, d, (int) H, T, g.rms_eps, st);
    quant(d, D, T, xq2);
    mm(Y.wo, xq2, y, T);
}

void Glm5Engine::Impl::attn_dsa(Layer& Y, int T, int64_t pos0) {
    const int64_t N = g.n_embd, H = g.n_head, QL = g.q_lora, KL = g.kv_lora;
    quant(xn, N, T, xq);
    mm(Y.q_a, xq, small1, T);
    mm(Y.kv_a, xq, small2, T);
    mm(Y.idx_k, xq, ikr, T);
    mm(Y.idx_gate, xq, igr, T);
    glm5k::f32_matvec(Y.idx_proj, xn, iw, (int) N, (int) g.idx_heads, T, st);
    glm5k::rms_norm(small1, Y.q_a_norm, small1, (int) QL, T, g.rms_eps, st);
    glm5k::rms_norm(small2, Y.kv_a_norm, small2, (int) KL, T, g.rms_eps, st);
    quant(small1, QL, T, xq2);
    mm(Y.q_b, xq2, a, T); // q [T][H][qk_dim]
    glm5k::kv_store(small2, Y.kv_cache, pos0, (int) KL, T, st);
    // the indexer's key and gate are stored for every cell, dense or not, and pooled as pools complete
    glm5k::idx_store(ikr, Y.idx_ln_w, Y.idx_ln_b, igr, Y.idx_kc, Y.idx_gc, pos0, T, g.ln_eps, st);
    glm5k::idx_pool(Y.idx_kc, Y.idx_gc, Y.idx_ape, Y.pooled, pos0, T, st);
    glm5k::q8_0_heads(Y.k_b.dev, a, e, (int) H, (int) g.qk_dim, (int) KL, T, st); // absorbed q [T][H][KL]
    const float scale = 1.f / std::sqrt((float) g.qk_dim);
    // a query sees at most idx_topk / kpool pools before the selection is anything but all of them
    const int k_pools = (int) (g.idx_topk / glm5k::kKpool);
    if ((pos0 + T) / glm5k::kKpool > k_pools) {
        mm(Y.idx_qb, xq2, iq, T);
        glm5k::idx_score(iq, iw, Y.pooled, pos0, T, (int) g.idx_heads,
                         1.f / std::sqrt((float) (g.idx_dim * g.idx_heads)), scores, score_stride, st);
        glm5k::idx_select(scores, score_stride, pos0, T, k_pools, sel, nsel, sel_stride, st);
        glm5k::mla_attend(e, Y.kv_cache, pos0, (int) H, (int) KL, T, scale, sel, nsel, sel_stride, b, part, st);
    } else {
        glm5k::mla_attend(e, Y.kv_cache, pos0, (int) H, (int) KL, T, scale, nullptr, nullptr, 0, b, part, st);
    }
    glm5k::q8_0_heads(Y.v_b.dev, b, c, (int) H, (int) KL, (int) g.v_dim, T, st); // [T][H][v_dim]
    quant(c, H * g.v_dim, T, xq2);
    mm(Y.wo, xq2, y, T);
}

bool Glm5Engine::Impl::layer_attention(int l, int T, int64_t pos0, bool save_mid, std::string& err) {
    Layer& Y = L[(size_t) l];
    const int64_t N = g.n_embd;
    // mHC pre-mix: mixes = hc_fn . rms_norm(R flat), then the stream sum, normed by attn_norm
    glm5k::rms_norm(R, nullptr, a, (int) (4 * N), T, g.rms_eps, st);
    quant(a, 4 * N, T, xq);
    mm(Y.hc_attn_fn, xq, mix, T);
    glm5k::hc_pre(mix, Y.hc_attn_scale, Y.hc_attn_base, R, Y.attn_norm, post, comb, xn, (int) N, T,
                  (int) g.sinkhorn_iters, g.hc_eps, g.rms_eps, st);
    if (Y.kda)
        attn_kda(Y, T, save_mid ? Y.kda_mid : nullptr, save_mid ? Y.conv_mid : nullptr);
    else
        attn_dsa(Y, T, pos0);
    glm5k::hc_post(y, R, post, comb, (int) N, T, st);
    (void) err;
    return true;
}

void Glm5Engine::Impl::cpu_ffn(const Layer& Y, int T) {
    // Groups: one FFN slice and the tokens it serves (a routed expert and the tokens routed to it, or the layer's own
    // FFN for every token when it lives on the CPU).  Units: (group, token) pairs, each with its hidden activation.
    // Phase 1, gate/up, over (group, row block): a group's tokens share the weight decode (multi-token kernels from
    // two tokens up).  Phase 2 quantizes the hidden activations.  Phase 3, down, over row blocks of the output, each
    // summing its rows over every unit of every token, so no two threads write one row.
    const int64_t N = g.n_embd;
    struct Group {
        CpuFfn f;
        std::vector<int> units; // indices into `unit_t` / `unit_w`
    };
    std::vector<Group> groups;
    std::vector<int> unit_t, unit_g;
    std::vector<float> unit_w;
    if (Y.moe) {
        std::vector<int> group_of((size_t) g.n_expert, -1);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < (int) g.n_used; ++j) {
                const int ex = h_ids[t * g.n_used + j];
                if (group_of[(size_t) ex] < 0) {
                    group_of[(size_t) ex] = (int) groups.size();
                    Group gr;
                    gr.f = Y.experts[(size_t) ex];
                    groups.push_back(std::move(gr));
                }
                const int gi = group_of[(size_t) ex];
                groups[(size_t) gi].units.push_back((int) unit_t.size());
                unit_t.push_back(t);
                unit_g.push_back(gi);
                unit_w.push_back(h_w[t * g.n_used + j]);
            }
    }
    if (!Y.ffn_gpu) {
        Group gr;
        gr.f = Y.ffn_cpu;
        const int gi = (int) groups.size();
        for (int t = 0; t < T; ++t) {
            gr.units.push_back((int) unit_t.size());
            unit_t.push_back(t);
            unit_g.push_back(gi);
            unit_w.push_back(1.f);
        }
        groups.push_back(std::move(gr));
    }
    const int U = (int) unit_t.size();

    // the quantized inputs: one per (token, activation type)
    std::map<int, size_t> act_off; // (vec_dot_type * kMaxT + t) -> offset
    std::vector<size_t> act_at((size_t) U);
    size_t total = 0;
    for (int u = 0; u < U; ++u) {
        const int at = act_type(groups[(size_t) unit_g[(size_t) u]].f.gu_type);
        const int key = at * kMaxT + unit_t[(size_t) u];
        auto it = act_off.find(key);
        if (it == act_off.end()) {
            it = act_off.emplace(key, total).first;
            total += ggml_row_size((ggml_type) at, N);
        }
        act_at[(size_t) u] = it->second;
    }
    static const bool timing = std::getenv("GUILD_GLM_TIMING") != nullptr;
    auto tp = std::chrono::steady_clock::now();
    auto lap = [&](int k) {
        if (!timing)
            return;
        const auto now = std::chrono::steady_clock::now();
        phase_ms[k] += std::chrono::duration<double, std::milli>(now - tp).count();
        tp = now;
    };
    actbuf.resize(total + 64);
    for (const auto& kv : act_off) {
        const int at = kv.first / kMaxT, t = kv.first % kMaxT;
        cpu_traits(at)->from_float(h_x + (size_t) t * N, actbuf.data() + kv.second, N);
    }
    std::vector<size_t> h_off((size_t) U + 1, 0), hq_off((size_t) U + 1, 0);
    for (int u = 0; u < U; ++u) {
        const CpuFfn& f = groups[(size_t) unit_g[(size_t) u]].f;
        h_off[(size_t) u + 1] = h_off[(size_t) u] + (size_t) f.n_ff;
        const int dt = (int) cpu_traits(f.d_type)->vec_dot_type;
        hq_off[(size_t) u + 1] = hq_off[(size_t) u] + ggml_row_size((ggml_type) dt, f.n_ff);
    }
    hbuf.resize(h_off[(size_t) U] * 4);
    hqbuf.resize(hq_off[(size_t) U] + 64);
    float* hv = (float*) hbuf.data();

    // the multi-token row kernel for a weight type, or null (ggml-cpu's single-token dot then)
    using RowsFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
    auto multi = [](int type) -> RowsFn {
        if (kcpu::cpu_avx512_ok() && kcpu::iq512_supported(type))
            return &kcpu::iq512_rows;
        if (kcpu::cpu_avx2_ok() && kcpu::iq256_supported(type))
            return &kcpu::iq256_rows;
        if (kcpu::cpu_avx2_ok() && kcpu::kq256_supported(type))
            return &kcpu::kq256_rows;
        return nullptr;
    };
    constexpr int RB = 64, MT = kcpu::MAXT;

    lap(0);
    // phase 1: gate/up, each row block on the queue of the node that holds it
    std::vector<std::pair<int, int>> items[2];
    for (int gi = 0; gi < (int) groups.size(); ++gi) {
        const CpuFfn& f = groups[(size_t) gi].f;
        for (int node = 0; node < 2; ++node) {
            const int64_t a = node ? f.gu_split : 0, b = node ? f.n_ff : f.gu_split;
            for (int64_t r = a; r < b; r += RB)
                items[node].emplace_back(gi, (int) r);
        }
    }
    pool->run_nodes((int) items[0].size(), (int) items[1].size(), [&](int node, int i) {
        const Group& gr = groups[(size_t) items[node][(size_t) i].first];
        const CpuFfn& f = gr.f;
        const int r0 = items[node][(size_t) i].second;
        const int r1 = (int) std::min<int64_t>(r0 + RB, node ? f.n_ff : f.gu_split), nr = r1 - r0;
        const uint8_t* wg = f.gate[node];
        const uint8_t* wu = f.up[node];
        const bool nl4 = f.gu_type == kcpu::kIq3sNl4Type;
        const ggml_type_traits_cpu* tr = nl4 ? nullptr : cpu_traits(f.gu_type);
        const RowsFn mt = !nl4 && gr.units.size() >= 2 ? multi(f.gu_type) : nullptr;
        float gbuf[MT][RB], ubuf[MT][RB];
        for (size_t c0 = 0; c0 < gr.units.size(); c0 += MT) {
            const int nt = (int) std::min<size_t>(MT, gr.units.size() - c0);
            const void* act[MT];
            float* go[MT];
            float* uo[MT];
            for (int k = 0; k < nt; ++k) {
                act[k] = actbuf.data() + act_at[(size_t) gr.units[c0 + k]];
                go[k] = gbuf[k] - r0; // the kernels index outputs by row
                uo[k] = ubuf[k] - r0;
            }
            if (nl4) {
                glm5cpu::nl4_rows(wg, f.gu_row, (int) N, act, nt, go, r0, r1);
                glm5cpu::nl4_rows(wu, f.gu_row, (int) N, act, nt, uo, r0, r1);
            } else if (mt && nt >= 2) {
                mt(f.gu_type, wg, f.gu_row, (int) N, act, nt, go, r0, r1);
                mt(f.gu_type, wu, f.gu_row, (int) N, act, nt, uo, r0, r1);
            } else {
                for (int k = 0; k < nt; ++k)
                    for (int r = r0; r < r1; ++r) {
                        tr->vec_dot((int) N, &go[k][r], 0, wg + (size_t) r * f.gu_row, 0, act[k], 0, 1);
                        tr->vec_dot((int) N, &uo[k][r], 0, wu + (size_t) r * f.gu_row, 0, act[k], 0, 1);
                    }
            }
            for (int k = 0; k < nt; ++k) {
                float* h = hv + h_off[(size_t) gr.units[c0 + k]];
                for (int j = 0; j < nr; ++j) {
                    float gt = gbuf[k][j], up = ubuf[k][j];
                    if (f.lim > 0.f) {
                        gt = std::min(gt, f.lim);
                        up = std::clamp(up, -f.lim, f.lim);
                    }
                    h[r0 + j] = gt / (1.f + std::exp(-gt)) * up;
                }
            }
        }
    });
    lap(1);
    // phase 2: the hidden activations
    pool->run(U, [&](int u) {
        const CpuFfn& f = groups[(size_t) unit_g[(size_t) u]].f;
        cpu_traits(cpu_traits(f.d_type)->vec_dot_type)
            ->from_float(hv + h_off[(size_t) u], hqbuf.data() + hq_off[(size_t) u], f.n_ff);
    });
    lap(2);
    // phase 3: down rows, summed over each token's units; the first half of the rows on node 0 (where the split puts
    // them), the second on node 1
    const int64_t half = (N / 2 + RB - 1) / RB * RB;
    const int nrb0 = (int) (half / RB), nrb1 = (int) ((N - half + RB - 1) / RB);
    pool->run_nodes(nrb0, nrb1, [&](int node, int i) {
        const int r0 = (int) (node ? half : 0) + i * RB, r1 = (int) std::min<int64_t>(r0 + RB, node ? N : half),
                  nr = r1 - r0;
        for (int t = 0; t < T; ++t)
            std::fill(h_y + (size_t) t * N + r0, h_y + (size_t) t * N + r1, 0.f);
        float dbuf[MT][RB];
        for (const Group& gr : groups) {
            const CpuFfn& f = gr.f;
            const ggml_type_traits_cpu* tr = cpu_traits(f.d_type);
            const RowsFn mt = gr.units.size() >= 2 ? multi(f.d_type) : nullptr;
            for (size_t c0 = 0; c0 < gr.units.size(); c0 += MT) {
                const int nt = (int) std::min<size_t>(MT, gr.units.size() - c0);
                const void* hq[MT];
                float* dout[MT];
                for (int k = 0; k < nt; ++k) {
                    hq[k] = hqbuf.data() + hq_off[(size_t) gr.units[c0 + k]];
                    dout[k] = dbuf[k] - r0;
                }
                // the block lies on one side of this group's split (the split is at `half` or past N)
                const uint8_t* wd = f.down[r0 >= f.d_split];
                if (mt && nt >= 2) {
                    mt(f.d_type, wd, f.d_row, (int) f.n_ff, hq, nt, dout, r0, r1);
                } else {
                    for (int k = 0; k < nt; ++k)
                        for (int r = r0; r < r1; ++r)
                            tr->vec_dot((int) f.n_ff, &dout[k][r], 0, wd + (size_t) r * f.d_row, 0, hq[k], 0, 1);
                }
                for (int k = 0; k < nt; ++k) {
                    const int u = gr.units[c0 + k];
                    float* out = h_y + (size_t) unit_t[(size_t) u] * N + r0;
                    const float w = unit_w[(size_t) u];
                    for (int j = 0; j < nr; ++j)
                        out[j] += w * dbuf[k][j];
                }
            }
        }
    });
    lap(3);
}

bool Glm5Engine::Impl::ffn(const Layer& Y, int T, std::string& err) {
    const int64_t N = g.n_embd;
    const bool cpu = Y.moe || !Y.ffn_gpu;
    if (Y.moe) {
        glm5k::f32_matvec(Y.router, xn, rlog, (int) N, (int) g.n_expert, T, st);
        glm5k::route(rlog, Y.exp_bias, (int) g.n_expert, (int) g.n_used, g.expert_norm, g.expert_scale, rids, rw, T,
                     st);
        cudaMemcpyAsync(h_ids, rids, (size_t) T * g.n_used * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(h_w, rw, (size_t) T * g.n_used * 4, cudaMemcpyDeviceToHost, st);
    }
    if (cpu) {
        cudaMemcpyAsync(h_x, xn, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st);
        cudaEventRecord(ev, st);
    }
    if (Y.ffn_gpu) { // the shared expert or the dense FFN, while the CPU computes the routed experts
        const int64_t F = Y.gate.n_out;
        quant(xn, N, T, xq);
        mm(Y.gate, xq, a, T);
        mm(Y.up, xq, b, T);
        glm5k::swiglu_clamp(a, b, c, (int) F, T, Y.lim_ffn > 0.f ? Y.lim_ffn : 3.0e38f, st);
        quant(c, F, T, xq2);
        mm(Y.down, xq2, y, T);
    }
    if (cpu) {
        if (cudaEventSynchronize(ev) != cudaSuccess) {
            err = "glm5: CUDA error before the CPU experts: " + std::string(cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        const auto t0 = std::chrono::steady_clock::now();
        cpu_ffn(Y, T);
        cpu_ms += ms_since(t0);
        if (Y.ffn_gpu) {
            cudaMemcpyAsync(ycpu, h_y, (size_t) T * N * 4, cudaMemcpyHostToDevice, st);
            glm5k::add(y, ycpu, T * N, st);
        } else {
            cudaMemcpyAsync(y, h_y, (size_t) T * N * 4, cudaMemcpyHostToDevice, st);
        }
    }
    return true;
}

bool Glm5Engine::Impl::layer_ffn(int l, int T, std::string& err) {
    Layer& Y = L[(size_t) l];
    const int64_t N = g.n_embd;
    glm5k::rms_norm(R, nullptr, a, (int) (4 * N), T, g.rms_eps, st);
    quant(a, 4 * N, T, xq);
    mm(Y.hc_ffn_fn, xq, mix, T);
    glm5k::hc_pre(mix, Y.hc_ffn_scale, Y.hc_ffn_base, R, Y.ffn_norm, post, comb, xn, (int) N, T, (int) g.sinkhorn_iters,
                  g.hc_eps, g.rms_eps, st);
    if (!ffn(Y, T, err))
        return false;
    glm5k::hc_post(y, R, post, comb, (int) N, T, st);
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
uint64_t Glm5Engine::vram_bytes() const {
    return p_->vram;
}
const std::string& Glm5Engine::placement() const {
    return p_->placement;
}
void Glm5Engine::print_timing(const char* what) const {
    if (std::getenv("GUILD_GLM_TIMING"))
        std::fprintf(stderr, "glm5 timing (%s): activations %.0f ms, gate/up %.0f, hidden %.0f, down %.0f\n", what,
                     p_->phase_ms[0], p_->phase_ms[1], p_->phase_ms[2], p_->phase_ms[3]);
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
    cudaStreamSynchronize(m.st);
}

bool Glm5Engine::snapshot(std::string& err) {
    Impl& m = *p_;
    const size_t sb = (size_t) m.g.n_head * 128 * 128 * 4, cb = (size_t) 3 * m.g.n_head * m.g.kda_dim * 3 * 4;
    const size_t n_kda = (size_t) std::count(m.g.is_kda.begin(), m.g.is_kda.end(), true);
    const size_t pb = (size_t) m.g.n_embd * 4; // the NextN block's pending row
    if (!m.snap && cudaMallocHost(&m.snap, n_kda * (sb + cb) + pb) != cudaSuccess) {
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
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: snapshot: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool Glm5Engine::restore(std::string& err) {
    Impl& m = *p_;
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
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: restore: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
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
    for (int l = 0; l < (int) g.n_layer; ++l) {
        if (!layer_attention(l, T, pos0, save_mid, err))
            return false;
        check(l, Y_kind(l));
        if (!layer_ffn(l, T, err))
            return false;
        check(l, "ffn");
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
        mm(head, xq, logits, 1);
        glm5k::argmax(logits, (int) g.n_vocab, d_arg, st);
        cudaMemcpyAsync(h_arg, d_arg, 4, cudaMemcpyDeviceToHost, st);
    }
    return true;
}

bool Glm5Engine::Impl::mtp_catch_up(const int32_t* tokens, int T, int64_t pos0, std::string& err) {
    // the NextN block reads (token p, hidden p - 1): the window's rows shifted by one, the pending row first
    const int64_t N = g.n_embd;
    cudaMemcpyAsync(hin, pending, (size_t) N * 4, cudaMemcpyDeviceToDevice, st);
    if (T > 1)
        cudaMemcpyAsync(hin + N, hn, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(pending, hn + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st);
    return mtp_window(tokens, T, pos0, hin, false, err);
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
    for (int base = 0; base < n; base += kMaxT) {
        const int T = std::min(kMaxT, n - base);
        if (!m.main_window(tokens + base, T, pos0 + base, false, err))
            return false;
        if (logits && base + T == n && !m.head_rows(T - 1, 1, logits, err))
            return false;
        if (m.has_mtp && !m.mtp_catch_up(tokens + base, T, pos0 + base, err))
            return false;
        if (cudaStreamSynchronize(m.st) != cudaSuccess) {
            err = std::string("glm5: CUDA error: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    (void) N;
    return true;
}

bool Glm5Engine::has_mtp() const {
    return p_->has_mtp;
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
    if (!m.mtp_window(&last, 1, pos, m.pending, true, err))
        return false;
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: CUDA error: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    draft = m.h_arg[0];
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
        if (n_keep > 1) {
            cudaMemcpyAsync(m.hin, m.hn, (size_t) (n_keep - 1) * N * 4, cudaMemcpyDeviceToDevice, m.st);
            if (!m.mtp_window(m.win_tok.data() + 1, n_keep - 1, m.win_pos + 1, m.hin, false, err))
                return false;
        }
        cudaMemcpyAsync(m.pending, m.hn + (size_t) (n_keep - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, m.st);
    }
    m.win_tok.clear();
    if (cudaStreamSynchronize(m.st) != cudaSuccess) {
        err = std::string("glm5: CUDA error: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

} // namespace guild::core
