// Layer-wise known-token scheduling. No head/sampling or speculative verifier.
// GPU projections use the existing <=8-column quantized arithmetic. Recurrent
// updates preserve temporal order; CPU jobs group an expert's rows in <=MAXT
// microbatches. Routing outputs remain in token/rank order for the GPU combine.
// All async-copy sources are owned until stream completion; no per-token GPU
// allocation. See bench/results/ornith-prefill/README.md for precision and gates.
#include "guild/runtime/batched_prefill.hpp"
#include "guild/core/native_head.hpp"
#include "guild/core/validation.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/kernels/native_mmvq.hpp"
#include "guild/kernels/native_qsa.hpp"
#include "guild/kernels/native_rope.hpp"
#include "guild/kernels/native_moe.hpp"
#include "guild/kernels/verify_kernels.hpp"
#include "guild/kernels/rope.hpp"
#include "guild/kernels/kv_q4.hpp"
#include "guild/kernels/kv_q8.hpp"
#include "guild/prefill/kernels.hpp"
#include "guild/kernels/qsa_decode_attn.hpp"
#include "guild/kernels/shared_expert.hpp"
#include "guild/kernels/elementwise.hpp"
#include "guild/kernels/bf16_gemv.hpp"
#include "guild/kernels/router_top10.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace guild::runtime {
namespace {
using namespace guild::kernels;
constexpr int kProjectionRows = kVerifyMaxT;
constexpr int kMaxBatch = 512;
constexpr int kAttentionQueries = 32; // bounded split-K scratch, independent of prompt length
constexpr float kEps = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
guild_exec_context_v1_t cuda_ctx(void* stream) {
    guild_exec_context_v1_t c{};
    c.struct_size = sizeof(c);
    c.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    c.backend = GUILD_BACKEND_CUDA;
    if (cudaGetDevice(&c.device_id) != cudaSuccess || c.device_id < 0) {
        cudaGetLastError();
        c.device_id = 0;
    }
    c.device_stream = stream;
    c.sync_mode = GUILD_SYNC_STREAM_ORDERED;
    return c;
}
void cuda_ok(guild_status_t st, const guild_error_info_t& e) {
    if (st == GUILD_STATUS_OK)
        return;
    throw std::runtime_error(e.message[0] ? e.message : "CUDA dense prefill execution failed");
}
void quant(const ResolvedPrefillKernels& fns, const float* x, void* dst, int width, int rows, void* stream) {
    if (fns.has_cuda_dense) {
        guild_cuda_quantize_q8_1_v1_t p{};
        p.struct_size = sizeof(p);
        p.x = x;
        p.q = dst;
        p.n_in = width;
        p.rows = rows;
        p.q_bytes = native_q8_1_bytes(width, rows);
        const auto c = cuda_ctx(stream);
        guild_error_info_t e{};
        cuda_ok(fns.cuda_dense.table.quantize_q8_1(&c, &p, &e), e);
        return;
    }
    fns.quantize_q8_1(x, dst, width, rows, stream);
}
void project(const ResolvedPrefillKernels& fns, const core::WeightRef& w, const void* x, float* y, int rows,
             void* stream) {
    if (!w.native_data)
        throw std::runtime_error("batched prefill: missing native projection");
    if (fns.has_cuda_dense) {
        guild_cuda_mmvq_v1_t p{};
        p.struct_size = sizeof(p);
        p.exact_multi = native_mmvq_multi_exact();
        p.weight_dtype = w.native_type;
        p.n_in = int(w.ne0);
        p.n_out = int(w.ne1);
        p.rows = rows;
        p.weights = w.native_data;
        p.q = x;
        p.y = y;
        const auto c = cuda_ctx(stream);
        guild_error_info_t e{};
        cuda_ok(fns.cuda_dense.table.mmvq(&c, &p, &e), e);
        return;
    }
    fns.mmvq(w.native_type, w.native_data, x, y, int(w.ne0), int(w.ne1), rows, stream);
}
const core::WeightRef& need(const core::LayerView& v, const char* key) {
    const auto* w = v.get(key);
    if (!w)
        throw std::runtime_error(v.name(key) + ": missing prefill weight");
    return *w;
}
} // namespace

struct BatchedPrefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    kernels::cpu::ExpertPool* pool = nullptr;
    cudaStream_t cs = nullptr;
    int capacity = 0, cap = 0, pos_stride = 0;
    uint64_t allocated = 0;
    std::vector<void*> device, pinned;
    int32_t *tokens = nullptr, *positions = nullptr, *steps = nullptr, *ident = nullptr, *keep = nullptr;
    int32_t *htokens = nullptr, *hpositions = nullptr, *hsteps = nullptr, *hkeep = nullptr, *hids = nullptr;
    float *R = nullptr, *mixed = nullptr, *bo = nullptr, *qkv = nullptr, *h = nullptr, *z = nullptr, *y = nullptr;
    float *gate = nullptr, *beta = nullptr, *qfull = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *attn = nullptr,
          *scratch = nullptr;
    float *logits = nullptr, *weights = nullptr, *shared = nullptr, *parts = nullptr, *sg = nullptr, *su = nullptr,
          *scalar = nullptr;
    float *hact = nullptr, *hparts = nullptr, *hweights = nullptr;
    int32_t* ids = nullptr;
    uint16_t* bf = nullptr;
    uint8_t* xq = nullptr;
    // Resident expert microbatch scratch.
    float *gx = nullptr, *gg = nullptr, *gu = nullptr, *go = nullptr;
    uint8_t* gq = nullptr;
    std::vector<uint8_t> act;
    std::vector<float> validation;
    std::vector<std::vector<int>> entries;
    std::vector<std::vector<uint8_t>> file_storage;
    std::vector<kernels::cpu::ExpertJobMulti> jobs;
    std::vector<const void*> job_nact;
    std::vector<float*> job_out;
    kernels::ResolvedPrefillKernels fns{};
    bool profiling = false;
    cudaEvent_t events[5] = {};
    std::FILE* profile_file = nullptr;
    ~Impl() {
        if (cs)
            cudaStreamSynchronize(cs);
        for (auto e : events)
            if (e)
                cudaEventDestroy(e);
        for (auto p : device)
            cudaFree(p);
        for (auto p : pinned)
            cudaFreeHost(p);
        if (profile_file)
            std::fclose(profile_file);
    }
    template <class T> void alloc(T*& p, size_t n, bool host = false) {
        void* raw = nullptr;
        ck(host ? cudaHostAlloc(&raw, n * sizeof(T), cudaHostAllocPortable) : cudaMalloc(&raw, n * sizeof(T)));
        (host ? pinned : device).push_back(raw);
        p = static_cast<T*>(raw);
        if (!host)
            allocated += n * sizeof(T);
    }
    void norm(const float* in, const core::WeightRef& w, float* out, int cols, int rows) {
        fns.rms_norm_weighted(in, static_cast<const float*>(w.data), out, cols, rows, kEps, cs);
    }
    void finite(const float* p, size_t n, const char* label) {
        validation.resize(n);
        ck(cudaMemcpyAsync(validation.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost, cs));
        ck(cudaStreamSynchronize(cs));
        std::string e;
        if (!core::validate_finite(validation.data(), n, label, e))
            throw std::runtime_error(e);
    }
    void stamp(int i) {
        if (profiling)
            ck(cudaEventRecord(events[i], cs));
    }
    double elapsed(int a, int b) {
        float value = 0;
        if (profiling)
            ck(cudaEventElapsedTime(&value, events[a], events[b]));
        return value;
    }
};

BatchedPrefill::BatchedPrefill() : impl_(std::make_unique<Impl>()) {
}
BatchedPrefill::~BatchedPrefill() = default;
void BatchedPrefill::set_expert_cache(const core::ExpertCache* cache) {
    impl_->cache = cache;
}
void BatchedPrefill::set_batch_size(int64_t n) {
    if (n < 1 || n > impl_->capacity)
        throw std::invalid_argument("prefill batch exceeds allocated capacity");
    batch_size_ = n;
}
const std::vector<std::string>& BatchedPrefill::bound_impl_ids() const {
    return impl_->fns.bound_impl_ids;
}
const std::vector<std::string>& BatchedPrefill::bound_package_ids() const {
    return impl_->fns.bound_package_ids;
}

bool BatchedPrefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                          core::ExpertSource& src, kernels::cpu::ExpertPool& pool, void* stream, std::string& err,
                          int64_t requested, const core::ExpertCache* cache) try {
    core::ScopedModelExecution exec_scope(ss.exec_state);
    auto& m = *impl_;
    if (m.wt)
        throw std::invalid_argument("prefill already initialized");
    if (!stream || g.hc != 1 || g.idx_q_heads != 0 || !g.is_dense_attention() || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4 || g.head_dim != 256 || g.n_head_kv < 1 || g.n_head % g.n_head_kv || ss.k < 1 || ss.k > 15)
        throw std::invalid_argument("batched prefill: unsupported execution geometry");
    if (!native_qsa_enabled() || !native_rope_enabled() || !native_moe_combine_enabled() ||
        !shared_expert_native_bf16_enabled() || !kernels::cpu::expert_layout().native)
        throw std::invalid_argument("batched prefill requires the validated native arithmetic contract");
    if (!kernels::KernelRegistry::instance().resolve_prefill_kernels(kernels::query_host_device_capabilities(), m.fns,
                                                                     err))
        throw std::runtime_error(err);
    m.g = &g;
    m.ss = &ss;
    m.src = &src;
    m.pool = &pool;
    m.cache = cache;
    m.cs = static_cast<cudaStream_t>(stream);
    m.profiling = std::getenv("GUILD_PREFILL_PROFILE") != nullptr;
    if (const char* path = std::getenv("GUILD_PREFILL_PROFILE_CSV")) {
        m.profile_file = std::fopen(path, "w");
        if (!m.profile_file)
            throw std::runtime_error("cannot open GUILD_PREFILL_PROFILE_CSV");
        m.profiling = true;
        std::fputs("prompt_tokens,chunk_pos,layer,rows,gpu_dense_interval_ms,gpu_mixer_interval_ms,cpu_expert_wall_ms,"
                   "cpu_combine_wall_ms,cpu_group_wall_ms,host_wait_ms,d2h_activation_bytes,h2d_activation_bytes\n",
                   m.profile_file);
    }
    if (m.profiling)
        for (auto& event : m.events)
            ck(cudaEventCreate(&event));
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const core::LayerView lv(wt, l);
        for (const auto* key :
             {"attn_norm.weight", "post_attention_norm.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
              "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
            need(lv, key);
    }
    // Use actual free device memory AFTER weights/KV/verifier, with the planner's reserve.
    size_t free = 0, total = 0;
    ck(cudaMemGetInfo(&free, &total));
    const int N = int(g.n_embd), C = int(g.ssm_conv_channels), V = int(g.ssm_value_dim), F = int(g.n_ff);
    const int NH = int(g.n_head), HD = int(g.head_dim), NK = int(g.n_head_kv), K = int(ss.k), NE = int(g.n_expert);
    QsaShapes shape = qsa_real_shapes();
    shape.n_head = NH;
    shape.n_head_kv = NK;
    shape.head_dim = HD;
    shape.idx_n_head = 0;
    m.cap = int(std::min<int64_t>(ss.max_cells, kTopkMaxCells));
    const uint64_t fixed = qsa_decode_attn_scratch_floats(m.cap, shape) * sizeof(float) * kAttentionQueries;
    const uint64_t perrow =
        4ull * (5 * N + 2 * C + 2 * V + 2 * g.ssm_v_heads + 4 * NH * HD + 2 * NK * HD + NE + K + 2 * F + 1 + K * N) +
        2ull * N + 36ull / 4 * std::max({N, V, NH * HD, F}) + 4ull * (NH + NK + kStepCount + 1);
    if (requested < 0 || requested > kMaxBatch)
        throw std::invalid_argument("prefill batch must be 0(auto) or 1..512");
    int chosen = requested ? int(requested) : 256;
    const uint64_t reserve = 512ull << 20;
    if (!requested)
        while (chosen > 1 && fixed + perrow * chosen + reserve > free)
            chosen /= 2;
    if (fixed + perrow * chosen + reserve > free)
        throw std::runtime_error("prefill scratch exceeds free VRAM and reserve");
    m.capacity = chosen;
    batch_size_ = chosen;
    m.pos_stride = NH + NK;
    const size_t B = chosen;
    m.alloc(m.tokens, B);
    m.alloc(m.positions, B * m.pos_stride);
    m.alloc(m.steps, B * kStepCount);
    m.alloc(m.keep, 1);
    m.alloc(m.ident, size_t(kAttentionQueries) * m.cap);
    m.alloc(m.R, B * N);
    m.alloc(m.mixed, B * N);
    m.alloc(m.bo, B * N);
    m.alloc(m.qkv, B * C);
    m.alloc(m.h, B * C);
    m.alloc(m.z, B * V);
    m.alloc(m.y, B * V);
    m.alloc(m.gate, B * g.ssm_v_heads);
    m.alloc(m.beta, B * g.ssm_v_heads);
    m.alloc(m.qfull, B * NH * HD * 2);
    m.alloc(m.q, B * NH * HD);
    m.alloc(m.k, B * NK * HD);
    m.alloc(m.v, B * NK * HD);
    m.alloc(m.attn, B * NH * HD);
    m.alloc(m.scratch, qsa_decode_attn_scratch_floats(m.cap, shape) * kAttentionQueries);
    m.alloc(m.logits, B * NE);
    m.alloc(m.ids, B * K);
    m.alloc(m.weights, B * K);
    m.alloc(m.shared, B * N);
    m.alloc(m.parts, B * K * N);
    m.alloc(m.sg, B * F);
    m.alloc(m.su, B * F);
    m.alloc(m.scalar, B);
    m.alloc(m.bf, B * N);
    m.alloc(m.xq, B * std::max({N, V, NH * HD, F}) / 32 * 36);
    m.alloc(m.gx, size_t(kProjectionRows) * N);
    m.alloc(m.gg, size_t(kProjectionRows) * F);
    m.alloc(m.gu, size_t(kProjectionRows) * F);
    m.alloc(m.go, size_t(kProjectionRows) * N);
    m.alloc(m.gq, size_t(kProjectionRows) * std::max(N, F) / 32 * 36);
    {
        struct HostAllocAffinity {
            kernels::cpu::ThreadAffinity prev;
            explicit HostAllocAffinity(int core) : prev(kernels::cpu::pin_current_thread_if_allowed(core)) {
            }
            ~HostAllocAffinity() {
                kernels::cpu::restore_thread_affinity(prev);
            }
        } local_alloc(pool.host_core());
        m.alloc(m.htokens, B, true);
        m.alloc(m.hpositions, B * m.pos_stride, true);
        m.alloc(m.hsteps, B * kStepCount, true);
        m.alloc(m.hkeep, 1, true);
        m.alloc(m.hact, B * N, true);
        m.alloc(m.hparts, B * K * N, true);
        m.alloc(m.hids, B * K, true);
        m.alloc(m.hweights, B * K, true);
    }
    std::vector<int32_t> identity(size_t(kAttentionQueries) * m.cap);
    for (size_t i = 0; i < identity.size(); ++i)
        identity[i] = int(i % m.cap);
    ck(cudaMemcpy(m.ident, identity.data(), identity.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    m.entries.resize(NE);
    m.file_storage.resize(NE);
    m.jobs.reserve(B * K);
    m.act.resize(B * kernels::cpu::kNativeActBytes);
    m.wt = &wt;
    err.clear();
    return true;
} catch (const std::exception& e) {
    err = e.what();
    return false;
}

bool BatchedPrefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) try {
    auto& m = *impl_;
    stats_ = {};
    stats_.device_bytes = m.allocated;
    if (!m.wt || n < 0 || pos0 < 0 || (n && !tokens))
        throw std::invalid_argument("uninitialized/invalid prefill request");
    if (n > m.ss->max_cells - pos0)
        throw std::invalid_argument("prefill exceeds allocated context");
    // The inherited dense decoder has no multi-tile reducer yet. Fail before any state change.
    if (n > m.cap - pos0)
        throw std::invalid_argument("dense history exceeds validated attention staging capacity");
    if (!n) {
        err.clear();
        return true;
    }
    const auto start = Clock::now();
    const auto& g = *m.g;
    auto& ss = *m.ss;
    core::ScopedModelExecution exec_scope(ss.exec_state);
    auto cs = m.cs;
    std::optional<ScopedCudaDense> dense_scope;
    if (m.fns.has_cuda_dense)
        dense_scope.emplace(m.fns.cuda_dense);
    std::optional<ScopedCudaAttn> attn_scope;
    if (m.fns.has_cuda_attn)
        attn_scope.emplace(m.fns.cuda_attn);
    std::optional<ScopedCudaMoE> moe_scope;
    if (m.fns.has_cuda_moe)
        moe_scope.emplace(m.fns.cuda_moe);
    const int N = int(g.n_embd), C = int(g.ssm_conv_channels), V = int(g.ssm_value_dim), F = int(g.n_ff);
    const int NH = int(g.n_head), NK = int(g.n_head_kv), HD = int(g.head_dim), HV = int(g.ssm_v_heads),
              HK = int(g.ssm_k_heads);
    const int K = int(ss.k), NE = int(g.n_expert);
    QsaShapes s = qsa_real_shapes();
    s.n_head = NH;
    s.n_head_kv = NK;
    s.head_dim = HD;
    s.idx_n_head = 0;
    s.idx_dim = 0;
    const size_t state_floats = size_t(g.ssm_state_size) * HV * g.ssm_state_size;
    const size_t gdn_floats = state_floats + size_t(C) * (g.ssm_d_conv - 1);
    const auto& layout = m.src ? m.src->layout() : kernels::cpu::expert_layout();
    const double gu0 = m.pool->ms_multi_gu, quant0 = m.pool->ms_multi_q, down0 = m.pool->ms_multi_down;
    for (int64_t c0 = 0; c0 < n; c0 += batch_size_) {
        if (should_stop && should_stop()) {
            ck(cudaStreamSynchronize(cs));
            err = "cancelled";
            return false;
        }
        const int T = int(std::min<int64_t>(batch_size_, n - c0));
        const int64_t p0 = pos0 + c0;
        auto a = Clock::now();
        for (int t = 0; t < T; ++t) {
            m.htokens[t] = int32_t(tokens[c0 + t]);
            qsa_step_fill(m.hsteps + t * kStepCount, p0 + t, s);
            std::fill_n(m.hpositions + t * NH, NH, int32_t(p0 + t));
            std::fill_n(m.hpositions + T * NH + t * NK, NK, int32_t(p0 + t));
        }
        *m.hkeep = T;
        ck(cudaMemcpyAsync(m.tokens, m.htokens, T * 4, cudaMemcpyHostToDevice, cs));
        ck(cudaMemcpyAsync(m.steps, m.hsteps, T * kStepCount * 4, cudaMemcpyHostToDevice, cs));
        ck(cudaMemcpyAsync(m.positions, m.hpositions, T * m.pos_stride * 4, cudaMemcpyHostToDevice, cs));
        ck(cudaMemcpyAsync(m.keep, m.hkeep, 4, cudaMemcpyHostToDevice, cs));
        if (const auto* emb = core::native_embed())
            emb->gather_dev(m.tokens, T, m.R, cs);
        else
            for (int t = 0; t < T; ++t)
                if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.R + size_t(t) * N, cs, err))
                    return false;
        stats_.ms_embed += ms(a);
        int qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (should_stop && should_stop()) {
                ck(cudaStreamSynchronize(cs));
                err = "cancelled";
                return false;
            }
            const core::LayerView lv(*m.wt, l);
            a = Clock::now();
            m.stamp(0);
            m.norm(m.R, need(lv, "attn_norm.weight"), m.mixed, N, T);
            quant(m.fns, m.mixed, m.xq, N, T, cs);
            if (!core::is_qsa_layer(g, l)) {
                project(m.fns, need(lv, "attn_qkv.weight"), m.xq, m.qkv, T, cs);
                project(m.fns, need(lv, "attn_gate.weight"), m.xq, m.z, T, cs);
                m.stamp(1);
                auto* state = ss.gdn_state + size_t(gi - ss.gdn_ord0) * gdn_floats;
                auto* conv = state + state_floats;
                m.fns.gdn_conv(conv, m.qkv, static_cast<const float*>(need(lv, "ssm_conv1d.weight").data), m.h, C,
                               2 * HK, kEps, T, cs);
                m.fns.gdn_ab(m.mixed, static_cast<const uint16_t*>(need(lv, "ssm_alpha.weight").data),
                             static_cast<const uint16_t*>(need(lv, "ssm_beta.weight").data),
                             static_cast<const float*>(need(lv, "ssm_dt.bias").data),
                             static_cast<const float*>(need(lv, "ssm_a").data), m.gate, m.beta, N, HV, T, cs);
                m.fns.gdn_recurrence(state, m.h, C, m.gate, m.beta, m.z,
                                     static_cast<const float*>(need(lv, "ssm_norm.weight").data), kEps, m.y, HK, HV, T,
                                     m.keep, g.gdn_gate_silu, cs);
                gdn_conv_commit(conv, m.qkv, C, m.keep, cs);
                m.stamp(2);
                quant(m.fns, m.y, m.xq, V, T, cs);
                project(m.fns, need(lv, "ssm_out.weight"), m.xq, m.bo, T, cs);
                ++gi;
            } else {
                project(m.fns, need(lv, "attn_q.weight"), m.xq, m.qfull, T, cs);
                project(m.fns, need(lv, "attn_k.weight"), m.xq, m.k, T, cs);
                project(m.fns, need(lv, "attn_v.weight"), m.xq, m.v, T, cs);
                m.stamp(1);
                const auto& st = ss.qsa_states[qi];
                // Append current chunk first; each query's width enforces its own causal boundary.
                ck(cudaMemcpy2DAsync(m.q, HD * 4, m.qfull, HD * 8, HD * 4, size_t(T) * NH, cudaMemcpyDeviceToDevice,
                                     cs));
                m.norm(m.q, need(lv, "attn_q_norm.weight"), m.q, HD, T * NH);
                m.norm(m.k, need(lv, "attn_k_norm.weight"), m.k, HD, T * NK);
                m.fns.rope_apply(m.q, m.q, T * NH, HD, int(s.n_rot), rope_scaling(), m.positions, cs);
                m.fns.rope_apply(m.k, m.k, T * NK, HD, int(s.n_rot), rope_scaling(), m.positions + T * NH, cs);
                if (st.kv_rot) {
                    fwht256_inplace_cuda(m.k, int64_t(T) * NK, cs);
                    fwht256_inplace_cuda(m.v, int64_t(T) * NK, cs);
                    fwht256_inplace_cuda(m.q, int64_t(T) * NH, cs);
                } else if (st.kv_hybrid) {
                    fwht256_inplace_cuda(m.v, int64_t(T) * NK, cs);
                }
                if (st.kv_mode == 3 && !st.shares_staging) {
                    // Claim new staging slots before kv_append_step so new cells are written directly into VRAM
                    // and st.host instead of round-tripping through st.host over PCIe.
                    core::qsa_kv_resolve(st, g, m.ident, m.steps + (T - 1) * kStepCount, 1, m.cap, cs);
                }
                if (!st.kv_hybrid && st.kv_q4 && HD == 256) {
                    kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.k, m.v, s, cs, &st.host);
                } else if (!st.kv_hybrid && !st.kv_q4 && NK == 2 && HD == 256) {
                    prefill::kv_append(m.k, m.v, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                       st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, cs,
                                       &st.host);
                } else {
                    for (int t = 0; t < T; ++t) {
                        float* kt = m.k + size_t(t) * NK * HD;
                        float* vt = m.v + size_t(t) * NK * HD;
                        const int32_t* step_t = m.steps + t * kStepCount;
                        if (st.kv_hybrid) {
                            kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t, kt, kt, s,
                                              cs, nullptr);
                            kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vt, vt, s, cs, nullptr);
                        } else if (st.kv_q4) {
                            kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kt, vt, s, cs, &st.host);
                        } else if (st.kv_int8) {
                            kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t, kt, vt, s,
                                              cs, &st.host);
                        } else {
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kt, vt, s, cs, &st.host);
                        }
                    }
                }
                if (!(st.kv_mode == 3 && !st.shares_staging)) {
                    core::qsa_kv_resolve(st, g, m.ident, m.steps + (T - 1) * kStepCount, 1, m.cap, cs);
                }
                for (int t = 0; t < T; t += kAttentionQueries) {
                    const int count = std::min(kAttentionQueries, T - t);
                    const int active_cap = std::max(1, int(p0 + t + count));
                    qsa_decode_attn_batch(m.q + size_t(t) * NH * HD, core::qsa_attn_pools(st), m.ident,
                                          m.steps + t * kStepCount, active_cap, s, m.scratch,
                                          m.attn + size_t(t) * NH * HD, count, cs, 0);
                }
                if (st.kv_rot || st.kv_hybrid)
                    fwht256_inplace_cuda(m.attn, int64_t(T) * NH, cs);
                native_qsa_gate_apply(m.attn, m.qfull, m.attn, T * NH, HD, cs);
                m.stamp(2);
                quant(m.fns, m.attn, m.xq, NH * HD, T, cs);
                project(m.fns, need(lv, "attn_output.weight"), m.xq, m.bo, T, cs);
                ++qi;
            }
            add_inplace(m.R, m.bo, size_t(T) * N, cs);
            m.stamp(3);
            stats_.ms_mixer += ms(a); // host submission + waits below; not a GPU-duration estimate
            a = Clock::now();
            m.norm(m.R, need(lv, "post_attention_norm.weight"), m.mixed, N, T);
            m.fns.router_projection(m.mixed, static_cast<const uint16_t*>(need(lv, "ffn_gate_inp.weight").data),
                                    m.logits, N, NE, T, cs);
            router_top10(m.logits, T, NE, K, m.ids, m.weights, cs);
            ck(cudaMemcpyAsync(m.hact, m.mixed, size_t(T) * N * 4, cudaMemcpyDeviceToHost, cs));
            ck(cudaMemcpyAsync(m.hids, m.ids, size_t(T) * K * 4, cudaMemcpyDeviceToHost, cs));
            ck(cudaMemcpyAsync(m.hweights, m.weights, size_t(T) * K * 4, cudaMemcpyDeviceToHost, cs));
            m.stamp(4);
            const auto wait_start = Clock::now();
            ck(cudaStreamSynchronize(cs));
            const double wait_ms = ms(wait_start);
            stats_.host_wait_ms += wait_ms;
            const double dense_ms = m.elapsed(0, 1) + m.elapsed(2, 3) + m.elapsed(3, 4), mixer_ms = m.elapsed(1, 2);
            stats_.gpu_dense_ms += dense_ms;
            if (core::is_qsa_layer(g, l))
                stats_.gpu_attention_ms += mixer_ms;
            else
                stats_.gpu_gdn_ms += mixer_ms;
            NativeSharedWeights sw;
            const auto& wg = need(lv, "ffn_gate_shexp.weight");
            const auto& wu = need(lv, "ffn_up_shexp.weight");
            const auto& wd = need(lv, "ffn_down_shexp.weight");
            sw.gate_type = wg.native_type;
            sw.gate_data = wg.native_data;
            sw.up_type = wu.native_type;
            sw.up_data = wu.native_data;
            sw.down_type = wd.native_type;
            sw.down_data = wd.native_data;
            sw.q8_1 = m.xq;
            shared_expert_multi(T, m.mixed, m.bf, sw,
                                static_cast<const uint16_t*>(need(lv, "ffn_gate_inp_shexp.weight").data), m.sg, m.su,
                                m.scalar, m.shared, N, F, cs);
            std::string e;
            if (!core::validate_finite(m.hweights, size_t(T) * K, "prefill routing weights", e))
                throw std::runtime_error(e);
            // Group by expert, but write every result to its original token/routing rank.
            const auto group_start = Clock::now();
            for (auto& list : m.entries)
                list.clear();
            m.jobs.clear();
            m.job_nact.clear();
            m.job_out.clear();
            if (m.job_nact.capacity() < size_t(T * K)) {
                m.job_nact.reserve(size_t(T * K));
                m.job_out.reserve(size_t(T * K));
            }
            const auto& fmt = layout.fmt[size_t(l)];
            if (!m.pool->quant_act_multi_native(fmt, m.hact, m.act.data(), T)) {
                if (!core::validate_finite(m.hact, size_t(T) * N, "prefill expert input", e))
                    throw std::runtime_error(e);
                throw std::runtime_error("prefill expert input: nonfinite value");
            }
            for (int i = 0; i < T * K; ++i) {
                if (m.hids[i] < 0 || m.hids[i] >= NE)
                    throw std::runtime_error("prefill routed expert ID outside vocabulary");
                m.entries[size_t(m.hids[i])].push_back(i);
            }
            m.src->begin_layer(l, m.hids, T * K);
            for (int expert = 0; expert < NE; ++expert) {
                const auto& list = m.entries[size_t(expert)];
                if (list.empty())
                    continue;
                const int slot = m.cache ? m.cache->slot_of(l, expert) : -1;
                const uint8_t* blob = nullptr;
                if (slot >= 0)
                    blob = m.cache->device_slot(slot);
                else if (m.src->transient(l, expert)) {
                    // A file source's staging leases can expire within a large prompt layer.
                    // Own each active file expert until every job finishes.
                    auto& storage = m.file_storage[size_t(expert)];
                    storage.resize(fmt.bytes);
                    if (m.src->copy_blob(l, expert, storage.data()))
                        blob = storage.data();
                } else
                    blob = m.src->blob(l, expert);
                if (!blob)
                    throw std::runtime_error("prefill: missing routed expert blob");
                if (slot >= 0) {
                    for (size_t b = 0; b < list.size(); b += kernels::cpu::MAXT) {
                        const int nt = int(std::min<size_t>(kernels::cpu::MAXT, list.size() - b));
                        // Native cached matrices keep their original quantization; no whole expert dequantization.
                        for (int j = 0; j < nt; ++j)
                            ck(cudaMemcpyAsync(m.gx + size_t(j) * N, m.mixed + size_t(list[b + j] / K) * N, N * 4,
                                               cudaMemcpyDeviceToDevice, cs));
                        quant(m.fns, m.gx, m.gq, N, nt, cs);
                        m.fns.mmvq(fmt.gu_type, blob, m.gq, m.gg, N, F, nt, cs);
                        m.fns.mmvq(fmt.gu_type, blob + fmt.up_off, m.gq, m.gu, N, F, nt, cs);
                        native_swiglu_quantize_q8_1(m.gg, m.gu, m.gq, F, nt, cs);
                        m.fns.mmvq(fmt.d_type, blob + fmt.down_off, m.gq, m.go, F, N, nt, cs);
                        for (int j = 0; j < nt; ++j)
                            ck(cudaMemcpyAsync(m.parts + size_t(list[b + j]) * N, m.go + size_t(j) * N, N * 4,
                                               cudaMemcpyDeviceToDevice, cs));
                        ++stats_.gpu_groups;
                        ++stats_.expert_groups;
                    }
                } else {
                    for (size_t b = 0; b < list.size(); b += kernels::cpu::kMaxMultiExt) {
                        const int nt = int(std::min<size_t>(kernels::cpu::kMaxMultiExt, list.size() - b));
                        kernels::cpu::ExpertJobMulti job;
                        job.blob = blob;
                        job.nt = nt;
                        if (nt <= kernels::cpu::MAXT) {
                            for (int j = 0; j < nt; ++j) {
                                const int i = list[b + j];
                                job.nact[j] = m.act.data() + size_t(i / K) * kernels::cpu::kNativeActBytes;
                                job.out[j] = m.hparts + size_t(i) * N;
                            }
                        } else {
                            const size_t base = m.job_nact.size();
                            for (int j = 0; j < nt; ++j) {
                                const int i = list[b + j];
                                m.job_nact.push_back(m.act.data() + size_t(i / K) * kernels::cpu::kNativeActBytes);
                                m.job_out.push_back(m.hparts + size_t(i) * N);
                            }
                            job.nact_ext = m.job_nact.data() + base;
                            job.out_ext = m.job_out.data() + base;
                        }
                        m.jobs.push_back(job);
                        stats_.expert_bytes += fmt.bytes;
                        ++stats_.expert_groups;
                    }
                }
            }
            std::stable_sort(m.jobs.begin(), m.jobs.end(),
                             [](const kernels::cpu::ExpertJobMulti& a, const kernels::cpu::ExpertJobMulti& b) {
                                 return a.nt > b.nt;
                             });
            const double group_ms = ms(group_start);
            stats_.cpu_group_ms += group_ms;
            const auto expert_start = m.profile_file ? Clock::now() : Clock::time_point{};
            m.pool->run_split_multi_native(fmt, m.jobs.data(), int(m.jobs.size()), true);
            const double expert_ms = m.profile_file ? ms(expert_start) : 0;
            const auto combine_start = m.profile_file ? Clock::now() : Clock::time_point{};
            uint64_t h2d_bytes = 0;
            // CPU row writes never overlap resident GPU rows. Preserve rank order, not expert-ID sum order.
            if (!m.cache || !m.cache->valid()) {
                if (!m.pool->combine_multi_native(m.hparts, m.hweights, m.hact, T, K, N)) {
                    if (!core::validate_finite(m.hparts, size_t(T) * K * N, "prefill expert output", e))
                        throw std::runtime_error(e);
                    throw std::runtime_error("prefill expert output: nonfinite value");
                }
                ck(cudaMemcpyAsync(m.bo, m.hact, size_t(T) * N * 4, cudaMemcpyHostToDevice, cs));
                h2d_bytes = uint64_t(T) * N * 4;
                add_inplace(m.bo, m.shared, size_t(T) * N, cs);
                add_inplace(m.R, m.bo, size_t(T) * N, cs);
            } else {
                for (int i = 0; i < T * K; ++i)
                    if (m.cache->slot_of(l, m.hids[i]) < 0) {
                        if (!core::validate_finite(m.hparts + size_t(i) * N, N, "prefill expert output", e))
                            throw std::runtime_error(e);
                        ck(cudaMemcpyAsync(m.parts + size_t(i) * N, m.hparts + size_t(i) * N, N * 4,
                                           cudaMemcpyHostToDevice, cs));
                        h2d_bytes += uint64_t(N) * 4;
                    }
                native_moe_combine_multi(m.parts, m.weights, m.shared, m.bo, N, K, T, cs);
                add_inplace(m.R, m.bo, size_t(T) * N, cs);
                ck(cudaStreamSynchronize(cs));
            }
            if (on_layer) {
                ck(cudaStreamSynchronize(cs));
                on_layer(p0, l, T, m.R, m.ids, m.weights);
            }
            stats_.activation_bytes += uint64_t(T) * (N + 2 * K) * 4 + h2d_bytes;
            stats_.ms_moe += ms(a);
            if (m.profile_file) {
                const double combine_ms = ms(combine_start);
                std::fprintf(m.profile_file, "%lld,%lld,%lld,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%llu,%llu\n",
                             (long long) n, (long long) p0, (long long) l, T, dense_ms, mixer_ms, expert_ms, combine_ms,
                             group_ms, wait_ms, (unsigned long long) (uint64_t(T) * (N + 2 * K) * 4),
                             (unsigned long long) h2d_bytes);
                std::fflush(m.profile_file);
            }
        }
        m.finite(m.R, size_t(T) * N, "prefill residual");
        ++stats_.chunks_processed;
    }
    ck(cudaStreamSynchronize(m.cs));
    stats_.prompt_tokens = n;
    stats_.ms_total = ms(start);
    stats_.tok_per_sec = n / (stats_.ms_total / 1000);
    stats_.ms_cpu_gu = m.pool->ms_multi_gu - gu0;
    stats_.ms_cpu_quant = m.pool->ms_multi_q - quant0;
    stats_.ms_cpu_down = m.pool->ms_multi_down - down0;
    err.clear();
    return true;
} catch (const std::exception& e) {
    if (impl_->cs)
        cudaStreamSynchronize(impl_->cs);
    err = std::string("batched prefill: ") + e.what();
    return false;
}
} // namespace guild::runtime
