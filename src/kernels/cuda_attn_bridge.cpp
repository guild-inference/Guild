// Orchestration-only adapters: no CUDA kernels or CPU numerical fallback.
#include "guild/kernels/cuda_attn.hpp"
#include "guild/kernels/rope_scaling.hpp"
#include "guild/kernels/mrope.hpp"
#include "guild/kernels/rope.hpp"
#include "guild/kernels/native_rope.hpp"
#include "guild/kernels/native_qsa.hpp"
#include "guild/kernels/gdn.hpp"
#include "guild/kernels/fused_gdn.hpp"
#include "guild/kernels/native_gdn.hpp"
#include "guild/kernels/native_gdn_preprocess.hpp"
#include "guild/kernels/gdn_multi.hpp"
#include "guild/kernels/qsa.hpp"
#include "guild/kernels/qsa_decode_attn.hpp"
#include "guild/kernels/qsa_prompt_attn.hpp"
#include "guild/kernels/qsa_select.hpp"
#include "guild/kernels/native_flash_attn.hpp"
#include "guild/kernels/native_qsa_indexer.hpp"
#include "guild/kernels/native_qsa_score.hpp"
#include "guild/kernels/kv_q8.hpp"
#include "guild/kernels/kv_q4.hpp"
#include "guild/kernels/kv_stream.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#if GUILD_BUNDLED_CUDA_ATTN
extern "C" const guild_cuda_attn_ext_v1_t* guild_cuda_attn_table_v1();
#endif

namespace guild::kernels {
namespace {

thread_local const guild_cuda_attn_ext_v1_t* active = nullptr;

std::atomic<bool> g_native_qsa{false};
std::atomic<bool> g_native_rope{false};
std::atomic<bool> g_native_gdn{false};
std::atomic<bool> g_native_qsa_indexer{false};
std::atomic<bool> g_native_qsa_score{false};
RopeScaling g_rope_scaling{};

constexpr int kMropeDevices = 64;
std::atomic<const int32_t*> g_mrope_tab[kMropeDevices] = {};

struct RopeReg {
    RopeTab tab;
    RopeScaling scaling;
};
RopeReg g_rope_tab[kMropeDevices] = {};

int current_dev() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kMropeDevices) d = 0;
    return d;
}

bool same_scaling(const RopeScaling& a, const RopeScaling& b) {
    return a.type == b.type && a.freq_base == b.freq_base && a.factor == b.factor &&
           a.freq_scale_in == b.freq_scale_in && a.orig_ctx == b.orig_ctx &&
           a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}

bool rope_table_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("GUILD_ROPE_TABLE");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

guild_exec_context_v1_t context(void* stream) {
    guild_exec_context_v1_t c{};
    c.struct_size = sizeof(c);
    c.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    c.backend = GUILD_BACKEND_CUDA;
    c.device_id = 0;
    c.device_stream = stream;
    c.sync_mode = GUILD_SYNC_STREAM_ORDERED;
    return c;
}

void checked(guild_status_t result, const guild_error_info_t& error) {
    if (result == GUILD_STATUS_OK) return;
    if (result == GUILD_STATUS_ERR_INVALID_ARGUMENT || result == GUILD_STATUS_ERR_ABI_MISMATCH)
        throw std::invalid_argument(error.message);
    throw std::runtime_error(error.message[0] ? error.message : "CUDA attention backend execution failed");
}

guild_cuda_rope_scaling_v1_t from_scaling(const RopeScaling& s) {
    guild_cuda_rope_scaling_v1_t out{};
    out.struct_size = sizeof(out);
    switch (s.type) {
    case RopeScalingType::None: out.scaling_type = GUILD_CUDA_ROPE_SCALING_NONE_V1; break;
    case RopeScalingType::Linear: out.scaling_type = GUILD_CUDA_ROPE_SCALING_LINEAR_V1; break;
    case RopeScalingType::YaRN: out.scaling_type = GUILD_CUDA_ROPE_SCALING_YARN_V1; break;
    }
    out.freq_base = s.freq_base;
    out.factor = s.factor;
    out.freq_scale_in = s.freq_scale_in;
    out.orig_ctx = s.orig_ctx;
    out.ext_factor = s.ext_factor;
    out.attn_factor = s.attn_factor;
    out.beta_fast = s.beta_fast;
    out.beta_slow = s.beta_slow;
    return out;
}

guild_cuda_qsa_shapes_v1_t from_qsa_shapes(const QsaShapes& s) {
    guild_cuda_qsa_shapes_v1_t out{};
    out.struct_size = sizeof(out);
    out.n_head = s.n_head;
    out.n_head_kv = s.n_head_kv;
    out.head_dim = s.head_dim;
    out.n_rot = s.n_rot;
    out.idx_n_head = s.idx_n_head;
    out.idx_dim = s.idx_dim;
    out.idx_block = s.idx_block;
    out.idx_top_k = s.idx_top_k;
    out.page_size = s.page_size;
    return out;
}

guild_cuda_gdn_shapes_v1_t from_gdn_shapes(const GdnShapes& s) {
    guild_cuda_gdn_shapes_v1_t out{};
    out.struct_size = sizeof(out);
    out.state_size = s.S;
    out.h_k = s.h_k;
    out.h_v = s.h_v;
    return out;
}

guild_cuda_kv_host_pools_v1_t from_host_pools(const KvHostPools& p) {
    guild_cuda_kv_host_pools_v1_t out{};
    out.struct_size = sizeof(out);
    out.k_pool = p.k_pool;
    out.v_pool = p.v_pool;
    out.k_q = p.k_q;
    out.v_q = p.v_q;
    out.k_scale = p.k_scale;
    out.v_scale = p.v_scale;
    out.k_q4 = p.k_q4;
    out.v_q4 = p.v_q4;
    return out;
}

guild_cuda_qsa_attn_pools_v1_t from_attn_pools(const QsaAttnPools& p) {
    guild_cuda_qsa_attn_pools_v1_t out{};
    out.struct_size = sizeof(out);
    out.k_pool = p.k_pool;
    out.v_pool = p.v_pool;
    out.k_q = p.k_q;
    out.v_q = p.v_q;
    out.k_scale = p.k_scale;
    out.v_scale = p.v_scale;
    out.k_q4 = p.k_q4;
    out.v_q4 = p.v_q4;
    out.page_table = p.page_table;
    return out;
}

guild_cuda_kv_stream_map_v1_t from_stream_map(const KvStreamMap& m) {
    guild_cuda_kv_stream_map_v1_t out{};
    out.struct_size = sizeof(out);
    out.page_table = m.page_table;
    out.slot_block = m.slot_block;
    out.slot_stamp = m.slot_stamp;
    out.slot_ref = m.slot_ref;
    out.ctl = m.ctl;
    out.miss_block = m.miss_block;
    out.miss_slot = m.miss_slot;
    out.n_blocks = m.n_blocks;
    out.n_slots = m.n_slots;
    return out;
}

guild_cuda_qsa_indexer_buffers_v1_t from_indexer_buffers(const QsaIndexerBuffers& b) {
    guild_cuda_qsa_indexer_buffers_v1_t out{};
    out.struct_size = sizeof(out);
    out.tail = b.tail;
    out.dead = b.dead;
    out.pooled = b.pooled;
    out.block_pos = b.block_pos;
    return out;
}

const guild_cuda_attn_ext_v1_t* try_backend() {
    if (active) return active;
#if GUILD_BUNDLED_CUDA_ATTN
    return guild_cuda_attn_table_v1();
#else
    return nullptr;
#endif
}

const guild_cuda_attn_ext_v1_t& backend() {
    if (const auto* b = try_backend()) return *b;
    throw std::runtime_error("CUDA attention/recurrent provider missing; bundled attention arithmetic is disabled; install a CDA1 CUDA .gkernel");
}

void sync_state(const guild_cuda_attn_ext_v1_t& tbl) {
    const int d = current_dev();
    guild_cuda_attn_config_v1_t cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.actions = GUILD_CUDA_ATTN_CFG_SET_TOGGLES_V1 |
                  GUILD_CUDA_ATTN_CFG_SET_ROPE_SCALING_V1 |
                  GUILD_CUDA_ATTN_CFG_SET_MROPE_TABLE_V1;
    cfg.toggle_mask = GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1 |
                      GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1 |
                      GUILD_CUDA_ATTN_TOGGLE_NATIVE_GDN_V1 |
                      GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_INDEXER_V1 |
                      GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_SCORE_V1;
    if (g_native_qsa.load(std::memory_order_relaxed)) cfg.toggle_values |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1;
    if (g_native_rope.load(std::memory_order_relaxed)) cfg.toggle_values |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1;
    if (g_native_gdn.load(std::memory_order_relaxed)) cfg.toggle_values |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_GDN_V1;
    if (g_native_qsa_indexer.load(std::memory_order_relaxed)) cfg.toggle_values |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_INDEXER_V1;
    if (g_native_qsa_score.load(std::memory_order_relaxed)) cfg.toggle_values |= GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_SCORE_V1;
    cfg.mrope_table = g_mrope_tab[d].load(std::memory_order_relaxed);
    cfg.rope_scaling = from_scaling(g_rope_scaling);
    if (g_rope_tab[d].tab.cos != nullptr) {
        cfg.actions |= GUILD_CUDA_ATTN_CFG_SET_ROPE_TABLE_V1;
        cfg.rope_cos_tab = g_rope_tab[d].tab.cos;
        cfg.rope_sin_tab = g_rope_tab[d].tab.sin;
        cfg.rope_max_pos = g_rope_tab[d].tab.max_pos;
        cfg.rope_scaling = from_scaling(g_rope_tab[d].scaling);
    }
    const auto c = context(nullptr);
    guild_error_info_t e{};
    checked(tbl.configure(&c, &cfg, &e), e);
}

void push_toggle(uint32_t flag, bool value) {
    if (const auto* b = try_backend()) {
        guild_cuda_attn_config_v1_t cfg{};
        cfg.struct_size = sizeof(cfg);
        cfg.actions = GUILD_CUDA_ATTN_CFG_SET_TOGGLES_V1;
        cfg.toggle_mask = flag;
        cfg.toggle_values = value ? flag : 0u;
        const auto c = context(nullptr);
        guild_error_info_t e{};
        checked(b->configure(&c, &cfg, &e), e);
    }
}

}  // namespace

ScopedCudaAttn::ScopedCudaAttn(const ResolvedCudaAttn& binding) : previous_(active) {
    active = &binding.table;
    sync_state(*active);
}

ScopedCudaAttn::~ScopedCudaAttn() {
    active = previous_;
}

void native_qsa_set_enabled(bool enabled) {
    g_native_qsa.store(enabled, std::memory_order_relaxed);
    push_toggle(GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1, enabled);
}
bool native_qsa_enabled() { return g_native_qsa.load(std::memory_order_relaxed); }

void native_rope_set_enabled(bool enabled) {
    g_native_rope.store(enabled, std::memory_order_relaxed);
    push_toggle(GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1, enabled);
}
bool native_rope_enabled() { return g_native_rope.load(std::memory_order_relaxed); }

void native_gdn_set_enabled(bool enabled) {
    g_native_gdn.store(enabled, std::memory_order_relaxed);
    push_toggle(GUILD_CUDA_ATTN_TOGGLE_NATIVE_GDN_V1, enabled);
}
bool native_gdn_enabled() { return g_native_gdn.load(std::memory_order_relaxed); }

void native_qsa_indexer_set_enabled(bool enabled) {
    g_native_qsa_indexer.store(enabled, std::memory_order_relaxed);
    push_toggle(GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_INDEXER_V1, enabled);
}
bool native_qsa_indexer_enabled() { return g_native_qsa_indexer.load(std::memory_order_relaxed); }

void native_qsa_score_set_enabled(bool enabled) {
    g_native_qsa_score.store(enabled, std::memory_order_relaxed);
    push_toggle(GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_SCORE_V1, enabled);
}
bool native_qsa_score_enabled() { return g_native_qsa_score.load(std::memory_order_relaxed); }

void rope_scaling_set(const RopeScaling& scaling) {
    g_rope_scaling = scaling;
    if (const auto* b = try_backend()) {
        guild_cuda_attn_config_v1_t cfg{};
        cfg.struct_size = sizeof(cfg);
        cfg.actions = GUILD_CUDA_ATTN_CFG_SET_ROPE_SCALING_V1;
        cfg.rope_scaling = from_scaling(scaling);
        const auto c = context(nullptr);
        guild_error_info_t e{};
        checked(b->configure(&c, &cfg, &e), e);
    }
}
const RopeScaling& rope_scaling() { return g_rope_scaling; }

void mrope_table_set(const int32_t* device_table) {
    g_mrope_tab[current_dev()].store(device_table, std::memory_order_relaxed);
    if (const auto* b = try_backend()) {
        guild_cuda_attn_config_v1_t cfg{};
        cfg.struct_size = sizeof(cfg);
        cfg.actions = GUILD_CUDA_ATTN_CFG_SET_MROPE_TABLE_V1;
        cfg.mrope_table = device_table;
        const auto c = context(nullptr);
        guild_error_info_t e{};
        checked(b->configure(&c, &cfg, &e), e);
    }
}
const int32_t* mrope_table() { return g_mrope_tab[current_dev()].load(std::memory_order_relaxed); }

void rope_table_set(const float* cos_tab, const float* sin_tab, int max_pos, const RopeScaling& scaling) {
    g_rope_tab[current_dev()] = RopeReg{RopeTab{cos_tab, sin_tab, max_pos}, scaling};
    if (const auto* b = try_backend()) {
        guild_cuda_attn_config_v1_t cfg{};
        cfg.struct_size = sizeof(cfg);
        cfg.actions = GUILD_CUDA_ATTN_CFG_SET_ROPE_TABLE_V1;
        cfg.rope_cos_tab = cos_tab;
        cfg.rope_sin_tab = sin_tab;
        cfg.rope_max_pos = max_pos;
        cfg.rope_scaling = from_scaling(scaling);
        const auto c = context(nullptr);
        guild_error_info_t e{};
        checked(b->configure(&c, &cfg, &e), e);
    }
}

void rope_table_release(const float* cos_tab) {
    for (RopeReg& r : g_rope_tab) {
        if (cos_tab != nullptr && r.tab.cos == cos_tab) r = RopeReg{};
    }
    if (const auto* b = try_backend()) {
        guild_cuda_attn_config_v1_t cfg{};
        cfg.struct_size = sizeof(cfg);
        cfg.actions = GUILD_CUDA_ATTN_CFG_RELEASE_ROPE_TABLE_V1;
        cfg.rope_cos_tab = cos_tab;
        const auto c = context(nullptr);
        guild_error_info_t e{};
        checked(b->configure(&c, &cfg, &e), e);
    }
}

RopeTab rope_table_for(const RopeScaling& scaling) {
    if (!rope_table_enabled()) return {};
    const RopeReg& r = g_rope_tab[current_dev()];
    return r.tab.cos != nullptr && same_scaling(r.scaling, scaling) ? r.tab : RopeTab{};
}

// Host float64 table builder for RoPE cos/sin arrays (no GPU kernel or CPU fallback).
void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;
            const double interp = fs * extrap;
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
    }
}

void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot,
                     const float* cos_tab, const float* sin_tab, const int* pos, void* stream) {
    guild_cuda_rope_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_ROPE_NEOX_APPLY_V1;
    p.x = x;
    p.out = out;
    p.cos_tab = cos_tab;
    p.sin_tab = sin_tab;
    p.positions = pos;
    p.rows = rows;
    p.head_dim = head_dim;
    p.n_rot = n_rot;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().rope(&c, &p, &e), e);
}

void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, const RopeScaling& scaling, const int* positions, void* stream) {
    guild_cuda_rope_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_ROPE_NATIVE_APPLY_V1;
    p.x = x;
    p.out = out;
    p.positions = positions;
    p.rows = rows;
    p.head_dim = head_dim;
    p.n_rot = n_rot;
    p.scaling = from_scaling(scaling);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().rope(&c, &p, &e), e);
}

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    guild_cuda_qsa_norm_gate_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_RMS_NORM_WEIGHTED_V1;
    p.x = input;
    p.aux = gamma;
    p.out = output;
    p.cols_or_head_dim = n_cols;
    p.rows_or_n_head = n_rows;
    p.epsilon = epsilon;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_norm_gate(&c, &p, &e), e);
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream) {
    guild_cuda_qsa_norm_gate_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_GATE_APPLY_NATIVE_V1;
    p.x = attn;
    p.aux = q_full;
    p.out = output;
    p.rows_or_n_head = n_head;
    p.cols_or_head_dim = head_dim;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_norm_gate(&c, &p, &e), e);
}

void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    guild_cuda_qsa_norm_gate_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_GATE_APPLY_F16_V1;
    p.x = attn;
    p.aux = q_full;
    p.out = out;
    p.shapes = from_qsa_shapes(s);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_norm_gate(&c, &p, &e), e);
}

void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    guild_cuda_qsa_norm_gate_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_GATE_APPLY_F32_V1;
    p.x = attn;
    p.aux = q_full;
    p.out = out;
    p.shapes = from_qsa_shapes(s);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_norm_gate(&c, &p, &e), e);
}

void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const GdnShapes& s, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_STEP_V1;
    p.state = state;
    p.x = q;
    p.k = k;
    p.v = v;
    p.gate = gate;
    p.beta = beta;
    p.out = o;
    p.shapes = from_gdn_shapes(s);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out,
                   int64_t channels, int64_t d_conv, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_CONV_STEP_V1;
    p.state = conv_state;
    p.x = x;
    p.k = kW;
    p.out = out;
    p.channels_or_rows = channels;
    p.cols_or_d_conv = d_conv;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_L2_NORM_V1;
    p.state = x;
    p.channels_or_rows = rows;
    p.cols_or_d_conv = cols;
    p.epsilon = eps;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_beta_gate(float* beta, int64_t h_v, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_BETA_GATE_V1;
    p.state = beta;
    p.channels_or_rows = h_v;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y,
                  int64_t h_v, int64_t S, float eps, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_OUT_NORM_V1;
    p.x = o;
    p.z = z;
    p.gamma = ssm_norm;
    p.out = y;
    p.channels_or_rows = h_v;
    p.cols_or_d_conv = S;
    p.epsilon = eps;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h,
                       int channels, int qk_heads, float eps, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_FUSED_CONV_L2_V1;
    p.state = history;
    p.x = qkv;
    p.k = conv_w;
    p.out = h;
    p.channels_or_rows = channels;
    p.qk_heads_or_h_k = qk_heads;
    p.epsilon = eps;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta,
                  const float* dt, const float* ssm_a, float* gate, float* beta,
                  int n_embd, int h_v, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_FUSED_AB_V1;
    p.x = x;
    p.w_alpha = w_alpha;
    p.w_beta = w_beta;
    p.k = dt;
    p.v = ssm_a;
    p.out = gate;
    p.aux_out = beta;
    p.channels_or_rows = n_embd;
    p.h_v = h_v;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v,
                         const float* gate, const float* beta, const float* z, const float* gamma,
                         float eps, float* y, int h_k, int h_v, void* stream, bool gate_silu) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_FUSED_STEP_NORM_V1;
    p.state = state;
    p.x = q;
    p.k = k;
    p.v = v;
    p.gate = gate;
    p.beta = beta;
    p.z = z;
    p.gamma = gamma;
    p.epsilon = eps;
    p.out = y;
    p.qk_heads_or_h_k = h_k;
    p.h_v = h_v;
    p.gate_silu = gate_silu ? 1u : 0u;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shapes, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_NATIVE_STEP_V1;
    p.state = state;
    p.x = q;
    p.k = k;
    p.v = v;
    p.gate = gate;
    p.beta = beta;
    p.out = output;
    p.shapes = from_gdn_shapes(shapes);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void native_gdn_conv_silu(float* history, const float* input, const float* weights,
                          float* raw_output, float* silu_output, int64_t channels,
                          int64_t d_conv, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_NATIVE_CONV_SILU_V1;
    p.state = history;
    p.x = input;
    p.k = weights;
    p.aux_out = raw_output;
    p.out = silu_output;
    p.channels_or_rows = channels;
    p.cols_or_d_conv = d_conv;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_NATIVE_L2_NORM_V1;
    p.state = input;
    p.channels_or_rows = rows;
    p.cols_or_d_conv = cols;
    p.epsilon = epsilon;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_NATIVE_BETA_GATE_V1;
    p.state = beta;
    p.channels_or_rows = heads;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
                     float* gate, int64_t heads, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_NATIVE_GATE_V1;
    p.x = alpha;
    p.k = dt;
    p.v = ssm_a;
    p.out = gate;
    p.channels_or_rows = heads;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void native_gdn_out_norm(const float* output, const float* z, const float* gamma,
                         float* destination, int64_t heads, int64_t cols,
                         float epsilon, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_NATIVE_OUT_NORM_V1;
    p.x = output;
    p.z = z;
    p.gamma = gamma;
    p.out = destination;
    p.channels_or_rows = heads;
    p.cols_or_d_conv = cols;
    p.epsilon = epsilon;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_prefill_conv(const float* history, const float* qkv, const float* conv_w, float* h,
                      int channels, int qk_heads, float eps, int n_tok, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_PREFILL_CONV_V1;
    p.x = history;
    p.v = qkv;
    p.k = conv_w;
    p.out = h;
    p.channels_or_rows = channels;
    p.qk_heads_or_h_k = qk_heads;
    p.epsilon = eps;
    p.n_tok = n_tok;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_prefill_recurrence(float* state, const float* h, int channels, const float* gate,
                            const float* beta, const float* z, const float* gamma, float eps,
                            float* y, int h_k, int h_v, int n_tok, const int32_t* keep,
                            bool gate_silu, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_PREFILL_RECURRENCE_V1;
    p.state = state;
    p.x = h;
    p.channels_or_rows = channels;
    p.gate = gate;
    p.beta = beta;
    p.z = z;
    p.gamma = gamma;
    p.epsilon = eps;
    p.out = y;
    p.qk_heads_or_h_k = h_k;
    p.h_v = h_v;
    p.n_tok = n_tok;
    p.n_keep = keep;
    p.gate_silu = gate_silu ? 1u : 0u;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h,
                       int channels, int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_CONV_L2_MULTI_V1;
    p.x = history;
    p.v = qkv;
    p.k = conv_w;
    p.out = h;
    p.channels_or_rows = channels;
    p.qk_heads_or_h_k = qk_heads;
    p.epsilon = eps;
    p.n_tok = n_tok;
    p.t_begin = t_begin;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_CONV_COMMIT_V1;
    p.state = history;
    p.x = qkv;
    p.channels_or_rows = channels;
    p.n_keep = n_keep;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta,
                  const float* dt, const float* ssm_a, float* gate, float* beta,
                  int n_embd, int h_v, int n_tok, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_AB_MULTI_V1;
    p.x = x;
    p.w_alpha = w_alpha;
    p.w_beta = w_beta;
    p.k = dt;
    p.v = ssm_a;
    p.out = gate;
    p.aux_out = beta;
    p.channels_or_rows = n_embd;
    p.h_v = h_v;
    p.n_tok = n_tok;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_ab_prefill(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta,
                    const float* dt, const float* ssm_a, float* gate, float* beta,
                    int n_embd, int h_v, int n_tok, void* stream) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_AB_PREFILL_V1;
    p.x = x;
    p.w_alpha = w_alpha;
    p.w_beta = w_beta;
    p.k = dt;
    p.v = ssm_a;
    p.out = gate;
    p.aux_out = beta;
    p.channels_or_rows = n_embd;
    p.h_v = h_v;
    p.n_tok = n_tok;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps,
                         float* y, int h_k, int h_v, int n_tok, const int32_t* n_keep,
                         void* stream, int t_out_begin, bool gate_silu) {
    guild_cuda_gdn_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_GDN_STEP_NORM_MULTI_V1;
    p.state = state;
    p.x = h;
    p.channels_or_rows = conv_channels;
    p.gate = gate;
    p.beta = beta;
    p.z = z;
    p.gamma = gamma;
    p.epsilon = eps;
    p.out = y;
    p.qk_heads_or_h_k = h_k;
    p.h_v = h_v;
    p.n_tok = n_tok;
    p.n_keep = n_keep;
    p.t_begin = t_out_begin;
    p.gate_silu = gate_silu ? 1u : 0u;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().gdn(&c, &p, &e), e);
}

// Host step/scratch/block-size calculations (no GPU arithmetic).
void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s) {
    if (host_step == nullptr) return;
    if (pos < 0) {
        std::fprintf(stderr, "qsa: qsa_step_fill: pos < 0\n");
        std::exit(1);
    }
    const int64_t n_kv = pos + 1;
    host_step[kStepPos] = (int32_t) pos;
    host_step[kStepNKv] = (int32_t) n_kv;
    host_step[kStepNBid] = (int32_t) (n_kv / s.idx_block);
    host_step[kStepWidth] = (int32_t) qsa_selection_width(n_kv, s);
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + 64 - 1) / 64;
    return (uint64_t) chunks * (uint64_t) s.n_head * (256 + 2) + 64;
}

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos,
               const float* kcur, const float* vcur, const QsaShapes& s, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_APPEND_SYNC_V1;
    p.k_pool = k_pool;
    p.v_pool = v_pool;
    p.page_table = page_table;
    p.pos_or_b0 = pos;
    p.kcur = kcur;
    p.vcur = vcur;
    p.shapes = from_qsa_shapes(s);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                    const KvHostPools* host) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_APPEND_STEP_F16_V1;
    p.k_pool = k_pool;
    p.v_pool = v_pool;
    p.page_table = page_table;
    p.step = step;
    p.kcur = kcur;
    p.vcur = vcur;
    p.shapes = from_qsa_shapes(s);
    if (host) {
        p.has_host_pools = 1u;
        p.host_pools = from_host_pools(*host);
    }
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_GATHER_SYNC_V1;
    p.k_pool_ro = k_pool;
    p.v_pool_ro = v_pool;
    p.page_table = page_table;
    p.ids = ids;
    p.count_or_b1 = n_ids;
    p.shapes = from_qsa_shapes(s);
    p.k_scratch = k_scratch;
    p.v_scratch = v_scratch;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_GATHER_STEP_F16_V1;
    p.k_pool_ro = k_pool;
    p.v_pool_ro = v_pool;
    p.page_table = page_table;
    p.ids = ids;
    p.step = step;
    p.cap_or_n_slots = max_ids;
    p.shapes = from_qsa_shapes(s);
    p.k_scratch = k_scratch;
    p.v_scratch = v_scratch;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* step, const float* kcur,
                       const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_APPEND_STEP_Q8_V1;
    p.k_pool = k_q;
    p.v_pool = v_q;
    p.k_scale = k_scale;
    p.v_scale = v_scale;
    p.page_table = page_table;
    p.step = step;
    p.kcur = kcur;
    p.vcur = vcur;
    p.shapes = from_qsa_shapes(s);
    if (host) {
        p.has_host_pools = 1u;
        p.host_pools = from_host_pools(*host);
    }
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_GATHER_STEP_Q8_V1;
    p.k_pool_ro = k_q;
    p.v_pool_ro = v_q;
    p.k_scale_ro = k_scale;
    p.v_scale_ro = v_scale;
    p.page_table = page_table;
    p.ids = ids;
    p.step = step;
    p.cap_or_n_slots = max_ids;
    p.shapes = from_qsa_shapes(s);
    p.k_scratch = k_scratch;
    p.v_scratch = v_scratch;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_FWHT256_V1;
    p.kcur = src;
    p.fwht_dst = dst;
    p.count_or_b1 = n_rows;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_APPEND_STEP_Q4_V1;
    p.k_pool = k_q4;
    p.v_pool = v_q4;
    p.page_table = page_table;
    p.step = step;
    p.kcur = kcur;
    p.vcur = vcur;
    p.shapes = from_qsa_shapes(s);
    if (host) {
        p.has_host_pools = 1u;
        p.host_pools = from_host_pools(*host);
    }
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T,
                  const float* K, const float* V, const QsaShapes& s, void* stream,
                  const KvHostPools* host, const KvHostPools* stage) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_APPEND_BATCH_Q4_V1;
    p.k_pool = k_q4;
    p.v_pool = v_q4;
    p.page_table = page_table;
    p.pos_or_b0 = pos0;
    p.count_or_b1 = T;
    p.kcur = K;
    p.vcur = V;
    p.shapes = from_qsa_shapes(s);
    if (host) {
        p.has_host_pools = 1u;
        p.host_pools = from_host_pools(*host);
    }
    if (stage) {
        p.has_stage_pools = 1u;
        p.stage_pools = from_host_pools(*stage);
    }
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table,
                       const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                       uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_GATHER_STEP_Q4_V1;
    p.k_pool_ro = k_q4;
    p.v_pool_ro = v_q4;
    p.page_table = page_table;
    p.ids = ids;
    p.step = step;
    p.cap_or_n_slots = max_ids;
    p.shapes = from_qsa_shapes(s);
    p.k_scratch = k_scratch;
    p.v_scratch = v_scratch;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_STREAM_RESET_V1;
    p.stream_map = from_stream_map(m);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_stream_invalidate(const KvStreamMap& m, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_STREAM_INVALIDATE_V1;
    p.stream_map = from_stream_map(m);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap,
                       const QsaShapes& s, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_STREAM_RESOLVE_V1;
    p.stream_map = from_stream_map(m);
    p.attn_pools = from_attn_pools(slots);
    p.host_pools = from_host_pools(host);
    p.kv_format = fmt;
    p.ids = ids;
    p.step = steps;
    p.count_or_b1 = n_q;
    p.cap_or_n_slots = cap;
    p.shapes = from_qsa_shapes(s);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_RING_TABLE_V1;
    p.page_table_mut = page_table;
    p.count_or_b1 = n_blocks;
    p.cap_or_n_slots = n_slots;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_RING_RESTORE_V1;
    p.attn_pools = from_attn_pools(slots);
    p.host_pools = from_host_pools(host);
    p.kv_format = fmt;
    p.pos_or_b0 = b0;
    p.count_or_b1 = b1;
    p.cap_or_n_slots = n_slots;
    p.shapes = from_qsa_shapes(s);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_STAGE_FROM_HOST_V1;
    p.attn_pools = from_attn_pools(stage);
    p.host_pools = from_host_pools(host);
    p.kv_format = fmt;
    p.count_or_b1 = n_blocks;
    p.shapes = from_qsa_shapes(s);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

void kv_unstage_to_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                        const QsaShapes& s, void* stream) {
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_UNSTAGE_TO_HOST_V1;
    p.attn_pools = from_attn_pools(stage);
    p.host_pools = from_host_pools(host);
    p.kv_format = fmt;
    p.pos_or_b0 = b0;
    p.count_or_b1 = b1;
    p.shapes = from_qsa_shapes(s);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    guild_cuda_kv_stream_counters_v1_t cnt{};
    cnt.struct_size = sizeof(cnt);
    guild_cuda_kv_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_KV_STREAM_COUNTERS_V1;
    p.stream_map = from_stream_map(m);
    p.out_counters = &cnt;
    const auto c = context(nullptr);
    guild_error_info_t e{};
    checked(backend().kv(&c, &p, &e), e);
    KvStreamCounters r{};
    r.misses = cnt.misses;
    r.lookups = cnt.lookups;
    r.calls = cnt.calls;
    r.overflow = cnt.overflow != 0;
    return r;
}

void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_INDEXER_APPEND_V1;
    p.raw_or_pooled = raw;
    p.step_or_pos_dev = pos_dev;
    p.pos_base = pos_base;
    p.q_idx_or_gamma = w_k_norm;
    p.epsilon = eps;
    p.buffers = from_indexer_buffers(b);
    p.shapes = from_qsa_shapes(s);
    p.bias_or_cos = cos_tab;
    p.sin_tab = sin_tab;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device,
                               int32_t pos_base, const float* gamma, float epsilon,
                               const QsaIndexerBuffers& buffers, const QsaShapes& shapes,
                               int64_t max_cells, const RopeScaling& scaling, void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_INDEXER_NATIVE_APPEND_V1;
    p.raw_or_pooled = raw;
    p.step_or_pos_dev = relative_pos_device;
    p.pos_base = pos_base;
    p.q_idx_or_gamma = gamma;
    p.epsilon = epsilon;
    p.buffers = from_indexer_buffers(buffers);
    p.shapes = from_qsa_shapes(shapes);
    p.max_cells_or_cap = max_cells;
    p.scaling = from_scaling(scaling);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base,
                                     const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                                     const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling,
                                     void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_INDEXER_NATIVE_BATCH_V1;
    p.raw_or_pooled = raw;
    p.n_or_nq = n;
    p.p0_or_n_bid = p0;
    p.pos_base = pos_base;
    p.q_idx_or_gamma = gamma;
    p.epsilon = epsilon;
    p.buffers = from_indexer_buffers(b);
    p.shapes = from_qsa_shapes(s);
    p.max_cells_or_cap = max_cells;
    p.scaling = from_scaling(scaling);
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias,
               const QsaShapes& s, int64_t n_kv, float* cell_scores, void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_INDEX_SYNC_V1;
    p.raw_or_pooled = pooled;
    p.p0_or_n_bid = n_bid;
    p.q_idx_or_gamma = q_idx;
    p.bias_or_cos = bias;
    p.shapes = from_qsa_shapes(s);
    p.n_or_nq = n_kv;
    p.scores_out = cell_scores;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_INDEX_STEP_V1;
    p.raw_or_pooled = pooled;
    p.q_idx_or_gamma = q_idx;
    p.bias_or_cos = bias;
    p.shapes = from_qsa_shapes(s);
    p.step_or_pos_dev = step;
    p.max_blocks = max_blocks;
    p.scores_out = cell_scores;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void native_qsa_score(const float* pooled, const float* query, const float* block_bias,
                      const QsaShapes& shapes, const int32_t* step,
                      int64_t max_blocks, int64_t max_cells,
                      float* cell_scores, void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_NATIVE_SCORE_V1;
    p.raw_or_pooled = pooled;
    p.dead_or_query = query;
    p.bias_or_cos = block_bias;
    p.shapes = from_qsa_shapes(shapes);
    p.step_or_pos_dev = step;
    p.max_blocks = max_blocks;
    p.max_cells_or_cap = max_cells;
    p.scores_out = cell_scores;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap,
              int32_t* ids, void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_TOPK_512_SYNC_V1;
    p.raw_or_pooled = cell_scores;
    p.n_or_nq = n_kv;
    p.shapes = from_qsa_shapes(s);
    p.max_cells_or_cap = cap;
    p.ids_out = ids;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap,
                   const int32_t* step, int32_t* ids, void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_TOPK_512_STEP_V1;
    p.raw_or_pooled = cell_scores;
    p.shapes = from_qsa_shapes(s);
    p.max_cells_or_cap = cap;
    p.step_or_pos_dev = step;
    p.ids_out = ids;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx,
                      const int32_t* steps, int64_t nq, int64_t max_blocks,
                      const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_BLOCK_SCORES_V1;
    p.raw_or_pooled = pooled;
    p.dead_or_query = dead;
    p.q_idx_or_gamma = q_idx;
    p.step_or_pos_dev = steps;
    p.n_or_nq = nq;
    p.max_blocks = max_blocks;
    p.shapes = from_qsa_shapes(s);
    p.scores_out = scores;
    p.active_blocks = active_blocks;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx,
                         const int32_t* steps, int64_t nq, int64_t max_blocks,
                         const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    uint32_t launched = 0;
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_BLOCK_SCORES_TC_V1;
    p.raw_or_pooled = pooled;
    p.dead_or_query = dead;
    p.q_idx_or_gamma = q_idx;
    p.step_or_pos_dev = steps;
    p.n_or_nq = nq;
    p.max_blocks = max_blocks;
    p.shapes = from_qsa_shapes(s);
    p.scores_out = scores;
    p.active_blocks = active_blocks;
    p.out_launched = &launched;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
    return launched != 0;
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                    int64_t cap, const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_BLOCK_TOPK_V1;
    p.raw_or_pooled = scores;
    p.step_or_pos_dev = steps;
    p.n_or_nq = nq;
    p.max_blocks = max_blocks;
    p.max_cells_or_cap = cap;
    p.shapes = from_qsa_shapes(s);
    p.ids_out = ids;
    p.active_blocks = active_blocks;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                            int64_t cap, const QsaShapes& s, int32_t* ids, void* stream) {
    uint32_t launched = 0;
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_BLOCK_TOPK_CLUSTER_V1;
    p.raw_or_pooled = scores;
    p.step_or_pos_dev = steps;
    p.n_or_nq = nq;
    p.max_blocks = max_blocks;
    p.max_cells_or_cap = cap;
    p.shapes = from_qsa_shapes(s);
    p.ids_out = ids;
    p.out_launched = &launched;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
    return launched != 0;
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                        int64_t cap, const QsaShapes& s, int32_t* ids, void* stream) {
    guild_cuda_qsa_index_select_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_QSA_BLOCK_TOPK_REF_V1;
    p.raw_or_pooled = scores;
    p.step_or_pos_dev = steps;
    p.n_or_nq = nq;
    p.max_blocks = max_blocks;
    p.max_cells_or_cap = cap;
    p.shapes = from_qsa_shapes(s);
    p.ids_out = ids;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().qsa_index_select(&c, &p, &e), e);
}

void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& s, float* attn, float* weights, void* stream) {
    guild_cuda_attention_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_ATTN_QSA_SYNC_V1;
    p.q = q;
    p.k_scratch = k_scratch;
    p.v_scratch = v_scratch;
    p.cap_or_n_ids = n_ids;
    p.shapes = from_qsa_shapes(s);
    p.attn_out = attn;
    p.weights_out = weights;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().attention(&c, &p, &e), e);
}

void qsa_attend_step(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch,
                     const int32_t* step, int64_t max_ids, const QsaShapes& s,
                     float* attn, float* weights, void* stream) {
    guild_cuda_attention_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_ATTN_QSA_STEP_V1;
    p.q = q;
    p.k_scratch = k_scratch;
    p.v_scratch = v_scratch;
    p.steps = step;
    p.cap_or_n_ids = max_ids;
    p.shapes = from_qsa_shapes(s);
    p.attn_out = attn;
    p.weights_out = weights;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().attention(&c, &p, &e), e);
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids,
                          const int32_t* step, int64_t cap, const QsaShapes& s,
                          float* scratch, float* attn, void* stream) {
    guild_cuda_attention_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_ATTN_DECODE_STEP_V1;
    p.q = q;
    p.pools = from_attn_pools(pools);
    p.ids = ids;
    p.steps = step;
    p.cap_or_n_ids = cap;
    p.shapes = from_qsa_shapes(s);
    p.scratch = scratch;
    p.attn_out = attn;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().attention(&c, &p, &e), e);
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids,
                           const int32_t* steps, int64_t cap, const QsaShapes& s,
                           float* scratch, float* attn, int64_t n_q, void* stream, int64_t id_stride) {
    guild_cuda_attention_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_ATTN_DECODE_BATCH_V1;
    p.q = q;
    p.pools = from_attn_pools(pools);
    p.ids = ids;
    p.steps = steps;
    p.cap_or_n_ids = cap;
    p.shapes = from_qsa_shapes(s);
    p.scratch = scratch;
    p.attn_out = attn;
    p.n_q = n_q;
    p.id_stride = id_stride;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().attention(&c, &p, &e), e);
}

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids,
                           const int32_t* steps, int64_t cap, const QsaShapes& s,
                           float* attn, int64_t n_q, void* stream) {
    uint32_t launched = 0;
    guild_cuda_attention_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_ATTN_PROMPT_BATCH_V1;
    p.q = q;
    p.pools = from_attn_pools(pools);
    p.ids = ids;
    p.steps = steps;
    p.cap_or_n_ids = cap;
    p.shapes = from_qsa_shapes(s);
    p.attn_out = attn;
    p.n_q = n_q;
    p.out_launched = &launched;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().attention(&c, &p, &e), e);
    return launched != 0;
}

void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v,
                                  const int32_t* step, int64_t capacity, int max_context,
                                  const QsaShapes& shapes, float* output, int32_t* status,
                                  const uint16_t* mask, void* stream) {
    guild_cuda_attention_op_v1_t p{};
    p.struct_size = sizeof(p);
    p.mode = GUILD_CUDA_ATTN_NATIVE_FLASH_SHORT_V1;
    p.q = q;
    p.k_scratch = k;
    p.v_scratch = v;
    p.steps = step;
    p.cap_or_n_ids = capacity;
    p.max_context = max_context;
    p.shapes = from_qsa_shapes(shapes);
    p.attn_out = output;
    p.status_out = status;
    p.mask = mask;
    const auto c = context(stream);
    guild_error_info_t e{};
    checked(backend().attention(&c, &p, &e), e);
}

}  // namespace guild::kernels
