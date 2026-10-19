// Orchestration-only CDM1 adapters. All device arithmetic belongs to the backend.
#include "guild/kernels/cuda_moe.hpp"
#include "guild/kernels/dequant_bf16.hpp"
#include "guild/kernels/dequant_s2.hpp"
#include "guild/kernels/quantize_act.hpp"
#include "guild/kernels/s_gemv.hpp"
#include "guild/kernels/s2_gemv.hpp"
#include "guild/kernels/s2_gemv_q8.hpp"
#include "guild/kernels/s2_expert_grouped.hpp"
#include "guild/kernels/shared_expert.hpp"
#include "guild/kernels/router_top10.hpp"
#include "guild/kernels/native_router.hpp"
#include "guild/kernels/native_moe.hpp"
#include "guild/kernels/iq_kernels.hpp"
#include "guild/kernels/elementwise.hpp"
#include "guild/kernels/sampler.hpp"
#include "guild/kernels/verify_kernels.hpp"
#include "guild/kernels/gr.hpp"
#include "guild/kernels/fused_gr.hpp"
#include "guild/kernels/native_gr_norm.hpp"
#include "guild/kernels/native_gr_postops.hpp"
#include "guild/kernels/ple.hpp"
#include "guild/kernels/native_ple_postops.hpp"
#include "guild/kernels/remote_expert_kernels.hpp"
#include "guild/kernels/cvec.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdlib>
#include <stdexcept>

#if GUILD_BUNDLED_CUDA_MOE
extern "C" const guild_cuda_moe_ext_v1_t* guild_cuda_moe_table_v1();
#endif

namespace guild::kernels {
namespace {
thread_local const guild_cuda_moe_ext_v1_t* active = nullptr;
bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] && v[0] != '0';
}
std::atomic<uint32_t> flags{(env_on("GUILD_OLD_IQ_MMVQ") ? uint32_t(GUILD_CUDA_MOE_FLAG_IQ_OLD_KERNELS_V1) : 0u) |
                            (env_on("GUILD_GROUPED_V1") ? uint32_t(GUILD_CUDA_MOE_FLAG_NATIVE_GROUPED_V1_V1) : 0u)};
std::atomic<int> grouped_old{-1}, gr_fast{-1}, expert_mode{-1}, expert_phase{0};

const guild_cuda_moe_ext_v1_t* try_backend() {
    if (active)
        return active;
#if GUILD_BUNDLED_CUDA_MOE
    return guild_cuda_moe_table_v1();
#else
    return nullptr;
#endif
}
const guild_cuda_moe_ext_v1_t& backend() {
    if (const auto* b = try_backend())
        return *b;
    throw std::runtime_error(
        "CUDA MoE provider missing; bundled MoE arithmetic is disabled; install a CDM1 CUDA .gkernel");
}
guild_exec_context_v1_t context(void* stream) {
    guild_exec_context_v1_t c{};
    c.struct_size = sizeof(c);
    c.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    c.backend = GUILD_BACKEND_CUDA;
    c.device_stream = stream;
    c.sync_mode = GUILD_SYNC_STREAM_ORDERED;
    if (cudaGetDevice(&c.device_id) != cudaSuccess || c.device_id < 0) {
        cudaGetLastError();
        c.device_id = 0;
    }
    return c;
}
void checked(guild_status_t status, const guild_error_info_t& e) {
    if (status == GUILD_STATUS_OK)
        return;
    const char* message = e.message[0] ? e.message : "CUDA MoE backend execution failed";
    if (status == GUILD_STATUS_ERR_ABI_MISMATCH || status == GUILD_STATUS_ERR_INVALID_ARGUMENT)
        throw std::invalid_argument(message);
    throw std::runtime_error(message);
}
template <class T> T command(uint32_t mode) {
    T p{};
    p.struct_size = sizeof(p);
    p.mode = mode;
    return p;
}
template <class T>
void call(guild_status_t (*fn)(const guild_exec_context_v1_t*, const T*, guild_error_info_t*), const T& p, void* s) {
    const auto c = context(s);
    guild_error_info_t e{};
    checked(fn(&c, &p, &e), e);
}
void sync_state(const guild_cuda_moe_ext_v1_t& b) {
    auto p = command<guild_cuda_moe_config_v1_t>(GUILD_CUDA_MOE_CFG_SET_FLAGS_V1);
    p.flags = flags.load(std::memory_order_relaxed);
    p.grouped_select_old = grouped_old.load(std::memory_order_relaxed);
    call(b.configure, p, nullptr);
    p.mode = GUILD_CUDA_MOE_CFG_FUSED_GR_SET_FAST_V1;
    p.value = gr_fast.load(std::memory_order_relaxed);
    call(b.configure, p, nullptr);
    p.mode = GUILD_CUDA_MOE_CFG_NATIVE_EXPERT_SET_MODE_V1;
    p.value = expert_mode.load(std::memory_order_relaxed);
    p.value2 = expert_phase.load(std::memory_order_relaxed);
    call(b.configure, p, nullptr);
}
void toggle(uint32_t bit, bool on) {
    if (on)
        flags.fetch_or(bit, std::memory_order_relaxed);
    else
        flags.fetch_and(~bit, std::memory_order_relaxed);
    if (const auto* b = try_backend())
        sync_state(*b);
}
bool enabled(uint32_t bit) {
    return (flags.load(std::memory_order_relaxed) & bit) != 0;
}
int query(uint32_t mode, int value = 0, int64_t n = 0, int64_t hc = 0, int64_t lr = 0) {
    auto p = command<guild_cuda_moe_config_v1_t>(mode);
    int32_t result = 0;
    p.out_value = &result;
    p.value = value;
    p.n_embd = n;
    p.hc = hc;
    p.hc_lr = lr;
    call(backend().configure, p, nullptr);
    return result;
}
guild_cuda_sform_v1_t form(const SForm& f) {
    return {sizeof(guild_cuda_sform_v1_t), f.code_bits, f.code_bias, f.group_elems, static_cast<int32_t>(f.codebook),
            f.has_offset ? 1u : 0u,        f.act_kind};
}
guild_cuda_native_shared_weights_v1_t native_shared(const NativeSharedWeights& w) {
    return {sizeof(guild_cuda_native_shared_weights_v1_t),
            w.gate_type,
            w.up_type,
            w.down_type,
            w.gate_data,
            w.up_data,
            w.down_data,
            w.q8_1,
            w.x_q8_1};
}
void qd(uint32_t mode, const void* src, void* dst, int64_t n, void* s, const float* scales = nullptr,
        float* out_scales = nullptr, int type = 0, int64_t row0 = 0, int64_t rows = 0, int64_t cols = 0,
        const void* src2 = nullptr, const int32_t* tokens = nullptr, uint64_t row_bytes = 0, int ncols = 0) {
    auto p = command<guild_cuda_quant_dequant_op_v1_t>(mode);
    p.src = src;
    p.dst = dst;
    p.n = n;
    p.scales = scales;
    p.out_scales = out_scales;
    p.ggml_type = type;
    p.row0 = row0;
    p.rows = rows;
    p.cols = cols;
    p.src2 = src2;
    p.tokens = tokens;
    p.row_bytes = row_bytes;
    p.ncols = ncols;
    call(backend().quantize_dequant, p, s);
}
void sgemv(uint32_t mode, const void* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
           int64_t n_in, int64_t n_out, const SForm& f, int tpr, bool stage, void* s) {
    auto p = command<guild_cuda_s_gemv_op_v1_t>(mode);
    p.x = x;
    p.codes = codes;
    p.scales = scales;
    p.offset = offset;
    p.y = y;
    p.n_in = n_in;
    p.n_out = n_out;
    p.form = form(f);
    p.threads_per_row = tpr;
    p.stage_x = stage;
    call(backend().s_gemv, p, s);
}
void moe(const guild_cuda_moe_op_v1_t& p, void* s) {
    call(backend().moe, p, s);
}
void elementwise(const guild_cuda_elementwise_op_v1_t& p, void* s) {
    call(backend().elementwise, p, s);
}
void verify(const guild_cuda_verify_op_v1_t& p, void* s) {
    call(backend().verify, p, s);
}
void qwen(const guild_cuda_qwen_op_v1_t& p, void* s) {
    call(backend().qwen, p, s);
}
} // namespace

ScopedCudaMoE::ScopedCudaMoE(const ResolvedCudaMoE& binding) : previous_(active) {
    // A failing configure must not leave a dangling thread-local binding.
    sync_state(binding.table);
    active = &binding.table;
}
ScopedCudaMoE::~ScopedCudaMoE() {
    active = previous_;
}

void shared_expert_set_native_bf16(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_SHARED_NATIVE_BF16_V1, v);
}
bool shared_expert_native_bf16_enabled() {
    return enabled(GUILD_CUDA_MOE_FLAG_SHARED_NATIVE_BF16_V1);
}
void native_router_set_enabled(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_NATIVE_ROUTER_V1, v);
}
bool native_router_enabled() {
    return enabled(GUILD_CUDA_MOE_FLAG_NATIVE_ROUTER_V1);
}
void native_moe_combine_set_enabled(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_NATIVE_MOE_COMBINE_V1, v);
}
bool native_moe_combine_enabled() {
    return enabled(GUILD_CUDA_MOE_FLAG_NATIVE_MOE_COMBINE_V1);
}
void gr_set_fp32_activations(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_GR_FP32_ACT_V1, v);
}
void gr_set_native_mmvf(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_GR_NATIVE_MMVF_V1, v);
}
void ple_set_native_bf16(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_PLE_NATIVE_BF16_V1, v);
}
bool ple_native_bf16_enabled() {
    return enabled(GUILD_CUDA_MOE_FLAG_PLE_NATIVE_BF16_V1);
}
void ple_set_native_postops(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_PLE_NATIVE_POSTOPS_V1, v);
}
bool ple_native_postops_enabled() {
    return enabled(GUILD_CUDA_MOE_FLAG_PLE_NATIVE_POSTOPS_V1);
}
void native_grouped_set_v1(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_NATIVE_GROUPED_V1_V1, v);
}
void iq_set_old_kernels(bool v) {
    toggle(GUILD_CUDA_MOE_FLAG_IQ_OLD_KERNELS_V1, v);
}
bool iq_old_kernels() {
    return enabled(GUILD_CUDA_MOE_FLAG_IQ_OLD_KERNELS_V1);
}
void moe_grouped_select_old(int v) {
    grouped_old = v;
    if (const auto* b = try_backend())
        sync_state(*b);
}
int moe_grouped_last_path() {
    return query(GUILD_CUDA_MOE_CFG_QUERY_GROUPED_LAST_PATH_V1);
}
void fused_gr_set_fast(int v) {
    gr_fast = v;
    if (const auto* b = try_backend())
        sync_state(*b);
}
void native_expert_set_mode(int mode, int phase) {
    expert_mode = mode;
    expert_phase = phase;
    if (const auto* b = try_backend())
        sync_state(*b);
}
bool fused_gr_supported(int64_t n, int64_t hc, int64_t lr) {
    return query(GUILD_CUDA_MOE_CFG_FUSED_GR_SUPPORTED_V1, 0, n, hc, lr) != 0;
}
void fused_gr_check() {
    query(GUILD_CUDA_MOE_CFG_FUSED_GR_CHECK_V1);
}
int fused_gr_variant() {
    return query(GUILD_CUDA_MOE_CFG_FUSED_GR_VARIANT_V1);
}
bool ple_block_available() {
    return query(GUILD_CUDA_MOE_CFG_PLE_AVAILABLE_V1) != 0;
}
size_t coupled_draft_scratch_bytes(int nv) {
    auto p = command<guild_cuda_moe_config_v1_t>(GUILD_CUDA_MOE_CFG_COUPLED_SCRATCH_BYTES_V1);
    uint64_t bytes = 0;
    p.out_bytes = &bytes;
    p.value = nv;
    call(backend().configure, p, nullptr);
    return bytes;
}

void dequant_bf16(int t, const void* src, int64_t row0, int64_t rows, int64_t cols, uint16_t* dst, void* s) {
    qd(GUILD_CUDA_QD_DEQUANT_BF16_V1, src, dst, 0, s, nullptr, nullptr, t, row0, rows, cols);
}
void dequant_f16(int t, const void* src, int64_t row0, int64_t rows, int64_t cols, uint16_t* dst, void* s) {
    qd(GUILD_CUDA_QD_DEQUANT_F16_V1, src, dst, 0, s, nullptr, nullptr, t, row0, rows, cols);
}
void dequant_f32(int t, const void* src, int64_t row0, int64_t rows, int64_t cols, float* dst, void* s) {
    qd(GUILD_CUDA_QD_DEQUANT_F32_V1, src, dst, 0, s, nullptr, nullptr, t, row0, rows, cols);
}
void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t blocks) {
    qd(GUILD_CUDA_QD_DEQUANT_S2_V1, codes, out, blocks, nullptr, scales);
}
void quantize_q8_0(const float* x, uint8_t* out, int64_t n, void* s) {
    qd(GUILD_CUDA_QD_QUANTIZE_Q8_0_V1, x, out, n, s);
}
void quantize_q8_0_scaled(const float* x, uint8_t* out, float* scales, int64_t n, void* s) {
    qd(GUILD_CUDA_QD_QUANTIZE_Q8_0_SCALED_V1, x, out, n, s, nullptr, scales);
}
void dequant_q8_0(const uint8_t* x, float* out, int64_t n, void* s) {
    qd(GUILD_CUDA_QD_DEQUANT_Q8_0_V1, x, out, n, s);
}
void quantize_q8_K(const float* x, uint8_t* out, int64_t n, void* s) {
    qd(GUILD_CUDA_QD_QUANTIZE_Q8_K_V1, x, out, n, s);
}
void dequant_q8_K(const uint8_t* x, float* out, int64_t n, void* s) {
    qd(GUILD_CUDA_QD_DEQUANT_Q8_K_V1, x, out, n, s);
}
void quantize_q8_1_rows(const float* x, int64_t rows, int64_t cols, void* out, void* s) {
    qd(GUILD_CUDA_QD_QUANTIZE_Q8_1_ROWS_V1, x, out, 0, s, nullptr, nullptr, 0, 0, rows, cols);
}
void iq_dequant_f16(int t, const void* x, int64_t n, uint16_t* out, void* s) {
    qd(GUILD_CUDA_QD_IQ_DEQUANT_F16_V1, x, out, n, s, nullptr, nullptr, t);
}
void iq_dequant_f32(int t, const void* x, int64_t n, float* out, void* s) {
    qd(GUILD_CUDA_QD_IQ_DEQUANT_F32_V1, x, out, n, s, nullptr, nullptr, t);
}
void iq_embed_rows(int t, const void* x, size_t rb, const int32_t* ids, int64_t rows, int64_t cols, float* out,
                   void* s) {
    qd(GUILD_CUDA_QD_IQ_EMBED_ROWS_V1, x, out, 0, s, nullptr, nullptr, t, 0, rows, cols, nullptr, ids, rb);
}
void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t ff, int64_t n, uint16_t* out, void* s) {
    qd(GUILD_CUDA_QD_IQ_DEQUANT_GU_F16_V1, gate, out, 0, s, nullptr, nullptr, t, 0, ff, n, up);
}
void iq_mmvq(int t, const void* w, const void* x, float* out, int k, int n, int rows, void* s) {
    qd(GUILD_CUDA_QD_IQ_MMVQ_V1, w, out, 0, s, nullptr, nullptr, t, 0, n, k, x, nullptr, 0, rows);
}
void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t k, int64_t n) {
    sgemv(GUILD_CUDA_SGEMV_S2_V1, x, codes, scales, nullptr, y, k, n, {}, 0, false, nullptr);
}
void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t k, int64_t n,
                   int tpr) {
    sgemv(GUILD_CUDA_SGEMV_S2_QUADS_V1, x, codes, scales, nullptr, y, k, n, {}, tpr, false, nullptr);
}
void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t k, int64_t n, int tpr,
                  bool stage) {
    sgemv(GUILD_CUDA_SGEMV_S2_FAST_V1, x, codes, scales, nullptr, y, k, n, {}, tpr, stage, nullptr);
}
void s2_gemv_q8(const uint8_t* x, const uint8_t* codes, const float* scales, float* y, int64_t k, int64_t n, int tpr,
                void* s) {
    sgemv(GUILD_CUDA_SGEMV_S2_Q8_V1, x, codes, scales, nullptr, y, k, n, {}, tpr, false, s);
}
void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* off, float* y, int64_t k,
            int64_t n, const SForm& f) {
    sgemv(GUILD_CUDA_SGEMV_S_V1, x, codes, scales, off, y, k, n, f, 0, false, nullptr);
}
void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* off, float* y, int64_t k,
                  int64_t n, const SForm& f, int tpr) {
    sgemv(GUILD_CUDA_SGEMV_S_SPLIT_V1, x, codes, scales, off, y, k, n, f, tpr, false, nullptr);
}
void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* off, float* y,
                        int64_t k, int64_t n, const SForm& f, int tpr, void* s) {
    sgemv(GUILD_CUDA_SGEMV_S_SPLIT_ASYNC_V1, x, codes, scales, off, y, k, n, f, tpr, false, s);
}
#define GUILD_SGEMV_ADAPTER(name, mode)                                                                                \
    void name(const uint8_t* x, const uint8_t* codes, const float* scales, const float* off, float* y, int64_t k,      \
              int64_t n, const SForm& f, void* s) {                                                                    \
        sgemv(mode, x, codes, scales, off, y, k, n, f, 0, false, s);                                                   \
    }
GUILD_SGEMV_ADAPTER(s_gemv_q8k, GUILD_CUDA_SGEMV_S_Q8K_V1)
GUILD_SGEMV_ADAPTER(s_gemv_q8k_split, GUILD_CUDA_SGEMV_S_Q8K_SPLIT_V1)
GUILD_SGEMV_ADAPTER(s_gemv_q8_0_split, GUILD_CUDA_SGEMV_S_Q8_0_SPLIT_V1)
#undef GUILD_SGEMV_ADAPTER

void router_top10(const float* logits, int nt, int ne, int k, int* ids, float* weights, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_ROUTER_TOP10_V1);
    p.logits = logits;
    p.n_tokens = nt;
    p.n_expert = ne;
    p.k = k;
    p.ids = ids;
    p.weights = weights;
    moe(p, s);
}
bool router_top10_variant(const float* logits, int nt, int ne, int k, int* ids, float* weights, void* s, int variant) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_ROUTER_TOP10_VARIANT_V1);
    uint32_t ok = 0;
    p.out_supported = &ok;
    p.variant = variant;
    p.logits = logits;
    p.n_tokens = nt;
    p.n_expert = ne;
    p.k = k;
    p.ids = ids;
    p.weights = weights;
    moe(p, s);
    return ok != 0;
}
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_NATIVE_ROUTER_TOP10_V1);
    p.logits = logits;
    p.ids = ids;
    p.weights = weights;
    moe(p, s);
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int nt, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_NATIVE_ROUTER_TOP10_MULTI_V1);
    p.logits = logits;
    p.ids = ids;
    p.weights = weights;
    p.n_tokens = nt;
    moe(p, s);
}
void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n, int64_t k,
                 void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_COMBINE_V1);
    p.parts = parts;
    p.weights = weights;
    p.shared = shared;
    p.out = y;
    p.n_embd = n;
    p.k = k;
    moe(p, s);
}
void native_moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n, int64_t k,
                        void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_NATIVE_COMBINE_V1);
    p.parts = parts;
    p.weights = weights;
    p.shared = shared;
    p.out = y;
    p.n_embd = n;
    p.k = k;
    moe(p, s);
}
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* y, int64_t n,
                              int64_t k, int nt, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_NATIVE_COMBINE_MULTI_V1);
    p.parts = parts;
    p.weights = weights;
    p.shared = shared;
    p.out = y;
    p.n_embd = n;
    p.k = k;
    p.n_tokens = nt;
    moe(p, s);
}
void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gf,
                   const uint8_t* gc, const float* gs, const float* go, const SForm& uf, const uint8_t* uc,
                   const float* us, const float* uo, const SForm& df, const uint8_t* dc, const float* ds,
                   const float* d_o, const uint16_t* gi, float* scratch, float* out, int64_t n, int64_t ff, int tpr,
                   void* s, const float* x_f32, const NativeSharedWeights* nw) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_SHARED_EXPERT_V1);
    p.x_q8_0 = x_q8_0;
    p.x_q8k = x_q8k;
    p.x_bf16 = x_bf16;
    p.x_f32 = x_f32;
    p.gate_form = form(gf);
    p.gate_codes = gc;
    p.gate_scales = gs;
    p.gate_off = go;
    p.up_form = form(uf);
    p.up_codes = uc;
    p.up_scales = us;
    p.up_off = uo;
    p.down_form = form(df);
    p.down_codes = dc;
    p.down_scales = ds;
    p.down_off = d_o;
    p.gate_inp_bf16 = gi;
    p.scratch = scratch;
    p.out = out;
    p.n_embd = n;
    p.n_ff = ff;
    p.tpr = tpr;
    if (nw) {
        p.has_native_shared = 1;
        p.native_shared = native_shared(*nw);
    }
    moe(p, s);
}
void shared_expert_multi(int nt, const float* x, const uint16_t* xb, const NativeSharedWeights& nw, const uint16_t* gi,
                         float* gate, float* up, float* g, float* out, int64_t n, int64_t ff, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_SHARED_EXPERT_MULTI_V1);
    p.n_tokens = nt;
    p.x_f32 = x;
    p.x_bf16 = xb;
    p.native_shared = native_shared(nw);
    p.has_native_shared = 1;
    p.gate_inp_bf16 = gi;
    p.gate_buf = gate;
    p.up_buf = up;
    p.g_buf = g;
    p.out = out;
    p.n_embd = n;
    p.n_ff = ff;
    moe(p, s);
}
void moe_hit_select(const int32_t* ids, const int32_t* res, int k, int ne, int32_t* slot, int32_t* dst, int32_t* count,
                    void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_HIT_SELECT_V1);
    p.ids = const_cast<int32_t*>(ids);
    p.res_row = res;
    p.k = k;
    p.n_expert = ne;
    p.slot_out = slot;
    p.dst_out = dst;
    p.count_out = count;
    moe(p, s);
}
void moe_hit_select_multi(const int32_t* ids, const int32_t* res, int n, int ne, int32_t* slot, int32_t* dst,
                          int32_t* count, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_HIT_SELECT_MULTI_V1);
    p.ids = const_cast<int32_t*>(ids);
    p.res_row = res;
    p.n_tokens = n;
    p.n_expert = ne;
    p.slot_out = slot;
    p.dst_out = dst;
    p.count_out = count;
    moe(p, s);
}
void moe_hit_add(float* parts, const float* hit, const int32_t* dst, const int32_t* count, int64_t cap, int64_t n,
                 void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_HIT_ADD_V1);
    p.out = parts;
    p.parts = hit;
    p.dst_index = dst;
    p.d_count = count;
    p.cap = cap;
    p.n_embd = n;
    moe(p, s);
}
namespace {
void hit(uint32_t mode, const uint8_t* base, const int32_t* slot, const int32_t* dst, const int32_t* count, int64_t cap,
         int64_t blob, const uint8_t* x, void* scratch, float* out, void* s, const float* scales, int k = 0,
         float* trace = nullptr) {
    auto p = command<guild_cuda_moe_op_v1_t>(mode);
    p.blob_base = base;
    p.slot_index = slot;
    p.dst_index = dst;
    p.d_count = count;
    p.cap = cap;
    p.n_hits = cap;
    p.blob_bytes = blob;
    p.x_q8_0 = x;
    p.scratch = scratch;
    p.out = out;
    p.x_scales = scales;
    p.k_per_token = k;
    p.gate_up_trace = trace;
    moe(p, s);
}
} // namespace
void moe_hit_grouped_s2(const uint8_t* b, const int32_t* slot, const int32_t* dst, int64_t n, int64_t blob,
                        const uint8_t* x, void* scratch, float* out, void* s, const float* scales) {
    hit(GUILD_CUDA_MOE_HIT_GROUPED_S2_V1, b, slot, dst, nullptr, n, blob, x, scratch, out, s, scales);
}
void moe_hit_grouped_s2_dev(const uint8_t* b, const int32_t* slot, const int32_t* dst, const int32_t* count,
                            int64_t cap, int64_t blob, const uint8_t* x, void* scratch, float* out, void* s,
                            const float* scales) {
    hit(GUILD_CUDA_MOE_HIT_GROUPED_S2_DEV_V1, b, slot, dst, count, cap, blob, x, scratch, out, s, scales);
}
void moe_hit_grouped_s2_multi(const uint8_t* b, const int32_t* slot, const int32_t* dst, const int32_t* count,
                              int64_t cap, int64_t blob, const uint8_t* x, const float* scales, int k, void* scratch,
                              float* out, void* s) {
    hit(GUILD_CUDA_MOE_HIT_GROUPED_S2_MULTI_V1, b, slot, dst, count, cap, blob, x, scratch, out, s, scales, k);
}
void moe_hit_grouped_s2_cpu_order(const uint8_t* b, const int32_t* slot, const int32_t* dst, int64_t n, int64_t blob,
                                  const uint8_t* x, void* scratch, float* out, void* s, const float* scales,
                                  float* trace) {
    hit(GUILD_CUDA_MOE_HIT_GROUPED_S2_CPU_ORDER_V1, b, slot, dst, nullptr, n, blob, x, scratch, out, s, scales, 0,
        trace);
}
void moe_grouped_s2(const unsigned long long* ptr, const int32_t* start, const int32_t* count, const int32_t* dst,
                    const int32_t* tok, int64_t groups, int64_t entries, const uint8_t* x, const float* scales,
                    void* scratch, float* out, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_GROUPED_S2_V1);
    p.grp_ptr = ptr;
    p.grp_start = start;
    p.n_groups = count;
    p.ent_dst = dst;
    p.ent_tok = tok;
    p.cap_groups = groups;
    p.cap_entries = entries;
    p.x_q8_0 = x;
    p.x_scales = scales;
    p.scratch = scratch;
    p.out = out;
    moe(p, s);
}
void moe_group_resident(const int32_t* ids, int n, int k, const uint8_t* base, int64_t blob, unsigned long long* ptr,
                        int32_t* start, int32_t* counts, int32_t* dst, int32_t* tok, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_GROUP_RESIDENT_V1);
    p.ids = const_cast<int32_t*>(ids);
    p.n_tokens = n;
    p.k_per_token = k;
    p.blob_base = base;
    p.blob_bytes = blob;
    p.grp_ptr_mut = ptr;
    p.grp_start_mut = start;
    p.count_out = counts;
    p.ent_dst_mut = dst;
    p.ent_tok_mut = tok;
    moe(p, s);
}
void native_expert_grouped(const NativeExpertLayout& l, const unsigned long long* ptr, const int32_t* start,
                           const int32_t* count, const int32_t* dst, const int32_t* tok, int64_t groups,
                           int64_t entries, const void* x, void* scratch, float* out, void* s, int64_t grid) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_NATIVE_EXPERT_GROUPED_V1);
    p.gu_type = l.gu_type;
    p.d_type = l.d_type;
    p.n_embd = l.n_embd;
    p.n_ff = l.n_ff;
    p.grp_ptr = ptr;
    p.grp_start = start;
    p.n_groups = count;
    p.ent_dst = dst;
    p.ent_tok = tok;
    p.cap_groups = groups;
    p.cap_entries = entries;
    p.x_q8_1 = x;
    p.scratch = scratch;
    p.out = out;
    p.grid_groups = grid;
    moe(p, s);
}
void remote_expert_reduce(const float* parts, const void* meta, float* out, int tokens, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_REMOTE_REDUCE_V1);
    p.parts = parts;
    p.reduce_meta = meta;
    p.out = out;
    p.n_tokens = tokens;
    moe(p, s);
}
void remote_expert_copy_cpu_rows(float* dst, const float* src, const int32_t* mask, const int32_t* hit_rows,
                                 const int32_t* count, int tokens, void* s) {
    auto p = command<guild_cuda_moe_op_v1_t>(GUILD_CUDA_MOE_REMOTE_COPY_CPU_ROWS_V1);
    p.out = dst;
    p.parts = src;
    p.mask = mask;
    p.dst_index = hit_rows;
    p.d_count = count;
    p.n_tokens = tokens;
    moe(p, s);
}

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets, int64_t n, int bits, int bias,
                      int group, float* out, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_EMBEDDING_GATHER_V1);
    p.codes = codes;
    p.scales = scales;
    p.offsets = offsets;
    p.n = n;
    p.code_bits = bits;
    p.code_bias = bias;
    p.group_elems = group;
    p.dst = out;
    elementwise(p, s);
}
void gdn_gate(const float* alpha, const float* dt, const float* a, float* gate, int64_t nt, int64_t hv, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_GDN_GATE_V1);
    p.src = alpha;
    p.dt = dt;
    p.ssm_a = a;
    p.dst = gate;
    p.rows = nt;
    p.cols = hv;
    elementwise(p, s);
}
void scale_inplace(float* x, int64_t n, float scale, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_SCALE_INPLACE_V1);
    p.dst = x;
    p.n = n;
    p.scale = scale;
    elementwise(p, s);
}
void add_inplace(float* dst, const float* src, int64_t n, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_ADD_INPLACE_V1);
    p.dst = dst;
    p.src = src;
    p.n = n;
    elementwise(p, s);
}
void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_F32_TO_F16_V1);
    p.src = x;
    p.dst_u16 = y;
    p.n = n;
    elementwise(p, s);
}
void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_F32_TO_BF16_V1);
    p.src = x;
    p.dst_u16 = y;
    p.n = n;
    elementwise(p, s);
}
void silu_inplace(float* x, int64_t n, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_SILU_INPLACE_V1);
    p.dst = x;
    p.n = n;
    elementwise(p, s);
}
void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_RMS_NORM_WEIGHTED_V1);
    p.dst = x;
    p.scales = w;
    p.rows = rows;
    p.cols = cols;
    p.eps = eps;
    elementwise(p, s);
}
void doorbell_ring(uint32_t* seq, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_DOORBELL_RING_V1);
    p.d_seq = seq;
    elementwise(p, s);
}
void doorbell_wait(const uint32_t* flag, const uint32_t* seq, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_DOORBELL_WAIT_V1);
    p.d_flag = flag;
    p.d_seq = const_cast<uint32_t*>(seq);
    elementwise(p, s);
}
void copy_from_mapped(float* dst, const float* src, int64_t n, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_COPY_FROM_MAPPED_V1);
    p.dst = dst;
    p.src = src;
    p.n = n;
    elementwise(p, s);
}
void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_SCATTER_ROWS_F32_V1);
    p.src = src;
    p.dst = dst;
    p.rows_idx = rows;
    p.rows = n;
    p.cols = width;
    elementwise(p, s);
}
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_COPY_ROWS_FROM_MAPPED_V1);
    p.dst = dst;
    p.src = src;
    p.rows = rows;
    p.cols = width;
    p.hit_rows = hit_rows;
    p.count = count;
    elementwise(p, s);
}
namespace {
void publish(uint32_t mode, const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* xo,
             int32_t* io, float* wo, uint32_t* seq, uint32_t value, const int32_t* res, int ne, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(mode);
    p.src = x;
    p.ids = ids;
    p.scales = weights;
    p.n = n;
    p.cols = k;
    p.dst = xo;
    p.ids_out = io;
    p.weights_out = wo;
    p.d_seq = seq;
    p.value = value;
    p.d_res = res;
    p.n_expert = ne;
    elementwise(p, s);
}
} // namespace
void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* xo,
                      int32_t* io, float* wo, uint32_t* seq, void* s) {
    publish(GUILD_CUDA_ELEM_DOORBELL_PUBLISH_V1, x, ids, weights, n, k, xo, io, wo, seq, 0, nullptr, 0, s);
}
void doorbell_publish_value(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* xo,
                            int32_t* io, float* wo, uint32_t* seq, uint32_t value, void* s) {
    publish(GUILD_CUDA_ELEM_DOORBELL_PUBLISH_VALUE_V1, x, ids, weights, n, k, xo, io, wo, seq, value, nullptr, 0, s);
}
void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* res, int ne, int64_t n, int64_t k,
                          float* xo, int32_t* io, uint32_t* seq, void* s) {
    publish(GUILD_CUDA_ELEM_DOORBELL_PUBLISH_RES_V1, x, ids, nullptr, n, k, xo, io, nullptr, seq, 0, res, ne, s);
}
void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* s) {
    auto p = command<guild_cuda_elementwise_op_v1_t>(GUILD_CUDA_ELEM_COPY_I32_FROM_MAPPED_V1);
    p.ids_out = dst;
    p.ids = src;
    p.n = n;
    elementwise(p, s);
}

void sample_tokens(const float* logits, int nt, int nv, const int* hist, int nh, const SamplerParams& a, int* out,
                   void* s) {
    auto p = command<guild_cuda_sampler_op_v1_t>(GUILD_CUDA_SAMPLER_SAMPLE_TOKENS_V1);
    p.logits = logits;
    p.n_tokens = nt;
    p.n_vocab = nv;
    p.history = hist;
    p.history_len = nh;
    p.out = out;
    p.params = {sizeof(p.params), a.top_k,          a.top_p,           a.min_p,        a.temperature,
                a.min_keep,       a.penalty_last_n, a.penalty_repeat,  a.penalty_freq, a.penalty_present,
                a.seed,           a.counter,        a.greedy ? 1u : 0u};
    call(backend().sampler, p, s);
}
bool sample_greedy_cluster(const float* logits, int nt, int nv, int* out, void* s) {
    auto p = command<guild_cuda_sampler_op_v1_t>(GUILD_CUDA_SAMPLER_GREEDY_CLUSTER_V1);
    uint32_t launched = 0;
    p.logits = logits;
    p.n_tokens = nt;
    p.n_vocab = nv;
    p.out = out;
    p.out_launched = &launched;
    call(backend().sampler, p, s);
    return launched != 0;
}
void coupled_draft_stage(const SamplerParams* mapped, const int32_t* hist, SamplerParams* params, int32_t* ring,
                         int cap, void* s) {
    auto p = command<guild_cuda_sampler_op_v1_t>(GUILD_CUDA_SAMPLER_COUPLED_STAGE_V1);
    p.mapped_params = mapped;
    p.mapped_hist = hist;
    p.dev_params = params;
    p.ring = ring;
    p.cap = cap;
    call(backend().sampler, p, s);
}
void coupled_draft_sample(float* logits, int nv, const int32_t* sub, const int32_t* inv, int id_vocab,
                          const SamplerParams* params, int32_t* ring, int cap, int j, const int32_t* step,
                          void* scratch, int32_t* out, float* prob, void* s) {
    auto p = command<guild_cuda_sampler_op_v1_t>(GUILD_CUDA_SAMPLER_COUPLED_SAMPLE_V1);
    p.logits_mut = logits;
    p.n_vocab = nv;
    p.sub_to_id = sub;
    p.id_to_sub = inv;
    p.id_vocab = id_vocab;
    p.dev_params = const_cast<SamplerParams*>(params);
    p.ring = ring;
    p.cap = cap;
    p.j = j;
    p.step_rec = step;
    p.scratch = scratch;
    p.out = out;
    p.out_prob = prob;
    call(backend().sampler, p, s);
}
void finite_first_bad(const float* values, uint64_t count, unsigned long long* bad, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_FINITE_FIRST_BAD_V1);
    p.values = values;
    p.count = count;
    p.u64_dst = bad;
    verify(p, s);
}
void wait_flag_ge(const uint32_t* flag, uint32_t value, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_WAIT_FLAG_GE_V1);
    p.u32_flag = flag;
    p.u32_val = value;
    verify(p, s);
}
void gpu_stamp(unsigned long long* buf, int i, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_GPU_STAMP_V1);
    p.u64_dst = buf;
    p.idx = i;
    verify(p, s);
}
void resident_plan(const int32_t* ids, int entries, int k, const int32_t* res, int ne, const uint8_t* base,
                   const unsigned long long* off, long long blob, int32_t* plan, long long cap, uint32_t* skip,
                   uint32_t ring, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_RESIDENT_PLAN_V1);
    p.i32_src = ids;
    p.n_tok = entries;
    p.k = k;
    p.i32_src2 = res;
    p.n_expert = ne;
    p.bytes_src = base;
    p.u64_src = off;
    p.blob_bytes = blob;
    p.i32_dst = plan;
    p.n = cap;
    p.u32_skip_mut = skip;
    p.u32_val = ring;
    verify(p, s);
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_WAIT_FLAG_GE_OR_V1);
    p.u32_flag = flag;
    p.u32_val = value;
    p.u32_skip = skip;
    verify(p, s);
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_COPY_I32_UNLESS_V1);
    p.i32_dst = dst;
    p.i32_src = src;
    p.n = n;
    p.u32_skip = skip;
    p.u32_val = value;
    verify(p, s);
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_COPY_OR_ZERO_V1);
    p.dst = dst;
    p.values = src;
    p.n = n;
    p.u32_skip = skip;
    p.u32_val = value;
    verify(p, s);
}
void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* ids, int nt,
                          int64_t n, int bits, int bias, int group, uint64_t rc, uint64_t rg, float* out, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_EMBEDDING_GATHER_DEV_V1);
    p.bytes_src = codes;
    p.scales = scales;
    p.offsets = offsets;
    p.i32_src = ids;
    p.n_tok = nt;
    p.n = n;
    p.code_bits = bits;
    p.code_bias = bias;
    p.group_elems = group;
    p.row_codes = rc;
    p.row_groups = rg;
    p.dst = out;
    verify(p, s);
}
void broadcast_streams(const float* x, float* r, int64_t n, int hc, int nt, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_BROADCAST_STREAMS_V1);
    p.values = x;
    p.dst = r;
    p.n = n;
    p.hc = hc;
    p.n_tok = nt;
    verify(p, s);
}
void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_COPY_INDEXED_V1);
    p.dst = dst;
    p.values = src;
    p.stride = stride;
    p.i32_src = index;
    p.n = n;
    verify(p, s);
}
void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob, int cap, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_FETCH_BLOBS_V1);
    p.u64_src = src;
    p.i32_src = n;
    p.bytes_dst = dst;
    p.blob_bytes = blob;
    p.cap = cap;
    verify(p, s);
}
void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_REBASE_PTRS_V1);
    p.u64_dst = ptr;
    p.i32_src = n;
    p.bytes_dst = base;
    p.blob_bytes = blob;
    verify(p, s);
}
void add_streams_broadcast(const float* h, const float* e, float* r, int64_t n, int hc, int nt, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_ADD_STREAMS_BROADCAST_V1);
    p.values = h;
    p.src2 = e;
    p.dst = r;
    p.n = n;
    p.hc = hc;
    p.n_tok = nt;
    verify(p, s);
}
void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_IDENT_HITS_V1);
    p.i32_src = ids;
    p.n_tok = n;
    p.i32_dst = slot;
    p.i32_dst2 = dst;
    p.i32_dst3 = count;
    verify(p, s);
}
void mtp_select(const float* r, int64_t stride, const int32_t* ids, const int32_t* row, float* dst, int32_t* tok,
                int32_t* out, int j, void* s, const float* probs, float* out_p) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_MTP_SELECT_V1);
    p.values = r;
    p.stride = stride;
    p.i32_src = ids;
    p.i32_src2 = row;
    p.dst = dst;
    p.i32_dst = tok;
    p.i32_dst2 = out;
    p.idx = j;
    p.src2 = probs;
    p.out_p = out_p;
    verify(p, s);
}
void gather_rows(const uint8_t* src, int64_t rb, const int32_t* ids, int64_t n, uint8_t* dst, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_GATHER_ROWS_V1);
    p.bytes_src = src;
    p.stride = rb;
    p.i32_src = ids;
    p.n = n;
    p.bytes_dst = dst;
    verify(p, s);
}
void map_ids(int32_t* ids, const int32_t* table, int n, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_MAP_IDS_V1);
    p.i32_dst = ids;
    p.i32_src = table;
    p.n_tok = n;
    verify(p, s);
}
void row_top_prob(const float* logits, int rows, int vocab, const int32_t* ids, float* probs, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_ROW_TOP_PROB_V1);
    p.values = logits;
    p.n_tok = rows;
    p.cap = vocab;
    p.i32_src = ids;
    p.dst = probs;
    verify(p, s);
}
void dense_steps(const int32_t* cells, int n, int32_t* steps, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_DENSE_STEPS_V1);
    p.i32_src = cells;
    p.n_tok = n;
    p.i32_dst = steps;
    verify(p, s);
}
void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t stride, void* s) {
    auto p = command<guild_cuda_verify_op_v1_t>(GUILD_CUDA_VERIFY_WINDOW_IDS_V1);
    p.i32_dst = steps;
    p.n_tok = n;
    p.cap = window;
    p.i32_dst2 = ids;
    p.stride = stride;
    verify(p, s);
}

namespace {
guild_cuda_gr_shapes_v1_t gr_shapes(const GrShapes& s) {
    return {sizeof(guild_cuda_gr_shapes_v1_t), s.n_embd, s.hc, s.hc_lr};
}
guild_cuda_fused_gr_args_v1_t fused_args(const FusedGrArgs& a) {
    return {sizeof(guild_cuda_fused_gr_args_v1_t),
            a.R,
            a.R_out,
            a.apply ? 1u : 0u,
            a.bo_prev,
            a.inj_prev,
            a.w_norm,
            a.w_down,
            a.w_up,
            a.w_inject,
            a.eps,
            a.lo,
            a.rs,
            a.inject_out,
            a.mixed};
}
guild_cuda_ple_weights_v1_t ple_weights(const PleWeights& w) {
    return {sizeof(guild_cuda_ple_weights_v1_t),
            w.key_codes,
            w.key_scales,
            w.value_bf16,
            w.norm_key,
            w.norm_query,
            w.norm_conv,
            w.conv1d_f16,
            w.key_native_data,
            w.key_native_type,
            w.key_native_q8_1,
            w.key_bf16};
}
guild_cuda_ple_out_v1_t ple_out(const PleOut& o) {
    return {sizeof(guild_cuda_ple_out_v1_t), o.key, o.value, o.gate, o.gated, o.normalized, o.conv, o.result};
}
} // namespace
void gr_read(const float* r, const float* norm, const uint16_t* down, const uint16_t* up, const uint16_t* inject,
             float eps, const GrShapes& shapes, const GrWorkspace& ws, float* mixed, float* inj, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_GR_READ_V1);
    p.R = r;
    p.w_norm = norm;
    p.w_down = down;
    p.w_up = up;
    p.w_inject = inject;
    p.eps = eps;
    p.gr_shapes = gr_shapes(shapes);
    p.gr_ws = {sizeof(p.gr_ws), ws.xn, ws.xq, ws.lq, ws.gated, ws.lo, ws.bytes};
    p.mixed = mixed;
    p.inject_out = inj;
    qwen(p, s);
}
void gr_write(const float* r, const float* bo, const float* inj, const GrShapes& shapes, float* out, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_GR_WRITE_V1);
    p.R = r;
    p.block_out = bo;
    p.inject = inj;
    p.gr_shapes = gr_shapes(shapes);
    p.R_out = out;
    qwen(p, s);
}
void fused_gr_read(const FusedGrArgs& a, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_FUSED_GR_READ_V1);
    p.fused_gr = fused_args(a);
    qwen(p, s);
}
void fused_gr_read_multi(const FusedGrArgs* a, int nt, float* scratch, void* s, unsigned long long* stamp, int i) {
    if (nt <= 0)
        return;
    if (!a || nt > kFusedGrMaxT)
        throw std::invalid_argument("fused_gr_read_multi: invalid arguments");
    guild_cuda_fused_gr_args_v1_t args[kFusedGrMaxT]{};
    for (int t = 0; t < nt; ++t)
        args[t] = fused_args(a[t]);
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_FUSED_GR_READ_MULTI_V1);
    p.fused_gr_multi = args;
    p.n_tok = nt;
    p.xn_scratch = scratch;
    p.stamp_buf = stamp;
    p.stamp_i0 = i;
    qwen(p, s);
}
void native_gr_rms_norm_weighted(const float* x, const float* gamma, float* out, int cols, int rows, float eps,
                                 void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_NATIVE_GR_RMS_NORM_V1);
    p.R = x;
    p.w_norm = gamma;
    p.R_out = out;
    p.n_cols = cols;
    p.n_rows = rows;
    p.eps = eps;
    qwen(p, s);
}
void native_gr_down_silu(float* lo, int lr, int hc, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_NATIVE_GR_DOWN_SILU_V1);
    p.R_out = lo;
    p.gr_shapes = gr_shapes({0, hc, lr});
    qwen(p, s);
}
void native_gr_pre_gated(const float* xn, float* gate, float* mixed, int n, int hc, bool layer, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_NATIVE_GR_PRE_GATED_V1);
    p.R = xn;
    p.gated = gate;
    p.mixed = mixed;
    p.gr_shapes = gr_shapes({n, hc, 0});
    p.fused_layer = layer;
    qwen(p, s);
}
void native_gr_post(const float* r, const float* bo, const float* inj, float* out, int n, int hc, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_NATIVE_GR_POST_V1);
    p.R = r;
    p.block_out = bo;
    p.inject = inj;
    p.R_out = out;
    p.gr_shapes = gr_shapes({n, hc, 0});
    qwen(p, s);
}
void ple_block(const float* emb, const float* hidden, const float* hist, const PleWeights& w, PleOut& out,
               void* scratch, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_PLE_BLOCK_V1);
    p.emb = emb;
    p.hidden = hidden;
    p.hist_rows = hist;
    p.ple_weights = ple_weights(w);
    p.ple_out = ple_out(out);
    p.scratch = scratch;
    qwen(p, s);
}
void ple_block_projected(const float* key, const float* value, const float* hidden, const float* hist,
                         const PleWeights& w, PleOut& out, void* scratch, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_PLE_BLOCK_PROJECTED_V1);
    p.projected_key = key;
    p.projected_value = value;
    p.hidden = hidden;
    p.hist_rows = hist;
    p.ple_weights = ple_weights(w);
    p.ple_out = ple_out(out);
    p.scratch = scratch;
    qwen(p, s);
}
void ple_history_advance(float* hist, const float* normalized, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_PLE_HISTORY_ADVANCE_V1);
    p.hist_mut = hist;
    p.normalized = normalized;
    qwen(p, s);
}
void native_ple_postops(const float* key, const float* hidden, const float* value, const float* hist,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_NATIVE_PLE_POSTOPS_V1);
    p.projected_key = key;
    p.hidden = hidden;
    p.projected_value = value;
    p.hist_rows = hist;
    p.ple_weights = ple_weights(w);
    p.ple_buffers = {sizeof(p.ple_buffers), b.key, b.query, b.gate, b.gated, b.normalized, b.conv, b.result};
    qwen(p, s);
}
void native_ple_postops_batch(float* key, float* hidden, const float* value, float* hist, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int nt, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_NATIVE_PLE_POSTOPS_BATCH_V1);
    p.key_mut = key;
    p.hidden_mut = hidden;
    p.projected_value = value;
    p.hist_mut = hist;
    p.ple_weights = ple_weights(w);
    p.query_norm = query_norm;
    p.gated = gated;
    p.gate = gate;
    p.n_tok = nt;
    qwen(p, s);
}
void cvec_launch_kernel(float* r, const float* dir, const float* scale, const int* on, int mode, int64_t layer,
                        int64_t nt, int64_t n, int64_t hc, int64_t rld, const float* bo, int64_t bld, const float* inj,
                        int64_t ild, bool write, void* s) {
    auto p = command<guild_cuda_qwen_op_v1_t>(GUILD_CUDA_QWEN_CVEC_APPLY_V1);
    p.R_out = r;
    p.cvec_dir = dir;
    p.cvec_s = scale;
    p.cvec_on = on;
    p.cvec_mode = mode;
    p.layer = layer;
    p.n_tok = static_cast<int32_t>(nt);
    p.gr_shapes = gr_shapes({n, hc, 0});
    p.r_ld = rld;
    p.block_out = bo;
    p.bo_ld = bld;
    p.inject = inj;
    p.inj_ld = ild;
    p.write = write;
    qwen(p, s);
}

} // namespace guild::kernels
