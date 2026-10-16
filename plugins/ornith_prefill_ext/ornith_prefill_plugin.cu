// plugins/ornith_prefill_ext/ornith_prefill_plugin.cu
// External .gkernel plugin built strictly against the public Guild Kernel SDK (kernel_abi.h).
// Provides:
//   1. ext.cuda.moe.router_topk_bf16 (GUILD_OP_MOE_ROUTER_TOPK, CUDA fast-path router projection)
//   2. ext.cpu.finite_check.v1       (GUILD_OP_FINITE_CHECK, CPU NaN/Inf validation op)
//
// Copyright (c) Guild Inference Contributors. MIT License.

#include "guild/abi/kernel_abi.h"

#if defined(GUILD_ENABLE_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef GUILD_EXT_PLUGIN_VERSION_STR
#define GUILD_EXT_PLUGIN_VERSION_STR "1.0.0"
#endif

#ifndef GUILD_EXT_PLUGIN_VERSION_MINOR
#define GUILD_EXT_PLUGIN_VERSION_MINOR 0
#endif

#ifndef GUILD_EXT_PLUGIN_VERSION_ENCODED
#define GUILD_EXT_PLUGIN_VERSION_ENCODED GUILD_KERNEL_ABI_MAKE_VERSION(1, GUILD_EXT_PLUGIN_VERSION_MINOR, 0)
#endif

namespace {

std::atomic<int64_t> g_ext_router_calls{0};
std::atomic<int64_t> g_ext_router_rows{0};
std::atomic<int64_t> g_ext_finite_calls{0};
std::atomic<int64_t> g_ext_shutdown_calls{0};

void set_err(guild_error_info_t* err, guild_status_t code, const char* comp, const char* msg) {
    if (!err) return;
    err->struct_size = sizeof(guild_error_info_t);
    err->code = code;
    err->source_component = comp;
    std::snprintf(err->message, sizeof(err->message), "%s", msg ? msg : "");
}

#if defined(GUILD_ENABLE_CUDA) || defined(__CUDACC__)

__device__ __forceinline__ float ext_f32_from_bf16(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

__device__ __forceinline__ float ext_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    }
    return value;
}

template <int BLOCK_SIZE>
__global__ void ext_bf16_f32_mmvf_1row_kernel(const float* __restrict__ x,
                                              const uint16_t* __restrict__ w,
                                              float* __restrict__ y,
                                              int n_in) {
    const int t = threadIdx.x;
    const uint16_t* row = w + static_cast<size_t>(blockIdx.x) * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    const float2* inputs2 = reinterpret_cast<const float2*>(x);
    __shared__ float partials[32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32) partials[t] = 0.0f;
        __syncthreads();
    }
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float2 input = inputs2[pair];
        acc = __fmaf_rn(ext_f32_from_bf16(static_cast<uint16_t>(weight)), input.x, acc);
        acc = __fmaf_rn(ext_f32_from_bf16(static_cast<uint16_t>(weight >> 16)), input.y, acc);
    }
    acc = ext_warp_sum(acc);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0) partials[t / 32] = acc;
        __syncthreads();
        if (t < 32) acc = ext_warp_sum(partials[t]);
    }
    if (t == 0) y[blockIdx.x] = acc;
}

template <int BLOCK_SIZE, int NT, bool Tiled>
__global__ void ext_bf16_f32_mmvf_multi_kernel(const float* __restrict__ x,
                                               int64_t ldx,
                                               const uint16_t* __restrict__ w,
                                               float* __restrict__ y,
                                               int64_t ldy,
                                               int n_in,
                                               int n_tok) {
    if constexpr (Tiled) {
        const int first = static_cast<int>(blockIdx.y) * NT;
        x += static_cast<size_t>(first) * ldx;
        y += static_cast<size_t>(first) * ldy;
        n_tok = min(NT, n_tok - first);
    }
    const int t = threadIdx.x;
    const uint16_t* row = w + static_cast<size_t>(blockIdx.x) * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    __shared__ float partials[NT][32];
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32) {
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t] = 0.0f;
        }
        __syncthreads();
    }
    float acc[NT];
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float w0 = ext_f32_from_bf16(static_cast<uint16_t>(weight));
        const float w1 = ext_f32_from_bf16(static_cast<uint16_t>(weight >> 16));
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const float2 input = reinterpret_cast<const float2*>(x + static_cast<size_t>(k) * ldx)[pair];
                acc[k] = __fmaf_rn(w0, input.x, acc[k]);
                acc[k] = __fmaf_rn(w1, input.y, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < NT; ++k) acc[k] = ext_warp_sum(acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0) {
#pragma unroll
            for (int k = 0; k < NT; ++k) partials[k][t / 32] = acc[k];
        }
        __syncthreads();
        if (t < 32) {
#pragma unroll
            for (int k = 0; k < NT; ++k) acc[k] = ext_warp_sum(partials[k][t]);
        }
    }
    if (t == 0) {
#pragma unroll
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) y[static_cast<size_t>(k) * ldy + blockIdx.x] = acc[k];
        }
    }
}

int ext_mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

void ext_router_projection_cuda(const float* x,
                                const uint16_t* w,
                                float* y,
                                int64_t n_in,
                                int64_t n_out,
                                int ncols,
                                void* stream) {
    g_ext_router_calls.fetch_add(1, std::memory_order_relaxed);
    g_ext_router_rows.fetch_add(ncols, std::memory_order_relaxed);

    if (ncols <= 0 || n_in <= 0 || n_out <= 0 || !x || !w || !y) return;
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const int bsize = ext_mmvf_block_size(n_in);

    if (ncols > 8) {
        const dim3 grid(static_cast<unsigned>(n_out), static_cast<unsigned>((ncols + 7) / 8));
#define EXT_LAUNCH_TILED(B) \
    case B: ext_bf16_f32_mmvf_multi_kernel<B, 8, true><<<grid, B, 0, st>>>(x, n_in, w, y, n_out, static_cast<int>(n_in), ncols); break
        switch (bsize) {
            EXT_LAUNCH_TILED(32); EXT_LAUNCH_TILED(64); EXT_LAUNCH_TILED(96); EXT_LAUNCH_TILED(128);
            EXT_LAUNCH_TILED(160); EXT_LAUNCH_TILED(192); EXT_LAUNCH_TILED(224); EXT_LAUNCH_TILED(256);
        }
#undef EXT_LAUNCH_TILED
        return;
    }

    for (int c0 = 0; c0 < ncols; c0 += 8) {
        const int nc = (ncols - c0 < 8) ? (ncols - c0) : 8;
        const float* xc = x + static_cast<size_t>(c0) * n_in;
        float* yc = y + static_cast<size_t>(c0) * n_out;
        if (nc == 1) {
#define EXT_LAUNCH_1ROW(B) \
    case B: ext_bf16_f32_mmvf_1row_kernel<B><<<static_cast<unsigned>(n_out), B, 0, st>>>(xc, w, yc, static_cast<int>(n_in)); break
            switch (bsize) {
                EXT_LAUNCH_1ROW(32); EXT_LAUNCH_1ROW(64); EXT_LAUNCH_1ROW(96); EXT_LAUNCH_1ROW(128);
                EXT_LAUNCH_1ROW(160); EXT_LAUNCH_1ROW(192); EXT_LAUNCH_1ROW(224); EXT_LAUNCH_1ROW(256);
            }
#undef EXT_LAUNCH_1ROW
        } else if (nc <= 4) {
#define EXT_LAUNCH_4ROW(B) \
    case B: ext_bf16_f32_mmvf_multi_kernel<B, 4, false><<<static_cast<unsigned>(n_out), B, 0, st>>>(xc, n_in, w, yc, n_out, static_cast<int>(n_in), nc); break
            switch (bsize) {
                EXT_LAUNCH_4ROW(32); EXT_LAUNCH_4ROW(64); EXT_LAUNCH_4ROW(96); EXT_LAUNCH_4ROW(128);
                EXT_LAUNCH_4ROW(160); EXT_LAUNCH_4ROW(192); EXT_LAUNCH_4ROW(224); EXT_LAUNCH_4ROW(256);
            }
#undef EXT_LAUNCH_4ROW
        } else {
#define EXT_LAUNCH_8ROW(B) \
    case B: ext_bf16_f32_mmvf_multi_kernel<B, 8, false><<<static_cast<unsigned>(n_out), B, 0, st>>>(xc, n_in, w, yc, n_out, static_cast<int>(n_in), nc); break
            switch (bsize) {
                EXT_LAUNCH_8ROW(32); EXT_LAUNCH_8ROW(64); EXT_LAUNCH_8ROW(96); EXT_LAUNCH_8ROW(128);
                EXT_LAUNCH_8ROW(160); EXT_LAUNCH_8ROW(192); EXT_LAUNCH_8ROW(224); EXT_LAUNCH_8ROW(256);
            }
#undef EXT_LAUNCH_8ROW
        }
    }
}

#endif // GUILD_ENABLE_CUDA || __CUDACC__

guild_status_t ext_finite_query_scratch(const guild_tensor_desc_t*,
                                        size_t,
                                        const guild_tensor_desc_t*,
                                        size_t,
                                        guild_scratch_request_t* out_scratch,
                                        guild_error_info_t* err) {
    if (!out_scratch || out_scratch->struct_size < sizeof(guild_scratch_request_t)) {
        set_err(err, GUILD_STATUS_ERR_INVALID_ARGUMENT, "ext.cpu.finite_check.v1", "invalid scratch request struct");
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    out_scratch->device_bytes = 0;
    out_scratch->device_alignment = 32;
    out_scratch->pinned_host_bytes = 0;
    out_scratch->pinned_host_alignment = 32;
    out_scratch->pageable_host_bytes = 0;
    return GUILD_STATUS_OK;
}

guild_status_t ext_finite_validate(const guild_device_capabilities_v1_t* caps,
                                   const guild_tensor_desc_t* inputs,
                                   size_t n_inputs,
                                   const guild_tensor_desc_t*,
                                   size_t,
                                   guild_error_info_t* err) {
    if (!caps || !inputs || n_inputs < 1) {
        set_err(err, GUILD_STATUS_ERR_INVALID_ARGUMENT, "ext.cpu.finite_check.v1", "missing capabilities or input tensor");
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    if (inputs[0].dtype != GUILD_DTYPE_F32) {
        set_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "ext.cpu.finite_check.v1", "finite_check requires F32 tensor");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    return GUILD_STATUS_OK;
}

guild_status_t ext_finite_execute(const guild_exec_context_v1_t*,
                                  const guild_tensor_desc_t* inputs,
                                  size_t n_inputs,
                                  const guild_tensor_desc_t*,
                                  size_t,
                                  const void*,
                                  size_t,
                                  guild_error_info_t* err) {
    g_ext_finite_calls.fetch_add(1, std::memory_order_relaxed);
    if (!inputs || n_inputs < 1 || !inputs[0].buffer.data) {
        set_err(err, GUILD_STATUS_ERR_INVALID_ARGUMENT, "ext.cpu.finite_check.v1", "missing input tensor buffer");
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    if (inputs[0].dtype != GUILD_DTYPE_F32) {
        set_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "ext.cpu.finite_check.v1", "finite_check requires F32 tensor");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    size_t elems = 1;
    for (int d = 0; d < inputs[0].n_dims; ++d) {
        if (inputs[0].dims[d] <= 0) {
            set_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "ext.cpu.finite_check.v1", "non-positive tensor dimension");
            return GUILD_STATUS_ERR_INVALID_TENSOR;
        }
        elems *= static_cast<size_t>(inputs[0].dims[d]);
    }
    const float* vals = static_cast<const float*>(inputs[0].buffer.data);
    for (size_t i = 0; i < elems; ++i) {
        if (!std::isfinite(vals[i])) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "nonfinite value at index %zu", i);
            set_err(err, GUILD_STATUS_ERR_NONFINITE_OUTPUT, "ext.cpu.finite_check.v1", msg);
            return GUILD_STATUS_ERR_NONFINITE_OUTPUT;
        }
    }
    return GUILD_STATUS_OK;
}

guild_status_t plugin_init(guild_core_host_t host, const guild_host_api_v1_t* api, guild_error_info_t* err) {
    if (!host || !api || api->struct_size < sizeof(guild_host_api_v1_t)) {
        set_err(err, GUILD_STATUS_ERR_ABI_MISMATCH, "ext.guild-kernels-ornith-prefill", "invalid host API table");
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    uint32_t negotiated = 0;
    if (api->negotiate_abi(GUILD_KERNEL_ABI_VERSION_V1, &negotiated) != GUILD_STATUS_OK) {
        set_err(err, GUILD_STATUS_ERR_ABI_MISMATCH, "ext.guild-kernels-ornith-prefill", "ABI negotiation failed");
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }

    // 1. Register CPU finite check op
    guild_kernel_op_desc_v1_t finite_desc{};
    finite_desc.struct_size = sizeof(finite_desc);
    finite_desc.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    finite_desc.op_kind = GUILD_OP_FINITE_CHECK;
    finite_desc.op_name = "finite_check";
    finite_desc.impl_id = "ext.cpu.finite_check.v1";
    finite_desc.package_id = "ext.guild-kernels-ornith-prefill";
    finite_desc.impl_version = GUILD_EXT_PLUGIN_VERSION_ENCODED;
    finite_desc.backend = GUILD_BACKEND_CPU;
    finite_desc.weight_dtype = GUILD_DTYPE_ANY;
    finite_desc.input_dtype = GUILD_DTYPE_F32;
    finite_desc.output_dtype = GUILD_DTYPE_I32;
    finite_desc.accum_dtype = GUILD_DTYPE_F32;
    finite_desc.max_batch_rows = 512;
    finite_desc.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC | GUILD_NUM_EXACT_SINGLE_COL_EQUIV | GUILD_NUM_FINITE_VALIDATED;
    finite_desc.supported_sync_modes = GUILD_SYNC_NONE | GUILD_SYNC_STREAM_ORDERED | GUILD_SYNC_HOST_BLOCKING;
    finite_desc.priority = 250;
    finite_desc.query_scratch = &ext_finite_query_scratch;
    finite_desc.validate = &ext_finite_validate;
    finite_desc.execute = &ext_finite_execute;

    guild_status_t st = api->register_kernel_op(host, &finite_desc, err);
    if (st != GUILD_STATUS_OK) return st;

#if defined(GUILD_ENABLE_CUDA) || defined(__CUDACC__)
    // 2. Register CUDA MoE router BF16 projection op (constructed on stack to prove deep-copy safety!)
    guild_prefill_kernel_ext_v1_t prefill_ext{};
    prefill_ext.header.struct_size = sizeof(prefill_ext);
    prefill_ext.header.tag = GUILD_PREFILL_KERNEL_EXT_TAG_V1;
    prefill_ext.header.version = GUILD_PREFILL_KERNEL_EXT_VERSION_V1;
    prefill_ext.gdn_ab = nullptr;
    prefill_ext.router_projection = &ext_router_projection_cuda;

    guild_kernel_op_desc_v1_t router_desc{};
    router_desc.struct_size = sizeof(router_desc);
    router_desc.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    router_desc.op_kind = GUILD_OP_MOE_ROUTER_TOPK;
    router_desc.op_name = "moe_router_topk";
    router_desc.impl_id = "ext.cuda.moe.router_topk_bf16";
    router_desc.package_id = "ext.guild-kernels-ornith-prefill";
    router_desc.impl_version = GUILD_EXT_PLUGIN_VERSION_ENCODED;
    router_desc.backend = GUILD_BACKEND_CUDA;
    router_desc.min_sm_version = 60;
    router_desc.weight_dtype = GUILD_DTYPE_F32;
    router_desc.input_dtype = GUILD_DTYPE_F32;
    router_desc.output_dtype = GUILD_DTYPE_F32;
    router_desc.accum_dtype = GUILD_DTYPE_F32;
    router_desc.max_batch_rows = 512;
    router_desc.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC | GUILD_NUM_EXACT_SINGLE_COL_EQUIV | GUILD_NUM_FINITE_VALIDATED;
    router_desc.supported_sync_modes = GUILD_SYNC_NONE | GUILD_SYNC_STREAM_ORDERED | GUILD_SYNC_HOST_BLOCKING | GUILD_SYNC_GRAPH_CAPTURE;
    router_desc.priority = 250; // Higher than built-in gpu.moe.router_topk_native (200)
    router_desc.direct_fn_table = nullptr;
    router_desc.next_ext = &prefill_ext;

    st = api->register_kernel_op(host, &router_desc, err);
    if (st != GUILD_STATUS_OK) return st;
#endif

    return GUILD_STATUS_OK;
}

void plugin_shutdown(guild_core_host_t) {
    g_ext_shutdown_calls.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

extern "C" {

GUILD_PLUGIN_EXPORT guild_status_t guild_plugin_entry_v1(
    uint32_t host_abi_version,
    guild_plugin_descriptor_v1_t* out_plugin_desc,
    guild_error_info_t* err) {
    if (!out_plugin_desc) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    if (GUILD_KERNEL_ABI_VERSION_MAJOR_OF(host_abi_version) != GUILD_KERNEL_ABI_VERSION_MAJOR) {
        set_err(err, GUILD_STATUS_ERR_ABI_MISMATCH, "ext.guild-kernels-ornith-prefill", "incompatible host ABI major version");
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    out_plugin_desc->struct_size = sizeof(*out_plugin_desc);
    out_plugin_desc->abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    out_plugin_desc->package_id = "ext.guild-kernels-ornith-prefill";
    out_plugin_desc->package_version = GUILD_EXT_PLUGIN_VERSION_STR;
    out_plugin_desc->publisher_id = "org.illatio.guild";
    out_plugin_desc->init = &plugin_init;
    out_plugin_desc->shutdown = &plugin_shutdown;
    return GUILD_STATUS_OK;
}

GUILD_PLUGIN_EXPORT int64_t guild_ext_ornith_router_call_count(void) {
    return g_ext_router_calls.load(std::memory_order_relaxed);
}

GUILD_PLUGIN_EXPORT int64_t guild_ext_ornith_router_row_count(void) {
    return g_ext_router_rows.load(std::memory_order_relaxed);
}

GUILD_PLUGIN_EXPORT int64_t guild_ext_ornith_finite_call_count(void) {
    return g_ext_finite_calls.load(std::memory_order_relaxed);
}

GUILD_PLUGIN_EXPORT int64_t guild_ext_ornith_shutdown_call_count(void) {
    return g_ext_shutdown_calls.load(std::memory_order_relaxed);
}

GUILD_PLUGIN_EXPORT void guild_ext_ornith_reset_counters(void) {
    g_ext_router_calls.store(0, std::memory_order_relaxed);
    g_ext_router_rows.store(0, std::memory_order_relaxed);
    g_ext_finite_calls.store(0, std::memory_order_relaxed);
}

} // extern "C"
