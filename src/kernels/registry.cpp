// src/kernels/registry.cpp - Hardware Kernel Operation Registry & Selection
#include "guild/kernels/registry.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include "guild/core/validation.hpp"

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
#include "guild/kernels/native_mmvq.hpp"
#include "guild/kernels/native_qsa.hpp"
#include "guild/kernels/native_rope.hpp"
#include "guild/kernels/native_moe.hpp"
#include "guild/kernels/native_gdn.hpp"
#include "guild/kernels/native_gdn_preprocess.hpp"
#include "guild/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace guild::kernels {
namespace {

void fill_err(guild_error_info_t* err, guild_status_t code, const char* comp, const std::string& msg) {
    if (!err) return;
    err->struct_size = sizeof(guild_error_info_t);
    err->code = code;
    err->source_component = comp;
    std::snprintf(err->message, sizeof(err->message), "%s", msg.c_str());
}

guild_status_t execute_cpu_finite_check(
    const guild_exec_context_v1_t* /*ctx*/,
    const guild_tensor_desc_t* inputs,
    size_t n_inputs,
    const guild_tensor_desc_t* /*outputs*/,
    size_t /*n_outputs*/,
    const void* /*op_params*/,
    size_t /*op_params_size*/,
    guild_error_info_t* err) {
    if (!inputs || n_inputs < 1) {
        fill_err(err, GUILD_STATUS_ERR_INVALID_ARGUMENT, "cpu.finite_check.simd64", "missing input tensor");
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    const guild_status_t v = validate_tensor_desc(inputs[0], err);
    if (v != GUILD_STATUS_OK) return v;
    if (inputs[0].dtype != GUILD_DTYPE_F32) {
        fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "cpu.finite_check.simd64", "finite check requires F32 tensor");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    size_t elems = 1;
    for (int i = 0; i < inputs[0].n_dims; ++i) elems *= size_t(inputs[0].dims[i]);
    std::string e;
    if (!core::validate_finite(static_cast<const float*>(inputs[0].buffer.data), elems, "tensor", e)) {
        fill_err(err, GUILD_STATUS_ERR_NONFINITE_OUTPUT, "cpu.finite_check.simd64", e);
        return GUILD_STATUS_ERR_NONFINITE_OUTPUT;
    }
    return GUILD_STATUS_OK;
}

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
const ResolvedPrefillKernels kBuiltinPrefillKernels{
    &native_quantize_q8_1,
    &native_mmvq,
    &native_qsa_rms_norm_weighted,
    &gdn_prefill_conv,
    &gdn_prefill_recurrence,
    &native_gdn_out_norm,
    &native_rope_apply
};
#endif

} // namespace

const char* status_to_string(guild_status_t status) {
    switch (status) {
    case GUILD_STATUS_OK: return "ok";
    case GUILD_STATUS_ERR_INVALID_ARGUMENT: return "invalid argument";
    case GUILD_STATUS_ERR_ABI_MISMATCH: return "ABI version mismatch";
    case GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE: return "unsupported hardware";
    case GUILD_STATUS_ERR_CAPABILITY_MISMATCH: return "capability mismatch";
    case GUILD_STATUS_ERR_MISSING_RUNNER: return "missing model runner";
    case GUILD_STATUS_ERR_MISSING_KERNEL: return "missing kernel operation";
    case GUILD_STATUS_ERR_DUPLICATE_REGISTRATION: return "duplicate registration";
    case GUILD_STATUS_ERR_DEPENDENCY_MISSING: return "missing required dependency";
    case GUILD_STATUS_ERR_INVALID_TENSOR: return "invalid tensor descriptor";
    case GUILD_STATUS_ERR_BUFFER_LIFETIME: return "invalid or expired buffer lifetime";
    case GUILD_STATUS_ERR_OUT_OF_MEMORY: return "scratch memory exceeds available budget";
    case GUILD_STATUS_ERR_EXECUTION_FAILED: return "kernel execution failed";
    case GUILD_STATUS_ERR_NONFINITE_OUTPUT: return "nonfinite value in output";
    case GUILD_STATUS_ERR_SECURITY_VIOLATION: return "package security violation";
    case GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY: return "active session holds kernel lease";
    default: return "internal error";
    }
}

const char* op_kind_to_string(guild_op_kind_t kind) {
    switch (kind) {
    case GUILD_OP_QUANTIZED_GEMV: return "quantized_gemv";
    case GUILD_OP_QUANTIZED_GEMM: return "quantized_gemm";
    case GUILD_OP_GROUPED_MOE_GEMM: return "grouped_moe_gemm";
    case GUILD_OP_MOE_ROUTER_TOPK: return "moe_router_topk";
    case GUILD_OP_MOE_DISPATCH: return "moe_dispatch";
    case GUILD_OP_BATCHED_EXPERT_EXEC: return "batched_expert_exec";
    case GUILD_OP_SHARED_EXPERT: return "shared_expert";
    case GUILD_OP_MOE_COMBINE: return "moe_combine";
    case GUILD_OP_ATTENTION_DECODE: return "attention_decode";
    case GUILD_OP_ATTENTION_PREFILL: return "attention_prefill";
    case GUILD_OP_QSA_INDEXER_SELECT: return "qsa_indexer_select";
    case GUILD_OP_ROPE_APPLY: return "rope_apply";
    case GUILD_OP_GDN_CONV_L2: return "gdn_conv_l2";
    case GUILD_OP_GDN_AB_PROJ: return "gdn_ab_proj";
    case GUILD_OP_GDN_RECURRENCE_STEP: return "gdn_recurrence_step";
    case GUILD_OP_GDN_OUT_NORM: return "gdn_out_norm";
    case GUILD_OP_QUANTIZE_ACT: return "quantize_act";
    case GUILD_OP_DEQUANTIZE: return "dequantize";
    case GUILD_OP_RMS_NORM: return "rms_norm";
    case GUILD_OP_GATED_RESIDUAL: return "gated_residual";
    case GUILD_OP_KV_APPEND: return "kv_append";
    case GUILD_OP_KV_GATHER: return "kv_gather";
    case GUILD_OP_PLE_COMPUTE: return "ple_compute";
    case GUILD_OP_SPEC_VERIFY: return "spec_verify";
    case GUILD_OP_FINITE_CHECK: return "finite_check";
    case GUILD_OP_EMBEDDING_GATHER: return "embedding_gather";
    default: return "invalid_op";
    }
}

guild_status_t negotiate_abi_version(uint32_t requested_version, uint32_t* out_negotiated_version) {
    const uint32_t req_major = GUILD_KERNEL_ABI_VERSION_MAJOR_OF(requested_version);
    const uint32_t req_minor = GUILD_KERNEL_ABI_VERSION_MINOR_OF(requested_version);
    if (req_major != GUILD_KERNEL_ABI_VERSION_MAJOR || req_minor > GUILD_KERNEL_ABI_VERSION_MINOR) {
        if (out_negotiated_version) *out_negotiated_version = GUILD_KERNEL_ABI_VERSION_V1;
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (out_negotiated_version) *out_negotiated_version = requested_version;
    return GUILD_STATUS_OK;
}

guild_status_t validate_buffer_desc(const guild_buffer_desc_t& buf, guild_error_info_t* err) {
    if (buf.struct_size < sizeof(guild_buffer_desc_t)) {
        fill_err(err, GUILD_STATUS_ERR_ABI_MISMATCH, "buffer", "guild_buffer_desc_t struct_size too small");
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (buf.lifetime == GUILD_LIFETIME_EXPIRED || buf.lifetime > GUILD_LIFETIME_PERSISTENT) {
        fill_err(err, GUILD_STATUS_ERR_BUFFER_LIFETIME, "buffer", "buffer lifetime is expired or invalid");
        return GUILD_STATUS_ERR_BUFFER_LIFETIME;
    }
    if (buf.ownership > GUILD_OWNERSHIP_TRANSIENT_COPY) {
        fill_err(err, GUILD_STATUS_ERR_BUFFER_LIFETIME, "buffer", "invalid buffer ownership kind");
        return GUILD_STATUS_ERR_BUFFER_LIFETIME;
    }
    if (!buf.data || buf.byte_size == 0) {
        fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "buffer", "buffer pointer is null or byte_size is zero");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    if (buf.alignment > 1 && (reinterpret_cast<uintptr_t>(buf.data) % buf.alignment) != 0) {
        fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "buffer", "buffer pointer violates declared alignment");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    return GUILD_STATUS_OK;
}

guild_status_t validate_tensor_desc(const guild_tensor_desc_t& tensor, guild_error_info_t* err) {
    if (tensor.struct_size < sizeof(guild_tensor_desc_t)) {
        fill_err(err, GUILD_STATUS_ERR_ABI_MISMATCH, "tensor", "guild_tensor_desc_t struct_size too small");
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    const guild_status_t bstat = validate_buffer_desc(tensor.buffer, err);
    if (bstat != GUILD_STATUS_OK) return bstat;
    if (tensor.n_dims < 1 || tensor.n_dims > GUILD_MAX_TENSOR_DIMS) {
        fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "tensor", "tensor n_dims must be in 1..4");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    int block_elems = 0, block_bytes = 0;
    if (tensor.dtype == GUILD_DTYPE_I32) {
        block_elems = 1;
        block_bytes = 4;
    } else if (!guild::block_geometry(static_cast<uint32_t>(tensor.dtype), block_elems, block_bytes)) {
        fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "tensor", "unsupported or unknown tensor dtype");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    if (tensor.dims[0] <= 0 || (tensor.dims[0] % block_elems) != 0) {
        fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "tensor", "contiguous dimension dims[0] is not a positive multiple of block size");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    uint64_t required_bytes = uint64_t(tensor.dims[0] / block_elems) * uint64_t(block_bytes);
    for (int d = 1; d < tensor.n_dims; ++d) {
        if (tensor.dims[d] <= 0) {
            fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "tensor", "tensor dimension must be positive");
            return GUILD_STATUS_ERR_INVALID_TENSOR;
        }
        if (tensor.strides_bytes[d] != 0 && tensor.strides_bytes[d] < required_bytes) {
            fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "tensor", "tensor stride is smaller than contiguous sub-tensor size");
            return GUILD_STATUS_ERR_INVALID_TENSOR;
        }
        const uint64_t step = tensor.strides_bytes[d] ? tensor.strides_bytes[d] : required_bytes;
        if (uint64_t(tensor.dims[d]) > UINT64_MAX / step) {
            fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "tensor", "tensor byte size overflow");
            return GUILD_STATUS_ERR_INVALID_TENSOR;
        }
        required_bytes = step * uint64_t(tensor.dims[d]);
    }
    if (tensor.buffer.byte_size < required_bytes) {
        fill_err(err, GUILD_STATUS_ERR_INVALID_TENSOR, "tensor", "backing buffer is smaller than tensor shape requires");
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    }
    return GUILD_STATUS_OK;
}

guild_device_capabilities_v1_t query_device_capabilities(const hardware::HardwareInfo& hw) {
    guild_device_capabilities_v1_t caps{};
    caps.struct_size = sizeof(guild_device_capabilities_v1_t);
    caps.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    caps.available_backends = GUILD_BACKEND_CPU;
    caps.cpu_isa_flags = GUILD_ISA_SSE42 | GUILD_ISA_AVX;
    if (hw.isa_avx2) caps.cpu_isa_flags |= GUILD_ISA_AVX2 | GUILD_ISA_FMA | GUILD_ISA_F16C | GUILD_ISA_BMI2;
    if (hw.isa_avx512) caps.cpu_isa_flags |= GUILD_ISA_AVX512F | GUILD_ISA_AVX512BW | GUILD_ISA_AVX512DQ | GUILD_ISA_AVX512VL;
    if (hw.isa_vnni) caps.cpu_isa_flags |= GUILD_ISA_AVX512VNNI;
    if (hw.isa_vbmi) caps.cpu_isa_flags |= GUILD_ISA_AVX512VBMI;
    caps.cpu_physical_cores = hw.cpu_physical_cores;
    caps.cpu_sockets = hw.cpu_sockets;
    caps.gpu_device_count = hw.gpu_count;
    caps.available_ram_bytes = hw.ram_free_bytes > 0 ? hw.ram_free_bytes : hw.ram_total_bytes;
    caps.available_vram_bytes = hw.gpu_vram_bytes;
    caps.core_cap_flags = GUILD_CORE_CAP_HOST_ONLY_KV | GUILD_CORE_CAP_PINNED_EXPERT_RAM |
                          GUILD_CORE_CAP_DENSE_CAUSAL_ATTN | GUILD_CORE_CAP_INDEXED_QSA_ATTN;
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
    if (hw.has_cuda && hw.gpu_count > 0) {
#if defined(GUILD_ENABLE_HIP)
        caps.available_backends |= GUILD_BACKEND_HIP;
#else
        caps.available_backends |= GUILD_BACKEND_CUDA;
#endif
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            caps.gpu_sm_version = prop.major * 10 + prop.minor;
            if (caps.gpu_sm_version >= 61) caps.gpu_cap_flags |= GUILD_GPU_CAP_DP4A;
            if (caps.gpu_sm_version >= 60) caps.gpu_cap_flags |= GUILD_GPU_CAP_FP16 | GUILD_GPU_CAP_BF16;
            if (caps.gpu_sm_version >= 70) caps.gpu_cap_flags |= GUILD_GPU_CAP_TENSOR_CORE;
            caps.gpu_cap_flags |= GUILD_GPU_CAP_CUDA_GRAPH | GUILD_GPU_CAP_MAPPED_HOST;
        }
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b > 0) {
            caps.available_vram_bytes = free_b;
        }
        caps.core_cap_flags |= GUILD_CORE_CAP_PCIE_EXPERT_SPLIT | GUILD_CORE_CAP_CUDA_GRAPH_VERIFY |
                               GUILD_CORE_CAP_BATCHED_PREFILL | GUILD_CORE_CAP_MTP_SPECULATIVE |
                               GUILD_CORE_CAP_PLE_NGRAM;
    }
#endif
    return caps;
}

guild_device_capabilities_v1_t query_host_device_capabilities() {
    return query_device_capabilities(hardware::detect_hardware());
}

KernelRegistry::KernelRegistry() = default;
KernelRegistry::~KernelRegistry() {
    for (auto it = plugins_.rbegin(); it != plugins_.rend(); ++it) {
        if (it->desc.shutdown) {
            it->desc.shutdown(reinterpret_cast<guild_core_host_t>(this));
        }
    }
}

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry reg;
    static std::once_flag once;
    std::call_once(once, [] { reg.register_builtin_kernels(); });
    return reg;
}

guild_status_t KernelRegistry::register_op(const guild_kernel_op_desc_v1_t& desc, std::string& err) {
    if (desc.struct_size < sizeof(guild_kernel_op_desc_v1_t)) {
        err = "kernel op descriptor struct_size is smaller than ABI v1";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    uint32_t negotiated = 0;
    if (negotiate_abi_version(desc.abi_version, &negotiated) != GUILD_STATUS_OK) {
        err = "kernel op descriptor ABI version is incompatible with host ABI v1";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (desc.op_kind <= GUILD_OP_INVALID || desc.op_kind >= GUILD_OP_COUNT ||
        !desc.op_name || !desc.op_name[0] || !desc.impl_id || !desc.impl_id[0] ||
        !desc.package_id || !desc.package_id[0] || desc.backend == GUILD_BACKEND_NONE) {
        err = "kernel op descriptor has missing op_kind, impl_id, package_id, or backend";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& existing : ops_) {
        if (existing->impl_id == desc.impl_id) {
            err = "duplicate kernel operation registration for impl_id: " + existing->impl_id;
            return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
        }
    }
    auto stored = std::make_unique<StoredOp>();
    stored->desc = desc;
    stored->op_name = desc.op_name;
    stored->impl_id = desc.impl_id;
    stored->package_id = desc.package_id;
    stored->desc.op_name = stored->op_name.c_str();
    stored->desc.impl_id = stored->impl_id.c_str();
    stored->desc.package_id = stored->package_id.c_str();
    if (desc.required_op_deps && desc.n_required_op_deps > 0) {
        stored->dep_storage.reserve(desc.n_required_op_deps);
        for (size_t i = 0; i < desc.n_required_op_deps; ++i) {
            if (!desc.required_op_deps[i] || !desc.required_op_deps[i][0]) {
                err = "kernel op descriptor contains null or empty dependency ID";
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            }
            stored->dep_storage.emplace_back(desc.required_op_deps[i]);
        }
        stored->dep_ptrs.reserve(stored->dep_storage.size());
        for (const auto& s : stored->dep_storage) stored->dep_ptrs.push_back(s.c_str());
        stored->desc.required_op_deps = stored->dep_ptrs.data();
    } else {
        stored->desc.required_op_deps = nullptr;
        stored->desc.n_required_op_deps = 0;
    }
    ops_.push_back(std::move(stored));
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t KernelRegistry::register_plugin(guild_plugin_entry_fn entry_fn, std::string& err) {
    if (!entry_fn) {
        err = "plugin entry function is null";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    guild_plugin_descriptor_v1_t pdesc{};
    guild_error_info_t einfo{};
    const guild_status_t st = entry_fn(GUILD_KERNEL_ABI_VERSION_V1, &pdesc, &einfo);
    if (st != GUILD_STATUS_OK) {
        err = einfo.message[0] ? einfo.message : status_to_string(st);
        return st;
    }
    if (pdesc.struct_size < sizeof(guild_plugin_descriptor_v1_t) ||
        negotiate_abi_version(pdesc.abi_version, nullptr) != GUILD_STATUS_OK) {
        err = "plugin descriptor ABI version mismatch";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (!pdesc.package_id || !pdesc.package_id[0] || !pdesc.init) {
        err = "plugin descriptor missing package_id or init callback";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    guild_host_api_v1_t host_api{};
    host_api.struct_size = sizeof(guild_host_api_v1_t);
    host_api.core_abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    host_api.negotiate_abi = &negotiate_abi_version;
    host_api.get_device_capabilities = [](guild_core_host_t, guild_device_capabilities_v1_t* out_caps) -> guild_status_t {
        if (!out_caps) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out_caps = query_host_device_capabilities();
        return GUILD_STATUS_OK;
    };
    host_api.register_kernel_op = [](guild_core_host_t host, const guild_kernel_op_desc_v1_t* d, guild_error_info_t* e) -> guild_status_t {
        if (!host || !d) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        std::string msg;
        const auto s = reinterpret_cast<KernelRegistry*>(host)->register_op(*d, msg);
        if (s != GUILD_STATUS_OK) fill_err(e, s, d->impl_id ? d->impl_id : "plugin", msg);
        return s;
    };
    host_api.register_runner = nullptr;
    const guild_status_t init_st = pdesc.init(reinterpret_cast<guild_core_host_t>(this), &host_api, &einfo);
    if (init_st != GUILD_STATUS_OK) {
        err = einfo.message[0] ? einfo.message : status_to_string(init_st);
        return init_st;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    LoadedPlugin lp;
    lp.desc = pdesc;
    lp.package_id = pdesc.package_id;
    lp.package_version = pdesc.package_version ? pdesc.package_version : "1.0.0";
    lp.publisher_id = pdesc.publisher_id ? pdesc.publisher_id : "builtin";
    lp.desc.package_id = lp.package_id.c_str();
    lp.desc.package_version = lp.package_version.c_str();
    lp.desc.publisher_id = lp.publisher_id.c_str();
    plugins_.push_back(std::move(lp));
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t KernelRegistry::unregister_package(const std::string& package_id, std::string& err) {
    if (active_sessions() > 0) {
        err = "cannot unregister kernel package '" + package_id + "' while active sessions are running";
        return GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t before = ops_.size();
    ops_.erase(std::remove_if(ops_.begin(), ops_.end(), [&](const std::unique_ptr<StoredOp>& op) {
        return op->package_id == package_id;
    }), ops_.end());
    if (ops_.size() == before) {
        err = "kernel package not found: " + package_id;
        return GUILD_STATUS_ERR_MISSING_KERNEL;
    }
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t KernelRegistry::validate_dependencies(std::string& err) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& op : ops_) {
        for (const auto& dep_id : op->dep_storage) {
            const bool found = std::any_of(ops_.begin(), ops_.end(), [&](const std::unique_ptr<StoredOp>& cand) {
                return cand->impl_id == dep_id;
            });
            if (!found) {
                err = "kernel '" + op->impl_id + "' requires missing operation dependency '" + dep_id + "'";
                return GUILD_STATUS_ERR_DEPENDENCY_MISSING;
            }
        }
    }
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t KernelRegistry::select(const KernelSelectionQuery& query,
                                      const guild_device_capabilities_v1_t& caps,
                                      const guild_kernel_op_desc_v1_t** out_desc,
                                      std::string& err) const {
    if (out_desc) *out_desc = nullptr;
    if (query.op_kind <= GUILD_OP_INVALID || query.op_kind >= GUILD_OP_COUNT) {
        err = "invalid operation kind in kernel selection query";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const StoredOp* best = nullptr;
    bool saw_op_kind = false;
    bool failed_hw = false;
    bool failed_caps = false;
    bool failed_mem = false;

    for (const auto& cand : ops_) {
        const auto& d = cand->desc;
        if (d.op_kind != query.op_kind) continue;
        if (!query.frozen_impl_id.empty()) {
            if (cand->impl_id != query.frozen_impl_id) continue;
            if (query.frozen_impl_version != 0 && d.impl_version != query.frozen_impl_version) {
                failed_caps = true;
                continue;
            }
        }
        saw_op_kind = true;

        // Check backend & hardware ISA / SM
        if ((d.backend & query.allowed_backends) == 0 || (d.backend & caps.available_backends) == 0) {
            failed_hw = true;
            continue;
        }
        if ((d.required_cpu_isa & caps.cpu_isa_flags) != d.required_cpu_isa) {
            failed_hw = true;
            continue;
        }
        if ((d.required_gpu_caps & caps.gpu_cap_flags) != d.required_gpu_caps) {
            failed_hw = true;
            continue;
        }
        if (d.min_sm_version > 0 && caps.gpu_sm_version < d.min_sm_version) {
            failed_hw = true;
            continue;
        }
        if (d.max_sm_version > 0 && caps.gpu_sm_version > d.max_sm_version) {
            failed_hw = true;
            continue;
        }

        // Check dtypes, dimensions, numerical guarantees, sync modes
        if (query.weight_dtype != GUILD_DTYPE_ANY && d.weight_dtype != GUILD_DTYPE_ANY && d.weight_dtype != query.weight_dtype) {
            failed_caps = true;
            continue;
        }
        if (query.input_dtype != GUILD_DTYPE_ANY && d.input_dtype != GUILD_DTYPE_ANY && d.input_dtype != query.input_dtype) {
            failed_caps = true;
            continue;
        }
        if (query.output_dtype != GUILD_DTYPE_ANY && d.output_dtype != GUILD_DTYPE_ANY && d.output_dtype != query.output_dtype) {
            failed_caps = true;
            continue;
        }
        if (query.accum_dtype != GUILD_DTYPE_ANY && d.accum_dtype != GUILD_DTYPE_ANY && d.accum_dtype != query.accum_dtype) {
            failed_caps = true;
            continue;
        }
        if (d.max_batch_rows > 0 && query.batch_rows > d.max_batch_rows) {
            failed_caps = true;
            continue;
        }
        if (query.dim_n > 0 && ((d.min_dim_n > 0 && query.dim_n < d.min_dim_n) ||
                                (d.max_dim_n > 0 && query.dim_n > d.max_dim_n))) {
            failed_caps = true;
            continue;
        }
        if (query.dim_k > 0 && ((d.min_dim_k > 0 && query.dim_k < d.min_dim_k) ||
                                (d.max_dim_k > 0 && query.dim_k > d.max_dim_k))) {
            failed_caps = true;
            continue;
        }
        if ((d.numerical_flags & query.required_numerical_flags) != query.required_numerical_flags) {
            failed_caps = true;
            continue;
        }
        if (query.required_sync_mode != GUILD_SYNC_NONE &&
            (d.supported_sync_modes & query.required_sync_mode) == 0) {
            failed_caps = true;
            continue;
        }

        // Check scratch memory if query_scratch is provided
        if (d.query_scratch) {
            guild_scratch_request_t sreq{};
            sreq.struct_size = sizeof(sreq);
            if (d.query_scratch(nullptr, 0, nullptr, 0, &sreq, nullptr) == GUILD_STATUS_OK) {
                const uint64_t vram_limit = std::min(query.available_vram_bytes, caps.available_vram_bytes);
                const uint64_t ram_limit = std::min(query.available_ram_bytes, caps.available_ram_bytes);
                if (sreq.device_bytes > vram_limit ||
                    sreq.pinned_host_bytes + sreq.pageable_host_bytes > ram_limit) {
                    failed_mem = true;
                    continue;
                }
            }
        }

        // Check operation dependencies
        bool deps_ok = true;
        for (const auto& dep_id : cand->dep_storage) {
            const bool dep_present = std::any_of(ops_.begin(), ops_.end(), [&](const std::unique_ptr<StoredOp>& o) {
                return o->impl_id == dep_id;
            });
            if (!dep_present) {
                deps_ok = false;
                break;
            }
        }
        if (!deps_ok) {
            err = "candidate kernel '" + cand->impl_id + "' has unsatisfied operation dependency";
            return GUILD_STATUS_ERR_DEPENDENCY_MISSING;
        }

        if (!best || d.priority > best->desc.priority) {
            best = cand.get();
        }
    }

    if (!best) {
        if (!saw_op_kind) {
            err = std::string("no registered kernel for operation '") + op_kind_to_string(query.op_kind) + "'";
            if (!query.frozen_impl_id.empty()) err += " (frozen impl_id '" + query.frozen_impl_id + "')";
            return GUILD_STATUS_ERR_MISSING_KERNEL;
        }
        if (failed_mem) {
            err = std::string("scratch memory requirement exceeds available budget for '") + op_kind_to_string(query.op_kind) + "'";
            return GUILD_STATUS_ERR_OUT_OF_MEMORY;
        }
        if (failed_hw && !failed_caps) {
            err = std::string("no compatible hardware backend/ISA for operation '") + op_kind_to_string(query.op_kind) + "'";
            return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
        }
        err = std::string("no kernel matched capability/shape/dtype constraints for '") + op_kind_to_string(query.op_kind) + "'";
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }

    if (out_desc) *out_desc = &best->desc;
    err.clear();
    return GUILD_STATUS_OK;
}

const guild_kernel_op_desc_v1_t* KernelRegistry::find_by_id(const std::string& impl_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& op : ops_) {
        if (op->impl_id == impl_id) return &op->desc;
    }
    return nullptr;
}

std::vector<guild_kernel_op_desc_v1_t> KernelRegistry::list_operations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<guild_kernel_op_desc_v1_t> out;
    out.reserve(ops_.size());
    for (const auto& op : ops_) out.push_back(op->desc);
    return out;
}

bool KernelRegistry::resolve_prefill_kernels(const guild_device_capabilities_v1_t& caps,
                                             ResolvedPrefillKernels& out_table,
                                             std::string& err) const {
    out_table = {};
    const guild_op_kind_t required_ops[] = {
        GUILD_OP_QUANTIZE_ACT,
        GUILD_OP_QUANTIZED_GEMM,
        GUILD_OP_RMS_NORM,
        GUILD_OP_GDN_CONV_L2,
        GUILD_OP_GDN_RECURRENCE_STEP,
        GUILD_OP_GDN_OUT_NORM,
        GUILD_OP_ROPE_APPLY,
        GUILD_OP_KV_APPEND,
        GUILD_OP_ATTENTION_DECODE,
        GUILD_OP_MOE_ROUTER_TOPK,
        GUILD_OP_SHARED_EXPERT,
        GUILD_OP_BATCHED_EXPERT_EXEC,
        GUILD_OP_MOE_COMBINE,
        GUILD_OP_FINITE_CHECK
    };
    for (guild_op_kind_t op : required_ops) {
        KernelSelectionQuery q;
        q.op_kind = op;
        q.required_numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC;
        const guild_kernel_op_desc_v1_t* d = nullptr;
        if (select(q, caps, &d, err) != GUILD_STATUS_OK || !d) return false;
        if (d->direct_fn_table) {
            out_table = *static_cast<const ResolvedPrefillKernels*>(d->direct_fn_table);
        }
    }
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
    if (!out_table.all_bound()) {
        out_table = kBuiltinPrefillKernels;
    }
#endif
    if (!out_table.all_bound()) {
        err = "prefill kernel resolution did not bind all required device entry points";
        return false;
    }
    return true;
}

void KernelRegistry::register_builtin_kernels() {
    auto make_base = [](guild_op_kind_t kind, const char* impl_id, const char* pkg,
                        guild_backend_t backend, int32_t prio) {
        guild_kernel_op_desc_v1_t d{};
        d.struct_size = sizeof(d);
        d.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        d.op_kind = kind;
        d.op_name = op_kind_to_string(kind);
        d.impl_id = impl_id;
        d.package_id = pkg;
        d.impl_version = GUILD_KERNEL_ABI_VERSION_V1;
        d.backend = backend;
        d.weight_dtype = GUILD_DTYPE_ANY;
        d.input_dtype = GUILD_DTYPE_ANY;
        d.output_dtype = GUILD_DTYPE_F32;
        d.accum_dtype = GUILD_DTYPE_F32;
        d.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC | GUILD_NUM_EXACT_SINGLE_COL_EQUIV | GUILD_NUM_FINITE_VALIDATED;
        d.supported_sync_modes = GUILD_SYNC_NONE | GUILD_SYNC_STREAM_ORDERED | GUILD_SYNC_HOST_BLOCKING | GUILD_SYNC_GRAPH_CAPTURE;
        d.priority = prio;
        return d;
    };

    std::string ignore_err;
    // Built-in CPU kernels (package: builtin.guild-kernels-cpu)
    {
        auto d = make_base(GUILD_OP_BATCHED_EXPERT_EXEC, "cpu.expert_pool.split_multi_native", "builtin.guild-kernels-cpu", GUILD_BACKEND_CPU, 100);
        d.required_cpu_isa = GUILD_ISA_AVX2;
        d.max_batch_rows = 512;
        d.max_dim_n = 2560;
        d.max_dim_k = 2560;
        register_op(d, ignore_err);
    }
    {
        auto d = make_base(GUILD_OP_BATCHED_EXPERT_EXEC, "cpu.expert_pool.split_multi_canonical_avx512", "builtin.guild-kernels-cpu", GUILD_BACKEND_CPU, 90);
        d.required_cpu_isa = GUILD_ISA_AVX512F | GUILD_ISA_AVX512BW | GUILD_ISA_AVX512DQ | GUILD_ISA_AVX512VL;
        d.weight_dtype = GUILD_DTYPE_Q2_0;
        d.max_batch_rows = 512;
        d.max_dim_n = 2560;
        d.max_dim_k = 2560;
        register_op(d, ignore_err);
    }
    {
        auto d = make_base(GUILD_OP_QUANTIZE_ACT, "cpu.quantize.act_q8_k_avx2", "builtin.guild-kernels-cpu", GUILD_BACKEND_CPU, 80);
        d.required_cpu_isa = GUILD_ISA_AVX2;
        d.input_dtype = GUILD_DTYPE_F32;
        d.output_dtype = GUILD_DTYPE_Q8_K;
        register_op(d, ignore_err);
    }
    {
        auto d = make_base(GUILD_OP_FINITE_CHECK, "cpu.finite_check.simd64", "builtin.guild-kernels-cpu", GUILD_BACKEND_CPU, 100);
        d.input_dtype = GUILD_DTYPE_F32;
        d.output_dtype = GUILD_DTYPE_I32;
        d.execute = &execute_cpu_finite_check;
        register_op(d, ignore_err);
    }

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
    const guild_backend_t gpu_backend =
#if defined(GUILD_ENABLE_HIP)
        GUILD_BACKEND_HIP;
#else
        GUILD_BACKEND_CUDA;
#endif
    const char* gpu_pkg =
#if defined(GUILD_ENABLE_HIP)
        "builtin.guild-kernels-amd";
#else
        "builtin.guild-kernels-cuda";
#endif

    auto register_gpu = [&](guild_op_kind_t kind, const char* id, int64_t max_rows, guild_dtype_t w_dt, guild_dtype_t in_dt, guild_dtype_t out_dt) {
        auto d = make_base(kind, id, gpu_pkg, gpu_backend, 200);
        d.min_sm_version = 60;
        d.max_batch_rows = max_rows;
        d.weight_dtype = w_dt;
        d.input_dtype = in_dt;
        d.output_dtype = out_dt;
        d.direct_fn_table = &kBuiltinPrefillKernels;
        register_op(d, ignore_err);
    };

    register_gpu(GUILD_OP_EMBEDDING_GATHER, "gpu.embed.native_gather", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_I32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_QUANTIZE_ACT, "gpu.quantize.q8_1_8col", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_Q8_1);
    register_gpu(GUILD_OP_DEQUANTIZE, "gpu.dequant.bf16_native", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_ANY, GUILD_DTYPE_BF16);
    register_gpu(GUILD_OP_QUANTIZED_GEMV, "gpu.mmvq.native_1col", 1, GUILD_DTYPE_ANY, GUILD_DTYPE_Q8_1, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_QUANTIZED_GEMM, "gpu.mmvq.native_8col", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_Q8_1, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_GROUPED_MOE_GEMM, "gpu.moe.grouped_hit_native", 128, GUILD_DTYPE_ANY, GUILD_DTYPE_Q8_1, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_MOE_ROUTER_TOPK, "gpu.moe.router_topk_native", 512, GUILD_DTYPE_F32, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_MOE_DISPATCH, "gpu.moe.dispatch_hybrid", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_SHARED_EXPERT, "gpu.moe.shared_expert_bf16", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_BF16, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_MOE_COMBINE, "gpu.moe.combine_native", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_RMS_NORM, "gpu.norm.qsa_rms_weighted", 512, GUILD_DTYPE_F32, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_GATED_RESIDUAL, "gpu.gr.fused_read_write", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_ROPE_APPLY, "gpu.rope.native_apply", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_KV_APPEND, "gpu.kv.append_step", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F16);
    register_gpu(GUILD_OP_KV_GATHER, "gpu.kv.gather_step", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_F16, GUILD_DTYPE_F16);
    register_gpu(GUILD_OP_QSA_INDEXER_SELECT, "gpu.qsa.indexer_topk", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_I32);
    register_gpu(GUILD_OP_ATTENTION_DECODE, "gpu.attention.qsa_decode_batch", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_ATTENTION_PREFILL, "gpu.attention.qsa_prompt_flash", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_GDN_CONV_L2, "gpu.gdn.prefill_conv_l2", 512, GUILD_DTYPE_F32, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_GDN_AB_PROJ, "gpu.gdn.ab_proj", 8, GUILD_DTYPE_BF16, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_GDN_RECURRENCE_STEP, "gpu.gdn.prefill_recurrence", 512, GUILD_DTYPE_F32, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_GDN_OUT_NORM, "gpu.gdn.out_norm_native", 512, GUILD_DTYPE_F32, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_PLE_COMPUTE, "gpu.ple.native_postops", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_SPEC_VERIFY, "gpu.verify.window_graph", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_I32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_FINITE_CHECK, "gpu.finite.first_bad", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_I32);
#endif
}

} // namespace guild::kernels
