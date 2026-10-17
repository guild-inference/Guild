// src/kernels/registry.cpp - Hardware Kernel Operation Registry & Selection
#include "guild/kernels/registry.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include "guild/core/validation.hpp"
#include "guild/abi/runner_abi.h"
#include "guild/abi/cpu_expert_abi.h"
#include "guild/abi/cpu_backend_abi.h"

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
#include "guild/kernels/native_mmvq.hpp"
#include "guild/kernels/native_qsa.hpp"
#include "guild/kernels/native_rope.hpp"
#include "guild/kernels/native_moe.hpp"
#include "guild/kernels/native_gdn.hpp"
#include "guild/kernels/native_gdn_preprocess.hpp"
#include "guild/kernels/verify_kernels.hpp"
#include "guild/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace guild::kernels {
#if GUILD_BUNDLED_CPU_ARITHMETIC && defined(GUILD_NATIVE_EXPERTS)
extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_backend_table_v2();
#endif
namespace {

size_t cpu_extension_size(const void* payload) {
    uint32_t size=0;
    std::memcpy(&size,payload,sizeof(size));
    if(size<sizeof(guild_cpu_expert_ext_v1_t)) return 0;
    guild_cpu_expert_ext_v1_t native{};
    std::memcpy(&native,payload,sizeof(native));
    if(native.tag!=GUILD_CPU_EXPERT_TAG_V1 || native.version!=1 ||
       !native.quant_act || !native.quant_h || !native.gate_up || !native.down) return 0;
    if(size==sizeof(native)) return sizeof(native);
    constexpr size_t base_size=offsetof(guild_cpu_backend_ext_v2_t,swiglu);
    if(size<base_size) return 0;
    guild_cpu_backend_ext_v2_t full{};
    const size_t bytes=size>=sizeof(full)?sizeof(full):base_size;
    if(size!=base_size && size<sizeof(full))return 0;
    std::memcpy(&full,payload,bytes);
    if(full.backend_version!=GUILD_CPU_BACKEND_VERSION_V2 || !full.format || !full.quant_planar ||
       !full.canonical_gate_up || !full.canonical_down || !full.q2_gguf_rows || !full.router_bf16 || !full.combine) return 0;
    if(bytes==sizeof(full) && !full.swiglu)return 0;
    return bytes;
}

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

bool read_prefill_extension(const guild_kernel_op_desc_v1_t& desc,
                            PrefillKernelExtensionV1& extension, std::string& err) {
    const auto incompatible = [&](const char* reason) {
        err = std::string("prefill kernel '") + desc.impl_id + "' (" + op_kind_to_string(desc.op_kind) +
              "): incompatible prefill extension: " + reason;
        return false;
    };
    if (!desc.next_ext) return incompatible("missing next_ext projection payload");
    uint32_t size = 0;
    std::memcpy(&size, desc.next_ext, sizeof(size));
    if (size < sizeof(PrefillKernelExtensionHeader))
        return incompatible("struct_size is smaller than the extension header");
    PrefillKernelExtensionHeader header;
    std::memcpy(&header, desc.next_ext, sizeof(header));
    if (header.tag != kPrefillKernelExtensionTag)
        return incompatible("unknown extension tag");
    if (header.version != kPrefillKernelExtensionVersion)
        return incompatible("unsupported extension version");
    if (size < sizeof(PrefillKernelExtensionV1))
        return incompatible("struct_size is smaller than the v1 projection payload");
    // Only the validated v1 prefix is read, even if a future payload has extra bytes.
    std::memcpy(&extension, desc.next_ext, sizeof(extension));
    if (desc.op_kind == GUILD_OP_GDN_AB_PROJ && !extension.gdn_ab)
        return incompatible("missing gdn_ab entry point");
    if (desc.op_kind == GUILD_OP_MOE_ROUTER_TOPK && !extension.router_projection)
        return incompatible("missing router_projection entry point");
    return true;
}

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
void gdn_ab_prefill_serial(const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt, const float* a,
                            float* gate, float* beta, int n, int heads, int rows, void* stream) {
    for (int t=0;t<rows;t+=kVerifyMaxT)
        gdn_ab_multi(x+size_t(t)*n,wa,wb,dt,a,gate+size_t(t)*heads,beta+size_t(t)*heads,
                     n,heads,std::min(kVerifyMaxT,rows-t),stream);
}
const PrefillKernelTableV1 kBuiltinPrefillKernels{
    &native_quantize_q8_1,
    &native_mmvq,
    &native_qsa_rms_norm_weighted,
    &gdn_prefill_conv,
    &gdn_prefill_recurrence,
    &native_gdn_out_norm,
    &native_rope_apply
};
const PrefillKernelExtensionV1 kBuiltinPrefillExtension = [] {
    PrefillKernelExtensionV1 extension;
    extension.gdn_ab = &gdn_ab_prefill_serial;
    extension.router_projection = &bf16_gemv_fp32_mmvf_cols;
    return extension;
}();
const PrefillKernelExtensionV1 kTiledPrefillExtension = [] {
    auto extension = kBuiltinPrefillExtension;
    extension.gdn_ab = &gdn_ab_prefill;
    return extension;
}();
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
    if (hw.isa_avx2) caps.cpu_isa_flags |= GUILD_ISA_AVX2 | GUILD_ISA_FMA | GUILD_ISA_F16C;
    if (hw.isa_bmi2) caps.cpu_isa_flags |= GUILD_ISA_BMI2;
    if (hw.isa_avx512) caps.cpu_isa_flags |= GUILD_ISA_AVX512F | GUILD_ISA_AVX512BW | GUILD_ISA_AVX512DQ | GUILD_ISA_AVX512VL;
    if (hw.isa_vnni) caps.cpu_isa_flags |= GUILD_ISA_AVX512VNNI;
    if (hw.isa_vbmi) caps.cpu_isa_flags |= GUILD_ISA_AVX512VBMI;
    if (hw.isa_avx512cd) caps.cpu_isa_flags |= GUILD_ISA_AVX512CD;
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
    // Deep-copy legacy prefill function pointer table for operations that bind from it,
    // so plugin descriptors constructed on the stack during init() never leave dangling pointers.
    if (desc.direct_fn_table != nullptr) {
        switch (desc.op_kind) {
        case GUILD_OP_QUANTIZE_ACT:
        case GUILD_OP_QUANTIZED_GEMM:
        case GUILD_OP_RMS_NORM:
        case GUILD_OP_GDN_CONV_L2:
        case GUILD_OP_GDN_RECURRENCE_STEP:
        case GUILD_OP_GDN_OUT_NORM:
        case GUILD_OP_ROPE_APPLY:
            stored->owned_direct_table = std::make_unique<PrefillKernelTableV1>();
            std::memcpy(stored->owned_direct_table.get(), desc.direct_fn_table, sizeof(PrefillKernelTableV1));
            stored->desc.direct_fn_table = stored->owned_direct_table.get();
            break;
        default:
            break;
        }
    }
    // Deep-copy prefill extension payload respecting exact validated prefix bounds.
    if (desc.next_ext && desc.op_kind == GUILD_OP_BATCHED_EXPERT_EXEC) {
        const size_t bytes=cpu_extension_size(desc.next_ext);
        if (!bytes) {
            err="invalid CPU expert extension"; return GUILD_STATUS_ERR_ABI_MISMATCH;
        }
        stored->owned_ext_bytes.resize(bytes);
        std::memcpy(stored->owned_ext_bytes.data(),desc.next_ext,bytes);
        stored->desc.next_ext=stored->owned_ext_bytes.data();
    }
    if (desc.next_ext != nullptr &&
        (desc.op_kind == GUILD_OP_GDN_AB_PROJ || desc.op_kind == GUILD_OP_MOE_ROUTER_TOPK)) {
        uint32_t ext_size = 0;
        std::memcpy(&ext_size, desc.next_ext, sizeof(ext_size));
        size_t copy_bytes = sizeof(uint32_t);
        if (ext_size >= sizeof(PrefillKernelExtensionHeader)) {
            PrefillKernelExtensionHeader hdr{};
            std::memcpy(&hdr, desc.next_ext, sizeof(hdr));
            if (hdr.tag != kPrefillKernelExtensionTag || hdr.version != kPrefillKernelExtensionVersion) {
                copy_bytes = sizeof(PrefillKernelExtensionHeader);
            } else if (ext_size < sizeof(PrefillKernelExtensionV1)) {
                copy_bytes = ext_size;
            } else {
                copy_bytes = sizeof(PrefillKernelExtensionV1);
            }
        }
        stored->owned_ext_bytes.resize(copy_bytes);
        std::memcpy(stored->owned_ext_bytes.data(), desc.next_ext, copy_bytes);
        stored->desc.next_ext = stored->owned_ext_bytes.data();
    }
    ops_.push_back(std::move(stored));
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t KernelRegistry::register_plugin(guild_plugin_entry_fn entry_fn, std::string& err) {
    return register_plugin_transactional(entry_fn, "", "", "", {}, {}, nullptr, nullptr, err, nullptr);
}

guild_status_t KernelRegistry::register_plugin_transactional(
    guild_plugin_entry_fn entry_fn,
    const std::string& expected_package_id,
    const std::string& expected_version,
    const std::string& expected_publisher,
    const std::vector<std::string>& declared_ops,
    const std::vector<std::string>& declared_runners,
    std::function<guild_status_t(const guild_runner_desc_v1_t&, std::string&)> runner_commit_fn,
    std::function<void(const std::string&)> runner_rollback_fn,
    std::string& err,
    std::shared_ptr<void> library_keepalive) {
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
    const std::string pkg_id = pdesc.package_id;
    const std::string pkg_ver = pdesc.package_version ? pdesc.package_version : "1.0.0";
    const std::string pub_id = pdesc.publisher_id ? pdesc.publisher_id : "builtin";
    if (!expected_package_id.empty() && pkg_id != expected_package_id) {
        err = "plugin descriptor package_id '" + pkg_id + "' does not match .gkmetadata package_id '" + expected_package_id + "'";
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }
    if (!expected_version.empty() && pkg_ver != expected_version) {
        err = "plugin descriptor version '" + pkg_ver + "' does not match .gkmetadata version '" + expected_version + "'";
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }
    if (!expected_publisher.empty() && pub_id != expected_publisher) {
        err = "plugin descriptor publisher_id '" + pub_id + "' does not match .gkmetadata publisher_id '" + expected_publisher + "'";
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& existing_lp : plugins_) {
            if (existing_lp.package_id == pkg_id) {
                err = "duplicate plugin registration for package_id: " + pkg_id;
                return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
            }
        }
    }

    struct StagingContext {
        KernelRegistry* self = nullptr;
        std::string package_id;
        std::vector<guild_kernel_op_desc_v1_t> staged_ops;
        std::vector<std::string> staged_op_names;
        std::vector<std::string> staged_impl_ids;
        std::vector<std::string> staged_pkg_ids;
        std::vector<std::vector<std::string>> staged_op_deps;
        std::vector<std::vector<const char*>> staged_op_dep_ptrs;
        std::vector<std::unique_ptr<PrefillKernelTableV1>> staged_direct_tables;
        std::vector<std::vector<uint8_t>> staged_ext_bytes;

        std::vector<guild_runner_desc_v1_t> staged_runners;
        std::vector<std::string> staged_runner_archs;
        std::vector<std::string> staged_runner_ids;
        std::vector<std::string> staged_runner_pkgs;
        std::vector<std::vector<guild_op_kind_t>> staged_runner_ops;
        std::vector<std::vector<std::string>> staged_runner_pkg_deps;
        std::vector<std::vector<const char*>> staged_runner_pkg_dep_ptrs;
        std::vector<std::unique_ptr<guild_runner_model_ext_v1_t>> staged_runner_extensions;
    } staging;
    staging.self = this;
    staging.package_id = pkg_id;

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
        auto* ctx = reinterpret_cast<StagingContext*>(host);
        if (d->struct_size < sizeof(guild_kernel_op_desc_v1_t) ||
            negotiate_abi_version(d->abi_version, nullptr) != GUILD_STATUS_OK) {
            fill_err(e, GUILD_STATUS_ERR_ABI_MISMATCH, "plugin", "kernel op descriptor ABI mismatch");
            return GUILD_STATUS_ERR_ABI_MISMATCH;
        }
        if (d->op_kind <= GUILD_OP_INVALID || d->op_kind >= GUILD_OP_COUNT ||
            !d->op_name || !d->op_name[0] || !d->impl_id || !d->impl_id[0] ||
            !d->package_id || !d->package_id[0] || d->backend == GUILD_BACKEND_NONE) {
            fill_err(e, GUILD_STATUS_ERR_INVALID_ARGUMENT, "plugin", "kernel op descriptor missing required fields");
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
        if (ctx->package_id != d->package_id) {
            fill_err(e, GUILD_STATUS_ERR_SECURITY_VIOLATION, d->impl_id, "kernel op package_id does not match owning plugin package_id");
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
        if (ctx->self->find_by_id(d->impl_id) != nullptr) {
            fill_err(e, GUILD_STATUS_ERR_DUPLICATE_REGISTRATION, d->impl_id, "duplicate kernel impl_id in registry");
            return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
        }
        for (const auto& existing_id : ctx->staged_impl_ids) {
            if (existing_id == d->impl_id) {
                fill_err(e, GUILD_STATUS_ERR_DUPLICATE_REGISTRATION, d->impl_id, "duplicate kernel impl_id in plugin transaction");
                return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
            }
        }
        ctx->staged_op_names.emplace_back(d->op_name);
        ctx->staged_impl_ids.emplace_back(d->impl_id);
        ctx->staged_pkg_ids.emplace_back(d->package_id);
        std::vector<std::string> deps;
        if (d->required_op_deps && d->n_required_op_deps > 0) {
            for (size_t i = 0; i < d->n_required_op_deps; ++i) {
                if (!d->required_op_deps[i] || !d->required_op_deps[i][0]) {
                    fill_err(e, GUILD_STATUS_ERR_INVALID_ARGUMENT, d->impl_id, "null or empty required_op_deps entry");
                    return GUILD_STATUS_ERR_INVALID_ARGUMENT;
                }
                deps.emplace_back(d->required_op_deps[i]);
            }
        }
        ctx->staged_op_deps.push_back(std::move(deps));
        ctx->staged_op_dep_ptrs.emplace_back();
        if (d->direct_fn_table != nullptr) {
            auto tbl = std::make_unique<PrefillKernelTableV1>();
            std::memcpy(tbl.get(), d->direct_fn_table, sizeof(PrefillKernelTableV1));
            ctx->staged_direct_tables.push_back(std::move(tbl));
        } else {
            ctx->staged_direct_tables.push_back(nullptr);
        }
        std::vector<uint8_t> ext_copy;
        if (d->next_ext && d->op_kind == GUILD_OP_BATCHED_EXPERT_EXEC) {
            const size_t bytes=cpu_extension_size(d->next_ext);
            if(!bytes) return GUILD_STATUS_ERR_ABI_MISMATCH;
            ext_copy.resize(bytes);
            std::memcpy(ext_copy.data(),d->next_ext,bytes);
        }
        if (d->next_ext != nullptr &&
            (d->op_kind == GUILD_OP_GDN_AB_PROJ || d->op_kind == GUILD_OP_MOE_ROUTER_TOPK)) {
            uint32_t ext_size = 0;
            std::memcpy(&ext_size, d->next_ext, sizeof(ext_size));
            size_t copy_bytes = sizeof(uint32_t);
            if (ext_size >= sizeof(PrefillKernelExtensionHeader)) {
                PrefillKernelExtensionHeader hdr{};
                std::memcpy(&hdr, d->next_ext, sizeof(hdr));
                if (hdr.tag != kPrefillKernelExtensionTag || hdr.version != kPrefillKernelExtensionVersion) {
                    copy_bytes = sizeof(PrefillKernelExtensionHeader);
                } else if (ext_size < sizeof(PrefillKernelExtensionV1)) {
                    copy_bytes = ext_size;
                } else {
                    copy_bytes = sizeof(PrefillKernelExtensionV1);
                }
            }
            ext_copy.resize(copy_bytes);
            std::memcpy(ext_copy.data(), d->next_ext, copy_bytes);
        }
        ctx->staged_ext_bytes.push_back(std::move(ext_copy));
        ctx->staged_ops.push_back(*d);
        return GUILD_STATUS_OK;
    };
    host_api.register_runner = [](guild_core_host_t host, const guild_runner_desc_v1_t* r, guild_error_info_t* e) -> guild_status_t {
        if (!host || !r) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        auto* ctx = reinterpret_cast<StagingContext*>(host);
        if (r->struct_size < sizeof(guild_runner_desc_v1_t) ||
            negotiate_abi_version(r->abi_version, nullptr) != GUILD_STATUS_OK) {
            fill_err(e, GUILD_STATUS_ERR_ABI_MISMATCH, "plugin", "runner descriptor ABI mismatch");
            return GUILD_STATUS_ERR_ABI_MISMATCH;
        }
        if (!r->arch_id || !r->arch_id[0] || !r->runner_id || !r->runner_id[0] ||
            !r->package_id || !r->package_id[0]) {
            fill_err(e, GUILD_STATUS_ERR_INVALID_ARGUMENT, "plugin", "runner descriptor missing required fields");
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
        if (ctx->package_id != r->package_id) {
            fill_err(e, GUILD_STATUS_ERR_SECURITY_VIOLATION, r->runner_id, "runner package_id does not match owning plugin package_id");
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
        for (const auto& existing_rid : ctx->staged_runner_ids) {
            if (existing_rid == r->runner_id) {
                fill_err(e, GUILD_STATUS_ERR_DUPLICATE_REGISTRATION, r->runner_id, "duplicate runner_id in plugin transaction");
                return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
            }
        }
        ctx->staged_runner_archs.emplace_back(r->arch_id);
        ctx->staged_runner_ids.emplace_back(r->runner_id);
        ctx->staged_runner_pkgs.emplace_back(r->package_id);
        std::vector<guild_op_kind_t> rops;
        if (r->required_op_kinds && r->n_required_op_kinds > 0) {
            rops.assign(r->required_op_kinds, r->required_op_kinds + r->n_required_op_kinds);
        }
        ctx->staged_runner_ops.push_back(std::move(rops));
        std::vector<std::string> pdeps;
        if (r->required_package_deps && r->n_required_package_deps > 0) {
            for (size_t i = 0; i < r->n_required_package_deps; ++i) {
                if (r->required_package_deps[i]) pdeps.emplace_back(r->required_package_deps[i]);
            }
        }
        ctx->staged_runner_pkg_deps.push_back(std::move(pdeps));
        ctx->staged_runner_pkg_dep_ptrs.emplace_back();
        std::unique_ptr<guild_runner_model_ext_v1_t> ext;
        if (r->next_ext) {
            uint32_t size=0;
            std::memcpy(&size, r->next_ext, sizeof(size));
            if (size < sizeof(guild_runner_model_ext_v1_t)) {
                fill_err(e, GUILD_STATUS_ERR_ABI_MISMATCH, "plugin", "truncated runner model extension");
                return GUILD_STATUS_ERR_ABI_MISMATCH;
            }
            ext=std::make_unique<guild_runner_model_ext_v1_t>();
            std::memcpy(ext.get(), r->next_ext, sizeof(*ext));
            if (ext->tag != GUILD_RUNNER_MODEL_TAG_V1 || ext->version != GUILD_RUNNER_MODEL_VERSION_V1 ||
                !ext->recognize || !ext->describe) {
                fill_err(e, GUILD_STATUS_ERR_ABI_MISMATCH, "plugin", "invalid runner model extension");
                return GUILD_STATUS_ERR_ABI_MISMATCH;
            }
        }
        ctx->staged_runner_extensions.push_back(std::move(ext));
        ctx->staged_runners.push_back(*r);
        return GUILD_STATUS_OK;
    };

    const guild_status_t init_st = pdesc.init(reinterpret_cast<guild_core_host_t>(&staging), &host_api, &einfo);
    if (init_st != GUILD_STATUS_OK) {
        err = einfo.message[0] ? einfo.message : status_to_string(init_st);
        return init_st;
    }

    // Verify manifest-declared provided_ops and provided_runners match what the plugin registered
    if (!expected_package_id.empty()) {
        for (const auto& reg_op : staging.staged_impl_ids) {
            if (std::find(declared_ops.begin(), declared_ops.end(), reg_op) == declared_ops.end()) {
                if (pdesc.shutdown) pdesc.shutdown(reinterpret_cast<guild_core_host_t>(this));
                err = "security violation: plugin '" + pkg_id + "' registered undeclared kernel '" + reg_op + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
        }
        for (const auto& dec_op : declared_ops) {
            if (std::find(staging.staged_impl_ids.begin(), staging.staged_impl_ids.end(), dec_op) == staging.staged_impl_ids.end()) {
                if (pdesc.shutdown) pdesc.shutdown(reinterpret_cast<guild_core_host_t>(this));
                err = "package '" + pkg_id + "' declared kernel '" + dec_op + "' in .gkmetadata but plugin init() did not register it";
                return GUILD_STATUS_ERR_MISSING_KERNEL;
            }
        }
        for (const auto& reg_run : staging.staged_runner_ids) {
            if (std::find(declared_runners.begin(), declared_runners.end(), reg_run) == declared_runners.end()) {
                if (pdesc.shutdown) pdesc.shutdown(reinterpret_cast<guild_core_host_t>(this));
                err = "security violation: plugin '" + pkg_id + "' registered undeclared runner '" + reg_run + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
        }
        for (const auto& dec_run : declared_runners) {
            if (std::find(staging.staged_runner_ids.begin(), staging.staged_runner_ids.end(), dec_run) == staging.staged_runner_ids.end()) {
                if (pdesc.shutdown) pdesc.shutdown(reinterpret_cast<guild_core_host_t>(this));
                err = "package '" + pkg_id + "' declared runner '" + dec_run + "' in .gkmetadata but plugin init() did not register it";
                return GUILD_STATUS_ERR_MISSING_RUNNER;
            }
        }
    }

    // Commit staged runners first (with rollback if any fails)
    std::vector<std::string> committed_runners;
    for (size_t i = 0; i < staging.staged_runners.size(); ++i) {
        if (!runner_commit_fn) {
            for (const auto& cr : committed_runners) if (runner_rollback_fn) runner_rollback_fn(cr);
            if (pdesc.shutdown) pdesc.shutdown(reinterpret_cast<guild_core_host_t>(this));
            err = "plugin attempted to register a model runner, but runner registration is not enabled in this context";
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
        guild_runner_desc_v1_t rdesc = staging.staged_runners[i];
        rdesc.arch_id = staging.staged_runner_archs[i].c_str();
        rdesc.runner_id = staging.staged_runner_ids[i].c_str();
        rdesc.package_id = staging.staged_runner_pkgs[i].c_str();
        rdesc.required_op_kinds = staging.staged_runner_ops[i].empty() ? nullptr : staging.staged_runner_ops[i].data();
        rdesc.n_required_op_kinds = staging.staged_runner_ops[i].size();
        staging.staged_runner_pkg_dep_ptrs[i].clear();
        for (const auto& s : staging.staged_runner_pkg_deps[i]) staging.staged_runner_pkg_dep_ptrs[i].push_back(s.c_str());
        rdesc.required_package_deps = staging.staged_runner_pkg_dep_ptrs[i].empty() ? nullptr : staging.staged_runner_pkg_dep_ptrs[i].data();
        rdesc.n_required_package_deps = staging.staged_runner_pkg_dep_ptrs[i].size();
        rdesc.next_ext = staging.staged_runner_extensions[i].get();

        const guild_status_t rst = runner_commit_fn(rdesc, err);
        if (rst != GUILD_STATUS_OK) {
            for (const auto& cr : committed_runners) if (runner_rollback_fn) runner_rollback_fn(cr);
            if (pdesc.shutdown) pdesc.shutdown(reinterpret_cast<guild_core_host_t>(this));
            return rst;
        }
        committed_runners.push_back(staging.staged_runner_ids[i]);
    }

    // Commit staged kernel ops atomically under mutex_
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& impl_id : staging.staged_impl_ids) {
            for (const auto& existing : ops_) {
                if (existing->impl_id == impl_id) {
                    for (const auto& cr : committed_runners) if (runner_rollback_fn) runner_rollback_fn(cr);
                    if (pdesc.shutdown) pdesc.shutdown(reinterpret_cast<guild_core_host_t>(this));
                    err = "duplicate kernel operation registration for impl_id: " + impl_id;
                    return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
                }
            }
        }
        for (size_t i = 0; i < staging.staged_ops.size(); ++i) {
            auto stored = std::make_unique<StoredOp>();
            stored->desc = staging.staged_ops[i];
            stored->op_name = std::move(staging.staged_op_names[i]);
            stored->impl_id = std::move(staging.staged_impl_ids[i]);
            stored->package_id = std::move(staging.staged_pkg_ids[i]);
            stored->desc.op_name = stored->op_name.c_str();
            stored->desc.impl_id = stored->impl_id.c_str();
            stored->desc.package_id = stored->package_id.c_str();
            stored->dep_storage = std::move(staging.staged_op_deps[i]);
            for (const auto& s : stored->dep_storage) stored->dep_ptrs.push_back(s.c_str());
            stored->desc.required_op_deps = stored->dep_ptrs.empty() ? nullptr : stored->dep_ptrs.data();
            stored->desc.n_required_op_deps = stored->dep_ptrs.size();
            if (staging.staged_direct_tables[i]) {
                stored->owned_direct_table = std::move(staging.staged_direct_tables[i]);
                stored->desc.direct_fn_table = stored->owned_direct_table.get();
            }
            if (!staging.staged_ext_bytes[i].empty()) {
                stored->owned_ext_bytes = std::move(staging.staged_ext_bytes[i]);
                stored->desc.next_ext = stored->owned_ext_bytes.data();
            }
            ops_.push_back(std::move(stored));
        }

        LoadedPlugin lp;
        lp.desc = pdesc;
        lp.package_id = pkg_id;
        lp.package_version = pkg_ver;
        lp.publisher_id = pub_id;
        lp.desc.package_id = lp.package_id.c_str();
        lp.desc.package_version = lp.package_version.c_str();
        lp.desc.publisher_id = lp.publisher_id.c_str();
        lp.library_keepalive = std::move(library_keepalive);
        plugins_.push_back(std::move(lp));
    }

    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t KernelRegistry::unregister_package(const std::string& package_id, std::string& err) {
    if (active_sessions() > 0 || active_package_leases(package_id) > 0) {
        err = "cannot unregister kernel package '" + package_id + "' while active sessions are running";
        return GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY;
    }
    synchronize_package_streams(package_id);
    std::vector<LoadedPlugin> unloaded_plugins;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t before_ops = ops_.size();
        ops_.erase(std::remove_if(ops_.begin(), ops_.end(), [&](const std::unique_ptr<StoredOp>& op) {
            return op->package_id == package_id;
        }), ops_.end());
        const size_t before_plugins = plugins_.size();
        for (auto it = plugins_.begin(); it != plugins_.end();) {
            if (it->package_id == package_id) {
                unloaded_plugins.push_back(std::move(*it));
                it = plugins_.erase(it);
            } else {
                ++it;
            }
        }
        disabled_packages_.erase(package_id);
        if (ops_.size() == before_ops && plugins_.size() == before_plugins) {
            err = "kernel package not found: " + package_id;
            return GUILD_STATUS_ERR_MISSING_KERNEL;
        }
    }
    // Synchronize device and invoke plugin shutdown before releasing shared library keepalive
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
    if (!unloaded_plugins.empty()) {
        int dev_count = 0;
        if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0) {
            cudaDeviceSynchronize();
        }
    }
#endif
    for (auto& lp : unloaded_plugins) {
        if (lp.desc.shutdown) {
            lp.desc.shutdown(reinterpret_cast<guild_core_host_t>(this));
        }
        lp.library_keepalive.reset();
    }
    err.clear();
    return GUILD_STATUS_OK;
}

void KernelRegistry::set_package_enabled(const std::string& package_id, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enabled) disabled_packages_.erase(package_id);
    else disabled_packages_.insert(package_id);
}

bool KernelRegistry::is_package_enabled(const std::string& package_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return disabled_packages_.count(package_id) == 0;
}

void KernelRegistry::acquire_package_lease(const std::string& package_id, void* device_stream) const {
    if (package_id.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = package_leases_[package_id];
    ++entry.count;
    if (device_stream) entry.streams.insert(device_stream);
}

void KernelRegistry::release_package_lease(const std::string& package_id, void* device_stream) const {
    if (package_id.empty()) return;
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
    if (device_stream) {
        if (cudaStreamSynchronize(static_cast<cudaStream_t>(device_stream)) != cudaSuccess) {
            cudaGetLastError();
        }
    }
#else
    (void)device_stream;
#endif
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = package_leases_.find(package_id);
    if (it != package_leases_.end()) {
        if (device_stream) it->second.streams.erase(device_stream);
        if (--it->second.count <= 0) {
            package_leases_.erase(it);
        }
    }
}

int64_t KernelRegistry::active_package_leases(const std::string& package_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = package_leases_.find(package_id);
    return it != package_leases_.end() ? it->second.count : 0;
}

void KernelRegistry::synchronize_package_streams(const std::string& package_id) const {
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
    std::vector<void*> streams;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = package_leases_.find(package_id);
        if (it != package_leases_.end()) {
            streams.assign(it->second.streams.begin(), it->second.streams.end());
        }
    }
    for (void* s : streams) {
        if (s) cudaStreamSynchronize(static_cast<cudaStream_t>(s));
    }
#else
    (void)package_id;
#endif
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
        if (disabled_packages_.count(cand->package_id) > 0) continue;
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

std::vector<KernelOperationSnapshot> KernelRegistry::list_operation_snapshots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<KernelOperationSnapshot> out;
    out.reserve(ops_.size());
    for (const auto& op : ops_) {
        KernelOperationSnapshot snap;
        snap.op_kind = op->desc.op_kind;
        snap.op_name = op->op_name;
        snap.impl_id = op->impl_id;
        snap.package_id = op->package_id;
        snap.impl_version = op->desc.impl_version;
        snap.backend = op->desc.backend;
        snap.priority = op->desc.priority;
        snap.enabled = disabled_packages_.count(op->package_id) == 0;
        const auto lit = package_leases_.find(op->package_id);
        snap.active_package_leases = (lit != package_leases_.end()) ? lit->second.count : 0;
        out.push_back(std::move(snap));
    }
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
        GUILD_OP_GDN_AB_PROJ,
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
        // Descriptor batch rows count input tokens, including for head-expanded RMS/RoPE.
        // Attention is submitted in at most eight-query microbatches; the other
        // requirements must cover a full 512-token prefill chunk.
        q.batch_rows = op == GUILD_OP_ATTENTION_DECODE ? 8 : 512;
        if (op == GUILD_OP_GDN_AB_PROJ) {
            if (const char* value=std::getenv("GUILD_PREFILL_BATCH_PROJECTIONS"); value && std::atoi(value)==0)
                q.frozen_impl_id="gpu.gdn.ab_proj_prefill_serial";
        }
        const guild_kernel_op_desc_v1_t* d = nullptr;
        if (select(q, caps, &d, err) != GUILD_STATUS_OK || !d) return false;
        out_table.bound_impl_ids.emplace_back(d->impl_id ? d->impl_id : "");
        if (d->package_id && d->package_id[0]) {
            const std::string pkg_id = d->package_id;
            if (std::find(out_table.bound_package_ids.begin(), out_table.bound_package_ids.end(), pkg_id) ==
                out_table.bound_package_ids.end()) {
                out_table.bound_package_ids.push_back(pkg_id);
            }
        }
        if (op == GUILD_OP_GDN_AB_PROJ || op == GUILD_OP_MOE_ROUTER_TOPK) {
            PrefillKernelExtensionV1 extension;
            if (!read_prefill_extension(*d, extension, err)) return false;
            if (op == GUILD_OP_GDN_AB_PROJ) out_table.gdn_ab = extension.gdn_ab;
            else out_table.router_projection = extension.router_projection;
            continue;
        }
        const auto bind_legacy = [&](auto member) {
            if (!d->direct_fn_table) return;
            // Old binary plugins expose exactly seven pointers. Copy only that frozen
            // payload, without aliasing it as the larger runtime convenience table.
            PrefillKernelTableV1 legacy;
            std::memcpy(&legacy, d->direct_fn_table, sizeof(legacy));
            out_table.*member = legacy.*member;
        };
        // Bind only the selected entry. Unrelated requirements must neither overwrite
        // earlier choices nor interpret their operation-specific direct_fn_table payloads.
        switch (op) {
        case GUILD_OP_QUANTIZE_ACT: bind_legacy(&PrefillKernelTableV1::quantize_q8_1); break;
        case GUILD_OP_QUANTIZED_GEMM: bind_legacy(&PrefillKernelTableV1::mmvq); break;
        case GUILD_OP_RMS_NORM: bind_legacy(&PrefillKernelTableV1::rms_norm_weighted); break;
        case GUILD_OP_GDN_CONV_L2: bind_legacy(&PrefillKernelTableV1::gdn_conv); break;
        case GUILD_OP_GDN_RECURRENCE_STEP: bind_legacy(&PrefillKernelTableV1::gdn_recurrence); break;
        case GUILD_OP_GDN_OUT_NORM: bind_legacy(&PrefillKernelTableV1::gdn_out_norm); break;
        case GUILD_OP_ROPE_APPLY: bind_legacy(&PrefillKernelTableV1::rope_apply); break;
        default: break;
        }
    }
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
#if GUILD_BUNDLED_CPU_ARITHMETIC
    {
        auto d = make_base(GUILD_OP_BATCHED_EXPERT_EXEC, "cpu.expert_pool.split_multi_native", "builtin.guild-kernels-cpu", GUILD_BACKEND_CPU, 100);
        d.required_cpu_isa = GUILD_ISA_AVX2;
        d.max_batch_rows = 512;
        d.max_dim_n = 2560;
        d.max_dim_k = 2560;
#if defined(GUILD_NATIVE_EXPERTS)
        d.next_ext=guild_cpu_backend_table_v2();
#endif
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
#endif
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
        if (kind == GUILD_OP_GDN_AB_PROJ || kind == GUILD_OP_MOE_ROUTER_TOPK)
            d.next_ext = &kBuiltinPrefillExtension;
        register_op(d, ignore_err);
    };

    register_gpu(GUILD_OP_EMBEDDING_GATHER, "gpu.embed.native_gather", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_I32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_QUANTIZE_ACT, "gpu.quantize.q8_1_8col", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_Q8_1);
    register_gpu(GUILD_OP_DEQUANTIZE, "gpu.dequant.bf16_native", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_ANY, GUILD_DTYPE_BF16);
    register_gpu(GUILD_OP_QUANTIZED_GEMV, "gpu.mmvq.native_1col", 1, GUILD_DTYPE_ANY, GUILD_DTYPE_Q8_1, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_QUANTIZED_GEMM, "gpu.mmvq.native_8col", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_Q8_1, GUILD_DTYPE_F32);
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
    register_gpu(GUILD_OP_GDN_AB_PROJ, "gpu.gdn.ab_proj_prefill_serial", 512, GUILD_DTYPE_BF16, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    {
        auto d=make_base(GUILD_OP_GDN_AB_PROJ,"gpu.gdn.ab_proj_prefill_tiled",gpu_pkg,gpu_backend,210);
        d.min_sm_version=60;d.max_batch_rows=512;d.weight_dtype=GUILD_DTYPE_BF16;
        d.input_dtype=GUILD_DTYPE_F32;d.direct_fn_table=&kBuiltinPrefillKernels;
        d.next_ext=&kTiledPrefillExtension;
        register_op(d,ignore_err);
    }
    register_gpu(GUILD_OP_GDN_RECURRENCE_STEP, "gpu.gdn.prefill_recurrence", 512, GUILD_DTYPE_F32, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_GDN_OUT_NORM, "gpu.gdn.out_norm_native", 512, GUILD_DTYPE_F32, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_PLE_COMPUTE, "gpu.ple.native_postops", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_SPEC_VERIFY, "gpu.verify.window_graph", 8, GUILD_DTYPE_ANY, GUILD_DTYPE_I32, GUILD_DTYPE_F32);
    register_gpu(GUILD_OP_FINITE_CHECK, "gpu.finite.first_bad", 512, GUILD_DTYPE_ANY, GUILD_DTYPE_F32, GUILD_DTYPE_I32);
#endif
}

} // namespace guild::kernels
