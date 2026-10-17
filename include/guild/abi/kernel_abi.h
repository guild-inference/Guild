// include/guild/abi/kernel_abi.h - Guild Kernel & Runner Binary Interface (ABI v1)
//
// Copyright (c) Guild Inference Contributors. MIT License.
// Preserves prior Strata/Guild kernel contracts (MIT License).
//
// This header is strictly C99/C11 and C++17 compatible. It defines the binary
// boundary between Guild Core, hardware kernel packages, and model architecture
// runners. No C++ classes, STL containers, exceptions, or compiler-specific
// layouts may cross this boundary.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// 1. ABI Version & Feature Negotiation
// =============================================================================

#define GUILD_KERNEL_ABI_VERSION_MAJOR 1u
#define GUILD_KERNEL_ABI_VERSION_MINOR 0u
#define GUILD_KERNEL_ABI_VERSION_PATCH 0u

#define GUILD_KERNEL_ABI_MAKE_VERSION(major, minor, patch) \
    ((((uint32_t)(major) & 0xFFu) << 24) | (((uint32_t)(minor) & 0xFFFu) << 12) | ((uint32_t)(patch) & 0xFFFu))

#define GUILD_KERNEL_ABI_VERSION_MAJOR_OF(v) (((uint32_t)(v) >> 24) & 0xFFu)
#define GUILD_KERNEL_ABI_VERSION_MINOR_OF(v) (((uint32_t)(v) >> 12) & 0xFFFu)
#define GUILD_KERNEL_ABI_VERSION_PATCH_OF(v) ((uint32_t)(v) & 0xFFFu)

#define GUILD_MAKE_ABI_VERSION(major, minor, patch) GUILD_KERNEL_ABI_MAKE_VERSION(major, minor, patch)
#define GUILD_ABI_VERSION_MAJOR(v) GUILD_KERNEL_ABI_VERSION_MAJOR_OF(v)
#define GUILD_ABI_VERSION_MINOR(v) GUILD_KERNEL_ABI_VERSION_MINOR_OF(v)
#define GUILD_ABI_VERSION_PATCH(v) GUILD_KERNEL_ABI_VERSION_PATCH_OF(v)

#define GUILD_KERNEL_ABI_VERSION_V1 \
    GUILD_KERNEL_ABI_MAKE_VERSION(GUILD_KERNEL_ABI_VERSION_MAJOR, GUILD_KERNEL_ABI_VERSION_MINOR, GUILD_KERNEL_ABI_VERSION_PATCH)

#define GUILD_PLUGIN_ENTRY_SYMBOL_V1 "guild_plugin_entry_v1"

#if defined(_WIN32) || defined(__CYGW32__)
#define GUILD_PLUGIN_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define GUILD_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define GUILD_PLUGIN_EXPORT
#endif

// =============================================================================
// 2. Status & Error Propagation
// =============================================================================

typedef int32_t guild_status_t;

enum guild_status_code_e {
    GUILD_STATUS_OK                        = 0,
    GUILD_STATUS_ERR_INVALID_ARGUMENT      = -1,
    GUILD_STATUS_ERR_ABI_MISMATCH          = -2,
    GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE  = -3,
    GUILD_STATUS_ERR_CAPABILITY_MISMATCH   = -4,
    GUILD_STATUS_ERR_MISSING_RUNNER        = -5,
    GUILD_STATUS_ERR_MISSING_KERNEL        = -6,
    GUILD_STATUS_ERR_DUPLICATE_REGISTRATION = -7,
    GUILD_STATUS_ERR_DEPENDENCY_MISSING    = -8,
    GUILD_STATUS_ERR_INVALID_TENSOR        = -9,
    GUILD_STATUS_ERR_BUFFER_LIFETIME       = -10,
    GUILD_STATUS_ERR_OUT_OF_MEMORY         = -11,
    GUILD_STATUS_ERR_EXECUTION_FAILED      = -12,
    GUILD_STATUS_ERR_NONFINITE_OUTPUT      = -13,
    GUILD_STATUS_ERR_SECURITY_VIOLATION    = -14,
    GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY   = -15,
    GUILD_STATUS_ERR_INTERNAL              = -99
};

typedef struct guild_error_info_s {
    uint32_t struct_size;
    guild_status_t code;
    char message[256];
    const char* source_component;
} guild_error_info_t;

// =============================================================================
// 3. Opaque Handles & Extension Header
// =============================================================================

typedef struct guild_core_host_s* guild_core_host_t;
typedef struct guild_plugin_handle_s* guild_plugin_handle_t;
typedef struct guild_session_state_handle_s* guild_session_state_handle_t;
typedef struct guild_stream_handle_s* guild_stream_handle_t;

typedef struct guild_ext_header_s {
    uint32_t struct_size;
    uint32_t ext_type;
    const void* next_ext;
} guild_ext_header_t;

// =============================================================================
// 4. Hardware Backends, Capabilities & Numerical Guarantees
// =============================================================================

typedef uint32_t guild_backend_t;
enum guild_backend_e {
    GUILD_BACKEND_NONE = 0,
    GUILD_BACKEND_CPU  = 1u << 0,
    GUILD_BACKEND_CUDA = 1u << 1,
    GUILD_BACKEND_HIP  = 1u << 2
};

typedef uint64_t guild_cpu_isa_flags_t;
enum guild_cpu_isa_e {
    GUILD_ISA_NONE       = 0ull,
    GUILD_ISA_SSE42      = 1ull << 0,
    GUILD_ISA_AVX        = 1ull << 1,
    GUILD_ISA_AVX2       = 1ull << 2,
    GUILD_ISA_FMA        = 1ull << 3,
    GUILD_ISA_F16C       = 1ull << 4,
    GUILD_ISA_BMI2       = 1ull << 5,
    GUILD_ISA_AVX512F    = 1ull << 6,
    GUILD_ISA_AVX512BW   = 1ull << 7,
    GUILD_ISA_AVX512DQ   = 1ull << 8,
    GUILD_ISA_AVX512VL   = 1ull << 9,
    GUILD_ISA_AVX512VNNI = 1ull << 10,
    GUILD_ISA_AVX512VBMI = 1ull << 11,
    GUILD_ISA_AVX512CD   = 1ull << 12
};

typedef uint64_t guild_gpu_cap_flags_t;
enum guild_gpu_cap_e {
    GUILD_GPU_CAP_NONE         = 0ull,
    GUILD_GPU_CAP_DP4A         = 1ull << 0,
    GUILD_GPU_CAP_FP16         = 1ull << 1,
    GUILD_GPU_CAP_BF16         = 1ull << 2,
    GUILD_GPU_CAP_TENSOR_CORE  = 1ull << 3,
    GUILD_GPU_CAP_CUDA_GRAPH   = 1ull << 4,
    GUILD_GPU_CAP_MAPPED_HOST  = 1ull << 5,
    GUILD_GPU_CAP_COOPERATIVE  = 1ull << 6
};

typedef uint64_t guild_core_cap_flags_t;
enum guild_core_cap_e {
    GUILD_CORE_CAP_NONE              = 0ull,
    GUILD_CORE_CAP_HOST_ONLY_KV      = 1ull << 0,
    GUILD_CORE_CAP_PINNED_EXPERT_RAM = 1ull << 1,
    GUILD_CORE_CAP_PCIE_EXPERT_SPLIT = 1ull << 2,
    GUILD_CORE_CAP_CUDA_GRAPH_VERIFY = 1ull << 3,
    GUILD_CORE_CAP_BATCHED_PREFILL   = 1ull << 4,
    GUILD_CORE_CAP_MTP_SPECULATIVE   = 1ull << 5,
    GUILD_CORE_CAP_PLE_NGRAM         = 1ull << 6,
    GUILD_CORE_CAP_DENSE_CAUSAL_ATTN = 1ull << 7,
    GUILD_CORE_CAP_INDEXED_QSA_ATTN  = 1ull << 8
};

typedef uint32_t guild_numerical_flags_t;
enum guild_numerical_guarantee_e {
    GUILD_NUM_NONE                    = 0u,
    GUILD_NUM_BITWISE_DETERMINISTIC   = 1u << 0,
    GUILD_NUM_EXACT_SINGLE_COL_EQUIV  = 1u << 1,
    GUILD_NUM_FINITE_VALIDATED        = 1u << 2,
    GUILD_NUM_BF16_RNE                = 1u << 3,
    GUILD_NUM_FAST_APPROX             = 1u << 4
};

typedef struct guild_device_capabilities_v1_s {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t available_backends;
    guild_cpu_isa_flags_t cpu_isa_flags;
    guild_gpu_cap_flags_t gpu_cap_flags;
    guild_core_cap_flags_t core_cap_flags;
    int32_t cpu_physical_cores;
    int32_t cpu_sockets;
    int32_t gpu_device_count;
    int32_t gpu_sm_version; // e.g. 61 for SM 6.1, 80 for SM 8.0, 0 if no GPU
    uint64_t available_ram_bytes;
    uint64_t available_vram_bytes;
    const void* next_ext;
} guild_device_capabilities_v1_t;

// =============================================================================
// 5. Data Types, Memory Descriptors & Tensor Descriptors
// =============================================================================

typedef int32_t guild_dtype_t;
enum guild_dtype_e {
    GUILD_DTYPE_ANY     = -1,
    GUILD_DTYPE_F32     = 0,
    GUILD_DTYPE_F16     = 1,
    GUILD_DTYPE_Q4_0    = 2,
    GUILD_DTYPE_Q5_0    = 6,
    GUILD_DTYPE_Q5_1    = 7,
    GUILD_DTYPE_Q8_0    = 8,
    GUILD_DTYPE_Q8_1    = 9,
    GUILD_DTYPE_Q2_K    = 10,
    GUILD_DTYPE_Q3_K    = 11,
    GUILD_DTYPE_Q4_K    = 12,
    GUILD_DTYPE_Q5_K    = 13,
    GUILD_DTYPE_Q6_K    = 14,
    GUILD_DTYPE_Q8_K    = 15,
    GUILD_DTYPE_IQ2_XXS = 16,
    GUILD_DTYPE_IQ2_XS  = 17,
    GUILD_DTYPE_IQ3_XXS = 18,
    GUILD_DTYPE_IQ4_NL  = 20,
    GUILD_DTYPE_IQ3_S   = 21,
    GUILD_DTYPE_IQ2_S   = 22,
    GUILD_DTYPE_IQ4_XS  = 23,
    GUILD_DTYPE_I8      = 24,
    GUILD_DTYPE_I32     = 26,
    GUILD_DTYPE_BF16    = 30,
    GUILD_DTYPE_Q2_0    = 42
};

typedef uint32_t guild_memory_space_t;
enum guild_memory_space_e {
    GUILD_MEM_HOST_PAGEABLE = 0,
    GUILD_MEM_HOST_PINNED   = 1,
    GUILD_MEM_HOST_MAPPED   = 2,
    GUILD_MEM_DEVICE_VRAM   = 3
};

typedef uint32_t guild_buffer_ownership_t;
enum guild_buffer_ownership_e {
    GUILD_OWNERSHIP_CORE_BORROWED  = 0, // Owned by Guild Core; kernel must not free or retain past call
    GUILD_OWNERSHIP_SESSION_SCOPED = 1, // Owned by session state; valid until session destroy
    GUILD_OWNERSHIP_MODEL_STATIC   = 2, // Owned by loaded model; valid until model unload
    GUILD_OWNERSHIP_TRANSIENT_COPY = 3  // Caller-copied staging buffer valid until stream sync
};

typedef uint32_t guild_buffer_lifetime_t;
enum guild_buffer_lifetime_e {
    GUILD_LIFETIME_EXPIRED       = 0,
    GUILD_LIFETIME_CALL_SYNC     = 1, // Valid only during synchronous host call
    GUILD_LIFETIME_STREAM_ORDERED = 2, // Valid until the launch stream completes
    GUILD_LIFETIME_SESSION       = 3, // Valid for lifetime of session
    GUILD_LIFETIME_PERSISTENT    = 4  // Valid for lifetime of loaded model
};

typedef struct guild_buffer_desc_s {
    uint32_t struct_size;
    void* data;
    uint64_t byte_size;
    uint64_t alignment;
    guild_memory_space_t memory_space;
    guild_buffer_ownership_t ownership;
    guild_buffer_lifetime_t lifetime;
    int32_t device_id;
} guild_buffer_desc_t;

#define GUILD_MAX_TENSOR_DIMS 4

typedef struct guild_tensor_desc_s {
    uint32_t struct_size;
    guild_dtype_t dtype;
    int32_t n_dims;
    int64_t dims[GUILD_MAX_TENSOR_DIMS];        // dims[0] is contiguous axis (ne0)
    uint64_t strides_bytes[GUILD_MAX_TENSOR_DIMS];
    guild_buffer_desc_t buffer;
} guild_tensor_desc_t;

// =============================================================================
// 6. Scratch Memory, Synchronization & Execution Contexts
// =============================================================================

typedef uint32_t guild_sync_mode_t;
enum guild_sync_mode_e {
    GUILD_SYNC_NONE           = 0, // Asynchronous on stream; no host wait
    GUILD_SYNC_STREAM_ORDERED = 1, // Ordered on provided device stream
    GUILD_SYNC_HOST_BLOCKING  = 2, // Must complete before execute_fn returns
    GUILD_SYNC_GRAPH_CAPTURE  = 3  // Must be legal inside CUDA/HIP stream capture
};

typedef struct guild_scratch_request_s {
    uint32_t struct_size;
    uint64_t device_bytes;
    uint64_t device_alignment;
    uint64_t pinned_host_bytes;
    uint64_t pinned_host_alignment;
    uint64_t pageable_host_bytes;
} guild_scratch_request_t;

typedef struct guild_exec_context_v1_s {
    uint32_t struct_size;
    uint32_t abi_version;
    guild_backend_t backend;
    int32_t device_id;
    void* device_stream; // cudaStream_t / hipStream_t or NULL for CPU
    guild_sync_mode_t sync_mode;
    guild_buffer_desc_t device_scratch;
    guild_buffer_desc_t host_scratch;
    void* worker_pool;   // Optional host worker pool handle
    const void* next_ext;
} guild_exec_context_v1_t;

// =============================================================================
// 7. Kernel Operation Categories & Descriptors
// =============================================================================

typedef int32_t guild_op_kind_t;
enum guild_op_kind_e {
    GUILD_OP_INVALID               = 0,
    GUILD_OP_QUANTIZED_GEMV        = 1,
    GUILD_OP_QUANTIZED_GEMM        = 2,
    GUILD_OP_GROUPED_MOE_GEMM      = 3,
    GUILD_OP_MOE_ROUTER_TOPK       = 4,
    GUILD_OP_MOE_DISPATCH          = 5,
    GUILD_OP_BATCHED_EXPERT_EXEC   = 6,
    GUILD_OP_SHARED_EXPERT         = 7,
    GUILD_OP_MOE_COMBINE           = 8,
    GUILD_OP_ATTENTION_DECODE      = 9,
    GUILD_OP_ATTENTION_PREFILL     = 10,
    GUILD_OP_QSA_INDEXER_SELECT    = 11,
    GUILD_OP_ROPE_APPLY            = 12,
    GUILD_OP_GDN_CONV_L2           = 13,
    GUILD_OP_GDN_AB_PROJ           = 14,
    GUILD_OP_GDN_RECURRENCE_STEP   = 15,
    GUILD_OP_GDN_OUT_NORM          = 16,
    GUILD_OP_QUANTIZE_ACT          = 17,
    GUILD_OP_DEQUANTIZE            = 18,
    GUILD_OP_RMS_NORM              = 19,
    GUILD_OP_GATED_RESIDUAL        = 20,
    GUILD_OP_KV_APPEND             = 21,
    GUILD_OP_KV_GATHER             = 22,
    GUILD_OP_PLE_COMPUTE           = 23,
    GUILD_OP_SPEC_VERIFY           = 24,
    GUILD_OP_FINITE_CHECK          = 25,
    GUILD_OP_EMBEDDING_GATHER      = 26,
    GUILD_OP_COUNT                 = 27
};

typedef guild_status_t (*guild_kernel_query_scratch_fn)(
    const guild_tensor_desc_t* inputs,
    size_t n_inputs,
    const guild_tensor_desc_t* outputs,
    size_t n_outputs,
    guild_scratch_request_t* out_scratch,
    guild_error_info_t* err);

typedef guild_status_t (*guild_kernel_validate_fn)(
    const guild_device_capabilities_v1_t* caps,
    const guild_tensor_desc_t* inputs,
    size_t n_inputs,
    const guild_tensor_desc_t* outputs,
    size_t n_outputs,
    guild_error_info_t* err);

typedef guild_status_t (*guild_kernel_execute_fn)(
    const guild_exec_context_v1_t* ctx,
    const guild_tensor_desc_t* inputs,
    size_t n_inputs,
    const guild_tensor_desc_t* outputs,
    size_t n_outputs,
    const void* op_params,
    size_t op_params_size,
    guild_error_info_t* err);

typedef struct guild_kernel_op_desc_v1_s {
    uint32_t struct_size;
    uint32_t abi_version;
    guild_op_kind_t op_kind;
    const char* op_name;       // e.g. "quantized_gemm"
    const char* impl_id;       // e.g. "cuda.mmvq.native_8col"
    const char* package_id;    // e.g. "builtin.guild-kernels-cuda"
    uint32_t impl_version;     // Semantic version encoded via GUILD_KERNEL_ABI_MAKE_VERSION
    guild_backend_t backend;
    guild_cpu_isa_flags_t required_cpu_isa;
    guild_gpu_cap_flags_t required_gpu_caps;
    int32_t min_sm_version;    // 0 if unconstrained
    int32_t max_sm_version;    // 0 if unconstrained
    guild_dtype_t weight_dtype;
    guild_dtype_t input_dtype;
    guild_dtype_t output_dtype;
    guild_dtype_t accum_dtype;
    int64_t max_batch_rows;    // 0 if unconstrained
    int64_t min_dim_n;         // 0 if unconstrained
    int64_t max_dim_n;         // 0 if unconstrained
    int64_t min_dim_k;         // 0 if unconstrained
    int64_t max_dim_k;         // 0 if unconstrained
    guild_numerical_flags_t numerical_flags;
    guild_sync_mode_t supported_sync_modes;
    int32_t priority;          // Higher priority wins among compatible candidates
    const char* const* required_op_deps; // Optional array of impl_id strings
    size_t n_required_op_deps;
    guild_kernel_query_scratch_fn query_scratch;
    guild_kernel_validate_fn validate;
    guild_kernel_execute_fn execute;
    const void* direct_fn_table; // Optional pre-bound fast-path function pointer table
    const void* next_ext;
} guild_kernel_op_desc_v1_t;

// Tagged extension payload for prefill projection fast paths (GDN A/B and MoE router).
#define GUILD_PREFILL_KERNEL_EXT_TAG_V1 0x50464b45u // "PFKE"
#define GUILD_PREFILL_KERNEL_EXT_VERSION_V1 1u

typedef struct guild_prefill_ext_header_s {
    uint32_t struct_size;
    uint32_t tag;
    uint32_t version;
} guild_prefill_ext_header_t;

typedef struct guild_prefill_kernel_ext_v1_s {
    guild_prefill_ext_header_t header;
    void (*gdn_ab)(const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt, const float* a,
                   float* gate, float* beta, int n, int heads, int rows, void* stream);
    void (*router_projection)(const float* x, const uint16_t* w, float* y,
                              int64_t n, int64_t outputs, int rows, void* stream);
} guild_prefill_kernel_ext_v1_t;

// =============================================================================
// 8. Model Architecture Runner ABI v1
// =============================================================================

typedef guild_status_t (*guild_runner_session_alloc_fn)(
    const void* model_config,
    int64_t max_context,
    guild_session_state_handle_t* out_state,
    guild_error_info_t* err);

typedef void (*guild_runner_session_free_fn)(
    guild_session_state_handle_t state);

typedef struct guild_runner_desc_v1_s {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* arch_id;       // Canonical GGUF architecture e.g. "qwen4exp", "qwen35moe"
    const char* runner_id;     // e.g. "runner.qwen35moe.v1"
    const char* package_id;    // e.g. "builtin.guild-kernels-ornith"
    uint32_t runner_version;
    guild_core_cap_flags_t required_core_caps;
    const guild_op_kind_t* required_op_kinds;
    size_t n_required_op_kinds;
    const char* const* required_package_deps;
    size_t n_required_package_deps;
    guild_runner_session_alloc_fn session_alloc;
    guild_runner_session_free_fn session_free;
    const void* next_ext;
} guild_runner_desc_v1_t;

// =============================================================================
// 9. Plugin Registration & Host Callback Table
// =============================================================================

typedef struct guild_host_api_v1_s {
    uint32_t struct_size;
    uint32_t core_abi_version;
    guild_status_t (*negotiate_abi)(uint32_t requested_version, uint32_t* out_negotiated_version);
    guild_status_t (*get_device_capabilities)(guild_core_host_t host, guild_device_capabilities_v1_t* out_caps);
    guild_status_t (*register_kernel_op)(guild_core_host_t host, const guild_kernel_op_desc_v1_t* desc, guild_error_info_t* err);
    guild_status_t (*register_runner)(guild_core_host_t host, const guild_runner_desc_v1_t* desc, guild_error_info_t* err);
} guild_host_api_v1_t;

typedef struct guild_plugin_descriptor_v1_s {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* package_id;
    const char* package_version;
    const char* publisher_id;
    guild_status_t (*init)(guild_core_host_t host, const guild_host_api_v1_t* host_api, guild_error_info_t* err);
    void (*shutdown)(guild_core_host_t host);
} guild_plugin_descriptor_v1_t;

typedef guild_status_t (*guild_plugin_entry_fn)(
    uint32_t host_abi_version,
    guild_plugin_descriptor_v1_t* out_plugin_desc,
    guild_error_info_t* err);

#ifdef __cplusplus
} // extern "C"
#endif
