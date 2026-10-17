// Public CUDA dense execution extension. Copyright Guild contributors, MIT.
// Kernel ABI v1 descriptor layouts and the legacy prefill table are unchanged.
#pragma once
#include "kernel_abi.h"
#ifdef __cplusplus
extern "C" {
#endif

#define GUILD_CUDA_DENSE_TAG_V1 0x43444431u /* CDD1 */
#define GUILD_CUDA_DENSE_VERSION_V1 1u

enum guild_cuda_bf16_mode_v1_e {
    GUILD_CUDA_BF16_ACT_AUTO_V1 = 0,
    GUILD_CUDA_BF16_ACT_SPLIT_V1 = 1,
    GUILD_CUDA_BF16_F32_SINGLE_V1 = 2,
    GUILD_CUDA_BF16_F32_MULTI_V1 = 3,
    GUILD_CUDA_BF16_F32_COLS_V1 = 4
};

// Borrowed device buffers, nonoverlapping and at least four-byte aligned;
// F32 activation rows additionally require eight-byte alignment and even ldx.
// W is row-major BF16 [n_out,n_in]. X/Y rows have element strides ldx/ldy.
// Caller validates backing allocation spans and keeps them alive until stream
// completion (or until graph destruction plus the final replay completes).
typedef struct guild_cuda_bf16_projection_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    const void* x;
    const uint16_t* w;
    float* y;
    int64_t n_in, n_out, ldx, ldy;
    int32_t rows;
    int32_t threads_per_row;
} guild_cuda_bf16_projection_v1_t;

// Q8_1: 36 bytes per 32 activation values, including FP16 scale and original
// FP16 sum. up==NULL quantizes x; otherwise quantizes silu(x)*up. Rows contiguous.
typedef struct guild_cuda_quantize_q8_1_v1_s {
    uint32_t struct_size;
    uint32_t reserved;
    const float* x;
    const float* up;
    void* q;
    uint64_t q_bytes;
    int32_t n_in, rows;
} guild_cuda_quantize_q8_1_v1_t;

// Stable GGML dtype IDs from kernel_abi.h. Weights retain their original GGUF
// block layouts. Q8_1 and Y activation rows are contiguous. exact_multi=1 keeps
// every row's single-token arithmetic order. Configuration is explicit per call.
typedef struct guild_cuda_mmvq_v1_s {
    uint32_t struct_size;
    uint32_t exact_multi;
    int32_t weight_dtype;
    int32_t n_in, n_out, rows;
    const void* weights;
    const void* q;
    float* y;
} guild_cuda_mmvq_v1_t;

// No C++ objects/exceptions or CUDA SDK types cross this boundary. All calls
// return status and fill an optional caller-owned error record. The normal
// explicit-stream path allocates nothing and performs no host synchronization.
// Capture records the same kernel topology as direct execution. Request records
// are consumed synchronously, device buffers have stream/graph lifetime.
typedef struct guild_cuda_dense_ext_v1_s {
    uint32_t struct_size;
    uint32_t tag;
    uint32_t version;
    uint32_t reserved;
    guild_status_t (*bf16)(const guild_exec_context_v1_t*, const guild_cuda_bf16_projection_v1_t*, guild_error_info_t*);
    guild_status_t (*quantize_q8_1)(const guild_exec_context_v1_t*, const guild_cuda_quantize_q8_1_v1_t*, guild_error_info_t*);
    guild_status_t (*mmvq)(const guild_exec_context_v1_t*, const guild_cuda_mmvq_v1_t*, guild_error_info_t*);
} guild_cuda_dense_ext_v1_t;

#ifdef __cplusplus
}
#endif
