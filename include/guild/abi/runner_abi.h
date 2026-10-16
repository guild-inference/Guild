// Copyright (c) Guild Inference Contributors. MIT License.
// Optional Runner ABI v1 extension. Does not change any Kernel ABI v1 layout.
#pragma once
#include "guild/abi/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GUILD_RUNNER_MODEL_TAG_V1 0x524d4431u /* RMD1 */
#define GUILD_RUNNER_MODEL_VERSION_V1 1u

// A read-only view, valid only for the duration of describe(). No mapped tensor
// payloads are exposed. Core retains file ownership and checks shard bounds.
typedef struct guild_model_reader_v1_s {
    uint32_t struct_size;
    void* user;
    guild_status_t (*get_i64)(void*, const char*, int64_t*);
    guild_status_t (*get_f64)(void*, const char*, double*);
    guild_status_t (*get_string)(void*, const char*, char*, size_t);
    guild_status_t (*get_tensor)(void*, const char*, guild_tensor_desc_t*);
} guild_model_reader_v1_t;

enum guild_runner_prefill_e {
    GUILD_RUNNER_PREFILL_SEQUENTIAL = 0,
    GUILD_RUNNER_PREFILL_CHUNKED = 1,
    GUILD_RUNNER_PREFILL_LAYER_BATCHED = 2
};

// Values are copied into Core-owned storage. All dimensions are signed and
// validated before allocation. Strings are UTF-8, NUL terminated, inline arrays.
// Topology v1 describes periodic recurrent/full-attention blocks; new topology
// kinds require a new negotiated extension, never an implicit Core assumption.
typedef struct guild_runner_model_v1_s {
    uint32_t struct_size;
    uint32_t version;
    char arch_id[128];
    char name[256];
    int64_t n_embd, n_layers, n_heads, n_kv_heads, head_dim;
    int64_t context_length, vocab_size, full_attention_interval;
    int64_t n_experts, active_experts, expert_ff, shared_ff, shared_experts;
    int64_t ssm_state_size, ssm_k_heads, ssm_v_heads, ssm_d_conv;
    int64_t ssm_conv_channels, ssm_value_dim, index_heads, index_dim;
    int64_t residual_streams, residual_rank;
    uint32_t indexed_attention, gdn_gate_silu, requires_ple, supports_mtp;
    uint32_t prefill_strategy;
    uint32_t rope_kind; /* 0 none, 1 linear, 2 yarn */
    double rope_base, rope_factor, rope_original_context;
    char embedding_tensor[128], output_tensor[128], output_norm_tensor[128];
    char ple_table_tensor[128], ple_key_tensor[128], layer_prefix[64];
} guild_runner_model_v1_t;

typedef struct guild_runner_model_ext_v1_s {
    uint32_t struct_size;
    uint32_t tag;
    uint32_t version;
    int32_t priority;
    guild_status_t (*recognize)(const guild_model_reader_v1_t*, uint32_t*);
    guild_status_t (*describe)(const guild_model_reader_v1_t*, guild_runner_model_v1_t*, guild_error_info_t*);
    guild_status_t (*validate_numerical)(const float*, size_t, guild_error_info_t*);
} guild_runner_model_ext_v1_t;

#ifdef __cplusplus
}
#endif
