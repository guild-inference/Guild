// Public CUDA attention, recurrent (GDN), RoPE, KV and selection extension.
// Copyright Guild contributors, MIT.
// Kernel ABI v1 descriptor layouts and the legacy prefill table are unchanged.
#pragma once
#include "kernel_abi.h"
#ifdef __cplusplus
extern "C" {
#endif

#define GUILD_CUDA_ATTN_TAG_V1 0x43444131u /* CDA1 */
#define GUILD_CUDA_ATTN_VERSION_V1 1u

enum guild_cuda_rope_scaling_type_v1_e {
    GUILD_CUDA_ROPE_SCALING_NONE_V1 = 0,
    GUILD_CUDA_ROPE_SCALING_LINEAR_V1 = 1,
    GUILD_CUDA_ROPE_SCALING_YARN_V1 = 2
};

typedef struct guild_cuda_rope_scaling_v1_s {
    uint32_t struct_size;
    int32_t scaling_type;
    double freq_base;
    double factor;
    double freq_scale_in;
    double orig_ctx;
    double ext_factor;
    double attn_factor;
    double beta_fast;
    double beta_slow;
} guild_cuda_rope_scaling_v1_t;

typedef struct guild_cuda_qsa_shapes_v1_s {
    uint32_t struct_size;
    uint32_t reserved;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t head_dim;
    int64_t n_rot;
    int64_t idx_n_head;
    int64_t idx_dim;
    int64_t idx_block;
    int64_t idx_top_k;
    int64_t page_size;
} guild_cuda_qsa_shapes_v1_t;

typedef struct guild_cuda_gdn_shapes_v1_s {
    uint32_t struct_size;
    uint32_t reserved;
    int64_t state_size;
    int64_t h_k;
    int64_t h_v;
} guild_cuda_gdn_shapes_v1_t;

typedef struct guild_cuda_kv_host_pools_v1_s {
    uint32_t struct_size;
    uint32_t reserved;
    uint16_t* k_pool;
    uint16_t* v_pool;
    int8_t* k_q;
    int8_t* v_q;
    uint16_t* k_scale;
    uint16_t* v_scale;
    uint8_t* k_q4;
    uint8_t* v_q4;
} guild_cuda_kv_host_pools_v1_t;

typedef struct guild_cuda_qsa_attn_pools_v1_s {
    uint32_t struct_size;
    uint32_t reserved;
    const uint16_t* k_pool;
    const uint16_t* v_pool;
    const int8_t* k_q;
    const int8_t* v_q;
    const uint16_t* k_scale;
    const uint16_t* v_scale;
    const uint8_t* k_q4;
    const uint8_t* v_q4;
    const int32_t* page_table;
} guild_cuda_qsa_attn_pools_v1_t;

typedef struct guild_cuda_kv_stream_map_v1_s {
    uint32_t struct_size;
    uint32_t reserved;
    int32_t* page_table;
    int32_t* slot_block;
    int32_t* slot_stamp;
    int32_t* slot_ref;
    int32_t* ctl;
    int32_t* miss_block;
    int32_t* miss_slot;
    int64_t n_blocks;
    int64_t n_slots;
} guild_cuda_kv_stream_map_v1_t;

typedef struct guild_cuda_kv_stream_counters_v1_s {
    uint32_t struct_size;
    uint32_t overflow;
    uint64_t misses;
    uint64_t lookups;
    uint64_t calls;
} guild_cuda_kv_stream_counters_v1_t;

typedef struct guild_cuda_qsa_indexer_buffers_v1_s {
    uint32_t struct_size;
    uint32_t reserved;
    float* tail;
    float* dead;
    float* pooled;
    int32_t* block_pos;
} guild_cuda_qsa_indexer_buffers_v1_t;

enum guild_cuda_attn_toggle_v1_e {
    GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_V1 = 1u << 0,
    GUILD_CUDA_ATTN_TOGGLE_NATIVE_ROPE_V1 = 1u << 1,
    GUILD_CUDA_ATTN_TOGGLE_NATIVE_GDN_V1 = 1u << 2,
    GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_INDEXER_V1 = 1u << 3,
    GUILD_CUDA_ATTN_TOGGLE_NATIVE_QSA_SCORE_V1 = 1u << 4
};

enum guild_cuda_attn_cfg_action_v1_e {
    GUILD_CUDA_ATTN_CFG_SET_TOGGLES_V1 = 1u << 0,
    GUILD_CUDA_ATTN_CFG_SET_ROPE_SCALING_V1 = 1u << 1,
    GUILD_CUDA_ATTN_CFG_SET_MROPE_TABLE_V1 = 1u << 2,
    GUILD_CUDA_ATTN_CFG_SET_ROPE_TABLE_V1 = 1u << 3,
    GUILD_CUDA_ATTN_CFG_RELEASE_ROPE_TABLE_V1 = 1u << 4
};

typedef struct guild_cuda_attn_config_v1_s {
    uint32_t struct_size;
    uint32_t actions;
    uint32_t toggle_mask;
    uint32_t toggle_values;
    uint32_t* out_toggle_values;
    const int32_t* mrope_table;
    const float* rope_cos_tab;
    const float* rope_sin_tab;
    int32_t rope_max_pos;
    int32_t reserved;
    guild_cuda_rope_scaling_v1_t rope_scaling;
} guild_cuda_attn_config_v1_t;

enum guild_cuda_rope_mode_v1_e {
    GUILD_CUDA_ROPE_NEOX_APPLY_V1 = 0,
    GUILD_CUDA_ROPE_NATIVE_APPLY_V1 = 1
};

typedef struct guild_cuda_rope_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    const float* x;
    float* out;
    const float* cos_tab;
    const float* sin_tab;
    const int32_t* positions;
    int64_t rows;
    int32_t head_dim;
    int32_t n_rot;
    guild_cuda_rope_scaling_v1_t scaling;
} guild_cuda_rope_op_v1_t;

enum guild_cuda_qsa_norm_gate_mode_v1_e {
    GUILD_CUDA_QSA_RMS_NORM_WEIGHTED_V1 = 0,
    GUILD_CUDA_QSA_GATE_APPLY_NATIVE_V1 = 1,
    GUILD_CUDA_QSA_GATE_APPLY_F16_V1 = 2,
    GUILD_CUDA_QSA_GATE_APPLY_F32_V1 = 3
};

typedef struct guild_cuda_qsa_norm_gate_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    const float* x;
    const float* aux;
    void* out;
    int64_t cols_or_head_dim;
    int64_t rows_or_n_head;
    float epsilon;
    uint32_t reserved;
    guild_cuda_qsa_shapes_v1_t shapes;
} guild_cuda_qsa_norm_gate_v1_t;

enum guild_cuda_gdn_mode_v1_e {
    GUILD_CUDA_GDN_STEP_V1 = 0,
    GUILD_CUDA_GDN_CONV_STEP_V1 = 1,
    GUILD_CUDA_GDN_L2_NORM_V1 = 2,
    GUILD_CUDA_GDN_BETA_GATE_V1 = 3,
    GUILD_CUDA_GDN_OUT_NORM_V1 = 4,
    GUILD_CUDA_GDN_FUSED_CONV_L2_V1 = 5,
    GUILD_CUDA_GDN_FUSED_AB_V1 = 6,
    GUILD_CUDA_GDN_FUSED_STEP_NORM_V1 = 7,
    GUILD_CUDA_GDN_NATIVE_STEP_V1 = 8,
    GUILD_CUDA_GDN_NATIVE_CONV_SILU_V1 = 9,
    GUILD_CUDA_GDN_NATIVE_L2_NORM_V1 = 10,
    GUILD_CUDA_GDN_NATIVE_BETA_GATE_V1 = 11,
    GUILD_CUDA_GDN_NATIVE_GATE_V1 = 12,
    GUILD_CUDA_GDN_NATIVE_OUT_NORM_V1 = 13,
    GUILD_CUDA_GDN_PREFILL_CONV_V1 = 14,
    GUILD_CUDA_GDN_PREFILL_RECURRENCE_V1 = 15,
    GUILD_CUDA_GDN_CONV_L2_MULTI_V1 = 16,
    GUILD_CUDA_GDN_CONV_COMMIT_V1 = 17,
    GUILD_CUDA_GDN_AB_MULTI_V1 = 18,
    GUILD_CUDA_GDN_AB_PREFILL_V1 = 19,
    GUILD_CUDA_GDN_STEP_NORM_MULTI_V1 = 20
};

typedef struct guild_cuda_gdn_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    float* state;
    const float* x;
    const float* k;
    const float* v;
    const float* gate;
    const float* beta;
    const float* z;
    const float* gamma;
    const uint16_t* w_alpha;
    const uint16_t* w_beta;
    const int32_t* n_keep;
    float* out;
    float* aux_out;
    int64_t channels_or_rows;
    int64_t cols_or_d_conv;
    int32_t qk_heads_or_h_k;
    int32_t h_v;
    int32_t n_tok;
    int32_t t_begin;
    float epsilon;
    uint32_t gate_silu;
    guild_cuda_gdn_shapes_v1_t shapes;
} guild_cuda_gdn_op_v1_t;

enum guild_cuda_kv_mode_v1_e {
    GUILD_CUDA_KV_APPEND_SYNC_V1 = 0,
    GUILD_CUDA_KV_APPEND_STEP_F16_V1 = 1,
    GUILD_CUDA_KV_GATHER_SYNC_V1 = 2,
    GUILD_CUDA_KV_GATHER_STEP_F16_V1 = 3,
    GUILD_CUDA_KV_APPEND_STEP_Q8_V1 = 4,
    GUILD_CUDA_KV_GATHER_STEP_Q8_V1 = 5,
    GUILD_CUDA_KV_FWHT256_V1 = 6,
    GUILD_CUDA_KV_APPEND_STEP_Q4_V1 = 7,
    GUILD_CUDA_KV_APPEND_BATCH_Q4_V1 = 8,
    GUILD_CUDA_KV_GATHER_STEP_Q4_V1 = 9,
    GUILD_CUDA_KV_STREAM_RESET_V1 = 10,
    GUILD_CUDA_KV_STREAM_INVALIDATE_V1 = 11,
    GUILD_CUDA_KV_STREAM_RESOLVE_V1 = 12,
    GUILD_CUDA_KV_RING_TABLE_V1 = 13,
    GUILD_CUDA_KV_RING_RESTORE_V1 = 14,
    GUILD_CUDA_KV_STAGE_FROM_HOST_V1 = 15,
    GUILD_CUDA_KV_UNSTAGE_TO_HOST_V1 = 16,
    GUILD_CUDA_KV_STREAM_COUNTERS_V1 = 17
};

typedef struct guild_cuda_kv_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    void* k_pool;
    void* v_pool;
    uint16_t* k_scale;
    uint16_t* v_scale;
    const void* k_pool_ro;
    const void* v_pool_ro;
    const uint16_t* k_scale_ro;
    const uint16_t* v_scale_ro;
    int32_t* page_table_mut;
    const int32_t* page_table;
    const int32_t* step;
    const int32_t* ids;
    const float* kcur;
    const float* vcur;
    float* fwht_dst;
    uint16_t* k_scratch;
    uint16_t* v_scratch;
    int64_t pos_or_b0;
    int64_t count_or_b1;
    int64_t cap_or_n_slots;
    int32_t kv_format;
    uint32_t has_host_pools;
    uint32_t has_stage_pools;
    uint32_t reserved;
    guild_cuda_qsa_shapes_v1_t shapes;
    guild_cuda_kv_host_pools_v1_t host_pools;
    guild_cuda_kv_host_pools_v1_t stage_pools;
    guild_cuda_qsa_attn_pools_v1_t attn_pools;
    guild_cuda_kv_stream_map_v1_t stream_map;
    guild_cuda_kv_stream_counters_v1_t* out_counters;
} guild_cuda_kv_op_v1_t;

enum guild_cuda_qsa_index_select_mode_v1_e {
    GUILD_CUDA_QSA_INDEXER_APPEND_V1 = 0,
    GUILD_CUDA_QSA_INDEXER_NATIVE_APPEND_V1 = 1,
    GUILD_CUDA_QSA_INDEXER_NATIVE_BATCH_V1 = 2,
    GUILD_CUDA_QSA_INDEX_SYNC_V1 = 3,
    GUILD_CUDA_QSA_INDEX_STEP_V1 = 4,
    GUILD_CUDA_QSA_NATIVE_SCORE_V1 = 5,
    GUILD_CUDA_QSA_TOPK_512_SYNC_V1 = 6,
    GUILD_CUDA_QSA_TOPK_512_STEP_V1 = 7,
    GUILD_CUDA_QSA_BLOCK_SCORES_V1 = 8,
    GUILD_CUDA_QSA_BLOCK_SCORES_TC_V1 = 9,
    GUILD_CUDA_QSA_BLOCK_TOPK_V1 = 10,
    GUILD_CUDA_QSA_BLOCK_TOPK_CLUSTER_V1 = 11,
    GUILD_CUDA_QSA_BLOCK_TOPK_REF_V1 = 12
};

typedef struct guild_cuda_qsa_index_select_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    const float* raw_or_pooled;
    const float* dead_or_query;
    const float* q_idx_or_gamma;
    const float* bias_or_cos;
    const float* sin_tab;
    const int32_t* step_or_pos_dev;
    float* scores_out;
    int32_t* ids_out;
    uint32_t* out_launched;
    int64_t n_or_nq;
    int64_t p0_or_n_bid;
    int64_t max_blocks;
    int64_t max_cells_or_cap;
    int64_t active_blocks;
    int32_t pos_base;
    float epsilon;
    guild_cuda_qsa_indexer_buffers_v1_t buffers;
    guild_cuda_qsa_shapes_v1_t shapes;
    guild_cuda_rope_scaling_v1_t scaling;
} guild_cuda_qsa_index_select_v1_t;

enum guild_cuda_attention_mode_v1_e {
    GUILD_CUDA_ATTN_QSA_SYNC_V1 = 0,
    GUILD_CUDA_ATTN_QSA_STEP_V1 = 1,
    GUILD_CUDA_ATTN_DECODE_STEP_V1 = 2,
    GUILD_CUDA_ATTN_DECODE_BATCH_V1 = 3,
    GUILD_CUDA_ATTN_PROMPT_BATCH_V1 = 4,
    GUILD_CUDA_ATTN_NATIVE_FLASH_SHORT_V1 = 5
};

typedef struct guild_cuda_attention_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    const float* q;
    const uint16_t* k_scratch;
    const uint16_t* v_scratch;
    const uint16_t* mask;
    const int32_t* ids;
    const int32_t* steps;
    float* scratch;
    float* attn_out;
    float* weights_out;
    int32_t* status_out;
    uint32_t* out_launched;
    int64_t cap_or_n_ids;
    int64_t n_q;
    int64_t id_stride;
    int32_t max_context;
    uint32_t reserved;
    guild_cuda_qsa_attn_pools_v1_t pools;
    guild_cuda_qsa_shapes_v1_t shapes;
} guild_cuda_attention_op_v1_t;

// No C++ objects/exceptions or CUDA SDK types cross this boundary. All calls
// return status and fill an optional caller-owned error record. Explicit-stream
// operations allocate nothing and perform no host synchronization.
typedef struct guild_cuda_attn_ext_v1_s {
    uint32_t struct_size;
    uint32_t tag;
    uint32_t version;
    uint32_t reserved;
    guild_status_t (*configure)(const guild_exec_context_v1_t*, const guild_cuda_attn_config_v1_t*, guild_error_info_t*);
    guild_status_t (*rope)(const guild_exec_context_v1_t*, const guild_cuda_rope_op_v1_t*, guild_error_info_t*);
    guild_status_t (*qsa_norm_gate)(const guild_exec_context_v1_t*, const guild_cuda_qsa_norm_gate_v1_t*, guild_error_info_t*);
    guild_status_t (*gdn)(const guild_exec_context_v1_t*, const guild_cuda_gdn_op_v1_t*, guild_error_info_t*);
    guild_status_t (*kv)(const guild_exec_context_v1_t*, const guild_cuda_kv_op_v1_t*, guild_error_info_t*);
    guild_status_t (*qsa_index_select)(const guild_exec_context_v1_t*, const guild_cuda_qsa_index_select_v1_t*, guild_error_info_t*);
    guild_status_t (*attention)(const guild_exec_context_v1_t*, const guild_cuda_attention_op_v1_t*, guild_error_info_t*);
} guild_cuda_attn_ext_v1_t;

#ifdef __cplusplus
}
#endif
