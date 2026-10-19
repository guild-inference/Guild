// include/guild/abi/cuda_moe_abi.h - Public C MoE, S-family GEMV/dequant, elementwise,
// sampler, speculative verify, and isolated Qwen specialization CUDA contracts.
// No C++ types, STL containers or private Guild headers may appear here.
#ifndef GUILD_ABI_CUDA_MOE_ABI_H
#define GUILD_ABI_CUDA_MOE_ABI_H

#include "guild/abi/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GUILD_CUDA_MOE_TAG_V1 0x43444d31u // "CDM1"
#define GUILD_CUDA_MOE_VERSION_V1 1u

enum guild_cuda_moe_config_flags_v1_e {
    GUILD_CUDA_MOE_FLAG_NONE_V1 = 0u,
    GUILD_CUDA_MOE_FLAG_SHARED_NATIVE_BF16_V1 = 1u << 0,
    GUILD_CUDA_MOE_FLAG_NATIVE_ROUTER_V1 = 1u << 1,
    GUILD_CUDA_MOE_FLAG_NATIVE_MOE_COMBINE_V1 = 1u << 2,
    GUILD_CUDA_MOE_FLAG_GR_FP32_ACT_V1 = 1u << 3,
    GUILD_CUDA_MOE_FLAG_GR_NATIVE_MMVF_V1 = 1u << 4,
    GUILD_CUDA_MOE_FLAG_PLE_NATIVE_BF16_V1 = 1u << 5,
    GUILD_CUDA_MOE_FLAG_PLE_NATIVE_POSTOPS_V1 = 1u << 6,
    GUILD_CUDA_MOE_FLAG_NATIVE_GROUPED_V1_V1 = 1u << 7,
    GUILD_CUDA_MOE_FLAG_IQ_OLD_KERNELS_V1 = 1u << 8
};

enum guild_cuda_moe_config_mode_v1_e {
    GUILD_CUDA_MOE_CFG_SET_FLAGS_V1 = 0u,
    GUILD_CUDA_MOE_CFG_QUERY_GROUPED_LAST_PATH_V1 = 1u,
    GUILD_CUDA_MOE_CFG_DEQUANT_BF16_SUPPORTED_V1 = 2u,
    GUILD_CUDA_MOE_CFG_COUPLED_SCRATCH_BYTES_V1 = 3u,
    GUILD_CUDA_MOE_CFG_FUSED_GR_SUPPORTED_V1 = 4u,
    GUILD_CUDA_MOE_CFG_FUSED_GR_SET_FAST_V1 = 5u,
    GUILD_CUDA_MOE_CFG_FUSED_GR_CHECK_V1 = 6u,
    GUILD_CUDA_MOE_CFG_FUSED_GR_VARIANT_V1 = 7u,
    GUILD_CUDA_MOE_CFG_PLE_AVAILABLE_V1 = 8u,
    GUILD_CUDA_MOE_CFG_NATIVE_EXPERT_SET_MODE_V1 = 9u
};

typedef struct guild_cuda_moe_config_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    uint32_t flags;
    int32_t grouped_select_old; // -2 = unchanged, -1 = env, 0 = new, 1 = old
    int32_t value;              // ggml_type, n_vocab, fast flag, or expert mode
    int32_t value2;             // expert phase
    int64_t n_embd;
    int64_t hc;
    int64_t hc_lr;
    int32_t* out_value;
    uint64_t* out_bytes;
} guild_cuda_moe_config_v1_t;

typedef struct guild_cuda_sform_v1_s {
    uint32_t struct_size;
    int32_t code_bits;
    int32_t code_bias;
    int32_t group_elems;
    int32_t codebook; // 0 = Affine, 1 = Iq4Nl
    uint32_t has_offset;
    int32_t act_kind; // 0 = Q8_0, 1 = Q8_K
} guild_cuda_sform_v1_t;

typedef struct guild_cuda_native_shared_weights_v1_s {
    uint32_t struct_size;
    int32_t gate_type;
    int32_t up_type;
    int32_t down_type;
    const void* gate_data;
    const void* up_data;
    const void* down_data;
    void* q8_1;
    const void* x_q8_1;
} guild_cuda_native_shared_weights_v1_t;

typedef struct guild_cuda_sampler_params_v1_s {
    uint32_t struct_size;
    int32_t top_k;
    float top_p;
    float min_p;
    float temperature;
    int32_t min_keep;
    int32_t penalty_last_n;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
    uint64_t seed;
    uint64_t counter;
    uint32_t greedy;
} guild_cuda_sampler_params_v1_t;

typedef struct guild_cuda_gr_shapes_v1_s {
    uint32_t struct_size;
    int64_t n_embd;
    int64_t hc;
    int64_t hc_lr;
} guild_cuda_gr_shapes_v1_t;

typedef struct guild_cuda_gr_workspace_v1_s {
    uint32_t struct_size;
    float* xn;
    uint16_t* xq;
    uint16_t* lq;
    float* gated;
    float* lo;
    uint64_t bytes;
} guild_cuda_gr_workspace_v1_t;

typedef struct guild_cuda_fused_gr_args_v1_s {
    uint32_t struct_size;
    const float* R;
    float* R_out;
    uint32_t apply;
    const float* bo_prev;
    const float* inj_prev;
    const float* w_norm;
    const uint16_t* w_down;
    const uint16_t* w_up;
    const uint16_t* w_inject;
    float eps;
    float* lo;
    float* rs;
    float* inject_out;
    float* mixed;
} guild_cuda_fused_gr_args_v1_t;

typedef struct guild_cuda_ple_weights_v1_s {
    uint32_t struct_size;
    const uint8_t* key_codes;
    const float* key_scales;
    const uint16_t* value_bf16;
    const float* norm_key;
    const float* norm_query;
    const float* norm_conv;
    const uint16_t* conv1d_f16;
    const void* key_native_data;
    int32_t key_native_type;
    void* key_native_q8_1;
    const uint16_t* key_bf16;
} guild_cuda_ple_weights_v1_t;

typedef struct guild_cuda_ple_out_v1_s {
    uint32_t struct_size;
    float* key;
    float* value;
    float* gate;
    float* gated;
    float* normalized;
    float* conv;
    float* result;
} guild_cuda_ple_out_v1_t;

typedef struct guild_cuda_native_ple_postops_buffers_v1_s {
    uint32_t struct_size;
    float* key;
    float* query;
    float* gate;
    float* gated;
    float* normalized;
    float* conv;
    float* result;
} guild_cuda_native_ple_postops_buffers_v1_t;

enum guild_cuda_quant_dequant_mode_v1_e {
    GUILD_CUDA_QD_DEQUANT_BF16_V1 = 0u,
    GUILD_CUDA_QD_DEQUANT_F16_V1 = 1u,
    GUILD_CUDA_QD_DEQUANT_F32_V1 = 2u,
    GUILD_CUDA_QD_DEQUANT_S2_V1 = 3u,
    GUILD_CUDA_QD_QUANTIZE_Q8_0_V1 = 4u,
    GUILD_CUDA_QD_QUANTIZE_Q8_0_SCALED_V1 = 5u,
    GUILD_CUDA_QD_DEQUANT_Q8_0_V1 = 6u,
    GUILD_CUDA_QD_QUANTIZE_Q8_K_V1 = 7u,
    GUILD_CUDA_QD_DEQUANT_Q8_K_V1 = 8u,
    GUILD_CUDA_QD_QUANTIZE_Q8_1_ROWS_V1 = 9u,
    GUILD_CUDA_QD_IQ_DEQUANT_F16_V1 = 10u,
    GUILD_CUDA_QD_IQ_DEQUANT_F32_V1 = 11u,
    GUILD_CUDA_QD_IQ_EMBED_ROWS_V1 = 12u,
    GUILD_CUDA_QD_IQ_DEQUANT_GU_F16_V1 = 13u,
    GUILD_CUDA_QD_IQ_MMVQ_V1 = 14u
};

typedef struct guild_cuda_quant_dequant_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    int32_t ggml_type;
    const void* src;
    const void* src2;
    const float* scales;
    const int32_t* tokens;
    void* dst;
    float* out_scales;
    uint64_t row_bytes;
    int64_t row0;
    int64_t rows;
    int64_t cols;
    int64_t n;
    int32_t ncols;
} guild_cuda_quant_dequant_op_v1_t;

enum guild_cuda_s_gemv_mode_v1_e {
    GUILD_CUDA_SGEMV_S2_V1 = 0u,
    GUILD_CUDA_SGEMV_S2_QUADS_V1 = 1u,
    GUILD_CUDA_SGEMV_S2_FAST_V1 = 2u,
    GUILD_CUDA_SGEMV_S2_Q8_V1 = 3u,
    GUILD_CUDA_SGEMV_S_V1 = 4u,
    GUILD_CUDA_SGEMV_S_SPLIT_V1 = 5u,
    GUILD_CUDA_SGEMV_S_SPLIT_ASYNC_V1 = 6u,
    GUILD_CUDA_SGEMV_S_Q8K_V1 = 7u,
    GUILD_CUDA_SGEMV_S_Q8K_SPLIT_V1 = 8u,
    GUILD_CUDA_SGEMV_S_Q8_0_SPLIT_V1 = 9u
};

typedef struct guild_cuda_s_gemv_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    guild_cuda_sform_v1_t form;
    const void* x;
    const uint8_t* codes;
    const float* scales;
    const float* offset;
    float* y;
    int64_t n_in;
    int64_t n_out;
    int32_t threads_per_row;
    uint32_t stage_x;
} guild_cuda_s_gemv_op_v1_t;

enum guild_cuda_moe_mode_v1_e {
    GUILD_CUDA_MOE_ROUTER_TOP10_V1 = 0u,
    GUILD_CUDA_MOE_ROUTER_TOP10_VARIANT_V1 = 1u,
    GUILD_CUDA_MOE_NATIVE_ROUTER_TOP10_V1 = 2u,
    GUILD_CUDA_MOE_NATIVE_ROUTER_TOP10_MULTI_V1 = 3u,
    GUILD_CUDA_MOE_COMBINE_V1 = 4u,
    GUILD_CUDA_MOE_NATIVE_COMBINE_V1 = 5u,
    GUILD_CUDA_MOE_NATIVE_COMBINE_MULTI_V1 = 6u,
    GUILD_CUDA_MOE_SHARED_EXPERT_V1 = 7u,
    GUILD_CUDA_MOE_SHARED_EXPERT_MULTI_V1 = 8u,
    GUILD_CUDA_MOE_HIT_SELECT_V1 = 9u,
    GUILD_CUDA_MOE_HIT_SELECT_MULTI_V1 = 10u,
    GUILD_CUDA_MOE_HIT_ADD_V1 = 11u,
    GUILD_CUDA_MOE_HIT_GROUPED_S2_V1 = 12u,
    GUILD_CUDA_MOE_HIT_GROUPED_S2_DEV_V1 = 13u,
    GUILD_CUDA_MOE_HIT_GROUPED_S2_MULTI_V1 = 14u,
    GUILD_CUDA_MOE_HIT_GROUPED_S2_CPU_ORDER_V1 = 15u,
    GUILD_CUDA_MOE_GROUPED_S2_V1 = 16u,
    GUILD_CUDA_MOE_GROUP_RESIDENT_V1 = 17u,
    GUILD_CUDA_MOE_REMOTE_REDUCE_V1 = 18u,
    GUILD_CUDA_MOE_REMOTE_COPY_CPU_ROWS_V1 = 19u,
    GUILD_CUDA_MOE_NATIVE_EXPERT_GROUPED_V1 = 20u
};

typedef struct guild_cuda_moe_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    // Router / Combine / Remote fields
    const float* logits;
    const float* parts;
    const float* weights;
    const float* shared;
    float* out;
    int32_t* ids;
    int32_t n_tokens;
    int32_t n_expert;
    int64_t n_embd;
    int64_t n_ff;
    int64_t k;
    int32_t variant;
    int32_t gu_type;
    int32_t d_type;
    int64_t grid_groups;
    uint32_t* out_supported;
    // Shared expert fields
    const uint8_t* x_q8_0;
    const uint8_t* x_q8k;
    const uint16_t* x_bf16;
    const float* x_f32;
    const void* x_q8_1;
    guild_cuda_sform_v1_t gate_form;
    const uint8_t* gate_codes;
    const float* gate_scales;
    const float* gate_off;
    guild_cuda_sform_v1_t up_form;
    const uint8_t* up_codes;
    const float* up_scales;
    const float* up_off;
    guild_cuda_sform_v1_t down_form;
    const uint8_t* down_codes;
    const float* down_scales;
    const float* down_off;
    const uint16_t* gate_inp_bf16;
    void* scratch;
    float* gate_buf;
    float* up_buf;
    float* g_buf;
    int32_t tpr;
    uint32_t has_native_shared;
    guild_cuda_native_shared_weights_v1_t native_shared;
    // Grouped S2 / Residency fields
    const uint8_t* blob_base;
    const int32_t* res_row;
    int32_t* slot_out;
    int32_t* dst_out;
    int32_t* count_out;
    const int32_t* slot_index;
    const int32_t* dst_index;
    const int32_t* d_count;
    int64_t n_hits;
    int64_t cap;
    int64_t blob_bytes;
    const float* x_scales;
    int32_t k_per_token;
    float* gate_up_trace;
    const unsigned long long* grp_ptr;
    unsigned long long* grp_ptr_mut;
    const int32_t* grp_start;
    int32_t* grp_start_mut;
    const int32_t* n_groups;
    const int32_t* ent_dst;
    int32_t* ent_dst_mut;
    const int32_t* ent_tok;
    int32_t* ent_tok_mut;
    int64_t cap_groups;
    int64_t cap_entries;
    const void* reduce_meta;
    const int32_t* mask;
} guild_cuda_moe_op_v1_t;

enum guild_cuda_elementwise_mode_v1_e {
    GUILD_CUDA_ELEM_EMBEDDING_GATHER_V1 = 0u,
    GUILD_CUDA_ELEM_GDN_GATE_V1 = 1u,
    GUILD_CUDA_ELEM_SCALE_INPLACE_V1 = 2u,
    GUILD_CUDA_ELEM_ADD_INPLACE_V1 = 3u,
    GUILD_CUDA_ELEM_F32_TO_F16_V1 = 4u,
    GUILD_CUDA_ELEM_F32_TO_BF16_V1 = 5u,
    GUILD_CUDA_ELEM_SILU_INPLACE_V1 = 6u,
    GUILD_CUDA_ELEM_RMS_NORM_WEIGHTED_V1 = 7u,
    GUILD_CUDA_ELEM_DOORBELL_RING_V1 = 8u,
    GUILD_CUDA_ELEM_DOORBELL_WAIT_V1 = 9u,
    GUILD_CUDA_ELEM_COPY_FROM_MAPPED_V1 = 10u,
    GUILD_CUDA_ELEM_SCATTER_ROWS_F32_V1 = 11u,
    GUILD_CUDA_ELEM_COPY_ROWS_FROM_MAPPED_V1 = 12u,
    GUILD_CUDA_ELEM_DOORBELL_PUBLISH_V1 = 13u,
    GUILD_CUDA_ELEM_DOORBELL_PUBLISH_VALUE_V1 = 14u,
    GUILD_CUDA_ELEM_DOORBELL_PUBLISH_RES_V1 = 15u,
    GUILD_CUDA_ELEM_COPY_I32_FROM_MAPPED_V1 = 16u
};

typedef struct guild_cuda_elementwise_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    const uint8_t* codes;
    const float* scales;
    const float* offsets;
    const float* src;
    const float* dt;
    const float* ssm_a;
    float* dst;
    uint16_t* dst_u16;
    const int32_t* rows_idx;
    const int32_t* hit_rows;
    const int32_t* count;
    const int32_t* ids;
    const int32_t* d_res;
    int32_t* ids_out;
    float* weights_out;
    uint32_t* d_seq;
    const uint32_t* d_flag;
    uint32_t value;
    int64_t n;
    int64_t rows;
    int64_t cols;
    int32_t code_bits;
    int32_t code_bias;
    int32_t group_elems;
    int32_t n_expert;
    float scale;
    float eps;
} guild_cuda_elementwise_op_v1_t;

enum guild_cuda_sampler_mode_v1_e {
    GUILD_CUDA_SAMPLER_SAMPLE_TOKENS_V1 = 0u,
    GUILD_CUDA_SAMPLER_GREEDY_CLUSTER_V1 = 1u,
    GUILD_CUDA_SAMPLER_COUPLED_STAGE_V1 = 2u,
    GUILD_CUDA_SAMPLER_COUPLED_SAMPLE_V1 = 3u
};

typedef struct guild_cuda_sampler_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    const float* logits;
    float* logits_mut;
    int32_t n_tokens;
    int32_t n_vocab;
    const int32_t* history;
    int32_t history_len;
    guild_cuda_sampler_params_v1_t params;
    const void* mapped_params;
    const int32_t* mapped_hist;
    void* dev_params;
    int32_t* ring;
    int32_t cap;
    int32_t j;
    const int32_t* sub_to_id;
    const int32_t* id_to_sub;
    int32_t id_vocab;
    const int32_t* step_rec;
    void* scratch;
    int32_t* out;
    float* out_prob;
    uint32_t* out_launched;
} guild_cuda_sampler_op_v1_t;

enum guild_cuda_verify_mode_v1_e {
    GUILD_CUDA_VERIFY_FINITE_FIRST_BAD_V1 = 0u,
    GUILD_CUDA_VERIFY_WAIT_FLAG_GE_V1 = 1u,
    GUILD_CUDA_VERIFY_GPU_STAMP_V1 = 2u,
    GUILD_CUDA_VERIFY_RESIDENT_PLAN_V1 = 3u,
    GUILD_CUDA_VERIFY_WAIT_FLAG_GE_OR_V1 = 4u,
    GUILD_CUDA_VERIFY_COPY_I32_UNLESS_V1 = 5u,
    GUILD_CUDA_VERIFY_COPY_OR_ZERO_V1 = 6u,
    GUILD_CUDA_VERIFY_EMBEDDING_GATHER_DEV_V1 = 7u,
    GUILD_CUDA_VERIFY_BROADCAST_STREAMS_V1 = 8u,
    GUILD_CUDA_VERIFY_COPY_INDEXED_V1 = 9u,
    GUILD_CUDA_VERIFY_FETCH_BLOBS_V1 = 10u,
    GUILD_CUDA_VERIFY_REBASE_PTRS_V1 = 11u,
    GUILD_CUDA_VERIFY_ADD_STREAMS_BROADCAST_V1 = 12u,
    GUILD_CUDA_VERIFY_IDENT_HITS_V1 = 13u,
    GUILD_CUDA_VERIFY_MTP_SELECT_V1 = 14u,
    GUILD_CUDA_VERIFY_GATHER_ROWS_V1 = 15u,
    GUILD_CUDA_VERIFY_MAP_IDS_V1 = 16u,
    GUILD_CUDA_VERIFY_ROW_TOP_PROB_V1 = 17u,
    GUILD_CUDA_VERIFY_DENSE_STEPS_V1 = 18u,
    GUILD_CUDA_VERIFY_WINDOW_IDS_V1 = 19u
};

typedef struct guild_cuda_verify_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    const float* values;
    const float* src2;
    float* dst;
    float* out_p;
    unsigned long long* u64_dst;
    const unsigned long long* u64_src;
    const uint8_t* bytes_src;
    uint8_t* bytes_dst;
    const float* scales;
    const float* offsets;
    const int32_t* i32_src;
    const int32_t* i32_src2;
    int32_t* i32_dst;
    int32_t* i32_dst2;
    int32_t* i32_dst3;
    const uint32_t* u32_flag;
    const uint32_t* u32_skip;
    uint32_t* u32_skip_mut;
    uint32_t u32_val;
    uint64_t count;
    uint64_t row_codes;
    uint64_t row_groups;
    int64_t n;
    int64_t stride;
    int64_t blob_bytes;
    int32_t n_tok;
    int32_t k;
    int32_t n_expert;
    int32_t hc;
    int32_t code_bits;
    int32_t code_bias;
    int32_t group_elems;
    int32_t cap;
    int32_t idx;
} guild_cuda_verify_op_v1_t;

enum guild_cuda_qwen_mode_v1_e {
    GUILD_CUDA_QWEN_GR_READ_V1 = 0u,
    GUILD_CUDA_QWEN_GR_WRITE_V1 = 1u,
    GUILD_CUDA_QWEN_FUSED_GR_READ_V1 = 2u,
    GUILD_CUDA_QWEN_FUSED_GR_READ_MULTI_V1 = 3u,
    GUILD_CUDA_QWEN_NATIVE_GR_RMS_NORM_V1 = 4u,
    GUILD_CUDA_QWEN_NATIVE_GR_DOWN_SILU_V1 = 5u,
    GUILD_CUDA_QWEN_NATIVE_GR_PRE_GATED_V1 = 6u,
    GUILD_CUDA_QWEN_NATIVE_GR_POST_V1 = 7u,
    GUILD_CUDA_QWEN_PLE_BLOCK_V1 = 8u,
    GUILD_CUDA_QWEN_PLE_BLOCK_PROJECTED_V1 = 9u,
    GUILD_CUDA_QWEN_PLE_HISTORY_ADVANCE_V1 = 10u,
    GUILD_CUDA_QWEN_NATIVE_PLE_POSTOPS_V1 = 11u,
    GUILD_CUDA_QWEN_NATIVE_PLE_POSTOPS_BATCH_V1 = 12u,
    GUILD_CUDA_QWEN_CVEC_APPLY_V1 = 13u
};

typedef struct guild_cuda_qwen_op_v1_s {
    uint32_t struct_size;
    uint32_t mode;
    guild_cuda_gr_shapes_v1_t gr_shapes;
    guild_cuda_gr_workspace_v1_t gr_ws;
    guild_cuda_fused_gr_args_v1_t fused_gr;
    const guild_cuda_fused_gr_args_v1_t* fused_gr_multi;
    guild_cuda_ple_weights_v1_t ple_weights;
    guild_cuda_ple_out_v1_t ple_out;
    guild_cuda_native_ple_postops_buffers_v1_t ple_buffers;
    const float* R;
    const float* w_norm;
    const uint16_t* w_down;
    const uint16_t* w_up;
    const uint16_t* w_inject;
    const float* block_out;
    const float* inject;
    float* mixed;
    float* inject_out;
    float* R_out;
    float* xn_scratch;
    unsigned long long* stamp_buf;
    int32_t stamp_i0;
    int32_t n_tok;
    int32_t n_cols;
    int32_t n_rows;
    float eps;
    uint32_t fused_layer;
    const float* emb;
    const float* hidden;
    float* hidden_mut;
    const float* hist_rows;
    float* hist_mut;
    const float* projected_key;
    float* key_mut;
    const float* projected_value;
    const float* normalized;
    float* query_norm;
    float* gated;
    float* gate;
    void* scratch;
    // cvec fields
    const float* cvec_dir;
    const float* cvec_s;
    const int32_t* cvec_on;
    int32_t cvec_mode;
    int64_t layer;
    int64_t r_ld;
    int64_t bo_ld;
    int64_t inj_ld;
    uint32_t write;
} guild_cuda_qwen_op_v1_t;

typedef struct guild_cuda_moe_ext_v1_s {
    uint32_t struct_size;
    uint32_t tag;
    uint32_t version;
    uint32_t reserved;
    guild_status_t (*configure)(const guild_exec_context_v1_t* ctx, const guild_cuda_moe_config_v1_t* params,
                                guild_error_info_t* err);
    guild_status_t (*quantize_dequant)(const guild_exec_context_v1_t* ctx,
                                       const guild_cuda_quant_dequant_op_v1_t* params, guild_error_info_t* err);
    guild_status_t (*s_gemv)(const guild_exec_context_v1_t* ctx, const guild_cuda_s_gemv_op_v1_t* params,
                             guild_error_info_t* err);
    guild_status_t (*moe)(const guild_exec_context_v1_t* ctx, const guild_cuda_moe_op_v1_t* params,
                          guild_error_info_t* err);
    guild_status_t (*elementwise)(const guild_exec_context_v1_t* ctx, const guild_cuda_elementwise_op_v1_t* params,
                                  guild_error_info_t* err);
    guild_status_t (*sampler)(const guild_exec_context_v1_t* ctx, const guild_cuda_sampler_op_v1_t* params,
                              guild_error_info_t* err);
    guild_status_t (*verify)(const guild_exec_context_v1_t* ctx, const guild_cuda_verify_op_v1_t* params,
                             guild_error_info_t* err);
    guild_status_t (*qwen)(const guild_exec_context_v1_t* ctx, const guild_cuda_qwen_op_v1_t* params,
                           guild_error_info_t* err);
} guild_cuda_moe_ext_v1_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GUILD_ABI_CUDA_MOE_ABI_H
