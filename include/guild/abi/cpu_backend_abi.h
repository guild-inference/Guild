// Copyright (c) Guild Inference Contributors. MIT License.
// Append-only CPU backend extension. The CPE1 prefix remains binary compatible.
#pragma once
#include "guild/abi/cpu_expert_abi.h"
#if defined(_MSC_VER)
#define GUILD_CPU_ALIGN64 __declspec(align(64))
#else
#define GUILD_CPU_ALIGN64 __attribute__((aligned(64)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GUILD_CPU_BACKEND_VERSION_V2 2u
#define GUILD_CPU_PLANAR_EMBED_V1 2560
#define GUILD_CPU_PLANAR_FF_V1 640
#define GUILD_CPU_PLANAR_CHUNKS_V1 80

// Canonical Q2 storage contract, not a private C++ object. All fields, padding
// and 64-byte alignment are specified; caller owns buffers and task scheduling.
typedef struct GUILD_CPU_ALIGN64 guild_cpu_planar_act_v1_s {
    int8_t q[GUILD_CPU_PLANAR_EMBED_V1];
    float scale[GUILD_CPU_PLANAR_CHUNKS_V1];
    int32_t sum[GUILD_CPU_PLANAR_CHUNKS_V1];
    float hx[GUILD_CPU_PLANAR_CHUNKS_V1];
    int32_t nchunks;
    uint8_t reserved[60];
} guild_cpu_planar_act_v1_t;

typedef struct guild_cpu_backend_ext_v2_s {
    guild_cpu_expert_ext_v1_t native; // original CPE1 prefix; struct_size = sizeof(v2)
    uint32_t backend_version;
    uint32_t reserved;
    guild_status_t (*format)(int32_t, int32_t, int64_t, int64_t, guild_cpu_expert_format_v1_t*, guild_error_info_t*);
    void (*quant_planar)(const float*, int32_t, guild_cpu_planar_act_v1_t*);
    void (*canonical_gate_up)(const uint8_t*, const guild_cpu_planar_act_v1_t* const*, int32_t, float* const*, int32_t,
                              int32_t);
    void (*canonical_down)(const uint8_t*, const guild_cpu_planar_act_v1_t* const*, int32_t, float* const*, int32_t,
                           int32_t);
    void (*q2_gguf_rows)(const uint8_t*, uint64_t, int32_t, const guild_cpu_planar_act_v1_t* const*, int32_t,
                         float* const*, int32_t, int32_t);
    void (*router_bf16)(const uint16_t*, int32_t, int32_t, const float*, int32_t, float*);
    // Ordered fused multiply-add in routed-expert order; returns NONFINITE_OUTPUT
    // on non-finite expert values. No fallback is attempted on numerical failure.
    guild_status_t (*combine)(const float*, const float*, float*, int32_t, int32_t);
    // Optional append-only tail (old v2 prefixes end before this field).
    void (*swiglu)(const float*, const float*, float*, int32_t);
} guild_cpu_backend_ext_v2_t;

#ifdef __cplusplus
}
static_assert(sizeof(guild_cpu_planar_act_v1_t) == 3584, "canonical activation layout must remain frozen");
#endif
#undef GUILD_CPU_ALIGN64
