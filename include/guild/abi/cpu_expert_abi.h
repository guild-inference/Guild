// Copyright (c) Guild Inference Contributors. MIT License.
#pragma once
#include "guild/abi/kernel_abi.h"
#ifdef __cplusplus
extern "C" {
#define GUILD_CPU_DEFAULT(v) = v
#else
#define GUILD_CPU_DEFAULT(v)
#endif
#define GUILD_CPU_EXPERT_TAG_V1 0x43504531u /* CPE1 */
// Public format values, never backend state. Sizes/offsets are bytes and fixed width.
typedef struct guild_cpu_expert_format_v1_s {
    int32_t gu_type GUILD_CPU_DEFAULT(-1), d_type GUILD_CPU_DEFAULT(-1);
    int32_t gu_act GUILD_CPU_DEFAULT(-1), d_act GUILD_CPU_DEFAULT(-1);
    int64_t n_embd GUILD_CPU_DEFAULT(0), n_ff GUILD_CPU_DEFAULT(0);
    uint64_t gu_row GUILD_CPU_DEFAULT(0), d_row GUILD_CPU_DEFAULT(0);
    uint64_t up_off GUILD_CPU_DEFAULT(0), down_off GUILD_CPU_DEFAULT(0);
    uint64_t bytes GUILD_CPU_DEFAULT(0), act_bytes GUILD_CPU_DEFAULT(0), h_bytes GUILD_CPU_DEFAULT(0);
} guild_cpu_expert_format_v1_t;

// Pre-bound, host-synchronous arithmetic. The caller validates dimensions, owns
// all buffers and parallel scheduling, and holds the provider's package lease.
typedef struct guild_cpu_expert_ext_v1_s {
    uint32_t struct_size, tag, version;
    void (*quant_act)(const guild_cpu_expert_format_v1_t*, const float*, void*);
    void (*quant_h)(const guild_cpu_expert_format_v1_t*, const float*, void*);
    void (*gate_up)(const guild_cpu_expert_format_v1_t*, const uint8_t*, const void* const*, int32_t, float* const*,
                    int32_t, int32_t);
    void (*down)(const guild_cpu_expert_format_v1_t*, const uint8_t*, const void* const*, int32_t, float* const*,
                 int32_t, int32_t);
} guild_cpu_expert_ext_v1_t;
#undef GUILD_CPU_DEFAULT
#ifdef __cplusplus
}
#endif
