// Copyright (c) Guild Inference Contributors. MIT License.
// Optional, independently versioned operation contract; Kernel ABI v1 is unchanged.
#pragma once
#include "guild/abi/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GUILD_MODULE_ABI_V1 1u
#define GUILD_MODULE_ABI_TAG_V1 "guild.module.v1"
#define GUILD_OP_CUSTOM 65536
#define GUILD_OPERATION_TAG_V1 0x474f5031u /* GOP1 */
#define GUILD_LAYOUT_CONTIGUOUS 1u
#define GUILD_LAYOUT_STRIDED 2u
#define GUILD_SYNC_BIT(mode) (1u << (mode))
#define GUILD_MEMORY_BIT(space) (1u << (space))

// Required next_ext for GUILD_OP_CUSTOM. op_name is a namespaced semantic ID,
// independent of impl_id and package_id. All fields are copied at registration.
// Capability bits belong to the operation's versioned contract, not Core.
typedef struct guild_operation_ext_v1_s {
    uint32_t struct_size, tag, version;
    uint32_t operation_version; /* packed semantic version, nonzero major */
    uint64_t capabilities;
    uint32_t layouts;       /* nonempty GUILD_LAYOUT_* mask */
    uint32_t memory_spaces; /* nonempty GUILD_MEMORY_BIT mask */
    uint32_t sync_modes;    /* GUILD_SYNC_BIT; unlike legacy supported_sync_modes */
    uint32_t min_rank, max_rank;
    uint64_t alignment; /* power of two, >= 1 */
} guild_operation_ext_v1_t;

typedef struct guild_operation_query_v1_s {
    uint32_t struct_size, version;
    const char* operation_id;
    uint32_t min_version, max_version; /* inclusive, same major, max required */
    uint32_t allowed_backends;
    uint64_t required_capabilities;
    uint32_t numerical_flags;
    uint32_t sync_mode;
    uint64_t available_ram_bytes, available_vram_bytes;
    const char* exact_impl_id; /* NULL: compatible fallback permitted; otherwise strict */
    const guild_tensor_desc_t* inputs;
    size_t n_inputs;
    const guild_tensor_desc_t* outputs;
    size_t n_outputs;
} guild_operation_query_v1_t;

typedef struct guild_operation_binding_s* guild_operation_binding_t;

// Copy this table during plugin init; it is not the temporary registration host.
// user remains valid for the lifetime of the owning registry. Handles own provider
// bindings. Buffers/params stay caller-owned. resolve copies tensor descriptors.
// Device capabilities next_ext must be NULL in v1; no borrowed tail is retained.
// execute uses those buffers, validates context/scratch and never reselects.
// ctx.host_scratch contains pinned_host_bytes + pageable_host_bytes; the entire
// region must be pinned if pinned_host_bytes is nonzero. Device scratch is separate.
// release waits for recorded GPU streams. For graph capture, keep the handle until
// the graph is destroyed. Do not release concurrently with calls on that handle.
// CPU execute is synchronous; GPU execution is on ctx.device_stream only.
typedef struct guild_operation_api_v1_s {
    uint32_t struct_size, version;
    void* user;
    guild_status_t (*resolve)(void*, const guild_operation_query_v1_t*, const guild_device_capabilities_v1_t*,
                              guild_operation_binding_t*, guild_error_info_t*);
    const guild_kernel_op_desc_v1_t* (*describe)(guild_operation_binding_t);
    guild_status_t (*scratch)(guild_operation_binding_t, guild_scratch_request_t*, guild_error_info_t*);
    guild_status_t (*execute)(guild_operation_binding_t, const guild_exec_context_v1_t*, const void*, size_t,
                              guild_error_info_t*);
    void (*release)(guild_operation_binding_t);
} guild_operation_api_v1_t;

#ifdef __cplusplus
}
#endif
