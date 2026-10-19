// plugins/runner_smoke_ext/runner_smoke_plugin.cpp
// External .gkernel model runner smoke plugin built strictly against the public Guild Kernel SDK (kernel_abi.h).
// Registers external runner `runner.ornith_ext_smoke.v1` for architecture `ornith_ext_smoke`.
//
// Copyright (c) Guild Inference Contributors. MIT License.

#include "guild/abi/kernel_abi.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace {

std::atomic<int64_t> g_runner_sessions_allocated{0};
std::atomic<int64_t> g_runner_shutdown_calls{0};

struct SmokeSessionState {
    int64_t max_context = 0;
};

guild_status_t smoke_session_alloc(const void*, int64_t max_context, guild_session_state_handle_t* out_state,
                                   guild_error_info_t* err) {
    if (!out_state || max_context <= 0) {
        if (err) {
            err->struct_size = sizeof(*err);
            err->code = GUILD_STATUS_ERR_INVALID_ARGUMENT;
            err->source_component = "runner.ornith_ext_smoke.v1";
            std::snprintf(err->message, sizeof(err->message), "invalid max_context for smoke runner session");
        }
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    auto* st = new SmokeSessionState{max_context};
    *out_state = reinterpret_cast<guild_session_state_handle_t>(st);
    g_runner_sessions_allocated.fetch_add(1, std::memory_order_relaxed);
    return GUILD_STATUS_OK;
}

void smoke_session_free(guild_session_state_handle_t state) {
    if (state) {
        delete reinterpret_cast<SmokeSessionState*>(state);
        g_runner_sessions_allocated.fetch_sub(1, std::memory_order_relaxed);
    }
}

guild_status_t runner_plugin_init(guild_core_host_t host, const guild_host_api_v1_t* api, guild_error_info_t* err) {
    if (!host || !api || api->struct_size < sizeof(guild_host_api_v1_t) || !api->register_runner) {
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    uint32_t negotiated = 0;
    if (api->negotiate_abi(GUILD_KERNEL_ABI_VERSION_V1, &negotiated) != GUILD_STATUS_OK) {
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }

    const guild_op_kind_t req_ops[] = {GUILD_OP_BATCHED_EXPERT_EXEC, GUILD_OP_FINITE_CHECK};
    const char* pkg_deps[] = {"builtin.guild-kernels-cpu"};

    guild_runner_desc_v1_t rdesc{};
    rdesc.struct_size = sizeof(rdesc);
    rdesc.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    rdesc.arch_id = "ornith_ext_smoke";
    rdesc.runner_id = "runner.ornith_ext_smoke.v1";
    rdesc.package_id = "ext.guild-runner-smoke";
    rdesc.runner_version = GUILD_KERNEL_ABI_VERSION_V1;
    rdesc.required_core_caps = GUILD_CORE_CAP_HOST_ONLY_KV;
    rdesc.required_op_kinds = req_ops;
    rdesc.n_required_op_kinds = sizeof(req_ops) / sizeof(req_ops[0]);
    rdesc.required_package_deps = pkg_deps;
    rdesc.n_required_package_deps = 1;
    rdesc.session_alloc = &smoke_session_alloc;
    rdesc.session_free = &smoke_session_free;

    return api->register_runner(host, &rdesc, err);
}

void runner_plugin_shutdown(guild_core_host_t) {
    g_runner_shutdown_calls.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

extern "C" {

GUILD_PLUGIN_EXPORT guild_status_t guild_plugin_entry_v1(uint32_t host_abi_version,
                                                         guild_plugin_descriptor_v1_t* out_plugin_desc,
                                                         guild_error_info_t*) {
    if (!out_plugin_desc)
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    if (GUILD_KERNEL_ABI_VERSION_MAJOR_OF(host_abi_version) != GUILD_KERNEL_ABI_VERSION_MAJOR) {
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    out_plugin_desc->struct_size = sizeof(*out_plugin_desc);
    out_plugin_desc->abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    out_plugin_desc->package_id = "ext.guild-runner-smoke";
    out_plugin_desc->package_version = "1.0.0";
    out_plugin_desc->publisher_id = "org.illatio.guild";
    out_plugin_desc->init = &runner_plugin_init;
    out_plugin_desc->shutdown = &runner_plugin_shutdown;
    return GUILD_STATUS_OK;
}

GUILD_PLUGIN_EXPORT int64_t guild_ext_runner_smoke_active_sessions(void) {
    return g_runner_sessions_allocated.load(std::memory_order_relaxed);
}

} // extern "C"
