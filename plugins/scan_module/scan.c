// Minimal specialized sequence primitive; no model or modality code in Core.
// Build this file using only the installed GuildKernelSDK (C99).
#include <guild/abi/module_abi.h>
#include <string.h>
#ifdef SCAN_CUDA
#include <cuda_runtime.h>
__global__ static void scan_kernel(const float* in, float* out, size_t n, int exclusive) {
    float sum = 0;
    for (size_t i = 0; i < n; ++i) {
        const float next = sum + in[i]; out[i] = exclusive ? sum : next; sum = next;
    }
}
#endif

#ifndef SCAN_PACKAGE
#define SCAN_PACKAGE "example.scan"
#endif
#ifndef SCAN_VERSION
#define SCAN_VERSION "1.0.0"
#endif
#ifndef SCAN_PRIORITY
#define SCAN_PRIORITY 10
#endif
#ifdef SCAN_DELEGATE
// Immutable after init, which runs under the provider lifecycle transaction.
static guild_operation_api_v1_t operation_api;
static guild_device_capabilities_v1_t device_caps;
#endif

static guild_status_t validate(const guild_device_capabilities_v1_t* caps,
    const guild_tensor_desc_t* in, size_t ni, const guild_tensor_desc_t* out, size_t no, guild_error_info_t* err) {
    (void)caps; (void)err;
    if (ni != 1 || no != 1 || !in || !out || in->dtype != GUILD_DTYPE_F32 || out->dtype != GUILD_DTYPE_F32 ||
        in->n_dims != 1 || out->n_dims != 1 || in->dims[0] != out->dims[0]) return GUILD_STATUS_ERR_INVALID_TENSOR;
    return GUILD_STATUS_OK;
}
static guild_status_t scratch(const guild_tensor_desc_t* in, size_t ni,
    const guild_tensor_desc_t* out, size_t no, guild_scratch_request_t* result, guild_error_info_t* err) {
    (void)in; (void)ni; (void)out; (void)no; (void)err;
    memset(result, 0, sizeof(*result)); result->struct_size = sizeof(*result);
    // A real, bounded scratch requirement to demonstrate budget negotiation.
    result->pageable_host_bytes = sizeof(float);
#ifdef SCAN_CUDA
    result->pageable_host_bytes = 0;
#endif
    return GUILD_STATUS_OK;
}
static guild_status_t execute_scan(const guild_exec_context_v1_t* ctx,
    const guild_tensor_desc_t* in, size_t ni, const guild_tensor_desc_t* out, size_t no,
    const void* params, size_t size, guild_error_info_t* err, int exclusive) {
    (void)params;
#if defined(SCAN_DELEGATE)
    guild_operation_query_v1_t query = {0};
    guild_operation_binding_t binding = 0;
    guild_status_t status;
    query.struct_size = sizeof(query); query.version = 1;
    query.operation_id = exclusive ? "example.sequence.exclusive_sum" : "example.sequence.prefix_sum";
    query.min_version = query.max_version = GUILD_MAKE_ABI_VERSION(1,0,0);
    query.allowed_backends = GUILD_BACKEND_CPU; query.required_capabilities = 1;
    query.sync_mode = GUILD_SYNC_HOST_BLOCKING; query.available_ram_bytes = UINT64_MAX;
    query.exact_impl_id = exclusive ? "example.scan.exclusive_sum" : "example.scan.prefix_sum";
    query.inputs = in; query.n_inputs = ni; query.outputs = out; query.n_outputs = no;
    status = operation_api.resolve(operation_api.user, &query, &device_caps, &binding, err);
    if (status) return status;
    status = operation_api.execute(binding, ctx, params, size, err);
    operation_api.release(binding);
    return status;
#elif defined(SCAN_CUDA)
    if (size || !ctx || validate(0, in, ni, out, no, err)) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    scan_kernel<<<1,1,0,(cudaStream_t)ctx->device_stream>>>((const float*)in->buffer.data,
        (float*)out->buffer.data, (size_t)in->dims[0], exclusive);
    return cudaGetLastError() == cudaSuccess ? GUILD_STATUS_OK : GUILD_STATUS_ERR_EXECUTION_FAILED;
#else
    size_t i; float sum = 0;
    if (size || !ctx || !ctx->host_scratch.data || validate(0, in, ni, out, no, err)) return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    for (i = 0; i < (size_t)in->dims[0]; ++i) {
        const float next = sum + ((const float*)in->buffer.data)[i];
        ((float*)out->buffer.data)[i] = exclusive ? sum : next;
        sum = next;
    }
    *(float*)ctx->host_scratch.data = sum;
    return GUILD_STATUS_OK;
#endif
}
static guild_status_t execute(const guild_exec_context_v1_t* ctx,
    const guild_tensor_desc_t* in, size_t ni, const guild_tensor_desc_t* out, size_t no,
    const void* params, size_t size, guild_error_info_t* err) {
    return execute_scan(ctx, in, ni, out, no, params, size, err, 0);
}
static guild_status_t execute_exclusive(const guild_exec_context_v1_t* ctx,
    const guild_tensor_desc_t* in, size_t ni, const guild_tensor_desc_t* out, size_t no,
    const void* params, size_t size, guild_error_info_t* err) {
    return execute_scan(ctx, in, ni, out, no, params, size, err, 1);
}
static guild_status_t init(guild_core_host_t host, const guild_host_api_v1_t* api, guild_error_info_t* err) {
    guild_kernel_op_desc_v1_t op = {0};
    guild_operation_ext_v1_t ext = {0};
    if (!api || api->struct_size < sizeof(*api) || !api->operations ||
        api->operations->struct_size < sizeof(*api->operations) || api->operations->version != 1)
        return GUILD_STATUS_ERR_ABI_MISMATCH;
#ifdef SCAN_DELEGATE
    // This fixture supports one owning registry per loaded library instance.
    // Reinitialization must not overwrite a service used by active executions.
    if (operation_api.version) return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
    operation_api = *api->operations;
    if (api->get_device_capabilities(host, &device_caps)) return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
#endif
    ext.struct_size = sizeof(ext); ext.tag = GUILD_OPERATION_TAG_V1; ext.version = 1;
    ext.operation_version = GUILD_MAKE_ABI_VERSION(1,0,0); ext.capabilities = 1;
    ext.layouts = GUILD_LAYOUT_CONTIGUOUS;
    ext.memory_spaces = GUILD_MEMORY_BIT(GUILD_MEM_HOST_PAGEABLE) | GUILD_MEMORY_BIT(GUILD_MEM_HOST_PINNED);
    ext.sync_modes = GUILD_SYNC_BIT(GUILD_SYNC_HOST_BLOCKING); ext.min_rank = ext.max_rank = 1; ext.alignment = sizeof(float);
    op.struct_size = sizeof(op); op.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    op.op_kind = GUILD_OP_CUSTOM; op.op_name = "example.sequence.prefix_sum";
    op.impl_id = SCAN_PACKAGE ".prefix_sum"; op.package_id = SCAN_PACKAGE;
    op.impl_version = GUILD_MAKE_ABI_VERSION(1,0,0); op.backend = GUILD_BACKEND_CPU;
    op.weight_dtype = GUILD_DTYPE_ANY; op.input_dtype = op.output_dtype = op.accum_dtype = GUILD_DTYPE_F32;
    op.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC;
    op.priority = SCAN_PRIORITY; op.next_ext = &ext;
    op.query_scratch = scratch; op.validate = validate; op.execute = execute;
#ifdef SCAN_CUDA
    op.backend = GUILD_BACKEND_CUDA;
    ext.memory_spaces = GUILD_MEMORY_BIT(GUILD_MEM_DEVICE_VRAM);
    ext.sync_modes = GUILD_SYNC_BIT(GUILD_SYNC_HOST_BLOCKING) | GUILD_SYNC_BIT(GUILD_SYNC_STREAM_ORDERED) |
        GUILD_SYNC_BIT(GUILD_SYNC_GRAPH_CAPTURE);
#endif
#ifdef SCAN_BAD_ABI
    ext.version = 99;
#endif
    // Both descriptors are on the stack: the host must copy them before return.
    {
        guild_status_t status = api->register_kernel_op(host, &op, err);
        if (status) return status;
#ifdef SCAN_DUPLICATE
        return api->register_kernel_op(host, &op, err);
#else
        op.op_name = "example.sequence.exclusive_sum";
        op.impl_id = SCAN_PACKAGE ".exclusive_sum";
        op.execute = execute_exclusive;
        return api->register_kernel_op(host, &op, err);
#endif
    }
}
#ifdef __cplusplus
extern "C"
#endif
GUILD_PLUGIN_EXPORT guild_status_t guild_plugin_entry_v1(uint32_t abi, guild_plugin_descriptor_v1_t* out, guild_error_info_t* err) {
    (void)err;
    if (!out || GUILD_ABI_VERSION_MAJOR(abi) != 1) return GUILD_STATUS_ERR_ABI_MISMATCH;
    memset(out, 0, sizeof(*out)); out->struct_size = sizeof(*out); out->abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    out->package_id = SCAN_PACKAGE; out->package_version = SCAN_VERSION; out->publisher_id = "org.illatio.guild";
    out->init = init;
    return GUILD_STATUS_OK;
}
