// Namespaced operations use the existing registry and KernelBinding ownership.
#include "guild/kernels/registry.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include "operation_internal.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <set>
#include <string_view>
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
#include <cuda_runtime.h>
#endif

struct guild_operation_binding_s {
    guild::kernels::KernelBinding binding;
    guild_operation_query_v1_t query{};
    guild_device_capabilities_v1_t caps{};
    std::string name, exact;
    std::vector<guild_tensor_desc_t> inputs, outputs;
    std::mutex mutex;
    std::set<std::pair<int, void*>> streams;
    ~guild_operation_binding_s() {
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
        if (streams.empty())
            return;
        int original = 0;
        cudaGetDevice(&original);
        for (const auto& stream : streams) {
            cudaSetDevice(stream.first);
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream.second));
        }
        cudaSetDevice(original);
#endif
    }
};

namespace guild::kernels {
namespace {
guild_status_t error(guild_error_info_t* e, guild_status_t st, std::string_view text) noexcept {
    if (e) {
        e->struct_size = sizeof(*e);
        e->code = st;
        e->source_component = "operation";
        const size_t size = std::min(text.size(), sizeof(e->message) - 1);
        std::memcpy(e->message, text.data(), size);
        e->message[size] = '\0';
    }
    return st;
}
bool power_of_two(uint64_t x) {
    return x && !(x & (x - 1));
}
bool operation_id(const char* id) {
    if (!id)
        return false;
    bool dot = false;
    size_t n = 0;
    for (; n < 128 && id[n]; ++n) {
        const char c = id[n];
        if (c == '.')
            dot = true;
        else if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-'))
            return false;
    }
    return dot && n > 2 && n < 128 && id[0] != '.' && id[n - 1] != '.';
}
} // namespace

guild_status_t validate_operation_query(const guild_operation_query_v1_t& q, std::string& err) {
    if (q.struct_size < sizeof(q) || q.version != 1) {
        err = "incompatible operation query ABI";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (!operation_id(q.operation_id) || !GUILD_ABI_VERSION_MAJOR(q.min_version) || q.max_version < q.min_version ||
        GUILD_ABI_VERSION_MAJOR(q.min_version) != GUILD_ABI_VERSION_MAJOR(q.max_version) || !q.allowed_backends ||
        (q.allowed_backends & ~7u) || q.sync_mode > GUILD_SYNC_GRAPH_CAPTURE || !q.n_inputs || !q.n_outputs ||
        q.n_inputs > 64 || q.n_outputs > 64 || !q.inputs || !q.outputs) {
        err = "invalid operation identity, version range, backend, sync mode or tensor arrays";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    for (size_t group = 0; group < 2; ++group) {
        const auto* tensors = group ? q.outputs : q.inputs;
        const auto count = group ? q.n_outputs : q.n_inputs;
        for (size_t i = 0; i < count; ++i) {
            guild_error_info_t e{};
            const auto st = validate_tensor_desc(tensors[i], &e);
            if (st != GUILD_STATUS_OK) {
                err = e.message;
                return st;
            }
        }
    }
    return GUILD_STATUS_OK;
}

guild_status_t copy_operation_extension(const guild_kernel_op_desc_v1_t& d, guild_operation_ext_v1_t& ext,
                                        std::string& err) {
    uint32_t size = 0;
    if (d.next_ext)
        std::memcpy(&size, d.next_ext, sizeof(size));
    if (size < sizeof(ext)) {
        err = "missing or truncated GOP1 descriptor";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    std::memcpy(&ext, d.next_ext, sizeof(ext));
    if (ext.tag != GUILD_OPERATION_TAG_V1 || ext.version != 1) {
        err = "unsupported GOP1 descriptor version";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    // Only this prefix is retained. Never advertise uncopied future tail bytes.
    ext.struct_size = sizeof(ext);
    if (!operation_id(d.op_name) || !GUILD_ABI_VERSION_MAJOR(ext.operation_version) || !ext.layouts ||
        (ext.layouts & ~3u) || !ext.memory_spaces || (ext.memory_spaces & ~15u) || !ext.sync_modes ||
        (ext.sync_modes & ~15u) || !ext.min_rank || ext.max_rank > GUILD_MAX_TENSOR_DIMS ||
        ext.min_rank > ext.max_rank || !power_of_two(ext.alignment) || !d.execute || !d.validate || !d.query_scratch ||
        d.direct_fn_table ||
        (d.backend != GUILD_BACKEND_CPU && d.backend != GUILD_BACKEND_CUDA && d.backend != GUILD_BACKEND_HIP) ||
        (d.backend == GUILD_BACKEND_CPU && ext.sync_modes != GUILD_SYNC_BIT(GUILD_SYNC_HOST_BLOCKING)) ||
        d.min_dim_n < 0 || d.max_dim_n < 0 || d.min_dim_k < 0 || d.max_dim_k < 0) {
        err = "invalid GOP1 capabilities, callbacks, shape bounds or execution contract";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    return GUILD_STATUS_OK;
}

guild_status_t match_operation(const guild_kernel_op_desc_v1_t& d, const guild_operation_query_v1_t& q,
                               const guild_device_capabilities_v1_t& caps, guild_scratch_request_t& scratch,
                               std::string& err) {
    guild_operation_ext_v1_t ext{};
    auto st = copy_operation_extension(d, ext, err);
    if (st != GUILD_STATUS_OK)
        return st;
    auto fail = [&](guild_status_t code, const char* why) {
        err = why;
        return code;
    };
    if (!(d.backend & q.allowed_backends & caps.available_backends) ||
        (d.required_cpu_isa & caps.cpu_isa_flags) != d.required_cpu_isa ||
        (d.required_gpu_caps & caps.gpu_cap_flags) != d.required_gpu_caps ||
        (d.min_sm_version && caps.gpu_sm_version < d.min_sm_version) ||
        (d.max_sm_version && caps.gpu_sm_version > d.max_sm_version))
        return fail(GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE, "backend/ISA/device architecture mismatch");
    if (ext.operation_version < q.min_version || ext.operation_version > q.max_version ||
        (ext.capabilities & q.required_capabilities) != q.required_capabilities ||
        (d.numerical_flags & q.numerical_flags) != q.numerical_flags || !(ext.sync_modes & GUILD_SYNC_BIT(q.sync_mode)))
        return fail(GUILD_STATUS_ERR_CAPABILITY_MISMATCH,
                    "operation version, capabilities, numerical or sync mismatch");
    for (size_t group = 0; group < 2; ++group) {
        const auto* tensors = group ? q.outputs : q.inputs;
        const auto count = group ? q.n_outputs : q.n_inputs;
        const auto dtype = group ? d.output_dtype : d.input_dtype;
        for (size_t i = 0; i < count; ++i) {
            const auto& t = tensors[i];
            if ((dtype != GUILD_DTYPE_ANY && t.dtype != dtype) || uint32_t(t.n_dims) < ext.min_rank ||
                uint32_t(t.n_dims) > ext.max_rank || t.buffer.memory_space > GUILD_MEM_DEVICE_VRAM ||
                !(ext.memory_spaces & GUILD_MEMORY_BIT(t.buffer.memory_space)) ||
                reinterpret_cast<uintptr_t>(t.buffer.data) % ext.alignment)
                return fail(GUILD_STATUS_ERR_CAPABILITY_MISMATCH,
                            "tensor dtype, rank, memory space or alignment mismatch");
            int elems = 1, bytes = 4;
            if (t.dtype != GUILD_DTYPE_I32 && !block_geometry(uint32_t(t.dtype), elems, bytes))
                return fail(GUILD_STATUS_ERR_INVALID_TENSOR, "unknown dtype geometry");
            uint64_t stride = uint64_t(bytes);
            bool contiguous = !t.strides_bytes[0] || t.strides_bytes[0] == stride;
            stride *= uint64_t(t.dims[0] / elems);
            for (int axis = 1; axis < t.n_dims; ++axis) {
                contiguous &= !t.strides_bytes[axis] || t.strides_bytes[axis] == stride;
                stride *= uint64_t(t.dims[axis]);
            }
            if (!(ext.layouts & (contiguous ? GUILD_LAYOUT_CONTIGUOUS : GUILD_LAYOUT_STRIDED)))
                return fail(GUILD_STATUS_ERR_CAPABILITY_MISMATCH, "tensor layout mismatch");
            if ((d.min_dim_n && t.dims[0] < d.min_dim_n) || (d.max_dim_n && t.dims[0] > d.max_dim_n) ||
                (t.n_dims > 1 &&
                 ((d.min_dim_k && t.dims[1] < d.min_dim_k) || (d.max_dim_k && t.dims[1] > d.max_dim_k))) ||
                (d.max_batch_rows && (t.n_dims > 1 ? t.dims[1] : 1) > d.max_batch_rows))
                return fail(GUILD_STATUS_ERR_CAPABILITY_MISMATCH, "tensor dimension bounds mismatch");
            if (q.sync_mode != GUILD_SYNC_HOST_BLOCKING && t.buffer.lifetime < GUILD_LIFETIME_STREAM_ORDERED)
                return fail(GUILD_STATUS_ERR_BUFFER_LIFETIME, "async operation requires stream-ordered buffers");
        }
    }
    guild_error_info_t e{};
    st = d.validate(&caps, q.inputs, q.n_inputs, q.outputs, q.n_outputs, &e);
    if (st != GUILD_STATUS_OK) {
        err = e.message[0] ? e.message : "provider rejected tensor contract";
        return st;
    }
    scratch = {};
    scratch.struct_size = sizeof(scratch);
    st = d.query_scratch(q.inputs, q.n_inputs, q.outputs, q.n_outputs, &scratch, &e);
    if (st != GUILD_STATUS_OK) {
        err = e.message[0] ? e.message : "provider scratch query failed";
        return st;
    }
    if (scratch.struct_size < sizeof(scratch) || (scratch.device_bytes && !power_of_two(scratch.device_alignment)) ||
        (scratch.pinned_host_bytes && !power_of_two(scratch.pinned_host_alignment)))
        return fail(GUILD_STATUS_ERR_INVALID_ARGUMENT, "invalid scratch descriptor");
    const auto ram = std::min(q.available_ram_bytes, caps.available_ram_bytes);
    if (scratch.device_bytes > std::min(q.available_vram_bytes, caps.available_vram_bytes) ||
        scratch.pinned_host_bytes > ram || scratch.pageable_host_bytes > ram - scratch.pinned_host_bytes)
        return fail(GUILD_STATUS_ERR_OUT_OF_MEMORY, "scratch exceeds RAM/VRAM budget");
    return GUILD_STATUS_OK;
}

guild_operation_api_v1_t KernelRegistry::operation_api() {
    guild_operation_api_v1_t api{};
    api.struct_size = sizeof(api);
    api.version = 1;
    api.user = this;
    api.resolve = [](void* user, const guild_operation_query_v1_t* q, const guild_device_capabilities_v1_t* caps,
                     guild_operation_binding_t* out, guild_error_info_t* e) -> guild_status_t {
        if (out)
            *out = nullptr;
        if (!user || !q || !caps || !out)
            return error(e, GUILD_STATUS_ERR_INVALID_ARGUMENT, "null resolve argument");
        try {
            std::string err;
            auto st = validate_operation_query(*q, err);
            if (st != GUILD_STATUS_OK)
                return error(e, st, err);
            if (caps->struct_size < sizeof(*caps) || caps->next_ext ||
                negotiate_abi_version(caps->abi_version, nullptr) != GUILD_STATUS_OK)
                return error(e, GUILD_STATUS_ERR_ABI_MISMATCH, "incompatible device capabilities");
            KernelSelectionQuery query;
            query.op_kind = GUILD_OP_CUSTOM;
            query.operation = q;
            query.allowed_backends = q->allowed_backends;
            if (q->exact_impl_id)
                query.frozen_impl_id = q->exact_impl_id;
            auto h = std::make_unique<guild_operation_binding_s>();
            st = static_cast<KernelRegistry*>(user)->select(query, *caps, &h->binding, err);
            if (st != GUILD_STATUS_OK)
                return error(e, st, std::string(q->operation_id) + ": " + err);
            h->query = *q;
            h->caps = *caps;
            h->name = q->operation_id;
            h->caps.struct_size = sizeof(h->caps);
            if (q->exact_impl_id)
                h->exact = q->exact_impl_id;
            h->inputs.assign(q->inputs, q->inputs + q->n_inputs);
            h->outputs.assign(q->outputs, q->outputs + q->n_outputs);
            for (auto* tensors : {&h->inputs, &h->outputs}) {
                for (auto& tensor : *tensors) {
                    tensor.struct_size = sizeof(tensor);
                    tensor.buffer.struct_size = sizeof(tensor.buffer);
                }
            }
            h->query.operation_id = h->name.c_str();
            h->query.exact_impl_id = h->exact.c_str();
            h->query.inputs = h->inputs.data();
            h->query.outputs = h->outputs.data();
            *out = h.release();
            return GUILD_STATUS_OK;
        } catch (...) {
            return error(e, GUILD_STATUS_ERR_INTERNAL, "operation resolve exception");
        }
    };
    api.describe = [](guild_operation_binding_t h) -> const guild_kernel_op_desc_v1_t* {
        return h ? h->binding.get() : nullptr;
    };
    api.scratch = [](guild_operation_binding_t h, guild_scratch_request_t* out,
                     guild_error_info_t* e) -> guild_status_t {
        if (!h || !out || out->struct_size < sizeof(*out))
            return error(e, GUILD_STATUS_ERR_INVALID_ARGUMENT, "invalid scratch query");
        try {
            std::lock_guard<std::mutex> lock(h->mutex);
            std::string err;
            const auto st = match_operation(*h->binding, h->query, h->caps, *out, err);
            return st == GUILD_STATUS_OK ? st : error(e, st, err);
        } catch (...) {
            return error(e, GUILD_STATUS_ERR_INTERNAL, "scratch query exception");
        }
    };
    api.execute = [](guild_operation_binding_t h, const guild_exec_context_v1_t* ctx, const void* params, size_t bytes,
                     guild_error_info_t* e) -> guild_status_t {
        if (!h || !ctx || (bytes && !params))
            return error(e, GUILD_STATUS_ERR_INVALID_ARGUMENT, "invalid execute argument");
        try {
            std::lock_guard<std::mutex> lock(h->mutex);
            if (ctx->struct_size < sizeof(*ctx) || negotiate_abi_version(ctx->abi_version, nullptr) != GUILD_STATUS_OK)
                return error(e, GUILD_STATUS_ERR_ABI_MISMATCH, "incompatible execution context");
            if (ctx->backend != h->binding->backend || ctx->sync_mode != h->query.sync_mode ||
                (ctx->backend == GUILD_BACKEND_CPU && ctx->device_stream))
                return error(e, GUILD_STATUS_ERR_CAPABILITY_MISMATCH,
                             "execution backend/stream/sync differs from binding");
            guild_scratch_request_t scratch{};
            std::string err;
            auto st = match_operation(*h->binding, h->query, h->caps, scratch, err);
            if (st != GUILD_STATUS_OK)
                return error(e, st, err);
            auto buffer_ok = [&](const guild_buffer_desc_t& b, uint64_t required, uint64_t alignment, bool device) {
                if (!required)
                    return true;
                if (validate_buffer_desc(b) != GUILD_STATUS_OK || b.byte_size < required ||
                    (alignment && reinterpret_cast<uintptr_t>(b.data) % alignment))
                    return false;
                if (ctx->sync_mode != GUILD_SYNC_HOST_BLOCKING && b.lifetime < GUILD_LIFETIME_STREAM_ORDERED)
                    return false;
                return device ? b.memory_space == GUILD_MEM_DEVICE_VRAM && b.device_id == ctx->device_id
                              : (scratch.pinned_host_bytes ? b.memory_space == GUILD_MEM_HOST_PINNED
                                                           : b.memory_space <= GUILD_MEM_HOST_MAPPED);
            };
            if (!buffer_ok(ctx->device_scratch, scratch.device_bytes, scratch.device_alignment, true) ||
                !buffer_ok(ctx->host_scratch, scratch.pinned_host_bytes + scratch.pageable_host_bytes,
                           scratch.pinned_host_alignment, false))
                return error(e, GUILD_STATUS_ERR_OUT_OF_MEMORY, "missing, misaligned or invalid scratch buffers");
            if (ctx->backend != GUILD_BACKEND_CPU) {
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
                int device = -1;
                if (cudaGetDevice(&device) != cudaSuccess || device != ctx->device_id)
                    return error(e, GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE, "caller must select context device");
                for (const auto* group : {&h->inputs, &h->outputs})
                    for (const auto& t : *group)
                        if (t.buffer.memory_space == GUILD_MEM_DEVICE_VRAM && t.buffer.device_id != device)
                            return error(e, GUILD_STATUS_ERR_INVALID_TENSOR, "tensor belongs to another device");
                // Record before launch, including failed launches that queued partial work.
                h->streams.emplace(device, ctx->device_stream);
#else
                return error(e, GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE, "GPU operations require a GPU-enabled Core");
#endif
            }
            st = h->binding->execute(ctx, h->inputs.data(), h->inputs.size(), h->outputs.data(), h->outputs.size(),
                                     params, bytes, e);
            if (e)
                e->source_component = "operation"; // never return a borrowed DSO string
#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
            if (ctx->backend != GUILD_BACKEND_CPU && ctx->sync_mode == GUILD_SYNC_HOST_BLOCKING &&
                cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->device_stream)) != cudaSuccess)
                return error(e, GUILD_STATUS_ERR_EXECUTION_FAILED, "device stream synchronization failed");
#endif
            return st;
        } catch (...) {
            return error(e, GUILD_STATUS_ERR_INTERNAL, "operation execute exception");
        }
    };
    api.release = [](guild_operation_binding_t h) { delete h; };
    return api;
}
} // namespace guild::kernels
