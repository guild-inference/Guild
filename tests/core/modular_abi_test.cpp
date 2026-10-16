// tests/core/modular_abi_test.cpp - Modular Runner, Kernel Registry, ABI v1 & .gkmetadata Security Tests
#include "../check.hpp"
#include "guild/abi/gkernel_package.hpp"
#include "guild/abi/kernel_abi.h"
#include "guild/kernels/registry.hpp"
#include "guild/models/sha256.hpp"
#include "guild/model/runner.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace fs = std::filesystem;
using namespace guild;

namespace {

int g_plugin_shutdown_calls = 0;

guild_status_t test_plugin_init(guild_core_host_t host, const guild_host_api_v1_t* api, guild_error_info_t* err) {
    if (!host || !api || api->struct_size < sizeof(guild_host_api_v1_t)) return GUILD_STATUS_ERR_ABI_MISMATCH;
    uint32_t negotiated = 0;
    if (api->negotiate_abi(GUILD_KERNEL_ABI_VERSION_V1, &negotiated) != GUILD_STATUS_OK) return GUILD_STATUS_ERR_ABI_MISMATCH;
    guild_device_capabilities_v1_t caps{};
    if (api->get_device_capabilities(host, &caps) != GUILD_STATUS_OK) return GUILD_STATUS_ERR_INTERNAL;

    guild_kernel_op_desc_v1_t op{};
    op.struct_size = sizeof(op);
    op.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    op.op_kind = GUILD_OP_DEQUANTIZE;
    op.op_name = "dequantize";
    op.impl_id = "plugin.cpu.dequant.q4_0";
    op.package_id = "ext.guild-kernels-test";
    op.impl_version = GUILD_KERNEL_ABI_VERSION_V1;
    op.backend = GUILD_BACKEND_CPU;
    op.weight_dtype = GUILD_DTYPE_Q4_0;
    op.input_dtype = GUILD_DTYPE_Q4_0;
    op.output_dtype = GUILD_DTYPE_F32;
    op.accum_dtype = GUILD_DTYPE_F32;
    op.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC;
    op.supported_sync_modes = GUILD_SYNC_HOST_BLOCKING;
    op.priority = 150;
    return api->register_kernel_op(host, &op, err);
}

void test_plugin_shutdown(guild_core_host_t) {
    ++g_plugin_shutdown_calls;
}

guild_status_t valid_plugin_entry(uint32_t host_abi, guild_plugin_descriptor_v1_t* out_desc, guild_error_info_t*) {
    if (GUILD_KERNEL_ABI_VERSION_MAJOR_OF(host_abi) != 1) return GUILD_STATUS_ERR_ABI_MISMATCH;
    out_desc->struct_size = sizeof(*out_desc);
    out_desc->abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    out_desc->package_id = "ext.guild-kernels-test";
    out_desc->package_version = "1.0.0";
    out_desc->publisher_id = "org.illatio.guild";
    out_desc->init = &test_plugin_init;
    out_desc->shutdown = &test_plugin_shutdown;
    return GUILD_STATUS_OK;
}

guild_status_t bad_abi_plugin_entry(uint32_t, guild_plugin_descriptor_v1_t* out_desc, guild_error_info_t*) {
    out_desc->struct_size = sizeof(*out_desc);
    out_desc->abi_version = GUILD_KERNEL_ABI_MAKE_VERSION(2, 0, 0);
    out_desc->package_id = "ext.bad-abi";
    out_desc->init = &test_plugin_init;
    return GUILD_STATUS_OK;
}

// Synthetic CPU descriptors exercise prefill binding without a GPU or kernel launches.
// Allocate exactly the old seven-pointer payload so an ASan run catches any larger read.
static_assert(sizeof(kernels::PrefillKernelTableV1) ==
              7 * sizeof(decltype(kernels::PrefillKernelTableV1::quantize_q8_1)),
              "the legacy prefill payload must remain seven function pointers");

constexpr guild_op_kind_t kPrefillTestOps[] = {
    GUILD_OP_QUANTIZE_ACT, GUILD_OP_QUANTIZED_GEMM, GUILD_OP_RMS_NORM,
    GUILD_OP_GDN_CONV_L2, GUILD_OP_GDN_AB_PROJ, GUILD_OP_GDN_RECURRENCE_STEP,
    GUILD_OP_GDN_OUT_NORM, GUILD_OP_ROPE_APPLY, GUILD_OP_KV_APPEND,
    GUILD_OP_ATTENTION_DECODE, GUILD_OP_MOE_ROUTER_TOPK, GUILD_OP_SHARED_EXPERT,
    GUILD_OP_BATCHED_EXPERT_EXEC, GUILD_OP_MOE_COMBINE, GUILD_OP_FINITE_CHECK
};

const char* prefill_test_id(guild_op_kind_t op) {
    // Also satisfy the explicitly frozen serial policy when it is set in the environment.
    return op == GUILD_OP_GDN_AB_PROJ ? "gpu.gdn.ab_proj_prefill_serial" : kernels::op_kind_to_string(op);
}

struct PrefillTestRegistry {
    std::unique_ptr<kernels::PrefillKernelTableV1> legacy = std::make_unique<kernels::PrefillKernelTableV1>();
    kernels::PrefillKernelExtensionV1 extension;
    guild_device_capabilities_v1_t caps{};
    kernels::KernelRegistry reg;

    PrefillTestRegistry() {
        legacy->quantize_q8_1 = +[](const float*, void*, int, int, void*) {};
        legacy->mmvq = +[](int, const void*, const void*, float*, int, int, int, void*) {};
        legacy->rms_norm_weighted = +[](const float*, const float*, float*, int, int, float, void*) {};
        legacy->gdn_conv = +[](const float*, const float*, const float*, float*, int, int, float, int, void*) {};
        legacy->gdn_recurrence = +[](float*, const float*, int, const float*, const float*, const float*,
                                    const float*, float, float*, int, int, int, const int32_t*, bool, void*) {};
        legacy->gdn_out_norm = +[](const float*, const float*, const float*, float*, int64_t, int64_t, float, void*) {};
        legacy->rope_apply = +[](const float*, float*, int, int, int, const kernels::RopeScaling&, const int*, void*) {};
        extension.gdn_ab = +[](const float*, const uint16_t*, const uint16_t*, const float*, const float*,
                               float*, float*, int, int, int, void*) {};
        extension.router_projection = +[](const float*, const uint16_t*, float*, int64_t, int64_t, int, void*) {};
        caps.struct_size = sizeof(caps);
        caps.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        caps.available_backends = GUILD_BACKEND_CPU;
        caps.available_ram_bytes = 1ull << 30;
        std::string err;
        for (auto op : kPrefillTestOps) CHECK(reg.register_op(descriptor(op), err) == GUILD_STATUS_OK);
    }

    guild_kernel_op_desc_v1_t descriptor(guild_op_kind_t op) const {
        guild_kernel_op_desc_v1_t desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        desc.op_kind = op;
        desc.op_name = kernels::op_kind_to_string(op);
        desc.impl_id = prefill_test_id(op);
        desc.package_id = desc.impl_id;
        desc.impl_version = GUILD_KERNEL_ABI_VERSION_V1;
        desc.backend = GUILD_BACKEND_CPU;
        desc.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC;
        desc.max_batch_rows = op == GUILD_OP_ATTENTION_DECODE ? 8 : 512;
        desc.direct_fn_table = legacy.get();
        if (op == GUILD_OP_GDN_AB_PROJ || op == GUILD_OP_MOE_ROUTER_TOPK) desc.next_ext = &extension;
        desc.priority = 100;
        return desc;
    }

    void replace(guild_op_kind_t op, const void* table, const void* next_ext, int64_t rows = 512) {
        std::string err;
        CHECK(reg.unregister_package(prefill_test_id(op), err) == GUILD_STATUS_OK);
        auto desc = descriptor(op);
        desc.direct_fn_table = table;
        desc.next_ext = next_ext;
        desc.max_batch_rows = rows;
        CHECK(reg.register_op(desc, err) == GUILD_STATUS_OK);
    }
};

} // namespace

int main() {
    std::string err;

    // -------------------------------------------------------------------------
    // 1. ABI Version Negotiation
    // -------------------------------------------------------------------------
    {
        uint32_t negotiated = 0;
        CHECK(kernels::negotiate_abi_version(GUILD_KERNEL_ABI_VERSION_V1, &negotiated) == GUILD_STATUS_OK);
        CHECK(negotiated == GUILD_KERNEL_ABI_VERSION_V1);
        CHECK(kernels::negotiate_abi_version(GUILD_KERNEL_ABI_MAKE_VERSION(2, 0, 0), &negotiated) == GUILD_STATUS_ERR_ABI_MISMATCH);
        CHECK(kernels::negotiate_abi_version(GUILD_KERNEL_ABI_MAKE_VERSION(0, 9, 0), &negotiated) == GUILD_STATUS_ERR_ABI_MISMATCH);
        CHECK(kernels::negotiate_abi_version(GUILD_KERNEL_ABI_MAKE_VERSION(1, 5, 0), &negotiated) == GUILD_STATUS_ERR_ABI_MISMATCH);
    }

    // -------------------------------------------------------------------------
    // 2. Buffer Lifetime & Tensor Descriptor Validation
    // -------------------------------------------------------------------------
    {
        alignas(64) float data[64] = {};
        guild_buffer_desc_t buf{};
        buf.struct_size = sizeof(buf);
        buf.data = data;
        buf.byte_size = sizeof(data);
        buf.alignment = 32;
        buf.memory_space = GUILD_MEM_HOST_PAGEABLE;
        buf.ownership = GUILD_OWNERSHIP_CORE_BORROWED;
        buf.lifetime = GUILD_LIFETIME_CALL_SYNC;
        CHECK(kernels::validate_buffer_desc(buf) == GUILD_STATUS_OK);

        // Expired buffer lifetime
        guild_buffer_desc_t expired = buf;
        expired.lifetime = GUILD_LIFETIME_EXPIRED;
        CHECK(kernels::validate_buffer_desc(expired) == GUILD_STATUS_ERR_BUFFER_LIFETIME);

        // Invalid ownership
        guild_buffer_desc_t bad_own = buf;
        bad_own.ownership = static_cast<guild_buffer_ownership_t>(99);
        CHECK(kernels::validate_buffer_desc(bad_own) == GUILD_STATUS_ERR_BUFFER_LIFETIME);

        // Truncated struct_size
        guild_buffer_desc_t bad_struct = buf;
        bad_struct.struct_size = 8;
        CHECK(kernels::validate_buffer_desc(bad_struct) == GUILD_STATUS_ERR_ABI_MISMATCH);

        // Valid 2D F32 tensor
        guild_tensor_desc_t tensor{};
        tensor.struct_size = sizeof(tensor);
        tensor.dtype = GUILD_DTYPE_F32;
        tensor.n_dims = 2;
        tensor.dims[0] = 32;
        tensor.dims[1] = 2;
        tensor.strides_bytes[0] = 4;
        tensor.strides_bytes[1] = 128;
        tensor.buffer = buf;
        CHECK(kernels::validate_tensor_desc(tensor) == GUILD_STATUS_OK);

        // Invalid rank
        guild_tensor_desc_t bad_rank = tensor;
        bad_rank.n_dims = 0;
        CHECK(kernels::validate_tensor_desc(bad_rank) == GUILD_STATUS_ERR_INVALID_TENSOR);

        // Unaligned quantized block dimension (Q4_0 requires multiple of 32)
        guild_tensor_desc_t bad_block = tensor;
        bad_block.dtype = GUILD_DTYPE_Q4_0;
        bad_block.dims[0] = 30;
        CHECK(kernels::validate_tensor_desc(bad_block) == GUILD_STATUS_ERR_INVALID_TENSOR);

        // Stride smaller than contiguous row
        guild_tensor_desc_t bad_stride = tensor;
        bad_stride.strides_bytes[1] = 64; // < 32 * 4 = 128
        CHECK(kernels::validate_tensor_desc(bad_stride) == GUILD_STATUS_ERR_INVALID_TENSOR);

        // Backing buffer smaller than tensor shape
        guild_tensor_desc_t small_buf = tensor;
        small_buf.dims[1] = 4; // requires 512 bytes, buffer has 256
        CHECK(kernels::validate_tensor_desc(small_buf) == GUILD_STATUS_ERR_INVALID_TENSOR);
    }

    // -------------------------------------------------------------------------
    // 3. Built-in & Custom Model Runner Registration, Missing Runner, Isolation
    // -------------------------------------------------------------------------
    {
        auto& builtin_runners = model::ModelRunnerRegistry::instance();
        const model::IModelRunner* qwen_runner = nullptr;
        const model::IModelRunner* ornith_runner = nullptr;
        CHECK(builtin_runners.resolve_by_arch("qwen4exp", &qwen_runner, err) == GUILD_STATUS_OK && qwen_runner != nullptr);
        CHECK(builtin_runners.resolve_by_arch("qwen35moe", &ornith_runner, err) == GUILD_STATUS_OK && ornith_runner != nullptr);
        CHECK(builtin_runners.resolve_by_arch("ornith", &ornith_runner, err) == GUILD_STATUS_OK && ornith_runner != nullptr);

        // Missing runner for unregistered architecture
        const model::IModelRunner* missing = nullptr;
        CHECK(builtin_runners.resolve_by_arch("nonexistent_arch", &missing, err) == GUILD_STATUS_ERR_MISSING_RUNNER);
        CHECK(missing == nullptr);

        // Isolated runner registry for registration/duplicate/cleanup tests
        model::ModelRunnerRegistry reg;
        reg.register_builtin_runners();

        // Duplicate runner registration rejected
        const guild_op_kind_t glm_ops[] = {GUILD_OP_QUANTIZED_GEMV, GUILD_OP_BATCHED_EXPERT_EXEC};
        guild_runner_desc_v1_t glm_desc{};
        glm_desc.struct_size = sizeof(glm_desc);
        glm_desc.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        glm_desc.arch_id = "glm";
        glm_desc.runner_id = "runner.glm.v1";
        glm_desc.package_id = "ext.guild-kernels-glm";
        glm_desc.runner_version = GUILD_KERNEL_ABI_VERSION_V1;
        glm_desc.required_core_caps = GUILD_CORE_CAP_HOST_ONLY_KV;
        glm_desc.required_op_kinds = glm_ops;
        glm_desc.n_required_op_kinds = 2;

        CHECK(reg.register_abi_runner(glm_desc, err) == GUILD_STATUS_OK);
        CHECK(reg.register_abi_runner(glm_desc, err) == GUILD_STATUS_ERR_DUPLICATE_REGISTRATION);

        // ABI mismatch on runner registration
        guild_runner_desc_v1_t bad_abi_runner = glm_desc;
        bad_abi_runner.runner_id = "runner.glm.v2";
        bad_abi_runner.arch_id = "glm_v2";
        bad_abi_runner.abi_version = GUILD_KERNEL_ABI_MAKE_VERSION(3, 0, 0);
        CHECK(reg.register_abi_runner(bad_abi_runner, err) == GUILD_STATUS_ERR_ABI_MISMATCH);

        // Active session lease prevents runner unregistration
        {
            model::SessionRunnerLease lease(&reg);
            CHECK(reg.active_sessions() == 1);
            CHECK(reg.unregister_runner("runner.glm.v1", err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
        }
        CHECK(reg.active_sessions() == 0);
        CHECK(reg.unregister_runner("runner.glm.v1", err) == GUILD_STATUS_OK);
        CHECK(reg.find_by_id("runner.glm.v1") == nullptr);
    }

    // -------------------------------------------------------------------------
    // 4. Kernel Operation Registry: Selection, Missing Kernel, Hardware,
    //    Capability, Scratch OOM, Dependencies, Dispatch Failure & Leases
    // -------------------------------------------------------------------------
    {
        g_plugin_shutdown_calls = 0;
        {
            kernels::KernelRegistry kreg;
            kreg.register_builtin_kernels();

            guild_device_capabilities_v1_t caps{};
            caps.struct_size = sizeof(caps);
            caps.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
            caps.available_backends = GUILD_BACKEND_CPU;
            caps.cpu_isa_flags = GUILD_ISA_SSE42 | GUILD_ISA_AVX | GUILD_ISA_AVX2 | GUILD_ISA_FMA;
            caps.core_cap_flags = GUILD_CORE_CAP_HOST_ONLY_KV;
            caps.available_ram_bytes = 1ull << 30;
            caps.available_vram_bytes = 0;

            // Select built-in CPU expert kernel
            kernels::KernelSelectionQuery q;
            q.op_kind = GUILD_OP_BATCHED_EXPERT_EXEC;
            q.allowed_backends = GUILD_BACKEND_CPU;
            q.batch_rows = 8;
            q.dim_n = 2048;
            q.dim_k = 512;
            const guild_kernel_op_desc_v1_t* selected = nullptr;
            CHECK(kreg.select(q, caps, &selected, err) == GUILD_STATUS_OK && selected != nullptr);
            CHECK(std::strcmp(selected->impl_id, "cpu.expert_pool.split_multi_native") == 0);

            // Unsupported hardware (request Q2_0 canonical kernel requiring AVX-512 on an AVX2-only cap struct)
            kernels::KernelSelectionQuery q_avx512 = q;
            q_avx512.frozen_impl_id = "cpu.expert_pool.split_multi_canonical_avx512";
            CHECK(kreg.select(q_avx512, caps, &selected, err) == GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE);

            // Capability mismatch (batch_rows exceeds max_batch_rows = 512)
            kernels::KernelSelectionQuery q_big_batch = q;
            q_big_batch.batch_rows = 4096;
            CHECK(kreg.select(q_big_batch, caps, &selected, err) == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);

            // Missing kernel (unregistered frozen_impl_id)
            kernels::KernelSelectionQuery q_missing = q;
            q_missing.frozen_impl_id = "cpu.nonexistent.kernel";
            CHECK(kreg.select(q_missing, caps, &selected, err) == GUILD_STATUS_ERR_MISSING_KERNEL);

            // Plugin registration & duplicate rejection
            CHECK(kreg.register_plugin(&bad_abi_plugin_entry, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
            CHECK(kreg.register_plugin(&valid_plugin_entry, err) == GUILD_STATUS_OK);
            CHECK(kreg.register_plugin(&valid_plugin_entry, err) == GUILD_STATUS_ERR_DUPLICATE_REGISTRATION);

            // Scratch memory OOM check
            guild_kernel_op_desc_v1_t huge_scratch_op{};
            huge_scratch_op.struct_size = sizeof(huge_scratch_op);
            huge_scratch_op.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
            huge_scratch_op.op_kind = GUILD_OP_ATTENTION_PREFILL;
            huge_scratch_op.op_name = "attention_prefill";
            huge_scratch_op.impl_id = "cpu.attn.huge_scratch";
            huge_scratch_op.package_id = "ext.huge";
            huge_scratch_op.impl_version = GUILD_KERNEL_ABI_VERSION_V1;
            huge_scratch_op.backend = GUILD_BACKEND_CPU;
            huge_scratch_op.weight_dtype = GUILD_DTYPE_ANY;
            huge_scratch_op.input_dtype = GUILD_DTYPE_F32;
            huge_scratch_op.output_dtype = GUILD_DTYPE_F32;
            huge_scratch_op.accum_dtype = GUILD_DTYPE_F32;
            huge_scratch_op.query_scratch = [](const guild_tensor_desc_t*, size_t, const guild_tensor_desc_t*, size_t,
                                               guild_scratch_request_t* out, guild_error_info_t*) -> guild_status_t {
                out->pageable_host_bytes = 64ull << 30; // 64 GiB > 1 GiB available_ram_bytes
                return GUILD_STATUS_OK;
            };
            CHECK(kreg.register_op(huge_scratch_op, err) == GUILD_STATUS_OK);
            kernels::KernelSelectionQuery q_oom;
            q_oom.op_kind = GUILD_OP_ATTENTION_PREFILL;
            q_oom.allowed_backends = GUILD_BACKEND_CPU;
            q_oom.frozen_impl_id = "cpu.attn.huge_scratch";
            CHECK(kreg.select(q_oom, caps, &selected, err) == GUILD_STATUS_ERR_OUT_OF_MEMORY);

            // Dependency validation failure when a kernel references a missing dependency
            const char* missing_deps[] = {"gpu.nonexistent.dependency"};
            guild_kernel_op_desc_v1_t dep_op = huge_scratch_op;
            dep_op.impl_id = "cpu.attn.with_missing_dep";
            dep_op.package_id = "ext.dep";
            dep_op.query_scratch = nullptr;
            dep_op.required_op_deps = missing_deps;
            dep_op.n_required_op_deps = 1;
            CHECK(kreg.register_op(dep_op, err) == GUILD_STATUS_OK);
            CHECK(kreg.validate_dependencies(err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);

            // Active session lease prevents unregistering package
            {
                kernels::SessionKernelLease klease(&kreg);
                CHECK(kreg.unregister_package("ext.dep", err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
            }
            CHECK(kreg.unregister_package("ext.dep", err) == GUILD_STATUS_OK);
            CHECK(kreg.validate_dependencies(err) == GUILD_STATUS_OK);

            // Kernel dispatch failure propagation (NaN detection via cpu.finite_check.simd64)
            const auto* finite_op = kreg.find_by_id("cpu.finite_check.simd64");
            CHECK(finite_op != nullptr && finite_op->execute != nullptr);
            alignas(64) float values[32] = {};
            values[17] = std::numeric_limits<float>::quiet_NaN();
            guild_tensor_desc_t t{};
            t.struct_size = sizeof(t);
            t.dtype = GUILD_DTYPE_F32;
            t.n_dims = 1;
            t.dims[0] = 32;
            t.buffer.struct_size = sizeof(t.buffer);
            t.buffer.data = values;
            t.buffer.byte_size = sizeof(values);
            t.buffer.alignment = 32;
            t.buffer.memory_space = GUILD_MEM_HOST_PAGEABLE;
            t.buffer.ownership = GUILD_OWNERSHIP_CORE_BORROWED;
            t.buffer.lifetime = GUILD_LIFETIME_CALL_SYNC;
            guild_error_info_t einfo{};
            CHECK(finite_op->execute(nullptr, &t, 1, nullptr, 0, nullptr, 0, &einfo) == GUILD_STATUS_ERR_NONFINITE_OUTPUT);
            CHECK(std::strstr(einfo.message, "17") != nullptr);
        }
        // Plugin shutdown callback executed on registry destruction
        CHECK(g_plugin_shutdown_calls == 1);
    }

    // Legacy descriptors can still override individual entries with an exact seven-pointer
    // allocation and no extension. Projection-only descriptors can supply just next_ext.
    // Later unrelated descriptors must preserve every independently selected pointer.
    {
        PrefillTestRegistry f;
        kernels::ResolvedPrefillKernels resolved;
        CHECK(f.reg.resolve_prefill_kernels(f.caps, resolved, err) && resolved.all_bound());

        auto legacy = std::make_unique<kernels::PrefillKernelTableV1>();
        legacy->rms_norm_weighted = +[](const float*, const float*, float*, int, int, float, void*) {};
        auto norm_desc = f.descriptor(GUILD_OP_RMS_NORM);
        norm_desc.impl_id = "test.prefill.norm_override";
        norm_desc.direct_fn_table = legacy.get();
        norm_desc.next_ext = nullptr;
        norm_desc.priority = 10000;
        CHECK(f.reg.register_op(norm_desc, err) == GUILD_STATUS_OK);

        kernels::PrefillKernelExtensionV1 gdn_extension;
        gdn_extension.gdn_ab = +[](const float*, const uint16_t*, const uint16_t*, const float*, const float*,
                                   float*, float*, int, int, int, void*) {};
        // Keep the serial implementation ID so this test also works with the frozen A/B policy.
        f.replace(GUILD_OP_GDN_AB_PROJ, nullptr, &gdn_extension);
        kernels::PrefillKernelExtensionV1 router_extension;
        router_extension.router_projection = +[](const float*, const uint16_t*, float*, int64_t, int64_t, int, void*) {};
        auto router_desc = f.descriptor(GUILD_OP_MOE_ROUTER_TOPK);
        router_desc.impl_id = "test.prefill.router_override";
        router_desc.direct_fn_table = nullptr;
        router_desc.next_ext = &router_extension;
        router_desc.priority = 10000;
        CHECK(f.reg.register_op(router_desc, err) == GUILD_STATUS_OK);

        // An operation checked only for availability can have its own small fast-path
        // payload; it must never be reinterpreted as a legacy prefill table.
        const uint32_t attention_payload = 0;
        auto attention_desc = f.descriptor(GUILD_OP_ATTENTION_DECODE);
        attention_desc.impl_id = "test.prefill.attention_payload";
        attention_desc.priority = 10000;
        attention_desc.direct_fn_table = &attention_payload;
        CHECK(f.reg.register_op(attention_desc, err) == GUILD_STATUS_OK);

        // Fused GR remains an eight-row decode operation, outside the prefill requirements.
        auto gr_desc = f.descriptor(GUILD_OP_GATED_RESIDUAL);
        gr_desc.max_batch_rows = 8;
        gr_desc.direct_fn_table = nullptr;
        CHECK(f.reg.register_op(gr_desc, err) == GUILD_STATUS_OK);
        CHECK(f.reg.resolve_prefill_kernels(f.caps, resolved, err) && resolved.all_bound());
        CHECK(resolved.quantize_q8_1 == f.legacy->quantize_q8_1);
        CHECK(resolved.mmvq == f.legacy->mmvq);
        CHECK(resolved.rms_norm_weighted == legacy->rms_norm_weighted);
        CHECK(resolved.gdn_ab == gdn_extension.gdn_ab);
        CHECK(resolved.router_projection == router_extension.router_projection);
    }

    // An old binary plugin's table does not contain projection pointers. Null next_ext
    // must produce an explicit compatibility error, never a read beyond those seven slots.
    for (auto op : {GUILD_OP_GDN_AB_PROJ, GUILD_OP_MOE_ROUTER_TOPK}) {
        PrefillTestRegistry f;
        f.replace(op, f.legacy.get(), nullptr);
        kernels::ResolvedPrefillKernels resolved;
        CHECK(!f.reg.resolve_prefill_kernels(f.caps, resolved, err));
        CHECK(err.find(prefill_test_id(op)) != std::string::npos);
        CHECK(err.find("incompatible prefill extension: missing next_ext") != std::string::npos);
        CHECK(!resolved.all_bound());
        if (op == GUILD_OP_GDN_AB_PROJ) CHECK(!resolved.gdn_ab);
        else CHECK(!resolved.router_projection);
    }

    // Reject malformed extensions before reading any projection pointers. Several cases
    // intentionally allocate only the advertised header/prefix, rather than a full table.
    for (auto op : {GUILD_OP_GDN_AB_PROJ, GUILD_OP_MOE_ROUTER_TOPK}) {
        PrefillTestRegistry f;
        kernels::ResolvedPrefillKernels resolved;
        const auto rejects = [&](const void* extension, const char* reason) {
            f.replace(op, f.legacy.get(), extension);
            CHECK(!f.reg.resolve_prefill_kernels(f.caps, resolved, err));
            CHECK(err.find(prefill_test_id(op)) != std::string::npos);
            CHECK(err.find("incompatible prefill extension") != std::string::npos);
            CHECK(err.find(reason) != std::string::npos);
            CHECK(!resolved.all_bound());
        };
        for (uint32_t size_only : {0u, uint32_t(sizeof(uint32_t))})
            rejects(&size_only, "smaller than the extension header");
        kernels::PrefillKernelExtensionHeader header;
        rejects(&header, "smaller than the v1 projection payload");
        struct TruncatedProjection {
            kernels::PrefillKernelExtensionHeader header;
            decltype(kernels::PrefillKernelExtensionV1::gdn_ab) gdn_ab;
        } truncated{{sizeof(TruncatedProjection), kernels::kPrefillKernelExtensionTag,
                     kernels::kPrefillKernelExtensionVersion}, f.extension.gdn_ab};
        rejects(&truncated, "smaller than the v1 projection payload");

        header.struct_size = sizeof(kernels::PrefillKernelExtensionV1);
        header.tag ^= 1u;
        rejects(&header, "unknown extension tag");
        header.tag = kernels::kPrefillKernelExtensionTag;
        for (uint32_t version : {0u, kernels::kPrefillKernelExtensionVersion + 1}) {
            header.version = version;
            rejects(&header, "unsupported extension version");
        }
        auto missing_entry = f.extension;
        if (op == GUILD_OP_GDN_AB_PROJ) missing_entry.gdn_ab = nullptr;
        else missing_entry.router_projection = nullptr;
        rejects(&missing_entry, op == GUILD_OP_GDN_AB_PROJ ? "missing gdn_ab" : "missing router_projection");

        // Extra trailing data is allowed for the recognized v1 prefix.
        struct ExtendedProjection {
            kernels::PrefillKernelExtensionV1 v1;
            uint64_t extra = 0;
        } extended{f.extension};
        extended.v1.header.struct_size = sizeof(extended);
        f.replace(op, f.legacy.get(), &extended);
        CHECK(f.reg.resolve_prefill_kernels(f.caps, resolved, err) && resolved.all_bound());
    }

    // A high-priority decode-width router must not displace the full-prefill implementation.
    {
        PrefillTestRegistry f;
        kernels::PrefillKernelExtensionV1 limited_extension;
        limited_extension.router_projection = +[](const float*, const uint16_t*, float*, int64_t, int64_t, int, void*) {};
        for (int rows : {1, 8, 511}) {
            auto desc = f.descriptor(GUILD_OP_MOE_ROUTER_TOPK);
            const std::string id = "test.prefill.limited_router_" + std::to_string(rows);
            desc.impl_id = id.c_str();
            desc.priority = 10000;
            desc.max_batch_rows = rows;
            desc.direct_fn_table = nullptr;
            desc.next_ext = &limited_extension;
            CHECK(f.reg.register_op(desc, err) == GUILD_STATUS_OK);
        }
        kernels::ResolvedPrefillKernels resolved;
        CHECK(f.reg.resolve_prefill_kernels(f.caps, resolved, err) && resolved.all_bound());
        CHECK(resolved.router_projection == f.extension.router_projection);
    }

    // Check every required operation's selection boundary: eight attention queries and
    // 512 input tokens for the rest, including head-expanded RMS/RoPE and shared experts.
    for (auto op : kPrefillTestOps) {
        PrefillTestRegistry f;
        const int64_t required = op == GUILD_OP_ATTENTION_DECODE ? 8 : 512;
        f.replace(op, f.legacy.get(), f.descriptor(op).next_ext, required - 1);
        kernels::ResolvedPrefillKernels resolved;
        CHECK(!f.reg.resolve_prefill_kernels(f.caps, resolved, err));
        CHECK(err.find("capability/shape/dtype constraints") != std::string::npos);
    }

    // -------------------------------------------------------------------------
    // 5. Declarative .gkmetadata Parsing, Security Policy, Dependency Graph & GEXEC Lock
    // -------------------------------------------------------------------------
    {
        const fs::path temp_dir = fs::temp_directory_path() / "guild_gkernel_test";
        fs::remove_all(temp_dir);
        fs::create_directories(temp_dir / "lib");

        const std::string dummy_so = "ELF_DUMMY_KERNEL_PAYLOAD_V1";
        {
            std::ofstream f(temp_dir / "lib" / "libguild_ornith_ext.so", std::ios::binary);
            f.write(dummy_so.data(), static_cast<std::streamsize>(dummy_so.size()));
        }
        const std::string art_sha = models::Sha256::hash_string(dummy_so);
        const std::string sig_hex(128, 'a');
        const std::string payload_sha(64, 'b');

        const std::string valid_json = R"({
          "format_version": 1,
          "package_id": "guild-kernels-ornith",
          "scope": "hybrid",
          "version": "1.2.0",
          "publisher_id": "org.illatio.guild",
          "kernel_abi": { "major": 1, "minor_min": 0, "minor_max": 0 },
          "required_core_caps": 1,
          "required_cpu_isa": 4,
          "required_gpu_caps": 0,
          "min_sm_version": 0,
          "backends": ["cpu"],
          "supported_architectures": ["qwen35moe"],
          "provided_ops": ["cpu.expert_pool.split_multi_native"],
          "provided_runners": ["runner.qwen35moe.v1"],
          "required_ops": [],
          "dependencies": [],
          "artifacts": [
            { "path": "lib/libguild_ornith_ext.so", "sha256": ")" + art_sha + R"(", "byte_size": )" + std::to_string(dummy_so.size()) + R"( }
          ],
          "signature": {
            "algorithm": "ed25519",
            "key_id": "illatio-release-2026-01",
            "signature_hex": ")" + sig_hex + R"(",
            "payload_sha256": ")" + payload_sha + R"("
          },
          "provenance": {
            "spdx_license": "MIT",
            "source_repo": "https://github.com/guild-inference/Guild",
            "source_revision": "40a4612000000000000000000000000000000000"
          }
        })";

        abi::GKernelMetadata meta;
        CHECK(abi::GKernelMetadata::parse_json(valid_json, meta, err) == GUILD_STATUS_OK);

        guild_device_capabilities_v1_t host_caps{};
        host_caps.struct_size = sizeof(host_caps);
        host_caps.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        host_caps.available_backends = GUILD_BACKEND_CPU;
        host_caps.cpu_isa_flags = GUILD_ISA_AVX2;
        host_caps.core_cap_flags = GUILD_CORE_CAP_HOST_ONLY_KV;

        abi::PackageSecurityPolicy policy;
        // Untrusted publisher rejected first
        CHECK(abi::validate_gkernel_package(meta, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Trust publisher key -> passes validation and on-disk SHA-256 verification
        policy.trust_publisher_key("org.illatio.guild", "illatio-release-2026-01");
        CHECK(abi::validate_gkernel_package(meta, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_OK);

        // Tamper with artifact hash -> rejected
        abi::GKernelMetadata tampered = meta;
        tampered.artifacts[0].sha256 = std::string(64, '0');
        CHECK(abi::validate_gkernel_package(tampered, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Revoke signing key -> rejected
        abi::PackageSecurityPolicy revoked_key_policy = policy;
        revoked_key_policy.revoke_key("illatio-release-2026-01");
        CHECK(abi::validate_gkernel_package(meta, revoked_key_policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Forbidden install_script hook -> rejected during parse without executing anything
        const std::string script_json = R"({
          "format_version": 1,
          "package_id": "malicious-pkg",
          "scope": "cpu-kernels",
          "version": "1.0.0",
          "publisher_id": "org.illatio.guild",
          "install_script": "sh -c 'echo pwned'",
          "kernel_abi": { "major": 1, "minor_min": 0, "minor_max": 0 }
        })";
        abi::GKernelMetadata bad_meta;
        CHECK(abi::GKernelMetadata::parse_json(script_json, bad_meta, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Dependency graph cycle detection
        abi::GKernelMetadata pkg_a = meta;
        pkg_a.package_id = "pkg-a";
        pkg_a.dependencies = {{"pkg-b", "1.0.0", "org.illatio.guild"}};
        abi::GKernelMetadata pkg_b = meta;
        pkg_b.package_id = "pkg-b";
        pkg_b.dependencies = {{"pkg-a", "1.0.0", "org.illatio.guild"}};
        CHECK(abi::validate_package_dependency_graph({pkg_a, pkg_b}, policy, host_caps, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);

        // GEXEC lock verification
        abi::GexecKernelLock lock;
        lock.package_id = "guild-kernels-ornith";
        lock.package_version = "1.2.0";
        lock.artifact_sha256 = art_sha;
        lock.impl_id = "cpu.expert_pool.split_multi_native";
        lock.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        const auto* op_desc = kernels::KernelRegistry::instance().find_by_id("cpu.expert_pool.split_multi_native");
        CHECK(op_desc != nullptr);
        CHECK(abi::verify_gexec_kernel_lock(lock, meta, *op_desc, err) == GUILD_STATUS_OK);

        // Rolling version change breaks GEXEC lock unless explicitly re-bound
        abi::GKernelMetadata newer_meta = meta;
        newer_meta.version = "1.3.0";
        CHECK(abi::verify_gexec_kernel_lock(lock, newer_meta, *op_desc, err) == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);

        fs::remove_all(temp_dir);
    }

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)
    // -------------------------------------------------------------------------
    // 6. Real GPU Capability Discovery, Runner Requirement Resolution & Prefill Table Binding
    // -------------------------------------------------------------------------
    {
        const auto host_caps = kernels::query_host_device_capabilities();
        if (host_caps.gpu_device_count > 0) {
            CHECK((host_caps.available_backends & (GUILD_BACKEND_CUDA | GUILD_BACKEND_HIP)) != 0);
            CHECK(host_caps.gpu_sm_version >= 60);

            kernels::ResolvedPrefillKernels table{};
            CHECK(kernels::KernelRegistry::instance().resolve_prefill_kernels(host_caps, table, err));
            CHECK(table.all_bound());

            for (const char* arch : {"qwen4exp", "qwen35moe"}) {
                const model::IModelRunner* runner = nullptr;
                CHECK(model::ModelRunnerRegistry::instance().resolve_by_arch(arch, &runner, err) == GUILD_STATUS_OK && runner);
                model::ModelDescriptor desc;
                desc.arch_name = arch;
                desc.archetype = runner->archetype();
                desc.attn.n_embd = ( std::strcmp(arch, "qwen4exp") == 0 ) ? 2560 : 2048;
                desc.attn.n_layers = ( std::strcmp(arch, "qwen4exp") == 0 ) ? 48 : 40;
                desc.attn.n_heads = ( std::strcmp(arch, "qwen4exp") == 0 ) ? 24 : 16;
                desc.attn.n_kv_heads = 2;
                desc.attn.head_dim = 256;
                desc.attn.full_attn_interval = 4;
                desc.moe.n_routed_experts = ( std::strcmp(arch, "qwen4exp") == 0 ) ? 512 : 256;
                desc.moe.k_active_experts = ( std::strcmp(arch, "qwen4exp") == 0 ) ? 10 : 8;
                desc.moe.expert_dim_ff = ( std::strcmp(arch, "qwen4exp") == 0 ) ? 640 : 512;

                core::ModelGeometry geom;
                int64_t k_act = 0;
                CHECK(runner->configure_geometry(desc, geom, k_act, err));

                memory::ExecutionPlan plan;
                plan.context_length = 2048;
                plan.kv_mode = memory::KvMode::HostOnly;
                plan.kv_format = memory::KvPrecision::FP16;

                model::RunnerExecutionPlan exec_plan;
                CHECK(model::ModelRunnerRegistry::instance().validate_runner_requirements(
                          *runner, desc, geom, plan, kernels::KernelRegistry::instance(),
                          host_caps, exec_plan, err) == GUILD_STATUS_OK);
                CHECK(!exec_plan.resolved_kernel_ids.empty());
                CHECK(exec_plan.resolved_kernel_ids.size() == exec_plan.required_ops.size());

                // Capability mismatch if core_cap_flags is stripped
                guild_device_capabilities_v1_t stripped_caps = host_caps;
                stripped_caps.core_cap_flags = GUILD_CORE_CAP_NONE;
                CHECK(model::ModelRunnerRegistry::instance().validate_runner_requirements(
                          *runner, desc, geom, plan, kernels::KernelRegistry::instance(),
                          stripped_caps, exec_plan, err) == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);
            }
        }
    }
#endif

    std::cout << "modular_abi_test: ABI v1, runner registry, kernel registry, .gkmetadata security, and GEXEC lock PASS\n";
    return 0;
}
