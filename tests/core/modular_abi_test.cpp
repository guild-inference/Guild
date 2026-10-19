// tests/core/modular_abi_test.cpp - Modular Runner, Kernel Registry, ABI v1 & .gkmetadata Security Tests
#include "../check.hpp"
#include "../signing_fixture.hpp"
#include "guild/abi/gkernel_package.hpp"
#include "guild/abi/kernel_abi.h"
#include "guild/kernels/registry.hpp"
#include "guild/models/sha256.hpp"
#include "guild/model/runner.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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
    {
        hardware::HardwareInfo synthetic;
        synthetic.isa_avx2=true;
        auto caps=kernels::query_device_capabilities(synthetic);
        CHECK((caps.cpu_isa_flags&GUILD_ISA_BMI2)==0);
        CHECK((caps.cpu_isa_flags&GUILD_ISA_AVX512CD)==0);
        synthetic.isa_bmi2=true;synthetic.isa_avx512cd=true;
        caps=kernels::query_device_capabilities(synthetic);
        CHECK((caps.cpu_isa_flags&GUILD_ISA_BMI2)!=0);
        CHECK((caps.cpu_isa_flags&GUILD_ISA_AVX512CD)!=0);
    }
    {
        const auto policy=abi::PackageSecurityPolicy::default_builtin_policy();
        CHECK(policy.require_signature && policy.verify_artifact_hashes && !policy.allow_unsigned_dev);
        CHECK(policy.trusted_public_keys.empty());
        const auto root=fs::temp_directory_path()/"guild_corrupt_policy_test";
        fs::remove_all(root);fs::create_directories(root);
        { std::ofstream f(root/"trust_store.json");f<<"{broken"; }
        abi::GKernelPackageManager manager(root.string());
        CHECK(!manager.init(err));
        CHECK(manager.load_enabled_packages(err)==GUILD_STATUS_ERR_SECURITY_VIOLATION);
        fs::remove_all(root);
    }

    // -------------------------------------------------------------------------
    // 1. ABI Version Negotiation & Bit-Packing
    // -------------------------------------------------------------------------
    {
        uint32_t negotiated = 0;
        CHECK(kernels::negotiate_abi_version(GUILD_KERNEL_ABI_VERSION_V1, &negotiated) == GUILD_STATUS_OK);
        CHECK(negotiated == GUILD_KERNEL_ABI_VERSION_V1);
        CHECK(kernels::negotiate_abi_version(GUILD_KERNEL_ABI_MAKE_VERSION(2, 0, 0), &negotiated) == GUILD_STATUS_ERR_ABI_MISMATCH);
        CHECK(kernels::negotiate_abi_version(GUILD_KERNEL_ABI_MAKE_VERSION(0, 9, 0), &negotiated) == GUILD_STATUS_ERR_ABI_MISMATCH);
        CHECK(kernels::negotiate_abi_version(GUILD_KERNEL_ABI_MAKE_VERSION(1, 5, 0), &negotiated) == GUILD_STATUS_ERR_ABI_MISMATCH);

        // Verify 8/12/12 bit-packing across minor >= 16 and aliases
        const uint32_t v_test = GUILD_MAKE_ABI_VERSION(1, 25, 7);
        CHECK(GUILD_ABI_VERSION_MAJOR(v_test) == 1u);
        CHECK(GUILD_ABI_VERSION_MINOR(v_test) == 25u);
        CHECK(GUILD_ABI_VERSION_PATCH(v_test) == 7u);
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
        std::shared_ptr<const model::IModelRunner> qwen_runner;
        std::shared_ptr<const model::IModelRunner> ornith_runner;
        CHECK(builtin_runners.resolve_by_arch("qwen4exp", &qwen_runner, err) == GUILD_STATUS_OK && qwen_runner != nullptr);
        CHECK(builtin_runners.resolve_by_arch("qwen35moe", &ornith_runner, err) == GUILD_STATUS_OK && ornith_runner != nullptr);
        CHECK(builtin_runners.resolve_by_arch("ornith", &ornith_runner, err) == GUILD_STATUS_OK && ornith_runner != nullptr);

        // Missing runner for unregistered architecture
        std::shared_ptr<const model::IModelRunner> missing;
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
            model::SessionRunnerLease lease(&reg, "ext.guild-kernels-glm");
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
            kernels::KernelBinding selected;
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
                kernels::SessionKernelLease klease(&kreg, {"ext.dep"});
                CHECK(kreg.unregister_package("ext.dep", err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
            }
            CHECK(kreg.unregister_package("ext.dep", err) == GUILD_STATUS_OK);
            CHECK(kreg.validate_dependencies(err) == GUILD_STATUS_OK);

            // Kernel dispatch failure propagation (NaN detection via cpu.finite_check.simd64)
            const auto finite_op = kreg.find_by_id("cpu.finite_check.simd64");
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
        resolved = {};
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
        resolved = {};
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

    // CDD1 CUDA dense extension validation and deep-copy on GUILD_OP_QUANTIZED_GEMV.
    {
        kernels::KernelRegistry reg;
        guild_cuda_dense_ext_v1_t valid_dense{};
        valid_dense.struct_size = sizeof(valid_dense);
        valid_dense.tag = GUILD_CUDA_DENSE_TAG_V1;
        valid_dense.version = GUILD_CUDA_DENSE_VERSION_V1;
        valid_dense.bf16 = +[](const guild_exec_context_v1_t*, const guild_cuda_bf16_projection_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };
        valid_dense.quantize_q8_1 = +[](const guild_exec_context_v1_t*, const guild_cuda_quantize_q8_1_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };
        valid_dense.mmvq = +[](const guild_exec_context_v1_t*, const guild_cuda_mmvq_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };

        guild_kernel_op_desc_v1_t desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        desc.op_kind = GUILD_OP_QUANTIZED_GEMV;
        desc.op_name = "cuda_dense";
        desc.impl_id = "test.cuda.dense.v1";
        desc.package_id = "test.pkg.cuda";
        desc.impl_version = GUILD_KERNEL_ABI_VERSION_V1;
        desc.backend = GUILD_BACKEND_CUDA;
        desc.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC;
        desc.max_batch_rows = 512;
        desc.priority = 300;

        auto bad_ext = valid_dense;
        bad_ext.struct_size = sizeof(valid_dense) - 1;
        desc.next_ext = &bad_ext;
        CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_ABI_MISMATCH);

        bad_ext = valid_dense;
        bad_ext.tag ^= 1u;
        CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_ABI_MISMATCH);

        bad_ext = valid_dense;
        bad_ext.mmvq = nullptr;
        CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_ABI_MISMATCH);

        {
            guild_cuda_dense_ext_v1_t stack_copy = valid_dense;
            desc.next_ext = &stack_copy;
            CHECK(reg.register_op(desc, err) == GUILD_STATUS_OK);
            stack_copy = {};
        }

        guild_device_capabilities_v1_t cuda_caps{};
        cuda_caps.struct_size = sizeof(cuda_caps);
        cuda_caps.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        cuda_caps.available_backends = GUILD_BACKEND_CUDA;
        cuda_caps.available_vram_bytes = 1ull << 30;
        cuda_caps.available_ram_bytes = 1ull << 30;

        kernels::ResolvedCudaDense resolved_dense{};
        CHECK(reg.resolve_cuda_dense(cuda_caps, resolved_dense, err));
        CHECK(resolved_dense.impl_id == "test.cuda.dense.v1");
        CHECK(resolved_dense.package_id == "test.pkg.cuda");
        CHECK(resolved_dense.table.bf16 == valid_dense.bf16);
        CHECK(resolved_dense.table.quantize_q8_1 == valid_dense.quantize_q8_1);
        CHECK(resolved_dense.table.mmvq == valid_dense.mmvq);
    }

    // CDA1 CUDA attention extension validation and deep-copy on GUILD_OP_ATTENTION_DECODE.
    {
        kernels::KernelRegistry reg;
        guild_cuda_attn_ext_v1_t valid_attn{};
        valid_attn.struct_size = sizeof(valid_attn);
        valid_attn.tag = GUILD_CUDA_ATTN_TAG_V1;
        valid_attn.version = GUILD_CUDA_ATTN_VERSION_V1;
        valid_attn.configure = +[](const guild_exec_context_v1_t*, const guild_cuda_attn_config_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };
        valid_attn.rope = +[](const guild_exec_context_v1_t*, const guild_cuda_rope_op_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };
        valid_attn.qsa_norm_gate = +[](const guild_exec_context_v1_t*, const guild_cuda_qsa_norm_gate_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };
        valid_attn.gdn = +[](const guild_exec_context_v1_t*, const guild_cuda_gdn_op_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };
        valid_attn.kv = +[](const guild_exec_context_v1_t*, const guild_cuda_kv_op_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };
        valid_attn.qsa_index_select = +[](const guild_exec_context_v1_t*, const guild_cuda_qsa_index_select_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };
        valid_attn.attention = +[](const guild_exec_context_v1_t*, const guild_cuda_attention_op_v1_t*, guild_error_info_t*) -> guild_status_t {
            return GUILD_STATUS_OK;
        };

        guild_kernel_op_desc_v1_t desc{};
        desc.struct_size = sizeof(desc);
        desc.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        desc.op_kind = GUILD_OP_ATTENTION_DECODE;
        desc.op_name = "cuda_attn";
        desc.impl_id = "test.cuda.attn.v1";
        desc.package_id = "test.pkg.cuda";
        desc.impl_version = GUILD_KERNEL_ABI_VERSION_V1;
        desc.backend = GUILD_BACKEND_CUDA;
        desc.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC;
        desc.max_batch_rows = 8;
        desc.priority = 300;

        auto bad_ext = valid_attn;
        bad_ext.struct_size = sizeof(valid_attn) - 1;
        desc.next_ext = &bad_ext;
        CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_ABI_MISMATCH);

        bad_ext = valid_attn;
        bad_ext.tag ^= 1u;
        CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_ABI_MISMATCH);

        bad_ext = valid_attn;
        bad_ext.attention = nullptr;
        CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_ABI_MISMATCH);

        {
            guild_cuda_attn_ext_v1_t stack_copy = valid_attn;
            desc.next_ext = &stack_copy;
            CHECK(reg.register_op(desc, err) == GUILD_STATUS_OK);
            stack_copy = {};
        }

        guild_device_capabilities_v1_t cuda_caps{};
        cuda_caps.struct_size = sizeof(cuda_caps);
        cuda_caps.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        cuda_caps.available_backends = GUILD_BACKEND_CUDA;
        cuda_caps.available_vram_bytes = 1ull << 30;
        cuda_caps.available_ram_bytes = 1ull << 30;

        kernels::ResolvedCudaAttn resolved_attn{};
        CHECK(reg.resolve_cuda_attn(cuda_caps, resolved_attn, err));
        CHECK(resolved_attn.impl_id == "test.cuda.attn.v1");
        CHECK(resolved_attn.package_id == "test.pkg.cuda");
        CHECK(resolved_attn.table.configure == valid_attn.configure);
        CHECK(resolved_attn.table.rope == valid_attn.rope);
        CHECK(resolved_attn.table.qsa_norm_gate == valid_attn.qsa_norm_gate);
        CHECK(resolved_attn.table.gdn == valid_attn.gdn);
        CHECK(resolved_attn.table.kv == valid_attn.kv);
        CHECK(resolved_attn.table.qsa_index_select == valid_attn.qsa_index_select);
        CHECK(resolved_attn.table.attention == valid_attn.attention);
    }

    // -------------------------------------------------------------------------
    // CDM1 payload ownership, malformed tables and model-time selection.
    {
        kernels::KernelRegistry reg;
        guild_cuda_moe_ext_v1_t table{};
        table.struct_size = sizeof(table); table.tag = GUILD_CUDA_MOE_TAG_V1; table.version = GUILD_CUDA_MOE_VERSION_V1;
#define CDM_CALLBACK(member, type) \
        table.member = +[](const guild_exec_context_v1_t*, const type*, guild_error_info_t*) -> guild_status_t { return GUILD_STATUS_OK; }
        CDM_CALLBACK(configure, guild_cuda_moe_config_v1_t);
        CDM_CALLBACK(quantize_dequant, guild_cuda_quant_dequant_op_v1_t);
        CDM_CALLBACK(s_gemv, guild_cuda_s_gemv_op_v1_t);
        CDM_CALLBACK(moe, guild_cuda_moe_op_v1_t);
        CDM_CALLBACK(elementwise, guild_cuda_elementwise_op_v1_t);
        CDM_CALLBACK(sampler, guild_cuda_sampler_op_v1_t);
        CDM_CALLBACK(verify, guild_cuda_verify_op_v1_t);
        CDM_CALLBACK(qwen, guild_cuda_qwen_op_v1_t);
#undef CDM_CALLBACK
        guild_kernel_op_desc_v1_t d{};
        d.struct_size = sizeof(d); d.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        d.op_kind = GUILD_OP_MOE_COMBINE; d.op_name = "moe_combine";
        d.impl_id = "test.cuda.moe.v1"; d.package_id = "test.pkg.cuda.moe";
        d.backend = GUILD_BACKEND_CUDA; d.max_batch_rows = 512; d.priority = 300;
        d.numerical_flags = GUILD_NUM_BITWISE_DETERMINISTIC;
        auto bad = table; d.next_ext = &bad;
        bad.struct_size = sizeof(uint32_t);
        CHECK(reg.register_op(d, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
        bad = table; bad.tag ^= 1;
        CHECK(reg.register_op(d, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
        bad = table; bad.version += 1;
        CHECK(reg.register_op(d, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
        bad = table; bad.sampler = nullptr;
        CHECK(reg.register_op(d, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
        {
            auto temporary = table; d.next_ext = &temporary;
            CHECK(reg.register_op(d, err) == GUILD_STATUS_OK);
            temporary = {};
        }
        guild_device_capabilities_v1_t caps{};
        caps.struct_size = sizeof(caps); caps.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        caps.available_backends = GUILD_BACKEND_CUDA;
        caps.available_vram_bytes = caps.available_ram_bytes = 1ull << 30;
        kernels::ResolvedCudaMoE resolved{};
        CHECK(reg.resolve_cuda_moe(caps, resolved, err));
        CHECK(resolved.table.sampler == table.sampler && resolved.table.qwen == table.qwen);
        CHECK(resolved.impl_id == d.impl_id && resolved.package_id == d.package_id);
        reg.set_package_enabled(d.package_id, false);
        CHECK(!reg.resolve_cuda_moe(caps, resolved, err));
    }

    // 5. RFC 8032 Ed25519 Cryptographic Signatures, Declarative .gkmetadata,
    //    Three-Tier (.gkmodule) Boundary, Hardened Staging & GEXEC Lock
    // -------------------------------------------------------------------------
    {
        // 5a. Official RFC 8032 Section 7.1 Ed25519 Test Vectors
        {
            // Test Vector 1 (empty message)
            const std::string tv1_seed = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
            const std::string tv1_pub  = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
            const std::string tv1_sig  = "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
                                         "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b";
            abi::crypto::Ed25519KeyPair kp1;
            CHECK(abi::crypto::ed25519_keypair_from_seed_hex(tv1_seed, kp1));
            CHECK(kp1.public_key_hex == tv1_pub);
            std::string sig1_out;
            CHECK(abi::crypto::ed25519_sign_hex("", tv1_seed, sig1_out));
            CHECK(sig1_out == tv1_sig);
            CHECK(abi::crypto::ed25519_verify_hex("", tv1_sig, tv1_pub));
            CHECK(!abi::crypto::ed25519_verify_hex("a", tv1_sig, tv1_pub));

            // Test Vector 2 (1-byte message 0x72)
            const std::string tv2_seed = "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb";
            const std::string tv2_pub  = "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c";
            const std::string tv2_sig  = "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
                                         "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00";
            std::string sig2_out;
            const std::string msg2(1, '\x72');
            CHECK(abi::crypto::ed25519_sign_hex(msg2, tv2_seed, sig2_out));
            CHECK(sig2_out == tv2_sig);
            CHECK(abi::crypto::ed25519_verify_hex(msg2, tv2_sig, tv2_pub));

            // Non-canonical S >= L (all 0xFF in S[32..63]) must be rejected
            std::string bad_s_sig = tv1_sig.substr(0, 64) + std::string(64, 'f');
            CHECK(!abi::crypto::ed25519_verify_hex("", bad_s_sig, tv1_pub));
        }

        const fs::path temp_dir = fs::temp_directory_path() / "guild_gkernel_test";
        fs::remove_all(temp_dir);
        fs::create_directories(temp_dir / "lib");

        const std::string dummy_so = "ELF_DUMMY_KERNEL_PAYLOAD_V1";
        {
            std::ofstream f(temp_dir / "lib" / "libguild_ornith_ext.so", std::ios::binary);
            f.write(dummy_so.data(), static_cast<std::streamsize>(dummy_so.size()));
        }
        const std::string art_sha = models::Sha256::hash_string(dummy_so);

        const std::string valid_json = R"({
          "format_version": 1,
          "package_kind": "gkernel",
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
          "module_dependencies": [
            { "module_id": "guild-math-simd", "min_version": "1.0.0", "abi_tag": "avx2", "publisher_id": "org.illatio.guild", "sha256": ")" + art_sha + R"(" }
          ],
          "artifacts": [
            { "path": "lib/libguild_ornith_ext.so", "sha256": ")" + art_sha + R"(", "byte_size": )" + std::to_string(dummy_so.size()) + R"( }
          ],
          "signature": {
            "algorithm": "ed25519",
            "key_id": "illatio-release-2026-01",
            "signature_hex": ")" + std::string(128, 'a') + R"(",
            "payload_sha256": ")" + std::string(64, 'b') + R"("
          },
          "provenance": {
            "spdx_license": "MIT",
            "source_repo": "https://github.com/guild-inference/Guild",
            "source_revision": "40a4612000000000000000000000000000000000"
          }
        })";

        abi::GKernelMetadata meta;
        CHECK(abi::GKernelMetadata::parse_json(valid_json, meta, err) == GUILD_STATUS_OK);
        CHECK(meta.module_dependencies.size() == 1 && meta.module_dependencies[0].module_id == "guild-math-simd");

        guild_device_capabilities_v1_t host_caps{};
        host_caps.struct_size = sizeof(host_caps);
        host_caps.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        host_caps.available_backends = GUILD_BACKEND_CPU;
        host_caps.cpu_isa_flags = GUILD_ISA_AVX2;
        host_caps.core_cap_flags = GUILD_CORE_CAP_HOST_ONLY_KV;

        abi::PackageSecurityPolicy policy;
        // Untrusted publisher rejected first
        CHECK(abi::validate_gkernel_package(meta, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Trust publisher key, but fake 'aaaa...' signature must still be rejected by real Ed25519 verification!
        abi::trust_test_signing_key(policy);
        CHECK(abi::validate_gkernel_package(meta, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Now sign with the real Ed25519 private seed -> passes cryptographic signature + on-disk SHA-256 verification
        CHECK(meta.sign_with_seed_hex("illatio-release-2026-01", abi::kOfficialGuildDevSeedHex, err));
        CHECK(abi::validate_gkernel_package(meta, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_OK);

        // Tamper with any manifest field after signing -> rejected because payload_sha256 / Ed25519 signature mismatches
        abi::GKernelMetadata tampered_ver = meta;
        tampered_ver.version = "1.2.1";
        CHECK(abi::validate_gkernel_package(tampered_ver, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Tamper with signature bit -> rejected by Ed25519 verifier
        abi::GKernelMetadata tampered_sig = meta;
        tampered_sig.signature.signature_hex[0] = (tampered_sig.signature.signature_hex[0] == '0') ? '1' : '0';
        CHECK(abi::validate_gkernel_package(tampered_sig, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Tamper with artifact hash -> rejected
        abi::GKernelMetadata tampered = meta;
        tampered.artifacts[0].sha256 = std::string(64, '0');
        CHECK(abi::validate_gkernel_package(tampered, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Tamper with artifact file on disk after signing -> rejected
        {
            std::ofstream f(temp_dir / "lib" / "libguild_ornith_ext.so", std::ios::binary | std::ios::app);
            f.put('X');
        }
        CHECK(abi::validate_gkernel_package(meta, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
        // Restore valid artifact file
        {
            std::ofstream f(temp_dir / "lib" / "libguild_ornith_ext.so", std::ios::binary | std::ios::trunc);
            f.write(dummy_so.data(), static_cast<std::streamsize>(dummy_so.size()));
        }
        CHECK(abi::validate_gkernel_package(meta, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_OK);

        // .gkmodule packages cannot be directly loaded/validated as executable plugins by Guild Core
        abi::GKernelMetadata gkmod = meta;
        gkmod.package_kind = "gkmodule";
        CHECK(gkmod.sign_with_seed_hex("illatio-release-2026-01", abi::kOfficialGuildDevSeedHex, err));
        CHECK(abi::validate_gkernel_package(gkmod, policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_INVALID_ARGUMENT);

        // Revoke signing key -> rejected
        abi::PackageSecurityPolicy revoked_key_policy = policy;
        revoked_key_policy.revoke_key("illatio-release-2026-01");
        CHECK(abi::validate_gkernel_package(meta, revoked_key_policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Revoke publisher -> rejected
        abi::PackageSecurityPolicy revoked_pub_policy = policy;
        revoked_pub_policy.revoke_publisher("org.illatio.guild");
        CHECK(abi::validate_gkernel_package(meta, revoked_pub_policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Revoke release -> rejected
        abi::PackageSecurityPolicy revoked_rel_policy = policy;
        revoked_rel_policy.revoke_release("guild-kernels-ornith", "1.2.0");
        CHECK(abi::validate_gkernel_package(meta, revoked_rel_policy, host_caps, temp_dir.string(), err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

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

        // Unknown top-level key -> rejected during parse
        const std::string unknown_key_json = R"({
          "format_version": 1,
          "package_id": "unknown-key-pkg",
          "scope": "cpu-kernels",
          "version": "1.0.0",
          "publisher_id": "org.illatio.guild",
          "unexpected_field": 123,
          "kernel_abi": { "major": 1, "minor_min": 0, "minor_max": 0 }
        })";
        CHECK(abi::GKernelMetadata::parse_json(unknown_key_json, bad_meta, err) == GUILD_STATUS_ERR_INVALID_ARGUMENT);

        // Unsupported format_version -> rejected with ABI_MISMATCH
        const std::string future_fmt_json = R"({
          "format_version": 2,
          "package_id": "future-pkg",
          "scope": "cpu-kernels",
          "version": "1.0.0",
          "publisher_id": "org.illatio.guild",
          "kernel_abi": { "major": 1, "minor_min": 0, "minor_max": 0 }
        })";
        CHECK(abi::GKernelMetadata::parse_json(future_fmt_json, bad_meta, err) == GUILD_STATUS_ERR_ABI_MISMATCH);

        // Duplicate artifact path in .gkmetadata -> rejected with SECURITY_VIOLATION
        const std::string dup_art_json = R"({
          "format_version": 1,
          "package_id": "dup-art-pkg",
          "scope": "cpu-kernels",
          "version": "1.0.0",
          "publisher_id": "org.illatio.guild",
          "kernel_abi": { "major": 1, "minor_min": 0, "minor_max": 0 },
          "artifacts": [
            { "path": "lib/a.so", "sha256": ")" + art_sha + R"(", "byte_size": 27 },
            { "path": "lib/a.so", "sha256": ")" + art_sha + R"(", "byte_size": 27 }
          ]
        })";
        CHECK(abi::GKernelMetadata::parse_json(dup_art_json, bad_meta, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Dependency graph cycle detection
        abi::GKernelMetadata pkg_a = meta;
        pkg_a.package_id = "pkg-a";
        pkg_a.dependencies = {{"pkg-b", "1.0.0", "org.illatio.guild", ""}};
        CHECK(pkg_a.sign_with_seed_hex("illatio-release-2026-01", abi::kOfficialGuildDevSeedHex, err));
        abi::GKernelMetadata pkg_b = meta;
        pkg_b.package_id = "pkg-b";
        pkg_b.dependencies = {{"pkg-a", "1.0.0", "org.illatio.guild", ""}};
        CHECK(pkg_b.sign_with_seed_hex("illatio-release-2026-01", abi::kOfficialGuildDevSeedHex, err));
        CHECK(abi::validate_package_dependency_graph({pkg_a, pkg_b}, policy, host_caps, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);

        // GEXEC lock verification
        abi::GexecKernelLock lock;
        lock.package_id = "guild-kernels-ornith";
        lock.package_version = "1.2.0";
        lock.artifact_sha256 = art_sha;
        lock.impl_id = "cpu.expert_pool.split_multi_native";
        lock.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        const auto op_desc = kernels::KernelRegistry::instance().find_by_id("cpu.expert_pool.split_multi_native");
        CHECK(op_desc != nullptr);
        CHECK(abi::verify_gexec_kernel_lock(lock, meta, *op_desc, err) == GUILD_STATUS_OK);

        // Rolling version change breaks GEXEC lock unless explicitly re-bound
        abi::GKernelMetadata newer_meta = meta;
        newer_meta.version = "1.3.0";
        CHECK(abi::verify_gexec_kernel_lock(lock, newer_meta, *op_desc, err) == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);

        // 5b. Archive Bundling & Hardened Staging Security Tests (Symlink, Path Traversal, Undeclared .so, Zip Bomb)
        {
            std::ofstream mf(temp_dir / ".gkmetadata", std::ios::binary | std::ios::trunc);
            const std::string m_str = meta.to_json_string();
            mf.write(m_str.data(), static_cast<std::streamsize>(m_str.size()));
        }
        const fs::path bundle_path = temp_dir.parent_path() / "guild_test_pkg.gkernel";
        const fs::path stage_out = temp_dir.parent_path() / "guild_test_stage";
        CHECK(abi::create_gkernel_bundle(temp_dir.string(), bundle_path.string(), err) == GUILD_STATUS_OK);
        abi::GKernelMetadata staged_meta;
        CHECK(abi::stage_gkernel_source(bundle_path.string(), stage_out.string(), staged_meta, err) == GUILD_STATUS_OK);
        CHECK(abi::validate_gkernel_package(staged_meta, policy, host_caps, stage_out.string(), err) == GUILD_STATUS_OK);

        // Zip/archive bomb budget check (max_total_bytes = 8 bytes < actual payload)
        CHECK(abi::stage_gkernel_source(bundle_path.string(), stage_out.string(), staged_meta, err, 8, 64) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Max entries budget check (max_entries = 1 < 2 entries)
        CHECK(abi::stage_gkernel_source(bundle_path.string(), stage_out.string(), staged_meta, err, 1024 * 1024, 1) == GUILD_STATUS_ERR_SECURITY_VIOLATION);

        // Undeclared shared library or non-library file in package directory -> rejected
        {
            std::ofstream extra_so(temp_dir / "lib" / "undeclared_evil.so", std::ios::binary);
            extra_so << "UNDECLARED_SO";
        }
        CHECK(abi::stage_gkernel_source(temp_dir.string(), stage_out.string(), staged_meta, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
        fs::remove(temp_dir / "lib" / "undeclared_evil.so");
        {
            std::ofstream extra_txt(temp_dir / "undeclared_payload.txt", std::ios::binary);
            extra_txt << "UNDECLARED_PAYLOAD";
        }
        CHECK(abi::stage_gkernel_source(temp_dir.string(), stage_out.string(), staged_meta, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
        fs::remove(temp_dir / "undeclared_payload.txt");

        // GKPK archive with trailing unauthenticated bytes -> rejected
        {
            const fs::path trailing_archive = temp_dir.parent_path() / "trailing_bytes.gkernel";
            fs::copy_file(bundle_path, trailing_archive, fs::copy_options::overwrite_existing);
            {
                std::ofstream out(trailing_archive, std::ios::binary | std::ios::app);
                out.write("TRAILING", 8);
            }
            CHECK(abi::stage_gkernel_source(trailing_archive.string(), stage_out.string(), staged_meta, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
            fs::remove(trailing_archive);
        }

        // Truncated POSIX .tar archive header -> rejected
        {
            const fs::path trunc_tar = temp_dir.parent_path() / "truncated.tar";
            {
                std::ofstream out(trunc_tar, std::ios::binary | std::ios::trunc);
                char partial_block[200] = {'t', 'e', 's', 't'};
                out.write(partial_block, sizeof(partial_block));
            }
            CHECK(abi::stage_gkernel_source(trunc_tar.string(), stage_out.string(), staged_meta, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
            fs::remove(trunc_tar);
        }

#if !defined(_WIN32)
        // Symlink inside package directory -> rejected
        std::error_code sym_ec;
        fs::create_symlink("/etc/hosts", temp_dir / "lib" / "symlink_evil.txt", sym_ec);
        if (!sym_ec) {
            CHECK(abi::stage_gkernel_source(temp_dir.string(), stage_out.string(), staged_meta, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
            fs::remove(temp_dir / "lib" / "symlink_evil.txt", sym_ec);
        }
#endif

        // Malicious GKPK archive with "../escape.so" path traversal -> rejected before writing outside staging
        {
            const fs::path evil_archive = temp_dir.parent_path() / "evil_traversal.gkernel";
            std::ofstream out(evil_archive, std::ios::binary | std::ios::trunc);
            const char magic[8] = {'G', 'K', 'P', 'K', 1, 0, 0, 0};
            out.write(magic, 8);
            const uint32_t count = 1;
            out.write(reinterpret_cast<const char*>(&count), 4);
            const std::string evil_path = "../escaped_file.so";
            const uint32_t plen = static_cast<uint32_t>(evil_path.size());
            out.write(reinterpret_cast<const char*>(&plen), 4);
            out.write(evil_path.data(), plen);
            const uint64_t fsize = 4;
            out.write(reinterpret_cast<const char*>(&fsize), 8);
            out.write("EVIL", 4);
            out.close();
            CHECK(abi::stage_gkernel_source(evil_archive.string(), stage_out.string(), staged_meta, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
            CHECK(!fs::exists(stage_out.parent_path() / "escaped_file.so"));
            fs::remove(evil_archive);
        }

        fs::remove_all(temp_dir);
        fs::remove_all(stage_out);
        fs::remove(bundle_path);
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
                std::shared_ptr<const model::IModelRunner> runner;
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

#if !defined(_WIN32) && defined(GUILD_TEST_EXT_PREFILL_SO) && defined(GUILD_TEST_EXT_RUNNER_SO)
    // -------------------------------------------------------------------------
    // 7. Dynamic Shared Library .gkernel Loading (dlopen), Transactional
    //    Registration, Per-Package Leases, Concurrency, Upgrade/Rollback & CLI
    // -------------------------------------------------------------------------
    {
        const fs::path pkg_work = fs::temp_directory_path() / "guild_dyn_plugin_test";
        fs::remove_all(pkg_work);
        fs::create_directories(pkg_work / "store");

        auto build_signed_pkg = [&](const std::string& pkg_id,
                                    const std::string& ver,
                                    const std::string& scope,
                                    const std::string& so_src_path,
                                    const std::string& so_rel_name,
                                    const std::vector<std::string>& ops,
                                    const std::vector<std::string>& runners,
                                    const fs::path& out_dir,
                                    const fs::path& out_bundle) {
            fs::remove_all(out_dir);
            fs::create_directories(out_dir / "lib");
            const fs::path dst_so = out_dir / "lib" / so_rel_name;
            fs::copy_file(so_src_path, dst_so, fs::copy_options::overwrite_existing);

            abi::GKernelMetadata m;
            m.format_version = 1;
            m.package_kind = "gkernel";
            m.package_id = pkg_id;
            m.scope = scope;
            m.version = ver;
            m.publisher_id = abi::kOfficialGuildPublisherId;
            m.entrypoint_symbol = GUILD_PLUGIN_ENTRY_SYMBOL_V1;
            m.library_path = "lib/" + so_rel_name;
            m.abi_major = 1;
            m.abi_minor_min = 0;
            m.abi_minor_max = 0;
            m.supported_backends = GUILD_BACKEND_CPU | GUILD_BACKEND_CUDA;
            m.provided_ops = ops;
            m.provided_runners = runners;
            abi::ArtifactHashEntry art;
            art.relative_path = "lib/" + so_rel_name;
            art.byte_size = fs::file_size(dst_so);
            art.sha256 = models::Sha256::hash_file(dst_so.string());
            art.role = "shared_library";
            m.artifacts.push_back(art);
            m.provenance.spdx_license = "MIT";
            m.provenance.source_repo = "https://github.com/guild-inference/Guild";
            m.provenance.source_revision = "9d77159000000000000000000000000000000000";

            std::string sign_err;
            CHECK(m.sign_with_seed_hex(abi::kOfficialGuildReleaseKeyId, abi::kOfficialGuildDevSeedHex, sign_err));
            {
                std::ofstream mf(out_dir / ".gkmetadata", std::ios::binary | std::ios::trunc);
                const std::string j = m.to_json_string();
                mf.write(j.data(), static_cast<std::streamsize>(j.size()));
            }
            if (!out_bundle.empty()) {
                CHECK(abi::create_gkernel_bundle(out_dir.string(), out_bundle.string(), sign_err) == GUILD_STATUS_OK);
            }
        };

        std::vector<std::string> prefill_ops = {"ext.cpu.finite_check.v1"};
#if defined(GUILD_ENABLE_CUDA)
        prefill_ops.push_back("ext.cuda.moe.router_topk_bf16");
#endif
        const fs::path pkg_v10_dir = pkg_work / "src_prefill_v10";
        const fs::path pkg_v10_bundle = pkg_work / "ornith_prefill_v10.gkernel";
        build_signed_pkg("ext.guild-kernels-ornith-prefill", "1.0.0", "hybrid",
                         GUILD_TEST_EXT_PREFILL_SO, "libguild_ext_ornith_prefill.so",
                         prefill_ops, {}, pkg_v10_dir, pkg_v10_bundle);

        const fs::path pkg_v11_dir = pkg_work / "src_prefill_v11";
        const fs::path pkg_v11_bundle = pkg_work / "ornith_prefill_v11.gkernel";
        build_signed_pkg("ext.guild-kernels-ornith-prefill", "1.1.0", "hybrid",
                         GUILD_TEST_EXT_PREFILL_V11_SO, "libguild_ext_ornith_prefill_v11.so",
                         prefill_ops, {}, pkg_v11_dir, pkg_v11_bundle);

        const fs::path runner_dir = pkg_work / "src_runner_smoke";
        const fs::path runner_bundle = pkg_work / "runner_smoke.gkernel";
        build_signed_pkg("ext.guild-runner-smoke", "1.0.0", "runner",
                         GUILD_TEST_EXT_RUNNER_SO, "libguild_ext_runner_smoke.so",
                         {}, {"runner.ornith_ext_smoke.v1"}, runner_dir, runner_bundle);

        // Isolated registries and package manager
        kernels::KernelRegistry iso_kreg;
        iso_kreg.register_builtin_kernels();
        model::ModelRunnerRegistry iso_rreg;
        iso_rreg.register_builtin_runners();
        abi::GKernelPackageManager mgr((pkg_work / "store").string(), &iso_kreg, &iso_rreg);
        CHECK(mgr.init(err));
        abi::trust_test_signing_key(mgr.security_policy());

        // 7a. Transactional rollback test: if .gkmetadata declares an extra op that the plugin does not register,
        // installation must fail AND roll back all partial registrations from the plugin!
        {
            const fs::path bad_decl_dir = pkg_work / "src_bad_decl";
            const fs::path bad_decl_bundle = pkg_work / "bad_decl.gkernel";
            auto bad_ops = prefill_ops;
            bad_ops.push_back("ext.nonexistent.declared.op");
            build_signed_pkg("ext.guild-kernels-ornith-prefill", "1.0.0", "hybrid",
                             GUILD_TEST_EXT_PREFILL_SO, "libguild_ext_ornith_prefill.so",
                             bad_ops, {}, bad_decl_dir, bad_decl_bundle);
            CHECK(mgr.install_package(bad_decl_bundle.string(), true, err) == GUILD_STATUS_ERR_MISSING_KERNEL);
            // Verify transactional rollback: ext.cpu.finite_check.v1 was NOT left behind in iso_kreg
            CHECK(iso_kreg.find_by_id("ext.cpu.finite_check.v1") == nullptr);
        }

        // 7b. Install valid v1.0.0 kernel bundle and runner smoke bundle
        CHECK(mgr.install_package(pkg_v10_bundle.string(), true, err) == GUILD_STATUS_OK);
        CHECK(mgr.install_package(runner_bundle.string(), true, err) == GUILD_STATUS_OK);

        // Verify external runner is resolved
        std::shared_ptr<const model::IModelRunner> smoke_runner;
        CHECK(iso_rreg.resolve_by_arch("ornith_ext_smoke", &smoke_runner, err) == GUILD_STATUS_OK && smoke_runner != nullptr);
        CHECK(std::strcmp(smoke_runner->runner_id(), "runner.ornith_ext_smoke.v1") == 0);
        const auto r_abi = smoke_runner->abi_descriptor();
        CHECK(r_abi.session_alloc != nullptr && r_abi.session_free != nullptr);
        guild_session_state_handle_t smoke_state = nullptr;
        CHECK(r_abi.session_alloc(nullptr, 4096, &smoke_state, nullptr) == GUILD_STATUS_OK && smoke_state != nullptr);
        r_abi.session_free(smoke_state);
        smoke_runner.reset();

        // Verify external kernel is selected over built-in cpu.finite_check.simd64 (priority 250 > 100)
        const auto host_caps = kernels::query_host_device_capabilities();
        kernels::KernelSelectionQuery fq;
        fq.op_kind = GUILD_OP_FINITE_CHECK;
        fq.allowed_backends = GUILD_BACKEND_CPU;
        kernels::KernelBinding sel_finite;
        CHECK(iso_kreg.select(fq, host_caps, &sel_finite, err) == GUILD_STATUS_OK && sel_finite != nullptr);
        CHECK(std::strcmp(sel_finite->impl_id, "ext.cpu.finite_check.v1") == 0);
        CHECK(sel_finite->impl_version == GUILD_KERNEL_ABI_MAKE_VERSION(1, 0, 0));

        // Execute the external plugin's finite_check kernel on real data and on NaN data
        alignas(64) float ok_vals[16] = {1.0f, 2.0f, -3.5f, 4.25f};
        guild_tensor_desc_t t_ok{};
        t_ok.struct_size = sizeof(t_ok);
        t_ok.dtype = GUILD_DTYPE_F32;
        t_ok.n_dims = 1;
        t_ok.dims[0] = 16;
        t_ok.buffer.struct_size = sizeof(t_ok.buffer);
        t_ok.buffer.data = ok_vals;
        t_ok.buffer.byte_size = sizeof(ok_vals);
        t_ok.buffer.alignment = 32;
        t_ok.buffer.memory_space = GUILD_MEM_HOST_PAGEABLE;
        t_ok.buffer.ownership = GUILD_OWNERSHIP_CORE_BORROWED;
        t_ok.buffer.lifetime = GUILD_LIFETIME_CALL_SYNC;
        guild_error_info_t einfo{};
        CHECK(sel_finite->execute(nullptr, &t_ok, 1, nullptr, 0, nullptr, 0, &einfo) == GUILD_STATUS_OK);
        ok_vals[9] = std::numeric_limits<float>::infinity();
        CHECK(sel_finite->execute(nullptr, &t_ok, 1, nullptr, 0, nullptr, 0, &einfo) == GUILD_STATUS_ERR_NONFINITE_OUTPUT);
        sel_finite.reset();

#if defined(GUILD_ENABLE_CUDA)
        if (host_caps.gpu_device_count > 0) {
            kernels::ResolvedPrefillKernels pf_table{};
            CHECK(iso_kreg.resolve_prefill_kernels(host_caps, pf_table, err));
            CHECK(pf_table.all_bound());
            CHECK(std::find(pf_table.bound_impl_ids.begin(), pf_table.bound_impl_ids.end(),
                            "ext.cuda.moe.router_topk_bf16") != pf_table.bound_impl_ids.end());
            CHECK(std::find(pf_table.bound_package_ids.begin(), pf_table.bound_package_ids.end(),
                            "ext.guild-kernels-ornith-prefill") != pf_table.bound_package_ids.end());
        }
#endif

        // 7c. Active per-package lease protection & 8-thread concurrency stress test
        {
            kernels::SessionKernelLease pkg_lease(&iso_kreg, {"ext.guild-kernels-ornith-prefill"});
            CHECK(iso_kreg.active_package_leases("ext.guild-kernels-ornith-prefill") == 1);
            CHECK(mgr.disable_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
            CHECK(mgr.remove_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);

            std::atomic<bool> stop_threads{false};
            std::vector<std::thread> workers;
            for (int w = 0; w < 8; ++w) {
                workers.emplace_back([&, w]() {
                    alignas(64) float local_vals[16] = {1.0f, static_cast<float>(w)};
                    guild_tensor_desc_t tl = t_ok;
                    tl.buffer.data = local_vals;
                    for (int iter = 0; iter < 50 && !stop_threads.load(std::memory_order_relaxed); ++iter) {
                        kernels::KernelBinding d;
                        std::string lerr;
                        CHECK(iso_kreg.select(fq, host_caps, &d, lerr) == GUILD_STATUS_OK && d != nullptr);
                        guild_error_info_t le{};
                        CHECK(d->execute(nullptr, &tl, 1, nullptr, 0, nullptr, 0, &le) == GUILD_STATUS_OK);
                        auto snaps = iso_kreg.list_operation_snapshots();
                        CHECK(!snaps.empty());
                    }
                });
            }
            for (auto& th : workers) th.join();
        }
        CHECK(iso_kreg.active_package_leases("ext.guild-kernels-ornith-prefill") == 0);

        // 7d. Same-version immutability & idempotency tests (Stage D)
        // Re-installing the exact same v1.0.0 bundle is idempotent
        CHECK(mgr.install_package(pkg_v10_bundle.string(), true, err) == GUILD_STATUS_OK);

        // Building a different v1.0.0 bundle for the same package_id@1.0.0 must be rejected
        const fs::path pkg_v10_tampered_dir = pkg_work / "src_v10_tampered";
        const fs::path pkg_v10_tampered_bundle = pkg_work / "ornith_prefill_v1_0_0_tampered.gkernel";
        build_signed_pkg("ext.guild-kernels-ornith-prefill", "1.0.0", "cuda-kernels",
                         GUILD_TEST_EXT_PREFILL_SO, "libguild_ext_ornith_prefill.so",
                         prefill_ops, {}, pkg_v10_tampered_dir, pkg_v10_tampered_bundle);
        CHECK(mgr.install_package(pkg_v10_tampered_bundle.string(), true, err) == GUILD_STATUS_ERR_DUPLICATE_REGISTRATION);

        // Unrelated package lease isolation test (Stage B):
        // Holding a lease on ext.guild-runner-smoke must NOT block disabling/enabling ext.guild-kernels-ornith-prefill
        {
            model::SessionRunnerLease unrelated_lease(&iso_rreg, "ext.guild-runner-smoke");
            CHECK(iso_rreg.active_package_leases("ext.guild-runner-smoke") == 1);
            CHECK(mgr.disable_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);
            CHECK(mgr.enable_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);
            // Meanwhile, disabling ext.guild-runner-smoke itself IS blocked by its lease
            CHECK(mgr.disable_package("ext.guild-runner-smoke", err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
        }

        // Upgrade (1.0.0 -> 1.1.0) and Rollback (1.1.0 -> 1.0.0)
        CHECK(mgr.install_package(pkg_v11_bundle.string(), true, err) == GUILD_STATUS_OK);
        CHECK(iso_kreg.select(fq, host_caps, &sel_finite, err) == GUILD_STATUS_OK && sel_finite != nullptr);
        CHECK(sel_finite->impl_version == GUILD_KERNEL_ABI_MAKE_VERSION(1, 1, 0));
        sel_finite.reset();

        // Even while 1.1.0 is active, 1.0.0 is retained in the store and remains immutable against different 1.0.0 content
        CHECK(mgr.install_package(pkg_v10_tampered_bundle.string(), true, err) == GUILD_STATUS_ERR_DUPLICATE_REGISTRATION);

        CHECK(mgr.rollback_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);
        CHECK(iso_kreg.select(fq, host_caps, &sel_finite, err) == GUILD_STATUS_OK && sel_finite != nullptr);
        CHECK(sel_finite->impl_version == GUILD_KERNEL_ABI_MAKE_VERSION(1, 0, 0));
        sel_finite.reset();

        // 7d-faults. Fault-injection across all 6 transaction boundaries (Stage D)
        for (abi::PackageFaultPoint fp : {
                 abi::PackageFaultPoint::ArchiveRead,
                 abi::PackageFaultPoint::AfterStagingBeforeVerify,
                 abi::PackageFaultPoint::AfterVerifyBeforePublish,
                 abi::PackageFaultPoint::AfterPublishBeforeLoad,
                 abi::PackageFaultPoint::DuringRegistryWrite}) {
            mgr.set_fault_injection(fp, false);
            CHECK(mgr.install_package(pkg_v11_bundle.string(), true, err) == GUILD_STATUS_ERR_INTERNAL);
            // Verify v1.0.0 remains active, loaded, and selectable after every injected fault
            CHECK(iso_kreg.select(fq, host_caps, &sel_finite, err) == GUILD_STATUS_OK && sel_finite != nullptr);
            CHECK(sel_finite->impl_version == GUILD_KERNEL_ABI_MAKE_VERSION(1, 0, 0));
            sel_finite.reset();
            abi::InstalledPackageRecord rec_check;
            CHECK(mgr.get_package("ext.guild-kernels-ornith-prefill", rec_check));
            CHECK(rec_check.active_version == "1.0.0" && rec_check.enabled && rec_check.loaded_in_process);
        }

        // Fault injection during remove_package registry commit: package must remain installed and loaded
        mgr.set_fault_injection(abi::PackageFaultPoint::DuringRegistryWrite, false);
        CHECK(mgr.remove_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_ERR_INTERNAL);
        CHECK(iso_kreg.select(fq, host_caps, &sel_finite, err) == GUILD_STATUS_OK && sel_finite != nullptr);
        CHECK(sel_finite->impl_version == GUILD_KERNEL_ABI_MAKE_VERSION(1, 0, 0));
        sel_finite.reset();

        // Upgrade to 1.1.0, then inject fault during rollback load -> must stay on 1.1.0
        CHECK(mgr.install_package(pkg_v11_bundle.string(), true, err) == GUILD_STATUS_OK);
        mgr.set_fault_injection(abi::PackageFaultPoint::DuringRollbackLoad, false);
        CHECK(mgr.rollback_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_ERR_INTERNAL);
        CHECK(iso_kreg.select(fq, host_caps, &sel_finite, err) == GUILD_STATUS_OK && sel_finite != nullptr);
        CHECK(sel_finite->impl_version == GUILD_KERNEL_ABI_MAKE_VERSION(1, 1, 0));
        sel_finite.reset();

        // Roll back cleanly to 1.0.0
        CHECK(mgr.rollback_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);

#if !defined(_WIN32)
        // Real subprocess crash test (_Exit(99) during registry write & after publish) + startup recovery
        {
            const fs::path crash_store = pkg_work / "crash_store";
            {
                kernels::KernelRegistry ckreg;
                ckreg.register_builtin_kernels();
                model::ModelRunnerRegistry crreg;
                crreg.register_builtin_runners();
                abi::GKernelPackageManager cmgr(crash_store.string(), &ckreg, &crreg);
                CHECK(cmgr.init(err));
                abi::trust_test_signing_key(cmgr.security_policy());
                CHECK(cmgr.save_security_policy(err));
                CHECK(cmgr.install_package(pkg_v10_bundle.string(), true, err) == GUILD_STATUS_OK);
            }
            for (abi::PackageFaultPoint crash_fp : {
                     abi::PackageFaultPoint::AfterPublishBeforeLoad,
                     abi::PackageFaultPoint::DuringRegistryWrite}) {
                const pid_t pid = ::fork();
                CHECK(pid >= 0);
                if (pid == 0) {
                    kernels::KernelRegistry child_kreg;
                    child_kreg.register_builtin_kernels();
                    model::ModelRunnerRegistry child_rreg;
                    child_rreg.register_builtin_runners();
                    abi::GKernelPackageManager child_mgr(crash_store.string(), &child_kreg, &child_rreg);
                    std::string cerr;
                    if (!child_mgr.init(cerr)) std::_Exit(2);
                    child_mgr.set_fault_injection(crash_fp, true);
                    (void)child_mgr.install_package(pkg_v11_bundle.string(), true, cerr);
                    std::_Exit(0);
                }
                int wstatus = 0;
                CHECK(::waitpid(pid, &wstatus, 0) == pid);
                CHECK(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 99);

                // Recover store in parent process
                kernels::KernelRegistry rec_kreg;
                rec_kreg.register_builtin_kernels();
                model::ModelRunnerRegistry rec_rreg;
                rec_rreg.register_builtin_runners();
                abi::GKernelPackageManager rec_mgr(crash_store.string(), &rec_kreg, &rec_rreg);
                CHECK(rec_mgr.init(err));
                CHECK(!fs::exists(crash_store / "registry.json.tmp"));
                CHECK(fs::is_empty(crash_store / "staging"));
                CHECK(!fs::exists(crash_store / "packages" / "ext.guild-kernels-ornith-prefill" / "1.1.0"));
                CHECK(rec_mgr.load_enabled_packages(err) == GUILD_STATUS_OK);
                kernels::KernelBinding rec_sel;
                CHECK(rec_kreg.select(fq, host_caps, &rec_sel, err) == GUILD_STATUS_OK && rec_sel != nullptr);
                CHECK(rec_sel->impl_version == GUILD_KERNEL_ABI_MAKE_VERSION(1, 0, 0));
            }
            fs::remove_all(crash_store);
        }
#endif

        // Adversarial 12-thread concurrent selection, execution, inspection & lifecycle replacement test (Stages B & C)
        {
            std::atomic<bool> done{false};
            std::vector<std::thread> adv_threads;
            for (int w = 0; w < 8; ++w) {
                adv_threads.emplace_back([&, w]() {
                    alignas(64) float local_vals[16] = {1.0f, static_cast<float>(w)};
                    guild_tensor_desc_t tl = t_ok;
                    tl.buffer.data = local_vals;
                    for (int iter = 0; iter < 60 && !done.load(std::memory_order_relaxed); ++iter) {
                        kernels::KernelBinding d;
                        std::string lerr;
                        if (iso_kreg.select(fq, host_caps, &d, lerr) == GUILD_STATUS_OK && d != nullptr) {
                            guild_error_info_t le{};
                            CHECK(d->execute(nullptr, &tl, 1, nullptr, 0, nullptr, 0, &le) == GUILD_STATUS_OK);
                        }
                        std::shared_ptr<const model::IModelRunner> r;
                        if (iso_rreg.resolve_by_arch("ornith_ext_smoke", &r, lerr) == GUILD_STATUS_OK && r != nullptr) {
                            CHECK(std::strcmp(r->runner_id(), "runner.ornith_ext_smoke.v1") == 0);
                        }
                    }
                });
            }
            for (int insp = 0; insp < 2; ++insp) {
                adv_threads.emplace_back([&]() {
                    for (int iter = 0; iter < 60 && !done.load(std::memory_order_relaxed); ++iter) {
                        auto pkgs = mgr.list_packages();
                        CHECK(!pkgs.empty());
                        auto ops = iso_kreg.list_operation_snapshots();
                        CHECK(!ops.empty());
                        auto rns = iso_rreg.list_runners();
                        CHECK(!rns.empty());
                    }
                });
            }
            for (int mod = 0; mod < 2; ++mod) {
                adv_threads.emplace_back([&]() {
                    for (int iter = 0; iter < 15; ++iter) {
                        std::string merr;
                        guild_status_t st = mgr.install_package(pkg_v11_bundle.string(), true, merr);
                        CHECK(st == GUILD_STATUS_OK || st == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY ||
                              st == GUILD_STATUS_ERR_DUPLICATE_REGISTRATION);
                        st = mgr.rollback_package("ext.guild-kernels-ornith-prefill", merr);
                        CHECK(st == GUILD_STATUS_OK || st == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY ||
                              st == GUILD_STATUS_ERR_INVALID_ARGUMENT);
                    }
                });
            }
            for (auto& th : adv_threads) th.join();
            done.store(true);
            // Ensure package is enabled and on v1.0.0 after adversarial stress
            CHECK(mgr.enable_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);
            abi::InstalledPackageRecord final_rec;
            CHECK(mgr.get_package("ext.guild-kernels-ornith-prefill", final_rec));
            if (final_rec.active_version != "1.0.0") {
                CHECK(mgr.rollback_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);
            }
        }

        // 7e. Disable package -> automatically falls back to built-in cpu.finite_check.simd64
        sel_finite.reset();
        CHECK(mgr.disable_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);
        CHECK(iso_kreg.select(fq, host_caps, &sel_finite, err) == GUILD_STATUS_OK && sel_finite != nullptr);
        CHECK(std::strcmp(sel_finite->impl_id, "cpu.finite_check.simd64") == 0);

        // Re-enable -> selects external plugin again
        CHECK(mgr.enable_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);
        CHECK(iso_kreg.select(fq, host_caps, &sel_finite, err) == GUILD_STATUS_OK && sel_finite != nullptr);
        CHECK(std::strcmp(sel_finite->impl_id, "ext.cpu.finite_check.v1") == 0);

        // Remove packages cleanly
        sel_finite.reset();
        CHECK(mgr.remove_package("ext.guild-kernels-ornith-prefill", err) == GUILD_STATUS_OK);
        CHECK(mgr.remove_package("ext.guild-runner-smoke", err) == GUILD_STATUS_OK);
        CHECK(mgr.list_packages().empty());

#if defined(GUILD_TEST_CLI_BIN)
        // 7f. End-to-End CLI Package Management Subcommands Test (`guild kernel ...`)
        const fs::path cli_data_dir = pkg_work / "cli_data";
        const std::string cli_bin = GUILD_TEST_CLI_BIN;
        auto run_cli = [&](const std::string& args) -> int {
            const std::string cmd = cli_bin + " kernel " + args + " > /dev/null 2>&1";
            return std::system(cmd.c_str());
        };
        CHECK(run_cli("keygen --seed " + std::string(abi::kOfficialGuildDevSeedHex) + " --json") == 0);
        CHECK(run_cli("trust-key --publisher org.illatio.guild --key-id illatio-release-2026-01 --pubkey " +
                      abi::official_guild_public_key_hex() + " --data-dir " + cli_data_dir.string()) == 0);
        CHECK(run_cli("install " + pkg_v10_bundle.string() + " --data-dir " + cli_data_dir.string() + " --json") == 0);
        CHECK(run_cli("list --data-dir " + cli_data_dir.string() + " --json") == 0);
        CHECK(run_cli("inspect ext.guild-kernels-ornith-prefill --data-dir " + cli_data_dir.string() + " --json") == 0);
        CHECK(run_cli("active --data-dir " + cli_data_dir.string() + " --json") == 0);
        CHECK(run_cli("install " + pkg_v11_bundle.string() + " --data-dir " + cli_data_dir.string()) == 0);
        CHECK(run_cli("rollback ext.guild-kernels-ornith-prefill --data-dir " + cli_data_dir.string()) == 0);
        CHECK(run_cli("disable ext.guild-kernels-ornith-prefill --data-dir " + cli_data_dir.string()) == 0);
        CHECK(run_cli("enable ext.guild-kernels-ornith-prefill --data-dir " + cli_data_dir.string()) == 0);
        CHECK(run_cli("remove ext.guild-kernels-ornith-prefill --data-dir " + cli_data_dir.string()) == 0);
#endif

        fs::remove_all(pkg_work);
    }
#endif

    std::cout << "modular_abi_test: ABI v1, Ed25519 crypto, .gkmetadata, dlopen plugin lifecycle, concurrency, and CLI PASS\n";
    return 0;
}
