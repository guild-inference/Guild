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
