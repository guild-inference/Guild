#include "../check.hpp"
#include "../signing_fixture.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/model/runner.hpp"
#include "guild/models/sha256.hpp"
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using namespace guild;
namespace {
constexpr auto v1 = GUILD_MAKE_ABI_VERSION(1, 0, 0);
std::atomic<int> partial_shutdowns{0};
guild_status_t ignored_failure_entry(uint32_t, guild_plugin_descriptor_v1_t* out, guild_error_info_t*) {
    *out = {}; out->struct_size = sizeof(*out); out->abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    out->package_id = "example.ignored_failure";
    out->init = [](guild_core_host_t host, const guild_host_api_v1_t* api, guild_error_info_t* e) -> guild_status_t {
        guild_kernel_op_desc_v1_t truncated{};
        CHECK(api->register_kernel_op(host, &truncated, e) == GUILD_STATUS_ERR_ABI_MISMATCH);
        return GUILD_STATUS_OK; // deliberately broken plugin ignores registration failure
    };
    out->shutdown = [](guild_core_host_t) { ++partial_shutdowns; };
    return GUILD_STATUS_OK;
}
void ok(guild_status_t st, const std::string& err) {
    if (st) std::cerr << "status=" << st << " " << err << '\n';
    CHECK(st == GUILD_STATUS_OK);
}
guild_tensor_desc_t tensor(float* data, size_t count) {
    guild_tensor_desc_t t{};
    t.struct_size = sizeof(t); t.dtype = GUILD_DTYPE_F32; t.n_dims = 1; t.dims[0] = count; t.strides_bytes[0] = 4;
    t.buffer.struct_size = sizeof(t.buffer); t.buffer.data = data; t.buffer.byte_size = count * 4;
    t.buffer.alignment = 4; t.buffer.lifetime = GUILD_LIFETIME_CALL_SYNC;
    return t;
}
struct Call {
    float input[4]{1,2,3,4}, output[4]{}, scratch = 0;
    guild_tensor_desc_t in = tensor(input, 4), out = tensor(output, 4);
    guild_operation_query_v1_t query{};
    guild_exec_context_v1_t ctx{};
    Call() {
        query.struct_size = sizeof(query); query.version = 1; query.operation_id = "example.sequence.prefix_sum";
        query.min_version = query.max_version = v1; query.allowed_backends = GUILD_BACKEND_CPU;
        query.required_capabilities = 1; query.sync_mode = GUILD_SYNC_HOST_BLOCKING;
        query.available_ram_bytes = query.available_vram_bytes = UINT64_MAX;
        query.inputs = &in; query.n_inputs = 1; query.outputs = &out; query.n_outputs = 1;
        ctx.struct_size = sizeof(ctx); ctx.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        ctx.backend = GUILD_BACKEND_CPU; ctx.sync_mode = GUILD_SYNC_HOST_BLOCKING;
        ctx.host_scratch = tensor(&scratch, 1).buffer;
    }
};
abi::GKernelMetadata package(const fs::path& dir, const char* lib, const char* id, const char* version = "1.0.0") {
    fs::create_directories(dir / "lib");
    fs::copy_file(lib, dir / "lib/module.so", fs::copy_options::overwrite_existing);
    abi::GKernelMetadata m;
    m.package_id = id; m.package_kind = "gkmodule"; m.module_abi = 1; m.scope = "module";
    m.version = version; m.publisher_id = abi::kOfficialGuildPublisherId;
    m.library_path = "lib/module.so"; m.supported_backends = GUILD_BACKEND_CPU;
    m.provided_ops = {std::string(id) + ".prefix_sum", std::string(id) + ".exclusive_sum"};
    m.artifacts.push_back({m.library_path, models::Sha256::hash_file((dir / m.library_path).string()), fs::file_size(dir / m.library_path), "shared_library"});
    return m;
}
void write(const fs::path& dir, abi::GKernelMetadata& m) {
    std::string err;
    CHECK(m.sign_with_seed_hex(abi::kOfficialGuildReleaseKeyId, abi::kOfficialGuildDevSeedHex, err));
    std::ofstream f(dir / ".gkmetadata"); f << m.to_json_string(); CHECK(f.good());
}
}

int main() {
    std::string err;
    {
        kernels::KernelRegistry reg;
        CHECK(reg.register_plugin(ignored_failure_entry, err) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
        CHECK(partial_shutdowns == 1 && reg.list_operations().empty());
    }
    const fs::path root = fs::current_path() / "gkmodule-test-work";
    CHECK(!fs::exists(root)); fs::create_directories(root);
    auto base = package(root / "base", GUILD_SCAN_SO, "example.scan"); write(root / "base", base);
    auto fast = package(root / "fast", GUILD_SCAN_FAST_SO, "example.scan.fast");
    fast.module_dependencies.push_back({base.package_id, "1.0.0", GUILD_MODULE_ABI_TAG_V1, base.publisher_id, base.compute_payload_sha256(), "1.0.0"});
    write(root / "fast", fast);
    auto upgrade = package(root / "upgrade", GUILD_SCAN_V11_SO, "example.scan", "1.1.0"); write(root / "upgrade", upgrade);
    auto invalid = package(root / "invalid", GUILD_SCAN_BAD_SO, "example.scan.invalid"); write(root / "invalid", invalid);
    auto duplicate = package(root / "duplicate", GUILD_SCAN_DUP_SO, "example.scan.duplicate"); write(root / "duplicate", duplicate);
    const auto caps = kernels::query_host_device_capabilities();
    auto policy = abi::PackageSecurityPolicy::default_builtin_policy(); abi::trust_test_signing_key(policy);
    {
        abi::GKernelMetadata parsed, incompatible = base;
        incompatible.module_abi = 2;
        CHECK(abi::GKernelMetadata::parse_json(incompatible.to_json_string(), parsed, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
        incompatible = base; incompatible.provided_runners = {"example.forbidden_runner"};
        CHECK(abi::validate_gkernel_package(incompatible, policy, caps, "", err) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
        incompatible = base; incompatible.module_dependencies.push_back({"example.missing", "1.0.0", "unknown.abi", "", "", ""});
        CHECK(abi::validate_gkernel_package(incompatible, policy, caps, "", err) == GUILD_STATUS_ERR_ABI_MISMATCH);
    }
    ok(abi::validate_package_dependency_graph({base, fast}, policy, caps, err), err);
    CHECK(abi::validate_package_dependency_graph({fast}, policy, caps, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
    auto cycle = base;
    cycle.module_dependencies.push_back({fast.package_id, "1.0.0", GUILD_MODULE_ABI_TAG_V1, "", "", ""});
    write(root / "base", cycle);
    auto cyclic_fast = fast; cyclic_fast.module_dependencies[0].sha256.clear(); write(root / "fast", cyclic_fast);
    CHECK(abi::validate_package_dependency_graph({cycle, cyclic_fast}, policy, caps, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
    CHECK(err.find("cyclic") != std::string::npos);
    write(root / "base", base); write(root / "fast", fast);
    auto bad = fast; bad.module_dependencies[0].expected_publisher_id = "wrong.publisher"; write(root / "fast", bad);
    CHECK(abi::validate_package_dependency_graph({base, bad}, policy, caps, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
    bad = fast; bad.module_dependencies[0].sha256 = std::string(64, '0'); write(root / "fast", bad);
    CHECK(abi::validate_package_dependency_graph({base, bad}, policy, caps, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
    write(root / "fast", fast);

    {
        kernels::KernelRegistry reg; model::ModelRunnerRegistry runners;
        abi::GKernelPackageManager pm((root / "store").string(), &reg, &runners);
        CHECK(pm.init(err));
        CHECK(pm.install_package((root / "base").string(), true, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
        abi::trust_test_signing_key(pm.security_policy()); CHECK(pm.save_security_policy(err));
        CHECK(pm.install_package((root / "fast").string(), true, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
        CHECK(pm.install_package((root / "invalid").string(), true, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
        CHECK(pm.install_package((root / "duplicate").string(), true, err) == GUILD_STATUS_ERR_DUPLICATE_REGISTRATION);
        CHECK(reg.list_operations().empty()); CHECK(pm.list_packages().empty());
        ok(abi::create_gkernel_bundle((root / "base").string(), (root / "scan.gkmodule").string(), err), err);
        ok(pm.install_package((root / "scan.gkmodule").string(), true, err), err);
        ok(pm.install_package((root / "fast").string(), true, err), err);
        CHECK(reg.list_operations().size() == 4);
        CHECK(pm.disable_package(base.package_id, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
        CHECK(pm.install_package((root / "upgrade").string(), true, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
        auto api = reg.operation_api(); Call call; guild_error_info_t e{}; guild_operation_binding_t h = nullptr;
        auto resolve = [&] { return api.resolve(api.user, &call.query, &caps, &h, &e); };
        ok(resolve(), e.message);
        CHECK(std::string(api.describe(h)->package_id) == fast.package_id);
        guild_scratch_request_t scratch{}; scratch.struct_size = sizeof(scratch);
        ok(api.scratch(h, &scratch, &e), e.message); CHECK(scratch.pageable_host_bytes == 4);
        ok(api.execute(h, &call.ctx, nullptr, 0, &e), e.message);
        CHECK(call.output[0] == 1 && call.output[1] == 3 && call.output[2] == 6 && call.output[3] == 10);
        CHECK(pm.disable_package(fast.package_id, err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
        CHECK(pm.remove_package(fast.package_id, err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
        auto invalid_ctx = call.ctx; invalid_ctx.host_scratch = {};
        CHECK(api.execute(h, &invalid_ctx, nullptr, 0, &e) == GUILD_STATUS_ERR_OUT_OF_MEMORY);
        invalid_ctx = call.ctx; invalid_ctx.backend = GUILD_BACKEND_CUDA;
        CHECK(api.execute(h, &invalid_ctx, nullptr, 0, &e) == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);
        api.release(h); h = nullptr;
        call.query.operation_id = "example.sequence.exclusive_sum";
        ok(resolve(), e.message); ok(api.execute(h, &call.ctx, nullptr, 0, &e), e.message);
        CHECK(call.output[0] == 0 && call.output[1] == 1 && call.output[2] == 3 && call.output[3] == 6);
        api.release(h); h = nullptr; call.query.operation_id = "example.sequence.prefix_sum";
        call.query.required_capabilities = 2; CHECK(resolve() == GUILD_STATUS_ERR_CAPABILITY_MISMATCH && !h);
        call.query.required_capabilities = 1; call.query.available_ram_bytes = 3;
        CHECK(resolve() == GUILD_STATUS_ERR_OUT_OF_MEMORY);
        call.query.available_ram_bytes = UINT64_MAX; call.query.min_version = call.query.max_version = v1 + 1;
        CHECK(resolve() == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);
        call.query.min_version = call.query.max_version = v1; call.query.exact_impl_id = "absent";
        CHECK(resolve() == GUILD_STATUS_ERR_MISSING_KERNEL);
        call.query.exact_impl_id = nullptr; call.in.strides_bytes[0] = 8;
        CHECK(resolve() == GUILD_STATUS_ERR_INVALID_TENSOR);
        call.in.strides_bytes[0] = 4; call.in.dtype = GUILD_DTYPE_I32;
        CHECK(resolve() == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);
        call.in.dtype = GUILD_DTYPE_F32; call.query.struct_size = 4;
        CHECK(resolve() == GUILD_STATUS_ERR_ABI_MISMATCH);
        call.query.struct_size = sizeof(call.query); call.query.inputs = nullptr;
        CHECK(resolve() == GUILD_STATUS_ERR_INVALID_ARGUMENT); call.query.inputs = &call.in;
        ok(pm.disable_package(fast.package_id, err), err);
        ok(resolve(), e.message); CHECK(std::string(api.describe(h)->package_id) == base.package_id); api.release(h);
        ok(pm.disable_package(base.package_id, err), err);
        CHECK(pm.enable_package(fast.package_id, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
        CHECK(pm.load_installed_package(fast.package_id, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
        abi::InstalledPackageRecord disabled; CHECK(pm.get_package(fast.package_id, disabled)); CHECK(!disabled.enabled);
        ok(pm.enable_package(base.package_id, err), err);
        ok(pm.remove_package(fast.package_id, err), err);

        // Real dlopen/resolve/execute races against package unloading/reloading.
        std::atomic<bool> start{false}; std::atomic<int> calls{0}, ready{0};
        std::vector<std::thread> workers;
        for (int t = 0; t < 8; ++t) workers.emplace_back([&] {
            Call local; guild_error_info_t le{};
            guild_operation_binding_t initial = nullptr;
            CHECK(api.resolve(api.user, &local.query, &caps, &initial, &le) == GUILD_STATUS_OK);
            ++ready;
            while (!start.load()) std::this_thread::yield();
            CHECK(api.execute(initial, &local.ctx, nullptr, 0, &le) == GUILD_STATUS_OK);
            ++calls; api.release(initial);
            for (int i = 0; i < 150; ++i) {
                guild_operation_binding_t binding = nullptr;
                auto st = api.resolve(api.user, &local.query, &caps, &binding, &le);
                CHECK(st == GUILD_STATUS_OK || st == GUILD_STATUS_ERR_MISSING_KERNEL);
                if (!st) {
                    CHECK(api.execute(binding, &local.ctx, nullptr, 0, &le) == GUILD_STATUS_OK);
                    CHECK(local.output[3] == 10); ++calls; api.release(binding);
                }
            }
        });
        while (ready.load() != 8) std::this_thread::yield();
        start = true;
        for (int i = 0; i < 30; ++i) {
            const auto st = pm.disable_package(base.package_id, err);
            CHECK(st == GUILD_STATUS_OK || st == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
            if (!st) ok(pm.enable_package(base.package_id, err), err);
        }
        for (auto& thread : workers) thread.join();
        CHECK(calls > 0);

        // A descriptor binding alone prevents unload, before any explicit lease.
        auto binding = reg.find_by_id("example.scan.prefix_sum"); CHECK(binding);
        CHECK(pm.disable_package(base.package_id, err) == GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY);
        // Equal priority is independent of registration order. Incompatible
        // higher-priority candidates fall back only when the query permits it.
        for (bool reverse : {false, true}) {
            kernels::KernelRegistry isolated;
            auto candidate = *binding;
            candidate.package_id = "example.tie";
            for (const char* id : reverse ? std::vector<const char*>{"example.a", "example.z"}
                                          : std::vector<const char*>{"example.z", "example.a"}) {
                candidate.impl_id = id;
                ok(isolated.register_op(candidate, err), err);
            }
            auto service = isolated.operation_api(); Call request; guild_operation_binding_t selected = nullptr;
            ok(service.resolve(service.user, &request.query, &caps, &selected, &e), e.message);
            CHECK(std::string(service.describe(selected)->impl_id) == "example.a"); service.release(selected);
            candidate.impl_id = "example.high"; candidate.priority += 100; candidate.min_dim_n = 8;
            ok(isolated.register_op(candidate, err), err);
            ok(service.resolve(service.user, &request.query, &caps, &selected, &e), e.message);
            CHECK(std::string(service.describe(selected)->impl_id) == "example.a"); service.release(selected);
            request.query.exact_impl_id = "example.high";
            CHECK(service.resolve(service.user, &request.query, &caps, &selected, &e) == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);
            CHECK(!selected && std::string(e.message).find("dimension") != std::string::npos);
            request.query.exact_impl_id = nullptr; request.query.allowed_backends = GUILD_BACKEND_CUDA;
            CHECK(service.resolve(service.user, &request.query, &caps, &selected, &e) == GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE);
            request.query.allowed_backends = GUILD_BACKEND_CPU; request.query.sync_mode = GUILD_SYNC_STREAM_ORDERED;
            CHECK(service.resolve(service.user, &request.query, &caps, &selected, &e) == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);
            request.query.sync_mode = GUILD_SYNC_HOST_BLOCKING; request.query.numerical_flags = GUILD_NUM_FAST_APPROX;
            CHECK(service.resolve(service.user, &request.query, &caps, &selected, &e) == GUILD_STATUS_ERR_CAPABILITY_MISMATCH);
        }
        auto desc = *binding;
        guild_operation_ext_v1_t ext{}; std::memcpy(&ext, desc.next_ext, sizeof(ext));
        desc.impl_id = "invalid.extension"; desc.next_ext = &ext;
        ext.struct_size = 4; CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
        ext.struct_size = sizeof(ext); ext.version = 2; CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_ABI_MISMATCH);
        ext.version = 1; ext.alignment = 3; CHECK(reg.register_op(desc, err) == GUILD_STATUS_ERR_INVALID_ARGUMENT);
        ext.alignment = 4; ext.struct_size = sizeof(ext) + 64;
        desc.package_id = "example.future";
        ok(reg.register_op(desc, err), err);
        {
            auto copied = reg.find_by_id(desc.impl_id); CHECK(copied);
            uint32_t actual = 0; std::memcpy(&actual, copied->next_ext, sizeof(actual));
            CHECK(actual == sizeof(ext));
        }
        ok(reg.unregister_package("example.future", err), err);
        auto extended_caps = caps; extended_caps.next_ext = &ext;
        h = nullptr;
        CHECK(api.resolve(api.user, &call.query, &extended_caps, &h, &e) == GUILD_STATUS_ERR_ABI_MISMATCH && !h);
        const char* operation_dependencies[]{"example.scan.prefix_sum"};
        desc.package_id = "example.consumer"; desc.impl_id = "example.consumer.scan";
        desc.required_op_deps = operation_dependencies; desc.n_required_op_deps = 1;
        ok(reg.register_op(desc, err), err);
        binding.reset();
        CHECK(reg.can_unregister_package(base.package_id, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
        CHECK(pm.disable_package(base.package_id, err) == GUILD_STATUS_ERR_DEPENDENCY_MISSING);
        ok(reg.unregister_package("example.consumer", err), err);
        pm.set_fault_injection(abi::PackageFaultPoint::DuringRegistryWrite);
        CHECK(pm.disable_package(base.package_id, err) == GUILD_STATUS_ERR_INTERNAL);
        pm.clear_fault_injection();
        CHECK(reg.find_by_id("example.scan.prefix_sum"));
        ok(pm.disable_package(base.package_id, err), err);
        pm.set_fault_injection(abi::PackageFaultPoint::DuringRegistryWrite);
        CHECK(pm.enable_package(base.package_id, err) == GUILD_STATUS_ERR_INTERNAL);
        pm.clear_fault_injection();
        CHECK(!reg.find_by_id("example.scan.prefix_sum"));
        pm.set_fault_injection(abi::PackageFaultPoint::DuringRegistryWrite);
        CHECK(pm.install_package((root / "base").string(), true, err) == GUILD_STATUS_ERR_INTERNAL);
        pm.clear_fault_injection();
        CHECK(!reg.find_by_id("example.scan.prefix_sum"));
        CHECK(pm.get_package(base.package_id, disabled)); CHECK(!disabled.enabled);
        ok(pm.enable_package(base.package_id, err), err);
        for (auto point : {abi::PackageFaultPoint::AfterStagingBeforeVerify, abi::PackageFaultPoint::AfterVerifyBeforePublish,
                           abi::PackageFaultPoint::AfterPublishBeforeLoad, abi::PackageFaultPoint::DuringRegistryWrite}) {
            pm.set_fault_injection(point);
            CHECK(pm.install_package((root / "upgrade").string(), true, err) == GUILD_STATUS_ERR_INTERNAL);
            pm.clear_fault_injection();
            abi::InstalledPackageRecord record; CHECK(pm.get_package(base.package_id, record)); CHECK(record.active_version == "1.0.0");
            CHECK(reg.find_by_id("example.scan.prefix_sum"));
        }
        ok(pm.install_package((root / "upgrade").string(), true, err), err);
        ok(pm.rollback_package(base.package_id, err), err);
        ok(pm.disable_package(base.package_id, err), err);
        // Re-enable must revalidate signatures and dependencies.
        pm.security_policy().revoke_release(base.package_id, base.version);
        CHECK(pm.enable_package(base.package_id, err) == GUILD_STATUS_ERR_SECURITY_VIOLATION);
        pm.security_policy().revoked_releases.clear();
        ok(pm.enable_package(base.package_id, err), err);
        ok(pm.remove_package(base.package_id, err), err);
    }
    // Fresh registry: explicitly loading a dependent first loads its verified
    // prerequisites. Startup loading the same store is idempotent.
    for (bool restart : {false, true}) {
        kernels::KernelRegistry reg; model::ModelRunnerRegistry runners;
        abi::GKernelPackageManager pm((root / "restart-store").string(), &reg, &runners);
        CHECK(pm.init(err));
        if (!restart) {
            abi::trust_test_signing_key(pm.security_policy()); CHECK(pm.save_security_policy(err));
            ok(pm.install_package((root / "base").string(), true, err), err);
            ok(pm.install_package((root / "fast").string(), true, err), err);
        } else {
            ok(pm.load_installed_package(fast.package_id, err), err);
            CHECK(reg.list_operations().size() == 4);
            ok(pm.load_enabled_packages(err), err);
            CHECK(reg.list_operations().size() == 4);
        }
    }
#if !defined(_WIN32)
    // Process death after publication: recover the last committed module version.
    for (auto point : {abi::PackageFaultPoint::AfterPublishBeforeLoad, abi::PackageFaultPoint::DuringRegistryWrite}) {
        const auto store = root / "crash-store";
        {
            kernels::KernelRegistry reg; model::ModelRunnerRegistry runners;
            abi::GKernelPackageManager pm(store.string(), &reg, &runners); CHECK(pm.init(err));
            abi::trust_test_signing_key(pm.security_policy()); CHECK(pm.save_security_policy(err));
            ok(pm.install_package((root / "base").string(), true, err), err);
        }
        const auto child = fork(); CHECK(child >= 0);
        if (!child) {
            kernels::KernelRegistry reg; model::ModelRunnerRegistry runners;
            abi::GKernelPackageManager pm(store.string(), &reg, &runners); CHECK(pm.init(err));
            pm.set_fault_injection(point, true); pm.install_package((root / "upgrade").string(), true, err); _Exit(3);
        }
        int status = 0; CHECK(waitpid(child, &status, 0) == child); CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 99);
        {
            kernels::KernelRegistry reg; model::ModelRunnerRegistry runners;
            abi::GKernelPackageManager pm(store.string(), &reg, &runners); CHECK(pm.init(err));
            ok(pm.load_enabled_packages(err), err);
            abi::InstalledPackageRecord record; CHECK(pm.get_package(base.package_id, record)); CHECK(record.active_version == "1.0.0");
            CHECK(!fs::exists(store / "packages/example.scan/1.1.0"));
        }
        fs::remove_all(store);
    }
#endif
    fs::remove_all(root);
    std::cout << "gkmodule registration, negotiation, invocation, lifecycle and recovery passed\n";
}
