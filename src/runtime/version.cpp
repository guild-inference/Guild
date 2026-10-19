#include "guild/runtime/version.hpp"
#include "guild/build_metadata.hpp"
#include "guild/abi/gkernel_package.hpp"
#include "guild/model/runner.hpp"
#include "guild/server/json.hpp"
#include <sstream>

namespace guild::runtime {
bool version_report(bool json, std::string& output, std::string& error) {
    using server::json::JsonValue;
    JsonValue report;
    if (!JsonValue::parse(build::metadata_json, report, error))
        return false;
    auto& manager = abi::GKernelPackageManager::instance();
    const auto status = manager.load_enabled_packages(error);
    report["package_load_ok"] = status == GUILD_STATUS_OK;
    if (status != GUILD_STATUS_OK)
        report["package_error"] = error;
    std::vector<JsonValue> packages, kernel_values, runner_values;
    for (const auto& package : manager.list_packages()) {
        JsonValue value(std::vector<std::pair<std::string, JsonValue>>{});
        value["package_id"] = package.package_id;
        value["version"] = package.metadata.version;
        value["enabled"] = package.enabled;
        value["loaded_in_process"] = package.loaded_in_process;
        value["publisher_id"] = package.metadata.publisher_id;
        packages.push_back(std::move(value));
    }
    for (const auto& kernel : kernels::KernelRegistry::instance().list_operation_snapshots()) {
        JsonValue value(std::vector<std::pair<std::string, JsonValue>>{});
        value["impl_id"] = kernel.impl_id;
        value["package_id"] = kernel.package_id;
        value["enabled"] = kernel.enabled;
        value["impl_version_encoded"] = int64_t(kernel.impl_version);
        value["impl_version"] = std::to_string(GUILD_ABI_VERSION_MAJOR(kernel.impl_version)) + "." +
                                std::to_string(GUILD_ABI_VERSION_MINOR(kernel.impl_version)) + "." +
                                std::to_string(GUILD_ABI_VERSION_PATCH(kernel.impl_version));
        kernel_values.push_back(std::move(value));
    }
    for (const auto& runner : model::ModelRunnerRegistry::instance().list_runner_snapshots()) {
        JsonValue value(std::vector<std::pair<std::string, JsonValue>>{});
        value["runner_id"] = runner.runner_id;
        value["package_id"] = runner.package_id;
        value["enabled"] = runner.enabled;
        value["runner_version_encoded"] = int64_t(runner.runner_version);
        runner_values.push_back(std::move(value));
    }
    report["external_packages"] = JsonValue(std::move(packages));
    report["registered_kernels"] = JsonValue(std::move(kernel_values));
    report["registered_runners"] = JsonValue(std::move(runner_values));
    report["selection_scope"] = "process registry; model-specific bindings are reported by inference telemetry/tests";
    if (json)
        output = report.serialize_pretty() + "\n";
    else {
        std::ostringstream text;
        text << "Guild " << report["display_version"].as_string() << "\nChannel: " << report["channel"].as_string()
             << "\nCommit: " << report["commit"].as_string()
             << (report["tracked_source_dirty"].as_bool() ? " (tracked changes)" : "")
             << "\nBuild: " << report["build_timestamp_utc"].as_string() << "; " << report["build_type"].as_string()
             << "; " << report["compiler"].as_string() << "\nGPU toolchain: " << report["gpu_toolchain"].as_string()
             << "\nBackend: " << report["backend"].as_string()
             << "\nKernel ABI: " << report["kernel_abi_version"].as_string()
             << "\nKernel SDK: " << report["kernel_sdk_version"].as_string()
             << "\nCPU kernels: " << (report["bundled_cpu"]["enabled"].as_bool() ? "bundled " : "disabled ")
             << report["bundled_cpu"]["version"].as_string() << " ("
             << report["bundled_cpu"]["source_revision"].as_string() << ")\n";
        if (report["bundled_cuda"]["enabled"].as_bool()) {
            const bool dense_b = report["bundled_cuda"]["dense_bundled"].as_bool();
            const bool attn_b = report["bundled_cuda"]["attn_bundled"].as_bool();
            const bool moe_b = report["bundled_cuda"]["moe_bundled"].as_bool();
            text << "CUDA kernels: dense=" << (dense_b ? "bundled" : "external")
                 << " attention=" << (attn_b ? "bundled" : "external") << " moe=" << (moe_b ? "bundled" : "external")
                 << " " << report["bundled_cuda"]["version"].as_string() << " ("
                 << report["bundled_cuda"]["source_revision"].as_string() << ")\n";
        }
        if (report["backend"].as_string() == "none")
            text << "CPU-only inference: unavailable (management/ABI build)\n";
        for (const auto& package : manager.list_packages())
            text << "External package: " << package.package_id << "@" << package.metadata.version
                 << (package.enabled ? " enabled" : " disabled") << (package.loaded_in_process ? " loaded" : "")
                 << "\n";
        for (const auto& runner : model::ModelRunnerRegistry::instance().list_runner_snapshots())
            text << "Runner: " << runner.runner_id << " (" << runner.package_id << ") "
                 << (runner.enabled ? "enabled" : "disabled") << "\n";
        if (status != GUILD_STATUS_OK)
            text << "Package error: " << error << "\n";
        output = text.str();
    }
    return status == GUILD_STATUS_OK;
}
} // namespace guild::runtime
