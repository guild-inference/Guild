#include "guild/gexec/gexec_contract.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/models/sha256.hpp"
#include "guild/server/json.hpp"

#if defined(GUILD_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace guild::gexec {

using namespace guild::server::json;

namespace {
void erase_json_key(JsonValue& obj, const std::string& key) {
    obj.obj_val.erase(
        std::remove_if(obj.obj_val.begin(), obj.obj_val.end(),
                       [&](const auto& kv) { return kv.first == key; }),
        obj.obj_val.end());
}
} // namespace

const char* invalidation_reason_to_string(GExecInvalidationReason reason) noexcept {
    switch (reason) {
        case GExecInvalidationReason::None: return "valid";
        case GExecInvalidationReason::MagicOrVersionMismatch: return "magic or format version mismatch";
        case GExecInvalidationReason::CorruptedDigest: return "corrupted cache digest";
        case GExecInvalidationReason::GPackIdentityMismatch: return "model GPACK manifest SHA-256 mismatch";
        case GExecInvalidationReason::CoreAbiMismatch: return "Guild Core or Module ABI version mismatch";
        case GExecInvalidationReason::ModuleOrKernelDigestMismatch: return "bound kernel or module package digest mismatch";
        case GExecInvalidationReason::RunnerIdentityMismatch: return "model runner identity or version mismatch";
        case GExecInvalidationReason::CpuIsaMismatch: return "CPU vendor or ISA feature flags mismatch";
        case GExecInvalidationReason::NumaTopologyMismatch: return "NUMA node count or CPU core topology mismatch";
        case GExecInvalidationReason::GpuArchitectureMismatch: return "GPU device architecture or VRAM capacity mismatch";
        case GExecInvalidationReason::CudaRuntimeOrDriverMismatch: return "CUDA runtime or driver version mismatch";
        case GExecInvalidationReason::ExecutionPlanMismatch: return "execution plan context/KV/batch/spec configuration mismatch";
    }
    return "unknown invalidation reason";
}

GExecHardwareFingerprint GExecHardwareFingerprint::detect() {
    GExecHardwareFingerprint fp;
    const auto hw = guild::hardware::detect_hardware();
    const auto caps = guild::kernels::query_host_device_capabilities();

    fp.cpu_vendor = "x86_64";
    fp.cpu_brand = hw.cpu_model;
    fp.cpu_feature_flags = static_cast<uint32_t>(caps.cpu_isa_flags);
    fp.cpu_physical_cores = hw.cpu_physical_cores;
    fp.cpu_logical_threads = hw.cpu_logical_threads;
    fp.numa_nodes = std::max(1, hw.cpu_sockets);

    {
        std::ifstream thp("/sys/kernel/mm/transparent_hugepage/enabled");
        if (thp.is_open()) {
            std::string line;
            std::getline(thp, line);
            fp.thp_madvise_available = (line.find("[always]") != std::string::npos ||
                                        line.find("[madvise]") != std::string::npos);
        }
    }

    fp.gpu_count = hw.gpu_count;
    fp.gpu_name = hw.gpu_name;
    fp.gpu_cc_major = caps.gpu_sm_version / 10;
    fp.gpu_cc_minor = caps.gpu_sm_version % 10;
    fp.gpu_vram_bytes = hw.gpu_vram_bytes;

#if defined(GUILD_ENABLE_CUDA)
    int rt_ver = 0, drv_ver = 0;
    if (cudaRuntimeGetVersion(&rt_ver) == cudaSuccess) fp.cuda_runtime_version = rt_ver;
    if (cudaDriverGetVersion(&drv_ver) == cudaSuccess) fp.cuda_driver_version = drv_ver;
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count > 0) {
        fp.gpu_count = dev_count;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            fp.gpu_name = prop.name;
            fp.gpu_cc_major = prop.major;
            fp.gpu_cc_minor = prop.minor;
            fp.gpu_vram_bytes = static_cast<uint64_t>(prop.totalGlobalMem);
            fp.gpu_multiprocessors = prop.multiProcessorCount;
        }
    }
#endif
    return fp;
}

std::string GExecHardwareFingerprint::to_json() const {
    JsonValue obj;
    obj["cpu_vendor"] = cpu_vendor;
    obj["cpu_brand"] = cpu_brand;
    obj["cpu_feature_flags"] = static_cast<int64_t>(cpu_feature_flags);
    obj["cpu_physical_cores"] = static_cast<int64_t>(cpu_physical_cores);
    obj["cpu_logical_threads"] = static_cast<int64_t>(cpu_logical_threads);
    obj["numa_nodes"] = static_cast<int64_t>(numa_nodes);
    obj["thp_madvise_available"] = thp_madvise_available;
    obj["gpu_count"] = static_cast<int64_t>(gpu_count);
    obj["gpu_name"] = gpu_name;
    obj["gpu_cc_major"] = static_cast<int64_t>(gpu_cc_major);
    obj["gpu_cc_minor"] = static_cast<int64_t>(gpu_cc_minor);
    obj["gpu_vram_bytes"] = static_cast<int64_t>(gpu_vram_bytes);
    obj["gpu_multiprocessors"] = static_cast<int64_t>(gpu_multiprocessors);
    obj["cuda_runtime_version"] = static_cast<int64_t>(cuda_runtime_version);
    obj["cuda_driver_version"] = static_cast<int64_t>(cuda_driver_version);
    return obj.dump();
}

std::string GExecHardwareFingerprint::fingerprint_sha256() const {
    return models::Sha256::hash_string(to_json());
}

std::string GExecArtifactIdentity::to_json() const {
    JsonValue root;
    root["format_version"] = static_cast<int64_t>(format_version);
    root["gpack_manifest_sha256"] = gpack_manifest_sha256;
    root["core_abi_version"] = static_cast<int64_t>(core_abi_version);
    root["module_abi_version"] = static_cast<int64_t>(module_abi_version);
    root["runner_id"] = runner_id;
    root["runner_version"] = static_cast<int64_t>(runner_version);

    std::vector<GExecBoundPackageDigest> sorted_pkgs = bound_packages;
    std::sort(sorted_pkgs.begin(), sorted_pkgs.end(),
              [](const GExecBoundPackageDigest& a, const GExecBoundPackageDigest& b) {
                  return a.package_id < b.package_id;
              });
    JsonValue pkgs_arr(std::vector<JsonValue>{});
    for (const auto& p : sorted_pkgs) {
        JsonValue po;
        po["package_id"] = p.package_id;
        po["version"] = p.version;
        po["payload_sha256"] = p.payload_sha256;
        pkgs_arr.arr_val.push_back(std::move(po));
    }
    root["bound_packages"] = std::move(pkgs_arr);

    std::string perr;
    JsonValue hw_val;
    JsonValue::parse(hardware.to_json(), hw_val, perr);
    root["hardware"] = std::move(hw_val);

    root["kv_format"] = kv_format;
    root["kv_mode"] = kv_mode;
    root["max_context"] = static_cast<int64_t>(max_context);
    root["prefill_batch"] = static_cast<int64_t>(prefill_batch);
    root["mtp_spec_tokens"] = static_cast<int64_t>(mtp_spec_tokens);
    return root.dump();
}

std::string GExecArtifactIdentity::compute_cache_key() const {
    return models::Sha256::hash_string(to_json());
}

bool GExecArtifactIdentity::from_json(const std::string& json_str, GExecArtifactIdentity& out, std::string& err) {
    JsonValue root;
    if (!JsonValue::parse(json_str, root, err) || !root.is_object()) {
        if (err.empty()) err = "GExecArtifactIdentity JSON must be an object";
        return false;
    }
    out = GExecArtifactIdentity{};
    out.format_version = static_cast<uint16_t>(root["format_version"].as_int(0));
    out.gpack_manifest_sha256 = root["gpack_manifest_sha256"].as_string();
    out.core_abi_version = static_cast<uint32_t>(root["core_abi_version"].as_int(0));
    out.module_abi_version = static_cast<uint32_t>(root["module_abi_version"].as_int(0));
    out.runner_id = root["runner_id"].as_string();
    out.runner_version = static_cast<uint32_t>(root["runner_version"].as_int(0));

    if (root.contains("bound_packages") && root["bound_packages"].is_array()) {
        for (const auto& item : root["bound_packages"].arr_val) {
            if (!item.is_object()) continue;
            GExecBoundPackageDigest d;
            d.package_id = item["package_id"].as_string();
            d.version = item["version"].as_string();
            d.payload_sha256 = item["payload_sha256"].as_string();
            out.bound_packages.push_back(std::move(d));
        }
    }
    if (root.contains("hardware") && root["hardware"].is_object()) {
        const auto& hw = root["hardware"];
        out.hardware.cpu_vendor = hw["cpu_vendor"].as_string();
        out.hardware.cpu_brand = hw["cpu_brand"].as_string();
        out.hardware.cpu_feature_flags = static_cast<uint32_t>(hw["cpu_feature_flags"].as_int(0));
        out.hardware.cpu_physical_cores = static_cast<int32_t>(hw["cpu_physical_cores"].as_int(0));
        out.hardware.cpu_logical_threads = static_cast<int32_t>(hw["cpu_logical_threads"].as_int(0));
        out.hardware.numa_nodes = static_cast<int32_t>(hw["numa_nodes"].as_int(1));
        out.hardware.thp_madvise_available = hw["thp_madvise_available"].as_bool(false);
        out.hardware.gpu_count = static_cast<int32_t>(hw["gpu_count"].as_int(0));
        out.hardware.gpu_name = hw["gpu_name"].as_string();
        out.hardware.gpu_cc_major = static_cast<int32_t>(hw["gpu_cc_major"].as_int(0));
        out.hardware.gpu_cc_minor = static_cast<int32_t>(hw["gpu_cc_minor"].as_int(0));
        out.hardware.gpu_vram_bytes = static_cast<uint64_t>(hw["gpu_vram_bytes"].as_int(0));
        out.hardware.gpu_multiprocessors = static_cast<int32_t>(hw["gpu_multiprocessors"].as_int(0));
        out.hardware.cuda_runtime_version = static_cast<int32_t>(hw["cuda_runtime_version"].as_int(0));
        out.hardware.cuda_driver_version = static_cast<int32_t>(hw["cuda_driver_version"].as_int(0));
    }
    out.kv_format = root["kv_format"].as_string();
    out.kv_mode = root["kv_mode"].as_string();
    out.max_context = root["max_context"].as_int(0);
    out.prefill_batch = root["prefill_batch"].as_int(0);
    out.mtp_spec_tokens = static_cast<int32_t>(root["mtp_spec_tokens"].as_int(0));
    return true;
}

GExecInvalidationReason GExecArtifactIdentity::validate(
    const GExecArtifactIdentity& cached,
    const GExecArtifactIdentity& current,
    std::string& reason_msg) {
    if (cached.format_version != kGExecVersionV1 || cached.format_version != current.format_version) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::MagicOrVersionMismatch);
        return GExecInvalidationReason::MagicOrVersionMismatch;
    }
    if (cached.gpack_manifest_sha256.empty() || cached.gpack_manifest_sha256 != current.gpack_manifest_sha256) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::GPackIdentityMismatch);
        return GExecInvalidationReason::GPackIdentityMismatch;
    }
    if (cached.core_abi_version != current.core_abi_version ||
        cached.module_abi_version != current.module_abi_version) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::CoreAbiMismatch);
        return GExecInvalidationReason::CoreAbiMismatch;
    }
    if (cached.runner_id != current.runner_id || cached.runner_version != current.runner_version) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::RunnerIdentityMismatch);
        return GExecInvalidationReason::RunnerIdentityMismatch;
    }
    auto sort_pkgs = [](std::vector<GExecBoundPackageDigest> v) {
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.package_id < b.package_id; });
        return v;
    };
    const auto p_cached = sort_pkgs(cached.bound_packages);
    const auto p_curr = sort_pkgs(current.bound_packages);
    if (p_cached.size() != p_curr.size()) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::ModuleOrKernelDigestMismatch);
        return GExecInvalidationReason::ModuleOrKernelDigestMismatch;
    }
    for (size_t i = 0; i < p_cached.size(); ++i) {
        if (p_cached[i].package_id != p_curr[i].package_id ||
            p_cached[i].version != p_curr[i].version ||
            p_cached[i].payload_sha256 != p_curr[i].payload_sha256) {
            reason_msg = invalidation_reason_to_string(GExecInvalidationReason::ModuleOrKernelDigestMismatch);
            return GExecInvalidationReason::ModuleOrKernelDigestMismatch;
        }
    }
    if (cached.hardware.cpu_vendor != current.hardware.cpu_vendor ||
        cached.hardware.cpu_feature_flags != current.hardware.cpu_feature_flags) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::CpuIsaMismatch);
        return GExecInvalidationReason::CpuIsaMismatch;
    }
    if (cached.hardware.numa_nodes != current.hardware.numa_nodes ||
        cached.hardware.cpu_physical_cores != current.hardware.cpu_physical_cores) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::NumaTopologyMismatch);
        return GExecInvalidationReason::NumaTopologyMismatch;
    }
    if (cached.hardware.gpu_count != current.hardware.gpu_count ||
        cached.hardware.gpu_cc_major != current.hardware.gpu_cc_major ||
        cached.hardware.gpu_cc_minor != current.hardware.gpu_cc_minor ||
        cached.hardware.gpu_name != current.hardware.gpu_name ||
        cached.hardware.gpu_vram_bytes != current.hardware.gpu_vram_bytes) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::GpuArchitectureMismatch);
        return GExecInvalidationReason::GpuArchitectureMismatch;
    }
    if (cached.hardware.cuda_runtime_version != current.hardware.cuda_runtime_version ||
        cached.hardware.cuda_driver_version != current.hardware.cuda_driver_version) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::CudaRuntimeOrDriverMismatch);
        return GExecInvalidationReason::CudaRuntimeOrDriverMismatch;
    }
    if (cached.kv_format != current.kv_format ||
        cached.kv_mode != current.kv_mode ||
        cached.max_context != current.max_context ||
        cached.prefill_batch != current.prefill_batch ||
        cached.mtp_spec_tokens != current.mtp_spec_tokens) {
        reason_msg = invalidation_reason_to_string(GExecInvalidationReason::ExecutionPlanMismatch);
        return GExecInvalidationReason::ExecutionPlanMismatch;
    }
    reason_msg.clear();
    return GExecInvalidationReason::None;
}

void pack_expert_gate_up_interleaved(
    const uint8_t* src_blob,
    uint8_t* dst_blob,
    size_t n_ff,
    size_t gate_row_bytes,
    size_t up_row_bytes,
    size_t down_bytes) {
    const uint8_t* gate_base = src_blob;
    const uint8_t* up_base = src_blob + n_ff * gate_row_bytes;
    const uint8_t* down_base = up_base + n_ff * up_row_bytes;
    uint8_t* out = dst_blob;
    for (size_t r = 0; r < n_ff; ++r) {
        std::memcpy(out, gate_base + r * gate_row_bytes, gate_row_bytes);
        out += gate_row_bytes;
        std::memcpy(out, up_base + r * up_row_bytes, up_row_bytes);
        out += up_row_bytes;
    }
    if (down_bytes > 0) {
        std::memcpy(out, down_base, down_bytes);
    }
}

void unpack_expert_gate_up_interleaved(
    const uint8_t* packed_blob,
    uint8_t* dst_blob,
    size_t n_ff,
    size_t gate_row_bytes,
    size_t up_row_bytes,
    size_t down_bytes) {
    uint8_t* gate_base = dst_blob;
    uint8_t* up_base = dst_blob + n_ff * gate_row_bytes;
    uint8_t* down_base = up_base + n_ff * up_row_bytes;
    const uint8_t* in = packed_blob;
    for (size_t r = 0; r < n_ff; ++r) {
        std::memcpy(gate_base + r * gate_row_bytes, in, gate_row_bytes);
        in += gate_row_bytes;
        std::memcpy(up_base + r * up_row_bytes, in, up_row_bytes);
        in += up_row_bytes;
    }
    if (down_bytes > 0) {
        std::memcpy(down_base, in, down_bytes);
    }
}

const GExecKernelTuneRecord* GExecAutotuneTable::find(
    uint32_t op_kind,
    const std::string& operation_name,
    GExecWorkloadPhase phase,
    int32_t quant_type,
    int64_t m_tokens) const {
    for (const auto& rec : records) {
        if (rec.op_kind == op_kind &&
            rec.operation_name == operation_name &&
            rec.phase == phase &&
            rec.quant_type == quant_type &&
            rec.m_tokens == m_tokens) {
            return &rec;
        }
    }
    return nullptr;
}

std::string GExecAutotuneTable::to_json() const {
    JsonValue root;
    std::string perr;
    JsonValue id_val;
    JsonValue::parse(identity.to_json(), id_val, perr);
    root["identity"] = std::move(id_val);

    JsonValue arr(std::vector<JsonValue>{});
    for (const auto& r : records) {
        JsonValue ro;
        ro["op_kind"] = static_cast<int64_t>(r.op_kind);
        ro["operation_name"] = r.operation_name;
        ro["phase"] = static_cast<int64_t>(r.phase);
        ro["backend"] = static_cast<int64_t>(r.backend);
        ro["quant_type"] = static_cast<int64_t>(r.quant_type);
        ro["m_tokens"] = r.m_tokens;
        ro["n_cols"] = r.n_cols;
        ro["k_cols"] = r.k_cols;
        ro["selected_impl_id"] = r.selected_impl_id;
        ro["selected_package_id"] = r.selected_package_id;
        ro["selected_impl_version"] = static_cast<int64_t>(r.selected_impl_version);
        ro["fallback_impl_id"] = r.fallback_impl_id;
        ro["median_latency_us"] = r.median_latency_us;
        ro["candidate_count"] = static_cast<int64_t>(r.candidate_count);
        arr.arr_val.push_back(std::move(ro));
    }
    root["records"] = std::move(arr);
    root["digest"] = models::Sha256::hash_string(root.dump());
    return root.dump_pretty();
}

bool GExecAutotuneTable::from_json(const std::string& json_str, GExecAutotuneTable& out, std::string& err) {
    JsonValue root;
    if (!JsonValue::parse(json_str, root, err) || !root.is_object()) {
        if (err.empty()) err = "malformed GExecAutotuneTable JSON";
        return false;
    }
    if (!root.contains("identity") || !root.contains("records") || !root.contains("digest")) {
        err = "GExecAutotuneTable missing required fields";
        return false;
    }
    const std::string recorded_digest = root["digest"].as_string();
    JsonValue check_root = root;
    erase_json_key(check_root, "digest");
    const std::string actual_digest = models::Sha256::hash_string(check_root.dump());
    if (recorded_digest != actual_digest) {
        err = invalidation_reason_to_string(GExecInvalidationReason::CorruptedDigest);
        return false;
    }
    out = GExecAutotuneTable{};
    if (!GExecArtifactIdentity::from_json(root["identity"].dump(), out.identity, err)) {
        return false;
    }
    if (root["records"].is_array()) {
        for (const auto& item : root["records"].arr_val) {
            if (!item.is_object()) continue;
            GExecKernelTuneRecord r;
            r.op_kind = static_cast<uint32_t>(item["op_kind"].as_int(0));
            r.operation_name = item["operation_name"].as_string();
            r.phase = static_cast<GExecWorkloadPhase>(item["phase"].as_int(0));
            r.backend = static_cast<uint32_t>(item["backend"].as_int(0));
            r.quant_type = static_cast<int32_t>(item["quant_type"].as_int(-1));
            r.m_tokens = item["m_tokens"].as_int(1);
            r.n_cols = item["n_cols"].as_int(0);
            r.k_cols = item["k_cols"].as_int(0);
            r.selected_impl_id = item["selected_impl_id"].as_string();
            r.selected_package_id = item["selected_package_id"].as_string();
            r.selected_impl_version = static_cast<uint32_t>(item["selected_impl_version"].as_int(0));
            r.fallback_impl_id = item["fallback_impl_id"].as_string();
            r.median_latency_us = item["median_latency_us"].as_double(0.0);
            r.candidate_count = static_cast<uint32_t>(item["candidate_count"].as_int(0));
            out.records.push_back(std::move(r));
        }
    }
    return true;
}

std::string GExecHeterogeneousPlan::to_json() const {
    JsonValue root;
    std::string perr;
    JsonValue id_val;
    JsonValue::parse(identity.to_json(), id_val, perr);
    root["identity"] = std::move(id_val);
    root["numa_policy"] = numa_policy;
    root["use_thp_hugepages"] = use_thp_hugepages;
    root["measured_formats"] = static_cast<int64_t>(measured_formats);
    root["calibrated_cpu_ms"] = calibrated_cpu_ms;
    root["calibrated_gpu_ms"] = calibrated_gpu_ms;
    root["calibrated_copy_ms"] = calibrated_copy_ms;
    root["expert_cache_slots"] = static_cast<int64_t>(expert_cache_slots);

    JsonValue gl(std::vector<JsonValue>{});
    for (uint8_t v : gpu_layers) gl.arr_val.push_back(JsonValue(static_cast<int64_t>(v)));
    root["gpu_layers"] = std::move(gl);

    JsonValue pn(std::vector<JsonValue>{});
    for (int32_t v : pcie_numerators) pn.arr_val.push_back(JsonValue(static_cast<int64_t>(v)));
    root["pcie_numerators"] = std::move(pn);

    root["digest"] = models::Sha256::hash_string(root.dump());
    return root.dump_pretty();
}

bool GExecHeterogeneousPlan::from_json(const std::string& json_str, GExecHeterogeneousPlan& out, std::string& err) {
    JsonValue root;
    if (!JsonValue::parse(json_str, root, err) || !root.is_object()) {
        if (err.empty()) err = "malformed GExecHeterogeneousPlan JSON";
        return false;
    }
    if (!root.contains("identity") || !root.contains("digest")) {
        err = "GExecHeterogeneousPlan missing required fields";
        return false;
    }
    const std::string recorded_digest = root["digest"].as_string();
    JsonValue check_root = root;
    erase_json_key(check_root, "digest");
    const std::string actual_digest = models::Sha256::hash_string(check_root.dump());
    if (recorded_digest != actual_digest) {
        err = invalidation_reason_to_string(GExecInvalidationReason::CorruptedDigest);
        return false;
    }
    out = GExecHeterogeneousPlan{};
    if (!GExecArtifactIdentity::from_json(root["identity"].dump(), out.identity, err)) {
        return false;
    }
    out.numa_policy = root["numa_policy"].as_string();
    out.use_thp_hugepages = root["use_thp_hugepages"].as_bool(true);
    out.measured_formats = static_cast<int32_t>(root["measured_formats"].as_int(0));
    out.calibrated_cpu_ms = root["calibrated_cpu_ms"].as_double(0.0);
    out.calibrated_gpu_ms = root["calibrated_gpu_ms"].as_double(0.0);
    out.calibrated_copy_ms = root["calibrated_copy_ms"].as_double(0.0);
    out.expert_cache_slots = static_cast<int32_t>(root["expert_cache_slots"].as_int(0));

    if (root.contains("gpu_layers") && root["gpu_layers"].is_array()) {
        for (const auto& v : root["gpu_layers"].arr_val) {
            out.gpu_layers.push_back(static_cast<uint8_t>(v.as_int(0)));
        }
    }
    if (root.contains("pcie_numerators") && root["pcie_numerators"].is_array()) {
        for (const auto& v : root["pcie_numerators"].arr_val) {
            out.pcie_numerators.push_back(static_cast<int32_t>(v.as_int(0)));
        }
    }
    return true;
}

std::string GExecCompilationKey::compute_key() const {
    std::ostringstream oss;
    oss << "cuda_rt=" << cuda_runtime_version
        << ";cuda_drv=" << cuda_driver_version
        << ";sm=" << gpu_cc_major << "." << gpu_cc_minor
        << ";pkg=" << kernel_package_sha256;
    return models::Sha256::hash_string(oss.str());
}

std::string GExecWorkspaceRecipe::to_json() const {
    JsonValue root;
    root["dense_arena_bytes"] = static_cast<int64_t>(dense_arena_bytes);
    root["native_dense_arena_bytes"] = static_cast<int64_t>(native_dense_arena_bytes);
    root["session_device_bytes"] = static_cast<int64_t>(session_device_bytes);
    root["session_host_kv_bytes"] = static_cast<int64_t>(session_host_kv_bytes);
    root["verifier_arena_bytes"] = static_cast<int64_t>(verifier_arena_bytes);
    root["prefill_arena_bytes"] = static_cast<int64_t>(prefill_arena_bytes);
    root["capture_commit_graph"] = capture_commit_graph;
    JsonValue cw(std::vector<JsonValue>{});
    for (int32_t w : capture_window_tokens) cw.arr_val.push_back(JsonValue(static_cast<int64_t>(w)));
    root["capture_window_tokens"] = std::move(cw);
    root["digest"] = models::Sha256::hash_string(root.dump());
    return root.dump_pretty();
}

bool GExecWorkspaceRecipe::from_json(const std::string& json_str, GExecWorkspaceRecipe& out, std::string& err) {
    JsonValue root;
    if (!JsonValue::parse(json_str, root, err) || !root.is_object()) {
        if (err.empty()) err = "malformed GExecWorkspaceRecipe JSON";
        return false;
    }
    if (!root.contains("digest")) {
        err = "GExecWorkspaceRecipe missing digest";
        return false;
    }
    const std::string recorded_digest = root["digest"].as_string();
    JsonValue check_root = root;
    erase_json_key(check_root, "digest");
    if (recorded_digest != models::Sha256::hash_string(check_root.dump())) {
        err = invalidation_reason_to_string(GExecInvalidationReason::CorruptedDigest);
        return false;
    }
    out = GExecWorkspaceRecipe{};
    out.dense_arena_bytes = static_cast<uint64_t>(root["dense_arena_bytes"].as_int(0));
    out.native_dense_arena_bytes = static_cast<uint64_t>(root["native_dense_arena_bytes"].as_int(0));
    out.session_device_bytes = static_cast<uint64_t>(root["session_device_bytes"].as_int(0));
    out.session_host_kv_bytes = static_cast<uint64_t>(root["session_host_kv_bytes"].as_int(0));
    out.verifier_arena_bytes = static_cast<uint64_t>(root["verifier_arena_bytes"].as_int(0));
    out.prefill_arena_bytes = static_cast<uint64_t>(root["prefill_arena_bytes"].as_int(0));
    out.capture_commit_graph = root["capture_commit_graph"].as_bool(true);
    if (root.contains("capture_window_tokens") && root["capture_window_tokens"].is_array()) {
        for (const auto& v : root["capture_window_tokens"].arr_val) {
            out.capture_window_tokens.push_back(static_cast<int32_t>(v.as_int(0)));
        }
    }
    return true;
}

std::string GExecCompilationProvenance::to_json() const {
    JsonValue root;
    JsonValue ko;
    ko["cuda_runtime_version"] = static_cast<int64_t>(key.cuda_runtime_version);
    ko["cuda_driver_version"] = static_cast<int64_t>(key.cuda_driver_version);
    ko["gpu_cc_major"] = static_cast<int64_t>(key.gpu_cc_major);
    ko["gpu_cc_minor"] = static_cast<int64_t>(key.gpu_cc_minor);
    ko["kernel_package_sha256"] = key.kernel_package_sha256;
    ko["computed_key"] = key.compute_key();
    root["key"] = std::move(ko);
    root["compiler_id"] = compiler_id;
    root["guild_version"] = guild_version;
    root["git_commit"] = git_commit;
    root["created_unix_seconds"] = created_unix_seconds;
    root["optimization_mode"] = optimization_mode;
    root["optimization_wall_ms"] = optimization_wall_ms;
    root["digest"] = models::Sha256::hash_string(root.dump());
    return root.dump_pretty();
}

bool GExecCompilationProvenance::from_json(const std::string& json_str, GExecCompilationProvenance& out, std::string& err) {
    JsonValue root;
    if (!JsonValue::parse(json_str, root, err) || !root.is_object()) {
        if (err.empty()) err = "malformed GExecCompilationProvenance JSON";
        return false;
    }
    if (!root.contains("digest")) {
        err = "GExecCompilationProvenance missing digest";
        return false;
    }
    const std::string recorded_digest = root["digest"].as_string();
    JsonValue check_root = root;
    erase_json_key(check_root, "digest");
    if (recorded_digest != models::Sha256::hash_string(check_root.dump())) {
        err = invalidation_reason_to_string(GExecInvalidationReason::CorruptedDigest);
        return false;
    }
    out = GExecCompilationProvenance{};
    if (root.contains("key") && root["key"].is_object()) {
        const auto& ko = root["key"];
        out.key.cuda_runtime_version = static_cast<int32_t>(ko["cuda_runtime_version"].as_int(0));
        out.key.cuda_driver_version = static_cast<int32_t>(ko["cuda_driver_version"].as_int(0));
        out.key.gpu_cc_major = static_cast<int32_t>(ko["gpu_cc_major"].as_int(0));
        out.key.gpu_cc_minor = static_cast<int32_t>(ko["gpu_cc_minor"].as_int(0));
        out.key.kernel_package_sha256 = ko["kernel_package_sha256"].as_string();
    }
    out.compiler_id = root["compiler_id"].as_string();
    out.guild_version = root["guild_version"].as_string();
    out.git_commit = root["git_commit"].as_string();
    out.created_unix_seconds = root["created_unix_seconds"].as_int(0);
    out.optimization_mode = root["optimization_mode"].as_string();
    out.optimization_wall_ms = root["optimization_wall_ms"].as_double(0.0);
    return true;
}

std::string GExecExpertPackMeta::to_json() const {
    JsonValue root;
    root["layout_kind"] = static_cast<int64_t>(layout_kind);
    root["embedded_blob_present"] = embedded_blob_present;
    root["fused_single_barrier_pipeline"] = fused_single_barrier_pipeline;
    root["n_layers"] = n_layers;
    root["n_expert"] = n_expert;
    root["total_expert_bytes"] = static_cast<int64_t>(total_expert_bytes);
    JsonValue arr(std::vector<JsonValue>{});
    for (const auto& s : layer_specs) {
        JsonValue so;
        so["layout_kind"] = static_cast<int64_t>(s.layout_kind);
        so["gu_type"] = static_cast<int64_t>(s.gu_type);
        so["d_type"] = static_cast<int64_t>(s.d_type);
        so["n_embd"] = s.n_embd;
        so["n_ff"] = s.n_ff;
        so["alignment"] = static_cast<int64_t>(s.alignment);
        arr.arr_val.push_back(std::move(so));
    }
    root["layer_specs"] = std::move(arr);
    root["digest"] = models::Sha256::hash_string(root.dump());
    return root.dump_pretty();
}

bool GExecExpertPackMeta::from_json(const std::string& json_str, GExecExpertPackMeta& out, std::string& err) {
    JsonValue root;
    if (!JsonValue::parse(json_str, root, err) || !root.is_object()) {
        if (err.empty()) err = "malformed GExecExpertPackMeta JSON";
        return false;
    }
    if (!root.contains("digest")) {
        err = "GExecExpertPackMeta missing digest";
        return false;
    }
    const std::string recorded_digest = root["digest"].as_string();
    JsonValue check_root = root;
    erase_json_key(check_root, "digest");
    if (recorded_digest != models::Sha256::hash_string(check_root.dump())) {
        err = invalidation_reason_to_string(GExecInvalidationReason::CorruptedDigest);
        return false;
    }
    out = GExecExpertPackMeta{};
    out.layout_kind = static_cast<ExpertPackLayoutKind>(root["layout_kind"].as_int(0));
    out.embedded_blob_present = root["embedded_blob_present"].as_bool(false);
    out.fused_single_barrier_pipeline = root["fused_single_barrier_pipeline"].as_bool(true);
    out.n_layers = root["n_layers"].as_int(0);
    out.n_expert = root["n_expert"].as_int(0);
    out.total_expert_bytes = static_cast<uint64_t>(root["total_expert_bytes"].as_int(0));
    if (root.contains("layer_specs") && root["layer_specs"].is_array()) {
        for (const auto& item : root["layer_specs"].arr_val) {
            if (!item.is_object()) continue;
            GExecExpertPackSpec s;
            s.layout_kind = static_cast<ExpertPackLayoutKind>(item["layout_kind"].as_int(0));
            s.gu_type = static_cast<int32_t>(item["gu_type"].as_int(-1));
            s.d_type = static_cast<int32_t>(item["d_type"].as_int(-1));
            s.n_embd = item["n_embd"].as_int(0);
            s.n_ff = item["n_ff"].as_int(0);
            s.alignment = static_cast<uint64_t>(item["alignment"].as_int(64));
            out.layer_specs.push_back(s);
        }
    }
    return true;
}

std::string GExecDiagnosticLog::to_json() const {
    JsonValue root;
    root["baseline_decode_ms"] = baseline_decode_ms;
    root["optimized_decode_ms"] = optimized_decode_ms;
    root["speedup_ratio"] = speedup_ratio;
    root["break_even_tokens"] = break_even_tokens;

    JsonValue notes_arr(std::vector<JsonValue>{});
    for (const auto& n : notes) notes_arr.arr_val.push_back(JsonValue(n));
    root["notes"] = std::move(notes_arr);

    JsonValue rej_arr(std::vector<JsonValue>{});
    for (const auto& r : rejected_candidates) {
        JsonValue ro;
        ro["op_kind"] = static_cast<int64_t>(r.op_kind);
        ro["operation_name"] = r.operation_name;
        ro["candidate_impl_id"] = r.candidate_impl_id;
        ro["stage"] = r.stage;
        ro["reason"] = r.reason;
        ro["measured_max_abs_err"] = r.measured_max_abs_err;
        ro["measured_latency_us"] = r.measured_latency_us;
        rej_arr.arr_val.push_back(std::move(ro));
    }
    root["rejected_candidates"] = std::move(rej_arr);
    root["digest"] = models::Sha256::hash_string(root.dump());
    return root.dump_pretty();
}

bool GExecDiagnosticLog::from_json(const std::string& json_str, GExecDiagnosticLog& out, std::string& err) {
    JsonValue root;
    if (!JsonValue::parse(json_str, root, err) || !root.is_object()) {
        if (err.empty()) err = "malformed GExecDiagnosticLog JSON";
        return false;
    }
    if (!root.contains("digest")) {
        err = "GExecDiagnosticLog missing digest";
        return false;
    }
    const std::string recorded_digest = root["digest"].as_string();
    JsonValue check_root = root;
    erase_json_key(check_root, "digest");
    if (recorded_digest != models::Sha256::hash_string(check_root.dump())) {
        err = invalidation_reason_to_string(GExecInvalidationReason::CorruptedDigest);
        return false;
    }
    out = GExecDiagnosticLog{};
    out.baseline_decode_ms = root["baseline_decode_ms"].as_double(0.0);
    out.optimized_decode_ms = root["optimized_decode_ms"].as_double(0.0);
    out.speedup_ratio = root["speedup_ratio"].as_double(1.0);
    out.break_even_tokens = root["break_even_tokens"].as_double(0.0);
    if (root.contains("notes") && root["notes"].is_array()) {
        for (const auto& n : root["notes"].arr_val) out.notes.push_back(n.as_string());
    }
    if (root.contains("rejected_candidates") && root["rejected_candidates"].is_array()) {
        for (const auto& item : root["rejected_candidates"].arr_val) {
            if (!item.is_object()) continue;
            GExecCandidateRejection r;
            r.op_kind = static_cast<uint32_t>(item["op_kind"].as_int(0));
            r.operation_name = item["operation_name"].as_string();
            r.candidate_impl_id = item["candidate_impl_id"].as_string();
            r.stage = item["stage"].as_string();
            r.reason = item["reason"].as_string();
            r.measured_max_abs_err = item["measured_max_abs_err"].as_double(0.0);
            r.measured_latency_us = item["measured_latency_us"].as_double(0.0);
            out.rejected_candidates.push_back(std::move(r));
        }
    }
    return true;
}

const char* section_kind_to_string(GExecSectionKind kind) noexcept {
    switch (kind) {
        case GExecSectionKind::IdentityAndManifest: return "IdentityAndManifest";
        case GExecSectionKind::AutotuneTable: return "AutotuneTable";
        case GExecSectionKind::HeterogeneousPlan: return "HeterogeneousPlan";
        case GExecSectionKind::WorkspaceRecipe: return "WorkspaceRecipe";
        case GExecSectionKind::CompilationProvenance: return "CompilationProvenance";
        case GExecSectionKind::ExpertPackMeta: return "ExpertPackMeta";
        case GExecSectionKind::PackedExpertsBlob: return "PackedExpertsBlob";
        case GExecSectionKind::DiagnosticLog: return "DiagnosticLog";
        default: return "Invalid";
    }
}

namespace {

bool hex_to_raw_sha256(const std::string& hex, uint8_t out[32]) noexcept {
    if (hex.size() != 64) return false;
    for (size_t i = 0; i < 32; ++i) {
        auto nyb = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        const int hi = nyb(hex[2 * i]);
        const int lo = nyb(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

std::string raw_sha256_to_hex(const uint8_t raw[32]) {
    static const char* kHex = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        out[2 * i + 0] = kHex[(raw[i] >> 4) & 0xF];
        out[2 * i + 1] = kHex[raw[i] & 0xF];
    }
    return out;
}

void compute_raw_sha256(const void* data, size_t len, uint8_t out[32]) {
    models::Sha256 hasher;
    hasher.update(data, len);
    hex_to_raw_sha256(hasher.finalize_hex(), out);
}

void compute_header_and_table_sha256(const GExecFileHeader& hdr,
                                     const GExecSectionEntry* entries,
                                     size_t count,
                                     uint8_t out[32]) {
    models::Sha256 hasher;
    hasher.update(&hdr, offsetof(GExecFileHeader, header_and_table_sha256));
    if (entries && count > 0) {
        hasher.update(entries, count * sizeof(GExecSectionEntry));
    }
    hex_to_raw_sha256(hasher.finalize_hex(), out);
}

bool is_power_of_two_u64(uint64_t x) noexcept {
    return x != 0 && (x & (x - 1)) == 0;
}

uint64_t align_up_u64(uint64_t v, uint64_t a) noexcept {
    if (a <= 1) return v;
    return (v + a - 1) & ~(a - 1);
}

} // namespace

GExecReader::~GExecReader() {
    close();
}

GExecReader::GExecReader(GExecReader&& other) noexcept {
    *this = std::move(other);
}

GExecReader& GExecReader::operator=(GExecReader&& other) noexcept {
    if (this == &other) return *this;
    close();
    path_ = std::move(other.path_);
    mapped_base_ = other.mapped_base_;
    mapped_bytes_ = other.mapped_bytes_;
    fd_ = other.fd_;
    header_ = other.header_;
    sections_ = std::move(other.sections_);
    identity_ = std::move(other.identity_);
    autotune_ = std::move(other.autotune_);
    hplan_ = std::move(other.hplan_);
    workspace_ = std::move(other.workspace_);
    provenance_ = std::move(other.provenance_);
    pack_meta_ = std::move(other.pack_meta_);
    diag_log_ = std::move(other.diag_log_);

    other.mapped_base_ = nullptr;
    other.mapped_bytes_ = 0;
    other.fd_ = -1;
    return *this;
}

void GExecReader::close() {
#if defined(_WIN32)
    if (mapped_base_) UnmapViewOfFile(mapped_base_);
#else
    if (mapped_base_ != nullptr) {
        munmap(const_cast<uint8_t*>(mapped_base_), static_cast<size_t>(mapped_bytes_));
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
#endif
    mapped_base_ = nullptr;
    mapped_bytes_ = 0;
    fd_ = -1;
    path_.clear();
    header_ = GExecFileHeader{};
    sections_.clear();
    identity_ = GExecArtifactIdentity{};
    autotune_ = GExecAutotuneTable{};
    hplan_ = GExecHeterogeneousPlan{};
    workspace_ = GExecWorkspaceRecipe{};
    provenance_ = GExecCompilationProvenance{};
    pack_meta_ = GExecExpertPackMeta{};
    diag_log_ = GExecDiagnosticLog{};
}

bool GExecReader::open(const std::string& path,
                       std::string& err,
                       GExecInvalidationReason* out_reason,
                       bool verify_expert_blob_digest) {
    close();
    auto fail = [&](GExecInvalidationReason r, const std::string& msg) {
        if (out_reason) *out_reason = r;
        err = msg;
        close();
        return false;
    };

#if !defined(_WIN32)
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return fail(GExecInvalidationReason::MagicOrVersionMismatch,
                    "cannot open GEXEC file: " + path);
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(GExecFileHeader))) {
        ::close(fd);
        return fail(GExecInvalidationReason::CorruptedDigest,
                    "GEXEC file is truncated or smaller than 64-byte header: " + path);
    }
    const uint64_t fsize = static_cast<uint64_t>(st.st_size);
    void* view = mmap(nullptr, static_cast<size_t>(fsize), PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) {
        ::close(fd);
        return fail(GExecInvalidationReason::MagicOrVersionMismatch,
                    "mmap failed on GEXEC file: " + path);
    }
    fd_ = fd;
    mapped_base_ = static_cast<const uint8_t*>(view);
    mapped_bytes_ = fsize;
#else
    const auto wide_path = std::filesystem::u8path(path).wstring();
    HANDLE file = CreateFileW(wide_path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        return fail(GExecInvalidationReason::MagicOrVersionMismatch, "cannot open GEXEC file: " + path);
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size) || size.QuadPart < static_cast<LONGLONG>(sizeof(GExecFileHeader)) ||
        static_cast<uint64_t>(size.QuadPart) > SIZE_MAX) {
        CloseHandle(file);
        return fail(GExecInvalidationReason::CorruptedDigest, "invalid GEXEC file size");
    }
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    const void* view = mapping ? MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (mapping) CloseHandle(mapping);
    CloseHandle(file);
    if (!view) return fail(GExecInvalidationReason::MagicOrVersionMismatch, "cannot map GEXEC file: " + path);
    mapped_base_ = static_cast<const uint8_t*>(view);
    mapped_bytes_ = static_cast<uint64_t>(size.QuadPart);
#endif

    path_ = path;
    std::memcpy(&header_, mapped_base_, sizeof(GExecFileHeader));

    if (header_.magic != kGExecMagic ||
        header_.format_version != kGExecVersionV1 ||
        header_.endianness_tag != kGExecLittleEndianTag) {
        return fail(GExecInvalidationReason::MagicOrVersionMismatch,
                    invalidation_reason_to_string(GExecInvalidationReason::MagicOrVersionMismatch));
    }
    if (header_.total_file_bytes != mapped_bytes_) {
        return fail(GExecInvalidationReason::CorruptedDigest,
                    "GEXEC file size does not match header total_file_bytes (truncated or modified)");
    }
    if (header_.section_count == 0 || header_.section_count > 64) {
        return fail(GExecInvalidationReason::CorruptedDigest,
                    "GEXEC header has invalid section_count");
    }
    const uint64_t table_bytes = static_cast<uint64_t>(header_.section_count) * sizeof(GExecSectionEntry);
    if (header_.section_table_offset < sizeof(GExecFileHeader) ||
        header_.section_table_offset > mapped_bytes_ ||
        table_bytes > mapped_bytes_ - header_.section_table_offset) {
        return fail(GExecInvalidationReason::CorruptedDigest,
                    "GEXEC section directory extends past end of file");
    }

    sections_.resize(header_.section_count);
    std::memcpy(sections_.data(),
                mapped_base_ + static_cast<size_t>(header_.section_table_offset),
                 static_cast<size_t>(table_bytes));

    uint8_t expected_hdr_hash[32] = {};
    compute_header_and_table_sha256(header_, sections_.data(), sections_.size(), expected_hdr_hash);
    if (std::memcmp(expected_hdr_hash, header_.header_and_table_sha256, 32) != 0) {
        return fail(GExecInvalidationReason::CorruptedDigest,
                    "GEXEC header/section-table SHA-256 mismatch");
    }

    const uint64_t min_data_offset = header_.section_table_offset + table_bytes;
    for (size_t i = 0; i < sections_.size(); ++i) {
        const auto& s = sections_[i];
        if (s.kind == static_cast<uint32_t>(GExecSectionKind::Invalid) ||
            s.kind > static_cast<uint32_t>(GExecSectionKind::DiagnosticLog)) {
            return fail(GExecInvalidationReason::CorruptedDigest,
                        "GEXEC section directory contains unknown section kind");
        }
        if (!is_power_of_two_u64(s.alignment) || (s.offset % s.alignment) != 0) {
            return fail(GExecInvalidationReason::CorruptedDigest,
                        "GEXEC section violates declared alignment");
        }
        if ((s.flags & kGExecSectionFlagPageAligned) != 0 && (s.offset % 4096u) != 0) {
            return fail(GExecInvalidationReason::CorruptedDigest,
                        "GEXEC page-aligned section is not 4096-byte aligned");
        }
        if (s.offset < min_data_offset || s.offset > mapped_bytes_ ||
            s.byte_size > mapped_bytes_ - s.offset) {
            return fail(GExecInvalidationReason::CorruptedDigest,
                        "GEXEC section payload extends out of file bounds");
        }
        for (size_t j = 0; j < i; ++j) {
            if (sections_[j].kind == s.kind) {
                return fail(GExecInvalidationReason::CorruptedDigest,
                            "GEXEC section directory contains duplicate section kind");
            }
            const uint64_t a0 = sections_[j].offset, a1 = a0 + sections_[j].byte_size;
            const uint64_t b0 = s.offset, b1 = b0 + s.byte_size;
            if (std::max(a0, b0) < std::min(a1, b1)) {
                return fail(GExecInvalidationReason::CorruptedDigest,
                            "GEXEC sections overlap in file");
            }
        }
        const auto skind = static_cast<GExecSectionKind>(s.kind);
        if (skind != GExecSectionKind::PackedExpertsBlob && s.byte_size > (16ull << 20))
            return fail(GExecInvalidationReason::CorruptedDigest, "GEXEC metadata section exceeds 16 MiB");
        if (skind != GExecSectionKind::PackedExpertsBlob || verify_expert_blob_digest) {
            uint8_t actual_sha[32] = {};
            compute_raw_sha256(mapped_base_ + static_cast<size_t>(s.offset),
                               static_cast<size_t>(s.byte_size),
                               actual_sha);
            if (std::memcmp(actual_sha, s.sha256, 32) != 0) {
                return fail(GExecInvalidationReason::CorruptedDigest,
                            std::string("GEXEC section SHA-256 mismatch in ") + section_kind_to_string(skind));
            }
        }
    }

    if (!has_section(GExecSectionKind::IdentityAndManifest)) {
        return fail(GExecInvalidationReason::CorruptedDigest,
                    "GEXEC container is missing required IdentityAndManifest section");
    }

    auto read_str = [&](GExecSectionKind k) -> std::string {
        uint64_t sz = 0;
        const uint8_t* ptr = section_data(k, &sz);
        if (!ptr || sz == 0) return {};
        return std::string(reinterpret_cast<const char*>(ptr), static_cast<size_t>(sz));
    };

    std::string parse_err;
    if (!GExecArtifactIdentity::from_json(read_str(GExecSectionKind::IdentityAndManifest), identity_, parse_err)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "invalid IdentityAndManifest: " + parse_err);
    }
    if (has_section(GExecSectionKind::AutotuneTable) &&
        !GExecAutotuneTable::from_json(read_str(GExecSectionKind::AutotuneTable), autotune_, parse_err)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "invalid AutotuneTable: " + parse_err);
    }
    if (has_section(GExecSectionKind::HeterogeneousPlan) &&
        !GExecHeterogeneousPlan::from_json(read_str(GExecSectionKind::HeterogeneousPlan), hplan_, parse_err)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "invalid HeterogeneousPlan: " + parse_err);
    }
    if (has_section(GExecSectionKind::WorkspaceRecipe) &&
        !GExecWorkspaceRecipe::from_json(read_str(GExecSectionKind::WorkspaceRecipe), workspace_, parse_err)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "invalid WorkspaceRecipe: " + parse_err);
    }
    if (has_section(GExecSectionKind::CompilationProvenance) &&
        !GExecCompilationProvenance::from_json(read_str(GExecSectionKind::CompilationProvenance), provenance_, parse_err)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "invalid CompilationProvenance: " + parse_err);
    }
    if (has_section(GExecSectionKind::ExpertPackMeta) &&
        !GExecExpertPackMeta::from_json(read_str(GExecSectionKind::ExpertPackMeta), pack_meta_, parse_err)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "invalid ExpertPackMeta: " + parse_err);
    }
    if (has_section(GExecSectionKind::DiagnosticLog) &&
        !GExecDiagnosticLog::from_json(read_str(GExecSectionKind::DiagnosticLog), diag_log_, parse_err)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "invalid DiagnosticLog: " + parse_err);
    }

    const auto key = identity_.compute_cache_key();
    if ((has_section(GExecSectionKind::AutotuneTable) && autotune_.identity.compute_cache_key() != key) ||
        (has_section(GExecSectionKind::HeterogeneousPlan) && hplan_.identity.compute_cache_key() != key)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "GEXEC section identity differs from manifest");
    }
    const auto* blob = find_section(GExecSectionKind::PackedExpertsBlob);
    if (blob && (!has_section(GExecSectionKind::ExpertPackMeta) || !pack_meta_.embedded_blob_present ||
                 blob->byte_size != pack_meta_.total_expert_bytes || blob->offset % 4096 != 0)) {
        return fail(GExecInvalidationReason::CorruptedDigest, "GEXEC packed blob does not match layout metadata");
    }
    if (pack_meta_.embedded_blob_present && !blob)
        return fail(GExecInvalidationReason::CorruptedDigest, "GEXEC embedded experts section is missing");

    if (out_reason) *out_reason = GExecInvalidationReason::None;
    err.clear();
    return true;
}

bool GExecReader::validate_against(const GExecArtifactIdentity& current_identity,
                                   std::string& reason_msg,
                                   GExecInvalidationReason* out_reason) const {
    if (!is_open()) {
        reason_msg = "GEXEC reader is not open";
        if (out_reason) *out_reason = GExecInvalidationReason::MagicOrVersionMismatch;
        return false;
    }
    const auto r = GExecArtifactIdentity::validate(identity_, current_identity, reason_msg);
    if (out_reason) *out_reason = r;
    return r == GExecInvalidationReason::None;
}

bool GExecReader::verify_section_digest(GExecSectionKind kind, std::string& err) const {
    if (!is_open()) {
        err = "GEXEC reader is not open";
        return false;
    }
    const auto* sec = find_section(kind);
    if (!sec) {
        err = std::string("section not found: ") + section_kind_to_string(kind);
        return false;
    }
    uint8_t actual_sha[32] = {};
    compute_raw_sha256(mapped_base_ + static_cast<size_t>(sec->offset),
                       static_cast<size_t>(sec->byte_size),
                       actual_sha);
    if (std::memcmp(actual_sha, sec->sha256, 32) != 0) {
        err = std::string("SHA-256 digest mismatch in section ") + section_kind_to_string(kind);
        return false;
    }
    err.clear();
    return true;
}

bool GExecReader::verify_all_digests(std::string& err) const {
    if (!is_open()) {
        err = "GEXEC reader is not open";
        return false;
    }
    for (const auto& s : sections_) {
        if (!verify_section_digest(static_cast<GExecSectionKind>(s.kind), err)) return false;
    }
    err.clear();
    return true;
}

bool GExecReader::has_section(GExecSectionKind kind) const noexcept {
    return find_section(kind) != nullptr;
}

const GExecSectionEntry* GExecReader::find_section(GExecSectionKind kind) const noexcept {
    const uint32_t k = static_cast<uint32_t>(kind);
    for (const auto& s : sections_) {
        if (s.kind == k) return &s;
    }
    return nullptr;
}

const uint8_t* GExecReader::section_data(GExecSectionKind kind, uint64_t* out_size) const noexcept {
    if (!is_open()) return nullptr;
    const auto* sec = find_section(kind);
    if (!sec) return nullptr;
    if (out_size) *out_size = sec->byte_size;
    return mapped_base_ + static_cast<size_t>(sec->offset);
}

const uint8_t* GExecReader::packed_experts_blob(uint64_t* out_bytes) const noexcept {
    return section_data(GExecSectionKind::PackedExpertsBlob, out_bytes);
}

std::string GExecReader::inspect_json() const {
    JsonValue root;
    root["path"] = path_;
    root["magic"] = "GEXC";
    root["format_version"] = static_cast<int64_t>(header_.format_version);
    root["flags"] = static_cast<int64_t>(header_.flags);
    root["is_partial"] = is_partial();
    root["total_file_bytes"] = static_cast<int64_t>(header_.total_file_bytes);
    root["header_and_table_sha256"] = raw_sha256_to_hex(header_.header_and_table_sha256);
    root["cache_key"] = identity_.compute_cache_key();

    JsonValue sec_arr(std::vector<JsonValue>{});
    for (const auto& s : sections_) {
        JsonValue so;
        so["kind"] = static_cast<int64_t>(s.kind);
        so["name"] = section_kind_to_string(static_cast<GExecSectionKind>(s.kind));
        so["flags"] = static_cast<int64_t>(s.flags);
        so["offset"] = static_cast<int64_t>(s.offset);
        so["byte_size"] = static_cast<int64_t>(s.byte_size);
        so["alignment"] = static_cast<int64_t>(s.alignment);
        so["sha256"] = raw_sha256_to_hex(s.sha256);
        sec_arr.arr_val.push_back(std::move(so));
    }
    root["sections"] = std::move(sec_arr);

    std::string perr;
    auto embed_json = [&](const char* key, const std::string& jstr) {
        JsonValue v;
        if (JsonValue::parse(jstr, v, perr)) root[key] = std::move(v);
    };
    embed_json("identity", identity_.to_json());
    if (has_section(GExecSectionKind::AutotuneTable)) embed_json("autotune_table", autotune_.to_json());
    if (has_section(GExecSectionKind::HeterogeneousPlan)) embed_json("heterogeneous_plan", hplan_.to_json());
    if (has_section(GExecSectionKind::WorkspaceRecipe)) embed_json("workspace_recipe", workspace_.to_json());
    if (has_section(GExecSectionKind::CompilationProvenance)) embed_json("provenance", provenance_.to_json());
    if (has_section(GExecSectionKind::ExpertPackMeta)) embed_json("expert_pack_meta", pack_meta_.to_json());
    if (has_section(GExecSectionKind::DiagnosticLog)) embed_json("diagnostic_log", diag_log_.to_json());
    return root.dump_pretty();
}

void GExecBuilder::upsert_section(GExecSectionKind kind, uint32_t flags, uint64_t alignment, std::vector<uint8_t> data) {
    for (auto& s : sections_) {
        if (s.kind == kind) {
            s.flags = flags;
            s.alignment = alignment;
            s.data = std::move(data);
            s.stream_bytes = 0;
            s.read = {};
            return;
        }
    }
    PendingSection ps;
    ps.kind = kind;
    ps.flags = flags;
    ps.alignment = alignment;
    ps.data = std::move(data);
    sections_.push_back(std::move(ps));
}

void GExecBuilder::set_identity(const GExecArtifactIdentity& id) {
    const std::string s = id.to_json();
    upsert_section(GExecSectionKind::IdentityAndManifest,
                   kGExecSectionFlagRequired,
                   64,
                   std::vector<uint8_t>(s.begin(), s.end()));
}

void GExecBuilder::set_autotune_table(const GExecAutotuneTable& table) {
    const std::string s = table.to_json();
    upsert_section(GExecSectionKind::AutotuneTable,
                   kGExecSectionFlagNone,
                   64,
                   std::vector<uint8_t>(s.begin(), s.end()));
}

void GExecBuilder::set_heterogeneous_plan(const GExecHeterogeneousPlan& plan) {
    const std::string s = plan.to_json();
    upsert_section(GExecSectionKind::HeterogeneousPlan,
                   kGExecSectionFlagNone,
                   64,
                   std::vector<uint8_t>(s.begin(), s.end()));
}

void GExecBuilder::set_workspace_recipe(const GExecWorkspaceRecipe& recipe) {
    const std::string s = recipe.to_json();
    upsert_section(GExecSectionKind::WorkspaceRecipe,
                   kGExecSectionFlagNone,
                   64,
                   std::vector<uint8_t>(s.begin(), s.end()));
}

void GExecBuilder::set_provenance(const GExecCompilationProvenance& prov) {
    const std::string s = prov.to_json();
    upsert_section(GExecSectionKind::CompilationProvenance,
                   kGExecSectionFlagNone,
                   64,
                   std::vector<uint8_t>(s.begin(), s.end()));
}

void GExecBuilder::set_expert_pack_meta(const GExecExpertPackMeta& meta) {
    const std::string s = meta.to_json();
    if (meta.fused_single_barrier_pipeline) flags_ |= kGExecFlagFusedPipeline;
    else flags_ &= ~kGExecFlagFusedPipeline;
    upsert_section(GExecSectionKind::ExpertPackMeta,
                   kGExecSectionFlagNone,
                   64,
                   std::vector<uint8_t>(s.begin(), s.end()));
}

void GExecBuilder::set_diagnostic_log(const GExecDiagnosticLog& log) {
    const std::string s = log.to_json();
    upsert_section(GExecSectionKind::DiagnosticLog,
                   kGExecSectionFlagNone,
                   64,
                   std::vector<uint8_t>(s.begin(), s.end()));
}

void GExecBuilder::set_packed_experts_blob(std::vector<uint8_t> blob, uint64_t alignment) {
    flags_ |= kGExecFlagHasEmbeddedExperts;
    upsert_section(GExecSectionKind::PackedExpertsBlob,
                   kGExecSectionFlagPageAligned,
                   std::max<uint64_t>(4096, alignment),
                   std::move(blob));
}

bool GExecBuilder::import_from(const GExecReader& reader, std::string& err, bool stream_experts) {
    if (!reader.is_open()) {
        err = "cannot import from closed GExecReader";
        return false;
    }
    flags_ = reader.header().flags;
    sections_.clear();
    for (const auto& sec : reader.sections()) {
        const auto kind = static_cast<GExecSectionKind>(sec.kind);
        uint64_t sz = 0;
        const uint8_t* ptr = reader.section_data(kind, &sz);
        if (!ptr) {
            err = "missing section data while importing GExecReader";
            return false;
        }
        if (stream_experts && kind == GExecSectionKind::PackedExpertsBlob) {
            set_packed_experts_stream(sz, [ptr, sz](uint64_t offset, uint8_t* dst, size_t count, std::string& error) {
                if (offset > sz || count > sz - offset) {
                    error = "imported expert blob read out of bounds";
                    return false;
                }
                std::memcpy(dst, ptr + offset, count);
                return true;
            });
            sections_.back().flags = sec.flags;
            sections_.back().alignment = sec.alignment;
        } else {
            upsert_section(kind, sec.flags, sec.alignment, std::vector<uint8_t>(ptr, ptr + sz));
        }
    }
    return true;
}

void GExecBuilder::set_packed_experts_stream(uint64_t bytes, BlobReader read) {
    set_packed_experts_blob({});
    for (auto& s : sections_) {
        if (s.kind == GExecSectionKind::PackedExpertsBlob) {
            s.stream_bytes = bytes;
            s.read = std::move(read);
            break;
        }
    }
}

bool GExecBuilder::write_file(const std::string& dest_path, std::string& err, uint64_t max_disk_bytes) const {
    if (sections_.empty()) {
        err = "cannot write empty GEXEC container";
        return false;
    }
    bool has_identity = false;
    for (const auto& s : sections_) {
        if (s.kind == GExecSectionKind::IdentityAndManifest) has_identity = true;
    }
    if (!has_identity) {
        err = "GEXEC container requires IdentityAndManifest section";
        return false;
    }

    // Order sections deterministically by kind (PackedExpertsBlob last so metadata is compact at the front)
    std::vector<const PendingSection*> ordered;
    ordered.reserve(sections_.size());
    for (const auto& s : sections_) ordered.push_back(&s);
    std::sort(ordered.begin(), ordered.end(), [](const PendingSection* a, const PendingSection* b) {
        const bool a_blob = (a->kind == GExecSectionKind::PackedExpertsBlob);
        const bool b_blob = (b->kind == GExecSectionKind::PackedExpertsBlob);
        if (a_blob != b_blob) return !a_blob;
        return static_cast<uint32_t>(a->kind) < static_cast<uint32_t>(b->kind);
    });

    GExecFileHeader hdr{};
    hdr.magic = kGExecMagic;
    hdr.format_version = kGExecVersionV1;
    hdr.endianness_tag = kGExecLittleEndianTag;
    hdr.flags = flags_;
    hdr.section_count = static_cast<uint32_t>(ordered.size());
    hdr.section_table_offset = sizeof(GExecFileHeader);

    std::vector<GExecSectionEntry> entries(ordered.size());
    std::vector<uint8_t> stream_buffer(1u << 20);
    uint64_t cursor = sizeof(GExecFileHeader) + ordered.size() * sizeof(GExecSectionEntry);
    for (size_t i = 0; i < ordered.size(); ++i) {
        const auto* ps = ordered[i];
        const uint64_t align = is_power_of_two_u64(ps->alignment) ? ps->alignment : 64;
        cursor = align_up_u64(cursor, align);
        entries[i].kind = static_cast<uint32_t>(ps->kind);
        entries[i].flags = ps->flags;
        entries[i].offset = cursor;
        entries[i].byte_size = ps->read ? ps->stream_bytes : ps->data.size();
        entries[i].alignment = align;
        if (entries[i].byte_size > UINT64_MAX - cursor) {
            err = "GEXEC section size overflow";
            return false;
        }
        cursor += entries[i].byte_size;
        if (max_disk_bytes && cursor > max_disk_bytes) {
            err = "GEXEC artifact exceeds configured disk budget";
            return false;
        }
        if (ps->read) {
            models::Sha256 digest;
            for (uint64_t offset = 0; offset < ps->stream_bytes;) {
                const auto count = static_cast<size_t>(std::min<uint64_t>(stream_buffer.size(), ps->stream_bytes - offset));
                if (!ps->read(offset, stream_buffer.data(), count, err)) return false;
                digest.update(stream_buffer.data(), count);
                offset += count;
            }
            hex_to_raw_sha256(digest.finalize_hex(), entries[i].sha256);
        } else compute_raw_sha256(ps->data.data(), ps->data.size(), entries[i].sha256);
    }
    hdr.total_file_bytes = cursor;
    compute_header_and_table_sha256(hdr, entries.data(), entries.size(), hdr.header_and_table_sha256);

    std::error_code ec;
    const std::filesystem::path target(dest_path);
    if (target.has_parent_path()) {
        std::filesystem::create_directories(target.parent_path(), ec);
    }
    const auto space = std::filesystem::space(target.has_parent_path() ? target.parent_path() : std::filesystem::path("."), ec);
    if (ec || space.available < hdr.total_file_bytes) {
        err = "insufficient free disk space for atomic GEXEC installation";
        return false;
    }
    static std::atomic<uint64_t> tmp_seq{0};
    const uint64_t seq = ++tmp_seq;
#if defined(_WIN32)
    const auto pid = GetCurrentProcessId();
#else
    const auto pid = ::getpid();
#endif
    const std::string tmp_path = dest_path + ".tmp." + std::to_string(pid) + "." + std::to_string(seq);

#if !defined(_WIN32)
    const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) {
        err = "cannot create temporary GEXEC file: " + tmp_path;
        return false;
    }
    auto write_all = [&](const void* buf, size_t len) -> bool {
        const uint8_t* p = static_cast<const uint8_t*>(buf);
        size_t rem = len;
        while (rem > 0) {
            const ssize_t nw = ::write(fd, p, std::min<size_t>(rem, 1ull << 26));
            if (nw < 0 && errno == EINTR) continue;
            if (nw <= 0) return false;
            p += nw;
            rem -= static_cast<size_t>(nw);
        }
        return true;
    };
#else
    const auto wide_tmp = std::filesystem::u8path(tmp_path).wstring();
    HANDLE fd = CreateFileW(wide_tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fd == INVALID_HANDLE_VALUE) {
        err = "cannot create temporary GEXEC file: " + tmp_path;
        return false;
    }
    auto write_all = [&](const void* buf, size_t len) {
        const auto* p = static_cast<const uint8_t*>(buf);
        while (len) {
            DWORD n = 0;
            if (!WriteFile(fd, p, static_cast<DWORD>(std::min<size_t>(len, 1ull << 26)), &n, nullptr) || !n) return false;
            p += n;
            len -= n;
        }
        return true;
    };
#endif
    bool ok = write_all(&hdr, sizeof(hdr)) &&
              write_all(entries.data(), entries.size() * sizeof(GExecSectionEntry));
    uint64_t written = sizeof(hdr) + entries.size() * sizeof(GExecSectionEntry);
    std::vector<uint8_t> zeros(4096, 0);
    for (size_t i = 0; ok && i < ordered.size(); ++i) {
        while (ok && written < entries[i].offset) {
            const size_t pad = static_cast<size_t>(std::min<uint64_t>(zeros.size(), entries[i].offset - written));
            ok = write_all(zeros.data(), pad);
            written += pad;
        }
        if (ok && ordered[i]->read) {
            models::Sha256 digest;
            for (uint64_t offset = 0; ok && offset < ordered[i]->stream_bytes;) {
                const auto count = static_cast<size_t>(std::min<uint64_t>(stream_buffer.size(), ordered[i]->stream_bytes - offset));
                ok = ordered[i]->read(offset, stream_buffer.data(), count, err) && write_all(stream_buffer.data(), count);
                if (ok) digest.update(stream_buffer.data(), count);
                offset += count;
                written += count;
            }
            uint8_t actual[32];
            hex_to_raw_sha256(digest.finalize_hex(), actual);
            if (std::memcmp(actual, entries[i].sha256, 32) != 0) ok = false;
        } else if (ok && !ordered[i]->data.empty()) {
            ok = write_all(ordered[i]->data.data(), ordered[i]->data.size());
            written += ordered[i]->data.size();
        }
    }
#if defined(_WIN32)
    if (ok && !FlushFileBuffers(fd)) ok = false;
    if (!CloseHandle(fd)) ok = false;
#else
    if (ok && ::fsync(fd) != 0) ok = false;
    if (::close(fd) != 0) ok = false;
#endif
    if (!ok) {
        std::filesystem::remove(tmp_path, ec);
        err = "failed writing or fsyncing temporary GEXEC file: " + tmp_path;
        return false;
    }
#if defined(_WIN32)
    const auto wide_dest = std::filesystem::u8path(dest_path).wstring();
    const bool renamed = MoveFileExW(wide_tmp.c_str(), wide_dest.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
#else
    const bool renamed = ::rename(tmp_path.c_str(), dest_path.c_str()) == 0;
#endif
    if (!renamed) {
        std::filesystem::remove(tmp_path, ec);
        err = "failed atomic rename of GEXEC file to: " + dest_path;
        return false;
    }
#if !defined(_WIN32)
    {
        const auto parent = target.has_parent_path() ? target.parent_path() : std::filesystem::path(".");
        const int dir_fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
        if (dir_fd >= 0) {
            (void) ::fsync(dir_fd);
            ::close(dir_fd);
        }
    }
#endif
    err.clear();
    return true;
}

} // namespace guild::gexec
