#include "guild/gexec/gexec_contract.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/models/sha256.hpp"
#include "guild/server/json.hpp"

#if defined(GUILD_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

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

} // namespace guild::gexec
