// src/abi/gkernel_package.cpp - Declarative .gkmetadata Parser, Validator & Security Policy
#include "guild/abi/gkernel_package.hpp"
#include "guild/models/sha256.hpp"
#include "guild/server/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <functional>

namespace guild::abi {
namespace {

using server::json::JsonValue;

bool is_valid_hex(const std::string& s, size_t expected_len) {
    if (s.size() != expected_len) return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

bool is_safe_relative_path(const std::string& p) {
    if (p.empty() || p[0] == '/' || p[0] == '\\') return false;
    if (p.find("..") != std::string::npos || p.find(':') != std::string::npos) return false;
    return true;
}

bool parse_semver(const std::string& s, uint32_t& maj, uint32_t& min, uint32_t& pat) {
    unsigned a = 0, b = 0, c = 0;
    char tail = 0;
    if (std::sscanf(s.c_str(), "%u.%u.%u%c", &a, &b, &c, &tail) != 3) return false;
    maj = a;
    min = b;
    pat = c;
    return true;
}

bool semver_gte(const std::string& actual, const std::string& required_min) {
    uint32_t a0 = 0, a1 = 0, a2 = 0, r0 = 0, r1 = 0, r2 = 0;
    if (!parse_semver(actual, a0, a1, a2) || !parse_semver(required_min, r0, r1, r2)) return false;
    if (a0 != r0) return a0 > r0;
    if (a1 != r1) return a1 > r1;
    return a2 >= r2;
}

} // namespace

guild_status_t GKernelMetadata::parse_json(const std::string& json_text,
                                           GKernelMetadata& out_meta,
                                           std::string& err) {
    out_meta = {};
    JsonValue root;
    if (!JsonValue::parse(json_text, root, err) || !root.is_object()) {
        if (err.empty()) err = ".gkmetadata root must be a JSON object";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }

    // Security rule: reject any installation scripts or arbitrary command hooks
    const char* forbidden_keys[] = {
        "install_script", "post_install", "pre_install", "build_script", "shell_command", "exec_hook"
    };
    for (const char* fkey : forbidden_keys) {
        if (root.contains(fkey)) {
            err = std::string("security violation: forbidden executable hook '") + fkey + "' in .gkmetadata";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
    }

    out_meta.format_version = static_cast<uint32_t>(root["format_version"].as_int(1));
    if (out_meta.format_version != 1) {
        err = "unsupported .gkmetadata format_version: " + std::to_string(out_meta.format_version);
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }

    out_meta.package_id = root["package_id"].as_string("");
    out_meta.scope = root["scope"].as_string("");
    out_meta.version = root["version"].as_string("");
    out_meta.publisher_id = root["publisher_id"].as_string("");

    uint32_t vmaj = 0, vmin = 0, vpat = 0;
    if (out_meta.package_id.empty() || out_meta.scope.empty() ||
        out_meta.publisher_id.empty() || !parse_semver(out_meta.version, vmaj, vmin, vpat)) {
        err = ".gkmetadata missing required package_id, scope, publisher_id, or valid semantic version";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    if (out_meta.scope != "cpu-kernels" && out_meta.scope != "cuda-kernels" &&
        out_meta.scope != "hip-kernels" && out_meta.scope != "runner" && out_meta.scope != "hybrid") {
        err = "invalid .gkmetadata scope: " + out_meta.scope;
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }

    const JsonValue& abi = root["kernel_abi"];
    if (!abi.is_object()) {
        err = ".gkmetadata missing 'kernel_abi' object";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    out_meta.abi_major = static_cast<uint32_t>(abi["major"].as_int(0));
    out_meta.abi_minor_min = static_cast<uint32_t>(abi["minor_min"].as_int(0));
    out_meta.abi_minor_max = static_cast<uint32_t>(abi["minor_max"].as_int(out_meta.abi_minor_min));

    out_meta.required_core_caps = static_cast<guild_core_cap_flags_t>(root["required_core_caps"].as_int(0));
    out_meta.required_cpu_isa = static_cast<guild_cpu_isa_flags_t>(root["required_cpu_isa"].as_int(0));
    out_meta.required_gpu_caps = static_cast<guild_gpu_cap_flags_t>(root["required_gpu_caps"].as_int(0));
    out_meta.min_sm_version = static_cast<int32_t>(root["min_sm_version"].as_int(0));

    const JsonValue& backends = root["backends"];
    if (backends.is_array()) {
        for (const auto& b : backends.arr_val) {
            const std::string bs = b.as_string("");
            if (bs == "cpu") out_meta.supported_backends |= GUILD_BACKEND_CPU;
            else if (bs == "cuda") out_meta.supported_backends |= GUILD_BACKEND_CUDA;
            else if (bs == "hip") out_meta.supported_backends |= GUILD_BACKEND_HIP;
        }
    }

    auto read_str_array = [](const JsonValue& arr, std::vector<std::string>& dst) {
        if (!arr.is_array()) return;
        for (const auto& v : arr.arr_val) {
            if (v.is_string() && !v.as_string().empty()) dst.push_back(v.as_string());
        }
    };
    read_str_array(root["supported_architectures"], out_meta.supported_architectures);
    read_str_array(root["provided_ops"], out_meta.provided_ops);
    read_str_array(root["provided_runners"], out_meta.provided_runners);
    read_str_array(root["required_ops"], out_meta.required_ops);

    const JsonValue& deps = root["dependencies"];
    if (deps.is_array()) {
        for (const auto& d : deps.arr_val) {
            if (!d.is_object()) continue;
            PackageDependency dep;
            dep.package_id = d["package_id"].as_string("");
            dep.min_version = d["min_version"].as_string("1.0.0");
            dep.expected_publisher_id = d["publisher_id"].as_string("");
            if (dep.package_id.empty()) {
                err = ".gkmetadata dependency entry is missing package_id";
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            }
            out_meta.dependencies.push_back(std::move(dep));
        }
    }

    const JsonValue& tconstraints = root["tensor_constraints"];
    if (tconstraints.is_array()) {
        for (const auto& tc : tconstraints.arr_val) {
            if (!tc.is_object()) continue;
            TensorConstraintSpec spec;
            spec.op_name = tc["op_name"].as_string("");
            read_str_array(tc["allowed_dtypes"], spec.allowed_dtypes);
            spec.max_batch_rows = tc["max_batch_rows"].as_int(0);
            spec.alignment_bytes = tc["alignment_bytes"].as_int(32);
            out_meta.tensor_constraints.push_back(std::move(spec));
        }
    }

    const JsonValue& arts = root["artifacts"];
    if (arts.is_array()) {
        for (const auto& a : arts.arr_val) {
            if (!a.is_object()) continue;
            ArtifactHashEntry entry;
            entry.relative_path = a["path"].as_string("");
            entry.sha256 = a["sha256"].as_string("");
            entry.byte_size = static_cast<uint64_t>(a["byte_size"].as_int(0));
            if (!is_safe_relative_path(entry.relative_path)) {
                err = "security violation: unsafe artifact path '" + entry.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            if (!is_valid_hex(entry.sha256, 64)) {
                err = "invalid artifact SHA-256 digest for '" + entry.relative_path + "'";
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            }
            out_meta.artifacts.push_back(std::move(entry));
        }
    }

    const JsonValue& sig = root["signature"];
    if (sig.is_object()) {
        out_meta.signature.algorithm = sig["algorithm"].as_string("");
        out_meta.signature.key_id = sig["key_id"].as_string("");
        out_meta.signature.signature_hex = sig["signature_hex"].as_string("");
        out_meta.signature.payload_sha256 = sig["payload_sha256"].as_string("");
    }

    const JsonValue& prov = root["provenance"];
    if (prov.is_object()) {
        out_meta.provenance.spdx_license = prov["spdx_license"].as_string("");
        out_meta.provenance.source_repo = prov["source_repo"].as_string("");
        out_meta.provenance.source_revision = prov["source_revision"].as_string("");
    }

    const JsonValue& fixtures = root["validation_fixtures"];
    if (fixtures.is_array()) {
        for (const auto& fx : fixtures.arr_val) {
            if (!fx.is_object()) continue;
            ValidationFixtureSpec fspec;
            fspec.fixture_id = fx["fixture_id"].as_string("");
            fspec.relative_path = fx["path"].as_string("");
            fspec.sha256 = fx["sha256"].as_string("");
            fspec.max_abs_tol = fx["max_abs_tol"].as_double(0.0);
            fspec.rel_l2_tol = fx["rel_l2_tol"].as_double(0.0);
            if (!fspec.relative_path.empty() && !is_safe_relative_path(fspec.relative_path)) {
                err = "security violation: unsafe fixture path '" + fspec.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            out_meta.validation_fixtures.push_back(std::move(fspec));
        }
    }

    err.clear();
    return GUILD_STATUS_OK;
}

std::string GKernelMetadata::to_json_string() const {
    JsonValue root = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    root["format_version"] = static_cast<int64_t>(format_version);
    root["package_id"] = package_id;
    root["scope"] = scope;
    root["version"] = version;
    root["publisher_id"] = publisher_id;

    JsonValue abi = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    abi["major"] = static_cast<int64_t>(abi_major);
    abi["minor_min"] = static_cast<int64_t>(abi_minor_min);
    abi["minor_max"] = static_cast<int64_t>(abi_minor_max);
    root["kernel_abi"] = abi;

    root["required_core_caps"] = static_cast<int64_t>(required_core_caps);
    root["required_cpu_isa"] = static_cast<int64_t>(required_cpu_isa);
    root["required_gpu_caps"] = static_cast<int64_t>(required_gpu_caps);
    root["min_sm_version"] = static_cast<int64_t>(min_sm_version);

    std::vector<JsonValue> b_arr;
    if (supported_backends & GUILD_BACKEND_CPU) b_arr.emplace_back("cpu");
    if (supported_backends & GUILD_BACKEND_CUDA) b_arr.emplace_back("cuda");
    if (supported_backends & GUILD_BACKEND_HIP) b_arr.emplace_back("hip");
    root["backends"] = JsonValue(std::move(b_arr));

    auto write_str_array = [](const std::vector<std::string>& src) {
        std::vector<JsonValue> arr;
        for (const auto& s : src) arr.emplace_back(s);
        return JsonValue(std::move(arr));
    };
    root["supported_architectures"] = write_str_array(supported_architectures);
    root["provided_ops"] = write_str_array(provided_ops);
    root["provided_runners"] = write_str_array(provided_runners);
    root["required_ops"] = write_str_array(required_ops);

    std::vector<JsonValue> dep_arr;
    for (const auto& d : dependencies) {
        JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
        o["package_id"] = d.package_id;
        o["min_version"] = d.min_version;
        o["publisher_id"] = d.expected_publisher_id;
        dep_arr.push_back(std::move(o));
    }
    root["dependencies"] = JsonValue(std::move(dep_arr));

    std::vector<JsonValue> art_arr;
    for (const auto& a : artifacts) {
        JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
        o["path"] = a.relative_path;
        o["sha256"] = a.sha256;
        o["byte_size"] = static_cast<int64_t>(a.byte_size);
        art_arr.push_back(std::move(o));
    }
    root["artifacts"] = JsonValue(std::move(art_arr));

    JsonValue sig = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    sig["algorithm"] = signature.algorithm;
    sig["key_id"] = signature.key_id;
    sig["signature_hex"] = signature.signature_hex;
    sig["payload_sha256"] = signature.payload_sha256;
    root["signature"] = sig;

    JsonValue prov = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    prov["spdx_license"] = provenance.spdx_license;
    prov["source_repo"] = provenance.source_repo;
    prov["source_revision"] = provenance.source_revision;
    root["provenance"] = prov;

    return root.serialize_pretty();
}

guild_status_t validate_gkernel_package(const GKernelMetadata& meta,
                                        const PackageSecurityPolicy& policy,
                                        const guild_device_capabilities_v1_t& host_caps,
                                        const std::string& package_dir,
                                        std::string& err) {
    // 1. Check release revocation
    const std::string release_tag = meta.package_id + "@" + meta.version;
    if (policy.revoked_releases.count(release_tag) > 0) {
        err = "security violation: package release '" + release_tag + "' has been revoked";
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }

    // 2. Check cryptographic signature & publisher trust policy
    if (policy.require_signature) {
        if (meta.signature.algorithm.empty() || meta.signature.key_id.empty() ||
            !is_valid_hex(meta.signature.signature_hex, 128) ||
            !is_valid_hex(meta.signature.payload_sha256, 64)) {
            err = "security violation: unsigned or malformed signature in package '" + meta.package_id + "'";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
        if (policy.revoked_key_ids.count(meta.signature.key_id) > 0) {
            err = "security violation: signing key '" + meta.signature.key_id + "' has been revoked";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
        const auto pub_it = policy.trusted_publishers.find(meta.publisher_id);
        if (pub_it == policy.trusted_publishers.end() ||
            pub_it->second.count(meta.signature.key_id) == 0) {
            err = "security violation: publisher '" + meta.publisher_id + "' or key '" +
                  meta.signature.key_id + "' is not trusted";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
    }

    // 3. Check Kernel ABI compatibility
    if (meta.abi_major != GUILD_KERNEL_ABI_VERSION_MAJOR ||
        meta.abi_minor_min > GUILD_KERNEL_ABI_VERSION_MINOR) {
        err = "package '" + meta.package_id + "' requires incompatible Kernel ABI version";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }

    // 4. Check Guild Core capabilities
    if ((meta.required_core_caps & host_caps.core_cap_flags) != meta.required_core_caps) {
        err = "package '" + meta.package_id + "' requires Guild Core capabilities not supported by this host";
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }

    // 5. Check hardware backend / ISA / SM compatibility
    if (meta.supported_backends != GUILD_BACKEND_NONE &&
        (meta.supported_backends & host_caps.available_backends) == 0) {
        err = "package '" + meta.package_id + "' requires a hardware backend not available on this host";
        return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
    }
    if ((meta.required_cpu_isa & host_caps.cpu_isa_flags) != meta.required_cpu_isa) {
        err = "package '" + meta.package_id + "' requires CPU ISA features not supported by this CPU";
        return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
    }
    if ((meta.required_gpu_caps & host_caps.gpu_cap_flags) != meta.required_gpu_caps ||
        (meta.min_sm_version > 0 && host_caps.gpu_sm_version < meta.min_sm_version)) {
        err = "package '" + meta.package_id + "' requires GPU capabilities/SM version not supported by this GPU";
        return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
    }

    // 6. Verify content-addressed artifact hashes on disk if package_dir is provided
    if (policy.verify_artifact_hashes && !package_dir.empty()) {
        for (const auto& art : meta.artifacts) {
            if (!is_safe_relative_path(art.relative_path)) {
                err = "security violation: unsafe artifact path '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            const std::filesystem::path full_path = std::filesystem::path(package_dir) / art.relative_path;
            std::error_code ec;
            if (!std::filesystem::is_regular_file(full_path, ec)) {
                err = "security violation: missing declared artifact '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            if (art.byte_size > 0 && std::filesystem::file_size(full_path, ec) != art.byte_size) {
                err = "security violation: artifact size mismatch for '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            const std::string actual_hash = models::Sha256::hash_file(full_path.string());
            if (actual_hash != art.sha256) {
                err = "security violation: artifact SHA-256 mismatch for '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
        }
    }

    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t validate_package_dependency_graph(const std::vector<GKernelMetadata>& packages,
                                                 const PackageSecurityPolicy& policy,
                                                 const guild_device_capabilities_v1_t& host_caps,
                                                 std::string& err) {
    std::map<std::string, const GKernelMetadata*> by_id;
    std::set<std::string> provided_ops;
    for (const auto& pkg : packages) {
        if (by_id.count(pkg.package_id) > 0) {
            err = "duplicate package in dependency graph: " + pkg.package_id;
            return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
        }
        const guild_status_t st = validate_gkernel_package(pkg, policy, host_caps, "", err);
        if (st != GUILD_STATUS_OK) return st;
        by_id[pkg.package_id] = &pkg;
        for (const auto& op : pkg.provided_ops) provided_ops.insert(op);
    }

    // Check package dependencies, versions, publisher pinning, and required_ops
    for (const auto& pkg : packages) {
        for (const auto& dep : pkg.dependencies) {
            const auto it = by_id.find(dep.package_id);
            if (it == by_id.end()) {
                err = "package '" + pkg.package_id + "' requires missing package dependency '" + dep.package_id + "'";
                return GUILD_STATUS_ERR_DEPENDENCY_MISSING;
            }
            if (!semver_gte(it->second->version, dep.min_version)) {
                err = "package '" + pkg.package_id + "' requires '" + dep.package_id + "' >= " +
                      dep.min_version + " (found " + it->second->version + ")";
                return GUILD_STATUS_ERR_DEPENDENCY_MISSING;
            }
            if (!dep.expected_publisher_id.empty() &&
                it->second->publisher_id != dep.expected_publisher_id) {
                err = "security violation: dependency '" + dep.package_id + "' publisher mismatch";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
        }
        for (const auto& req_op : pkg.required_ops) {
            if (provided_ops.count(req_op) == 0) {
                err = "package '" + pkg.package_id + "' requires operation '" + req_op + "' not provided by any loaded package";
                return GUILD_STATUS_ERR_DEPENDENCY_MISSING;
            }
        }
    }

    // Cycle detection via DFS
    std::map<std::string, int> visit_state; // 0 = unvisited, 1 = visiting, 2 = done
    std::function<bool(const std::string&)> has_cycle = [&](const std::string& id) -> bool {
        visit_state[id] = 1;
        const auto* node = by_id.at(id);
        for (const auto& dep : node->dependencies) {
            const int st = visit_state[dep.package_id];
            if (st == 1) return true;
            if (st == 0 && has_cycle(dep.package_id)) return true;
        }
        visit_state[id] = 2;
        return false;
    };
    for (const auto& pkg : packages) {
        if (visit_state[pkg.package_id] == 0 && has_cycle(pkg.package_id)) {
            err = "cyclic dependency detected involving package '" + pkg.package_id + "'";
            return GUILD_STATUS_ERR_DEPENDENCY_MISSING;
        }
    }

    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t verify_gexec_kernel_lock(const GexecKernelLock& lock,
                                        const GKernelMetadata& meta,
                                        const guild_kernel_op_desc_v1_t& op_desc,
                                        std::string& err) {
    if (meta.package_id != lock.package_id || meta.version != lock.package_version) {
        err = "GEXEC lock mismatch: expected " + lock.package_id + "@" + lock.package_version +
              ", found " + meta.package_id + "@" + meta.version;
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }
    if (!op_desc.impl_id || lock.impl_id != op_desc.impl_id || op_desc. abi_version != lock.abi_version) {
        err = "GEXEC lock mismatch: kernel implementation ID or ABI version changed";
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }
    if (!lock.artifact_sha256.empty()) {
        const bool match = std::any_of(meta.artifacts.begin(), meta.artifacts.end(), [&](const ArtifactHashEntry& a) {
            return a.sha256 == lock.artifact_sha256;
        });
        if (!match) {
            err = "GEXEC lock mismatch: locked artifact SHA-256 digest is not present in package";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
    }
    err.clear();
    return GUILD_STATUS_OK;
}

} // namespace guild::abi
