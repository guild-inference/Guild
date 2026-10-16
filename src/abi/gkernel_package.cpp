// src/abi/gkernel_package.cpp - Declarative .gkmetadata Parser, Ed25519 Verifier, Archive Stager & Package Manager
#include "guild/abi/gkernel_package.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/model/runner.hpp"
#include "guild/models/sha256.hpp"
#include "guild/server/json.hpp"

#if !defined(_WIN32)
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>

namespace guild::abi {
namespace {

namespace fs = std::filesystem;
using server::json::JsonValue;

bool is_valid_hex(const std::string& s, size_t expected_len) {
    if (s.size() != expected_len) return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

bool is_safe_identifier(const std::string& s) {
    if (s.empty() || s.size() > 128) return false;
    if (s.find("..") != std::string::npos) return false;
    for (unsigned char c : s) {
        if (!(std::isalnum(c) || c == '_' || c == '-' || c == '.')) return false;
    }
    return s.front() != '.' && s.back() != '.';
}

bool is_valid_c_identifier(const std::string& s) {
    if (s.empty() || s.size() > 128) return false;
    const unsigned char f = static_cast<unsigned char>(s[0]);
    if (!(std::isalpha(f) || f == '_')) return false;
    for (unsigned char c : s) {
        if (!(std::isalnum(c) || c == '_')) return false;
    }
    return true;
}

bool is_safe_relative_path(const std::string& p) {
    if (p.empty() || p.size() > 512) return false;
    if (p[0] == '/' || p[0] == '\\' || p.back() == '/' || p.back() == '\\') return false;
    for (unsigned char c : p) {
        if (c < 0x20 || c == 0x7f || c == '\\' || c == ':') return false;
    }
    // Check each segment between '/'
    size_t start = 0;
    while (start < p.size()) {
        size_t slash = p.find('/', start);
        if (slash == std::string::npos) slash = p.size();
        const std::string seg = p.substr(start, slash - start);
        if (seg.empty() || seg == "." || seg == "..") return false;
        start = slash + 1;
    }
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

bool semver_lte(const std::string& actual, const std::string& required_max) {
    uint32_t a0 = 0, a1 = 0, a2 = 0, r0 = 0, r1 = 0, r2 = 0;
    if (!parse_semver(actual, a0, a1, a2) || !parse_semver(required_max, r0, r1, r2)) return false;
    if (a0 != r0) return a0 < r0;
    if (a1 != r1) return a1 < r1;
    return a2 <= r2;
}

std::string host_os_name() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

std::string host_arch_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#else
    return "unknown";
#endif
}

bool is_shared_lib_extension(const std::string& path) {
    auto ends_with = [&](const char* suffix) {
        const size_t slen = std::strlen(suffix);
        return path.size() >= slen && path.compare(path.size() - slen, slen, suffix) == 0;
    };
    return ends_with(".so") || ends_with(".dll") || ends_with(".dylib");
}

bool has_any_symlink_in_tree(const fs::path& root, const fs::path& rel) {
    std::error_code ec;
    if (fs::is_symlink(fs::symlink_status(root, ec))) return true;
    fs::path cur = root;
    for (const auto& part : rel) {
        cur /= part;
        if (fs::is_symlink(fs::symlink_status(cur, ec))) return true;
    }
    return false;
}

JsonValue build_unsigned_json_object(const GKernelMetadata& m) {
    JsonValue root = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    root["format_version"] = static_cast<int64_t>(m.format_version);
    root["package_kind"] = m.package_kind.empty() ? std::string("gkernel") : m.package_kind;
    root["package_id"] = m.package_id;
    root["scope"] = m.scope;
    root["version"] = m.version;
    root["publisher_id"] = m.publisher_id;
    root["entrypoint_symbol"] = m.entrypoint_symbol.empty() ? std::string(GUILD_PLUGIN_ENTRY_SYMBOL_V1) : m.entrypoint_symbol;
    if (!m.library_path.empty()) {
        root["library_path"] = m.library_path;
    }

    JsonValue abi = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    abi["major"] = static_cast<int64_t>(m.abi_major);
    abi["minor_min"] = static_cast<int64_t>(m.abi_minor_min);
    abi["minor_max"] = static_cast<int64_t>(m.abi_minor_max);
    root["kernel_abi"] = abi;

    if (m.target_platform.os != "any" || m.target_platform.arch != "any") {
        JsonValue tp = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
        tp["os"] = m.target_platform.os.empty() ? std::string("any") : m.target_platform.os;
        tp["arch"] = m.target_platform.arch.empty() ? std::string("any") : m.target_platform.arch;
        root["target_platform"] = tp;
    }

    root["required_core_caps"] = static_cast<int64_t>(m.required_core_caps);
    root["required_cpu_isa"] = static_cast<int64_t>(m.required_cpu_isa);
    root["required_gpu_caps"] = static_cast<int64_t>(m.required_gpu_caps);
    root["min_sm_version"] = static_cast<int64_t>(m.min_sm_version);
    if (m.max_sm_version > 0) {
        root["max_sm_version"] = static_cast<int64_t>(m.max_sm_version);
    }

    std::vector<JsonValue> b_arr;
    if (m.supported_backends & GUILD_BACKEND_CPU) b_arr.emplace_back("cpu");
    if (m.supported_backends & GUILD_BACKEND_CUDA) b_arr.emplace_back("cuda");
    if (m.supported_backends & GUILD_BACKEND_HIP) b_arr.emplace_back("hip");
    root["backends"] = JsonValue(std::move(b_arr));

    auto write_str_array = [](const std::vector<std::string>& src) {
        std::vector<JsonValue> arr;
        for (const auto& s : src) arr.emplace_back(s);
        return JsonValue(std::move(arr));
    };
    root["supported_architectures"] = write_str_array(m.supported_architectures);
    root["provided_ops"] = write_str_array(m.provided_ops);
    root["provided_runners"] = write_str_array(m.provided_runners);
    root["required_ops"] = write_str_array(m.required_ops);

    std::vector<JsonValue> dep_arr;
    for (const auto& d : m.dependencies) {
        JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
        o["package_id"] = d.package_id;
        o["min_version"] = d.min_version;
        o["publisher_id"] = d.expected_publisher_id;
        if (!d.max_version.empty()) o["max_version"] = d.max_version;
        dep_arr.push_back(std::move(o));
    }
    root["dependencies"] = JsonValue(std::move(dep_arr));

    if (!m.module_dependencies.empty()) {
        std::vector<JsonValue> mdep_arr;
        for (const auto& md : m.module_dependencies) {
            JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
            o["module_id"] = md.module_id;
            o["min_version"] = md.min_version;
            o["abi_tag"] = md.abi_tag;
            o["publisher_id"] = md.expected_publisher_id;
            o["sha256"] = md.sha256;
            mdep_arr.push_back(std::move(o));
        }
        root["module_dependencies"] = JsonValue(std::move(mdep_arr));
    }

    if (!m.tensor_constraints.empty()) {
        std::vector<JsonValue> tc_arr;
        for (const auto& tc : m.tensor_constraints) {
            JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
            o["op_name"] = tc.op_name;
            o["allowed_dtypes"] = write_str_array(tc.allowed_dtypes);
            o["max_batch_rows"] = tc.max_batch_rows;
            o["alignment_bytes"] = tc.alignment_bytes;
            tc_arr.push_back(std::move(o));
        }
        root["tensor_constraints"] = JsonValue(std::move(tc_arr));
    }

    std::vector<JsonValue> art_arr;
    for (const auto& a : m.artifacts) {
        JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
        o["path"] = a.relative_path;
        o["sha256"] = a.sha256;
        o["byte_size"] = static_cast<int64_t>(a.byte_size);
        if (!a.role.empty()) o["role"] = a.role;
        art_arr.push_back(std::move(o));
    }
    root["artifacts"] = JsonValue(std::move(art_arr));

    JsonValue prov = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    prov["spdx_license"] = m.provenance.spdx_license;
    prov["source_repo"] = m.provenance.source_repo;
    prov["source_revision"] = m.provenance.source_revision;
    root["provenance"] = prov;

    if (!m.validation_fixtures.empty()) {
        std::vector<JsonValue> fx_arr;
        for (const auto& fx : m.validation_fixtures) {
            JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
            o["fixture_id"] = fx.fixture_id;
            o["path"] = fx.relative_path;
            o["sha256"] = fx.sha256;
            o["max_abs_tol"] = fx.max_abs_tol;
            o["rel_l2_tol"] = fx.rel_l2_tol;
            fx_arr.push_back(std::move(o));
        }
        root["validation_fixtures"] = JsonValue(std::move(fx_arr));
    }

    return root;
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

    // 1. Security rule: reject any installation scripts or arbitrary command hooks
    const char* forbidden_keys[] = {
        "install_script", "post_install", "pre_install", "build_script",
        "shell_command", "exec_hook", "python_hook", "init_script", "uninstall_script"
    };
    for (const char* fkey : forbidden_keys) {
        if (root.contains(fkey)) {
            err = std::string("security violation: forbidden executable hook '") + fkey + "' in .gkmetadata";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
    }

    // 2. Reject unknown top-level keys so malformed/drifted manifests never silently pass
    const std::vector<std::string> allowed_keys = {
        "format_version", "package_kind", "package_id", "scope", "version", "publisher_id",
        "entrypoint_symbol", "library_path", "kernel_abi", "target_platform",
        "required_core_caps", "required_cpu_isa", "required_gpu_caps",
        "min_sm_version", "max_sm_version", "backends", "supported_architectures",
        "provided_ops", "provided_runners", "required_ops", "dependencies",
        "module_dependencies", "tensor_constraints", "artifacts",
        "signature", "provenance", "validation_fixtures"
    };
    std::string unknown_key;
    if (!root.check_unsupported_keys(allowed_keys, unknown_key)) {
        err = "unsupported or unknown key '" + unknown_key + "' in .gkmetadata";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }

    if (!root.contains("format_version") || !root["format_version"].is_number()) {
        err = ".gkmetadata missing numeric 'format_version'";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    out_meta.format_version = static_cast<uint32_t>(root["format_version"].as_int(0));
    if (out_meta.format_version != 1) {
        err = "unsupported .gkmetadata format_version: " + std::to_string(out_meta.format_version);
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }

    out_meta.package_kind = root["package_kind"].as_string("gkernel");
    if (out_meta.package_kind != "gkernel" && out_meta.package_kind != "gkmodule") {
        err = "invalid .gkmetadata package_kind: " + out_meta.package_kind + " (must be 'gkernel' or 'gkmodule')";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }

    out_meta.package_id = root["package_id"].as_string("");
    out_meta.scope = root["scope"].as_string("");
    out_meta.version = root["version"].as_string("");
    out_meta.publisher_id = root["publisher_id"].as_string("");
    out_meta.entrypoint_symbol = root["entrypoint_symbol"].as_string(GUILD_PLUGIN_ENTRY_SYMBOL_V1);
    out_meta.library_path = root["library_path"].as_string("");

    uint32_t vmaj = 0, vmin = 0, vpat = 0;
    if (!is_safe_identifier(out_meta.package_id) || out_meta.scope.empty() ||
        !is_safe_identifier(out_meta.publisher_id) || !parse_semver(out_meta.version, vmaj, vmin, vpat)) {
        err = ".gkmetadata missing or invalid package_id, scope, publisher_id, or semantic version";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    if (!is_valid_c_identifier(out_meta.entrypoint_symbol)) {
        err = "security violation: invalid entrypoint_symbol '" + out_meta.entrypoint_symbol + "'";
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }
    if (!out_meta.library_path.empty() && !is_safe_relative_path(out_meta.library_path)) {
        err = "security violation: unsafe library_path '" + out_meta.library_path + "'";
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }

    const bool valid_gkernel_scope =
        (out_meta.scope == "cpu-kernels" || out_meta.scope == "cuda-kernels" ||
         out_meta.scope == "hip-kernels" || out_meta.scope == "runner" || out_meta.scope == "hybrid");
    const bool valid_gkmodule_scope = (out_meta.scope == "module" || valid_gkernel_scope);
    if ((out_meta.package_kind == "gkernel" && !valid_gkernel_scope) ||
        (out_meta.package_kind == "gkmodule" && !valid_gkmodule_scope)) {
        err = "invalid .gkmetadata scope '" + out_meta.scope + "' for package_kind '" + out_meta.package_kind + "'";
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
    if (out_meta.abi_minor_min > out_meta.abi_minor_max) {
        err = ".gkmetadata kernel_abi minor_min exceeds minor_max";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }

    if (root.contains("target_platform")) {
        const JsonValue& tp = root["target_platform"];
        if (!tp.is_object()) {
            err = ".gkmetadata 'target_platform' must be an object";
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
        out_meta.target_platform.os = tp["os"].as_string("any");
        out_meta.target_platform.arch = tp["arch"].as_string("any");
    }

    out_meta.required_core_caps = static_cast<guild_core_cap_flags_t>(root["required_core_caps"].as_int(0));
    out_meta.required_cpu_isa = static_cast<guild_cpu_isa_flags_t>(root["required_cpu_isa"].as_int(0));
    out_meta.required_gpu_caps = static_cast<guild_gpu_cap_flags_t>(root["required_gpu_caps"].as_int(0));
    out_meta.min_sm_version = static_cast<int32_t>(root["min_sm_version"].as_int(0));
    out_meta.max_sm_version = static_cast<int32_t>(root["max_sm_version"].as_int(0));

    const JsonValue& backends = root["backends"];
    if (backends.is_array()) {
        for (const auto& b : backends.arr_val) {
            const std::string bs = b.as_string("");
            if (bs == "cpu") out_meta.supported_backends |= GUILD_BACKEND_CPU;
            else if (bs == "cuda") out_meta.supported_backends |= GUILD_BACKEND_CUDA;
            else if (bs == "hip") out_meta.supported_backends |= GUILD_BACKEND_HIP;
            else {
                err = "invalid backend '" + bs + "' in .gkmetadata";
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            }
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
            dep.max_version = d["max_version"].as_string("");
            uint32_t d0 = 0, d1 = 0, d2 = 0;
            if (!is_safe_identifier(dep.package_id) || !parse_semver(dep.min_version, d0, d1, d2) ||
                (!dep.max_version.empty() && !parse_semver(dep.max_version, d0, d1, d2))) {
                err = ".gkmetadata dependency entry has invalid package_id or version range";
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            }
            out_meta.dependencies.push_back(std::move(dep));
        }
    }

    const JsonValue& mdeps = root["module_dependencies"];
    if (mdeps.is_array()) {
        for (const auto& md : mdeps.arr_val) {
            if (!md.is_object()) continue;
            ModuleDependencySpec mspec;
            mspec.module_id = md["module_id"].as_string("");
            mspec.min_version = md["min_version"].as_string("1.0.0");
            mspec.abi_tag = md["abi_tag"].as_string("");
            mspec.expected_publisher_id = md["publisher_id"].as_string("");
            mspec.sha256 = md["sha256"].as_string("");
            uint32_t d0 = 0, d1 = 0, d2 = 0;
            if (!is_safe_identifier(mspec.module_id) || !parse_semver(mspec.min_version, d0, d1, d2) ||
                (!mspec.sha256.empty() && !is_valid_hex(mspec.sha256, 64))) {
                err = ".gkmetadata module_dependencies entry has invalid module_id, min_version, or sha256";
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            }
            out_meta.module_dependencies.push_back(std::move(mspec));
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
        std::set<std::string> seen_paths;
        for (const auto& a : arts.arr_val) {
            if (!a.is_object()) continue;
            ArtifactHashEntry entry;
            entry.relative_path = a["path"].as_string("");
            entry.sha256 = a["sha256"].as_string("");
            entry.byte_size = static_cast<uint64_t>(a["byte_size"].as_int(0));
            entry.role = a["role"].as_string("");
            if (!is_safe_relative_path(entry.relative_path)) {
                err = "security violation: unsafe artifact path '" + entry.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            if (!seen_paths.insert(entry.relative_path).second) {
                err = "security violation: duplicate artifact path '" + entry.relative_path + "' in .gkmetadata";
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

std::string GKernelMetadata::canonical_signing_payload() const {
    return build_unsigned_json_object(*this).dump();
}

std::string GKernelMetadata::compute_payload_sha256() const {
    return models::Sha256::hash_string(canonical_signing_payload());
}

std::string GKernelMetadata::signing_message() const {
    return "GUILD-GKERNEL-SIG-V1:" + compute_payload_sha256();
}

bool GKernelMetadata::sign_with_seed_hex(const std::string& key_id,
                                         const std::string& private_seed_hex,
                                         std::string& err) {
    if (key_id.empty()) {
        err = "signing key_id cannot be empty";
        return false;
    }
    const std::string payload_sha = compute_payload_sha256();
    const std::string msg = "GUILD-GKERNEL-SIG-V1:" + payload_sha;
    std::string sig_hex;
    if (!crypto::ed25519_sign_hex(msg, private_seed_hex, sig_hex)) {
        err = "invalid 64-character hex Ed25519 private seed";
        return false;
    }
    signature.algorithm = "ed25519";
    signature.key_id = key_id;
    signature.payload_sha256 = payload_sha;
    signature.signature_hex = sig_hex;
    err.clear();
    return true;
}

std::string GKernelMetadata::to_json_string() const {
    JsonValue root = build_unsigned_json_object(*this);
    JsonValue sig = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    sig["algorithm"] = signature.algorithm;
    sig["key_id"] = signature.key_id;
    sig["signature_hex"] = signature.signature_hex;
    sig["payload_sha256"] = signature.payload_sha256;
    root["signature"] = sig;
    return root.serialize_pretty();
}

std::string GKernelMetadata::resolve_primary_library_path() const {
    if (!library_path.empty()) return library_path;
    for (const auto& a : artifacts) {
        if (a.role == "shared_library" || is_shared_lib_extension(a.relative_path)) {
            return a.relative_path;
        }
    }
    return "";
}

PackageSecurityPolicy PackageSecurityPolicy::default_builtin_policy() {
    PackageSecurityPolicy pol;
    pol.require_signature = true;
    pol.allow_unsigned_dev = false;
    pol.verify_artifact_hashes = true;
    // No private release identity is embedded in Core. Trust public keys supplied
    // by the operator/distribution; published development seeds cannot authenticate releases.
    return pol;
}

bool PackageSecurityPolicy::save_to_file(const std::string& path, std::string& err) const {
    JsonValue root = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    root["require_signature"] = require_signature;
    root["allow_unsigned_dev"] = allow_unsigned_dev;
    root["verify_artifact_hashes"] = verify_artifact_hashes;

    std::vector<JsonValue> keys_arr;
    for (const auto& [pub, key_set] : trusted_publishers) {
        for (const auto& kid : key_set) {
            JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
            o["publisher_id"] = pub;
            o["key_id"] = kid;
            const auto pk_it = trusted_public_keys.find({pub, kid});
            if (pk_it != trusted_public_keys.end()) {
                o["public_key_hex"] = pk_it->second;
            }
            keys_arr.push_back(std::move(o));
        }
    }
    root["trusted_keys"] = JsonValue(std::move(keys_arr));

    auto set_to_json = [](const std::set<std::string>& s) {
        std::vector<JsonValue> arr;
        for (const auto& item : s) arr.emplace_back(item);
        return JsonValue(std::move(arr));
    };
    root["revoked_key_ids"] = set_to_json(revoked_key_ids);
    root["revoked_publishers"] = set_to_json(revoked_publishers);
    root["revoked_releases"] = set_to_json(revoked_releases);

    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        if (!f.good()) {
            err = "failed to write trust store file: " + tmp_path;
            return false;
        }
        const std::string s = root.serialize_pretty();
        f.write(s.data(), static_cast<std::streamsize>(s.size()));
    }
    fs::rename(tmp_path, path, ec);
    if (ec) {
        err = "failed to commit trust store file: " + ec.message();
        return false;
    }
    err.clear();
    return true;
}

bool PackageSecurityPolicy::load_from_file(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        err = "cannot open trust store file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    JsonValue root;
    if (!JsonValue::parse(ss.str(), root, err) || !root.is_object()) {
        return false;
    }
    require_signature = root["require_signature"].as_bool(true);
    allow_unsigned_dev = root["allow_unsigned_dev"].as_bool(false);
    verify_artifact_hashes = root["verify_artifact_hashes"].as_bool(true);
    trusted_publishers.clear();
    trusted_public_keys.clear();
    revoked_key_ids.clear();
    revoked_publishers.clear();
    revoked_releases.clear();

    const JsonValue& tkeys = root["trusted_keys"];
    if (tkeys.is_array()) {
        for (const auto& tk : tkeys.arr_val) {
            if (!tk.is_object()) continue;
            const std::string pub = tk["publisher_id"].as_string("");
            const std::string kid = tk["key_id"].as_string("");
            const std::string pk = tk["public_key_hex"].as_string("");
            if (!pub.empty() && !kid.empty()) {
                trust_publisher_key(pub, kid, pk);
            }
        }
    }
    auto json_to_set = [](const JsonValue& arr, std::set<std::string>& dst) {
        if (!arr.is_array()) return;
        for (const auto& v : arr.arr_val) {
            if (v.is_string() && !v.as_string().empty()) dst.insert(v.as_string());
        }
    };
    json_to_set(root["revoked_key_ids"], revoked_key_ids);
    json_to_set(root["revoked_publishers"], revoked_publishers);
    json_to_set(root["revoked_releases"], revoked_releases);
    err.clear();
    return true;
}

guild_status_t validate_gkernel_package(const GKernelMetadata& meta,
                                        const PackageSecurityPolicy& policy,
                                        const guild_device_capabilities_v1_t& host_caps,
                                        const std::string& package_dir,
                                        std::string& err) {
    // 0. Enforce three-tier responsibility boundary: .gkmodule is a non-executable building block
    if (!meta.package_kind.empty() && meta.package_kind != "gkernel") {
        err = "package '" + meta.package_id + "' has package_kind '" + meta.package_kind +
              "': .gkmodule packages are reusable building blocks consumed by .gkernel packages and cannot be directly loaded or registered by Guild Core";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }

    // 1. Check release & publisher revocation
    const std::string release_tag = meta.package_id + "@" + meta.version;
    if (policy.revoked_releases.count(release_tag) > 0) {
        err = "security violation: package release '" + release_tag + "' has been revoked";
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }
    if (policy.revoked_publishers.count(meta.publisher_id) > 0) {
        err = "security violation: publisher '" + meta.publisher_id + "' has been revoked";
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }

    // 2. Check cryptographic Ed25519 signature & publisher trust policy
    if (policy.require_signature && !policy.allow_unsigned_dev) {
        if (meta.signature.algorithm != "ed25519" || meta.signature.key_id.empty() ||
            !is_valid_hex(meta.signature.signature_hex, 128) ||
            !is_valid_hex(meta.signature.payload_sha256, 64)) {
            err = "security violation: unsigned or malformed Ed25519 signature in package '" + meta.package_id + "'";
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
        const auto pk_it = policy.trusted_public_keys.find({meta.publisher_id, meta.signature.key_id});
        if (pk_it == policy.trusted_public_keys.end() || !is_valid_hex(pk_it->second, 64)) {
            err = "security violation: no pinned Ed25519 public key configured for publisher '" +
                  meta.publisher_id + "' key '" + meta.signature.key_id + "'";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
        const std::string expected_payload_sha = meta.compute_payload_sha256();
        if (meta.signature.payload_sha256 != expected_payload_sha) {
            err = "security violation: .gkmetadata payload_sha256 mismatch in package '" + meta.package_id +
                  "' (manifest content was modified after signing)";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
        if (!crypto::ed25519_verify_hex(meta.signing_message(), meta.signature.signature_hex, pk_it->second)) {
            err = "security violation: Ed25519 cryptographic signature verification failed for package '" +
                  meta.package_id + "'";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
    }

    // 3. Check Kernel ABI compatibility
    const uint32_t host_abi_minor = GUILD_KERNEL_ABI_VERSION_MINOR;
    if (meta.abi_major != GUILD_KERNEL_ABI_VERSION_MAJOR ||
        meta.abi_minor_min > host_abi_minor ||
        meta.abi_minor_max < host_abi_minor) {
        err = "package '" + meta.package_id + "' requires incompatible Kernel ABI version";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }

    // 4. Check target OS & CPU architecture compatibility
    if (!meta.target_platform.os.empty() && meta.target_platform.os != "any" &&
        meta.target_platform.os != host_os_name()) {
        err = "package '" + meta.package_id + "' targets OS '" + meta.target_platform.os +
              "', incompatible with host OS '" + host_os_name() + "'";
        return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
    }
    if (!meta.target_platform.arch.empty() && meta.target_platform.arch != "any" &&
        meta.target_platform.arch != host_arch_name()) {
        err = "package '" + meta.package_id + "' targets CPU architecture '" + meta.target_platform.arch +
              "', incompatible with host architecture '" + host_arch_name() + "'";
        return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
    }

    // 5. Check Guild Core capabilities
    if ((meta.required_core_caps & host_caps.core_cap_flags) != meta.required_core_caps) {
        err = "package '" + meta.package_id + "' requires Guild Core capabilities not supported by this host";
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }

    // 6. Check hardware backend / ISA / SM compatibility
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
        (meta.min_sm_version > 0 && host_caps.gpu_sm_version < meta.min_sm_version) ||
        (meta.max_sm_version > 0 && host_caps.gpu_sm_version > meta.max_sm_version)) {
        err = "package '" + meta.package_id + "' requires GPU capabilities/SM version not supported by this GPU";
        return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
    }

    // 7. Verify content-addressed artifact hashes & symlink safety on disk if package_dir is provided
    if (policy.verify_artifact_hashes && !package_dir.empty()) {
        std::set<std::string> seen_rel_paths;
        for (const auto& art : meta.artifacts) {
            if (!is_safe_relative_path(art.relative_path)) {
                err = "security violation: unsafe artifact path '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            if (!seen_rel_paths.insert(art.relative_path).second) {
                err = "security violation: duplicate artifact path '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            const fs::path root_p(package_dir);
            const fs::path rel_p(art.relative_path);
            if (has_any_symlink_in_tree(root_p, rel_p)) {
                err = "security violation: symlink detected in artifact path '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            const fs::path full_path = root_p / rel_p;
            std::error_code ec;
            const auto st = fs::symlink_status(full_path, ec);
            if (ec || fs::is_symlink(st) || !fs::is_regular_file(st)) {
                err = "security violation: missing or non-regular declared artifact '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            if (art.byte_size > 0 && fs::file_size(full_path, ec) != art.byte_size) {
                err = "security violation: artifact size mismatch for '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            const std::string actual_hash = models::Sha256::hash_file(full_path.string());
            if (actual_hash != art.sha256) {
                err = "security violation: artifact SHA-256 mismatch for '" + art.relative_path + "'";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
        }
        if (!meta.library_path.empty() && seen_rel_paths.count(meta.library_path) == 0) {
            err = "security violation: library_path '" + meta.library_path + "' is not listed in artifacts";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
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
    for (const auto& op : kernels::KernelRegistry::instance().list_operation_snapshots()) {
        if (op.enabled) provided_ops.insert(op.impl_id);
    }
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
            // Allow built-in pseudo-packages
            if (dep.package_id == "builtin.guild-kernels-cpu" ||
                dep.package_id == "builtin.guild-kernels-cuda" ||
                dep.package_id == "builtin.guild-kernels-amd" ||
                dep.package_id == "builtin.guild-kernels-qwen" ||
                dep.package_id == "builtin.guild-kernels-ornith") {
                continue;
            }
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
            if (!dep.max_version.empty() && !semver_lte(it->second->version, dep.max_version)) {
                err = "package '" + pkg.package_id + "' requires '" + dep.package_id + "' <= " +
                      dep.max_version + " (found " + it->second->version + ")";
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
            if (by_id.count(dep.package_id) == 0) continue;
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
    if (!op_desc.impl_id || lock.impl_id != op_desc.impl_id || op_desc.abi_version != lock.abi_version) {
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

// =============================================================================
// Archive Bundling & Hardened Staging (`GKPK` v1 + POSIX `.tar` + Directory)
// =============================================================================

namespace {

constexpr char kGkpkMagic[8] = {'G', 'K', 'P', 'K', 1, 0, 0, 0};

void write_le32(std::ostream& os, uint32_t v) {
    uint8_t b[4] = {
        static_cast<uint8_t>(v & 0xff),
        static_cast<uint8_t>((v >> 8) & 0xff),
        static_cast<uint8_t>((v >> 16) & 0xff),
        static_cast<uint8_t>((v >> 24) & 0xff)
    };
    os.write(reinterpret_cast<const char*>(b), 4);
}

void write_le64(std::ostream& os, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        const uint8_t byte = static_cast<uint8_t>((v >> (8 * i)) & 0xff);
        os.write(reinterpret_cast<const char*>(&byte), 1);
    }
}

bool read_le32(std::istream& is, uint32_t& out) {
    uint8_t b[4];
    if (!is.read(reinterpret_cast<char*>(b), 4)) return false;
    out = static_cast<uint32_t>(b[0]) |
          (static_cast<uint32_t>(b[1]) << 8) |
          (static_cast<uint32_t>(b[2]) << 16) |
          (static_cast<uint32_t>(b[3]) << 24);
    return true;
}

bool read_le64(std::istream& is, uint64_t& out) {
    uint8_t b[8];
    if (!is.read(reinterpret_cast<char*>(b), 8)) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out |= (static_cast<uint64_t>(b[i]) << (8 * i));
    return true;
}

bool write_staged_file(const fs::path& staging_root,
                       const std::string& rel_path,
                       const char* data,
                       size_t len,
                       std::string& err) {
    if (rel_path != ".gkmetadata" && !is_safe_relative_path(rel_path)) {
        err = "security violation: unsafe archive entry path '" + rel_path + "'";
        return false;
    }
    const fs::path dst = staging_root / rel_path;
    std::error_code ec;
    fs::create_directories(dst.parent_path(), ec);
    if (ec) {
        err = "failed to create staging parent directory: " + ec.message();
        return false;
    }
    if (fs::exists(fs::symlink_status(dst, ec))) {
        err = "security violation: duplicate file entry '" + rel_path + "' in package";
        return false;
    }
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        err = "failed to open staging file for writing: " + dst.string();
        return false;
    }
    if (len > 0) {
        out.write(data, static_cast<std::streamsize>(len));
        if (!out.good()) {
            err = "failed to write staging file: " + dst.string();
            return false;
        }
    }
    out.close();
#if !defined(_WIN32)
    ::chmod(dst.c_str(), is_shared_lib_extension(rel_path) ? 0500 : 0400);
#endif
    return true;
}

} // namespace

guild_status_t create_gkernel_bundle(const std::string& source_dir,
                                     const std::string& output_bundle_path,
                                     std::string& err) {
    const fs::path src_root(source_dir);
    const fs::path meta_path = src_root / ".gkmetadata";
    std::error_code ec;
    if (!fs::is_regular_file(fs::symlink_status(meta_path, ec))) {
        err = "create_gkernel_bundle: missing regular .gkmetadata file in " + source_dir;
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    std::string meta_json;
    {
        std::ifstream mf(meta_path, std::ios::binary);
        std::ostringstream ss;
        ss << mf.rdbuf();
        meta_json = ss.str();
    }
    GKernelMetadata meta;
    const guild_status_t pst = GKernelMetadata::parse_json(meta_json, meta, err);
    if (pst != GUILD_STATUS_OK) return pst;

    std::vector<std::pair<std::string, std::string>> files;
    files.emplace_back(".gkmetadata", meta_json);
    for (const auto& art : meta.artifacts) {
        if (has_any_symlink_in_tree(src_root, art.relative_path)) {
            err = "security violation: symlink in source artifact '" + art.relative_path + "'";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
        const fs::path ap = src_root / art.relative_path;
        if (!fs::is_regular_file(fs::symlink_status(ap, ec))) {
            err = "missing declared artifact '" + art.relative_path + "'";
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
        std::ifstream af(ap, std::ios::binary);
        std::ostringstream ss;
        ss << af.rdbuf();
        files.emplace_back(art.relative_path, ss.str());
    }
    for (const auto& fx : meta.validation_fixtures) {
        if (fx.relative_path.empty()) continue;
        const fs::path fp = src_root / fx.relative_path;
        if (!fs::is_regular_file(fs::symlink_status(fp, ec))) {
            err = "missing declared fixture '" + fx.relative_path + "'";
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
        std::ifstream ff(fp, std::ios::binary);
        std::ostringstream ss;
        ss << ff.rdbuf();
        files.emplace_back(fx.relative_path, ss.str());
    }

    fs::create_directories(fs::path(output_bundle_path).parent_path(), ec);
    std::ofstream out(output_bundle_path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        err = "cannot open output bundle path: " + output_bundle_path;
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    out.write(kGkpkMagic, sizeof(kGkpkMagic));
    write_le32(out, static_cast<uint32_t>(files.size()));
    for (const auto& [rel_path, content] : files) {
        write_le32(out, static_cast<uint32_t>(rel_path.size()));
        out.write(rel_path.data(), static_cast<std::streamsize>(rel_path.size()));
        write_le64(out, static_cast<uint64_t>(content.size()));
        if (!content.empty()) {
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
        }
    }
    if (!out.good()) {
        err = "failed while writing bundle: " + output_bundle_path;
        return GUILD_STATUS_ERR_INTERNAL;
    }
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t stage_gkernel_source(const std::string& source_path,
                                    const std::string& staging_dir,
                                    GKernelMetadata& out_meta,
                                    std::string& err,
                                    uint64_t max_total_bytes,
                                    size_t max_entries) {
    std::error_code ec;
    const fs::path src(source_path);
    const auto src_status = fs::symlink_status(src, ec);
    if (ec) {
        err = "cannot stat package source: " + source_path;
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    if (fs::is_symlink(src_status)) {
        err = "security violation: package source path cannot be a symlink: " + source_path;
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }

    const fs::path stage(staging_dir);
    fs::remove_all(stage, ec);
    fs::create_directories(stage, ec);
#if !defined(_WIN32)
    ::chmod(stage.c_str(), 0700);
#endif

    uint64_t total_extracted = 0;
    size_t entry_count = 0;
    std::set<std::string> staged_rel_paths;

    if (fs::is_directory(src_status)) {
        // Walk directory without following symlinks
        for (fs::recursive_directory_iterator it(src, fs::directory_options::none, ec), end; it != end; it.increment(ec)) {
            if (ec) {
                err = "error iterating package directory: " + ec.message();
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            const auto st = it->symlink_status(ec);
            if (fs::is_symlink(st)) {
                err = "security violation: symlink encountered inside package directory: " + it->path().string();
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            if (fs::is_directory(st)) continue;
            if (!fs::is_regular_file(st)) {
                err = "security violation: non-regular file encountered inside package directory";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            if (++entry_count > max_entries) {
                err = "security violation: package exceeds maximum file count limit (" + std::to_string(max_entries) + ")";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            const uint64_t fsize = fs::file_size(it->path(), ec);
            if (fsize > max_total_bytes || total_extracted + fsize > max_total_bytes) {
                err = "security violation: package exceeds maximum extracted byte budget";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            total_extracted += fsize;
            const std::string rel = fs::relative(it->path(), src, ec).generic_string();
            std::ifstream in(it->path(), std::ios::binary);
            std::string buf(fsize, '\0');
            if (fsize > 0 && !in.read(&buf[0], static_cast<std::streamsize>(fsize))) {
                err = "failed to read file from package directory: " + rel;
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            if (!write_staged_file(stage, rel, buf.data(), buf.size(), err)) {
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            staged_rel_paths.insert(rel);
        }
    } else if (fs::is_regular_file(src_status)) {
        const uint64_t archive_size = fs::file_size(src, ec);
        if (archive_size < 8 || archive_size > max_total_bytes + (4ull << 20)) {
            err = "security violation: package archive size is invalid or exceeds maximum limit";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
        std::ifstream in(src, std::ios::binary);
        char header[8] = {0};
        if (!in.read(header, 8)) {
            err = "failed to read package archive header";
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
        if (std::memcmp(header, kGkpkMagic, 8) == 0) {
            uint32_t count = 0;
            if (!read_le32(in, count) || count == 0 || count > max_entries) {
                err = "security violation: invalid or excessive entry count in GKPK archive";
                return GUILD_STATUS_ERR_SECURITY_VIOLATION;
            }
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t path_len = 0;
                if (!read_le32(in, path_len) || path_len == 0 || path_len > 512) {
                    err = "security violation: invalid path length in GKPK archive";
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                std::string rel_path(path_len, '\0');
                if (!in.read(&rel_path[0], path_len)) {
                    err = "security violation: truncated path in GKPK archive";
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                uint64_t fsize = 0;
                if (!read_le64(in, fsize) || fsize > max_total_bytes || total_extracted + fsize > max_total_bytes) {
                    err = "security violation: entry size in GKPK archive exceeds maximum budget";
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                total_extracted += fsize;
                std::string data(fsize, '\0');
                if (fsize > 0 && !in.read(&data[0], static_cast<std::streamsize>(fsize))) {
                    err = "security violation: truncated payload in GKPK archive for '" + rel_path + "'";
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                if (!write_staged_file(stage, rel_path, data.data(), data.size(), err)) {
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                staged_rel_paths.insert(rel_path);
            }
        } else {
            // Check for POSIX USTAR .tar archive (512-byte blocks)
            in.seekg(0, std::ios::beg);
            while (true) {
                char block[512];
                if (!in.read(block, 512)) break;
                bool all_zero = true;
                for (char c : block) if (c != 0) { all_zero = false; break; }
                if (all_zero) break;
                if (++entry_count > max_entries) {
                    err = "security violation: tar archive exceeds maximum entry count";
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                std::string name(block, strnlen(block, 100));
                if (!name.empty() && name.rfind("./", 0) == 0) name = name.substr(2);
                const char typeflag = block[156];
                // Parse octal size at [124..135]
                uint64_t fsize = 0;
                for (int i = 124; i < 136 && block[i] != '\0' && block[i] != ' '; ++i) {
                    if (block[i] < '0' || block[i] > '7') {
                        err = "security violation: malformed tar header size field";
                        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                    }
                    fsize = (fsize << 3) + static_cast<uint64_t>(block[i] - '0');
                }
                if (typeflag == '5') {
                    // Directory entry
                    continue;
                }
                if (typeflag != '0' && typeflag != '\0') {
                    err = "security violation: forbidden tar entry type (symlink/hardlink/device) for '" + name + "'";
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                if (fsize > max_total_bytes || total_extracted + fsize > max_total_bytes) {
                    err = "security violation: tar entry exceeds maximum byte budget";
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                total_extracted += fsize;
                std::string data(fsize, '\0');
                if (fsize > 0 && !in.read(&data[0], static_cast<std::streamsize>(fsize))) {
                    err = "security violation: truncated tar entry payload for '" + name + "'";
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                const size_t pad = (512 - (fsize % 512)) % 512;
                if (pad > 0) in.ignore(static_cast<std::streamsize>(pad));
                if (!write_staged_file(stage, name, data.data(), data.size(), err)) {
                    return GUILD_STATUS_ERR_SECURITY_VIOLATION;
                }
                staged_rel_paths.insert(name);
            }
            if (staged_rel_paths.empty()) {
                err = "unrecognized or empty .gkernel archive format";
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            }
        }
    } else {
        err = "package source is neither a directory nor a regular file";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }

    const fs::path staged_meta_path = stage / ".gkmetadata";
    if (!fs::is_regular_file(fs::symlink_status(staged_meta_path, ec))) {
        err = "package is missing required .gkmetadata manifest";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    std::string meta_text;
    {
        std::ifstream mf(staged_meta_path, std::ios::binary);
        std::ostringstream ss;
        ss << mf.rdbuf();
        meta_text = ss.str();
    }
    const guild_status_t pst = GKernelMetadata::parse_json(meta_text, out_meta, err);
    if (pst != GUILD_STATUS_OK) return pst;

    // Reject any undeclared shared libraries in the staged package
    std::set<std::string> declared_artifacts;
    for (const auto& a : out_meta.artifacts) declared_artifacts.insert(a.relative_path);
    for (const auto& p : staged_rel_paths) {
        if (p == ".gkmetadata") continue;
        if (is_shared_lib_extension(p) && declared_artifacts.count(p) == 0) {
            err = "security violation: undeclared shared library '" + p + "' present in package";
            return GUILD_STATUS_ERR_SECURITY_VIOLATION;
        }
    }

    err.clear();
    return GUILD_STATUS_OK;
}

// =============================================================================
// GKernelPackageManager & Dynamic Shared Library Loader
// =============================================================================

namespace {

struct SharedLibHandle {
    void* handle = nullptr;
    std::string path;
    explicit SharedLibHandle(void* h, std::string p) : handle(h), path(std::move(p)) {}
    ~SharedLibHandle() {
#if !defined(_WIN32)
        if (handle) {
            ::dlclose(handle);
            handle = nullptr;
        }
#endif
    }
};

std::string default_kernels_store_dir() {
    if (const char* env_dir = std::getenv("GUILD_DATA_DIR"); env_dir && env_dir[0]) {
        return (fs::path(env_dir) / "kernels").string();
    }
    if (const char* home = std::getenv("HOME"); home && home[0]) {
        return (fs::path(home) / ".guild" / "kernels").string();
    }
    return (fs::temp_directory_path() / "guild_kernels").string();
}

} // namespace

GKernelPackageManager::GKernelPackageManager(std::string store_root_dir,
                                             kernels::KernelRegistry* kernel_reg,
                                             model::ModelRunnerRegistry* runner_reg)
    : root_dir_(store_root_dir.empty() ? default_kernels_store_dir() : std::move(store_root_dir)),
      kernel_reg_(kernel_reg ? kernel_reg : &kernels::KernelRegistry::instance()),
      runner_reg_(runner_reg ? runner_reg : &model::ModelRunnerRegistry::instance()),
      policy_(PackageSecurityPolicy::default_builtin_policy()) {}

GKernelPackageManager::~GKernelPackageManager() = default;

GKernelPackageManager& GKernelPackageManager::instance() {
    static GKernelPackageManager mgr;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string ignore_err;
        mgr.init(ignore_err);
    });
    return mgr;
}

bool GKernelPackageManager::init(std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
    initialization_error_ = "package store initialization did not finish";
    std::error_code ec;
    fs::create_directories(fs::path(root_dir_) / "packages", ec);
    fs::create_directories(fs::path(root_dir_) / "staging", ec);
    if (ec) {
        err = "failed to create kernel package store directories: " + ec.message();
        return false;
    }
    const fs::path trust_path = fs::path(root_dir_) / "trust_store.json";
    if (fs::is_regular_file(trust_path, ec)) {
        if (!policy_.load_from_file(trust_path.string(), err)) {
            initialization_error_ = err;
            return false;
        }
    } else {
        policy_ = PackageSecurityPolicy::default_builtin_policy();
        if (!policy_.save_to_file(trust_path.string(), err)) {
            initialization_error_ = err;
            return false;
        }
    }
    const fs::path reg_path = fs::path(root_dir_) / "registry.json";
    if (fs::is_regular_file(reg_path, ec)) {
        initialized_ = load_registry_unlocked(err);
        initialization_error_ = err;
        return initialized_;
    }
    initialized_ = true;
    initialization_error_.clear();
    err.clear();
    return true;
}

bool GKernelPackageManager::save_security_policy(std::string& err) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const fs::path trust_path = fs::path(root_dir_) / "trust_store.json";
    return policy_.save_to_file(trust_path.string(), err);
}

bool GKernelPackageManager::save_registry_unlocked(std::string& err) const {
    JsonValue root = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
    root["format_version"] = static_cast<int64_t>(1);
    std::vector<JsonValue> pkg_arr;
    for (const auto& [id, rec] : packages_) {
        JsonValue o = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
        o["package_id"] = rec.package_id;
        o["active_version"] = rec.active_version;
        o["previous_version"] = rec.previous_version;
        o["enabled"] = rec.enabled;
        o["staged_dir"] = rec.staged_dir;
        JsonValue vmap = JsonValue(std::vector<std::pair<std::string, JsonValue>>{});
        for (const auto& [ver, vdir] : rec.version_dirs) {
            vmap[ver] = vdir;
        }
        o["version_dirs"] = vmap;
        pkg_arr.push_back(std::move(o));
    }
    root["packages"] = JsonValue(std::move(pkg_arr));

    std::error_code ec;
    fs::create_directories(root_dir_, ec);
    const std::string reg_path = (fs::path(root_dir_) / "registry.json").string();
    const std::string tmp_path = reg_path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        if (!f.good()) {
            err = "failed to write kernel package registry: " + tmp_path;
            return false;
        }
        const std::string s = root.serialize_pretty();
        f.write(s.data(), static_cast<std::streamsize>(s.size()));
    }
    fs::rename(tmp_path, reg_path, ec);
    if (ec) {
        err = "failed to commit kernel package registry: " + ec.message();
        return false;
    }
    err.clear();
    return true;
}

bool GKernelPackageManager::load_registry_unlocked(std::string& err) {
    const std::string reg_path = (fs::path(root_dir_) / "registry.json").string();
    std::ifstream f(reg_path, std::ios::binary);
    if (!f.good()) {
        err = "cannot open kernel package registry: " + reg_path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    JsonValue root;
    if (!JsonValue::parse(ss.str(), root, err) || !root.is_object()) {
        return false;
    }
    packages_.clear();
    const JsonValue& arr = root["packages"];
    if (arr.is_array()) {
        for (const auto& item : arr.arr_val) {
            if (!item.is_object()) continue;
            InstalledPackageRecord rec;
            rec.package_id = item["package_id"].as_string("");
            rec.active_version = item["active_version"].as_string("");
            rec.previous_version = item["previous_version"].as_string("");
            rec.enabled = item["enabled"].as_bool(true);
            rec.staged_dir = item["staged_dir"].as_string("");
            const JsonValue& vdirs = item["version_dirs"];
            if (vdirs.is_object()) {
                for (const auto& kv : vdirs.obj_val) {
                    if (kv.second.is_string()) rec.version_dirs[kv.first] = kv.second.as_string();
                }
            }
            if (rec.package_id.empty() || rec.staged_dir.empty()) continue;
            const fs::path meta_p = fs::path(rec.staged_dir) / ".gkmetadata";
            std::ifstream mf(meta_p, std::ios::binary);
            if (!mf.good()) continue;
            std::ostringstream mss;
            mss << mf.rdbuf();
            std::string merr;
            if (GKernelMetadata::parse_json(mss.str(), rec.metadata, merr) == GUILD_STATUS_OK) {
                packages_[rec.package_id] = std::move(rec);
            }
        }
    }
    err.clear();
    return true;
}

guild_status_t GKernelPackageManager::load_staged_package_unlocked(InstalledPackageRecord& rec, std::string& err) {
    if (rec.loaded_in_process) {
        kernel_reg_->set_package_enabled(rec.package_id, true);
        runner_reg_->set_package_enabled(rec.package_id, true);
        err.clear();
        return GUILD_STATUS_OK;
    }

    const auto host_caps = kernels::query_host_device_capabilities();
    // Re-verify metadata, signature, and on-disk SHA-256 digests immediately before dlopen
    const guild_status_t vst = validate_gkernel_package(rec.metadata, policy_, host_caps, rec.staged_dir, err);
    if (vst != GUILD_STATUS_OK) return vst;

    const std::string rel_lib = rec.metadata.resolve_primary_library_path();
    if (rel_lib.empty()) {
        // Pure declarative metadata package without shared library
        rec.loaded_in_process = true;
        err.clear();
        return GUILD_STATUS_OK;
    }

    const fs::path lib_path = fs::path(rec.staged_dir) / rel_lib;
#if defined(_WIN32)
    err = "dynamic shared library loading on Windows is not enabled in this build";
    return GUILD_STATUS_ERR_UNSUPPORTED_HARDWARE;
#else
    ::dlerror();
    void* raw_handle = ::dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!raw_handle) {
        const char* dl_err = ::dlerror();
        err = std::string("dlopen failed for '") + lib_path.string() + "': " + (dl_err ? dl_err : "unknown error");
        return GUILD_STATUS_ERR_EXECUTION_FAILED;
    }
    auto lib_keepalive = std::make_shared<SharedLibHandle>(raw_handle, lib_path.string());

    const std::string sym_name = rec.metadata.entrypoint_symbol.empty()
                                     ? std::string(GUILD_PLUGIN_ENTRY_SYMBOL_V1)
                                     : rec.metadata.entrypoint_symbol;
    ::dlerror();
    void* sym = ::dlsym(raw_handle, sym_name.c_str());
    if (!sym && sym_name == GUILD_PLUGIN_ENTRY_SYMBOL_V1) {
        sym = ::dlsym(raw_handle, "guild_plugin_get_descriptor_v1");
    }
    if (!sym) {
        const char* dl_err = ::dlerror();
        err = std::string("missing plugin entry symbol '") + sym_name + "': " + (dl_err ? dl_err : "not found");
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }

    auto entry_fn = reinterpret_cast<guild_plugin_entry_fn>(sym);
    auto runner_commit = [this](const guild_runner_desc_v1_t& rdesc, std::string& rerr) -> guild_status_t {
        return runner_reg_->register_abi_runner(rdesc, rerr);
    };
    auto runner_rollback = [this](const std::string& runner_id) {
        std::string ignore;
        runner_reg_->unregister_runner(runner_id, ignore);
    };

    const guild_status_t rst = kernel_reg_->register_plugin_transactional(
        entry_fn,
        rec.metadata.package_id,
        rec.metadata.version,
        rec.metadata.publisher_id,
        rec.metadata.provided_ops,
        rec.metadata.provided_runners,
        runner_commit,
        runner_rollback,
        err,
        lib_keepalive);
    if (rst != GUILD_STATUS_OK) {
        return rst;
    }

    kernel_reg_->set_package_enabled(rec.package_id, rec.enabled);
    runner_reg_->set_package_enabled(rec.package_id, rec.enabled);
    rec.loaded_in_process = true;
    err.clear();
    return GUILD_STATUS_OK;
#endif
}

guild_status_t GKernelPackageManager::unload_package_unlocked(InstalledPackageRecord& rec, std::string& err) {
    if (!rec.loaded_in_process) {
        err.clear();
        return GUILD_STATUS_OK;
    }
    if (kernel_reg_->active_package_leases(rec.package_id) > 0 ||
        runner_reg_->active_package_leases(rec.package_id) > 0 ||
        (!rec.metadata.provided_ops.empty() && kernel_reg_->active_sessions() > 0) ||
        (!rec.metadata.provided_runners.empty() && runner_reg_->active_sessions() > 0)) {
        err = "cannot unload package '" + rec.package_id + "' while active model/session leases are held";
        return GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY;
    }
    if (!rec.metadata.provided_runners.empty()) {
        const guild_status_t rst = runner_reg_->unregister_package(rec.package_id, err);
        if (rst != GUILD_STATUS_OK && rst != GUILD_STATUS_ERR_MISSING_RUNNER) {
            return rst;
        }
    }
    const guild_status_t kst = kernel_reg_->unregister_package(rec.package_id, err);
    if (kst != GUILD_STATUS_OK && kst != GUILD_STATUS_ERR_MISSING_KERNEL) {
        return kst;
    }
    rec.loaded_in_process = false;
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t GKernelPackageManager::install_package(const std::string& source_path,
                                                      bool enable_now,
                                                      std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::error_code ec;
    fs::create_directories(fs::path(root_dir_) / "staging", ec);
    const fs::path temp_stage = fs::path(root_dir_) / "staging" /
                                ("stage_" + std::to_string(std::rand()) + "_" + std::to_string(reinterpret_cast<uintptr_t>(this)));

    struct StageCleanup {
        fs::path p;
        ~StageCleanup() {
            std::error_code ignore;
            if (!p.empty()) fs::remove_all(p, ignore);
        }
    } cleanup{temp_stage};

    GKernelMetadata meta;
    const guild_status_t sst = stage_gkernel_source(source_path, temp_stage.string(), meta, err);
    if (sst != GUILD_STATUS_OK) return sst;

    const auto host_caps = kernels::query_host_device_capabilities();
    const guild_status_t vst = validate_gkernel_package(meta, policy_, host_caps, temp_stage.string(), err);
    if (vst != GUILD_STATUS_OK) return vst;

    // Validate dependency graph together with currently enabled packages
    std::vector<GKernelMetadata> graph_pkgs;
    for (const auto& [id, rec] : packages_) {
        if (id != meta.package_id && rec.enabled) {
            graph_pkgs.push_back(rec.metadata);
        }
    }
    graph_pkgs.push_back(meta);
    const guild_status_t dst = validate_package_dependency_graph(graph_pkgs, policy_, host_caps, err);
    if (dst != GUILD_STATUS_OK) return dst;

    // Check if an existing version is currently loaded
    auto existing_it = packages_.find(meta.package_id);
    std::string prev_version;
    std::string prev_staged_dir;
    bool had_existing = (existing_it != packages_.end());
    if (had_existing) {
        if (existing_it->second.active_version == meta.version && existing_it->second.loaded_in_process) {
            err = "package '" + meta.package_id + "@" + meta.version + "' is already installed and active";
            return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
        }
        prev_version = existing_it->second.active_version;
        prev_staged_dir = existing_it->second.staged_dir;
        if (existing_it->second.loaded_in_process) {
            const guild_status_t ust = unload_package_unlocked(existing_it->second, err);
            if (ust != GUILD_STATUS_OK) return ust;
        }
    }

    // Move verified staged directory into immutable store path <root_dir>/packages/<package_id>/<version>
    const fs::path final_dir = fs::path(root_dir_) / "packages" / meta.package_id / meta.version;
    fs::remove_all(final_dir, ec);
    fs::create_directories(final_dir.parent_path(), ec);
    fs::rename(temp_stage, final_dir, ec);
    if (ec) {
        err = "failed to move verified package into store: " + ec.message();
        return GUILD_STATUS_ERR_INTERNAL;
    }
    cleanup.p.clear(); // moved

    InstalledPackageRecord rec;
    if (had_existing) rec = existing_it->second;
    rec.package_id = meta.package_id;
    rec.previous_version = (prev_version != meta.version) ? prev_version : rec.previous_version;
    rec.active_version = meta.version;
    rec.enabled = enable_now;
    rec.loaded_in_process = false;
    rec.staged_dir = final_dir.string();
    rec.metadata = meta;
    rec.version_dirs[meta.version] = final_dir.string();

    if (enable_now) {
        const guild_status_t lst = load_staged_package_unlocked(rec, err);
        if (lst != GUILD_STATUS_OK) {
            fs::remove_all(final_dir, ec);
            // Restore previous version if it was loaded
            if (had_existing && !prev_version.empty() && !prev_staged_dir.empty()) {
                std::string restore_err;
                load_staged_package_unlocked(existing_it->second, restore_err);
            }
            return lst;
        }
    }

    packages_[meta.package_id] = std::move(rec);
    std::string save_err;
    if (!save_registry_unlocked(save_err)) {
        err = save_err;
        return GUILD_STATUS_ERR_INTERNAL;
    }
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t GKernelPackageManager::load_installed_package(const std::string& package_id, std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = packages_.find(package_id);
    if (it == packages_.end()) {
        err = "package not found: " + package_id;
        return GUILD_STATUS_ERR_MISSING_KERNEL;
    }
    return load_staged_package_unlocked(it->second, err);
}

guild_status_t GKernelPackageManager::load_enabled_packages(std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        err = initialization_error_;
        return GUILD_STATUS_ERR_SECURITY_VIOLATION;
    }
    std::vector<GKernelMetadata> enabled_metas;
    for (const auto& [id, rec] : packages_) {
        if (rec.enabled) enabled_metas.push_back(rec.metadata);
    }
    if (enabled_metas.empty()) {
        err.clear();
        return GUILD_STATUS_OK;
    }
    const auto host_caps = kernels::query_host_device_capabilities();
    const guild_status_t gst = validate_package_dependency_graph(enabled_metas, policy_, host_caps, err);
    if (gst != GUILD_STATUS_OK) return gst;

    // Topological load order
    std::set<std::string> loaded;
    std::function<guild_status_t(const std::string&)> load_rec = [&](const std::string& id) -> guild_status_t {
        if (loaded.count(id) > 0) return GUILD_STATUS_OK;
        auto it = packages_.find(id);
        if (it == packages_.end() || !it->second.enabled) return GUILD_STATUS_OK;
        for (const auto& dep : it->second.metadata.dependencies) {
            if (packages_.count(dep.package_id) > 0) {
                const guild_status_t st = load_rec(dep.package_id);
                if (st != GUILD_STATUS_OK) return st;
            }
        }
        const guild_status_t st = load_staged_package_unlocked(it->second, err);
        if (st == GUILD_STATUS_OK) loaded.insert(id);
        return st;
    };

    for (const auto& [id, rec] : packages_) {
        if (rec.enabled) {
            const guild_status_t st = load_rec(id);
            if (st != GUILD_STATUS_OK) return st;
        }
    }
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t GKernelPackageManager::enable_package(const std::string& package_id, std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = packages_.find(package_id);
    if (it == packages_.end()) {
        err = "package not found: " + package_id;
        return GUILD_STATUS_ERR_MISSING_KERNEL;
    }
    it->second.enabled = true;
    const guild_status_t st = load_staged_package_unlocked(it->second, err);
    if (st != GUILD_STATUS_OK) {
        it->second.enabled = false;
        return st;
    }
    kernel_reg_->set_package_enabled(package_id, true);
    runner_reg_->set_package_enabled(package_id, true);
    if (!save_registry_unlocked(err)) return GUILD_STATUS_ERR_INTERNAL;
    return GUILD_STATUS_OK;
}

guild_status_t GKernelPackageManager::disable_package(const std::string& package_id, std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = packages_.find(package_id);
    if (it == packages_.end()) {
        err = "package not found: " + package_id;
        return GUILD_STATUS_ERR_MISSING_KERNEL;
    }
    // Check if any other enabled package depends on this package
    for (const auto& [other_id, other_rec] : packages_) {
        if (other_id == package_id || !other_rec.enabled) continue;
        for (const auto& dep : other_rec.metadata.dependencies) {
            if (dep.package_id == package_id) {
                err = "cannot disable package '" + package_id + "' because enabled package '" + other_id + "' depends on it";
                return GUILD_STATUS_ERR_DEPENDENCY_MISSING;
            }
        }
    }
    const guild_status_t ust = unload_package_unlocked(it->second, err);
    if (ust != GUILD_STATUS_OK) return ust;
    it->second.enabled = false;
    kernel_reg_->set_package_enabled(package_id, false);
    runner_reg_->set_package_enabled(package_id, false);
    if (!save_registry_unlocked(err)) return GUILD_STATUS_ERR_INTERNAL;
    return GUILD_STATUS_OK;
}

guild_status_t GKernelPackageManager::remove_package(const std::string& package_id, std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = packages_.find(package_id);
    if (it == packages_.end()) {
        err = "package not found: " + package_id;
        return GUILD_STATUS_ERR_MISSING_KERNEL;
    }
    for (const auto& [other_id, other_rec] : packages_) {
        if (other_id == package_id) continue;
        for (const auto& dep : other_rec.metadata.dependencies) {
            if (dep.package_id == package_id) {
                err = "cannot remove package '" + package_id + "' because installed package '" + other_id + "' depends on it";
                return GUILD_STATUS_ERR_DEPENDENCY_MISSING;
            }
        }
    }
    const guild_status_t ust = unload_package_unlocked(it->second, err);
    if (ust != GUILD_STATUS_OK) return ust;

    std::error_code ec;
    fs::remove_all(fs::path(root_dir_) / "packages" / package_id, ec);
    packages_.erase(it);
    if (!save_registry_unlocked(err)) return GUILD_STATUS_ERR_INTERNAL;
    return GUILD_STATUS_OK;
}

guild_status_t GKernelPackageManager::rollback_package(const std::string& package_id, std::string& err) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = packages_.find(package_id);
    if (it == packages_.end()) {
        err = "package not found: " + package_id;
        return GUILD_STATUS_ERR_MISSING_KERNEL;
    }
    auto& rec = it->second;
    if (rec.previous_version.empty() || rec.version_dirs.count(rec.previous_version) == 0) {
        err = "no previous version available to roll back for package '" + package_id + "'";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    const std::string target_ver = rec.previous_version;
    const std::string target_dir = rec.version_dirs.at(target_ver);
    const fs::path meta_p = fs::path(target_dir) / ".gkmetadata";
    std::ifstream mf(meta_p, std::ios::binary);
    if (!mf.good()) {
        err = "previous version directory is missing .gkmetadata: " + target_dir;
        return GUILD_STATUS_ERR_INTERNAL;
    }
    std::ostringstream ss;
    ss << mf.rdbuf();
    GKernelMetadata prev_meta;
    const guild_status_t pst = GKernelMetadata::parse_json(ss.str(), prev_meta, err);
    if (pst != GUILD_STATUS_OK) return pst;

    const guild_status_t ust = unload_package_unlocked(rec, err);
    if (ust != GUILD_STATUS_OK) return ust;

    const std::string old_active = rec.active_version;
    const std::string old_dir = rec.staged_dir;
    const GKernelMetadata old_meta = rec.metadata;

    rec.active_version = target_ver;
    rec.previous_version = old_active;
    rec.staged_dir = target_dir;
    rec.metadata = prev_meta;

    if (rec.enabled) {
        const guild_status_t lst = load_staged_package_unlocked(rec, err);
        if (lst != GUILD_STATUS_OK) {
            rec.active_version = old_active;
            rec.previous_version = target_ver;
            rec.staged_dir = old_dir;
            rec.metadata = old_meta;
            std::string restore_err;
            load_staged_package_unlocked(rec, restore_err);
            return lst;
        }
    }
    if (!save_registry_unlocked(err)) return GUILD_STATUS_ERR_INTERNAL;
    return GUILD_STATUS_OK;
}

std::vector<InstalledPackageRecord> GKernelPackageManager::list_packages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<InstalledPackageRecord> out;
    out.reserve(packages_.size());
    for (const auto& [id, rec] : packages_) out.push_back(rec);
    return out;
}

bool GKernelPackageManager::get_package(const std::string& package_id, InstalledPackageRecord& out_rec) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = packages_.find(package_id);
    if (it == packages_.end()) return false;
    out_rec = it->second;
    return true;
}

} // namespace guild::abi
