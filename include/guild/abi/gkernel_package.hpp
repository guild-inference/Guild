// include/guild/abi/gkernel_package.hpp - Declarative .gkmetadata Specification & Security Policy
#pragma once

#include "guild/abi/kernel_abi.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace guild::abi {

struct PackageDependency {
    std::string package_id;
    std::string min_version;
    std::string expected_publisher_id;
};

struct ArtifactHashEntry {
    std::string relative_path;
    std::string sha256;
    uint64_t byte_size = 0;
};

struct TensorConstraintSpec {
    std::string op_name;
    std::vector<std::string> allowed_dtypes;
    int64_t max_batch_rows = 0;
    int64_t alignment_bytes = 32;
};

struct ValidationFixtureSpec {
    std::string fixture_id;
    std::string relative_path;
    std::string sha256;
    double max_abs_tol = 0.0;
    double rel_l2_tol = 0.0;
};

struct SignatureMetadata {
    std::string algorithm;      // e.g. "ed25519"
    std::string key_id;         // e.g. "illatio-release-2026-01"
    std::string signature_hex;  // 128 hex chars (64-byte signature)
    std::string payload_sha256; // 64 hex chars (SHA-256 of canonical unsigned payload)
};

struct ProvenanceMetadata {
    std::string spdx_license;    // e.g. "MIT"
    std::string source_repo;     // e.g. "https://github.com/guild-inference/guild-kernels-cuda"
    std::string source_revision; // 40-char git commit SHA
};

struct GKernelMetadata {
    uint32_t format_version = 1;
    std::string package_id;      // e.g. "guild-kernels-cuda"
    std::string scope;           // "cpu-kernels", "cuda-kernels", "hip-kernels", "runner", "hybrid"
    std::string version;         // Semantic version "MAJOR.MINOR.PATCH"
    std::string publisher_id;    // e.g. "org.illatio.guild"

    // Kernel ABI compatibility range
    uint32_t abi_major = 1;
    uint32_t abi_minor_min = 0;
    uint32_t abi_minor_max = 0;

    // Capabilities & hardware support
    guild_core_cap_flags_t required_core_caps = GUILD_CORE_CAP_NONE;
    guild_backend_t supported_backends = GUILD_BACKEND_NONE;
    guild_cpu_isa_flags_t required_cpu_isa = GUILD_ISA_NONE;
    guild_gpu_cap_flags_t required_gpu_caps = GUILD_GPU_CAP_NONE;
    int32_t min_sm_version = 0;
    std::vector<std::string> supported_architectures; // e.g. ["qwen4exp", "qwen35moe"]

    // Operations & dependencies
    std::vector<std::string> provided_ops;
    std::vector<std::string> provided_runners;
    std::vector<std::string> required_ops;
    std::vector<PackageDependency> dependencies;

    // Constraints, artifacts, signing, provenance, fixtures
    std::vector<TensorConstraintSpec> tensor_constraints;
    std::vector<ArtifactHashEntry> artifacts;
    SignatureMetadata signature;
    ProvenanceMetadata provenance;
    std::vector<ValidationFixtureSpec> validation_fixtures;

    /// Parse a declarative `.gkmetadata` JSON string without executing any external code.
    /// Rejects any manifest containing installation scripts or unsafe paths.
    static guild_status_t parse_json(const std::string& json_text,
                                     GKernelMetadata& out_meta,
                                     std::string& err);

    /// Serialize to canonical JSON string.
    std::string to_json_string() const;
};

/// Publisher trust store and revocation policy for external `.gkernel` packages.
struct PackageSecurityPolicy {
    bool require_signature = true;
    bool verify_artifact_hashes = true;
    // Map of publisher_id -> set of trusted key_ids
    std::map<std::string, std::set<std::string>> trusted_publishers;
    // Revoked signing key_ids
    std::set<std::string> revoked_key_ids;
    // Revoked "package_id@version" releases
    std::set<std::string> revoked_releases;

    void trust_publisher_key(const std::string& publisher_id, const std::string& key_id) {
        trusted_publishers[publisher_id].insert(key_id);
    }
    void revoke_key(const std::string& key_id) { revoked_key_ids.insert(key_id); }
    void revoke_release(const std::string& package_id, const std::string& version) {
        revoked_releases.insert(package_id + "@" + version);
    }
};

/// Frozen kernel binding entry for GEXEC artifacts.
/// Ensures rolling kernel updates never silently change a compiled GEXEC plan.
struct GexecKernelLock {
    std::string package_id;
    std::string package_version;
    std::string artifact_sha256;
    std::string impl_id;
    uint32_t abi_version = GUILD_KERNEL_ABI_VERSION_V1;
};

/// Validate a parsed `.gkmetadata` against the security policy, host ABI, hardware capabilities,
/// and (optionally) on-disk artifact SHA-256 digests under `package_dir`.
guild_status_t validate_gkernel_package(const GKernelMetadata& meta,
                                        const PackageSecurityPolicy& policy,
                                        const guild_device_capabilities_v1_t& host_caps,
                                        const std::string& package_dir,
                                        std::string& err);

/// Validate a set of packages and their dependency graph (detecting missing dependencies,
/// untrusted dependencies, version mismatches, and dependency cycles).
guild_status_t validate_package_dependency_graph(const std::vector<GKernelMetadata>& packages,
                                                 const PackageSecurityPolicy& policy,
                                                 const guild_device_capabilities_v1_t& host_caps,
                                                 std::string& err);

/// Verify that a candidate package & kernel descriptor match a frozen GEXEC lock.
guild_status_t verify_gexec_kernel_lock(const GexecKernelLock& lock,
                                        const GKernelMetadata& meta,
                                        const guild_kernel_op_desc_v1_t& op_desc,
                                        std::string& err);

} // namespace guild::abi
