// include/guild/abi/gkernel_package.hpp - Declarative .gkmetadata Specification, Ed25519 Security Policy & Package Manager
#pragma once

#include "guild/abi/ed25519.hpp"
#include "guild/abi/kernel_abi.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace guild:: kernels {
class KernelRegistry;
}
namespace guild::model {
class ModelRunnerRegistry;
}

namespace guild::abi {

/// Official Guild release publisher & key constants used for default trust store and SDK signing.
inline constexpr const char* kOfficialGuildPublisherId = "org.illatio.guild";
inline constexpr const char* kOfficialGuildReleaseKeyId = "illatio-release-2026-01";

struct PackageDependency {
    std::string package_id;
    std::string min_version;
    std::string expected_publisher_id;
    std::string max_version;
};

/// Dependency boundary for reusable `.gkmodule` building blocks consumed by `.gkernel` packages.
/// `.gkmodule` packages are never directly executed or registered as plugins by Guild Core.
struct ModuleDependencySpec {
    std::string module_id;
    std::string min_version;
    std::string abi_tag;
    std::string expected_publisher_id;
    std::string sha256;
};

struct TargetPlatformSpec {
    std::string os = "any";   // "any", "linux", "windows", "macos"
    std::string arch = "any"; // "any", "x86_64", "aarch64"
};

struct ArtifactHashEntry {
    std::string relative_path;
    std::string sha256;
    uint64_t byte_size = 0;
    std::string role; // Optional: "shared_library", "fixture", "module", "metadata"
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
    std::string algorithm;      // "ed25519"
    std::string key_id;         // e.g. "illatio-release-2026-01"
    std::string signature_hex;  // 128 hex chars (64-byte Ed25519 signature)
    std::string payload_sha256; // 64 hex chars (SHA-256 of canonical unsigned manifest payload)
};

struct ProvenanceMetadata {
    std::string spdx_license;    // e.g. "MIT"
    std::string source_repo;     // e.g. "https://github.com/guild-inference/Guild"
    std::string source_revision; // 40-char git commit SHA
};

struct GKernelMetadata {
    uint32_t format_version = 1;
    std::string package_kind = "gkernel"; // "gkernel" (executable plugin) or "gkmodule" (non-executable module)
    std::string package_id;               // e.g. "guild-kernels-cuda"
    std::string scope;                    // "cpu-kernels", "cuda-kernels", "hip-kernels", "runner", "hybrid", "module"
    std::string version;                  // Semantic version "MAJOR.MINOR.PATCH"
    std::string publisher_id;             // e.g. "org.illatio.guild"
    std::string entrypoint_symbol = GUILD_PLUGIN_ENTRY_SYMBOL_V1;
    std::string library_path;             // Optional primary shared library relative path in artifacts

    // Kernel ABI compatibility range
    uint32_t abi_major = 1;
    uint32_t abi_minor_min = 0;
    uint32_t abi_minor_max = 0;

    // Platform & hardware capability constraints
    TargetPlatformSpec target_platform{};
    guild_core_cap_flags_t required_core_caps = GUILD_CORE_CAP_NONE;
    guild_backend_t supported_backends = GUILD_BACKEND_NONE;
    guild_cpu_isa_flags_t required_cpu_isa = GUILD_ISA_NONE;
    guild_gpu_cap_flags_t required_gpu_caps = GUILD_GPU_CAP_NONE;
    int32_t min_sm_version = 0;
    int32_t max_sm_version = 0;
    std::vector<std::string> supported_architectures; // e.g. ["qwen4exp", "qwen35moe"]

    // Operations & dependencies
    std::vector<std::string> provided_ops;
    std::vector<std::string> provided_runners;
    std::vector<std::string> required_ops;
    std::vector<PackageDependency> dependencies;
    std::vector<ModuleDependencySpec> module_dependencies;

    // Constraints, artifacts, signing, provenance, fixtures
    std::vector<TensorConstraintSpec> tensor_constraints;
    std::vector<ArtifactHashEntry> artifacts;
    SignatureMetadata signature;
    ProvenanceMetadata provenance;
    std::vector<ValidationFixtureSpec> validation_fixtures;

    /// Parse a declarative `.gkmetadata` JSON string without executing any external code.
    /// Rejects any manifest containing installation scripts, unknown keys, or unsafe paths.
    static guild_status_t parse_json(const std::string& json_text,
                                     GKernelMetadata& out_meta,
                                     std::string& err);

    /// Serialize canonical unsigned payload JSON string (deterministic field order, excludes `signature`).
    std::string canonical_signing_payload() const;

    /// Compute 64-char hex SHA-256 digest of `canonical_signing_payload()`.
    std::string compute_payload_sha256() const;

    /// Domain-separated message signed by Ed25519: "GUILD-GKERNEL-SIG-V1:<payload_sha256>".
    std::string signing_message() const;

    /// Sign this metadata in-place using an Ed25519 private 32-byte seed (64 hex chars).
    bool sign_with_seed_hex(const std::string& key_id,
                            const std::string& private_seed_hex,
                            std::string& err);

    /// Serialize full `.gkmetadata` to pretty-printed JSON string.
    std::string to_json_string() const;

    /// Resolve the primary shared library relative path from `library_path` or `artifacts`.
    std::string resolve_primary_library_path() const;
};

/// Publisher trust store and revocation policy for external `.gkernel` packages.
struct PackageSecurityPolicy {
    bool require_signature = true;
    bool allow_unsigned_dev = false;
    bool verify_artifact_hashes = true;
    // Map of publisher_id -> set of trusted key_ids
    std::map<std::string, std::set<std::string>> trusted_publishers;
    // Map of (publisher_id, key_id) -> 64-char hex Ed25519 public key
    std::map<std::pair<std::string, std::string>, std::string> trusted_public_keys;
    // Revoked signing key_ids
    std::set<std::string> revoked_key_ids;
    // Revoked publisher_ids
    std::set<std::string> revoked_publishers;
    // Revoked "package_id@version" releases
    std::set<std::string> revoked_releases;

    void trust_publisher_key(const std::string& publisher_id,
                             const std::string& key_id,
                             const std::string& public_key_hex = "") {
        trusted_publishers[publisher_id].insert(key_id);
        if (!public_key_hex.empty()) {
            trusted_public_keys[{publisher_id, key_id}] = public_key_hex;
        }
    }
    void revoke_key(const std::string& key_id) { revoked_key_ids.insert(key_id); }
    void revoke_publisher(const std::string& publisher_id) { revoked_publishers.insert(publisher_id); }
    void revoke_release(const std::string& package_id, const std::string& version) {
        revoked_releases.insert(package_id + "@" + version);
    }

    static PackageSecurityPolicy default_builtin_policy();
    bool save_to_file(const std::string& path, std::string& err) const;
    bool load_from_file(const std::string& path, std::string& err);
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

/// Create a deterministic single-file `.gkernel` archive bundle from a package directory containing `.gkmetadata`.
guild_status_t create_gkernel_bundle(const std::string& source_dir,
                                     const std::string& output_bundle_path,
                                     std::string& err);

/// Hardened staging & extraction of a `.gkernel` directory or archive (`GKPK` or `.tar`) into `staging_dir`.
/// Defends against path traversal, symlinks, duplicate entries, and zip/tar bombs before parsing `.gkmetadata`.
guild_status_t stage_gkernel_source(const std::string& source_path,
                                    const std::string& staging_dir,
                                    GKernelMetadata& out_meta,
                                    std::string& err,
                                    uint64_t max_total_bytes = (256ull << 20),
                                    size_t max_entries = 64);

/// Installed package record managed by GKernelPackageManager.
struct InstalledPackageRecord {
    std::string package_id;
    std::string active_version;
    std::string previous_version;
    bool enabled = true;
    bool loaded_in_process = false;
    std::string staged_dir;
    GKernelMetadata metadata;
    std::map<std::string, std::string> version_dirs; // version -> immutable staged path
};

/// Persistent local package manager & dynamic loader for `.gkernel` plugins.
class GKernelPackageManager {
public:
    explicit GKernelPackageManager(std::string store_root_dir = "",
                                   kernels::KernelRegistry* kernel_reg = nullptr,
                                   model::ModelRunnerRegistry* runner_reg = nullptr);
    ~GKernelPackageManager();

    /// Process-wide singleton using the default Guild store directory and global registries.
    static GKernelPackageManager& instance();

    /// Initialize store directories, load persisted trust policy and package registry.
    bool init(std::string& err);

    const std::string& store_root_dir() const { return root_dir_; }
    PackageSecurityPolicy& security_policy() { return policy_; }
    const PackageSecurityPolicy& security_policy() const { return policy_; }
    bool save_security_policy(std::string& err) const;

    /// Install a `.gkernel` package from a directory or `.gkernel` archive file.
    /// Verifies metadata, Ed25519 signature, ABI/hardware compatibility, dependency graph, and artifact hashes
    /// before staging into the immutable package store and (if `enable_now` is true) dynamically loading it.
    guild_status_t install_package(const std::string& source_path,
                                   bool enable_now,
                                   std::string& err);

    /// Load and register an already-installed package into the active registries.
    guild_status_t load_installed_package(const std::string& package_id, std::string& err);

    /// Load all enabled packages from the persistent store in topological dependency order.
    guild_status_t load_enabled_packages(std::string& err);

    /// Enable an installed package (loads its shared library and enables its kernels/runners).
    guild_status_t enable_package(const std::string& package_id, std::string& err);

    /// Disable an installed package (unregisters and unloads if no active session lease holds it).
    guild_status_t disable_package(const std::string& package_id, std::string& err);

    /// Remove an installed package (fails if active session lease holds it or another enabled package depends on it).
    guild_status_t remove_package(const std::string& package_id, std::string& err);

    /// Roll back `package_id` to its `previous_version` (if available).
    guild_status_t rollback_package(const std::string& package_id, std::string& err);

    /// List all installed packages.
    std::vector<InstalledPackageRecord> list_packages() const;

    /// Inspect a specific installed package by `package_id` (or inspect a path on disk before installation).
    bool get_package(const std::string& package_id, InstalledPackageRecord& out_rec) const;

private:
    guild_status_t load_staged_package_unlocked(InstalledPackageRecord& rec, std::string& err);
    guild_status_t unload_package_unlocked(InstalledPackageRecord& rec, std::string& err);
    bool save_registry_unlocked(std::string& err) const;
    bool load_registry_unlocked(std::string& err);

    std::string root_dir_;
    kernels::KernelRegistry* kernel_reg_ = nullptr;
    model::ModelRunnerRegistry* runner_reg_ = nullptr;
    PackageSecurityPolicy policy_;
    bool initialized_ = false;
    std::string initialization_error_;
    mutable std::mutex mutex_;
    std::map<std::string, InstalledPackageRecord> packages_;
};

} // namespace guild::abi
