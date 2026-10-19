// include/guild/model/runner.hpp - Modular Model Architecture Runner Interface & Registry
#pragma once

#include "guild/abi/kernel_abi.h"
#include "guild/artifact/gguf_reader.hpp"
#include "guild/core/layout.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/memory/plan.hpp"
#include "guild/model/archetype.hpp"
#include "guild/model/model_descriptor.hpp"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace guild::model {

enum class PrefillStrategy {
    SequentialVerifier = 0,
    DedicatedChunkedGemm = 1,
    LayerWiseBatched = 2
};

struct RunnerTensorMap {
    std::string embedding_tensor = "token_embd.weight";
    std::string output_tensor = "output.weight";
    std::string output_norm_tensor = "output_norm.weight";
    std::string ple_table_tensor = "per_layer_token_embd.weight";
    std::string ple_key_tensor = "blk.1.ple_key.weight";
    std::string layer_prefix = "blk.";
};

struct RunnerExecutionPlan {
    std::string runner_id;
    std::string arch_id;
    PrefillStrategy prefill_strategy = PrefillStrategy::SequentialVerifier;
    bool requires_ple = false;
    bool supports_mtp = false;
    guild_core_cap_flags_t required_core_caps = GUILD_CORE_CAP_NONE;
    std::vector<guild_op_kind_t> required_ops;
    std::vector<std::string> resolved_kernel_ids;
    std::vector<std::string> resolved_package_ids;
    std::vector<kernels::KernelBinding> bindings;
};

struct RunnerInfoSnapshot {
    std::string runner_id;
    std::string arch_id;
    std::string package_id;
    uint32_t runner_version = 0;
    guild_core_cap_flags_t required_core_caps = GUILD_CORE_CAP_NONE;
    std::vector<guild_op_kind_t> required_ops;
    std::vector<std::string> required_package_deps;
    bool enabled = true;
    int64_t active_package_leases = 0;
};

/// Interface implemented by each model architecture runner.
/// Guild Core delegates architecture recognition, tensor mapping, geometry
/// derivation, layer-topology validation, and capability requirements to
/// IModelRunner while retaining ownership of memory planning, expert caching,
/// session lifecycle, and kernel dispatch.
class IModelRunner {
public:
    virtual ~IModelRunner() = default;

    virtual const char* runner_id() const = 0;
    virtual const char* package_id() const = 0;
    virtual const char* arch_id() const = 0;
    virtual uint32_t runner_version() const = 0;
    virtual int32_t priority() const { return 0; }
    virtual ModelArchetype archetype() const = 0;

    /// Returns true if this runner supports the given architecture string or alias.
    virtual bool matches_arch_name(const std::string& name) const = 0;

    /// Returns true if this runner recognizes the GGUF metadata shard.
    virtual bool matches_gguf(const GgufFile& gguf) const = 0;

    /// Canonical tensor naming map for this architecture.
    virtual const RunnerTensorMap& tensor_map() const = 0;

    /// Validate GGUF metadata/tensors and populate ModelDescriptor.
    virtual bool validate_and_describe(const GgufModel& artifact,
                                       ModelDescriptor& out_desc,
                                       std::string& err) const = 0;

    /// Populate execution geometry and active expert count K from ModelDescriptor.
    virtual bool configure_geometry(const ModelDescriptor& desc,
                                    core::ModelGeometry& out_geom,
                                    int64_t& out_k_active,
                                    std::string& err) const = 0;

    /// Determine which canonical pack tensors should be skipped when loading WeightTable.
    virtual bool configure_weight_skip_set(bool native_pack,
                                           bool has_shards,
                                           const std::string& pack_dir,
                                           std::set<std::string>& inout_skip,
                                           std::string& err) const = 0;

    /// Build the execution capability and operation requirements for a loaded model and plan.
    virtual bool build_execution_plan(const ModelDescriptor& desc,
                                      const core::ModelGeometry& geom,
                                      const memory::ExecutionPlan& mem_plan,
                                      RunnerExecutionPlan& out_exec,
                                      std::string& err) const = 0;

    /// Architecture-specific numerical validation of output buffers.
    virtual bool validate_numerical(const float* values,
                                    size_t count,
                                    const std::string& label,
                                    std::string& err) const = 0;

    /// Export C-compatible ABI v1 runner descriptor.
    virtual guild_runner_desc_v1_t abi_descriptor() const = 0;
};

/// Registry of executable model architecture runners.
class ModelRunnerRegistry {
public:
    ModelRunnerRegistry();
    ~ModelRunnerRegistry();

    /// Process-wide singleton initialized with built-in Qwen4Exp and Qwen35MoE (Ornith) runners.
    static ModelRunnerRegistry& instance();

    /// Register built-in statically linked runners.
    void register_builtin_runners();

    /// Register a C++ runner implementation.
    guild_status_t register_runner(std::unique_ptr<IModelRunner> runner, std::string& err);

    /// Register a C ABI v1 runner descriptor.
    guild_status_t register_abi_runner(const guild_runner_desc_v1_t& desc, std::string& err,
                                      std::shared_ptr<void> library_keepalive = nullptr);

    /// Unregister a runner by runner_id. Fails with GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY if sessions are active.
    guild_status_t unregister_runner(const std::string& runner_id, std::string& err);

    /// Unregister all runners belonging to `package_id`.
    guild_status_t unregister_package(const std::string& package_id, std::string& err);

    /// Enable or disable all runners belonging to `package_id` for future resolutions.
    void set_package_enabled(const std::string& package_id, bool enabled);
    bool is_package_enabled(const std::string& package_id) const;

    /// Resolve a runner for a GGUF metadata file.
    guild_status_t resolve_for_gguf(const GgufFile& gguf,
                                    std::shared_ptr<const IModelRunner>* out_runner,
                                    std::string& err) const;

    /// Resolve a runner by architecture tag or alias.
    guild_status_t resolve_by_arch(const std::string& arch_name,
                                   std::shared_ptr<const IModelRunner>* out_runner,
                                   std::string& err) const;

    /// Resolve a runner by exact runner_id.
    std::shared_ptr<const IModelRunner> find_by_id(const std::string& runner_id) const;

    /// Validate that a runner's required Core capabilities and kernel operations can be satisfied.
    guild_status_t validate_runner_requirements(const IModelRunner& runner,
                                                const ModelDescriptor& desc,
                                                const core::ModelGeometry& geom,
                                                const memory::ExecutionPlan& mem_plan,
                                                const kernels::KernelRegistry& kernel_reg,
                                                const guild_device_capabilities_v1_t& caps,
                                                RunnerExecutionPlan& out_exec,
                                                std::string& err) const;

    /// Snapshot of registered runner ABI descriptors.
    std::vector<std::shared_ptr<const IModelRunner>> list_runners() const;

    /// Value-owning snapshots of registered runners (safe without holding registry lock).
    std::vector<RunnerInfoSnapshot> list_runner_snapshots() const;

    void acquire_session_lease() const { active_sessions_.fetch_add(1, std::memory_order_acq_rel); }
    void release_session_lease() const { active_sessions_.fetch_sub(1, std::memory_order_acq_rel); }
    int64_t active_sessions() const { return active_sessions_.load(std::memory_order_acquire); }

    void acquire_package_lease(const std::string& package_id) const;
    void release_package_lease(const std::string& package_id) const;
    int64_t active_package_leases(const std::string& package_id) const;

private:
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<IModelRunner>> runners_;
    std::set<std::string> disabled_packages_;
    mutable std::map<std::string, int64_t> package_leases_;
    mutable std::atomic<int64_t> active_sessions_{0};
};

/// RAII lease guard held by active GuildModel / GuildSession instances.
class SessionRunnerLease {
public:
    SessionRunnerLease() = default;
    explicit SessionRunnerLease(const ModelRunnerRegistry* reg, std::string package_id = "")
        : reg_(reg), package_id_(std::move(package_id)) {
        if (reg_) {
            reg_->acquire_session_lease();
            if (!package_id_.empty()) reg_->acquire_package_lease(package_id_);
        }
    }
    ~SessionRunnerLease() { reset(); }
    SessionRunnerLease(const SessionRunnerLease&) = delete;
    SessionRunnerLease& operator=(const SessionRunnerLease&) = delete;
    SessionRunnerLease(SessionRunnerLease&& other) noexcept
        : reg_(other.reg_), package_id_(std::move(other.package_id_)) {
        other.reg_ = nullptr;
        other.package_id_.clear();
    }
    SessionRunnerLease& operator=(SessionRunnerLease&& other) noexcept {
        if (this != &other) {
            reset();
            reg_ = other.reg_;
            package_id_ = std::move(other.package_id_);
            other.reg_ = nullptr;
            other.package_id_.clear();
        }
        return *this;
    }
    void reset() {
        if (reg_) {
            if (!package_id_.empty()) reg_->release_package_lease(package_id_);
            reg_->release_session_lease();
            reg_ = nullptr;
            package_id_.clear();
        }
    }
private:
    const ModelRunnerRegistry* reg_ = nullptr;
    std::string package_id_;
};

} // namespace guild::model
