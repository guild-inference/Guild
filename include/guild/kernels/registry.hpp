// include/guild/kernels/registry.hpp - Hardware Kernel Operation Registry & Selection
#pragma once

#include "guild/abi/kernel_abi.h"
#include "guild/hardware/hardware_info.hpp"
#include "guild/kernels/cuda_attn.hpp"
#include "guild/kernels/cuda_dense.hpp"
#include "guild/kernels/cuda_moe.hpp"
#include "guild/kernels/rope_scaling.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace guild::kernels {

/// Convert a status code to a human-readable string.
const char* status_to_string(guild_status_t status);

/// Convert an operation kind to its canonical name.
const char* op_kind_to_string(guild_op_kind_t kind);

/// Negotiate Kernel ABI v1 compatibility.
guild_status_t negotiate_abi_version(uint32_t requested_version, uint32_t* out_negotiated_version);

/// Validate a buffer descriptor (null data, zero size, expired lifetime, alignment).
guild_status_t validate_buffer_desc(const guild_buffer_desc_t& buf, guild_error_info_t* err = nullptr);

/// Validate a tensor descriptor (rank, positive dimensions, block alignment for quantized types, stride bounds, buffer).
guild_status_t validate_tensor_desc(const guild_tensor_desc_t& tensor, guild_error_info_t* err = nullptr);

/// Build a v1 device capability descriptor from HardwareInfo and active GPU device properties.
guild_device_capabilities_v1_t query_device_capabilities(const hardware::HardwareInfo& hw);
guild_device_capabilities_v1_t query_host_device_capabilities();

/// Selection criteria for resolving a hardware kernel implementation.
struct KernelSelectionQuery {
    guild_op_kind_t op_kind = GUILD_OP_INVALID;
    guild_backend_t allowed_backends = GUILD_BACKEND_CPU | GUILD_BACKEND_CUDA | GUILD_BACKEND_HIP;
    guild_dtype_t weight_dtype = GUILD_DTYPE_ANY;
    guild_dtype_t input_dtype = GUILD_DTYPE_ANY;
    guild_dtype_t output_dtype = GUILD_DTYPE_ANY;
    guild_dtype_t accum_dtype = GUILD_DTYPE_ANY;
    int64_t batch_rows = 1;
    int64_t dim_n = 0;
    int64_t dim_k = 0;
    guild_numerical_flags_t required_numerical_flags = GUILD_NUM_NONE;
    guild_sync_mode_t required_sync_mode = GUILD_SYNC_NONE;
    uint64_t available_vram_bytes = UINT64_MAX;
    uint64_t available_ram_bytes = UINT64_MAX;
    // Optional GEXEC lock: if non-empty, only this exact implementation ID (and optional version) may match.
    std::string frozen_impl_id;
    uint32_t frozen_impl_version = 0;
};

/// Frozen seven-pointer payload for guild_kernel_op_desc_v1_t::direct_fn_table.
/// Keep this layout unchanged so plugins built against the original table remain readable.
/// New entry points belong in the tagged next_ext payload below, never after these pointers.
struct PrefillKernelTableV1 {
    void (*quantize_q8_1)(const float* x, void* vx, int K, int ncols, void* stream) = nullptr;
    void (*mmvq)(int type, const void* w, const void* x, float* y, int ncols_x, int nrows_x, int ncols_y, void* stream) = nullptr;
    void (*rms_norm_weighted)(const float* x, const float* w, float* y, int ncols, int nrows, float eps, void* stream) = nullptr;
    void (*gdn_conv)(const float* history, const float* qkv, const float* conv_w, float* h,
                     int channels, int qk_heads, float eps, int n_tok, void* stream) = nullptr;
    void (*gdn_recurrence)(float* state, const float* h, int channels, const float* gate,
                           const float* beta, const float* z, const float* gamma, float eps,
                           float* y, int h_k, int h_v, int n_tok, const int32_t* keep,
                           bool gate_silu, void* stream) = nullptr;
    void (*gdn_out_norm)(const float* output, const float* z, const float* gamma,
                         float* destination, int64_t heads, int64_t cols,
                         float epsilon, void* stream) = nullptr;
    void (*rope_apply)(const float* x, float* out, int rows, int head_dim,
                         int n_rot, const RopeScaling& scaling, const int* positions, void* stream) = nullptr;
};

inline constexpr uint32_t kPrefillKernelExtensionTag = 0x50464b45u; // "PFKE"
inline constexpr uint32_t kPrefillKernelExtensionVersion = 1;

/// A next_ext payload starts with this header. Read struct_size before the other fields.
struct PrefillKernelExtensionHeader {
    uint32_t struct_size = sizeof(PrefillKernelExtensionHeader);
    uint32_t tag = kPrefillKernelExtensionTag;
    uint32_t version = kPrefillKernelExtensionVersion;
};

/// Optional guild_kernel_op_desc_v1_t::next_ext payload for prefill projections.
/// GDN A/B and router descriptors require this extension; legacy operations do not.
/// Validate header size, tag, version and full payload size before reading either pointer.
/// Both payloads are borrowed and must outlive the registry's active session leases.
struct PrefillKernelExtensionV1 {
    PrefillKernelExtensionHeader header{sizeof(PrefillKernelExtensionV1),
                                        kPrefillKernelExtensionTag, kPrefillKernelExtensionVersion};
    void (*gdn_ab)(const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt, const float* a,
                    float* gate, float* beta, int n, int heads, int rows, void* stream) = nullptr;
    void (*router_projection)(const float* x, const uint16_t* w, float* y,
                               int64_t n, int64_t outputs, int rows, void* stream) = nullptr;
};

static_assert(sizeof(PrefillKernelExtensionHeader) == sizeof(guild_prefill_ext_header_t),
              "PrefillKernelExtensionHeader must match C ABI guild_prefill_ext_header_t");
static_assert(sizeof(PrefillKernelExtensionV1) == sizeof(guild_prefill_kernel_ext_v1_t),
              "PrefillKernelExtensionV1 must match C ABI guild_prefill_kernel_ext_v1_t");

/// Runtime-only convenience table resolved once during BatchedPrefill::init().
/// This is not the descriptor payload: direct_fn_table exposes only PrefillKernelTableV1.
struct ResolvedPrefillKernels : PrefillKernelTableV1 {
    decltype(PrefillKernelExtensionV1::gdn_ab) gdn_ab = nullptr;
    decltype(PrefillKernelExtensionV1::router_projection) router_projection = nullptr;
    ResolvedCudaDense cuda_dense{};
    bool has_cuda_dense = false;
    ResolvedCudaAttn cuda_attn{};
    bool has_cuda_attn = false;
    ResolvedCudaMoE cuda_moe{};
    bool has_cuda_moe = false;
    std::vector<std::string> bound_impl_ids;
    std::vector<std::string> bound_package_ids;
    bool all_bound() const {
        return quantize_q8_1 && mmvq && rms_norm_weighted && gdn_conv &&
                gdn_recurrence && gdn_out_norm && rope_apply && gdn_ab && router_projection;
    }
};

/// Value-owning snapshot of a registered kernel operation (safe across concurrent unloads).
struct KernelOperationSnapshot {
    guild_op_kind_t op_kind = GUILD_OP_INVALID;
    std::string op_name;
    std::string impl_id;
    std::string package_id;
    uint32_t impl_version = 0;
    guild_backend_t backend = GUILD_BACKEND_NONE;
    int32_t priority = 0;
    bool enabled = true;
    int64_t active_package_leases = 0;
};

/// Registry of built-in and plugin-provided hardware kernel operations.
class KernelRegistry {
public:
    KernelRegistry();
    ~KernelRegistry();

    /// Process-wide singleton initialized with all built-in CPU and GPU kernels.
    static KernelRegistry& instance();

    /// Populate this registry with built-in statically linked CPU/CUDA/HIP kernels.
    void register_builtin_kernels();

    /// Register a single kernel operation descriptor (ABI v1).
    guild_status_t register_op(const guild_kernel_op_desc_v1_t& desc, std::string& err);

    /// Register a plugin via its ABI v1 entry point.
    guild_status_t register_plugin(guild_plugin_entry_fn entry_fn, std::string& err);

    /// Register a plugin transactionally with an optional runner registration callback and metadata verification.
    guild_status_t register_plugin_transactional(
        guild_plugin_entry_fn entry_fn,
        const std::string& expected_package_id,
        const std::string& expected_version,
        const std::string& expected_publisher,
        const std::vector<std::string>& declared_ops,
        const std::vector<std::string>& declared_runners,
        std::function<guild_status_t(const guild_runner_desc_v1_t&, std::string&)> runner_commit_fn,
        std::function<void(const std::string&)> runner_rollback_fn,
        std::string& err,
        std::shared_ptr<void> library_keepalive = nullptr);

    /// Unregister all operations belonging to `package_id`.
    /// Fails with GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY if any session lease is active.
    guild_status_t unregister_package(const std::string& package_id, std::string& err);

    /// Enable or disable all operations belonging to `package_id` for future selections.
    void set_package_enabled(const std::string& package_id, bool enabled);
    bool is_package_enabled(const std::string& package_id) const;

    /// Validate that all registered operations have their declared `required_op_deps` present.
    guild_status_t validate_dependencies(std::string& err) const;

    /// Select the highest-priority compatible kernel for `query` under `caps`.
    guild_status_t select(const KernelSelectionQuery& query,
                          const guild_device_capabilities_v1_t& caps,
                          const guild_kernel_op_desc_v1_t** out_desc,
                          std::string& err) const;

    /// Look up a registered operation by exact `impl_id`.
    const guild_kernel_op_desc_v1_t* find_by_id(const std::string& impl_id) const;

    /// Snapshot of all registered descriptors.
    std::vector<guild_kernel_op_desc_v1_t> list_operations() const;

    /// Value-owning snapshots of all registered operations (safe without holding registry lock).
    std::vector<KernelOperationSnapshot> list_operation_snapshots() const;

    /// Resolve direct function pointers for layer-wise batched prefill once per model init.
    bool resolve_prefill_kernels(const guild_device_capabilities_v1_t& caps,
                                 ResolvedPrefillKernels& out_table,
                                 std::string& err) const;

    /// Resolve a copied CDD1 CUDA dense extension table once per model init.
    bool resolve_cuda_dense(const guild_device_capabilities_v1_t& caps,
                            ResolvedCudaDense& out_dense,
                            std::string& err) const;

    /// Resolve a copied CDA1 CUDA attention extension table once per model init.
    bool resolve_cuda_attn(const guild_device_capabilities_v1_t& caps,
                           ResolvedCudaAttn& out_attn,
                           std::string& err) const;

    /// Resolve a copied CDM1 table once per model, retained by its package lease.
    bool resolve_cuda_moe(const guild_device_capabilities_v1_t& caps,
                          ResolvedCudaMoE& out_moe,
                          std::string& err) const;

    /// Active session & per-package lease tracking (prevents kernel replacement/removal during active inference).
    void acquire_session_lease() const { active_sessions_.fetch_add(1, std::memory_order_acq_rel); }
    void release_session_lease() const { active_sessions_.fetch_sub(1, std::memory_order_acq_rel); }
    int64_t active_sessions() const { return active_sessions_.load(std::memory_order_acquire); }

    void acquire_package_lease(const std::string& package_id, void* device_stream = nullptr) const;
    void release_package_lease(const std::string& package_id, void* device_stream = nullptr) const;
    int64_t active_package_leases(const std::string& package_id) const;
    void synchronize_package_streams(const std::string& package_id) const;

private:
    struct StoredOp {
        guild_kernel_op_desc_v1_t desc{};
        std::string op_name;
        std::string impl_id;
        std::string package_id;
        std::vector<std::string> dep_storage;
        std::vector<const char*> dep_ptrs;
        std::unique_ptr<PrefillKernelTableV1> owned_direct_table;
        std::vector<uint8_t> owned_ext_bytes;
    };
    struct LoadedPlugin {
        guild_plugin_descriptor_v1_t desc{};
        std::string package_id;
        std::string package_version;
        std::string publisher_id;
        std::shared_ptr<void> library_keepalive;
    };
    struct PackageLeaseEntry {
        int64_t count = 0;
        std::set<void*> streams;
    };

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<StoredOp>> ops_;
    std::vector<LoadedPlugin> plugins_;
    std::set<std::string> disabled_packages_;
    mutable std::map<std::string, PackageLeaseEntry> package_leases_;
    mutable std::atomic<int64_t> active_sessions_{0};
};

/// RAII lease guard held by active GuildModel / GuildSession instances.
class SessionKernelLease {
public:
    SessionKernelLease() = default;
    explicit SessionKernelLease(const KernelRegistry* reg,
                                std::vector<std::string> package_ids = {},
                                void* device_stream = nullptr)
        : reg_(reg), package_ids_(std::move(package_ids)), stream_(device_stream) {
        if (reg_) {
            reg_->acquire_session_lease();
            for (const auto& pkg : package_ids_) {
                reg_->acquire_package_lease(pkg, stream_);
            }
        }
    }
    ~SessionKernelLease() { reset(); }
    SessionKernelLease(const SessionKernelLease&) = delete;
    SessionKernelLease& operator=(const SessionKernelLease&) = delete;
    SessionKernelLease(SessionKernelLease&& other) noexcept
        : reg_(other.reg_), package_ids_(std::move(other.package_ids_)), stream_(other.stream_) {
        other.reg_ = nullptr;
        other.package_ids_.clear();
        other.stream_ = nullptr;
    }
    SessionKernelLease& operator=(SessionKernelLease&& other) noexcept {
        if (this != &other) {
            reset();
            reg_ = other.reg_;
            package_ids_ = std::move(other.package_ids_);
            stream_ = other.stream_;
            other.reg_ = nullptr;
            other.package_ids_.clear();
            other.stream_ = nullptr;
        }
        return *this;
    }
    void add_package(const std::string& package_id) {
        if (!reg_ || package_id.empty()) return;
        for (const auto& existing : package_ids_) {
            if (existing == package_id) return;
        }
        reg_->acquire_package_lease(package_id, stream_);
        package_ids_.push_back(package_id);
    }
    void reset() {
        if (reg_) {
            for (const auto& pkg : package_ids_) {
                reg_->release_package_lease(pkg, stream_);
            }
            reg_->release_session_lease();
            reg_ = nullptr;
            package_ids_.clear();
            stream_ = nullptr;
        }
    }
private:
    const KernelRegistry* reg_ = nullptr;
    std::vector<std::string> package_ids_;
    void* stream_ = nullptr;
};

} // namespace guild::kernels
