// include/guild/kernels/registry.hpp - Hardware Kernel Operation Registry & Selection
#pragma once

#include "guild/abi/kernel_abi.h"
#include "guild/hardware/hardware_info.hpp"
#include "guild/kernels/rope_scaling.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
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

/// Direct function-pointer table resolved once during BatchedPrefill::init()
/// so layer-wise prefill incurs zero virtual-dispatch or registry overhead in inner loops.
struct ResolvedPrefillKernels {
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
    bool all_bound() const {
        return quantize_q8_1 && mmvq && rms_norm_weighted && gdn_conv &&
               gdn_recurrence && gdn_out_norm && rope_apply;
    }
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

    /// Unregister all operations belonging to `package_id`.
    /// Fails with GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY if any session lease is active.
    guild_status_t unregister_package(const std::string& package_id, std::string& err);

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

    /// Resolve direct function pointers for layer-wise batched prefill once per model init.
    bool resolve_prefill_kernels(const guild_device_capabilities_v1_t& caps,
                                 ResolvedPrefillKernels& out_table,
                                 std::string& err) const;

    /// Active session lease tracking (prevents kernel replacement/removal during active inference).
    void acquire_session_lease() const { active_sessions_.fetch_add(1, std::memory_order_acq_rel); }
    void release_session_lease() const { active_sessions_.fetch_sub(1, std::memory_order_acq_rel); }
    int64_t active_sessions() const { return active_sessions_.load(std::memory_order_acquire); }

private:
    struct StoredOp {
        guild_kernel_op_desc_v1_t desc{};
        std::string op_name;
        std::string impl_id;
        std::string package_id;
        std::vector<std::string> dep_storage;
        std::vector<const char*> dep_ptrs;
    };
    struct LoadedPlugin {
        guild_plugin_descriptor_v1_t desc{};
        std::string package_id;
        std::string package_version;
        std::string publisher_id;
    };

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<StoredOp>> ops_;
    std::vector<LoadedPlugin> plugins_;
    mutable std::atomic<int64_t> active_sessions_{0};
};

/// RAII lease guard held by active GuildModel / GuildSession instances.
class SessionKernelLease {
public:
    SessionKernelLease() = default;
    explicit SessionKernelLease(const KernelRegistry* reg) : reg_(reg) {
        if (reg_) reg_->acquire_session_lease();
    }
    ~SessionKernelLease() { reset(); }
    SessionKernelLease(const SessionKernelLease&) = delete;
    SessionKernelLease& operator=(const SessionKernelLease&) = delete;
    SessionKernelLease(SessionKernelLease&& other) noexcept : reg_(other.reg_) { other.reg_ = nullptr; }
    SessionKernelLease& operator=(SessionKernelLease&& other) noexcept {
        if (this != &other) {
            reset();
            reg_ = other.reg_;
            other.reg_ = nullptr;
        }
        return *this;
    }
    void reset() {
        if (reg_) {
            reg_->release_session_lease();
            reg_ = nullptr;
        }
    }
private:
    const KernelRegistry* reg_ = nullptr;
};

} // namespace guild::kernels
