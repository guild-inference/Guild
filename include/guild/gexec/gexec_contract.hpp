#pragma once

#include "guild/abi/kernel_abi.h"
#include "guild/abi/module_abi.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace guild::gexec {

inline constexpr uint32_t kGExecMagic = 0x43584547u; // "GEXC"
inline constexpr uint16_t kGExecVersionV1 = 1u;

enum class GExecInvalidationReason : uint32_t {
    None = 0,
    MagicOrVersionMismatch = 1,
    CorruptedDigest = 2,
    GPackIdentityMismatch = 3,
    CoreAbiMismatch = 4,
    ModuleOrKernelDigestMismatch = 5,
    RunnerIdentityMismatch = 6,
    CpuIsaMismatch = 7,
    NumaTopologyMismatch = 8,
    GpuArchitectureMismatch = 9,
    CudaRuntimeOrDriverMismatch = 10,
    ExecutionPlanMismatch = 11,
};

const char* invalidation_reason_to_string(GExecInvalidationReason reason) noexcept;

struct GExecHardwareFingerprint {
    std::string cpu_vendor;
    std::string cpu_brand;
    uint32_t cpu_feature_flags = 0;
    int32_t cpu_physical_cores = 0;
    int32_t cpu_logical_threads = 0;
    int32_t numa_nodes = 1;
    bool thp_madvise_available = false;

    int32_t gpu_count = 0;
    std::string gpu_name;
    int32_t gpu_cc_major = 0;
    int32_t gpu_cc_minor = 0;
    uint64_t gpu_vram_bytes = 0;
    int32_t gpu_multiprocessors = 0;
    int32_t cuda_runtime_version = 0;
    int32_t cuda_driver_version = 0;

    static GExecHardwareFingerprint detect();
    std::string fingerprint_sha256() const;
    std::string to_json() const;
};

struct GExecBoundPackageDigest {
    std::string package_id;
    std::string version;
    std::string payload_sha256;
};

struct GExecArtifactIdentity {
    uint16_t format_version = kGExecVersionV1;
    std::string gpack_manifest_sha256;
    uint32_t core_abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    uint32_t module_abi_version = 1;
    std::string runner_id;
    uint32_t runner_version = 0;
    std::vector<GExecBoundPackageDigest> bound_packages;
    GExecHardwareFingerprint hardware;

    std::string kv_format = "fp16";
    std::string kv_mode = "host_only";
    int64_t max_context = 0;
    int64_t prefill_batch = 0;
    int32_t mtp_spec_tokens = 0;

    std::string compute_cache_key() const;
    std::string to_json() const;
    static bool from_json(const std::string& json_str, GExecArtifactIdentity& out, std::string& err);

    static GExecInvalidationReason validate(
        const GExecArtifactIdentity& cached,
        const GExecArtifactIdentity& current,
        std::string& reason_msg);
};

// Family A: Offline Expert Weight Packing Contract & Prototype
enum class ExpertPackLayoutKind : uint32_t {
    PortableContiguous     = 0, // [gate(n_ff) | up(n_ff) | down(n_embd)] (GPACK canonical)
    GateUpRowInterleaved   = 1, // [gate_row_0, up_row_0, ..., gate_row_{n_ff-1}, up_row_{n_ff-1} | down(n_embd)]
    SimdTilePacked         = 2, // CPU SIMD-tiled block arrangement (GEXEC Phase 5)
    GpuMmvqPacked          = 3, // GPU warp-tiled MMVQ arrangement (GEXEC Phase 5)
};

struct GExecExpertPackSpec {
    ExpertPackLayoutKind layout_kind = ExpertPackLayoutKind::PortableContiguous;
    int32_t gu_type = -1;
    int32_t d_type = -1;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    uint64_t alignment = 64;
};

void pack_expert_gate_up_interleaved(
    const uint8_t* src_blob,
    uint8_t* dst_blob,
    size_t n_ff,
    size_t gate_row_bytes,
    size_t up_row_bytes,
    size_t down_bytes);

void unpack_expert_gate_up_interleaved(
    const uint8_t* packed_blob,
    uint8_t* dst_blob,
    size_t n_ff,
    size_t gate_row_bytes,
    size_t up_row_bytes,
    size_t down_bytes);

// Family B: Kernel Autotuning & Exact Implementation Locking
enum class GExecWorkloadPhase : uint32_t {
    Prefill = 0,
    Decode  = 1,
    Verify  = 2,
};

struct GExecKernelTuneRecord {
    uint32_t op_kind = 0;
    std::string operation_name;
    GExecWorkloadPhase phase = GExecWorkloadPhase::Decode;
    uint32_t backend = GUILD_BACKEND_CPU;
    int32_t quant_type = -1;
    int64_t m_tokens = 1;
    int64_t n_cols = 0;
    int64_t k_cols = 0;
    std::string selected_impl_id;
    std::string selected_package_id;
    std::string fallback_impl_id;
    double median_latency_us = 0.0;
    uint32_t candidate_count = 0;
};

struct GExecAutotuneTable {
    GExecArtifactIdentity identity;
    std::vector<GExecKernelTuneRecord> records;

    const GExecKernelTuneRecord* find(
        uint32_t op_kind,
        const std::string& operation_name,
        GExecWorkloadPhase phase,
        int32_t quant_type,
        int64_t m_tokens) const;

    std::string to_json() const;
    static bool from_json(const std::string& json_str, GExecAutotuneTable& out, std::string& err);
};

// Family C: NUMA & Heterogeneous Execution Plan
struct GExecHeterogeneousPlan {
    GExecArtifactIdentity identity;
    std::string numa_policy = "interleave"; // "interleave", "process", "zero_copy_mmap"
    bool use_thp_hugepages = true;
    int32_t measured_formats = 0;
    double calibrated_cpu_ms = 0.0;
    double calibrated_gpu_ms = 0.0;
    double calibrated_copy_ms = 0.0;
    std::vector<uint8_t> gpu_layers;
    std::vector<int32_t> pcie_numerators;
    int32_t expert_cache_slots = 0;

    std::string to_json() const;
    static bool from_json(const std::string& json_str, GExecHeterogeneousPlan& out, std::string& err);
};

// Family D: Persistent Execution Workspace & Graph Capture Recipe
struct GExecWorkspaceRecipe {
    uint64_t dense_arena_bytes = 0;
    uint64_t native_dense_arena_bytes = 0;
    uint64_t session_device_bytes = 0;
    uint64_t session_host_kv_bytes = 0;
    uint64_t verifier_arena_bytes = 0;
    uint64_t prefill_arena_bytes = 0;
    std::vector<int32_t> capture_window_tokens;
    bool capture_commit_graph = true;
};

// Family E: Compilation Cache Provenance
struct GExecCompilationKey {
    int32_t cuda_runtime_version = 0;
    int32_t cuda_driver_version = 0;
    int32_t gpu_cc_major = 0;
    int32_t gpu_cc_minor = 0;
    std::string kernel_package_sha256;
    std::string compute_key() const;
};

} // namespace guild::gexec
