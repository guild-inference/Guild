#pragma once

#include "guild/abi/kernel_abi.h"
#include "guild/abi/module_abi.h"

#include <cstddef>
#include <cstdint>
#include <functional>
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

    static GExecInvalidationReason validate(const GExecArtifactIdentity& cached, const GExecArtifactIdentity& current,
                                            std::string& reason_msg);
};

// Family A: Offline Expert Weight Packing Contract & Prototype
enum class ExpertPackLayoutKind : uint32_t {
    PortableContiguous = 0,   // [gate(n_ff) | up(n_ff) | down(n_embd)] (GPACK canonical)
    GateUpRowInterleaved = 1, // [gate_row_0, up_row_0, ..., gate_row_{n_ff-1}, up_row_{n_ff-1} | down(n_embd)]
    SimdTilePacked = 2,       // CPU SIMD-tiled block arrangement (GEXEC Phase 5)
    GpuMmvqPacked = 3,        // GPU warp-tiled MMVQ arrangement (GEXEC Phase 5)
};

struct GExecExpertPackSpec {
    ExpertPackLayoutKind layout_kind = ExpertPackLayoutKind::PortableContiguous;
    int32_t gu_type = -1;
    int32_t d_type = -1;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    uint64_t alignment = 64;
};

void pack_expert_gate_up_interleaved(const uint8_t* src_blob, uint8_t* dst_blob, size_t n_ff, size_t gate_row_bytes,
                                     size_t up_row_bytes, size_t down_bytes);

void unpack_expert_gate_up_interleaved(const uint8_t* packed_blob, uint8_t* dst_blob, size_t n_ff,
                                       size_t gate_row_bytes, size_t up_row_bytes, size_t down_bytes);

// Family B: Kernel Autotuning & Exact Implementation Locking
enum class GExecWorkloadPhase : uint32_t {
    Prefill = 0,
    Decode = 1,
    Verify = 2,
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
    uint32_t selected_impl_version = 0;
    std::string fallback_impl_id;
    double median_latency_us = 0.0;
    uint32_t candidate_count = 0;
};

struct GExecAutotuneTable {
    GExecArtifactIdentity identity;
    std::vector<GExecKernelTuneRecord> records;

    const GExecKernelTuneRecord* find(uint32_t op_kind, const std::string& operation_name, GExecWorkloadPhase phase,
                                      int32_t quant_type, int64_t m_tokens) const;

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

    // Phase 6.6 / 6.6.1: Speculative verification & drafting cost calibration (window sizes T = 1..8).
    bool spec_calibrated = false;
    std::vector<double> verify_window_ms;
    std::vector<double> verify_cpu_expert_ms;
    std::vector<double> distinct_expert_ratio;
    double mtp_draft_step_ms = 0.0;
    double mtp_kv_catchup_ms = 0.0;
    double lookup_draft_ms = 0.0;
    double retrieval_draft_ms = 0.0;
    double kv_commit_ms = 0.0;
    double verify_noise_cv = 0.0;
    double concurrency_slope = 0.0;
    double context_attn_ms_per_1k = 0.0;

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

    std::string to_json() const;
    static bool from_json(const std::string& json_str, GExecWorkspaceRecipe& out, std::string& err);
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

struct GExecCompilationProvenance {
    GExecCompilationKey key;
    std::string compiler_id = "guild-gexec-v1";
    std::string guild_version;
    std::string git_commit;
    int64_t created_unix_seconds = 0;
    std::string optimization_mode = "auto";
    double optimization_wall_ms = 0.0;

    std::string to_json() const;
    static bool from_json(const std::string& json_str, GExecCompilationProvenance& out, std::string& err);
};

// Expert Packing Metadata Section
struct GExecExpertPackMeta {
    ExpertPackLayoutKind layout_kind = ExpertPackLayoutKind::PortableContiguous;
    bool embedded_blob_present = false;
    bool fused_single_barrier_pipeline = true;
    int64_t n_layers = 0;
    int64_t n_expert = 0;
    uint64_t total_expert_bytes = 0;
    std::vector<GExecExpertPackSpec> layer_specs;

    std::string to_json() const;
    static bool from_json(const std::string& json_str, GExecExpertPackMeta& out, std::string& err);
};

// Candidate Rejection & Diagnostic Log Section
struct GExecCandidateRejection {
    uint32_t op_kind = 0;
    std::string operation_name;
    std::string candidate_impl_id;
    std::string stage; // "compatibility", "correctness", "microbenchmark", "end_to_end"
    std::string reason;
    double measured_max_abs_err = 0.0;
    double measured_latency_us = 0.0;
};

struct GExecDiagnosticLog {
    std::vector<GExecCandidateRejection> rejected_candidates;
    std::vector<std::string> notes;
    double baseline_decode_ms = 0.0;
    double optimized_decode_ms = 0.0;
    double speedup_ratio = 1.0;
    double break_even_tokens = 0.0;

    std::string to_json() const;
    static bool from_json(const std::string& json_str, GExecDiagnosticLog& out, std::string& err);
};

// ============================================================================
// Phase 5 Durable Binary GEXEC v1 Container Format ("GEXC")
// ============================================================================

inline constexpr uint16_t kGExecLittleEndianTag = 0x0102u;

enum GExecHeaderFlags : uint32_t {
    kGExecFlagNone = 0u,
    kGExecFlagPartialArtifact = 1u << 0,
    kGExecFlagHasEmbeddedExperts = 1u << 1,
    kGExecFlagFusedPipeline = 1u << 2,
};

enum class GExecSectionKind : uint32_t {
    Invalid = 0,
    IdentityAndManifest = 1,
    AutotuneTable = 2,
    HeterogeneousPlan = 3,
    WorkspaceRecipe = 4,
    CompilationProvenance = 5,
    ExpertPackMeta = 6,
    PackedExpertsBlob = 7,
    DiagnosticLog = 8,
};

const char* section_kind_to_string(GExecSectionKind kind) noexcept;

enum GExecSectionFlags : uint32_t {
    kGExecSectionFlagNone = 0u,
    kGExecSectionFlagRequired = 1u << 0,
    kGExecSectionFlagPageAligned = 1u << 1,
};

#pragma pack(push, 1)
struct GExecFileHeader {
    uint32_t magic = kGExecMagic;                    // "GEXC" (0x43584547)
    uint16_t format_version = kGExecVersionV1;       // 1
    uint16_t endianness_tag = kGExecLittleEndianTag; // 0x0102
    uint32_t flags = 0;                              // GExecHeaderFlags bitmask
    uint32_t section_count = 0;                      // Number of GExecSectionEntry records
    uint64_t section_table_offset = 64;              // Byte offset of section directory (64)
    uint64_t total_file_bytes = 0;                   // Exact total file size in bytes
    uint8_t header_and_table_sha256[32] = {};        // SHA-256 over header[0..32) + section table
};

struct GExecSectionEntry {
    uint32_t kind = 0;       // GExecSectionKind
    uint32_t flags = 0;      // GExecSectionFlags
    uint64_t offset = 0;     // Byte offset in .gexec container
    uint64_t byte_size = 0;  // Payload byte size
    uint64_t alignment = 64; // Required byte alignment (power of 2)
    uint8_t sha256[32] = {}; // Raw 32-byte SHA-256 of section payload
};
#pragma pack(pop)

static_assert(sizeof(GExecFileHeader) == 64, "GExecFileHeader must be 64 bytes");
static_assert(sizeof(GExecSectionEntry) == 64, "GExecSectionEntry must be 64 bytes");

class GExecReader {
public:
    GExecReader() = default;
    ~GExecReader();
    GExecReader(const GExecReader&) = delete;
    GExecReader& operator=(const GExecReader&) = delete;
    GExecReader(GExecReader&& other) noexcept;
    GExecReader& operator=(GExecReader&& other) noexcept;

    // Opens and memory-maps a .gexec container, validating header magic, version,
    // endianness, file size, section bounds/alignments, and header+table SHA-256.
    // If verify_all_sections is true, also verifies SHA-256 of every metadata section
    // (and PackedExpertsBlob if verify_expert_blob_digest is true).
    bool open(const std::string& path, std::string& err, GExecInvalidationReason* out_reason = nullptr,
              bool verify_expert_blob_digest = false);

    // Validates that this opened .gexec container matches the current runtime identity.
    bool validate_against(const GExecArtifactIdentity& current_identity, std::string& reason_msg,
                          GExecInvalidationReason* out_reason = nullptr) const;

    // Verifies the cryptographic SHA-256 digest of a specific section (or all sections).
    bool verify_section_digest(GExecSectionKind kind, std::string& err) const;
    bool verify_all_digests(std::string& err) const;

    void close();

    bool is_open() const noexcept {
        return mapped_base_ != nullptr;
    }
    const std::string& path() const noexcept {
        return path_;
    }
    const GExecFileHeader& header() const noexcept {
        return header_;
    }
    const std::vector<GExecSectionEntry>& sections() const noexcept {
        return sections_;
    }

    bool is_partial() const noexcept {
        return (header_.flags & kGExecFlagPartialArtifact) != 0;
    }
    bool has_section(GExecSectionKind kind) const noexcept;
    const GExecSectionEntry* find_section(GExecSectionKind kind) const noexcept;
    const uint8_t* section_data(GExecSectionKind kind, uint64_t* out_size = nullptr) const noexcept;

    const GExecArtifactIdentity& identity() const noexcept {
        return identity_;
    }
    const GExecAutotuneTable& autotune_table() const noexcept {
        return autotune_;
    }
    const GExecHeterogeneousPlan& heterogeneous_plan() const noexcept {
        return hplan_;
    }
    const GExecWorkspaceRecipe& workspace_recipe() const noexcept {
        return workspace_;
    }
    const GExecCompilationProvenance& provenance() const noexcept {
        return provenance_;
    }
    const GExecExpertPackMeta& expert_pack_meta() const noexcept {
        return pack_meta_;
    }
    const GExecDiagnosticLog& diagnostic_log() const noexcept {
        return diag_log_;
    }

    const uint8_t* packed_experts_blob(uint64_t* out_bytes = nullptr) const noexcept;

    std::string inspect_json() const;

private:
    std::string path_;
    const uint8_t* mapped_base_ = nullptr;
    uint64_t mapped_bytes_ = 0;
    int fd_ = -1;

    GExecFileHeader header_{};
    std::vector<GExecSectionEntry> sections_;

    GExecArtifactIdentity identity_;
    GExecAutotuneTable autotune_;
    GExecHeterogeneousPlan hplan_;
    GExecWorkspaceRecipe workspace_;
    GExecCompilationProvenance provenance_;
    GExecExpertPackMeta pack_meta_;
    GExecDiagnosticLog diag_log_;
};

class GExecBuilder {
public:
    GExecBuilder() = default;

    void set_flags(uint32_t flags) {
        flags_ = flags;
    }
    uint32_t flags() const noexcept {
        return flags_;
    }

    void set_identity(const GExecArtifactIdentity& id);
    void set_autotune_table(const GExecAutotuneTable& table);
    void set_heterogeneous_plan(const GExecHeterogeneousPlan& plan);
    void set_workspace_recipe(const GExecWorkspaceRecipe& recipe);
    void set_provenance(const GExecCompilationProvenance& prov);
    void set_expert_pack_meta(const GExecExpertPackMeta& meta);
    void set_diagnostic_log(const GExecDiagnosticLog& log);

    // Sets or streams an embedded PackedExpertsBlob section (page-aligned to 4096 B).
    void set_packed_experts_blob(std::vector<uint8_t> blob, uint64_t alignment = 4096);
    using BlobReader = std::function<bool(uint64_t offset, uint8_t* dst, size_t size, std::string& err)>;
    // The callback and its source must remain valid through write_file; reads may repeat.
    void set_packed_experts_stream(uint64_t bytes, BlobReader read);

    // Imports all existing sections from an opened GExecReader so a single section
    // can be selectively regenerated without re-running unrelated passes.
    // With stream_experts, the reader must remain open through write_file.
    bool import_from(const GExecReader& reader, std::string& err, bool stream_experts = false);

    // Atomically writes the .gexec container to `dest_path` via a crash-safe temporary
    // file + fsync + rename in the same filesystem directory.
    bool write_file(const std::string& dest_path, std::string& err, uint64_t max_disk_bytes = 0) const;

private:
    struct PendingSection {
        GExecSectionKind kind = GExecSectionKind::Invalid;
        uint32_t flags = kGExecSectionFlagNone;
        uint64_t alignment = 64;
        std::vector<uint8_t> data;
        uint64_t stream_bytes = 0;
        BlobReader read;
    };
    void upsert_section(GExecSectionKind kind, uint32_t flags, uint64_t alignment, std::vector<uint8_t> data);

    uint32_t flags_ = 0;
    std::vector<PendingSection> sections_;
};

} // namespace guild::gexec
