// Copyright (c) Guild Inference Contributors. MIT License.
// Phase 5: GEXEC Ahead-of-Time Hardware-Specialized Optimization Compiler & Cache Manager.
#pragma once

#include "guild/gexec/gexec_contract.hpp"
#include "guild/artifact/gpack.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace guild::gexec {

enum class GExecOptMode : uint32_t {
    Skip       = 0,
    Quick      = 1,
    Auto       = 2,
    Exhaustive = 3,
};

const char* opt_mode_to_string(GExecOptMode mode) noexcept;
bool parse_opt_mode(const std::string& str, GExecOptMode& out) noexcept;

struct GExecOptBudget {
    double max_time_seconds = 300.0;       // Maximum wall-clock time budget (0 = unlimited)
    uint64_t max_memory_bytes = 0;         // Maximum temporary RAM budget (0 = auto from free RAM)
    uint64_t max_disk_bytes = 0;           // Maximum disk space budget (0 = auto from free disk)
    bool embed_packed_experts = false;     // Embed full PackedExpertsBlob in .gexec (vs zero-extra-pass on-load interleave)
    bool allow_partial_artifact = true;    // Save resumable partial .gexec if budget expires mid-run
    bool resume_interrupted = true;        // Resume from <output>.progress.json if valid
    GExecSectionKind only_section = GExecSectionKind::Invalid; // If != Invalid, regenerate only this section
};

struct GExecOptimizeOptions {
    std::string gpack_path;
    std::string output_gexec_path;         // Empty = auto path in cache_dir
    std::string cache_dir;                 // Empty = <gpack_dir>/.gexec_cache or $GUILD_GEXEC_CACHE_DIR
    GExecOptMode mode = GExecOptMode::Auto;
    GExecOptBudget budget;

    // Runtime execution plan parameters bound into GExecArtifactIdentity
    std::string kv_format = "fp16";
    std::string kv_mode = "host_only";
    int64_t max_context = 2048;
    int64_t prefill_batch = 64;
    int32_t mtp_spec_tokens = -1;          // -1 = auto from GPACK manifest (1 if has_mtp else 0)

    // Optional fault injection for testing interrupted runs / corruption recovery
    int32_t stop_after_pass = -1;          // If >= 0, simulates budget/interruption stop after pass N
};

struct GExecOptimizeResult {
    bool ok = false;
    bool skipped = false;
    bool resumed_from_checkpoint = false;
    bool partial_artifact = false;
    std::string output_path;
    std::string cache_key;
    std::string selected_cpu_expert_impl_id;
    uint32_t selected_cpu_expert_impl_version = 0;
    ExpertPackLayoutKind expert_layout_kind = ExpertPackLayoutKind::PortableContiguous;
    bool fused_single_barrier_pipeline = false;
    double optimization_wall_ms = 0.0;
    double baseline_decode_ms = 0.0;
    double optimized_decode_ms = 0.0;
    double speedup_ratio = 1.0;
    double break_even_tokens = 0.0;
    uint64_t artifact_bytes = 0;
    std::string error;
};

struct GExecCacheCleanResult {
    bool ok = false;
    uint32_t removed_files = 0;
    uint32_t kept_files = 0;
    uint64_t reclaimed_bytes = 0;
    std::vector<std::string> removed_paths;
    std::string error;
};

// Builds the canonical GExecArtifactIdentity for an opened GPACK container on the current host.
GExecArtifactIdentity build_current_gexec_identity(
    const artifact::GPackReader& gpack,
    const std::string& kv_format = "fp16",
    const std::string& kv_mode = "host_only",
    int64_t max_context = 2048,
    int64_t prefill_batch = 64,
    int32_t mtp_spec_tokens = -1);

// Computes the default .gexec artifact path for a given GPACK path and identity.
std::string default_gexec_path_for_gpack(
    const std::string& gpack_path,
    const GExecArtifactIdentity& identity,
    const std::string& cache_dir = "");

// Runs the GEXEC ahead-of-time optimization pipeline on a GPACK model.
GExecOptimizeResult optimize_gpack(
    const GExecOptimizeOptions& opts,
    std::function<void(const std::string& stage_msg)> progress_cb = nullptr);

// Regenerates a single section inside an existing .gexec file without re-running other passes.
bool regenerate_gexec_section(
    const std::string& gexec_path,
    const std::string& gpack_path,
    GExecSectionKind section_kind,
    std::string& err,
    const GExecOptBudget& budget = {});

// Resolves and opens a compatible .gexec artifact for a GPACK model if one exists.
// Never fails model loading: if absent, incompatible, or corrupted, returns false and populates `fallback_reason`.
bool resolve_gexec_for_gpack(
    const std::string& gpack_path,
    const std::string& explicit_gexec_path,
    const GExecArtifactIdentity& current_identity,
    GExecReader& out_reader,
    std::string& fallback_reason,
    GExecInvalidationReason* out_reason = nullptr);

// Cleans obsolete, corrupted, temporary, or all .gexec cache files in `cache_dir`.
GExecCacheCleanResult clean_gexec_cache(
    const std::string& cache_dir,
    bool remove_all = false,
    const GExecArtifactIdentity* keep_identity = nullptr);

} // namespace guild::gexec
