// Copyright (c) Guild Inference Contributors. MIT License.
// Phase 5: GEXEC Ahead-of-Time Hardware-Specialized Optimization Compiler & Cache Manager.

#include "guild/gexec/gexec_optimizer.hpp"
#include "guild/abi/gkernel_package.hpp"
#include "guild/build_metadata.hpp"
#include "guild/hardware/hardware_info.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include "guild/kernels/cpu/gexec_expert.hpp"
#include "guild/kernels/cpu/pool.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/model/runner.hpp"
#include "guild/models/sha256.hpp"
#include "guild/server/json.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace guild::gexec {

using namespace guild::server::json;

const char* opt_mode_to_string(GExecOptMode mode) noexcept {
    switch (mode) {
    case GExecOptMode::Skip: return "skip";
    case GExecOptMode::Quick: return "quick";
    case GExecOptMode::Auto: return "auto";
    case GExecOptMode::Exhaustive: return "exhaustive";
    }
    return "auto";
}

bool parse_opt_mode(const std::string& str, GExecOptMode& out) noexcept {
    if (str == "skip" || str == "none" || str == "0") {
        out = GExecOptMode::Skip;
        return true;
    }
    if (str == "quick" || str == "fast" || str == "1") {
        out = GExecOptMode::Quick;
        return true;
    }
    if (str == "auto" || str == "default" || str == "2") {
        out = GExecOptMode::Auto;
        return true;
    }
    if (str == "exhaustive" || str == "full" || str == "3") {
        out = GExecOptMode::Exhaustive;
        return true;
    }
    return false;
}

namespace {

class ScopedAdvisoryLock {
public:
    ScopedAdvisoryLock() = default;
    ~ScopedAdvisoryLock() {
        release();
    }
    ScopedAdvisoryLock(const ScopedAdvisoryLock&) = delete;
    ScopedAdvisoryLock& operator=(const ScopedAdvisoryLock&) = delete;

    bool acquire(const std::string& lock_path, std::string& err) {
        release();
        std::error_code ec;
        const std::filesystem::path p(lock_path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path(), ec);
        }
#if !defined(_WIN32)
        const int fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd < 0) {
            err = "cannot open GEXEC lock file: " + lock_path;
            return false;
        }
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(fd);
            err = "another GEXEC optimization process holds the lock on: " + lock_path;
            return false;
        }
        const std::string pid_str = std::to_string(::getpid()) + "\n";
        if (::ftruncate(fd, 0) == 0) {
            const ssize_t nw = ::write(fd, pid_str.data(), pid_str.size());
            (void) nw;
        }
        fd_ = fd;
        path_ = lock_path;
#else
        const auto wide = std::filesystem::u8path(lock_path).wstring();
        HANDLE file = CreateFileW(wide.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                  nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        OVERLAPPED overlap{};
        if (file == INVALID_HANDLE_VALUE ||
            !LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlap)) {
            if (file != INVALID_HANDLE_VALUE)
                CloseHandle(file);
            err = "cannot acquire GEXEC writer lock: " + lock_path;
            return false;
        }
        file_ = file;
#endif
        return true;
    }

    void release() {
#if !defined(_WIN32)
        if (fd_ >= 0) {
            (void) ::flock(fd_, LOCK_UN);
            ::close(fd_);
            fd_ = -1;
        }
        // Keep the lock inode: unlinking it can let another writer lock a new inode
        // while a contender still holds the old one.
        path_.clear();
#else
        if (file_ != INVALID_HANDLE_VALUE) {
            OVERLAPPED overlap{};
            UnlockFileEx(file_, 0, 1, 0, &overlap);
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
        }
#endif
    }

private:
    int fd_ = -1;
    std::string path_;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
#endif
};

std::string resolve_cache_dir(const std::string& gpack_path, const std::string& explicit_dir) {
    if (!explicit_dir.empty())
        return explicit_dir;
    if (const char* env_dir = std::getenv("GUILD_GEXEC_CACHE_DIR"); env_dir && env_dir[0] != '\0') {
        return std::string(env_dir);
    }
    const std::filesystem::path gp(gpack_path);
    const std::filesystem::path parent = gp.has_parent_path() ? gp.parent_path() : std::filesystem::path(".");
    return (parent / ".gexec_cache").string();
}

struct CheckpointState {
    std::string cache_key;
    int32_t last_completed_pass = 0;
    GExecArtifactIdentity identity;
    GExecExpertPackMeta pack_meta;
    GExecAutotuneTable autotune;
    GExecDiagnosticLog diag_log;
    GExecHeterogeneousPlan hplan;
    GExecWorkspaceRecipe workspace;
    GExecCompilationProvenance provenance;

    std::string to_json() const {
        JsonValue root;
        root["cache_key"] = cache_key;
        root["last_completed_pass"] = static_cast<int64_t>(last_completed_pass);
        std::string perr;
        auto put = [&](const char* k, const std::string& j) {
            JsonValue v;
            if (JsonValue::parse(j, v, perr))
                root[k] = std::move(v);
        };
        if (last_completed_pass >= 1)
            put("identity", identity.to_json());
        if (last_completed_pass >= 2)
            put("pack_meta", pack_meta.to_json());
        if (last_completed_pass >= 3) {
            put("autotune", autotune.to_json());
            put("diag_log", diag_log.to_json());
        }
        if (last_completed_pass >= 4)
            put("hplan", hplan.to_json());
        if (last_completed_pass >= 5) {
            put("workspace", workspace.to_json());
            put("provenance", provenance.to_json());
        }
        root["digest"] = models::Sha256::hash_string(root.dump());
        return root.dump_pretty();
    }

    static bool from_json(const std::string& jstr, CheckpointState& out) {
        std::string err;
        JsonValue root;
        if (!JsonValue::parse(jstr, root, err) || !root.is_object() || !root.contains("digest"))
            return false;
        const std::string rec_dig = root["digest"].as_string();
        JsonValue chk = root;
        chk.obj_val.erase(
            std::remove_if(chk.obj_val.begin(), chk.obj_val.end(), [](const auto& kv) { return kv.first == "digest"; }),
            chk.obj_val.end());
        if (rec_dig != models::Sha256::hash_string(chk.dump()))
            return false;
        out.cache_key = root["cache_key"].as_string();
        out.last_completed_pass = static_cast<int32_t>(root["last_completed_pass"].as_int(0));
        if (out.last_completed_pass >= 1 &&
            !GExecArtifactIdentity::from_json(root["identity"].dump(), out.identity, err))
            return false;
        if (out.last_completed_pass >= 2 &&
            !GExecExpertPackMeta::from_json(root["pack_meta"].dump(), out.pack_meta, err))
            return false;
        if (out.last_completed_pass >= 3) {
            if (!GExecAutotuneTable::from_json(root["autotune"].dump(), out.autotune, err))
                return false;
            if (!GExecDiagnosticLog::from_json(root["diag_log"].dump(), out.diag_log, err))
                return false;
        }
        if (out.last_completed_pass >= 4 && !GExecHeterogeneousPlan::from_json(root["hplan"].dump(), out.hplan, err))
            return false;
        if (out.last_completed_pass >= 5) {
            if (!GExecWorkspaceRecipe::from_json(root["workspace"].dump(), out.workspace, err))
                return false;
            if (!GExecCompilationProvenance::from_json(root["provenance"].dump(), out.provenance, err))
                return false;
        }
        return true;
    }
};

bool save_checkpoint_atomic(const std::string& progress_path, const CheckpointState& st) {
    const std::string tmp = progress_path + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs)
            return false;
        const std::string s = st.to_json();
        ofs.write(s.data(), static_cast<std::streamsize>(s.size()));
        if (!ofs.good())
            return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, progress_path, ec);
    return !ec;
}

} // namespace

GExecArtifactIdentity build_current_gexec_identity(const artifact::GPackReader& gpack, const std::string& kv_format,
                                                   const std::string& kv_mode, int64_t max_context,
                                                   int64_t prefill_batch, int32_t mtp_spec_tokens) {
    GExecArtifactIdentity id;
    id.format_version = kGExecVersionV1;
    id.gpack_manifest_sha256 = gpack.manifest_sha256_hex();
    id.core_abi_version = GUILD_KERNEL_ABI_VERSION_V1;
    id.module_abi_version = 1;

    std::shared_ptr<const model::IModelRunner> runner;
    std::string rerr;
    const std::string arch_str =
        !gpack.execution_metadata().arch_id.empty() ? gpack.execution_metadata().arch_id : gpack.descriptor().arch_name;
    if (model::ModelRunnerRegistry::instance().resolve_by_arch(arch_str, &runner, rerr) == GUILD_STATUS_OK && runner) {
        id.runner_id = runner->runner_id();
        id.runner_version = runner->runner_version();
    } else {
        id.runner_id = gpack.execution_metadata().runner_id;
        id.runner_version = 1;
    }

    // Collect bound package digests deterministically from KernelRegistry
    const auto snaps = kernels::KernelRegistry::instance().list_operation_snapshots();
    JsonValue build_info;
    JsonValue::parse(build::metadata_json, build_info, rerr);
    std::set<std::string> seen_pkgs;
    for (const auto& s : snaps) {
        if (!s.enabled || s.package_id.empty())
            continue;
        if (seen_pkgs.insert(s.package_id).second) {
            GExecBoundPackageDigest d;
            d.package_id = s.package_id;
            d.version = std::to_string(GUILD_KERNEL_ABI_VERSION_MAJOR_OF(s.impl_version)) + "." +
                        std::to_string(GUILD_KERNEL_ABI_VERSION_MINOR_OF(s.impl_version)) + "." +
                        std::to_string(GUILD_KERNEL_ABI_VERSION_PATCH_OF(s.impl_version));
            abi::InstalledPackageRecord package;
            if (abi::GKernelPackageManager::instance().get_package(s.package_id, package) &&
                package.loaded_in_process) {
                d.version = package.metadata.version;
                d.payload_sha256 = package.metadata.compute_payload_sha256();
            } else {
                d.payload_sha256 =
                    models::Sha256::hash_string(s.package_id + ":" + build_info["gexec_source_sha256"].as_string());
            }
            id.bound_packages.push_back(std::move(d));
        }
    }

    // Runner-only modules have no kernel operation snapshot.
    for (const auto& package : abi::GKernelPackageManager::instance().list_packages()) {
        if (package.loaded_in_process && package.enabled && seen_pkgs.insert(package.package_id).second)
            id.bound_packages.push_back(
                {package.package_id, package.metadata.version, package.metadata.compute_payload_sha256()});
    }

    id.hardware = GExecHardwareFingerprint::detect();
    id.kv_format = kv_format;
    std::transform(id.kv_format.begin(), id.kv_format.end(), id.kv_format.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    id.kv_mode = kv_mode;
    std::replace(id.kv_mode.begin(), id.kv_mode.end(), '-', '_');
    id.max_context = max_context > 0 ? max_context : 2048;
    id.prefill_batch = prefill_batch > 0 ? prefill_batch : 64;
    if (mtp_spec_tokens >= 0) {
        id.mtp_spec_tokens = mtp_spec_tokens;
    } else {
        id.mtp_spec_tokens = gpack.has_mtp() ? 1 : 0;
    }
    return id;
}

std::string default_gexec_path_for_gpack(const std::string& gpack_path, const GExecArtifactIdentity& identity,
                                         const std::string& cache_dir) {
    const std::string dir = resolve_cache_dir(gpack_path, cache_dir);
    const std::filesystem::path gp(gpack_path);
    const std::string stem = gp.stem().string().empty() ? "model" : gp.stem().string();
    const std::string key = identity.compute_cache_key();
    const std::string short_key = key.substr(0, 16);
    return (std::filesystem::path(dir) / (stem + "." + short_key + ".gexec")).string();
}

namespace {

bool read_expert(const artifact::GPackReader& gpack, size_t layer, int64_t expert, std::vector<uint8_t>& blob,
                 std::string& err) {
    const auto& lr = gpack.expert_layers().at(layer);
    if (expert < 0 || expert >= gpack.expert_count_per_layer() || lr.up_off > lr.down_off ||
        lr.down_off > lr.blob_bytes || lr.gate_off != 0) {
        err = "invalid GPACK expert geometry";
        return false;
    }
    blob.resize(static_cast<size_t>(lr.blob_bytes));
    if (gpack.has_embedded_experts()) {
        uint64_t size = 0;
        const auto* data = gpack.expert_blobs_data(&size);
        if (!data || lr.layer_offset > size || lr.blob_bytes == 0 ||
            static_cast<uint64_t>(expert) >= (size - lr.layer_offset) / lr.blob_bytes) {
            err = "GPACK expert exceeds embedded section bounds";
            return false;
        }
        std::memcpy(blob.data(), data + lr.layer_offset + expert * lr.blob_bytes, blob.size());
        return true;
    }
    const auto parent = std::filesystem::path(gpack.external_expert_primary_gguf()).parent_path();
    auto read_role = [&](const std::string& shard, uint64_t base, uint64_t offset, uint64_t bytes) {
        const auto path = parent / shard;
        std::ifstream in(path, std::ios::binary);
        const uint64_t start = base + static_cast<uint64_t>(expert) * bytes;
        if (start < base || start > static_cast<uint64_t>(INT64_MAX) || bytes > static_cast<uint64_t>(INT64_MAX))
            return false;
        in.seekg(static_cast<std::streamoff>(start));
        in.read(reinterpret_cast<char*>(blob.data() + offset), static_cast<std::streamsize>(bytes));
        return in.good();
    };
    if (!gpack.has_external_experts() || !read_role(lr.gate_shard, lr.gguf_gate_off, 0, lr.up_off) ||
        !read_role(lr.up_shard, lr.gguf_up_off, lr.up_off, lr.down_off - lr.up_off) ||
        !read_role(lr.down_shard, lr.gguf_down_off, lr.down_off, lr.blob_bytes - lr.down_off)) {
        err = "cannot read GPACK external expert roles";
        return false;
    }
    return true;
}

// Pass 3: Staged Kernel Autotuner + Multi-Expert Pool Validation
bool run_autotune_pass(const artifact::GPackReader& gpack, const GExecArtifactIdentity& identity, GExecOptMode mode,
                       GExecExpertPackMeta& inout_pack_meta, GExecAutotuneTable& out_table, GExecDiagnosticLog& out_log,
                       std::string& out_selected_cpu_impl, uint32_t& out_selected_cpu_ver, std::string& err,
                       const GExecOptBudget& budget = {}, const std::function<bool()>& should_stop = {}) {
    out_table = GExecAutotuneTable{};
    out_table.identity = identity;
    out_log = GExecDiagnosticLog{};
    auto interrupted = [&] {
        if (!should_stop || !should_stop())
            return false;
        err = "GEXEC autotuning time budget expired";
        return true;
    };

    const auto caps = kernels::query_host_device_capabilities();
    const auto all_ops = kernels::KernelRegistry::instance().list_operations();

    // Determine baseline reference CPU expert backend table
    const guild_cpu_backend_ext_v2_t* ref_table = nullptr;
    for (const auto& op : all_ops) {
        if (op->op_kind == GUILD_OP_BATCHED_EXPERT_EXEC && op->impl_id &&
            std::strcmp(op->impl_id, "cpu.expert_pool.split_multi_native") == 0 && op->next_ext) {
            ref_table = static_cast<const guild_cpu_backend_ext_v2_t*>(op->next_ext);
            break;
        }
    }
    inout_pack_meta.layout_kind = ExpertPackLayoutKind::PortableContiguous;
    inout_pack_meta.fused_single_barrier_pipeline = false;
    for (auto& s : inout_pack_meta.layer_specs)
        s.layout_kind = inout_pack_meta.layout_kind;
    if (!ref_table) {
        out_log.notes.push_back("No reference CPU backend is available; expert tuning skipped.");
        return true;
    }

    // One runtime pool binding must be correct for every format in the model.
    struct LayerFmtKey {
        int32_t gu_type = 12;
        int32_t d_type = 20;
        int64_t n_embd = 2048;
        int64_t n_ff = 512;
        bool operator<(const LayerFmtKey& o) const noexcept {
            if (gu_type != o.gu_type)
                return gu_type < o.gu_type;
            if (d_type != o.d_type)
                return d_type < o.d_type;
            if (n_embd != o.n_embd)
                return n_embd < o.n_embd;
            return n_ff < o.n_ff;
        }
    };
    const int64_t model_n_embd = gpack.descriptor().attn.n_embd;
    const int64_t model_n_ff = gpack.descriptor().moe.expert_dim_ff;
    std::vector<LayerFmtKey> distinct_formats;
    std::vector<size_t> format_layers;
    std::set<LayerFmtKey> seen_formats;
    std::map<LayerFmtKey, size_t> format_counts;
    if (gpack.has_expert_layout() && gpack.expert_native_format() && model_n_embd > 0 &&
        model_n_embd <= kernels::cpu::H && model_n_ff > 0 && model_n_ff <= kernels::cpu::FF) {
        for (size_t l = 0; l < gpack.expert_layers().size(); ++l) {
            const auto& lrow = gpack.expert_layers()[l];
            guild_cpu_expert_format_v1_t probe_fmt{};
            if (ref_table->format(lrow.gu_type, lrow.d_type, model_n_embd, model_n_ff, &probe_fmt, nullptr) ==
                    GUILD_STATUS_OK &&
                probe_fmt.bytes == lrow.blob_bytes && probe_fmt.up_off == lrow.up_off &&
                probe_fmt.down_off == lrow.down_off) {
                LayerFmtKey k{lrow.gu_type, lrow.d_type, model_n_embd, model_n_ff};
                ++format_counts[k];
                if (seen_formats.insert(k).second) {
                    distinct_formats.push_back(k);
                    format_layers.push_back(l);
                }
            } else {
                out_log.notes.push_back("Expert tuning skipped: unsupported native geometry at layer " +
                                        std::to_string(l));
                return true;
            }
        }
    }
    if (distinct_formats.empty()) {
        out_log.notes.push_back("No supported native expert layout; portable execution retained.");
        return true;
    }

    const int warmup_iters = (mode == GExecOptMode::Quick) ? 2 : (mode == GExecOptMode::Exhaustive ? 6 : 3);
    const int bench_iters = (mode == GExecOptMode::Quick) ? 5 : (mode == GExecOptMode::Exhaustive ? 19 : 9);

    std::string overall_best_impl = "cpu.expert_pool.split_multi_native";
    std::string overall_best_pkg = "builtin.guild-kernels-cpu";
    uint32_t overall_best_ver = 0;
    double total_baseline_us = 0.0;
    double total_best_us = 0.0;
    struct CandidateResult {
        std::string impl_id;
        std::string pkg_id;
        uint32_t impl_ver = 0;
        double median_us = 0.0;
    };
    std::map<std::string, std::vector<CandidateResult>> global_candidates;

    for (size_t fi = 0; fi < distinct_formats.size(); ++fi) {
        if (interrupted())
            return false;
        const auto& fk = distinct_formats[fi];
        guild_cpu_expert_format_v1_t fmt{};
        guild_error_info_t einfo{};
        if (ref_table->format(fk.gu_type, fk.d_type, fk.n_embd, fk.n_ff, &fmt, &einfo) != GUILD_STATUS_OK) {
            continue;
        }
        const uint64_t active = std::max<int64_t>(1, gpack.descriptor().moe.k_active_experts);
        const uint64_t temporary_bytes =
            (64ull << 20) + 2 * active * fmt.bytes +
            (static_cast<uint64_t>(std::max(1, identity.hardware.cpu_logical_threads)) << 20);
        if (budget.max_memory_bytes && temporary_bytes > budget.max_memory_bytes) {
            err = "GEXEC autotuning working set exceeds memory budget";
            return false;
        }

        // Use the real model bytes, including the actual per-block scales.
        std::vector<uint8_t> blob_contig;
        if (!read_expert(gpack, format_layers[fi], 0, blob_contig, err))
            return false;
        std::vector<uint8_t> blob_interleaved(blob_contig.size(), 0);
        pack_expert_gate_up_interleaved(blob_contig.data(), blob_interleaved.data(), static_cast<size_t>(fmt.n_ff),
                                        static_cast<size_t>(fmt.gu_row), static_cast<size_t>(fmt.gu_row),
                                        static_cast<size_t>(fmt.bytes - fmt.down_off));

        // Prepare deterministic input activations for nt = 1 and nt = 2
        std::vector<float> x_in(static_cast<size_t>(2 * fmt.n_embd), 0.0f);
        for (size_t i = 0; i < x_in.size(); ++i) {
            x_in[i] = std::sin(static_cast<float>(i) * 0.03125f) * 0.5f;
        }
        alignas(64) uint8_t qact0[kernels::cpu::kNativeActBytes] = {};
        alignas(64) uint8_t qact1[kernels::cpu::kNativeActBytes] = {};
        ref_table->native.quant_act(&fmt, x_in.data(), qact0);
        ref_table->native.quant_act(&fmt, x_in.data() + fmt.n_embd, qact1);
        const void* act_ptrs[2] = {qact0, qact1};

        // Compute reference outputs using baseline cpu.expert_pool.split_multi_native
        std::vector<float> ref_ff0(static_cast<size_t>(fmt.n_ff), 0.0f);
        std::vector<float> ref_ff1(static_cast<size_t>(fmt.n_ff), 0.0f);
        float* ref_ff_ptrs[2] = {ref_ff0.data(), ref_ff1.data()};
        ref_table->native.gate_up(&fmt, blob_contig.data(), act_ptrs, 2, ref_ff_ptrs, 0, static_cast<int>(fmt.n_ff));

        alignas(64) uint8_t ref_hq0[kernels::cpu::kNativeHBytes] = {};
        alignas(64) uint8_t ref_hq1[kernels::cpu::kNativeHBytes] = {};
        ref_table->native.quant_h(&fmt, ref_ff0.data(), ref_hq0);
        ref_table->native.quant_h(&fmt, ref_ff1.data(), ref_hq1);
        const void* ref_hq_ptrs[2] = {ref_hq0, ref_hq1};

        std::vector<float> ref_out0(static_cast<size_t>(fmt.n_embd), 0.0f);
        std::vector<float> ref_out1(static_cast<size_t>(fmt.n_embd), 0.0f);
        float* ref_out_ptrs[2] = {ref_out0.data(), ref_out1.data()};
        ref_table->native.down(&fmt, blob_contig.data(), ref_hq_ptrs, 2, ref_out_ptrs, 0, static_cast<int>(fmt.n_embd));

        std::vector<CandidateResult> valid_candidates;
        double fmt_baseline_us = 0.0;

        for (const auto& op : all_ops) {
            if (interrupted())
                return false;
            if (op->op_kind != GUILD_OP_BATCHED_EXPERT_EXEC)
                continue;
            const std::string cid = op->impl_id ? op->impl_id : "";
            const std::string cpkg = op->package_id ? op->package_id : "";
            if (cid != "cpu.expert_pool.split_multi_native" && !kernels::cpu::is_gexec_optimized_cpu_impl(cid.c_str()))
                continue;
            if (!kernels::KernelRegistry::instance().is_package_enabled(cpkg))
                continue;

            // Stage 1: Hardware / Capability Compatibility Filter
            if ((op->backend & GUILD_BACKEND_CPU) == 0) {
                out_log.rejected_candidates.push_back({GUILD_OP_BATCHED_EXPERT_EXEC, "batched_expert_exec", cid,
                                                       "compatibility", "non-CPU backend", 0.0, 0.0});
                continue;
            }
            if ((op->required_cpu_isa & caps.cpu_isa_flags) != op->required_cpu_isa) {
                out_log.rejected_candidates.push_back({GUILD_OP_BATCHED_EXPERT_EXEC, "batched_expert_exec", cid,
                                                       "compatibility", "host CPU lacks required ISA flags", 0.0, 0.0});
                continue;
            }
            if (!op->next_ext) {
                out_log.rejected_candidates.push_back({GUILD_OP_BATCHED_EXPERT_EXEC, "batched_expert_exec", cid,
                                                       "compatibility",
                                                       "missing guild_cpu_backend_ext_v2_t extension table", 0.0, 0.0});
                continue;
            }

            const auto* tbl = static_cast<const guild_cpu_backend_ext_v2_t*>(op->next_ext);
            if (tbl->native.struct_size < sizeof(*tbl) || tbl->native.tag != GUILD_CPU_EXPERT_TAG_V1 ||
                tbl->native.version != 1 || tbl->backend_version != GUILD_CPU_BACKEND_VERSION_V2 || !tbl->format ||
                !tbl->native.gate_up || !tbl->native.quant_h || !tbl->native.down)
                continue;
            guild_cpu_expert_format_v1_t candidate_fmt{};
            if (tbl->format(fk.gu_type, fk.d_type, fk.n_embd, fk.n_ff, &candidate_fmt, nullptr) != GUILD_STATUS_OK ||
                std::memcmp(&candidate_fmt, &fmt, sizeof(fmt)) != 0)
                continue;
            const bool is_interleaved = kernels::cpu::is_gexec_interleaved_impl(cid.c_str());
            const uint8_t* test_blob = is_interleaved ? blob_interleaved.data() : blob_contig.data();

            // Stage 2: Numerical Correctness & Bitwise/Tolerance Verification (nt = 1 and nt = 2)
            std::vector<float> cand_ff0(static_cast<size_t>(fmt.n_ff), 0.0f);
            std::vector<float> cand_ff1(static_cast<size_t>(fmt.n_ff), 0.0f);
            float* cand_ff_ptrs[2] = {cand_ff0.data(), cand_ff1.data()};

            // Test nt = 1
            ref_table->native.gate_up(&fmt, blob_contig.data(), act_ptrs, 1, ref_ff_ptrs, 0,
                                      static_cast<int>(fmt.n_ff));
            ref_table->native.quant_h(&fmt, ref_ff0.data(), ref_hq0);
            ref_table->native.down(&fmt, blob_contig.data(), ref_hq_ptrs, 1, ref_out_ptrs, 0,
                                   static_cast<int>(fmt.n_embd));
            tbl->native.gate_up(&fmt, test_blob, act_ptrs, 1, cand_ff_ptrs, 0, static_cast<int>(fmt.n_ff));
            alignas(64) uint8_t cand_hq0[kernels::cpu::kNativeHBytes] = {};
            tbl->native.quant_h(&fmt, cand_ff0.data(), cand_hq0);
            const void* cand_hq_ptrs[2] = {cand_hq0, ref_hq1};
            std::vector<float> cand_out0(static_cast<size_t>(fmt.n_embd), 0.0f);
            std::vector<float> cand_out1(static_cast<size_t>(fmt.n_embd), 0.0f);
            float* cand_out_ptrs[2] = {cand_out0.data(), cand_out1.data()};
            tbl->native.down(&fmt, test_blob, cand_hq_ptrs, 1, cand_out_ptrs, 0, static_cast<int>(fmt.n_embd));

            double max_err = 0.0;
            bool finite_ok = true;
            bool exact = std::memcmp(cand_ff0.data(), ref_ff0.data(), fmt.n_ff * sizeof(float)) == 0 &&
                         std::memcmp(cand_hq0, ref_hq0, fmt.h_bytes) == 0 &&
                         std::memcmp(cand_out0.data(), ref_out0.data(), fmt.n_embd * sizeof(float)) == 0;
            for (size_t i = 0; i < static_cast<size_t>(fmt.n_embd); ++i) {
                if (!std::isfinite(cand_out0[i])) {
                    finite_ok = false;
                    break;
                }
                max_err = std::max(max_err, static_cast<double>(std::fabs(cand_out0[i] - ref_out0[i])));
            }
            // Test nt = 2
            ref_table->native.gate_up(&fmt, blob_contig.data(), act_ptrs, 2, ref_ff_ptrs, 0,
                                      static_cast<int>(fmt.n_ff));
            ref_table->native.quant_h(&fmt, ref_ff0.data(), ref_hq0);
            ref_table->native.quant_h(&fmt, ref_ff1.data(), ref_hq1);
            ref_table->native.down(&fmt, blob_contig.data(), ref_hq_ptrs, 2, ref_out_ptrs, 0,
                                   static_cast<int>(fmt.n_embd));
            tbl->native.gate_up(&fmt, test_blob, act_ptrs, 2, cand_ff_ptrs, 0, static_cast<int>(fmt.n_ff));
            alignas(64) uint8_t cand_hq1[kernels::cpu::kNativeHBytes] = {};
            tbl->native.quant_h(&fmt, cand_ff0.data(), cand_hq0);
            tbl->native.quant_h(&fmt, cand_ff1.data(), cand_hq1);
            cand_hq_ptrs[0] = cand_hq0;
            cand_hq_ptrs[1] = cand_hq1;
            tbl->native.down(&fmt, test_blob, cand_hq_ptrs, 2, cand_out_ptrs, 0, static_cast<int>(fmt.n_embd));
            exact = exact && std::memcmp(cand_ff0.data(), ref_ff0.data(), fmt.n_ff * sizeof(float)) == 0 &&
                    std::memcmp(cand_ff1.data(), ref_ff1.data(), fmt.n_ff * sizeof(float)) == 0 &&
                    std::memcmp(cand_hq0, ref_hq0, fmt.h_bytes) == 0 &&
                    std::memcmp(cand_hq1, ref_hq1, fmt.h_bytes) == 0 &&
                    std::memcmp(cand_out0.data(), ref_out0.data(), fmt.n_embd * sizeof(float)) == 0 &&
                    std::memcmp(cand_out1.data(), ref_out1.data(), fmt.n_embd * sizeof(float)) == 0;
            for (size_t i = 0; i < static_cast<size_t>(fmt.n_embd); ++i) {
                if (!std::isfinite(cand_out0[i]) || !std::isfinite(cand_out1[i])) {
                    finite_ok = false;
                    break;
                }
                max_err = std::max(max_err, static_cast<double>(std::fabs(cand_out0[i] - ref_out0[i])));
                max_err = std::max(max_err, static_cast<double>(std::fabs(cand_out1[i] - ref_out1[i])));
            }

            if (!finite_ok || !exact) {
                out_log.rejected_candidates.push_back(
                    {GUILD_OP_BATCHED_EXPERT_EXEC, "batched_expert_exec", cid, "correctness",
                     finite_ok ? "bitwise parity mismatch" : "non-finite output", max_err, 0.0});
                continue;
            }

            // Stage 3: single-expert, single-token microbenchmark.
            for (int w = 0; w < warmup_iters; ++w) {
                tbl->native.gate_up(&fmt, test_blob, act_ptrs, 1, cand_ff_ptrs, 0, static_cast<int>(fmt.n_ff));
                tbl->native.quant_h(&fmt, cand_ff0.data(), cand_hq0);
                tbl->native.down(&fmt, test_blob, cand_hq_ptrs, 1, cand_out_ptrs, 0, static_cast<int>(fmt.n_embd));
            }
            std::vector<double> samples_us;
            samples_us.reserve(static_cast<size_t>(bench_iters));
            for (int it = 0; it < bench_iters; ++it) {
                const auto t0 = std::chrono::steady_clock::now();
                tbl->native.gate_up(&fmt, test_blob, act_ptrs, 1, cand_ff_ptrs, 0, static_cast<int>(fmt.n_ff));
                tbl->native.quant_h(&fmt, cand_ff0.data(), cand_hq0);
                tbl->native.down(&fmt, test_blob, cand_hq_ptrs, 1, cand_out_ptrs, 0, static_cast<int>(fmt.n_embd));
                const double us =
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                samples_us.push_back(us);
            }
            std::sort(samples_us.begin(), samples_us.end());
            const double med_us = samples_us[samples_us.size() / 2];

            if (cid == "cpu.expert_pool.split_multi_native") {
                fmt_baseline_us = med_us;
            }
            valid_candidates.push_back({cid, cpkg, op->impl_version, med_us});
            global_candidates[cid].push_back(valid_candidates.back());
        }

        if (valid_candidates.empty()) {
            err = "no valid CPU expert kernel candidates survived verification";
            return false;
        }

        // Keep the comparator transitive; noisy ties cannot define a sort order.
        std::sort(valid_candidates.begin(), valid_candidates.end(),
                  [](const CandidateResult& a, const CandidateResult& b) {
                      if (a.median_us == b.median_us)
                          return a.impl_id < b.impl_id;
                      return a.median_us < b.median_us;
                  });

        const auto& winner = valid_candidates.front();
        for (size_t i = 1; i < valid_candidates.size(); ++i) {
            out_log.rejected_candidates.push_back(
                {GUILD_OP_BATCHED_EXPERT_EXEC,
                 "batched_expert_exec(gu=" + std::to_string(fk.gu_type) + ",d=" + std::to_string(fk.d_type) + ")",
                 valid_candidates[i].impl_id, "microbenchmark", "slower than winning candidate " + winner.impl_id, 0.0,
                 valid_candidates[i].median_us});
        }

        total_baseline_us += (fmt_baseline_us > 0.0 ? fmt_baseline_us : winner.median_us);
        total_best_us += winner.median_us;
        overall_best_impl = winner.impl_id;
        overall_best_pkg = winner.pkg_id;
        overall_best_ver = winner.impl_ver;

        // Record for m_tokens = 1 (Decode) and m_tokens = 2 (Verify)
        for (int64_t m_tok : {int64_t(1), int64_t(2)}) {
            GExecKernelTuneRecord rec;
            rec.op_kind = GUILD_OP_BATCHED_EXPERT_EXEC;
            rec.operation_name = "batched_expert_exec";
            rec.phase = (m_tok == 1) ? GExecWorkloadPhase::Decode : GExecWorkloadPhase::Verify;
            rec.backend = GUILD_BACKEND_CPU;
            rec.quant_type = fk.gu_type;
            rec.m_tokens = m_tok;
            rec.n_cols = fk.n_ff;
            rec.k_cols = fk.n_embd;
            rec.selected_impl_id = winner.impl_id;
            rec.selected_package_id = winner.pkg_id;
            rec.selected_impl_version = winner.impl_ver;
            rec.fallback_impl_id = "cpu.expert_pool.split_multi_native";
            rec.median_latency_us = winner.median_us;
            rec.candidate_count = static_cast<uint32_t>(valid_candidates.size());
            out_table.records.push_back(std::move(rec));
        }
    }

    struct RankedCandidate {
        std::string impl_id;
        std::string pkg_id;
        uint32_t impl_ver = 0;
        double score = 0.0;
    };
    std::vector<RankedCandidate> ranked_candidates;
    for (const auto& [cid, results] : global_candidates) {
        if (results.size() != distinct_formats.size() || cid == "cpu.expert_pool.split_multi_native")
            continue;
        double score = 0;
        for (size_t i = 0; i < results.size(); ++i)
            score += results[i].median_us * format_counts.at(distinct_formats[i]);
        ranked_candidates.push_back({cid, results.front().pkg_id, results.front().impl_ver, score});
    }
    std::sort(ranked_candidates.begin(), ranked_candidates.end(),
              [](const RankedCandidate& a, const RankedCandidate& b) {
                  if (a.score == b.score)
                      return a.impl_id < b.impl_id;
                  return a.score < b.score;
              });
    const auto baseline = global_candidates.find("cpu.expert_pool.split_multi_native");
    if (ranked_candidates.empty() &&
        (baseline == global_candidates.end() || baseline->second.size() != distinct_formats.size())) {
        err = "no kernel passed every model format";
        return false;
    }

    // Stage 4: routed-expert pool trial, not a full-model decode measurement.
    // Evaluate all valid GEXEC candidates in a single shared ExpertPool so AVX-512
    // multi-token kernels and both layouts compete under the real worker schedule.
    std::vector<RankedCandidate>& pool_candidates = ranked_candidates;

    kernels::cpu::ExpertPool pool(/*n_workers=*/0, /*pin=*/false);
    double baseline_pool_ms = 0.0;
    std::vector<double> cand_pool_ms(pool_candidates.size(), 0.0);
    std::vector<bool> cand_exact(pool_candidates.size(), true);
    std::vector<double> cand_max_err(pool_candidates.size(), 0.0);

    for (size_t fi = 0; !pool_candidates.empty() && fi < distinct_formats.size(); ++fi) {
        if (interrupted())
            return false;
        const auto& fk = distinct_formats[fi];
        guild_cpu_expert_format_v1_t fmt{};
        ref_table->format(fk.gu_type, fk.d_type, fk.n_embd, fk.n_ff, &fmt, nullptr);

        const int k_active =
            std::max(1, static_cast<int>(
                            gpack.descriptor().moe.k_active_experts > 0 ? gpack.descriptor().moe.k_active_experts : 8));
        std::vector<std::vector<uint8_t>> blobs_contig(static_cast<size_t>(k_active));
        std::vector<std::vector<uint8_t>> blobs_inter(static_cast<size_t>(k_active));
        for (int e = 0; e < k_active; ++e) {
            if (!read_expert(gpack, format_layers[fi], e % gpack.expert_count_per_layer(),
                             blobs_contig[static_cast<size_t>(e)], err))
                return false;
            blobs_inter[static_cast<size_t>(e)].resize(blobs_contig[static_cast<size_t>(e)].size());
            pack_expert_gate_up_interleaved(blobs_contig[static_cast<size_t>(e)].data(),
                                            blobs_inter[static_cast<size_t>(e)].data(), static_cast<size_t>(fmt.n_ff),
                                            static_cast<size_t>(fmt.gu_row), static_cast<size_t>(fmt.gu_row),
                                            static_cast<size_t>(fmt.bytes - fmt.down_off));
        }

        const int pool_nt = (identity.mtp_spec_tokens > 0) ? 2 : 1;
        std::vector<float> x_in0(static_cast<size_t>(fmt.n_embd), 0.125f);
        std::vector<float> x_in1(static_cast<size_t>(fmt.n_embd), -0.0625f);
        alignas(64) uint8_t qact0[kernels::cpu::kNativeActBytes] = {};
        alignas(64) uint8_t qact1[kernels::cpu::kNativeActBytes] = {};
        ref_table->native.quant_act(&fmt, x_in0.data(), qact0);
        ref_table->native.quant_act(&fmt, x_in1.data(), qact1);

        const size_t n_embd_sz = static_cast<size_t>(fmt.n_embd);
        const size_t k_embd_sz = static_cast<size_t>(k_active) * n_embd_sz;
        std::vector<float> combine_w(static_cast<size_t>(k_active), 1.0f / static_cast<float>(k_active));
        std::vector<float> outs_base0(k_embd_sz, 0.0f), outs_base1(k_embd_sz, 0.0f);
        std::vector<float> comb_base0(n_embd_sz, 0.0f), comb_base1(n_embd_sz, 0.0f);
        std::vector<kernels::cpu::ExpertJobMulti> jobs_base(static_cast<size_t>(k_active));
        for (int e = 0; e < k_active; ++e) {
            jobs_base[static_cast<size_t>(e)].blob = blobs_contig[static_cast<size_t>(e)].data();
            jobs_base[static_cast<size_t>(e)].nt = pool_nt;
            jobs_base[static_cast<size_t>(e)].nact[0] = qact0;
            jobs_base[static_cast<size_t>(e)].nact[1] = qact1;
            jobs_base[static_cast<size_t>(e)].out[0] = outs_base0.data() + static_cast<size_t>(e) * n_embd_sz;
            jobs_base[static_cast<size_t>(e)].out[1] = outs_base1.data() + static_cast<size_t>(e) * n_embd_sz;
        }

        std::vector<std::vector<float>> outs_cands0(pool_candidates.size(), std::vector<float>(k_embd_sz, 0.0f));
        std::vector<std::vector<float>> outs_cands1(pool_candidates.size(), std::vector<float>(k_embd_sz, 0.0f));
        std::vector<std::vector<float>> comb_cands0(pool_candidates.size(), std::vector<float>(n_embd_sz, 0.0f));
        std::vector<std::vector<float>> comb_cands1(pool_candidates.size(), std::vector<float>(n_embd_sz, 0.0f));
        std::vector<std::vector<kernels::cpu::ExpertJobMulti>> jobs_cands(
            pool_candidates.size(), std::vector<kernels::cpu::ExpertJobMulti>(static_cast<size_t>(k_active)));
        std::vector<const guild_cpu_backend_ext_v2_t*> tables_cands(pool_candidates.size(), ref_table);

        for (size_t ci = 0; ci < pool_candidates.size(); ++ci) {
            const bool use_inter = kernels::cpu::is_gexec_interleaved_impl(pool_candidates[ci].impl_id.c_str());
            const auto bnd = kernels::KernelRegistry::instance().find_by_id(pool_candidates[ci].impl_id);
            if (bnd && bnd->next_ext) {
                tables_cands[ci] = static_cast<const guild_cpu_backend_ext_v2_t*>(bnd->next_ext);
            }
            for (int e = 0; e < k_active; ++e) {
                jobs_cands[ci][static_cast<size_t>(e)].blob = use_inter ? blobs_inter[static_cast<size_t>(e)].data()
                                                                        : blobs_contig[static_cast<size_t>(e)].data();
                jobs_cands[ci][static_cast<size_t>(e)].nt = pool_nt;
                jobs_cands[ci][static_cast<size_t>(e)].nact[0] = qact0;
                jobs_cands[ci][static_cast<size_t>(e)].nact[1] = qact1;
                jobs_cands[ci][static_cast<size_t>(e)].out[0] =
                    outs_cands0[ci].data() + static_cast<size_t>(e) * n_embd_sz;
                jobs_cands[ci][static_cast<size_t>(e)].out[1] =
                    outs_cands1[ci].data() + static_cast<size_t>(e) * n_embd_sz;
            }
        }

        auto run_base = [&]() {
            pool.bind_backend_kernels(*ref_table);
            pool.set_fused_pipeline(false);
            const auto start = std::chrono::steady_clock::now();
            pool.run_split_multi_native(fmt, jobs_base.data(), k_active);
            ref_table->combine(outs_base0.data(), combine_w.data(), comb_base0.data(), k_active,
                               static_cast<int>(fmt.n_embd));
            if (pool_nt >= 2) {
                ref_table->combine(outs_base1.data(), combine_w.data(), comb_base1.data(), k_active,
                                   static_cast<int>(fmt.n_embd));
            }
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        };
        auto run_cand = [&](size_t ci) {
            pool.bind_backend_kernels(*tables_cands[ci]);
            pool.set_fused_pipeline(kernels::cpu::is_gexec_optimized_cpu_impl(pool_candidates[ci].impl_id.c_str()));
            const auto start = std::chrono::steady_clock::now();
            pool.run_split_multi_native(fmt, jobs_cands[ci].data(), k_active);
            tables_cands[ci]->combine(outs_cands0[ci].data(), combine_w.data(), comb_cands0[ci].data(), k_active,
                                      static_cast<int>(fmt.n_embd));
            if (pool_nt >= 2) {
                tables_cands[ci]->combine(outs_cands1[ci].data(), combine_w.data(), comb_cands1[ci].data(), k_active,
                                          static_cast<int>(fmt.n_embd));
            }
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        };

        for (int w = 0; w < warmup_iters; ++w) {
            run_base();
            for (size_t ci = 0; ci < pool_candidates.size(); ++ci)
                run_cand(ci);
        }
        std::vector<double> base_samples;
        std::vector<std::vector<double>> cand_samples(pool_candidates.size());
        for (int it = 0; it < bench_iters; ++it) {
            if (interrupted())
                return false;
            if (it % 2 == 0) {
                base_samples.push_back(run_base());
                for (size_t ci = 0; ci < pool_candidates.size(); ++ci)
                    cand_samples[ci].push_back(run_cand(ci));
            } else {
                for (size_t ci = pool_candidates.size(); ci-- > 0;)
                    cand_samples[ci].push_back(run_cand(ci));
                base_samples.push_back(run_base());
            }
        }
        std::sort(base_samples.begin(), base_samples.end());
        const double base_med = base_samples[base_samples.size() / 2];
        baseline_pool_ms += base_med * format_counts.at(fk);

        for (size_t ci = 0; ci < pool_candidates.size(); ++ci) {
            std::sort(cand_samples[ci].begin(), cand_samples[ci].end());
            const double c_med = cand_samples[ci][cand_samples[ci].size() / 2];
            cand_pool_ms[ci] += c_med * format_counts.at(fk);

            double fmt_max_err = 0.0;
            bool exact = true;
            for (size_t i = 0; i < k_embd_sz; ++i) {
                if (!std::isfinite(outs_cands0[ci][i]) || !std::isfinite(outs_base0[i]))
                    exact = false;
                fmt_max_err = std::max(fmt_max_err, static_cast<double>(std::fabs(outs_cands0[ci][i] - outs_base0[i])));
                if (pool_nt >= 2) {
                    if (!std::isfinite(outs_cands1[ci][i]) || !std::isfinite(outs_base1[i]))
                        exact = false;
                    fmt_max_err =
                        std::max(fmt_max_err, static_cast<double>(std::fabs(outs_cands1[ci][i] - outs_base1[i])));
                }
            }
            exact = exact && std::memcmp(outs_cands0[ci].data(), outs_base0.data(), k_embd_sz * sizeof(float)) == 0;
            exact = exact && std::memcmp(comb_cands0[ci].data(), comb_base0.data(), n_embd_sz * sizeof(float)) == 0;
            if (pool_nt >= 2) {
                exact = exact && std::memcmp(outs_cands1[ci].data(), outs_base1.data(), k_embd_sz * sizeof(float)) == 0;
                exact = exact && std::memcmp(comb_cands1[ci].data(), comb_base1.data(), n_embd_sz * sizeof(float)) == 0;
            }
            cand_max_err[ci] = std::max(cand_max_err[ci], fmt_max_err);
            if (!exact)
                cand_exact[ci] = false;
            out_log.notes.push_back("ExpertPool format " + std::to_string(fi) + " (" + pool_candidates[ci].impl_id +
                                    "): baseline_ms=" + std::to_string(base_med) + ", candidate_ms=" +
                                    std::to_string(c_med) + ", max_abs_err=" + std::to_string(fmt_max_err));
        }
    }

    overall_best_impl = "cpu.expert_pool.split_multi_native";
    if (const auto bnd = kernels::KernelRegistry::instance().find_by_id(overall_best_impl)) {
        overall_best_ver = bnd->impl_version;
        overall_best_pkg = bnd->package_id;
    }
    double best_pool_ms = baseline_pool_ms;
    for (size_t ci = 0; ci < pool_candidates.size(); ++ci) {
        if (!cand_exact[ci]) {
            out_log.rejected_candidates.push_back(
                {GUILD_OP_BATCHED_EXPERT_EXEC, "batched_expert_exec", pool_candidates[ci].impl_id, "pool",
                 "failed multi-thread ExpertPool bitwise parity check", cand_max_err[ci], cand_pool_ms[ci] * 1000.0});
            continue;
        }
        if (cand_pool_ms[ci] < best_pool_ms) {
            best_pool_ms = cand_pool_ms[ci];
            overall_best_impl = pool_candidates[ci].impl_id;
            overall_best_pkg = pool_candidates[ci].pkg_id;
            overall_best_ver = pool_candidates[ci].impl_ver;
        } else {
            out_log.rejected_candidates.push_back(
                {GUILD_OP_BATCHED_EXPERT_EXEC, "batched_expert_exec", pool_candidates[ci].impl_id, "pool",
                 "slower in multi-thread ExpertPool trial", 0.0, cand_pool_ms[ci] * 1000.0});
        }
    }
    out_log.notes.push_back("Model-wide pool winner: " + overall_best_impl);
    for (size_t i = 0; i < out_table.records.size(); ++i) {
        auto& r = out_table.records[i];
        r.selected_impl_id = overall_best_impl;
        r.selected_package_id = overall_best_pkg;
        r.selected_impl_version = overall_best_ver;
        r.median_latency_us =
            r.phase == GExecWorkloadPhase::Decode ? global_candidates.at(overall_best_impl).at(i / 2).median_us : 0.0;
    }
    out_log.notes.push_back("Full-model decode and break-even have not been measured by this pool trial.");

    // Also lock GUILD_OP_FINITE_CHECK and GUILD_OP_QUANTIZE_ACT in the autotune table
    {
        GExecKernelTuneRecord frec;
        frec.op_kind = GUILD_OP_FINITE_CHECK;
        frec.operation_name = "finite_check";
        frec.phase = GExecWorkloadPhase::Decode;
        frec.backend = GUILD_BACKEND_CPU;
        frec.selected_impl_id = "cpu.finite_check.simd64";
        frec.selected_package_id = "builtin.guild-kernels-cpu";
        frec.selected_impl_version = GUILD_KERNEL_ABI_MAKE_VERSION(1, 0, 0);
        frec.fallback_impl_id = "cpu.finite_check.simd64";
        frec.candidate_count = 1;
        out_table.records.push_back(std::move(frec));
    }

    out_selected_cpu_impl = overall_best_impl;
    out_selected_cpu_ver = overall_best_ver;

    const bool winner_interleaved = kernels::cpu::is_gexec_interleaved_impl(overall_best_impl.c_str());
    inout_pack_meta.layout_kind =
        winner_interleaved ? ExpertPackLayoutKind::GateUpRowInterleaved : ExpertPackLayoutKind::PortableContiguous;
    inout_pack_meta.fused_single_barrier_pipeline =
        kernels::cpu::is_gexec_optimized_cpu_impl(overall_best_impl.c_str());
    for (auto& ls : inout_pack_meta.layer_specs) {
        ls.layout_kind = inout_pack_meta.layout_kind;
    }
    return true;
}

} // namespace

GExecOptimizeResult optimize_gpack(const GExecOptimizeOptions& opts,
                                   std::function<void(const std::string& stage_msg)> progress_cb) {
    GExecOptimizeResult res;
    if (!std::isfinite(opts.budget.max_time_seconds) || opts.budget.max_time_seconds < 0 || opts.max_context <= 0 ||
        opts.prefill_batch <= 0) {
        res.error = "invalid GEXEC time or execution-plan budget";
        return res;
    }
    GExecOptBudget budget = opts.budget;
    if (!budget.max_memory_bytes)
        budget.max_memory_bytes = hardware::detect_hardware().ram_free_bytes / 2;
    const auto wall_t0 = std::chrono::steady_clock::now();
    auto elapsed_sec = [&]() -> double {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_t0).count();
    };
    auto budget_expired = [&](int32_t pass_num) -> bool {
        if (opts.stop_after_pass >= 0 && pass_num >= opts.stop_after_pass)
            return true;
        if (opts.budget.max_time_seconds > 0.0 && elapsed_sec() >= opts.budget.max_time_seconds)
            return true;
        return false;
    };

    if (opts.mode == GExecOptMode::Skip) {
        res.ok = true;
        res.skipped = true;
        return res;
    }

    artifact::GPackReader gpack;
    std::string gerr;
    if (!gpack.open(opts.gpack_path, gerr, /*verify_hashes=*/false)) {
        res.error = "failed to open GPACK '" + opts.gpack_path + "': " + gerr;
        return res;
    }

    const GExecArtifactIdentity identity = build_current_gexec_identity(
        gpack, opts.kv_format, opts.kv_mode, opts.max_context, opts.prefill_batch, opts.mtp_spec_tokens);
    res.cache_key = identity.compute_cache_key();
    res.output_path = opts.output_gexec_path.empty()
                          ? default_gexec_path_for_gpack(opts.gpack_path, identity, opts.cache_dir)
                          : opts.output_gexec_path;

    if (budget.only_section != GExecSectionKind::Invalid) {
        if (!regenerate_gexec_section(res.output_path, opts.gpack_path, budget.only_section, res.error, budget))
            return res;
        GExecReader updated;
        if (!updated.open(res.output_path, res.error))
            return res;
        res.ok = true;
        res.cache_key = updated.identity().compute_cache_key();
        res.partial_artifact = updated.is_partial();
        res.expert_layout_kind = updated.expert_pack_meta().layout_kind;
        res.fused_single_barrier_pipeline = updated.expert_pack_meta().fused_single_barrier_pipeline;
        for (const auto& record : updated.autotune_table().records) {
            if (record.op_kind == GUILD_OP_BATCHED_EXPERT_EXEC) {
                res.selected_cpu_expert_impl_id = record.selected_impl_id;
                res.selected_cpu_expert_impl_version = record.selected_impl_version;
                break;
            }
        }
        res.artifact_bytes = updated.header().total_file_bytes;
        res.optimization_wall_ms = elapsed_sec() * 1000.0;
        return res;
    }

    ScopedAdvisoryLock lock;
    if (!lock.acquire(res.output_path + ".lock", res.error)) {
        return res;
    }

    const std::string progress_path = res.output_path + ".progress.json";
    CheckpointState ckpt;
    ckpt.cache_key = res.cache_key;

    if (opts.budget.resume_interrupted && std::filesystem::exists(progress_path)) {
        std::ifstream ifs(progress_path, std::ios::binary);
        std::string jstr((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        CheckpointState loaded;
        if (CheckpointState::from_json(jstr, loaded) && loaded.cache_key == res.cache_key) {
            ckpt = std::move(loaded);
            res.resumed_from_checkpoint = true;
            if (progress_cb) {
                progress_cb("Resumed interrupted GEXEC optimization from pass " +
                            std::to_string(ckpt.last_completed_pass));
            }
        }
    }

    auto flush_partial_if_allowed = [&]() -> bool {
        if (!opts.budget.allow_partial_artifact || ckpt.last_completed_pass < 1) {
            res.error = "GEXEC optimization budget expired before completion";
            return false;
        }
        GExecBuilder pbuilder;
        pbuilder.set_flags(kGExecFlagPartialArtifact |
                           (ckpt.pack_meta.fused_single_barrier_pipeline ? kGExecFlagFusedPipeline : 0u));
        pbuilder.set_identity(ckpt.identity);
        if (ckpt.last_completed_pass >= 2)
            pbuilder.set_expert_pack_meta(ckpt.pack_meta);
        if (ckpt.last_completed_pass >= 3) {
            pbuilder.set_autotune_table(ckpt.autotune);
            pbuilder.set_diagnostic_log(ckpt.diag_log);
        }
        if (ckpt.last_completed_pass >= 4)
            pbuilder.set_heterogeneous_plan(ckpt.hplan);
        if (!pbuilder.write_file(res.output_path, res.error, budget.max_disk_bytes))
            return false;
        res.ok = true;
        res.partial_artifact = true;
        res.optimization_wall_ms = elapsed_sec() * 1000.0;
        std::error_code ec;
        res.artifact_bytes = static_cast<uint64_t>(std::filesystem::file_size(res.output_path, ec));
        return true;
    };

    // Pass 1: Identity & Manifest
    if (ckpt.last_completed_pass < 1) {
        if (progress_cb)
            progress_cb("Pass 1/5: Computing hardware/software identity and ABI compatibility contract");
        ckpt.identity = identity;
        ckpt.last_completed_pass = 1;
        save_checkpoint_atomic(progress_path, ckpt);
    }
    if (budget_expired(1)) {
        flush_partial_if_allowed();
        return res;
    }

    // Pass 2: Expert Packing Metadata
    if (ckpt.last_completed_pass < 2) {
        if (progress_cb)
            progress_cb("Pass 2/5: Analyzing MoE expert layout and GateUpRowInterleaved specialization");
        GExecExpertPackMeta pmeta;
        pmeta.layout_kind = ExpertPackLayoutKind::PortableContiguous;
        pmeta.embedded_blob_present = false;
        pmeta.fused_single_barrier_pipeline = false;
        pmeta.n_layers = static_cast<int64_t>(gpack.expert_layers().size());
        pmeta.n_expert = gpack.expert_count_per_layer();
        pmeta.total_expert_bytes = gpack.total_expert_bytes();
        const int64_t m_embd = gpack.descriptor().attn.n_embd;
        const int64_t m_ff = gpack.descriptor().moe.expert_dim_ff;
        for (const auto& lrow : gpack.expert_layers()) {
            GExecExpertPackSpec s;
            s.layout_kind = pmeta.layout_kind;
            s.gu_type = lrow.gu_type;
            s.d_type = lrow.d_type;
            s.n_embd = m_embd;
            s.n_ff = m_ff;
            s.alignment = 64;
            pmeta.layer_specs.push_back(s);
        }
        ckpt.pack_meta = std::move(pmeta);
        ckpt.last_completed_pass = 2;
        save_checkpoint_atomic(progress_path, ckpt);
    }
    if (budget_expired(2)) {
        flush_partial_if_allowed();
        return res;
    }

    // Pass 3: Staged Kernel Autotuning & Numerical Verification
    if (ckpt.last_completed_pass < 3) {
        if (progress_cb)
            progress_cb("Pass 3/5: Staged kernel autotuning (compatibility -> parity -> microbench -> pool)");
        std::string sel_impl;
        uint32_t sel_ver = 0;
        if (!run_autotune_pass(gpack, ckpt.identity, opts.mode, ckpt.pack_meta, ckpt.autotune, ckpt.diag_log, sel_impl,
                               sel_ver, res.error, budget, [&] {
                                   return budget.max_time_seconds > 0 && elapsed_sec() >= budget.max_time_seconds;
                               })) {
            if (res.error.find("budget") != std::string::npos)
                flush_partial_if_allowed();
            return res;
        }
        ckpt.last_completed_pass = 3;
        save_checkpoint_atomic(progress_path, ckpt);
    }
    if (budget_expired(3)) {
        flush_partial_if_allowed();
        return res;
    }

    // Pass 4: NUMA & Heterogeneous Execution Plan
    if (ckpt.last_completed_pass < 4) {
        if (progress_cb)
            progress_cb("Pass 4/5: Computing NUMA/THP memory policy and heterogeneous placement plan");
        GExecHeterogeneousPlan hp;
        hp.identity = ckpt.identity;
        hp.numa_policy = (ckpt.identity.hardware.numa_nodes > 1) ? "interleave" : "process";
        hp.use_thp_hugepages =
            ckpt.identity.hardware.thp_madvise_available || (gpack.total_expert_bytes() >= (1ull << 30));
        hp.measured_formats = 0; // CPU pool trials do not calibrate GPU/PCIe placement.
        hp.calibrated_cpu_ms = 0.0;
        hp.calibrated_gpu_ms = 0.0;
        hp.calibrated_copy_ms = 0.0;
        hp.expert_cache_slots = 0;
        const size_t n_layers = gpack.expert_layers().size();
        hp.gpu_layers.assign(n_layers, 1u);
        hp.pcie_numerators.assign(n_layers, 0);
        ckpt.hplan = std::move(hp);
        ckpt.last_completed_pass = 4;
        save_checkpoint_atomic(progress_path, ckpt);
    }
    if (budget_expired(4)) {
        flush_partial_if_allowed();
        return res;
    }

    // Pass 5: Workspace Recipe & Compilation Provenance
    if (ckpt.last_completed_pass < 5) {
        if (progress_cb)
            progress_cb("Pass 5/5: Precomputing execution workspace recipe and provenance metadata");
        GExecWorkspaceRecipe wr;
        wr.dense_arena_bytes =
            gpack.dense_compacted_pool_bytes() > 0 ? gpack.dense_compacted_pool_bytes() : gpack.dense_pool_bytes();
        wr.native_dense_arena_bytes = gpack.native_dense_arena_bytes();
        wr.capture_commit_graph = false;
        ckpt.workspace = std::move(wr);

        GExecCompilationProvenance prov;
        prov.key.cuda_runtime_version = ckpt.identity.hardware.cuda_runtime_version;
        prov.key.cuda_driver_version = ckpt.identity.hardware.cuda_driver_version;
        prov.key.gpu_cc_major = ckpt.identity.hardware.gpu_cc_major;
        prov.key.gpu_cc_minor = ckpt.identity.hardware.gpu_cc_minor;
        prov.key.kernel_package_sha256 = ckpt.identity.compute_cache_key();
        JsonValue build_info;
        std::string parse_error;
        JsonValue::parse(build::metadata_json, build_info, parse_error);
        prov.compiler_id = build_info["compiler"].as_string();
        prov.guild_version = build_info["software_version"].as_string();
        prov.git_commit = build_info["commit"].as_string();
        prov.created_unix_seconds = static_cast<int64_t>(std::time(nullptr));
        prov.optimization_mode = opt_mode_to_string(opts.mode);
        prov.optimization_wall_ms = elapsed_sec() * 1000.0;

        // Compute break-even token count: optimization_wall_ms / max(1e-6, baseline_decode_ms - optimized_decode_ms)
        const double saved_ms_per_tok = ckpt.diag_log.baseline_decode_ms - ckpt.diag_log.optimized_decode_ms;
        if (saved_ms_per_tok > 0.01) {
            ckpt.diag_log.break_even_tokens = prov.optimization_wall_ms / saved_ms_per_tok;
        } else {
            ckpt.diag_log.break_even_tokens = 0.0;
        }

        ckpt.provenance = std::move(prov);
        ckpt.last_completed_pass = 5;
        save_checkpoint_atomic(progress_path, ckpt);
    }

    // Build final GEXEC binary container
    GExecBuilder builder;
    uint32_t hdr_flags = ckpt.pack_meta.fused_single_barrier_pipeline ? kGExecFlagFusedPipeline : 0u;

    if (budget.embed_packed_experts && gpack.total_expert_bytes()) {
        uint64_t max_blob = 0;
        for (const auto& layer : gpack.expert_layers())
            max_blob = std::max(max_blob, layer.blob_bytes);
        if (budget.max_memory_bytes && 2 * max_blob + (2ull << 20) > budget.max_memory_bytes) {
            res.error = "expert packing working set exceeds memory budget";
            return res;
        }
        const bool interleaved = ckpt.pack_meta.layout_kind == ExpertPackLayoutKind::GateUpRowInterleaved;
        builder.set_packed_experts_stream(
            gpack.total_expert_bytes(),
            [&, interleaved, cached_offset = UINT64_MAX, source = std::vector<uint8_t>{},
             packed = std::vector<uint8_t>{}](uint64_t offset, uint8_t* dst, size_t count, std::string& err) mutable {
                while (count) {
                    if (budget.max_time_seconds > 0 && elapsed_sec() >= budget.max_time_seconds) {
                        err = "expert packing time budget expired";
                        return false;
                    }
                    size_t l = 0;
                    for (; l < gpack.expert_layers().size(); ++l) {
                        const auto& row = gpack.expert_layers()[l];
                        if (offset >= row.layer_offset && row.blob_bytes &&
                            (offset - row.layer_offset) / row.blob_bytes <
                                static_cast<uint64_t>(gpack.expert_count_per_layer()))
                            break;
                    }
                    if (l == gpack.expert_layers().size()) {
                        err = "invalid GPACK packing offset";
                        return false;
                    }
                    const auto& row = gpack.expert_layers()[l];
                    const uint64_t expert = (offset - row.layer_offset) / row.blob_bytes;
                    const uint64_t begin = row.layer_offset + expert * row.blob_bytes;
                    if (begin != cached_offset) {
                        if (!read_expert(gpack, l, static_cast<int64_t>(expert), source, err))
                            return false;
                        packed.resize(source.size());
                        if (interleaved) {
                            const auto ff = static_cast<size_t>(gpack.descriptor().moe.expert_dim_ff);
                            if (!ff || row.up_off % ff || row.down_off != 2 * row.up_off) {
                                err = "invalid GPACK gate/up packing geometry";
                                return false;
                            }
                            pack_expert_gate_up_interleaved(source.data(), packed.data(), ff, row.up_off / ff,
                                                            row.up_off / ff, row.blob_bytes - row.down_off);
                        } else
                            packed = source;
                        cached_offset = begin;
                    }
                    const auto bytes =
                        static_cast<size_t>(std::min<uint64_t>(count, row.blob_bytes - (offset - begin)));
                    std::memcpy(dst, packed.data() + (offset - begin), bytes);
                    count -= bytes;
                    offset += bytes;
                    dst += bytes;
                }
                return true;
            });
        ckpt.pack_meta.embedded_blob_present = true;
        hdr_flags |= kGExecFlagHasEmbeddedExperts;
    }

    builder.set_flags(hdr_flags);
    builder.set_identity(ckpt.identity);
    builder.set_expert_pack_meta(ckpt.pack_meta);
    builder.set_autotune_table(ckpt.autotune);
    builder.set_heterogeneous_plan(ckpt.hplan);
    builder.set_workspace_recipe(ckpt.workspace);
    builder.set_provenance(ckpt.provenance);
    builder.set_diagnostic_log(ckpt.diag_log);

    if (!builder.write_file(res.output_path, res.error, budget.max_disk_bytes)) {
        return res;
    }

    // Remove checkpoint file upon successful final installation
    {
        std::error_code ec;
        std::filesystem::remove(progress_path, ec);
    }

    res.ok = true;
    res.partial_artifact = false;
    res.expert_layout_kind = ckpt.pack_meta.layout_kind;
    res.fused_single_barrier_pipeline = ckpt.pack_meta.fused_single_barrier_pipeline;
    if (!ckpt.autotune.records.empty()) {
        res.selected_cpu_expert_impl_id = ckpt.autotune.records.front().selected_impl_id;
        res.selected_cpu_expert_impl_version = ckpt.autotune.records.front().selected_impl_version;
    }
    res.optimization_wall_ms = ckpt.provenance.optimization_wall_ms;
    res.baseline_decode_ms = ckpt.diag_log.baseline_decode_ms;
    res.optimized_decode_ms = ckpt.diag_log.optimized_decode_ms;
    res.speedup_ratio = ckpt.diag_log.speedup_ratio;
    res.break_even_tokens = ckpt.diag_log.break_even_tokens;
    {
        std::error_code ec;
        res.artifact_bytes = static_cast<uint64_t>(std::filesystem::file_size(res.output_path, ec));
    }
    return res;
}

bool regenerate_gexec_section(const std::string& gexec_path, const std::string& gpack_path,
                              GExecSectionKind section_kind, std::string& err, const GExecOptBudget& budget) {
    const auto start = std::chrono::steady_clock::now();
    auto expired = [&] {
        return budget.max_time_seconds > 0 &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >=
                   budget.max_time_seconds;
    };
    GExecReader reader;
    ScopedAdvisoryLock lock;
    if (!lock.acquire(gexec_path + ".lock", err))
        return false;
    if (!reader.open(gexec_path, err, nullptr, true))
        return false;

    artifact::GPackReader gpack;
    if (!gpack.open(gpack_path, err, /*verify_hashes=*/false))
        return false;

    const auto& cached = reader.identity();
    const auto current = build_current_gexec_identity(gpack, cached.kv_format, cached.kv_mode, cached.max_context,
                                                      cached.prefill_batch, cached.mtp_spec_tokens);
    if (!reader.validate_against(current, err))
        return false;

    uint64_t working_bytes = 2ull << 20;
    for (const auto& section : reader.sections()) {
        if (section.kind != static_cast<uint32_t>(GExecSectionKind::PackedExpertsBlob))
            working_bytes += section.byte_size;
    }
    if (budget.max_memory_bytes && working_bytes > budget.max_memory_bytes) {
        err = "section regeneration exceeds memory budget";
        return false;
    }
    GExecBuilder builder;
    if (!builder.import_from(reader, err, true))
        return false;

    const auto id = reader.identity();
    if (section_kind == GExecSectionKind::HeterogeneousPlan) {
        GExecHeterogeneousPlan hp = reader.heterogeneous_plan();
        hp.identity = id;
        hp.numa_policy = (id.hardware.numa_nodes > 1) ? "interleave" : "process";
        hp.use_thp_hugepages = id.hardware.thp_madvise_available || (gpack.total_expert_bytes() >= (1ull << 30));
        builder.set_heterogeneous_plan(hp);
    } else if (section_kind == GExecSectionKind::WorkspaceRecipe) {
        GExecWorkspaceRecipe wr = reader.workspace_recipe();
        wr.dense_arena_bytes =
            gpack.dense_compacted_pool_bytes() > 0 ? gpack.dense_compacted_pool_bytes() : gpack.dense_pool_bytes();
        wr.native_dense_arena_bytes = gpack.native_dense_arena_bytes();
        wr.capture_window_tokens.clear();
        wr.capture_commit_graph = false;
        builder.set_workspace_recipe(wr);
    } else if (section_kind == GExecSectionKind::AutotuneTable) {
        GExecExpertPackMeta pmeta = reader.expert_pack_meta();
        GExecAutotuneTable atable;
        GExecDiagnosticLog dlog;
        std::string sel_impl;
        uint32_t sel_ver = 0;
        if (!run_autotune_pass(gpack, id, GExecOptMode::Auto, pmeta, atable, dlog, sel_impl, sel_ver, err, budget,
                               expired)) {
            return false;
        }
        if (reader.has_section(GExecSectionKind::PackedExpertsBlob) &&
            pmeta.layout_kind != reader.expert_pack_meta().layout_kind) {
            err = "retuning changes the embedded layout; regenerate the complete artifact";
            return false;
        }
        builder.set_expert_pack_meta(pmeta);
        builder.set_autotune_table(atable);
        builder.set_diagnostic_log(dlog);
    } else if (section_kind == GExecSectionKind::CompilationProvenance) {
        GExecCompilationProvenance prov = reader.provenance();
        prov.created_unix_seconds = static_cast<int64_t>(std::time(nullptr));
        builder.set_provenance(prov);
    } else {
        err =
            std::string("unsupported section kind for selective regeneration: ") + section_kind_to_string(section_kind);
        return false;
    }

    if (expired()) {
        err = "section regeneration time budget expired";
        return false;
    }
    return builder.write_file(gexec_path, err, budget.max_disk_bytes);
}

bool resolve_gexec_for_gpack(const std::string& gpack_path, const std::string& explicit_gexec_path,
                             const GExecArtifactIdentity& current_identity, GExecReader& out_reader,
                             std::string& fallback_reason, GExecInvalidationReason* out_reason) {
    out_reader.close();
    if (const char* env_skip = std::getenv("GUILD_DISABLE_GEXEC"); env_skip && std::atoi(env_skip) != 0) {
        fallback_reason = "disabled via GUILD_DISABLE_GEXEC=1";
        if (out_reason)
            *out_reason = GExecInvalidationReason::None;
        return false;
    }

    std::vector<std::string> candidates;
    if (!explicit_gexec_path.empty()) {
        candidates.push_back(explicit_gexec_path);
    } else if (const char* env_path = std::getenv("GUILD_GEXEC_PATH"); env_path && env_path[0] != '\0') {
        candidates.emplace_back(env_path);
    } else {
        candidates.push_back(default_gexec_path_for_gpack(gpack_path, current_identity));
        // Also check sibling <gpack_stem>.gexec next to the .gpack file
        const std::filesystem::path gp(gpack_path);
        std::filesystem::path sibling = gp;
        sibling.replace_extension(".gexec");
        if (sibling.string() != candidates.front()) {
            candidates.push_back(sibling.string());
        }
        // Also scan cache_dir for any .gexec file in case an outdated cache entry is present so we can report the exact
        // invalidation reason
        const std::string cdir = resolve_cache_dir(gpack_path, "");
        std::error_code ec;
        if (std::filesystem::exists(cdir, ec) && std::filesystem::is_directory(cdir, ec)) {
            for (const auto& entry : std::filesystem::directory_iterator(cdir, ec)) {
                if (entry.is_regular_file(ec) && entry.path().extension() == ".gexec") {
                    const std::string p = entry.path().string();
                    if (std::find(candidates.begin(), candidates.end(), p) == candidates.end()) {
                        candidates.push_back(p);
                    }
                }
            }
        }
    }

    std::string first_mismatch_reason;
    GExecInvalidationReason first_mismatch_code = GExecInvalidationReason::None;

    for (const auto& cand : candidates) {
        std::error_code ec;
        if (!std::filesystem::exists(cand, ec))
            continue;

        GExecReader reader;
        std::string open_err;
        GExecInvalidationReason rcode = GExecInvalidationReason::None;
        if (!reader.open(cand, open_err, &rcode, true)) {
            if (first_mismatch_reason.empty()) {
                first_mismatch_reason = "rejected '" + cand + "': " + open_err;
                first_mismatch_code = rcode;
            }
            continue;
        }
        std::string val_err;
        if (!reader.validate_against(current_identity, val_err, &rcode)) {
            if (first_mismatch_reason.empty()) {
                first_mismatch_reason = "invalidated '" + cand + "': " + val_err;
                first_mismatch_code = rcode;
            }
            continue;
        }
        artifact::GPackReader source;
        bool valid = !reader.is_partial() && source.open(gpack_path, val_err, false);
        const auto& meta = reader.expert_pack_meta();
        valid = valid && reader.has_section(GExecSectionKind::ExpertPackMeta) &&
                meta.n_layers == static_cast<int64_t>(source.expert_layers().size()) &&
                meta.n_expert == source.expert_count_per_layer() &&
                meta.total_expert_bytes == source.total_expert_bytes() &&
                meta.layer_specs.size() == source.expert_layers().size();
        for (size_t i = 0; valid && i < meta.layer_specs.size(); ++i) {
            const auto& spec = meta.layer_specs[i];
            const auto& layer = source.expert_layers()[i];
            valid = spec.layout_kind == meta.layout_kind && spec.gu_type == layer.gu_type &&
                    spec.d_type == layer.d_type && spec.n_embd == source.descriptor().attn.n_embd &&
                    spec.n_ff == source.descriptor().moe.expert_dim_ff;
        }
        std::string binding;
        uint32_t binding_version = 0;
        for (const auto& rec : reader.autotune_table().records) {
            if (rec.op_kind != GUILD_OP_BATCHED_EXPERT_EXEC)
                continue;
            if (binding.empty()) {
                binding = rec.selected_impl_id;
                binding_version = rec.selected_impl_version;
            }
            valid = valid && rec.selected_impl_id == binding && rec.selected_impl_version == binding_version;
        }
        const bool interleaved = meta.layout_kind == ExpertPackLayoutKind::GateUpRowInterleaved;
        valid = valid && (meta.layout_kind == ExpertPackLayoutKind::PortableContiguous || interleaved) &&
                interleaved == kernels::cpu::is_gexec_interleaved_impl(binding.c_str());
        if (!valid) {
            if (first_mismatch_reason.empty()) {
                first_mismatch_reason = reader.is_partial() ? "GEXEC artifact is incomplete"
                                                            : "GEXEC expert layout or bindings do not match GPACK";
                first_mismatch_code = GExecInvalidationReason::ExecutionPlanMismatch;
            }
            continue;
        }
        out_reader = std::move(reader);
        fallback_reason.clear();
        if (out_reason)
            *out_reason = GExecInvalidationReason::None;
        return true;
    }

    if (!first_mismatch_reason.empty()) {
        fallback_reason = first_mismatch_reason;
        if (out_reason)
            *out_reason = first_mismatch_code;
    } else {
        fallback_reason = "no GEXEC artifact found for GPACK";
        if (out_reason)
            *out_reason = GExecInvalidationReason::None;
    }
    return false;
}

GExecCacheCleanResult clean_gexec_cache(const std::string& cache_dir, bool remove_all,
                                        const GExecArtifactIdentity* keep_identity) {
    GExecCacheCleanResult res;
    std::error_code ec;
    if (!std::filesystem::exists(cache_dir, ec)) {
        res.ok = true;
        return res;
    }
    if (!std::filesystem::is_directory(cache_dir, ec)) {
        res.error = "cache path is not a directory: " + cache_dir;
        return res;
    }

    for (const auto& entry : std::filesystem::directory_iterator(cache_dir, ec)) {
        if (entry.is_symlink(ec) || !entry.is_regular_file(ec))
            continue;
        const std::filesystem::path p = entry.path();
        const std::string fname = p.filename().string();
        const size_t marker = fname.rfind(".gexec");
        if (marker == std::string::npos)
            continue;
        const std::string suffix = fname.substr(marker + 6);
        if (!suffix.empty() && suffix != ".progress.json" && suffix != ".progress.json.tmp" &&
            suffix.rfind(".tmp.", 0) != 0)
            continue;
        const auto owner = p.parent_path() / fname.substr(0, marker + 6);
        ScopedAdvisoryLock lock;
        std::string lock_error;
        if (!lock.acquire(owner.string() + ".lock", lock_error)) {
            if (suffix.empty())
                ++res.kept_files;
            continue;
        }
        const uint64_t fsize = static_cast<uint64_t>(entry.file_size(ec));

        bool should_remove = false;
        if (suffix.rfind(".tmp.", 0) == 0 || suffix == ".progress.json.tmp") {
            should_remove = true;
        } else if (suffix == ".progress.json") {
            should_remove = remove_all || !std::filesystem::exists(owner, ec);
        } else if (p.extension() == ".gexec") {
            if (remove_all) {
                should_remove = true;
            } else {
                GExecReader reader;
                std::string err;
                if (!reader.open(p.string(), err)) {
                    should_remove = true; // corrupted or unreadable
                } else if (reader.is_partial()) {
                    should_remove = true; // incomplete partial artifact
                } else if (keep_identity != nullptr &&
                           reader.identity().gpack_manifest_sha256 == keep_identity->gpack_manifest_sha256) {
                    std::string val_err;
                    if (!reader.validate_against(*keep_identity, val_err)) {
                        should_remove = true; // obsolete/invalidated for current identity
                    }
                }
            }
        }

        if (should_remove) {
            if (std::filesystem::remove(p, ec)) {
                ++res.removed_files;
                res.reclaimed_bytes += fsize;
                res.removed_paths.push_back(p.string());
            }
        } else if (p.extension() == ".gexec") {
            ++res.kept_files;
        }
    }
    res.ok = true;
    return res;
}

} // namespace guild::gexec
