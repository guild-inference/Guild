#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "model_impl.hpp"
#include "guild/abi/gkernel_package.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include "guild/model/archetype.hpp"
#include "guild/kernels/iq_kernels.hpp"
#include "guild/kernels/verify_kernels.hpp"
#include "guild/platform/device_policy.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <vector>

// GuildModel::load is the per-model startup sequence.  Each model instance owns its own ModelExecutionState and
// enters a ScopedModelExecution guard here so configuration settings never leak across models in the same process.
// A failure returns nullptr with `error_msg` set and unwinds the model's execution scope cleanly.
//
// The stages are also what the StartupProfile in `sp` reports, so the timing JSON of a load reads in this order:
//
//  1. device runtime policy, then the enabled `.gkmetadata` kernel packages.  Both must exist before any code
//     from a package is called.
//  2. the container: an explicit `.gpack` path (which must open, or the load fails) or the GGUF shards, with an
//     optional `model.gpack` sidecar that is a fallback, not a hard requirement.
//  3. the runner resolved from the container metadata, then the model geometry and its capability checks.
//  4. process-wide kernel settings (RoPE scaling, KV precision, shared-layer mode, KV-Resident size).  The kernels
//     module holds them as globals, so they have to be set before any weight is touched.
//  5. long-lived resources: the CPU expert pool (installs worker affinity before the workers touch expert pages),
//     the main CUDA stream, and the kernel lease that keeps the resolved providers loaded.
//  6. `.gexec` resolution.  It never fails the load: an unusable artifact falls back to the GPACK layout with the
//     reason recorded in the startup profile.
//  7. weights: dense weights and the output head, the expert tiers, the calibrated performance profile, the GPU
//     expert cache, and finally the RAM-resident complement (pinned or NUMA-interleaved).
//  8. state shared by every session: device KV staging, PLE tables, MTP drafter, verifier, prefill engine, and the
//     tokenizer.  After this point the model is ready and sessions can be created.

namespace guild::runtime {

GuildModel::GuildModel() : impl_(std::make_shared<Impl>()) {
    impl_->tokenizer = tokenizer_;
    impl_->ready = ready_;
}
GuildModel::~GuildModel() {
    if (ready_)
        ready_->store(false);
}

std::unique_ptr<GuildModel> GuildModel::load(const ModelPaths& paths, const model::ModelDescriptor& requested_desc,
                                             const memory::ExecutionPlan& plan, std::string& error_msg) try {
    const auto load_t0 = Clock::now();
    auto ms_since = [](Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };
    StartupProfile sp;
    // Device runtime policy first (CUDA scheduling flags, driver-level knobs): every later allocation and stream
    // creation happens under it, and it cannot be changed while allocations are live.
    const auto cuda_init_t0 = Clock::now();
    platform::configure_device_runtime_policy();
    sp.cuda_init_ms += ms_since(cuda_init_t0);
    error_msg.clear();
    if (paths.primary_model_path.empty()) {
        error_msg = "model load: primary model path is required";
        return nullptr;
    }
    {
        const auto pkg_t0 = Clock::now();
        std::string pkg_err;
        if (abi::GKernelPackageManager::instance().load_enabled_packages(pkg_err) != GUILD_STATUS_OK) {
            error_msg = "model load: enabled kernel package failed: " + pkg_err;
            return nullptr;
        }
        sp.package_load_ms += ms_since(pkg_t0);
    }

    // Container selection.  An explicitly named `.gpack` must open - silently running the GGUF path next to a
    // container the user asked for would hide a corrupt download.  A `model.gpack` found next to the pack
    // directory is a convenience: if it is damaged the load continues from the GGUF shards.
    auto has_gpack_ext = [](const std::string& p) {
        return p.size() >= 6 && p.compare(p.size() - 6, 6, ".gpack") == 0;
    };
    const bool explicit_gpack =
        has_gpack_ext(paths.primary_model_path) || artifact::GPackReader::is_gpack_file(paths.primary_model_path);
    std::unique_ptr<artifact::GPackReader> gpack_reader;
    std::string gpack_candidate_path;
    if (explicit_gpack) {
        gpack_candidate_path = paths.primary_model_path;
    } else if (!paths.pack_dir.empty()) {
        const auto sibling = std::filesystem::path(paths.pack_dir) / "model.gpack";
        if (std::filesystem::is_regular_file(sibling)) {
            gpack_candidate_path = sibling.string();
        }
    }
    if (!gpack_candidate_path.empty()) {
        const auto gopen_t0 = Clock::now();
        auto reader = std::make_unique<artifact::GPackReader>();
        std::string gerr;
        if (reader->open(gpack_candidate_path, gerr, false)) {
            gpack_reader = std::move(reader);
            sp.container_format = "gpack.v1";
            ++sp.container_opens;
            sp.model_open_parse_ms += ms_since(gopen_t0);
        } else if (explicit_gpack) {
            error_msg = "model load: " + gerr;
            return nullptr;
        } else {
            std::fprintf(stderr, "guild runtime: GPACK sidecar %s invalid (%s); falling back to GGUF\n",
                         gpack_candidate_path.c_str(), gerr.c_str());
        }
    }

    const bool use_gpack = (gpack_reader != nullptr);
    std::vector<std::string> shards;
    model::ModelDescriptor desc = requested_desc;
    std::shared_ptr<const model::IModelRunner> runner;

    if (use_gpack) {
        const auto run_t0 = Clock::now();
        desc = gpack_reader->descriptor();
        if (requested_desc.rope.type != "none" || requested_desc.rope.factor != 1.0 ||
            requested_desc.rope.orig_ctx > 0.0) {
            desc.rope = requested_desc.rope;
        }
        const std::string arch_id = !gpack_reader->execution_metadata().arch_id.empty()
                                        ? gpack_reader->execution_metadata().arch_id
                                        : desc.arch_name;
        if (model::ModelRunnerRegistry::instance().resolve_by_arch(arch_id, &runner, error_msg) != GUILD_STATUS_OK ||
            !runner) {
            return nullptr;
        }
        sp.runner_and_plan_ms += ms_since(run_t0);
    } else {
        if (paths.pack_dir.empty() ||
            !std::filesystem::is_regular_file(std::filesystem::path(paths.pack_dir) / "index.txt")) {
            error_msg = "model load: missing pack index.txt: " + paths.pack_dir;
            return nullptr;
        }
        // A missing shard is a load failure, not a reason to continue with a partial model.
        shards = guild::gguf_split_paths(paths.primary_model_path);
        if (shards.size() == 1)
            shards.insert(shards.end(), paths.additional_shards.begin(), paths.additional_shards.end());
        const auto parse_t0 = Clock::now();
        const guild::GgufModel artifact(shards);
        sp.container_opens += static_cast<int64_t>(shards.size());
        sp.model_open_parse_ms += ms_since(parse_t0);
        const auto run_t0 = Clock::now();
        if (model::ModelRunnerRegistry::instance().resolve_for_gguf(artifact.meta(), &runner, error_msg) !=
                GUILD_STATUS_OK ||
            !runner || !runner->validate_and_describe(artifact, desc, error_msg)) {
            return nullptr;
        }
        if (requested_desc.rope.type != "none" || requested_desc.rope.factor != 1.0 ||
            requested_desc.rope.orig_ctx > 0.0) {
            desc.rope = requested_desc.rope;
        }
        sp.runner_and_plan_ms += ms_since(run_t0);
    }
    if (plan.context_length <= 0 || plan.context_length > desc.attn.context_length || plan.mtp_spec_tokens < 0 ||
        plan.mtp_spec_tokens > kernels::kVerifyMaxT || plan.spec_max_draft < 0 ||
        plan.spec_max_draft >= kernels::kVerifyMaxT) {
        error_msg = "model load: invalid context or verification capacity";
        return nullptr;
    }
    spec::DecodeStrategyMode parsed_mode = spec::DecodeStrategyMode::Auto;
    if (!plan.decode_strategy.empty() && !spec::parse_decode_strategy_mode(plan.decode_strategy, parsed_mode)) {
        error_msg = "model load: unknown decode strategy '" + plan.decode_strategy + "'";
        return nullptr;
    }
    if (!std::isfinite(plan.pcie_expert_fraction) ||
        (plan.pcie_expert_fraction < 0 && plan.pcie_expert_fraction != -1) || plan.pcie_expert_fraction > 1 ||
        !std::isfinite(plan.spec_min_p) || plan.spec_min_p < 0 || plan.spec_min_p > 1 || plan.pool_workers < 0) {
        error_msg = "model load: invalid expert performance configuration";
        return nullptr;
    }
    auto model = std::unique_ptr<GuildModel>(new GuildModel());
    auto* impl = model->impl_.get();
    impl->exec_state = std::make_shared<core::ModelExecutionState>(core::ModelExecutionState::snapshot_current());
    core::ScopedModelExecution exec_scope(impl->exec_state);

    impl->paths = paths;
    impl->desc = desc;
    impl->plan = plan;
    impl->runner = runner;
    impl->gpack = std::move(gpack_reader);
    impl->runner_lease = model::SessionRunnerLease(&model::ModelRunnerRegistry::instance(),
                                                   runner && runner->package_id() ? runner->package_id() : "");
    impl->kernel_lease = kernels::SessionKernelLease(&kernels::KernelRegistry::instance());

    // Apply execution plan to runtime config
    impl->config.max_context = plan.context_length;
    impl->config.kv_format = memory::kv_precision_to_string(plan.kv_format);
    impl->config.kv_host_only = (plan.kv_mode == memory::KvMode::HostOnly);
    impl->config.kv_resident = plan.kv_resident;
    impl->config.shared_early = plan.shared_early;
    impl->config.decode_strategy = spec::decode_strategy_mode_to_string(parsed_mode);
    impl->config.spec_max_draft = plan.spec_max_draft;
    impl->config.spec_lookup_window = plan.spec_lookup_window > 0 ? plan.spec_lookup_window : 65536;
    impl->config.spec_cost_window = plan.spec_cost_window > 0 ? plan.spec_cost_window : 16;
    impl->config.spec_trace = plan.spec_trace;
    if (plan.spec_max_draft > 0) {
        impl->config.spec = std::clamp(1 + plan.spec_max_draft, 2, kernels::kVerifyMaxT);
    } else if (parsed_mode == spec::DecodeStrategyMode::Lookup || parsed_mode == spec::DecodeStrategyMode::Retrieval ||
               parsed_mode == spec::DecodeStrategyMode::Adaptive) {
        const int base_draft = (plan.mtp_spec_tokens >= 2) ? (plan.mtp_spec_tokens - 1) : 4;
        impl->config.spec = std::clamp(1 + base_draft, 2, kernels::kVerifyMaxT);
    } else {
        impl->config.spec = std::max(2, plan.mtp_spec_tokens);
    }
    impl->config.spec_min_p = plan.spec_min_p;
    impl->config.pool_workers = plan.pool_workers;
    impl->config.expert_cache_slots = plan.routed_experts_in_gpu;
    impl->config.prefill_chunk = plan.prefill_chunk > 0 ? plan.prefill_chunk : 512;
    impl->config.vram_reserve_mib = 512;
    impl->config.pcie_expert_fraction = plan.pcie_expert_fraction;

    model->desc_ = desc;
    model->plan_ = plan;
    model->paths_ = paths;
    model->config_ = impl->config;

    // Geometry & Runner Capability Validation
    const auto geom_t0 = Clock::now();
    if (!runner->configure_geometry(desc, impl->g, impl->K, error_msg)) {
        return nullptr;
    }
    const auto host_caps = kernels::query_host_device_capabilities();
    if (model::ModelRunnerRegistry::instance().validate_runner_requirements(
            *runner, desc, impl->g, plan, kernels::KernelRegistry::instance(), host_caps, impl->runner_exec,
            error_msg) != GUILD_STATUS_OK) {
        return nullptr;
    }
    sp.runner_and_plan_ms += ms_since(geom_t0);

    // Model-scoped execution settings, written into `impl->exec_state` via `exec_scope` above.  They are configured
    // here, before any weight is touched, because the kernels consult them while preparing quantized layouts and KV
    // buffers; `impl->ss.exec_state` retains the shared state for session execution and teardown.
    //
    // RoPE scaling
    kernels::RopeScaling rope_cfg;
    if (desc.rope.freq_base > 0)
        rope_cfg.freq_base = desc.rope.freq_base;
    if (desc.rope.factor > 0)
        rope_cfg.factor = desc.rope.factor;
    if (desc.rope.orig_ctx > 0)
        rope_cfg.orig_ctx = desc.rope.orig_ctx;
    if (desc.rope.type == "linear")
        rope_cfg.type = kernels::RopeScalingType::Linear;
    else if (desc.rope.type == "yarn")
        rope_cfg.type = kernels::RopeScalingType::YaRN;
    else
        rope_cfg.type = kernels::RopeScalingType::None;
    kernels::rope_scaling_set(rope_cfg);

    // KV configuration
    core::qsa_set_kv_int8(plan.kv_format == memory::KvPrecision::INT8);
    core::qsa_set_kv_q4(plan.kv_format == memory::KvPrecision::Q4_0);
    core::qsa_set_kv_hybrid(plan.kv_format == memory::KvPrecision::K8V4);
    if (plan.kv_mode == memory::KvMode::Streaming ||
        (plan.kv_format == memory::KvPrecision::K8V4 && impl->config.kv_host_only)) {
        error_msg = "model load: requested KV mode is not supported by this runtime configuration";
        return nullptr;
    }
    core::qsa_set_kv_host_only(impl->config.kv_host_only);
    core::qsa_set_kv_resident(impl->config.kv_resident);
    core::layer_set_shared_early(impl->config.shared_early);

    // Long-lived resources, in dependency order: the CPU expert pool first (its workers install the thread
    // affinity that decides where the expert pages they later touch physically land), then the main CUDA stream,
    // then the kernel lease.  The lease holds the resolved providers open for the life of the model, so a package
    // cannot be disabled or swapped while a session is executing against the artifacts it was compiled for.
    //
    // CPU expert pool
    const auto kprov_t0 = Clock::now();
    int n_workers = impl->config.pool_workers;
    if (const char* ew = std::getenv("GUILD_POOL_WORKERS")) {
        try {
            n_workers = std::stoi(ew);
        } catch (...) {}
    }
    impl->pool = std::make_unique<kernels::cpu::ExpertPool>(n_workers, true, true, kernels::cpu::PoolAffinity::All);
    impl->config.pool_workers = impl->pool->workers();
    sp.kernel_provider_resolve_ms += ms_since(kprov_t0);

    // CUDA stream
    const auto stream_t0 = Clock::now();
    if (cudaStreamCreateWithFlags(&impl->main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        error_msg = "failed to create CUDA main stream";
        return nullptr;
    }
    sp.cuda_init_ms += ms_since(stream_t0);
    const auto ksel_t0 = Clock::now();
    impl->kernel_lease = kernels::SessionKernelLease(&kernels::KernelRegistry::instance(),
                                                     impl->runner_exec.resolved_package_ids, impl->main_stream);

    // GEXEC ahead-of-time optimization resolution & validation.  The identity is rebuilt from the live container,
    // the requested KV/prefill/spec settings and this host; an artifact that does not match is *not* used (the
    // reason lands in `gexec_fallback_reason` and, for an explicitly requested artifact, on stderr).  With an
    // artifact the kernel selection below comes from what it recorded at tune time, which is what makes an
    // optimized run reproducible without re-tuning.
    if (use_gpack) {
        const auto cur_gexec_id = gexec::build_current_gexec_identity(
            *impl->gpack, impl->config.kv_format, impl->config.kv_host_only ? "host_only" : "resident",
            impl->config.max_context, plan.prefill_batch > 0 ? plan.prefill_batch : impl->config.prefill_chunk,
            plan.mtp_spec_tokens);
        auto greader = std::make_unique<gexec::GExecReader>();
        std::string gexec_fallback;
        gexec::GExecInvalidationReason gexec_reason = gexec::GExecInvalidationReason::None;
        if (gexec::resolve_gexec_for_gpack(gpack_candidate_path, paths.gexec_path, cur_gexec_id, *greader,
                                           gexec_fallback, &gexec_reason) &&
            !greader->is_partial()) {
            impl->gexec = std::move(greader);
            sp.container_format = "gpack.v1+gexec.v1";
            sp.gexec_active = true;
            sp.gexec_path = impl->gexec->path();
            sp.gexec_cache_key = impl->gexec->identity().compute_cache_key();
        } else {
            sp.gexec_active = false;
            sp.gexec_fallback_reason = gexec_fallback;
            if (!paths.gexec_path.empty() || gexec_reason != gexec::GExecInvalidationReason::None) {
                std::fprintf(stderr, "guild runtime: GEXEC fallback to GPACK (%s)\n", gexec_fallback.c_str());
            }
        }
    }

    {
        kernels::KernelSelectionQuery query;
        query.op_kind = GUILD_OP_BATCHED_EXPERT_EXEC;
        query.allowed_backends = GUILD_BACKEND_CPU;
        if (impl->gexec && impl->gexec->has_section(gexec::GExecSectionKind::AutotuneTable)) {
            const auto& at = impl->gexec->autotune_table();
            for (const auto& rec : at.records) {
                if (rec.op_kind == GUILD_OP_BATCHED_EXPERT_EXEC && !rec.selected_impl_id.empty()) {
                    query.frozen_impl_id = rec.selected_impl_id;
                    query.frozen_impl_version = rec.selected_impl_version;
                    break;
                }
            }
            // GPU decode kernels need contiguous gate/up weights. A profile alone
            // does not enable a GPU cache; its requested slot count does.
            const char* zcopy_env = std::getenv("GUILD_EXPERT_MMAP");
            const bool want_zero_copy = (zcopy_env != nullptr && std::atoi(zcopy_env) != 0) &&
                                        impl->gpack->has_embedded_experts() &&
                                        !impl->gexec->has_section(gexec::GExecSectionKind::PackedExpertsBlob);
            const char* pin_env = std::getenv("GUILD_PIN_EXPERTS");
            const bool force_pin = (pin_env != nullptr && std::atoi(pin_env) != 0);
            const uint64_t est_ram_budget =
                plan.expert_ram_budget_bytes ? std::min(plan.expert_ram_budget_bytes, impl->gpack->total_expert_bytes())
                : plan.routed_expert_total_bytes ? plan.routed_experts_ram_bytes
                                                 : impl->gpack->total_expert_bytes();
            const bool auto_pcie_pin = (plan.pcie_expert_fraction < 0.0) && (est_ram_budget > 0) &&
                                       (est_ram_budget <= (24ull << 30)) &&
                                       (impl->gexec->identity().hardware.numa_nodes <= 1);
            const bool gpu_decode_experts =
                (plan.routed_experts_in_gpu != 0) || (plan.pcie_expert_fraction > 0.0) || force_pin || auto_pcie_pin;
            const bool fused_pf_env =
                (std::getenv("GUILD_PF_FUSED") != nullptr && std::atoi(std::getenv("GUILD_PF_FUSED")) != 0);
            if ((want_zero_copy || gpu_decode_experts || plan.expert_cache_auto || fused_pf_env) &&
                kernels::cpu::is_gexec_interleaved_impl(query.frozen_impl_id.c_str())) {
                if (want_zero_copy || gpu_decode_experts || plan.expert_cache_auto) {
                    sp.gexec_fallback_reason =
                        "interleaved expert plan is incompatible with the requested mmap/GPU expert path";
                    std::fprintf(stderr, "guild runtime: GEXEC fallback to GPACK (%s)\n",
                                 sp.gexec_fallback_reason.c_str());
                    impl->gexec.reset();
                    sp.gexec_active = false;
                    sp.container_format = "gpack.v1";
                    query.frozen_impl_id.clear();
                    query.frozen_impl_version = 0;
                } else if (!impl->gexec->has_section(gexec::GExecSectionKind::PackedExpertsBlob)) {
                    // Preserve GEXEC single-barrier fused pipeline with PortableContiguous layout when fused prefill is
                    // active
                    if (query.frozen_impl_id == kernels::cpu::kGExecInterleavedAvx512ImplId)
                        query.frozen_impl_id = kernels::cpu::kGExecFusedAvx512ImplId;
                    else if (query.frozen_impl_id == kernels::cpu::kGExecInterleavedAvx2ImplId)
                        query.frozen_impl_id = kernels::cpu::kGExecFusedAvx2ImplId;
                }
            }
        }
        kernels::KernelBinding selected;
        if (kernels::KernelRegistry::instance().select(query, host_caps, &selected, error_msg) != GUILD_STATUS_OK) {
            if (!query.frozen_impl_id.empty()) {
                std::fprintf(
                    stderr,
                    "guild runtime: GEXEC frozen kernel '%s' unavailable (%s); falling back to default selection\n",
                    query.frozen_impl_id.c_str(), error_msg.c_str());
                sp.gexec_fallback_reason = error_msg;
                impl->gexec.reset();
                sp.gexec_active = false;
                sp.container_format = "gpack.v1";
                error_msg.clear();
                query.frozen_impl_id.clear();
                query.frozen_impl_version = 0;
                if (kernels::KernelRegistry::instance().select(query, host_caps, &selected, error_msg) !=
                    GUILD_STATUS_OK)
                    return nullptr;
            } else {
                return nullptr;
            }
        }
        if (selected && selected->impl_id) {
            sp.gexec_cpu_expert_impl = selected->impl_id;
        }
        if (impl->gexec) {
            impl->pool->set_fused_pipeline(impl->gexec->expert_pack_meta().fused_single_barrier_pipeline);
        }
        if (selected->next_ext) {
            impl->cpu_binding = selected;
            guild_cpu_expert_ext_v1_t table{};
            std::memcpy(&table, selected->next_ext, sizeof(table));
            if (table.struct_size < sizeof(table) || table.tag != GUILD_CPU_EXPERT_TAG_V1 || table.version != 1) {
                error_msg = "model load: invalid CPU expert contract";
                return nullptr;
            }
            impl->kernel_lease.add_package(selected->package_id);
            impl->pool->bind_native_kernels(table);
            impl->native_cpu_package = selected->package_id;
            if (table.struct_size >= offsetof(guild_cpu_backend_ext_v2_t, swiglu)) {
                guild_cpu_backend_ext_v2_t complete{};
                std::memcpy(&complete, selected->next_ext, std::min<size_t>(table.struct_size, sizeof(complete)));
                if (complete.backend_version != GUILD_CPU_BACKEND_VERSION_V2) {
                    error_msg = "invalid CPU backend contract";
                    return nullptr;
                }
                impl->cpu_backend = complete;
                impl->pool->bind_backend_kernels(complete);
            }
        }
#if defined(GUILD_NO_BUILTIN_NATIVE_CPU)
        if (impl->native_cpu_package.empty()) {
            error_msg = "model load: native CPU fallback disabled; install a compatible CPU .gkernel";
            return nullptr;
        }
#endif
#if defined(GUILD_NO_BUILTIN_FULL_CPU)
        if (!impl->cpu_backend || !impl->cpu_backend->swiglu) {
            error_msg = "model load: all bundled CPU arithmetic disabled; a complete CPU backend contract is required";
            return nullptr;
        }
#endif
    }
    std::optional<kernels::ScopedCudaDense> dense_scope;
    std::optional<kernels::ScopedCudaAttn> attn_scope;
    std::optional<kernels::ScopedCudaMoE> moe_scope;
#if defined(GUILD_ENABLE_CUDA) && !defined(GUILD_HIP_GFX906)
    {
        kernels::ResolvedCudaDense dense{};
        if (kernels::KernelRegistry::instance().resolve_cuda_dense(host_caps, dense, error_msg)) {
            impl->kernel_lease.add_package(dense.package_id);
            impl->cuda_dense = dense;
            dense_scope.emplace(*impl->cuda_dense);
        }
#if !GUILD_BUNDLED_CUDA_DENSE
        if (!impl->cuda_dense) {
            error_msg = "model load: bundled CUDA dense arithmetic disabled; install a compatible CDD1 CUDA .gkernel";
            return nullptr;
        }
#endif
        kernels::ResolvedCudaMoE moe{};
        if (kernels::KernelRegistry::instance().resolve_cuda_moe(host_caps, moe, error_msg)) {
            impl->kernel_lease.add_package(moe.package_id);
            impl->cuda_moe = moe;
            moe_scope.emplace(*impl->cuda_moe);
        }
#if !GUILD_BUNDLED_CUDA_MOE
        if (!impl->cuda_moe) {
            error_msg = "model load: bundled CUDA MoE arithmetic disabled; install a compatible CDM1 CUDA .gkernel";
            return nullptr;
        }
#endif
        kernels::ResolvedCudaAttn attn{};
        if (kernels::KernelRegistry::instance().resolve_cuda_attn(host_caps, attn, error_msg)) {
            impl->kernel_lease.add_package(attn.package_id);
            impl->cuda_attn = attn;
            attn_scope.emplace(*impl->cuda_attn);
        }
#if !GUILD_BUNDLED_CUDA_ATTN
        if (!impl->cuda_attn) {
            error_msg =
                "model load: bundled CUDA attention arithmetic disabled; install a compatible CDA1 CUDA .gkernel";
            return nullptr;
        }
#endif
    }
#endif
    sp.kernel_provider_resolve_ms += ms_since(ksel_t0);

    // Doorbell
    const auto db_t0 = Clock::now();
    if (core::doorbell_init(impl->g, impl->K, impl->db) == 0) {
        error_msg = "doorbell_init failed";
        return nullptr;
    }
    sp.cuda_init_ms += ms_since(db_t0);

    // Expert layout from GPACK or pack directory
    const auto qlay_t0 = Clock::now();
    if (use_gpack) {
        if (!guild::kernels::cpu::expert_layout_load_gpack(*impl->gpack, impl->g.n_layers, impl->g.n_expert, error_msg,
                                                           impl->g.n_embd, impl->g.n_ff,
                                                           impl->cpu_backend ? &*impl->cpu_backend : nullptr)) {
            return nullptr;
        }
    } else {
        if (!guild::kernels::cpu::expert_layout_load(paths.pack_dir, impl->g.n_layers, impl->g.n_expert, error_msg,
                                                     impl->g.n_embd, impl->g.n_ff,
                                                     impl->cpu_backend ? &*impl->cpu_backend : nullptr)) {
            return nullptr;
        }
    }
    const bool use_gexec_interleaved = impl->gexec &&
                                       kernels::cpu::is_gexec_interleaved_impl(sp.gexec_cpu_expert_impl.c_str()) &&
                                       guild::kernels::cpu::expert_layout().native;
    guild::kernels::cpu::expert_layout_set_gate_up_interleaved(use_gexec_interleaved);
    sp.gexec_expert_layout = use_gexec_interleaved ? "GateUpRowInterleaved" : "PortableContiguous";
    sp.quant_layout_prep_ms += ms_since(qlay_t0);
    const bool native_pack = guild::kernels::cpu::expert_layout().native;
    const bool has_native_sources = use_gpack || !shards.empty();
    if (!native_pack && (impl->g.n_embd != GUILD_CPU_PLANAR_EMBED_V1 || impl->g.n_ff != GUILD_CPU_PLANAR_FF_V1)) {
        error_msg = "model load: canonical planar Q2 pack geometry does not match its fixed public storage contract";
        return nullptr;
    }
    const auto& layout = kernels::cpu::expert_layout();
    const uint64_t ram_budget = plan.expert_ram_budget_bytes     ? std::min(plan.expert_ram_budget_bytes, layout.total)
                                : plan.routed_expert_total_bytes ? plan.routed_experts_ram_bytes
                                                                 : layout.total;
    impl->config.resident_budget_gib = (ram_budget >> 30) + ((ram_budget & ((1ull << 30) - 1)) != 0);
    model->config_ = impl->config;
    if ((!native_pack && !kernels::cpu::cpu_avx512_ok() && !impl->cpu_backend) || impl->g.n_ff > kernels::cpu::FF ||
        impl->g.n_embd > kernels::cpu::H) {
        error_msg = "model load: expert geometry or CPU ISA exceeds this pool's supported capacities";
        return nullptr;
    }

    // Kernel settings (fast paths)
    core::layer_set_fast_attn(true);
    core::layer_set_publish_kernel(true);
    core::layer_set_fused_gdn(true);
    core::layer_set_fast_select(true);
    core::layer_set_fused_gr(true);
    kernels::gr_set_fp32_activations(false);
    kernels::gr_set_native_mmvf(true);
    core::layer_set_native_bf16(native_pack || has_native_sources);
    kernels::ple_set_native_bf16(native_pack || has_native_sources);
    kernels::shared_expert_set_native_bf16(native_pack || has_native_sources);
    kernels::native_moe_combine_set_enabled(native_pack || has_native_sources);
    kernels::native_gdn_set_enabled(native_pack || has_native_sources);
    kernels::native_router_set_enabled(native_pack || has_native_sources);
    kernels::native_qsa_set_enabled(native_pack || has_native_sources);
    kernels::native_qsa_indexer_set_enabled(native_pack || has_native_sources);
    kernels::native_rope_set_enabled(native_pack || has_native_sources);
    kernels::ple_set_native_postops(native_pack || has_native_sources);

    // Native token embeddings
    if (use_gpack) {
        const auto emb_t0 = Clock::now();
        if (!impl->native_embed.load_gpack(*impl->gpack, impl->g.n_embd, desc.attn.vocab_size, error_msg)) {
            return nullptr;
        }
        core::set_native_embed(&impl->native_embed);
        sp.embed_load_ms += ms_since(emb_t0);
    } else if (!shards.empty()) {
        const auto emb_t0 = Clock::now();
        if (!impl->native_embed.load(shards, impl->g.n_embd, desc.attn.vocab_size, error_msg)) {
            return nullptr;
        }
        core::set_native_embed(&impl->native_embed);
        sp.embed_load_ms += ms_since(emb_t0);
        sp.container_opens += static_cast<int64_t>(shards.size());
    }

    // Load weight table from GPACK or pack directory
    const auto dpack_t0 = Clock::now();
    std::set<std::string> skip;
    uint64_t pool_bytes = 0;
    if (use_gpack) {
        skip = impl->gpack->precomputed_skip_set();
        if (!core::WeightTable::pool_bytes_gpack(*impl->gpack, pool_bytes, error_msg, skip.empty() ? nullptr : &skip)) {
            return nullptr;
        }
        if (cudaMalloc(&impl->arena, pool_bytes) != cudaSuccess) {
            error_msg = "failed to allocate VRAM weight arena (" + std::to_string(pool_bytes >> 20) + " MiB)";
            return nullptr;
        }
        ++sp.dense_device_allocs;
        sp.dense_vram_bytes += pool_bytes;
        if (!impl->wt.load_gpack(*impl->gpack, impl->arena, pool_bytes, error_msg, skip.empty() ? nullptr : &skip)) {
            return nullptr;
        }
    } else {
        if (!shards.empty()) {
            if (!guild::core::NativeDense::served_names(shards, true, skip, error_msg)) {
                return nullptr;
            }
            sp.container_opens += static_cast<int64_t>(shards.size());
            if (native_pack && !guild::core::NativeDense::keep_unquantized_ple_key(paths.pack_dir, skip, error_msg)) {
                return nullptr;
            }
            if (native_pack)
                ++sp.index_parses;
        }
        if (!runner->configure_weight_skip_set(native_pack, !shards.empty(), paths.pack_dir, skip, error_msg)) {
            return nullptr;
        }
        ++sp.index_parses;

        if (!core::WeightTable::pool_bytes(paths.pack_dir, pool_bytes, error_msg, skip.empty() ? nullptr : &skip)) {
            return nullptr;
        }
        ++sp.index_parses;

        if (cudaMalloc(&impl->arena, pool_bytes) != cudaSuccess) {
            error_msg = "failed to allocate VRAM weight arena (" + std::to_string(pool_bytes >> 20) + " MiB)";
            return nullptr;
        }
        ++sp.dense_device_allocs;
        sp.dense_vram_bytes += pool_bytes;

        if (!impl->wt.load(paths.pack_dir, impl->arena, pool_bytes, error_msg, skip.empty() ? nullptr : &skip)) {
            return nullptr;
        }
        ++sp.index_parses;
    }
    sp.dense_index_and_pack_ms += ms_since(dpack_t0);

    // Load native dense projections if available
    const auto ndense_t0 = Clock::now();
    if (use_gpack) {
        if (impl->gpack->has_native_dense()) {
            if (!impl->native_dense.load_gpack(*impl->gpack, impl->wt, error_msg)) {
                return nullptr;
            }
            ++sp.dense_device_allocs; // single consolidated arena allocation!
            sp.dense_vram_bytes += impl->native_dense.weight_bytes();
        }
    } else if (!shards.empty()) {
        if (!impl->native_dense.load(shards, impl->wt, error_msg, true)) {
            return nullptr;
        }
        sp.container_opens += static_cast<int64_t>(shards.size());
        sp.dense_device_allocs += static_cast<int64_t>(skip.size());
        sp.dense_vram_bytes += impl->native_dense.weight_bytes();
    }
    if (!core::check_all(impl->wt, impl->g, error_msg))
        return nullptr;
    sp.native_dense_load_ms += ms_since(ndense_t0);

    // Load native head if available
    const guild::core::WeightRef* wo = impl->wt.find("output.weight");
    int64_t n_vocab = wo ? wo->ne1 : desc.attn.vocab_size;
    if (use_gpack) {
        const auto nhead_t0 = Clock::now();
        if (!impl->native_head.load_gpack(*impl->gpack, impl->g.n_embd, n_vocab, error_msg)) {
            return nullptr;
        }
        sp.native_head_load_ms += ms_since(nhead_t0);
        ++sp.dense_device_allocs;
        sp.dense_vram_bytes += impl->native_head.weight_bytes();
    } else if (!shards.empty()) {
        const auto nhead_t0 = Clock::now();
        if (!impl->native_head.load(shards, impl->g.n_embd, n_vocab, error_msg)) {
            return nullptr;
        }
        sp.native_head_load_ms += ms_since(nhead_t0);
        sp.container_opens += static_cast<int64_t>(shards.size());
        ++sp.dense_device_allocs;
        sp.dense_vram_bytes += impl->native_head.weight_bytes();
    }

    // Open expert source (FileExpertSource over GEXEC PackedExpertsBlob, GPACK ExpertBlobs, or pack directory / GGUF
    // shards)
    // Expert tiers, highest priority first: the GEXEC packed blob (already interleaved and fused, so it is used
    // as-is), a GPACK with the expert bytes embedded, a GPACK whose experts live in the pack directory, and
    // finally the raw GGUF shards.  Every branch leaves `src` with the same interface, so the calibration, cache
    // and pinning stages below do not care which one supplied the bytes; the interleave flag is the only
    // difference that survives.
    const auto esrc_t0 = Clock::now();
    if (impl->gexec && impl->gexec->has_section(gexec::GExecSectionKind::PackedExpertsBlob)) {
        uint64_t exp_bytes = 0;
        const uint8_t* exp_ptr = impl->gexec->packed_experts_blob(&exp_bytes);
        if (!impl->src.open_gpack(exp_ptr, exp_bytes, impl->g.n_layers, impl->g.n_expert, error_msg)) {
            return nullptr;
        }
        impl->src.set_gate_up_interleaved(false); // Embedded .gexec blob is already interleaved
    } else if (use_gpack && impl->gpack->has_embedded_experts()) {
        uint64_t exp_bytes = 0;
        const uint8_t* exp_ptr = impl->gpack->expert_blobs_data(&exp_bytes);
        if (!impl->src.open_gpack(exp_ptr, exp_bytes, impl->g.n_layers, impl->g.n_expert, error_msg)) {
            return nullptr;
        }
        impl->src.set_gate_up_interleaved(use_gexec_interleaved);
    } else if (use_gpack && impl->gpack->has_external_experts()) {
        impl->src.set_gguf(impl->gpack->external_expert_primary_gguf());
        if (!impl->src.open(paths.pack_dir, impl->g.n_layers, impl->g.n_expert, error_msg)) {
            return nullptr;
        }
        impl->src.set_gate_up_interleaved(use_gexec_interleaved);
        ++sp.container_opens;
    } else {
        if (!shards.empty()) {
            impl->src.set_gguf(shards.front());
        }
        if (!impl->src.open(paths.pack_dir, impl->g.n_layers, impl->g.n_expert, error_msg)) {
            return nullptr;
        }
        impl->src.set_gate_up_interleaved(use_gexec_interleaved);
        if (!shards.empty())
            sp.container_opens += static_cast<int64_t>(shards.size());
    }
    impl->srcp = &impl->src;
    sp.expert_source_open_ms += ms_since(esrc_t0);

    // Expert placement calibration: measures CPU vs GPU vs copy cost per layer, which is what decides how many
    // experts of each layer the GPU cache holds and which layers stay fully on the CPU.  When the container is a
    // GEXEC artifact whose heterogeneous plan was measured on this same fingerprint, that plan is reused instead of
    // re-measuring, so an optimized start is a load, not a benchmark run.
    const auto cal1_t0 = Clock::now();
    if (impl->gexec && impl->gexec->has_section(gexec::GExecSectionKind::HeterogeneousPlan) &&
        impl->gexec->heterogeneous_plan().measured_formats > 0 && plan.pcie_expert_fraction <= 0.0) {
        const auto& hp = impl->gexec->heterogeneous_plan();
        impl->expert_performance.gpu_layers = hp.gpu_layers;
        if (impl->expert_performance.gpu_layers.size() != static_cast<size_t>(impl->g.n_layers)) {
            impl->expert_performance.gpu_layers.assign(static_cast<size_t>(impl->g.n_layers), 1u);
        }
        impl->expert_performance.pcie_numerators = hp.pcie_numerators;
        if (impl->expert_performance.pcie_numerators.size() != static_cast<size_t>(impl->g.n_layers)) {
            impl->expert_performance.pcie_numerators.assign(static_cast<size_t>(impl->g.n_layers), 0);
        }
        impl->expert_performance.measured_formats = hp.measured_formats;
        impl->expert_performance.cpu_ms = hp.calibrated_cpu_ms;
        impl->expert_performance.gpu_ms = hp.calibrated_gpu_ms;
        impl->expert_performance.copy_ms = hp.calibrated_copy_ms;
    } else {
        if (!calibrate_experts(layout, impl->src, *impl->pool, 0, impl->main_stream, impl->expert_performance,
                               error_msg))
            return nullptr;
    }
    if (!native_pack) {
        const bool supported = impl->g.n_embd == kernels::cpu::H && impl->g.n_ff == kernels::cpu::FF;
        impl->expert_performance.gpu_layers.assign(size_t(impl->g.n_layers), supported);
    }
    sp.expert_calibration_pre_ms += ms_since(cal1_t0);

    // Load expert profile if provided
    const auto xcache_t0 = Clock::now();
    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!paths.expert_profile_path.empty()) {
        int64_t profile_slots = 0;
        if (!core::read_expert_profile(paths.expert_profile_path, impl->g.n_layers, impl->g.n_expert, profile,
                                       profile_slots, error_msg))
            return nullptr;
    }

    // Explicit cache counts are exact. Automatic caches wait for all fixed allocations.
    const bool auto_cache = plan.expert_cache_auto || impl->config.expert_cache_slots < 0;
    int64_t slots = auto_cache ? 0 : impl->config.expert_cache_slots;
    const size_t total_experts = static_cast<size_t>(impl->g.n_layers * impl->g.n_expert);
    impl->host_res.assign(total_experts, core::kNotResident);
    bool cache_allocation_failed = false;
    auto fill_cache = [&](const CacheSizing& sizing) {
        cache_allocation_failed = false;
        if (!impl->xcache.open_sized(sizing.slot_bytes, impl->g.n_layers, impl->g.n_expert, error_msg)) {
            cache_allocation_failed = true;
            return false;
        }
        std::fill(impl->host_res.begin(), impl->host_res.end(), core::kNotResident);
        std::vector<uint8_t> staging;
        for (const auto& pair : sizing.pairs) {
            const auto l = pair.first, e = pair.second;
            const auto slot = impl->xcache.admit(l, e);
            const auto bytes = int64_t(layout.blob_bytes(l));
            const uint8_t* blob = nullptr;
            if (impl->src.transient(l, e)) {
                // Cache population must not retain thousands of file-tier staging blobs.
                staging.resize(size_t(bytes));
                if (!impl->src.copy_blob(l, e, staging.data())) {
                    error_msg = "cache source copy failed";
                    return false;
                }
                blob = staging.data();
            } else
                blob = impl->src.blob(l, e);
            if (slot < 0 || !blob || !impl->xcache.fill_slot_blocking(slot, blob, error_msg, bytes) ||
                !impl->xcache.verify_slot(slot, blob, error_msg, bytes)) {
                if (error_msg.empty())
                    error_msg = "model load: GPU expert slot could not be filled and verified";
                return false;
            }
            impl->host_res[size_t(l * impl->g.n_expert + e)] = slot;
        }
        return true;
    };
    if (slots > 0) {
        const auto sizing = size_expert_cache(layout, profile, impl->expert_performance.gpu_layers, UINT64_MAX, slots);
        if (int64_t(sizing.pairs.size()) != slots) {
            error_msg = "model load: requested cache exceeds validated, GPU-capable profile pairs";
            return nullptr;
        }
        if (!fill_cache(sizing))
            return nullptr;
    }
    impl->config.expert_cache_slots = int(slots);

    if (cudaMalloc(&impl->d_res, total_experts * sizeof(int32_t)) != cudaSuccess ||
        cudaMemcpy(impl->d_res, impl->host_res.data(), total_experts * sizeof(int32_t), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        error_msg = "failed to allocate device expert residency table";
        return nullptr;
    }
    sp.expert_cache_init_ms += ms_since(xcache_t0);

    // Pin RAM expert complement (all remaining experts resident in RAM)
    // RAM-resident complement.  Experts that did not fit the GPU cache stay in RAM: either zero-copy out of the
    // mapped GPACK (no copy at all, cheapest, but only without a GPU-side pinned requirement), or copied into the
    // resident set.  Page-locking and NUMA interleaving are alternatives, not partners: locked pages are what
    // async H2D copies want, interleaved pages are what a dual-socket host wants, and `pin_cache_complement`
    // decides which one the budget and the requested PCIe offload allow.  The budget passed here is bounded on
    // purpose - retrying with an unbounded one would discard the caller's cap.
    if (impl->config.resident_budget_gib > 0) {
        const auto pin_t0 = Clock::now();
        const uint64_t budget = ram_budget;
        const char* pin_env = std::getenv("GUILD_PIN_EXPERTS");
        const bool force_pin = (pin_env != nullptr && std::atoi(pin_env) != 0);
        const bool pcie_offload_requested = force_pin || (plan.pcie_expert_fraction > 0.0) ||
                                            (plan.pcie_expert_fraction < 0.0 && budget <= (24ull << 30));
        const bool gpu_needs_pinned = pcie_offload_requested && std::any_of(impl->expert_performance.gpu_layers.begin(),
                                                                            impl->expert_performance.gpu_layers.end(),
                                                                            [](uint8_t v) { return v != 0; });
        const char* zcopy_env = std::getenv("GUILD_EXPERT_MMAP");
        const bool zero_copy_mmap = (zcopy_env != nullptr && std::atoi(zcopy_env) != 0) && use_gpack &&
                                    impl->gpack->has_embedded_experts() && !gpu_needs_pinned;
        if (zero_copy_mmap) {
            sp.expert_zero_copy = true;
            sp.expert_ram_bytes = impl->gpack->total_expert_bytes();
            sp.expert_copy_calls = 0;
        } else {
            // The source already clamps to available memory and falls back from pinning.
            // Retrying with an unbounded budget would discard the requested backing cap.
            if (!impl->src.pin_cache_complement(impl->xcache, error_msg, gpu_needs_pinned, {}, -1, 4ull << 30, budget,
                                                profile.empty() ? nullptr : &profile))
                return nullptr;
            sp.expert_ram_bytes = impl->src.resident_bytes();
            sp.expert_page_locked = impl->src.pinned_bytes() > 0;
            sp.expert_numa_interleaved = !sp.expert_page_locked;
            const int64_t rem_experts = static_cast<int64_t>(total_experts) - slots;
            sp.expert_copy_calls = (use_gpack && impl->gpack->has_embedded_experts()) ? rem_experts : rem_experts * 3;
        }
        sp.numa_expert_pin_ms += ms_since(pin_t0);
    }

    // Session state initialization (with resource-aware QSA staging fallback under tight VRAM)
    // Shared session state.  The device staging buffer is sized for the full context and shared by all sessions;
    // under a tight VRAM budget the first sizing is given up and re-tried with the resource-aware variant that
    // keeps only the resident KV on the device.  It is zeroed here rather than per session so that a fresh
    // session always starts from a known-clean V state without paying for it per request.
    const auto kv_t0 = Clock::now();
    core::session_set_kv_share_staging(-1);
    uint64_t sbytes = core::session_bytes(impl->g, impl->config.max_context, impl->K);
    {
        size_t free_vram = 0, total_vram = 0;
        if (cudaMemGetInfo(&free_vram, &total_vram) == cudaSuccess && free_vram > 0) {
            const uint64_t mtp_margin = (!paths.mtp_dir.empty() && plan.mtp_spec_tokens > 0)
                                            ? (plan.mtp_bytes ? plan.mtp_bytes : (896ull << 20))
                                            : 0;
            const uint64_t reserve_needed = mtp_margin + std::max<uint64_t>(plan.prompt_scratch_bytes, 1024ull << 20);
            if (free_vram < sbytes + reserve_needed &&
                core::session_own_qsa_staging(impl->g, impl->config.max_context)) {
                core::session_set_kv_share_staging(1);
                sbytes = core::session_bytes(impl->g, impl->config.max_context, impl->K);
            }
        }
    }
    if (cudaMalloc(&impl->sbuf, sbytes) != cudaSuccess) {
        if (core::session_own_qsa_staging(impl->g, impl->config.max_context)) {
            cudaGetLastError();
            core::session_set_kv_share_staging(1);
            sbytes = core::session_bytes(impl->g, impl->config.max_context, impl->K);
            cudaMalloc(&impl->sbuf, sbytes);
        }
    }
    if (!impl->sbuf ||
        core::session_init(impl->g, impl->config.max_context, impl->K, impl->sbuf, impl->ss, 0, -1) == 0) {
        error_msg = "failed to allocate session buffer (" + std::to_string(sbytes >> 20) + " MiB)";
        return nullptr;
    }
    core::session_zero(impl->ss, impl->g, nullptr, impl->main_stream);
    if (cudaStreamSynchronize(impl->main_stream) != cudaSuccess) {
        error_msg = "model load: session state initialization failed";
        return nullptr;
    }
    sp.kv_session_init_ms += ms_since(kv_t0);
    sp.kv_device_bytes = sbytes;
    uint64_t model_kv_host_bytes = 0;
    for (int64_t j = 0; impl->ss.qsa_states && j < impl->ss.qsa_alloc; ++j) {
        model_kv_host_bytes += impl->ss.qsa_states[impl->ss.qsa_ord0 + j].host_allocation_bytes;
    }
    sp.kv_host_bytes = model_kv_host_bytes;

    const auto ple_t0 = Clock::now();
    if (use_gpack) {
        if (!bind_required_ple_gpack(*impl, *impl->gpack, error_msg))
            return nullptr;
    } else {
        if (!bind_required_ple(*impl, shards, error_msg))
            return nullptr;
        if (impl->ple_required)
            sp.container_opens += static_cast<int64_t>(shards.size()) + 1;
    }
    sp.ple_bind_ms += ms_since(ple_t0);

    // MTP Drafter
    const auto mtp_t0 = Clock::now();
    const bool mode_wants_mtp =
        (parsed_mode == spec::DecodeStrategyMode::Auto || parsed_mode == spec::DecodeStrategyMode::Mtp ||
         parsed_mode == spec::DecodeStrategyMode::Adaptive);
    const bool use_mtp =
        !paths.mtp_dir.empty() && mode_wants_mtp &&
        (plan.mtp_spec_tokens > 0 ||
         ((parsed_mode == spec::DecodeStrategyMode::Mtp || parsed_mode == spec::DecodeStrategyMode::Adaptive) &&
          plan.spec_max_draft > 0));
    if (use_mtp) {
        impl->mtp = std::make_unique<core::MtpDrafter>();
        if (!impl->mtp->load(paths.mtp_dir, impl->g, impl->ss, impl->config.spec, error_msg)) {
            error_msg = "failed to load MTP draft layer: " + error_msg;
            return nullptr;
        }
    }
    sp.mtp_init_ms += ms_since(mtp_t0);

    // Initialize speculative cost calibration (from GEXEC artifact when present, or model geometry priors).
    {
        auto& cal = impl->spec_calibration;
        const int64_t n_layers = std::max<int64_t>(1, impl->g.n_layers);
        const int64_t n_exp = std::max<int64_t>(1, impl->g.n_expert);
        const int64_t k_act = std::max<int64_t>(1, impl->K);
        const double p_single = std::clamp((double) k_act / (double) n_exp, 0.001, 0.5);
        const double cpu_1 = (n_exp >= 512) ? (0.46 * (double) n_layers) : (0.31 * (double) n_layers);
        const double gpu_1 = (n_exp >= 512) ? (0.30 * (double) n_layers) : (0.18 * (double) n_layers);
        for (int T = 1; T <= spec::GExecSpecCalibration::kMaxWindowT; ++T) {
            const double eff_trials = 1.0 + 0.80 * (double) (T - 1);
            const double u_ratio =
                (T == 1) ? 1.0 : std::clamp((1.0 - std::pow(1.0 - p_single, eff_trials)) / p_single, 1.0, (double) T);
            const double uses_per_expert = (double) T / u_ratio;
            const double cpu_T = cpu_1 * u_ratio * (1.0 + 0.22 * (uses_per_expert - 1.0));
            const double gpu_T = gpu_1 * (1.0 + 0.14 * (double) (T - 1));
            cal.distinct_expert_ratio[T] = u_ratio;
            cal.verify_cpu_expert_ms[T] = cpu_T;
            cal.verify_window_ms[T] = cpu_T + gpu_T;
        }
        if (impl->gexec && impl->gexec->has_section(gexec::GExecSectionKind::HeterogeneousPlan)) {
            const auto& hp = impl->gexec->heterogeneous_plan();
            if (hp.spec_calibrated && hp.verify_window_ms.size() >= 8) {
                cal.calibrated = true;
                for (int T = 1; T <= 8; ++T) {
                    if (hp.verify_window_ms[(size_t) (T - 1)] > 0.0)
                        cal.verify_window_ms[T] = hp.verify_window_ms[(size_t) (T - 1)];
                    if (hp.verify_cpu_expert_ms.size() >= 8 && hp.verify_cpu_expert_ms[(size_t) (T - 1)] > 0.0)
                        cal.verify_cpu_expert_ms[T] = hp.verify_cpu_expert_ms[(size_t) (T - 1)];
                    if (hp.distinct_expert_ratio.size() >= 8 && hp.distinct_expert_ratio[(size_t) (T - 1)] > 0.0)
                        cal.distinct_expert_ratio[T] = hp.distinct_expert_ratio[(size_t) (T - 1)];
                }
                if (hp.mtp_draft_step_ms > 0.0)
                    cal.mtp_draft_step_ms = hp.mtp_draft_step_ms;
                if (hp.mtp_kv_catchup_ms > 0.0)
                    cal.mtp_kv_catchup_ms = hp.mtp_kv_catchup_ms;
                if (hp.lookup_draft_ms > 0.0)
                    cal.lookup_draft_ms = hp.lookup_draft_ms;
                if (hp.retrieval_draft_ms > 0.0)
                    cal.retrieval_draft_ms = hp.retrieval_draft_ms;
                if (hp.kv_commit_ms > 0.0)
                    cal.kv_commit_ms = hp.kv_commit_ms;
                if (hp.verify_noise_cv > 0.0)
                    cal.verify_noise_cv = hp.verify_noise_cv;
                if (hp.concurrency_slope > 0.0)
                    cal.concurrency_slope = hp.concurrency_slope;
                if (hp.context_attn_ms_per_1k > 0.0)
                    cal.context_attn_ms_per_1k = hp.context_attn_ms_per_1k;
            }
        }
    }

    // Verifier
    const auto ver_t0 = Clock::now();
    // Single-device commits and the MTP draft have independent state and streams.
    // The old driver overlapped them; preserve its GUILD_COMMIT_SYNC A/B switch.
    core::VerifyHits vh;
    vh.d_res = impl->d_res;
    vh.h_res = impl->host_res.data();
    vh.cache_base = slots > 0 ? static_cast<const uint8_t*>(impl->xcache.device_slot(0)) : nullptr;
    vh.n_slots = slots;
    vh.slot_off = slots > 0 ? impl->xcache.slot_offsets() : nullptr;
    vh.blob = static_cast<int64_t>(kernels::cpu::expert_layout().max_blob);

    impl->ver = std::make_unique<core::Verifier>();
    if (!impl->ver->init(impl->wt, impl->g, impl->ss, vh, impl->native_head.loaded() ? &impl->native_head : nullptr,
                         impl->config.spec, error_msg)) {
        error_msg = "Verifier::init failed: " + error_msg;
        return nullptr;
    }
    impl->ver->set_async_commit(true);
    sp.verifier_init_ms += ms_since(ver_t0);

    if (use_mtp) {
        const auto mtpb_t0 = Clock::now();
        if (!impl->mtp->bind(impl->wt, impl->native_head.loaded() ? &impl->native_head : nullptr,
                             impl->ver->final_R_all(), error_msg)) {
            error_msg = "MTP bind failed: " + error_msg;
            return nullptr;
        }
        sp.mtp_init_ms += ms_since(mtpb_t0);
    }

    // Prefill initialization
    const auto pf_t0 = Clock::now();
    impl->prefill = std::make_unique<prefill::Prefill>();
    std::string pf_err;
    if (!impl->prefill->init(impl->wt, impl->g, impl->ss, impl->srcp,
                             impl->xcache.slots() > 0 ? &impl->xcache : nullptr,
                             impl->host_res.empty() ? nullptr : impl->host_res.data(), impl->config.prefill_chunk,
                             impl->main_stream, pf_err)) {
        // Prefill engine is optimized for Qwen4Exp; architectures with different geometry fall back to Verifier
        // ingestion
        impl->prefill.reset();
    }

    if (!impl->prefill && impl->g.is_dense_attention()) {
        int64_t batch = plan.prefill_batch;
        if (const char* value = std::getenv("GUILD_PREFILL_BATCH")) {
            size_t used = 0;
            batch = std::stoll(value, &used);
            if (used != std::strlen(value) || batch < 1 || batch > 512) {
                error_msg = "GUILD_PREFILL_BATCH must be an integer in 1..512";
                return nullptr;
            }
        }
        if (impl->g.hc == 1 && impl->g.ssm_state_size == 128 && impl->g.ssm_d_conv == 4 && impl->g.head_dim == 256 &&
            kernels::cpu::expert_layout().native) {
            impl->batched_prefill = std::make_unique<BatchedPrefill>();
            if (!impl->batched_prefill->init(impl->wt, impl->g, impl->ss, *impl->srcp, *impl->pool, impl->main_stream,
                                             error_msg, batch, impl->xcache.valid() ? &impl->xcache : nullptr))
                return nullptr;
            for (const auto& pkg : impl->batched_prefill->bound_package_ids()) {
                impl->kernel_lease.add_package(pkg);
            }
            impl->config.prefill_chunk = impl->batched_prefill->batch_size();
            model->config_.prefill_chunk = impl->config.prefill_chunk;
        } else if (batch > 0) {
            error_msg = "requested batched prefill is unsupported for this execution geometry";
            return nullptr;
        }
    }
    sp.prefill_init_ms += ms_since(pf_t0);

    auto finalize_cache = [&]() -> bool {
        if (!auto_cache)
            return true;
        size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
            error_msg = "cache free-memory query failed";
            return false;
        }
        const uint64_t reserve = plan.safety_reserve_bytes ? plan.safety_reserve_bytes : (512ull << 20);
        uint64_t budget = free > reserve ? free - reserve : 0;
        std::vector<std::pair<int32_t, int32_t>> cache_ranking = profile;
        if (cache_ranking.empty()) {
            uint64_t all_bytes = 0;
            for (int64_t l = 0; l < layout.n_layers; ++l)
                if (impl->expert_performance.gpu_layers[size_t(l)])
                    all_bytes += ((layout.blob_bytes(l) + 255) & ~uint64_t(255)) * uint64_t(layout.n_expert);
            // No guessed partial cache without a profile. A complete capable tier needs no ranking.
            if (all_bytes <= budget)
                for (int32_t e = 0; e < layout.n_expert; ++e)
                    for (int32_t l = 0; l < layout.n_layers; ++l)
                        if (impl->expert_performance.gpu_layers[size_t(l)])
                            cache_ranking.emplace_back(l, e);
        }
        for (int attempt = 0; attempt < 8 && budget; ++attempt) {
            const auto sizing = size_expert_cache(layout, cache_ranking, impl->expert_performance.gpu_layers, budget,
                                                  int64_t(cache_ranking.size()));
            if (sizing.pairs.empty())
                break;
            if (!fill_cache(sizing)) {
                if (!cache_allocation_failed)
                    return false;
                const auto status = cudaGetLastError();
                if (status != cudaSuccess && status != cudaErrorMemoryAllocation)
                    return false;
                impl->xcache.close();
                error_msg.clear();
                budget = budget / 4 * 3;
                std::fill(impl->host_res.begin(), impl->host_res.end(), core::kNotResident);
                continue;
            }
            slots = impl->xcache.slots();
            if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
                error_msg = "cache memory query failed";
                return false;
            }
            if (free >= reserve)
                break;
            impl->xcache.close();
            slots = 0;
            std::fill(impl->host_res.begin(), impl->host_res.end(), core::kNotResident);
            budget = budget / 4 * 3;
        }
        impl->config.expert_cache_slots = int(slots);
        if (cudaMemcpy(impl->d_res, impl->host_res.data(), total_experts * sizeof(int32_t), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            error_msg = "cache residency upload failed";
            return false;
        }
        vh.n_slots = slots;
        vh.cache_base = slots ? impl->xcache.device_slot(0) : nullptr;
        vh.slot_off = slots ? impl->xcache.slot_offsets() : nullptr;
        if (slots && (!impl->ver->reset_graphs(error_msg) || !impl->ver->configure_expert_cache(vh, error_msg)))
            return false;
        if (impl->prefill)
            impl->prefill->set_expert_cache(slots ? &impl->xcache : nullptr, impl->host_res.data());
        if (impl->batched_prefill)
            impl->batched_prefill->set_expert_cache(slots ? &impl->xcache : nullptr);
        return true;
    };

    // Setup Drive dispatch
    impl->drive.d.pool = impl->pool.get();
    impl->drive.d.src = impl->srcp;
    impl->drive.d.n_expert = impl->g.n_expert;
    impl->drive.d.n_embd = impl->g.n_embd;
    impl->drive.d.jobs.resize(static_cast<size_t>(impl->K));
    impl->drive.d.split_rows = true;
    impl->drive.d.cache = slots > 0 ? &impl->xcache : nullptr;
    impl->drive.d.cache_base = slots > 0 ? static_cast<const uint8_t*>(impl->xcache.device_slot(0)) : nullptr;
    impl->drive.d.cache_blob = static_cast<int64_t>(kernels::cpu::expert_layout().max_blob);
    impl->drive.d.cache_slot_off = slots > 0 ? impl->xcache.slot_offsets() : nullptr;
    impl->drive.d.host_res = impl->host_res.data();
    // Second calibration pass, after pinning and cache sizing have settled: the PCIe split the planner asked for is
    // planned against the memory that actually exists (pinned complement and/or a GPU expert cache), and the
    // pre-pinning measurement cannot know that yet.  Without either, the split is zeroed rather than guessed.
    const auto cal2_t0 = Clock::now();
    const bool need_post_calibrate =
        (plan.pcie_expert_fraction != 0.0) && (impl->src.pinned_bytes() > 0 || impl->xcache.slots() > 0);
    if (need_post_calibrate) {
        if (!calibrate_experts(layout, impl->src, *impl->pool, plan.pcie_expert_fraction, impl->main_stream,
                               impl->expert_performance, error_msg))
            return nullptr;
    } else {
        impl->expert_performance.pcie_numerators.assign(size_t(impl->g.n_layers), 0);
    }
    if (!native_pack && impl->g.n_embd == kernels::cpu::H && impl->g.n_ff == kernels::cpu::FF) {
        impl->expert_performance.gpu_layers.assign(size_t(impl->g.n_layers), 1);
        const double fraction = plan.pcie_expert_fraction < 0 ? 0.2 : plan.pcie_expert_fraction;
        impl->expert_performance.pcie_numerators.assign(size_t(impl->g.n_layers), int(fraction * 256 + 0.5));
    }
    impl->drive.d.plan = impl->ver->plan_sink();
    impl->drive.d.pcie_layer_numerators = impl->expert_performance.pcie_numerators;
    const bool any_offload =
        std::any_of(impl->drive.d.pcie_layer_numerators.begin(), impl->drive.d.pcie_layer_numerators.end(),
                    [](int value) { return value > 0; });
    impl->ver->set_pcie_mode(any_offload ? 2 : 0);
    impl->ver->set_pcie_enabled(any_offload);
    if (plan.pcie_expert_fraction < 0 && any_offload &&
        !tune_expert_split(*impl->ver, impl->ss, impl->g, impl->drive.d, drive_pool_multi, &impl->drive,
                           impl->config.spec, impl->main_stream, impl->expert_performance, error_msg))
        return nullptr;
    sp.expert_calibration_post_ms += ms_since(cal2_t0);
    if (auto_cache) {
        const auto gcap_t0 = Clock::now();
        // Capture every verifier size before the free-memory query, so automatic
        // cache sizing includes graph residency rather than only tensor arenas.
        std::vector<int32_t> input(size_t(impl->config.spec), 0), output(input.size());
        for (int rows = 1; rows <= impl->config.spec; ++rows) {
            core::session_zero(impl->ss, impl->g, nullptr, impl->main_stream);
            if (cudaStreamSynchronize(impl->main_stream) != cudaSuccess ||
                !impl->ver->run(rows, input.data(), 0, drive_pool_multi, &impl->drive, output.data(), error_msg))
                return nullptr;
        }
        sp.graph_capture_startup_ms += ms_since(gcap_t0);
        const auto fcache_t0 = Clock::now();
        core::session_zero(impl->ss, impl->g, nullptr, impl->main_stream);
        if (cudaStreamSynchronize(impl->main_stream) != cudaSuccess || !finalize_cache())
            return nullptr;
        sp.expert_cache_init_ms += ms_since(fcache_t0);
        impl->drive.d.cache = slots ? &impl->xcache : nullptr;
        impl->drive.d.cache_base = slots ? impl->xcache.device_slot(0) : nullptr;
        impl->drive.d.cache_slot_off = slots ? impl->xcache.slot_offsets() : nullptr;
    }
    sp.expert_vram_bytes = static_cast<uint64_t>(impl->xcache.bytes());
    const auto& numerators = impl->drive.d.pcie_layer_numerators;
    impl->drive.d.pcie_num = numerators.empty() ? 0 : *std::max_element(numerators.begin(), numerators.end());
    impl->config.pcie_layer_numerators = numerators;
    std::fprintf(stderr,
                 "guild performance: %d measured formats, CPU %.3f ms, GPU %.3f ms, copy %.3f ms; "
                 "PCIe up to %d/256, GPU cache %lld slots (%lld bytes)\n",
                 impl->expert_performance.measured_formats, impl->expert_performance.cpu_ms,
                 impl->expert_performance.gpu_ms, impl->expert_performance.copy_ms, impl->drive.d.pcie_num,
                 static_cast<long long>(slots), static_cast<long long>(impl->xcache.bytes()));
    std::fprintf(stderr,
                 "guild performance: CPU workers %d + host %d; verifier capacity %d; "
                 "MTP %s; min draft probability %.3f; RAM expert backing %llu bytes\n",
                 impl->pool->workers(), int(impl->pool->host_works()), impl->config.spec, use_mtp ? "on" : "off",
                 double(impl->config.spec_min_p), static_cast<unsigned long long>(impl->src.resident_bytes()));
    model->config_ = impl->config;

    // Load tokenizer (from GPACK binary section if available, else tokenizer_dir)
    if (use_gpack && impl->gpack->has_tokenizer_bin()) {
        const auto tok_t0 = Clock::now();
        uint64_t tok_bytes = 0;
        const uint8_t* tok_data = impl->gpack->tokenizer_bin_data(&tok_bytes);
        std::string tok_err;
        if (!model->tokenizer_->load_binary(tok_data, static_cast<size_t>(tok_bytes), tok_err)) {
            error_msg = "model load: GPACK tokenizer: " + tok_err;
            return nullptr;
        }
        const auto& eids = model->tokenizer_->eos_token_ids();
        if (!eids.empty()) {
            impl->config.eos_token_ids.clear();
            for (int32_t id : eids)
                impl->config.eos_token_ids.push_back(id);
            model->config_.eos_token_ids = impl->config.eos_token_ids;
        }
        sp.tokenizer_load_ms += ms_since(tok_t0);
    } else if (!paths.tokenizer_dir.empty()) {
        const auto tok_t0 = Clock::now();
        std::string tok_err;
        if (!model->tokenizer_->load(paths.tokenizer_dir, tok_err)) {
            error_msg = "model load: tokenizer: " + tok_err;
            return nullptr;
        } else {
            const auto& eids = model->tokenizer_->eos_token_ids();
            if (!eids.empty()) {
                impl->config.eos_token_ids.clear();
                for (int32_t id : eids)
                    impl->config.eos_token_ids.push_back(id);
                model->config_.eos_token_ids = impl->config.eos_token_ids;
            }
        }
        sp.tokenizer_load_ms += ms_since(tok_t0);
    }

    sp.total_load_ms = ms_since(load_t0);
    impl->startup_profile = sp;
    model->startup_profile_ = sp;
    model->ready_->store(true);
    return model;
} catch (const std::exception& e) {
    error_msg = std::string("model load: ") + e.what();
    return nullptr;
}

std::unique_ptr<GuildSession> GuildModel::create_session(int64_t max_context) {
    if (!is_ready())
        return nullptr;
    int64_t ctx = max_context > 0 ? max_context : config_.max_context;
    if (ctx > config_.max_context)
        return nullptr;
    return std::make_unique<GuildSession>(this, ctx);
}

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
