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

namespace guild::runtime {

GuildModel::GuildModel() : impl_(std::make_unique<Impl>()) {}
GuildModel::~GuildModel() = default;

std::unique_ptr<GuildModel> GuildModel::load(
    const ModelPaths& paths,
    const model::ModelDescriptor& requested_desc,
    const memory::ExecutionPlan& plan,
    std::string& error_msg
) try {
    platform::configure_device_runtime_policy();
    error_msg.clear();
    if (paths.pack_dir.empty() || !std::filesystem::is_regular_file(std::filesystem::path(paths.pack_dir) / "index.txt")) {
        error_msg = "model load: missing pack index.txt: " + paths.pack_dir;
        return nullptr;
    }
    if (paths.primary_model_path.empty()) {
        error_msg = "model load: primary GGUF path is required";
        return nullptr;
    }
    // A missing shard is a load failure, not a reason to continue with a partial model.
    std::vector<std::string> shards = guild::gguf_split_paths(paths.primary_model_path);
    if (shards.size() == 1) shards.insert(shards.end(), paths.additional_shards.begin(), paths.additional_shards.end());
    {
        std::string pkg_err;
        if (abi::GKernelPackageManager::instance().load_enabled_packages(pkg_err) != GUILD_STATUS_OK) {
            error_msg = "model load: enabled kernel package failed: " + pkg_err;
            return nullptr;
        }
    }
    model::ModelDescriptor desc = requested_desc;
    const model::IModelRunner* runner = nullptr;
    {
        const guild::GgufModel artifact(shards);
        if (model::ModelRunnerRegistry::instance().resolve_for_gguf(artifact.meta(), &runner, error_msg) != GUILD_STATUS_OK ||
            !runner || !runner->validate_and_describe(artifact, desc, error_msg)) {
            return nullptr;
        }
    }
    if (plan.context_length <= 0 || plan.context_length > desc.attn.context_length ||
        plan.mtp_spec_tokens < 0 || plan.mtp_spec_tokens > kernels::kVerifyMaxT) {
        error_msg = "model load: invalid context or verification capacity";
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

    impl->paths = paths;
    impl->desc = desc;
    impl->plan = plan;
    impl->runner = runner;
    impl->runner_lease = model::SessionRunnerLease(&model::ModelRunnerRegistry::instance(),
                                                   runner && runner->package_id() ? runner->package_id() : "");
    impl->kernel_lease = kernels::SessionKernelLease(&kernels::KernelRegistry::instance());

    // Apply execution plan to runtime config
    impl->config.max_context = plan.context_length;
    impl->config.kv_format = memory::kv_precision_to_string(plan.kv_format);
    impl->config.kv_host_only = (plan.kv_mode == memory::KvMode::HostOnly);
    impl->config.kv_resident = 0;
    impl->config.spec = std::max(2, plan.mtp_spec_tokens);
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
    if (!runner->configure_geometry(desc, impl->g, impl->K, error_msg)) {
        return nullptr;
    }
    const auto host_caps = kernels::query_host_device_capabilities();
    if (model::ModelRunnerRegistry::instance().validate_runner_requirements(
            *runner, desc, impl->g, plan, kernels::KernelRegistry::instance(),
            host_caps, impl->runner_exec, error_msg) != GUILD_STATUS_OK) {
        return nullptr;
    }

    // RoPE scaling
    kernels::RopeScaling rope_cfg;
    if (desc.rope.freq_base > 0) rope_cfg.freq_base = desc.rope.freq_base;
    if (desc.rope.factor > 0) rope_cfg.factor = desc.rope.factor;
    if (desc.rope.orig_ctx > 0) rope_cfg.orig_ctx = desc.rope.orig_ctx;
    if (desc.rope.type == "linear") rope_cfg.type = kernels::RopeScalingType::Linear;
    else if (desc.rope.type == "yarn") rope_cfg.type = kernels::RopeScalingType::YaRN;
    else rope_cfg.type = kernels::RopeScalingType::None;
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
    core::layer_set_shared_early(true);

    // CPU expert pool
    int n_workers = impl->config.pool_workers;
    if (const char* ew = std::getenv("GUILD_POOL_WORKERS")) {
        try { n_workers = std::stoi(ew); } catch (...) {}
    }
    impl->pool = std::make_unique<kernels::cpu::ExpertPool>(n_workers, true, true, kernels::cpu::PoolAffinity::All);
    impl->config.pool_workers = impl->pool->workers();

    // CUDA stream
    if (cudaStreamCreateWithFlags(&impl->main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        error_msg = "failed to create CUDA main stream";
        return nullptr;
    }
    impl->kernel_lease = kernels::SessionKernelLease(&kernels::KernelRegistry::instance(),
                                                      impl->runner_exec.resolved_package_ids,
                                                      impl->main_stream);
    {
        kernels::KernelSelectionQuery query;
        query.op_kind=GUILD_OP_BATCHED_EXPERT_EXEC; query.allowed_backends=GUILD_BACKEND_CPU;
        const guild_kernel_op_desc_v1_t* selected=nullptr;
        if (kernels::KernelRegistry::instance().select(query,host_caps,&selected,error_msg) != GUILD_STATUS_OK)
            return nullptr;
        if (selected->next_ext) {
            guild_cpu_expert_ext_v1_t table{};
            std::memcpy(&table,selected->next_ext,sizeof(table));
            if (table.struct_size<sizeof(table) || table.tag!=GUILD_CPU_EXPERT_TAG_V1 || table.version!=1) {
                error_msg="model load: invalid CPU expert contract"; return nullptr;
            }
            impl->kernel_lease.add_package(selected->package_id);
            impl->pool->bind_native_kernels(table);
            impl->native_cpu_package=selected->package_id;
            if(table.struct_size>=sizeof(guild_cpu_backend_ext_v2_t)) {
                guild_cpu_backend_ext_v2_t complete{};
                std::memcpy(&complete,selected->next_ext,sizeof(complete));
                if(complete.backend_version!=GUILD_CPU_BACKEND_VERSION_V2){error_msg="invalid CPU backend contract";return nullptr;}
                impl->cpu_backend=complete;
                impl->pool->bind_backend_kernels(complete);
            }
        }
#if defined(GUILD_NO_BUILTIN_NATIVE_CPU)
        if (impl->native_cpu_package.empty()) {
            error_msg="model load: native CPU fallback disabled; install a compatible CPU .gkernel";
            return nullptr;
        }
#endif
    }

    // Doorbell
    if (core::doorbell_init(impl->g, impl->K, impl->db) == 0) {
        error_msg = "doorbell_init failed";
        return nullptr;
    }

    // Expert layout from pack directory
    if (!guild::kernels::cpu::expert_layout_load(paths.pack_dir, impl->g.n_layers, impl->g.n_expert, error_msg,
                                                  impl->g.n_embd, impl->g.n_ff,
                                                  impl->cpu_backend ? &*impl->cpu_backend : nullptr)) {
        return nullptr;
    }
    const bool native_pack = guild::kernels::cpu::expert_layout().native;
    const auto& layout = kernels::cpu::expert_layout();
    const uint64_t ram_budget = plan.expert_ram_budget_bytes ? std::min(plan.expert_ram_budget_bytes, layout.total)
                                : plan.routed_expert_total_bytes ? plan.routed_experts_ram_bytes : layout.total;
    impl->config.resident_budget_gib = (ram_budget >> 30) + ((ram_budget & ((1ull << 30) - 1)) != 0);
    model->config_ = impl->config;
    if ((!native_pack && !kernels::cpu::cpu_avx512_ok() && !impl->cpu_backend) ||
        impl->g.n_ff > kernels::cpu::FF || impl->g.n_embd > kernels::cpu::H) {
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
    core::layer_set_native_bf16(native_pack || !shards.empty());
    kernels::ple_set_native_bf16(native_pack || !shards.empty());
    kernels::shared_expert_set_native_bf16(native_pack || !shards.empty());
    kernels::native_moe_combine_set_enabled(native_pack || !shards.empty());
    kernels::native_gdn_set_enabled(native_pack || !shards.empty());
    kernels::native_router_set_enabled(native_pack || !shards.empty());
    kernels::native_qsa_set_enabled(native_pack || !shards.empty());
    kernels::native_qsa_indexer_set_enabled(native_pack || !shards.empty());
    kernels::native_rope_set_enabled(native_pack || !shards.empty());
    kernels::ple_set_native_postops(native_pack || !shards.empty());

    // Native token embeddings
    if (!shards.empty()) {
        if (!impl->native_embed.load(shards, impl->g.n_embd, desc.attn.vocab_size, error_msg)) {
            return nullptr;
        }
        core::set_native_embed(&impl->native_embed);
    }

    // Load weight table from pack directory
    std::set<std::string> skip;
    if (!shards.empty()) {
        if (!guild::core::NativeDense::served_names(shards, true, skip, error_msg)) {
            return nullptr;
        }
        if (native_pack && !guild::core::NativeDense::keep_unquantized_ple_key(paths.pack_dir, skip, error_msg)) {
            return nullptr;
        }
    }
    if (!runner->configure_weight_skip_set(native_pack, !shards.empty(), paths.pack_dir, skip, error_msg)) {
        return nullptr;
    }

    uint64_t pool_bytes = 0;
    if (!core::WeightTable::pool_bytes(paths.pack_dir, pool_bytes, error_msg, skip.empty() ? nullptr : &skip)) {
        return nullptr;
    }

    if (cudaMalloc(&impl->arena, pool_bytes) != cudaSuccess) {
        error_msg = "failed to allocate VRAM weight arena (" + std::to_string(pool_bytes >> 20) + " MiB)";
        return nullptr;
    }

    if (!impl->wt.load(paths.pack_dir, impl->arena, pool_bytes, error_msg, skip.empty() ? nullptr : &skip)) {
        return nullptr;
    }

    // Load native dense projections if available
    if (!shards.empty()) {
        if (!impl->native_dense.load(shards, impl->wt, error_msg, true)) {
            return nullptr;
        }
    }
    if (!core::check_all(impl->wt, impl->g, error_msg)) return nullptr;

    // Load native head if available
    const guild::core::WeightRef* wo = impl->wt.find("output.weight");
    int64_t n_vocab = wo ? wo->ne1 : desc.attn.vocab_size;
    if (!shards.empty()) {
        if (!impl->native_head.load(shards, impl->g.n_embd, n_vocab, error_msg)) {
            return nullptr;
        }
    }

    // Open expert source (FileExpertSource over pack directory / GGUF shards)
    if (!shards.empty()) {
        impl->src.set_gguf(shards.front());
    }
    if (!impl->src.open(paths.pack_dir, impl->g.n_layers, impl->g.n_expert, error_msg)) {
        return nullptr;
    }
    impl->srcp = &impl->src;
    if (!calibrate_experts(layout, impl->src, *impl->pool, 0, impl->main_stream,
                           impl->expert_performance, error_msg)) return nullptr;
    if (!native_pack) {
        const bool supported = impl->g.n_embd == kernels::cpu::H && impl->g.n_ff == kernels::cpu::FF;
        impl->expert_performance.gpu_layers.assign(size_t(impl->g.n_layers), supported);
    }

    // Load expert profile if provided
    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!paths.expert_profile_path.empty()) {
        int64_t profile_slots = 0;
        if (!core::read_expert_profile(paths.expert_profile_path, impl->g.n_layers, impl->g.n_expert,
                                       profile, profile_slots, error_msg)) return nullptr;
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
                if (!impl->src.copy_blob(l, e, staging.data())) { error_msg = "cache source copy failed"; return false; }
                blob = staging.data();
            } else blob = impl->src.blob(l, e);
            if (slot < 0 || !blob || !impl->xcache.fill_slot_blocking(slot, blob, error_msg, bytes) ||
                !impl->xcache.verify_slot(slot, blob, error_msg, bytes)) {
                if (error_msg.empty()) error_msg = "model load: GPU expert slot could not be filled and verified";
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
        if (!fill_cache(sizing)) return nullptr;
    }
    impl->config.expert_cache_slots = int(slots);

    if (cudaMalloc(&impl->d_res, total_experts * sizeof(int32_t)) != cudaSuccess ||
        cudaMemcpy(impl->d_res, impl->host_res.data(), total_experts * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        error_msg = "failed to allocate device expert residency table";
        return nullptr;
    }

    // Pin RAM expert complement (all remaining experts resident in RAM)
    if (impl->config.resident_budget_gib > 0) {
        const uint64_t budget = ram_budget;
        const bool gpu_needs_pinned = std::any_of(
            impl->expert_performance.gpu_layers.begin(), impl->expert_performance.gpu_layers.end(),
            [](uint8_t v) { return v != 0; });
        // The source already clamps to available memory and falls back from pinning.
        // Retrying with an unbounded budget would discard the requested backing cap.
        if (!impl->src.pin_cache_complement(impl->xcache, error_msg, gpu_needs_pinned, {}, -1, 4ull << 30, budget,
                                            profile.empty() ? nullptr : &profile)) return nullptr;
    }

    // Session state initialization
    const uint64_t sbytes = core::session_bytes(impl->g, impl->config.max_context, impl->K);
    if (cudaMalloc(&impl->sbuf, sbytes) != cudaSuccess ||
        core::session_init(impl->g, impl->config.max_context, impl->K, impl->sbuf, impl->ss, 0, -1) == 0) {
        error_msg = "failed to allocate session buffer (" + std::to_string(sbytes >> 20) + " MiB)";
        return nullptr;
    }
    core::session_zero(impl->ss, impl->g, nullptr, impl->main_stream);
    if (cudaStreamSynchronize(impl->main_stream) != cudaSuccess) {
        error_msg = "model load: session state initialization failed";
        return nullptr;
    }

    if (!bind_required_ple(*impl, shards, error_msg)) return nullptr;

    // MTP Drafter
    bool use_mtp = !paths.mtp_dir.empty() && plan.mtp_spec_tokens > 0;
    if (use_mtp) {
        impl->mtp = std::make_unique<core::MtpDrafter>();
        if (!impl->mtp->load(paths.mtp_dir, impl->g, impl->ss, impl->config.spec, error_msg)) {
            error_msg = "failed to load MTP draft layer: " + error_msg;
            return nullptr;
        }
    }

    // Verifier
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
    if (!impl->ver->init(impl->wt, impl->g, impl->ss, vh,
                        impl->native_head.loaded() ? &impl->native_head : nullptr,
                        impl->config.spec, error_msg)) {
        error_msg = "Verifier::init failed: " + error_msg;
        return nullptr;
    }
    impl->ver->set_async_commit(true);

    if (use_mtp) {
        if (!impl->mtp->bind(impl->wt,
                            impl->native_head.loaded() ? &impl->native_head : nullptr,
                            impl->ver->final_R_all(), error_msg)) {
            error_msg = "MTP bind failed: " + error_msg;
            return nullptr;
        }
    }

    // Prefill initialization
    impl->prefill = std::make_unique<prefill::Prefill>();
    std::string pf_err;
    if (!impl->prefill->init(impl->wt, impl->g, impl->ss, impl->srcp,
                            impl->xcache.slots() > 0 ? &impl->xcache : nullptr,
                            impl->host_res.empty() ? nullptr : impl->host_res.data(),
                            impl->config.prefill_chunk, impl->main_stream, pf_err)) {
        // Prefill engine is optimized for Qwen4Exp; architectures with different geometry fall back to Verifier ingestion
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
        if (impl->g.hc == 1 && impl->g.ssm_state_size == 128 && impl->g.ssm_d_conv == 4 &&
            impl->g.head_dim == 256 && kernels::cpu::expert_layout().native &&
            plan.kv_format == memory::KvPrecision::FP16) {
            impl->batched_prefill = std::make_unique<BatchedPrefill>();
            if (!impl->batched_prefill->init(impl->wt, impl->g, impl->ss, *impl->srcp, *impl->pool,
                impl->main_stream, error_msg, batch, impl->xcache.valid() ? &impl->xcache : nullptr)) return nullptr;
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

    auto finalize_cache = [&]() -> bool {
        if (!auto_cache) return true;
        size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess) { error_msg = "cache free-memory query failed"; return false; }
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
                        if (impl->expert_performance.gpu_layers[size_t(l)]) cache_ranking.emplace_back(l, e);
        }
        for (int attempt = 0; attempt < 8 && budget; ++attempt) {
            const auto sizing = size_expert_cache(layout, cache_ranking, impl->expert_performance.gpu_layers,
                                                   budget, int64_t(cache_ranking.size()));
            if (sizing.pairs.empty()) break;
            if (!fill_cache(sizing)) {
                if (!cache_allocation_failed) return false;
                const auto status = cudaGetLastError();
                if (status != cudaSuccess && status != cudaErrorMemoryAllocation) return false;
                impl->xcache.close(); error_msg.clear(); budget = budget / 4 * 3;
                std::fill(impl->host_res.begin(), impl->host_res.end(), core::kNotResident);
                continue;
            }
            slots = impl->xcache.slots();
            if (cudaMemGetInfo(&free, &total) != cudaSuccess) { error_msg = "cache memory query failed"; return false; }
            if (free >= reserve) break;
            impl->xcache.close(); slots = 0;
            std::fill(impl->host_res.begin(), impl->host_res.end(), core::kNotResident);
            budget = budget / 4 * 3;
        }
        impl->config.expert_cache_slots = int(slots);
        if (cudaMemcpy(impl->d_res, impl->host_res.data(), total_experts * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            error_msg = "cache residency upload failed"; return false;
        }
        vh.n_slots = slots;
        vh.cache_base = slots ? impl->xcache.device_slot(0) : nullptr;
        vh.slot_off = slots ? impl->xcache.slot_offsets() : nullptr;
        if (slots && (!impl->ver->reset_graphs(error_msg) || !impl->ver->configure_expert_cache(vh, error_msg))) return false;
        if (impl->prefill) impl->prefill->set_expert_cache(slots ? &impl->xcache : nullptr, impl->host_res.data());
        if (impl->batched_prefill) impl->batched_prefill->set_expert_cache(slots ? &impl->xcache : nullptr);
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
    if (!calibrate_experts(layout, impl->src, *impl->pool, plan.pcie_expert_fraction,
                           impl->main_stream, impl->expert_performance, error_msg)) return nullptr;
    if (!native_pack && impl->g.n_embd == kernels::cpu::H && impl->g.n_ff == kernels::cpu::FF) {
        impl->expert_performance.gpu_layers.assign(size_t(impl->g.n_layers), 1);
        const double fraction = plan.pcie_expert_fraction < 0 ? 0.2 : plan.pcie_expert_fraction;
        impl->expert_performance.pcie_numerators.assign(size_t(impl->g.n_layers), int(fraction * 256 + 0.5));
    }
    impl->drive.d.plan = impl->ver->plan_sink();
    impl->drive.d.pcie_layer_numerators = impl->expert_performance.pcie_numerators;
    const bool any_offload = std::any_of(impl->drive.d.pcie_layer_numerators.begin(),
                                        impl->drive.d.pcie_layer_numerators.end(), [](int value) { return value > 0; });
    impl->ver->set_pcie_mode(any_offload ? 2 : 0);
    if (plan.pcie_expert_fraction < 0 && any_offload &&
        !tune_expert_split(*impl->ver, impl->ss, impl->g, impl->drive.d, drive_pool_multi,
                            &impl->drive, impl->config.spec, impl->main_stream, impl->expert_performance, error_msg)) return nullptr;
    if (auto_cache) {
        // Capture every verifier size before the free-memory query, so automatic
        // cache sizing includes graph residency rather than only tensor arenas.
        std::vector<int32_t> input(size_t(impl->config.spec), 0), output(input.size());
        for (int rows = 1; rows <= impl->config.spec; ++rows) {
            core::session_zero(impl->ss, impl->g, nullptr, impl->main_stream);
            if (cudaStreamSynchronize(impl->main_stream) != cudaSuccess ||
                !impl->ver->run(rows, input.data(), 0, drive_pool_multi, &impl->drive, output.data(), error_msg)) return nullptr;
        }
        core::session_zero(impl->ss, impl->g, nullptr, impl->main_stream);
        if (cudaStreamSynchronize(impl->main_stream) != cudaSuccess || !finalize_cache()) return nullptr;
        impl->drive.d.cache = slots ? &impl->xcache : nullptr;
        impl->drive.d.cache_base = slots ? impl->xcache.device_slot(0) : nullptr;
        impl->drive.d.cache_slot_off = slots ? impl->xcache.slot_offsets() : nullptr;
    }
    const auto& numerators = impl->drive.d.pcie_layer_numerators;
    impl->drive.d.pcie_num = numerators.empty() ? 0 : *std::max_element(numerators.begin(), numerators.end());
    impl->config.pcie_layer_numerators = numerators;
    std::fprintf(stderr, "guild performance: %d measured formats, CPU %.3f ms, GPU %.3f ms, copy %.3f ms; "
                         "PCIe up to %d/256, GPU cache %lld slots (%lld bytes)\n",
                 impl->expert_performance.measured_formats, impl->expert_performance.cpu_ms,
                 impl->expert_performance.gpu_ms, impl->expert_performance.copy_ms, impl->drive.d.pcie_num,
                 static_cast<long long>(slots), static_cast<long long>(impl->xcache.bytes()));
    std::fprintf(stderr, "guild performance: CPU workers %d + host %d; verifier capacity %d; "
                         "MTP %s; min draft probability %.3f; RAM expert backing %llu bytes\n",
                 impl->pool->workers(), int(impl->pool->host_works()), impl->config.spec,
                 use_mtp ? "on" : "off", double(impl->config.spec_min_p),
                 static_cast<unsigned long long>(impl->src.resident_bytes()));
    model->config_ = impl->config;

    // Load tokenizer
    if (!paths.tokenizer_dir.empty()) {
        std::string tok_err;
        if (!model->tokenizer_.load(paths.tokenizer_dir, tok_err)) {
            error_msg = "model load: tokenizer: " + tok_err;
            return nullptr;
        } else {
            const auto& eids = model->tokenizer_.eos_token_ids();
            if (!eids.empty()) {
                impl->config.eos_token_ids.clear();
                for (int32_t id : eids) impl->config.eos_token_ids.push_back(id);
                model->config_.eos_token_ids = impl->config.eos_token_ids;
            }
        }
    }

    model->ready_ = true;
    return model;
} catch (const std::exception& e) {
    error_msg = std::string("model load: ") + e.what();
    return nullptr;
}

std::unique_ptr<GuildSession> GuildModel::create_session(int64_t max_context) {
    if (!ready_) return nullptr;
    int64_t ctx = max_context > 0 ? max_context : config_.max_context;
    if (ctx > config_.max_context) return nullptr;
    return std::make_unique<GuildSession>(this, ctx);
}

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
