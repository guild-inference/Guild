#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "model_impl.hpp"

#include <cuda_runtime.h>

#include <chrono>
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
    const model::ModelDescriptor& desc,
    const memory::ExecutionPlan& plan,
    std::string& error_msg
) {
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }

    auto model = std::unique_ptr<GuildModel>(new GuildModel());
    auto* impl = model->impl_.get();

    impl->paths = paths;
    impl->desc = desc;
    impl->plan = plan;

    // Apply execution plan to runtime config
    impl->config.max_context = plan.context_length;
    impl->config.kv_format = memory::kv_precision_to_string(plan.kv_format);
    impl->config.kv_host_only = (plan.kv_mode == memory::KvMode::HostOnly);
    impl->config.kv_resident = 0;
    impl->config.resident_budget_gib = plan.routed_expert_total_bytes >> 30;
    if (impl->config.resident_budget_gib < 56) impl->config.resident_budget_gib = 56;
    impl->config.spec = plan.mtp_spec_tokens > 0 ? plan.mtp_spec_tokens : 4;
    impl->config.spec_min_p = 0.5f;
    impl->config.expert_cache_slots = plan.routed_experts_in_gpu;
    impl->config.prefill_chunk = plan.prefill_chunk > 0 ? plan.prefill_chunk : 512;
    impl->config.vram_reserve_mib = 512;

    model->desc_ = desc;
    model->plan_ = plan;
    model->paths_ = paths;
    model->config_ = impl->config;

    // Geometry
    impl->g.apply_descriptor(desc);
    if (desc.moe.k_active_experts > 0) {
        impl->K = desc.moe.k_active_experts;
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
    core::qsa_set_kv_int8(impl->config.kv_format == "int8");
    core::qsa_set_kv_host_only(impl->config.kv_host_only);
    core::qsa_set_kv_resident(impl->config.kv_resident);
    core::layer_set_shared_early(true);

    // Kernel settings (fast paths)
    core::layer_set_fast_attn(true);
    core::layer_set_publish_kernel(true);
    core::layer_set_fused_gdn(true);
    core::layer_set_fast_select(true);
    core::layer_set_fused_gr(true);
    kernels::gr_set_fp32_activations(false);
    kernels::gr_set_native_mmvf(true);
    core::layer_set_native_bf16(false);
    kernels::ple_set_native_bf16(false);
    kernels::shared_expert_set_native_bf16(false);
    kernels::native_moe_combine_set_enabled(false);
    kernels::native_gdn_set_enabled(false);
    kernels::native_router_set_enabled(false);
    kernels::native_qsa_set_enabled(false);
    kernels::native_qsa_indexer_set_enabled(false);
    kernels::native_rope_set_enabled(false);
    kernels::ple_set_native_postops(false);

    // CPU expert pool
    impl->pool = std::make_unique<kernels::cpu::ExpertPool>(impl->config.pool_workers, true, true, kernels::cpu::PoolAffinity::All);

    // CUDA stream
    if (cudaStreamCreateWithFlags(&impl->main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        error_msg = "failed to create CUDA main stream";
        return nullptr;
    }

    // Doorbell
    if (core::doorbell_init(impl->g, impl->K, impl->db) == 0) {
        error_msg = "doorbell_init failed";
        return nullptr;
    }

    // Determine model shards
    std::vector<std::string> shards;
    if (!paths.primary_model_path.empty()) {
        try {
            shards = guild::gguf_split_paths(paths.primary_model_path);
        } catch (...) {
            shards.push_back(paths.primary_model_path);
            shards.insert(shards.end(), paths.additional_shards.begin(), paths.additional_shards.end());
        }
    }
    if (shards.empty() && !paths.additional_shards.empty()) {
        shards = paths.additional_shards;
    }

    // Native token embeddings
    if (!shards.empty()) {
        if (!impl->native_embed.load(shards, impl->g.n_embd, 248320, error_msg)) {
            return nullptr;
        }
        core::set_native_embed(&impl->native_embed);
    }

    // Load weight table from pack directory
    std::set<std::string> skip;
    skip.insert("output.weight");
    skip.insert("token_embd.weight");

    uint64_t pool_bytes = 0;
    if (!core::WeightTable::pool_bytes(paths.pack_dir, pool_bytes, error_msg, &skip)) {
        return nullptr;
    }

    if (cudaMalloc(&impl->arena, pool_bytes) != cudaSuccess) {
        error_msg = "failed to allocate VRAM weight arena (" + std::to_string(pool_bytes >> 20) + " MiB)";
        return nullptr;
    }

    if (!impl->wt.load(paths.pack_dir, impl->arena, pool_bytes, error_msg, &skip)) {
        return nullptr;
    }

    // Load native dense projections if available
    if (!shards.empty()) {
        if (!impl->native_dense.load(shards, impl->wt, error_msg, false)) {
            // Non-fatal, continue with pack weights
            error_msg.clear();
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

    // Load expert profile if provided
    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!paths.expert_profile_path.empty() && std::filesystem::exists(paths.expert_profile_path)) {
        std::ifstream pf(paths.expert_profile_path, std::ios::binary);
        if (pf.is_open()) {
            int32_t count = 0;
            if (pf.read(reinterpret_cast<char*>(&count), sizeof(count)) && count > 0) {
                profile.resize(count);
                pf.read(reinterpret_cast<char*>(profile.data()), count * sizeof(std::pair<int32_t, int32_t>));
            }
        }
    }

    // Expert cache in VRAM
    int64_t slots = impl->config.expert_cache_slots;
    if (slots < 0) {
        // Auto sizing from free VRAM
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t blob = static_cast<int64_t>(kernels::cpu::expert_layout().max_blob);
        int64_t reserve = (static_cast<int64_t>(impl->config.vram_reserve_mib) << 20) + (100LL << 20);
        slots = std::max<int64_t>(0, (static_cast<int64_t>(free_b) - reserve) / blob);
        if (!profile.empty()) slots = std::min<int64_t>(slots, static_cast<int64_t>(profile.size()));
    }
    impl->config.expert_cache_slots = static_cast<int>(slots);

    if (slots > 0) {
        std::string cerr;
        if (!impl->xcache.open(slots, impl->g.n_layers, impl->g.n_expert,
                               static_cast<int64_t>(kernels::cpu::expert_layout().max_blob), cerr)) {
            error_msg = "failed to open expert cache: " + cerr;
            return nullptr;
        }
    }

    // Expert residency table
    const size_t total_experts = static_cast<size_t>(impl->g.n_layers * impl->g.n_expert);
    impl->host_res.assign(total_experts, core::kNotResident);

    if (slots > 0 && !profile.empty()) {
        int64_t fill = std::min<int64_t>(slots, static_cast<int64_t>(profile.size()));
        for (int64_t i = 0; i < fill; ++i) {
            int32_t l = profile[i].first;
            int32_t e = profile[i].second;
            if (l >= 0 && l < impl->g.n_layers && e >= 0 && e < impl->g.n_expert) {
                impl->host_res[static_cast<size_t>(l * impl->g.n_expert + e)] = static_cast<int32_t>(i);
            }
        }
    }

    if (cudaMalloc(&impl->d_res, total_experts * sizeof(int32_t)) != cudaSuccess ||
        cudaMemcpy(impl->d_res, impl->host_res.data(), total_experts * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        error_msg = "failed to allocate device expert residency table";
        return nullptr;
    }

    // Pin RAM expert complement (all remaining experts resident in RAM)
    if (impl->config.resident_budget_gib > 0) {
        uint64_t budget = static_cast<uint64_t>(impl->config.resident_budget_gib) << 30;
        if (!impl->src.pin_cache_complement(impl->xcache, error_msg, true, {}, -1, 4ull << 30, budget,
                                            profile.empty() ? nullptr : &profile)) {
            // soft fallback: try budget what fits
            error_msg.clear();
            impl->src.pin_cache_complement(impl->xcache, error_msg, true, {}, -1, 4ull << 30,
                                           core::FileExpertSource::kResidentWhatFits,
                                           profile.empty() ? nullptr : &profile);
        }
    }

    // Setup dummy session state to initialize Verifier & MTP
    core::SessionState ss_temp;
    void* sbuf_temp = nullptr;
    const uint64_t sbytes = core::session_bytes(impl->g, impl->config.max_context, impl->K);
    if (cudaMalloc(&sbuf_temp, sbytes) != cudaSuccess ||
        core::session_init(impl->g, impl->config.max_context, impl->K, sbuf_temp, ss_temp, 0, -1) == 0) {
        error_msg = "failed to allocate initialization session buffer";
        return nullptr;
    }

    // MTP Drafter
    bool use_mtp = !paths.mtp_dir.empty() && std::filesystem::exists(paths.mtp_dir);
    if (use_mtp) {
        if (!impl->mtp.load(paths.mtp_dir, impl->g, ss_temp, impl->config.spec, error_msg)) {
            error_msg = "failed to load MTP draft layer: " + error_msg;
            cudaFree(sbuf_temp);
            return nullptr;
        }
    }

    // Verifier
    core::VerifyHits vh;
    vh.d_res = impl->d_res;
    vh.h_res = impl->host_res.data();
    vh.cache_base = slots > 0 ? static_cast<const uint8_t*>(impl->xcache.device_slot(0)) : nullptr;
    vh.n_slots = slots;
    vh.blob = static_cast<int64_t>(kernels::cpu::expert_layout().max_blob);

    if (!impl->ver.init(impl->wt, impl->g, ss_temp, vh, nullptr, impl->config.spec, error_msg)) {
        error_msg = "Verifier::init failed: " + error_msg;
        cudaFree(sbuf_temp);
        return nullptr;
    }

    if (use_mtp) {
        if (!impl->mtp.bind(impl->wt, nullptr, impl->ver.final_R_all(), error_msg)) {
            error_msg = "MTP bind failed: " + error_msg;
            cudaFree(sbuf_temp);
            return nullptr;
        }
    }

    cudaFree(sbuf_temp);

    // Setup Drive dispatch
    impl->drive.d.pool = impl->pool.get();
    impl->drive.d.src = impl->srcp;
    impl->drive.d.cache = slots > 0 ? &impl->xcache : nullptr;
    impl->drive.d.host_res = impl->host_res.data();

    // Load tokenizer
    if (!paths.tokenizer_dir.empty() && std::filesystem::exists(paths.tokenizer_dir)) {
        std::string tok_err;
        model->tokenizer_.load(paths.tokenizer_dir, tok_err);
    }

    model->ready_ = true;
    return model;
}

std::unique_ptr<GuildSession> GuildModel::create_session(int64_t max_context) {
    if (!ready_) return nullptr;
    int64_t ctx = max_context > 0 ? max_context : config_.max_context;
    return std::make_unique<GuildSession>(this, ctx);
}

} // namespace guild::runtime

#endif // GUILD_ENABLE_CUDA || GUILD_ENABLE_HIP
