#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

#if defined(GUILD_ENABLE_CUDA) || defined(GUILD_ENABLE_HIP)

#include "model_impl.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include "guild/model/archetype.hpp"
#include "guild/kernels/iq_kernels.hpp"
#include "guild/kernels/verify_kernels.hpp"

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
    const model::ModelDescriptor& requested_desc,
    const memory::ExecutionPlan& plan,
    std::string& error_msg
) try {
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
    model::ModelDescriptor desc = requested_desc;
    {
        const guild::GgufModel artifact(shards);
        error_msg = guild::check_architecture(artifact.meta());
        if (!error_msg.empty() || !model::ArchetypeRegistry::instance().describe_gguf(artifact.meta(), desc, error_msg)) return nullptr;
        const auto* embedding = artifact.find("token_embd.weight");
        const auto* head = artifact.find("output.weight");
        if (!embedding || !head || embedding->shape.size() != 2 || head->shape.size() != 2 ||
            embedding->shape != head->shape || embedding->shape[0] != uint64_t(desc.attn.n_embd)) {
            error_msg = "model load: incompatible embedding/output tensor dimensions";
            return nullptr;
        }
        desc.attn.vocab_size = int64_t(embedding->shape[1]);
    }
    if (plan.context_length <= 0 || plan.context_length > desc.attn.context_length ||
        plan.mtp_spec_tokens < 0 || plan.mtp_spec_tokens > kernels::kVerifyMaxT) {
        error_msg = "model load: invalid context or verification capacity";
        return nullptr;
    }
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

    // Expert layout from pack directory
    if (!guild::kernels::cpu::expert_layout_load(paths.pack_dir, impl->g.n_layers, impl->g.n_expert, error_msg,
                                                 impl->g.n_embd, impl->g.n_ff)) {
        return nullptr;
    }
    const bool native_pack = guild::kernels::cpu::expert_layout().native;
    if ((!native_pack && !kernels::cpu::cpu_avx512_ok()) ||
        impl->g.n_ff > kernels::cpu::FF || impl->g.n_embd > kernels::cpu::H) {
        error_msg = "model load: expert geometry or CPU ISA exceeds this pool's supported capacities";
        return nullptr;
    }
    if (native_pack && plan.routed_experts_in_gpu > 0) {
        const auto& lay = guild::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < static_cast<int64_t>(lay.fmt.size()); ++l) {
            const auto& f = lay.fmt[static_cast<size_t>(l)];
            if (!guild::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                error_msg = "layer " + std::to_string(l) + "'s experts are not supported by native GPU kernels";
                return nullptr;
            }
        }
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
        skip.insert("output.weight");
        if (!native_pack) {
            skip.erase("blk.1.ple_key.weight");
        }
        if (native_pack && !guild::core::NativeDense::keep_unquantized_ple_key(paths.pack_dir, skip, error_msg)) {
            return nullptr;
        }
        if (native_pack) {
            skip.insert("token_embd.weight");
        }
    } else {
        skip.insert("output.weight");
        skip.insert("token_embd.weight");
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

    // Load expert profile if provided
    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!paths.expert_profile_path.empty()) {
        int64_t profile_slots = 0;
        if (!core::read_expert_profile(paths.expert_profile_path, impl->g.n_layers, impl->g.n_expert,
                                       profile, profile_slots, error_msg)) return nullptr;
    }

    // Expert cache in VRAM
    int64_t slots = impl->config.expert_cache_slots;
    if (slots < 0) {
        // Auto sizing from free VRAM
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t blob = static_cast<int64_t>(kernels::cpu::expert_layout().max_blob);
        int64_t reserve = (static_cast<int64_t>(impl->config.vram_reserve_mib) << 20) + (100LL << 20);
        slots = blob > 0 ? std::max<int64_t>(0, (static_cast<int64_t>(free_b) - reserve) / blob) : 0;
        if (!profile.empty()) slots = std::min<int64_t>(slots, static_cast<int64_t>(profile.size()));
    }
    impl->config.expert_cache_slots = static_cast<int>(slots);

    if (slots > 0) {
        if (profile.empty()) {
            error_msg = "model load: GPU expert slots require a validated expert profile";
            return nullptr;
        }
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
            const int32_t slot = impl->xcache.admit(l, e);
            const uint8_t* blob = impl->src.blob(l, e);
            const int64_t bytes = int64_t(kernels::cpu::expert_layout().blob_bytes(l));
            if (slot < 0 || !blob || !impl->xcache.fill_slot_blocking(slot, blob, error_msg, bytes) ||
                !impl->xcache.verify_slot(slot, blob, error_msg, bytes)) {
                if (error_msg.empty()) error_msg = "model load: GPU expert slot could not be filled and verified";
                return nullptr;
            }
            impl->host_res[size_t(l * impl->g.n_expert + e)] = slot;
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
            if (!impl->src.pin_cache_complement(impl->xcache, error_msg, true, {}, -1, 4ull << 30,
                                            core::FileExpertSource::kResidentWhatFits,
                                            profile.empty() ? nullptr : &profile)) return nullptr;
        }
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
    core::VerifyHits vh;
    vh.d_res = impl->d_res;
    vh.h_res = impl->host_res.data();
    vh.cache_base = slots > 0 ? static_cast<const uint8_t*>(impl->xcache.device_slot(0)) : nullptr;
    vh.n_slots = slots;
    vh.blob = static_cast<int64_t>(kernels::cpu::expert_layout().max_blob);

    impl->ver = std::make_unique<core::Verifier>();
    if (!impl->ver->init(impl->wt, impl->g, impl->ss, vh,
                        impl->native_head.loaded() ? &impl->native_head : nullptr,
                        impl->config.spec, error_msg)) {
        error_msg = "Verifier::init failed: " + error_msg;
        return nullptr;
    }

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
    bool native_gpu_supported = true;
    if (native_pack) {
        const auto& lay = kernels::cpu::expert_layout();
        for (size_t l = 0; l < lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[l];
            if (!guild::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                native_gpu_supported = false;
                break;
            }
        }
    }

    if (plan.routed_experts_in_gpu > 0 && native_gpu_supported) {
        impl->ver->set_pcie_mode(2);
        impl->drive.d.plan = impl->ver->plan_sink();
        impl->drive.d.pcie_num = static_cast<int>((native_pack ? 0.55 : 0.2) * static_cast<double>(impl->g.n_expert) + 0.5);
    } else {
        impl->ver->set_pcie_mode(0);
        impl->drive.d.plan = impl->ver->plan_sink();
        impl->drive.d.pcie_num = 0;
    }

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
