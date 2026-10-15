// src/model/runner.cpp - Modular Model Architecture Runners & Registry
#include "guild/model/runner.hpp"
#include "guild/core/validation.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace guild::model {
namespace {

std::string to_lower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool validate_embedding_and_head(const GgufModel& artifact,
                                 const RunnerTensorMap& tmap,
                                 ModelDescriptor& desc,
                                 std::string& err) {
    const auto* embedding = artifact.find(tmap.embedding_tensor);
    const auto* head = artifact.find(tmap.output_tensor);
    if (!embedding || !head || embedding->shape.size() != 2 || head->shape.size() != 2 ||
        embedding->shape != head->shape || embedding->shape[0] != uint64_t(desc.attn.n_embd)) {
        err = "model load: incompatible embedding/output tensor dimensions";
        return false;
    }
    desc.attn.vocab_size = int64_t(embedding->shape[1]);
    return true;
}

// -----------------------------------------------------------------------------
// Built-in Runner 1: Qwen4ExpRunner (Qwen3.8-Flash-Next hybrid GDN + QSA + PLE)
// -----------------------------------------------------------------------------
class Qwen4ExpRunner final : public IModelRunner {
public:
    Qwen4ExpRunner() {
        required_ops_ = {
            GUILD_OP_EMBEDDING_GATHER,
            GUILD_OP_QUANTIZE_ACT,
            GUILD_OP_QUANTIZED_GEMV,
            GUILD_OP_QUANTIZED_GEMM,
            GUILD_OP_MOE_ROUTER_TOPK,
            GUILD_OP_MOE_DISPATCH,
            GUILD_OP_BATCHED_EXPERT_EXEC,
            GUILD_OP_SHARED_EXPERT,
            GUILD_OP_MOE_COMBINE,
            GUILD_OP_RMS_NORM,
            GUILD_OP_GATED_RESIDUAL,
            GUILD_OP_ROPE_APPLY,
            GUILD_OP_KV_APPEND,
            GUILD_OP_KV_GATHER,
            GUILD_OP_QSA_INDEXER_SELECT,
            GUILD_OP_ATTENTION_DECODE,
            GUILD_OP_GDN_CONV_L2,
            GUILD_OP_GDN_AB_PROJ,
            GUILD_OP_GDN_RECURRENCE_STEP,
            GUILD_OP_GDN_OUT_NORM,
            GUILD_OP_PLE_COMPUTE,
            GUILD_OP_SPEC_VERIFY,
            GUILD_OP_FINITE_CHECK
        };
    }

    const char* runner_id() const override { return "runner.qwen4exp.v1"; }
    const char* package_id() const override { return "builtin.guild-kernels-qwen"; }
    const char* arch_id() const override { return "qwen4exp"; }
    uint32_t runner_version() const override { return GUILD_KERNEL_ABI_VERSION_V1; }
    ModelArchetype archetype() const override { return ModelArchetype::Qwen4Exp; }

    bool matches_arch_name(const std::string& name) const override {
        const std::string s = to_lower_str(name);
        return s == "qwen4exp" || s == "qwen3.8-flash-next";
    }

    bool matches_gguf(const GgufFile& gguf) const override {
        if (const MetaValue* arch = gguf.get("general.architecture")) {
            if (arch->type == MetaType::STRING) return matches_arch_name(arch->s);
        }
        return gguf.get("qwen4exp.expert_count") != nullptr;
    }

    const RunnerTensorMap& tensor_map() const override { return tmap_; }

    bool validate_and_describe(const GgufModel& artifact,
                               ModelDescriptor& out_desc,
                               std::string& err) const override {
        const GgufFile& g = artifact.meta();
        const std::string arch_err = guild::check_architecture(g);
        if (!arch_err.empty()) {
            err = arch_err;
            return false;
        }
        if (!ArchetypeRegistry::instance().describe_gguf(g, out_desc, err)) return false;
        return validate_embedding_and_head(artifact, tmap_, out_desc, err);
    }

    bool configure_geometry(const ModelDescriptor& desc,
                            core::ModelGeometry& out_geom,
                            int64_t& out_k_active,
                            std::string& err) const override {
        if (desc.attn.n_embd <= 0 || desc.attn.n_layers <= 0 || desc.moe.n_routed_experts <= 0) {
            err = "qwen4exp runner: invalid model descriptor dimensions";
            return false;
        }
        out_geom.n_embd = desc.attn.n_embd;
        out_geom.n_layers = desc.attn.n_layers;
        out_geom.qsa_interval = desc.attn.full_attn_interval > 0 ? desc.attn.full_attn_interval : 4;
        out_geom.n_head = desc.attn.n_heads;
        out_geom.n_head_kv = desc.attn.n_kv_heads;
        out_geom.head_dim = desc.attn.head_dim;
        out_geom.n_expert = desc.moe.n_routed_experts;
        out_geom.n_ff = desc.moe.expert_dim_ff;
        out_geom.sync_qwen4exp();
        out_geom.attn_mechanism = AttentionMechanism::IndexedSparse;
        out_k_active = desc.moe.k_active_experts > 0 ? desc.moe.k_active_experts : 10;
        err.clear();
        return true;
    }

    bool configure_weight_skip_set(bool native_pack,
                                   bool has_shards,
                                   const std::string& /*pack_dir*/,
                                   std::set<std::string>& inout_skip,
                                   std::string& err) const override {
        inout_skip.insert(tmap_.output_tensor);
        if (has_shards) {
            if (!native_pack) {
                inout_skip.erase(tmap_.ple_key_tensor);
            }
            if (native_pack) {
                inout_skip.insert(tmap_.embedding_tensor);
            }
        } else {
            inout_skip.insert(tmap_.embedding_tensor);
        }
        err.clear();
        return true;
    }

    bool build_execution_plan(const ModelDescriptor& /*desc*/,
                              const core::ModelGeometry& /*geom*/,
                              const memory::ExecutionPlan& mem_plan,
                              RunnerExecutionPlan& out_exec,
                              std::string& err) const override {
        out_exec = {};
        out_exec.runner_id = runner_id();
        out_exec.arch_id = arch_id();
        out_exec.prefill_strategy = PrefillStrategy::DedicatedChunkedGemm;
        out_exec.requires_ple = true;
        out_exec.supports_mtp = true;
        out_exec.required_core_caps = GUILD_CORE_CAP_INDEXED_QSA_ATTN |
                                      GUILD_CORE_CAP_CUDA_GRAPH_VERIFY |
                                      GUILD_CORE_CAP_PLE_NGRAM;
        if (mem_plan.kv_mode == memory::KvMode::HostOnly) {
            out_exec.required_core_caps |= GUILD_CORE_CAP_HOST_ONLY_KV;
        }
        out_exec.required_ops = required_ops_;
        err.clear();
        return true;
    }

    bool validate_numerical(const float* values,
                            size_t count,
                            const std::string& label,
                            std::string& err) const override {
        return core::validate_finite(values, count, label, err);
    }

    guild_runner_desc_v1_t abi_descriptor() const override {
        guild_runner_desc_v1_t d{};
        d.struct_size = sizeof(d);
        d.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        d.arch_id = arch_id();
        d.runner_id = runner_id();
        d.package_id = package_id();
        d.runner_version = runner_version();
        d.required_core_caps = GUILD_CORE_CAP_INDEXED_QSA_ATTN | GUILD_CORE_CAP_CUDA_GRAPH_VERIFY | GUILD_CORE_CAP_PLE_NGRAM;
        d.required_op_kinds = required_ops_.data();
        d.n_required_op_kinds = required_ops_.size();
        return d;
    }

private:
    RunnerTensorMap tmap_{};
    std::vector<guild_op_kind_t> required_ops_;
};

// -----------------------------------------------------------------------------
// Built-in Runner 2: Qwen35MoERunner (Ornith-1.5-35B hybrid GDN + Dense QSA)
// -----------------------------------------------------------------------------
class Qwen35MoERunner final : public IModelRunner {
public:
    Qwen35MoERunner() {
        required_ops_ = {
            GUILD_OP_EMBEDDING_GATHER,
            GUILD_OP_QUANTIZE_ACT,
            GUILD_OP_QUANTIZED_GEMV,
            GUILD_OP_QUANTIZED_GEMM,
            GUILD_OP_MOE_ROUTER_TOPK,
            GUILD_OP_MOE_DISPATCH,
            GUILD_OP_BATCHED_EXPERT_EXEC,
            GUILD_OP_SHARED_EXPERT,
            GUILD_OP_MOE_COMBINE,
            GUILD_OP_RMS_NORM,
            GUILD_OP_ROPE_APPLY,
            GUILD_OP_KV_APPEND,
            GUILD_OP_ATTENTION_DECODE,
            GUILD_OP_GDN_CONV_L2,
            GUILD_OP_GDN_AB_PROJ,
            GUILD_OP_GDN_RECURRENCE_STEP,
            GUILD_OP_GDN_OUT_NORM,
            GUILD_OP_SPEC_VERIFY,
            GUILD_OP_FINITE_CHECK
        };
    }

    const char* runner_id() const override { return "runner.qwen35moe.v1"; }
    const char* package_id() const override { return "builtin.guild-kernels-ornith"; }
    const char* arch_id() const override { return "qwen35moe"; }
    uint32_t runner_version() const override { return GUILD_KERNEL_ABI_VERSION_V1; }
    ModelArchetype archetype() const override { return ModelArchetype::Qwen35MoE; }

    bool matches_arch_name(const std::string& name) const override {
        const std::string s = to_lower_str(name);
        return s == "qwen35moe" || s == "qwen2moe" || s == "ornith";
    }

    bool matches_gguf(const GgufFile& gguf) const override {
        if (const MetaValue* arch = gguf.get("general.architecture")) {
            if (arch->type == MetaType::STRING) return matches_arch_name(arch->s);
        }
        return gguf.get("qwen35moe.expert_count") != nullptr || gguf.get("qwen2moe.expert_count") != nullptr;
    }

    const RunnerTensorMap& tensor_map() const override { return tmap_; }

    bool validate_and_describe(const GgufModel& artifact,
                               ModelDescriptor& out_desc,
                               std::string& err) const override {
        const GgufFile& g = artifact.meta();
        const std::string arch_err = guild::check_architecture(g);
        if (!arch_err.empty()) {
            err = arch_err;
            return false;
        }
        if (!ArchetypeRegistry::instance().describe_gguf(g, out_desc, err)) return false;
        return validate_embedding_and_head(artifact, tmap_, out_desc, err);
    }

    bool configure_geometry(const ModelDescriptor& desc,
                            core::ModelGeometry& out_geom,
                            int64_t& out_k_active,
                            std::string& err) const override {
        if (desc.attn.n_embd <= 0 || desc.attn.n_layers <= 0 || desc.moe.n_routed_experts <= 0) {
            err = "qwen35moe runner: invalid model descriptor dimensions";
            return false;
        }
        out_geom.n_embd = desc.attn.n_embd;
        out_geom.n_layers = desc.attn.n_layers;
        out_geom.qsa_interval = desc.attn.full_attn_interval > 0 ? desc.attn.full_attn_interval : 4;
        out_geom.n_head = desc.attn.n_heads;
        out_geom.n_head_kv = desc.attn.n_kv_heads;
        out_geom.head_dim = desc.attn.head_dim;
        out_geom.n_expert = desc.moe.n_routed_experts;
        out_geom.n_ff = desc.moe.expert_dim_ff;
        out_geom.ssm_state_size = 128;
        out_geom.ssm_k_heads = 16;
        out_geom.ssm_v_heads = 32;
        out_geom.ssm_d_conv = 4;
        out_geom.ssm_conv_channels = 8192;
        out_geom.ssm_value_dim = 4096;
        out_geom.idx_q_heads = 0;
        out_geom.idx_key_dim = 0;
        out_geom.hc = 1;
        out_geom.hc_lr = 0;
        out_geom.gdn_gate_silu = true;
        out_geom.attn_mechanism = AttentionMechanism::DenseCausal;
        out_k_active = desc.moe.k_active_experts > 0 ? desc.moe.k_active_experts : 8;
        err.clear();
        return true;
    }

    bool configure_weight_skip_set(bool native_pack,
                                   bool has_shards,
                                   const std::string& /*pack_dir*/,
                                   std::set<std::string>& inout_skip,
                                   std::string& err) const override {
        inout_skip.insert(tmap_.output_tensor);
        if (!has_shards || native_pack) {
            inout_skip.insert(tmap_.embedding_tensor);
        }
        err.clear();
        return true;
    }

    bool build_execution_plan(const ModelDescriptor& /*desc*/,
                              const core::ModelGeometry& geom,
                              const memory::ExecutionPlan& mem_plan,
                              RunnerExecutionPlan& out_exec,
                              std::string& err) const override {
        out_exec = {};
        out_exec.runner_id = runner_id();
        out_exec.arch_id = arch_id();
        const bool batched_ok = (geom.hc == 1 && geom.ssm_state_size == 128 &&
                                 geom.ssm_d_conv == 4 && geom.head_dim == 256 &&
                                 mem_plan.kv_format == memory::KvPrecision::FP16);
        out_exec.prefill_strategy = batched_ok ? PrefillStrategy::LayerWiseBatched
                                               : PrefillStrategy::SequentialVerifier;
        out_exec.requires_ple = false;
        out_exec.supports_mtp = true;
        out_exec.required_core_caps = GUILD_CORE_CAP_DENSE_CAUSAL_ATTN |
                                      GUILD_CORE_CAP_CUDA_GRAPH_VERIFY;
        if (batched_ok) out_exec.required_core_caps |= GUILD_CORE_CAP_BATCHED_PREFILL;
        if (mem_plan.kv_mode == memory::KvMode::HostOnly) {
            out_exec.required_core_caps |= GUILD_CORE_CAP_HOST_ONLY_KV;
        }
        out_exec.required_ops = required_ops_;
        err.clear();
        return true;
    }

    bool validate_numerical(const float* values,
                            size_t count,
                            const std::string& label,
                            std::string& err) const override {
        return core::validate_finite(values, count, label, err);
    }

    guild_runner_desc_v1_t abi_descriptor() const override {
        guild_runner_desc_v1_t d{};
        d.struct_size = sizeof(d);
        d.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        d.arch_id = arch_id();
        d.runner_id = runner_id();
        d.package_id = package_id();
        d.runner_version = runner_version();
        d.required_core_caps = GUILD_CORE_CAP_DENSE_CAUSAL_ATTN | GUILD_CORE_CAP_CUDA_GRAPH_VERIFY | GUILD_CORE_CAP_BATCHED_PREFILL;
        d.required_op_kinds = required_ops_.data();
        d.n_required_op_kinds = required_ops_.size();
        return d;
    }

private:
    RunnerTensorMap tmap_{};
    std::vector<guild_op_kind_t> required_ops_;
};

// -----------------------------------------------------------------------------
// Adapter for C ABI v1 Registered Runners
// -----------------------------------------------------------------------------
class AbiModelRunnerAdapter final : public IModelRunner {
public:
    explicit AbiModelRunnerAdapter(const guild_runner_desc_v1_t& desc)
        : arch_id_(desc.arch_id),
          runner_id_(desc.runner_id),
          package_id_(desc.package_id),
          version_(desc.runner_version),
          core_caps_(desc.required_core_caps),
          session_alloc_(desc.session_alloc),
          session_free_(desc.session_free) {
        if (desc.required_op_kinds && desc.n_required_op_kinds > 0) {
            required_ops_.assign(desc.required_op_kinds, desc.required_op_kinds + desc.n_required_op_kinds);
        }
        if (desc.required_package_deps && desc.n_required_package_deps > 0) {
            for (size_t i = 0; i < desc.n_required_package_deps; ++i) {
                if (desc.required_package_deps[i]) pkg_deps_.emplace_back(desc.required_package_deps[i]);
            }
            for (const auto& s : pkg_deps_) pkg_dep_ptrs_.push_back(s.c_str());
        }
    }

    const char* runner_id() const override { return runner_id_.c_str(); }
    const char* package_id() const override { return package_id_.c_str(); }
    const char* arch_id() const override { return arch_id_.c_str(); }
    uint32_t runner_version() const override { return version_; }
    ModelArchetype archetype() const override { return archetype_from_string(arch_id_); }

    bool matches_arch_name(const std::string& name) const override {
        return to_lower_str(name) == to_lower_str(arch_id_);
    }

    bool matches_gguf(const GgufFile& gguf) const override {
        if (const MetaValue* arch = gguf.get("general.architecture")) {
            if (arch->type == MetaType::STRING) return matches_arch_name(arch->s);
        }
        return false;
    }

    const RunnerTensorMap& tensor_map() const override { return tmap_; }

    bool validate_and_describe(const GgufModel& artifact,
                               ModelDescriptor& out_desc,
                               std::string& err) const override {
        if (!ArchetypeRegistry::instance().describe_gguf(artifact.meta(), out_desc, err)) {
            out_desc.arch_name = arch_id_;
            out_desc.archetype = archetype();
        }
        return validate_embedding_and_head(artifact, tmap_, out_desc, err);
    }

    bool configure_geometry(const ModelDescriptor& desc,
                            core::ModelGeometry& out_geom,
                            int64_t& out_k_active,
                            std::string& err) const override {
        if (desc.attn.n_embd <= 0 || desc.attn.n_layers <= 0) {
            err = "ABI runner: invalid model descriptor dimensions";
            return false;
        }
        out_geom.apply_descriptor(desc);
        out_k_active = desc.moe.k_active_experts > 0 ? desc.moe.k_active_experts : 8;
        err.clear();
        return true;
    }

    bool configure_weight_skip_set(bool native_pack,
                                   bool has_shards,
                                   const std::string& /*pack_dir*/,
                                   std::set<std::string>& inout_skip,
                                   std::string& err) const override {
        inout_skip.insert(tmap_.output_tensor);
        if (!has_shards || native_pack) inout_skip.insert(tmap_.embedding_tensor);
        err.clear();
        return true;
    }

    bool build_execution_plan(const ModelDescriptor& /*desc*/,
                              const core::ModelGeometry& /*geom*/,
                              const memory::ExecutionPlan& /*mem_plan*/,
                              RunnerExecutionPlan& out_exec,
                              std::string& err) const override {
        out_exec = {};
        out_exec.runner_id = runner_id_;
        out_exec.arch_id = arch_id_;
        out_exec.prefill_strategy = PrefillStrategy::SequentialVerifier;
        out_exec.required_core_caps = core_caps_;
        out_exec.required_ops = required_ops_;
        err.clear();
        return true;
    }

    bool validate_numerical(const float* values,
                            size_t count,
                            const std::string& label,
                            std::string& err) const override {
        return core::validate_finite(values, count, label, err);
    }

    guild_runner_desc_v1_t abi_descriptor() const override {
        guild_runner_desc_v1_t d{};
        d.struct_size = sizeof(d);
        d.abi_version = GUILD_KERNEL_ABI_VERSION_V1;
        d.arch_id = arch_id_.c_str();
        d.runner_id = runner_id_.c_str();
        d.package_id = package_id_.c_str();
        d.runner_version = version_;
        d.required_core_caps = core_caps_;
        d.required_op_kinds = required_ops_.empty() ? nullptr : required_ops_.data();
        d.n_required_op_kinds = required_ops_.size();
        d.required_package_deps = pkg_dep_ptrs_.empty() ? nullptr : pkg_dep_ptrs_.data();
        d.n_required_package_deps = pkg_dep_ptrs_.size();
        d.session_alloc = session_alloc_;
        d.session_free = session_free_;
        return d;
    }

private:
    std::string arch_id_;
    std::string runner_id_;
    std::string package_id_;
    uint32_t version_;
    guild_core_cap_flags_t core_caps_;
    std::vector<guild_op_kind_t> required_ops_;
    std::vector<std::string> pkg_deps_;
    std::vector<const char*> pkg_dep_ptrs_;
    guild_runner_session_alloc_fn session_alloc_ = nullptr;
    guild_runner_session_free_fn session_free_ = nullptr;
    RunnerTensorMap tmap_{};
};

} // namespace

ModelRunnerRegistry::ModelRunnerRegistry() = default;
ModelRunnerRegistry::~ModelRunnerRegistry() = default;

ModelRunnerRegistry& ModelRunnerRegistry::instance() {
    static ModelRunnerRegistry reg;
    static std::once_flag once;
    std::call_once(once, [] { reg.register_builtin_runners(); });
    return reg;
}

void ModelRunnerRegistry::register_builtin_runners() {
    std::string ignore_err;
    register_runner(std::make_unique<Qwen4ExpRunner>(), ignore_err);
    register_runner(std::make_unique<Qwen35MoERunner>(), ignore_err);
}

guild_status_t ModelRunnerRegistry::register_runner(std::unique_ptr<IModelRunner> runner, std::string& err) {
    if (!runner || !runner->runner_id() || !runner->runner_id()[0] ||
        !runner->arch_id() || !runner->arch_id()[0] ||
        !runner->package_id() || !runner->package_id()[0]) {
        err = "runner descriptor is null or missing runner_id/arch_id/package_id";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& existing : runners_) {
        if (std::strcmp(existing->runner_id(), runner->runner_id()) == 0) {
            err = std::string("duplicate runner registration for runner_id: ") + runner->runner_id();
            return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
        }
        if (to_lower_str(existing->arch_id()) == to_lower_str(runner->arch_id())) {
            err = std::string("duplicate runner registration for architecture: ") + runner->arch_id();
            return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
        }
    }
    runners_.push_back(std::move(runner));
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t ModelRunnerRegistry::register_abi_runner(const guild_runner_desc_v1_t& desc, std::string& err) {
    if (desc.struct_size < sizeof(guild_runner_desc_v1_t)) {
        err = "runner ABI descriptor struct_size is too small";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (kernels::negotiate_abi_version(desc.abi_version, nullptr) != GUILD_STATUS_OK) {
        err = "runner ABI version is incompatible with Guild Core ABI v1";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (!desc.arch_id || !desc.arch_id[0] || !desc.runner_id || !desc.runner_id[0] ||
        !desc.package_id || !desc.package_id[0]) {
        err = "runner ABI descriptor missing arch_id, runner_id, or package_id";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < desc.n_required_op_kinds; ++i) {
        if (!desc.required_op_kinds || desc.required_op_kinds[i] <= GUILD_OP_INVALID ||
            desc.required_op_kinds[i] >= GUILD_OP_COUNT) {
            err = "runner ABI descriptor declares invalid required operation kind";
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
    }
    return register_runner(std::make_unique<AbiModelRunnerAdapter>(desc), err);
}

guild_status_t ModelRunnerRegistry::unregister_runner(const std::string& runner_id, std::string& err) {
    if (active_sessions() > 0) {
        err = "cannot unregister model runner '" + runner_id + "' while active sessions are running";
        return GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t before = runners_.size();
    runners_.erase(std::remove_if(runners_.begin(), runners_.end(), [&](const std::unique_ptr<IModelRunner>& r) {
        return runner_id == r->runner_id();
    }), runners_.end());
    if (runners_.size() == before) {
        err = "model runner not found: " + runner_id;
        return GUILD_STATUS_ERR_MISSING_RUNNER;
    }
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t ModelRunnerRegistry::resolve_for_gguf(const GgufFile& gguf,
                                                     const IModelRunner** out_runner,
                                                     std::string& err) const {
    if (out_runner) *out_runner = nullptr;
    const MetaValue* arch = gguf.get("general.architecture");
    if (!arch || arch->type != MetaType::STRING || arch->s.empty()) {
        err = "missing general.architecture";
        return GUILD_STATUS_ERR_MISSING_RUNNER;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : runners_) {
        if (r->matches_gguf(gguf)) {
            if (out_runner) *out_runner = r.get();
            err.clear();
            return GUILD_STATUS_OK;
        }
    }
    err = "architecture is '" + arch->s + "', no compatible model runner is registered";
    return GUILD_STATUS_ERR_MISSING_RUNNER;
}

guild_status_t ModelRunnerRegistry::resolve_by_arch(const std::string& arch_name,
                                                    const IModelRunner** out_runner,
                                                    std::string& err) const {
    if (out_runner) *out_runner = nullptr;
    if (arch_name.empty()) {
        err = "architecture name cannot be empty";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : runners_) {
        if (r->matches_arch_name(arch_name)) {
            if (out_runner) *out_runner = r.get();
            err.clear();
            return GUILD_STATUS_OK;
        }
    }
    err = "no model runner registered for architecture '" + arch_name + "'";
    return GUILD_STATUS_ERR_MISSING_RUNNER;
}

const IModelRunner* ModelRunnerRegistry::find_by_id(const std::string& runner_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : runners_) {
        if (runner_id == r->runner_id()) return r.get();
    }
    return nullptr;
}

guild_status_t ModelRunnerRegistry::validate_runner_requirements(const IModelRunner& runner,
                                                                 const ModelDescriptor& desc,
                                                                 const core::ModelGeometry& geom,
                                                                 const memory::ExecutionPlan& mem_plan,
                                                                 const kernels::KernelRegistry& kernel_reg,
                                                                 const guild_device_capabilities_v1_t& caps,
                                                                 RunnerExecutionPlan& out_exec,
                                                                 std::string& err) const {
    if (!runner.build_execution_plan(desc, geom, mem_plan, out_exec, err)) {
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }
    if ((out_exec.required_core_caps & caps.core_cap_flags) != out_exec.required_core_caps) {
        err = std::string("runner '") + runner.runner_id() + "' requires Guild Core capabilities not present on this host/device";
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }
    out_exec.resolved_kernel_ids.clear();
    out_exec.resolved_kernel_ids.reserve(out_exec.required_ops.size());
    for (guild_op_kind_t op : out_exec.required_ops) {
        kernels::KernelSelectionQuery q;
        q.op_kind = op;
        q.dim_n = geom.n_embd;
        q.dim_k = geom.n_ff;
        const guild_kernel_op_desc_v1_t* kdesc = nullptr;
        const guild_status_t st = kernel_reg.select(q, caps, &kdesc, err);
        if (st != GUILD_STATUS_OK || !kdesc) {
            return st != GUILD_STATUS_OK ? st : GUILD_STATUS_ERR_MISSING_KERNEL;
        }
        out_exec.resolved_kernel_ids.emplace_back(kdesc->impl_id);
    }
    err.clear();
    return GUILD_STATUS_OK;
}

std::vector<guild_runner_desc_v1_t> ModelRunnerRegistry::list_runners() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<guild_runner_desc_v1_t> out;
    out.reserve(runners_.size());
    for (const auto& r : runners_) out.push_back(r->abi_descriptor());
    return out;
}

} // namespace guild::model
