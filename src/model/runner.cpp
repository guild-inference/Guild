// src/model/runner.cpp - Modular Model Architecture Runners & Registry
#include "guild/model/runner.hpp"
#include "guild/abi/lifecycle.hpp"
#include "guild/core/validation.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cmath>
#include <cstdio>

#if !defined(GUILD_NO_BUILTIN_QWEN_RUNNERS)
extern "C" guild_status_t guild_register_bundled_qwen_runners(guild_core_host_t host, const guild_host_api_v1_t* api,
                                                              guild_error_info_t* e);
#endif

namespace guild::model {
namespace {

std::string to_lower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

guild_model_reader_v1_t gguf_file_reader(const GgufFile& gguf) {
    guild_model_reader_v1_t reader{};
    reader.struct_size = sizeof(reader);
    reader.user = const_cast<GgufFile*>(&gguf);
    reader.get_i64 = [](void* u, const char* key, int64_t* out) -> guild_status_t {
        if (!key || !out)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* v = static_cast<GgufFile*>(u)->get(key);
        if (!v || !v->is_num())
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out = static_cast<int64_t>(v->num());
        return GUILD_STATUS_OK;
    };
    reader.get_f64 = [](void* u, const char* key, double* out) -> guild_status_t {
        if (!key || !out)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* v = static_cast<GgufFile*>(u)->get(key);
        if (!v || !v->is_num())
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out = v->num();
        return GUILD_STATUS_OK;
    };
    reader.get_string = [](void* u, const char* key, char* out, size_t n) -> guild_status_t {
        if (!key || !out || !n)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* v = static_cast<GgufFile*>(u)->get(key);
        if (!v || v->type != MetaType::STRING || v->s.size() >= n)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        std::memcpy(out, v->s.c_str(), v->s.size() + 1);
        return GUILD_STATUS_OK;
    };
    reader.get_tensor = [](void*, const char*, guild_tensor_desc_t*) -> guild_status_t {
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    };
    return reader;
}

struct DescReaderCtx {
    const ModelDescriptor* desc = nullptr;
    std::string fallback_arch;
};

guild_model_reader_v1_t descriptor_model_reader(DescReaderCtx& ctx) {
    guild_model_reader_v1_t reader{};
    reader.struct_size = sizeof(reader);
    reader.user = &ctx;
    reader.get_string = [](void* u, const char* key, char* out, size_t n) -> guild_status_t {
        if (!key || !out || !n)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* c = static_cast<const DescReaderCtx*>(u);
        const std::string arch = c->desc->arch_name.empty() ? c->fallback_arch : c->desc->arch_name;
        if (std::strcmp(key, "general.architecture") == 0) {
            if (arch.empty() || arch.size() >= n)
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            std::memcpy(out, arch.c_str(), arch.size() + 1);
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(key, "general.name") == 0) {
            const std::string& nm = c->desc->name.empty() ? arch : c->desc->name;
            if (nm.empty() || nm.size() >= n)
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            std::memcpy(out, nm.c_str(), nm.size() + 1);
            return GUILD_STATUS_OK;
        }
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    };
    reader.get_i64 = [](void* u, const char* key, int64_t* out) -> guild_status_t {
        if (!key || !out)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* c = static_cast<const DescReaderCtx*>(u);
        const ModelDescriptor& d = *c->desc;
        const char* dot = std::strchr(key, '.');
        const char* sfx = dot ? (dot + 1) : key;
        if (std::strcmp(sfx, "block_count") == 0 && d.attn.n_layers > 0) {
            *out = d.attn.n_layers;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "embedding_length") == 0 && d.attn.n_embd > 0) {
            *out = d.attn.n_embd;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "expert_count") == 0 && d.moe.n_routed_experts > 0) {
            *out = d.moe.n_routed_experts;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "expert_used_count") == 0 && d.moe.k_active_experts > 0) {
            *out = d.moe.k_active_experts;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "attention.head_count") == 0 && d.attn.n_heads > 0) {
            *out = d.attn.n_heads;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "attention.head_count_kv") == 0 && d.attn.n_kv_heads > 0) {
            *out = d.attn.n_kv_heads;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "attention.key_length") == 0 && d.attn.head_dim > 0) {
            *out = d.attn.head_dim;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "context_length") == 0 && d.attn.context_length > 0) {
            *out = d.attn.context_length;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "expert_feed_forward_length") == 0 && d.moe.expert_dim_ff > 0) {
            *out = d.moe.expert_dim_ff;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "expert_shared_feed_forward_length") == 0 && d.moe.shared_dim_ff > 0) {
            *out = d.moe.shared_dim_ff;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "expert_shared_count") == 0 && d.moe.n_shared_experts > 0) {
            *out = d.moe.n_shared_experts;
            return GUILD_STATUS_OK;
        }
        if (std::strcmp(sfx, "full_attention_interval") == 0 && d.attn.full_attn_interval > 0) {
            *out = d.attn.full_attn_interval;
            return GUILD_STATUS_OK;
        }
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    };
    reader.get_f64 = [](void*, const char*, double*) -> guild_status_t { return GUILD_STATUS_ERR_INVALID_ARGUMENT; };
    reader.get_tensor = [](void* u, const char* key, guild_tensor_desc_t* out) -> guild_status_t {
        if (!key || !out || out->struct_size < sizeof(*out))
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* c = static_cast<const DescReaderCtx*>(u);
        if (std::strcmp(key, "token_embd.weight") == 0 || std::strcmp(key, "output.weight") == 0) {
            if (c->desc->attn.n_embd <= 0)
                return GUILD_STATUS_ERR_INVALID_TENSOR;
            *out = {};
            out->struct_size = sizeof(*out);
            out->n_dims = 2;
            out->dims[0] = c->desc->attn.n_embd;
            out->dims[1] = c->desc->attn.vocab_size > 0 ? c->desc->attn.vocab_size : 248320;
            return GUILD_STATUS_OK;
        }
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    };
    return reader;
}

guild_model_reader_v1_t model_reader(const GgufModel& model) {
    guild_model_reader_v1_t reader{};
    reader.struct_size = sizeof(reader);
    reader.user = const_cast<GgufModel*>(&model);
    reader.get_i64 = [](void* u, const char* key, int64_t* out) -> guild_status_t {
        if (!key || !out)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* v = static_cast<GgufModel*>(u)->meta().get(key);
        if (!v || !v->is_num())
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out = static_cast<int64_t>(v->num());
        return GUILD_STATUS_OK;
    };
    reader.get_f64 = [](void* u, const char* key, double* out) -> guild_status_t {
        if (!key || !out)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* v = static_cast<GgufModel*>(u)->meta().get(key);
        if (!v || !v->is_num())
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out = v->num();
        return GUILD_STATUS_OK;
    };
    reader.get_string = [](void* u, const char* key, char* out, size_t n) -> guild_status_t {
        if (!key || !out || !n)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* v = static_cast<GgufModel*>(u)->meta().get(key);
        if (!v || v->type != MetaType::STRING || v->s.size() >= n)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        std::memcpy(out, v->s.c_str(), v->s.size() + 1);
        return GUILD_STATUS_OK;
    };
    reader.get_tensor = [](void* u, const char* key, guild_tensor_desc_t* out) -> guild_status_t {
        if (!key || !out || out->struct_size < sizeof(*out))
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* t = static_cast<GgufModel*>(u)->find(key);
        if (!t || t->shape.empty() || t->shape.size() > GUILD_MAX_TENSOR_DIMS)
            return GUILD_STATUS_ERR_INVALID_TENSOR;
        *out = {};
        out->struct_size = sizeof(*out);
        out->dtype = t->type;
        out->n_dims = int32_t(t->shape.size());
        for (size_t i = 0; i < t->shape.size(); ++i)
            out->dims[i] = int64_t(t->shape[i]);
        return GUILD_STATUS_OK;
    };
    return reader;
}

bool import_runner_model(const guild_runner_model_v1_t& m, ModelDescriptor& d, std::string& err) {
    const auto terminated = [](const auto& s) { return std::memchr(s, 0, sizeof(s)) != nullptr; };
    if (m.struct_size < sizeof(m) || m.version != GUILD_RUNNER_MODEL_VERSION_V1 || !terminated(m.arch_id) ||
        !terminated(m.name) || !terminated(m.embedding_tensor) || !terminated(m.output_tensor) ||
        !terminated(m.output_norm_tensor) || !terminated(m.ple_table_tensor) || !terminated(m.ple_key_tensor) ||
        !terminated(m.layer_prefix) || m.n_embd <= 0 || m.n_embd > 1048576 || m.n_layers <= 0 || m.n_layers > 4096 ||
        m.n_heads <= 0 || m.n_heads > 4096 || m.n_kv_heads <= 0 || m.n_heads % m.n_kv_heads || m.head_dim <= 0 ||
        m.context_length <= 0 || m.vocab_size <= 0 || m.full_attention_interval <= 0 || m.n_experts <= 0 ||
        m.active_experts <= 0 || m.active_experts > m.n_experts || m.expert_ff <= 0 || m.residual_streams <= 0 ||
        m.residual_streams > 64 || m.ssm_state_size < 0 || m.ssm_state_size > 4096 || m.ssm_v_heads < 0 ||
        m.ssm_v_heads > 4096 || m.prefill_strategy > GUILD_RUNNER_PREFILL_LAYER_BATCHED ||
        !std::isfinite(m.rope_base) || m.rope_base <= 0 || !std::isfinite(m.rope_factor) || m.rope_factor <= 0) {
        err = "external runner: invalid model description or unsupported topology version";
        return false;
    }
    if (std::strcmp(m.embedding_tensor, "token_embd.weight") || std::strcmp(m.output_tensor, "output.weight") ||
        std::strcmp(m.output_norm_tensor, "output_norm.weight") || std::strcmp(m.layer_prefix, "blk.")) {
        err = "external runner: this execution contract requires canonical tensor role names";
        return false;
    }
    d.runner_model = m;
    d.arch_name = m.arch_id;
    d.name = m.name;
    d.archetype = archetype_from_string(m.arch_id);
    d.attn.n_embd = m.n_embd;
    d.attn.n_layers = m.n_layers;
    d.attn.n_heads = m.n_heads;
    d.attn.n_kv_heads = m.n_kv_heads;
    d.attn.head_dim = m.head_dim;
    d.attn.context_length = m.context_length;
    d.attn.vocab_size = m.vocab_size;
    d.attn.full_attn_interval = m.full_attention_interval;
    d.attn.pattern = m.full_attention_interval > 1 ? AttentionPattern::HybridGDN : AttentionPattern::Standard;
    d.attn.mechanism = m.indexed_attention ? AttentionMechanism::IndexedSparse : AttentionMechanism::DenseCausal;
    d.moe.n_routed_experts = m.n_experts;
    d.moe.k_active_experts = m.active_experts;
    d.moe.expert_dim_ff = m.expert_ff;
    d.moe.shared_dim_ff = m.shared_ff;
    d.moe.n_shared_experts = m.shared_experts;
    d.rope.freq_base = m.rope_base;
    d.rope.factor = m.rope_factor;
    d.rope.orig_ctx = m.rope_original_context;
    d.rope.type = m.rope_kind == 1 ? "linear" : m.rope_kind == 2 ? "yarn" : "none";
    return true;
}

bool validate_embedding_and_head(const GgufModel& artifact, const RunnerTensorMap& tmap, ModelDescriptor& desc,
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
// Adapter for C ABI v1 Registered Runners (Bundled and External)
// -----------------------------------------------------------------------------
class AbiModelRunnerAdapter final : public IModelRunner {
public:
    explicit AbiModelRunnerAdapter(const guild_runner_desc_v1_t& desc, std::shared_ptr<void> keepalive)
        : library_keepalive_(std::move(keepalive)),
          arch_id_(desc.arch_id),
          runner_id_(desc.runner_id),
          package_id_(desc.package_id),
          version_(desc.runner_version),
          core_caps_(desc.required_core_caps),
          session_alloc_(desc.session_alloc),
          session_free_(desc.session_free) {
        if (desc.next_ext) {
            uint32_t size = 0;
            std::memcpy(&size, desc.next_ext, sizeof(size));
            if (size >= sizeof(guild_runner_model_ext_v1_t)) {
                guild_runner_model_ext_v1_t candidate{};
                std::memcpy(&candidate, desc.next_ext, sizeof(candidate));
                if (candidate.tag == GUILD_RUNNER_MODEL_TAG_V1 && candidate.version == GUILD_RUNNER_MODEL_VERSION_V1)
                    extension_ = candidate;
            }
        }
        if (desc.required_op_kinds && desc.n_required_op_kinds > 0) {
            required_ops_.assign(desc.required_op_kinds, desc.required_op_kinds + desc.n_required_op_kinds);
        }
        if (desc.required_package_deps && desc.n_required_package_deps > 0) {
            for (size_t i = 0; i < desc.n_required_package_deps; ++i) {
                if (desc.required_package_deps[i])
                    pkg_deps_.emplace_back(desc.required_package_deps[i]);
            }
            for (const auto& s : pkg_deps_)
                pkg_dep_ptrs_.push_back(s.c_str());
        }
    }

    const char* runner_id() const override {
        return runner_id_.c_str();
    }
    const char* package_id() const override {
        return package_id_.c_str();
    }
    const char* arch_id() const override {
        return arch_id_.c_str();
    }
    uint32_t runner_version() const override {
        return version_;
    }
    int32_t priority() const override {
        return extension_ ? extension_->priority : 0;
    }
    ModelArchetype archetype() const override {
        return archetype_from_string(arch_id_);
    }

    bool matches_arch_name(const std::string& name) const override {
        const std::string lower_name = to_lower_str(name);
        if (lower_name == to_lower_str(arch_id_))
            return true;
        if (extension_ && extension_->recognize) {
            guild_model_reader_v1_t reader{};
            reader.struct_size = sizeof(reader);
            reader.user = const_cast<std::string*>(&lower_name);
            reader.get_string = [](void* u, const char* k, char* out, size_t n) -> guild_status_t {
                if (!k || !out || !n || std::strcmp(k, "general.architecture") != 0)
                    return GUILD_STATUS_ERR_INVALID_ARGUMENT;
                const auto* s = static_cast<const std::string*>(u);
                if (s->empty() || s->size() >= n)
                    return GUILD_STATUS_ERR_INVALID_ARGUMENT;
                std::memcpy(out, s->c_str(), s->size() + 1);
                return GUILD_STATUS_OK;
            };
            reader.get_i64 = [](void*, const char*, int64_t*) -> guild_status_t {
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            };
            reader.get_f64 = [](void*, const char*, double*) -> guild_status_t {
                return GUILD_STATUS_ERR_INVALID_ARGUMENT;
            };
            reader.get_tensor = [](void*, const char*, guild_tensor_desc_t*) -> guild_status_t {
                return GUILD_STATUS_ERR_INVALID_TENSOR;
            };
            uint32_t recognized = 0;
            if (extension_->recognize(&reader, &recognized) == GUILD_STATUS_OK && recognized != 0)
                return true;
        }
        return false;
    }

    bool matches_gguf(const GgufFile& gguf) const override {
        if (extension_ && extension_->recognize) {
            auto reader = gguf_file_reader(gguf);
            uint32_t recognized = 0;
            return extension_->recognize(&reader, &recognized) == GUILD_STATUS_OK && recognized != 0;
        }
        if (const MetaValue* arch = gguf.get("general.architecture")) {
            if (arch->type == MetaType::STRING)
                return matches_arch_name(arch->s);
        }
        return false;
    }

    const RunnerTensorMap& tensor_map() const override {
        return tmap_;
    }

    bool validate_and_describe(const GgufModel& artifact, ModelDescriptor& out_desc, std::string& err) const override {
        if (extension_) {
            auto reader = model_reader(artifact);
            uint32_t recognized = 0;
            guild_error_info_t error{};
            guild_runner_model_v1_t m{};
            m.struct_size = sizeof(m);
            m.version = GUILD_RUNNER_MODEL_VERSION_V1;
            if (extension_->recognize(&reader, &recognized) != GUILD_STATUS_OK || !recognized ||
                extension_->describe(&reader, &m, &error) != GUILD_STATUS_OK) {
                err = error.message[0] ? error.message : "external runner rejected GGUF";
                return false;
            }
            if (!import_runner_model(m, out_desc, err))
                return false;
            out_desc.file_path = artifact.meta().path();
            // import_runner_model already requires canonical tensor roles. Never
            // mutate registry-shared strings while other models read tensor_map().
            return validate_embedding_and_head(artifact, tmap_, out_desc, err);
        }
        if (!ArchetypeRegistry::instance().describe_gguf(artifact.meta(), out_desc, err)) {
            out_desc.arch_name = arch_id_;
            out_desc.archetype = archetype();
        }
        return validate_embedding_and_head(artifact, tmap_, out_desc, err);
    }

    bool configure_geometry(const ModelDescriptor& desc, core::ModelGeometry& out_geom, int64_t& out_k_active,
                            std::string& err) const override {
        if (extension_) {
            guild_runner_model_v1_t m{};
            if (desc.runner_model) {
                m = *desc.runner_model;
            } else {
                if (desc.attn.n_embd <= 0 || desc.attn.n_layers <= 0 || desc.moe.n_routed_experts <= 0) {
                    err = arch_id_ + " runner: invalid model descriptor dimensions";
                    return false;
                }
                DescReaderCtx ctx{&desc, arch_id_};
                auto reader = descriptor_model_reader(ctx);
                guild_error_info_t abi_err{};
                m.struct_size = sizeof(m);
                m.version = GUILD_RUNNER_MODEL_VERSION_V1;
                if (extension_->describe(&reader, &m, &abi_err) != GUILD_STATUS_OK) {
                    err = abi_err.message[0] ? abi_err.message : "external runner model description missing";
                    return false;
                }
            }
            out_geom.n_embd = m.n_embd;
            out_geom.n_layers = m.n_layers;
            out_geom.qsa_interval = m.full_attention_interval;
            out_geom.n_head = m.n_heads;
            out_geom.n_head_kv = m.n_kv_heads;
            out_geom.head_dim = m.head_dim;
            out_geom.n_expert = m.n_experts;
            out_geom.n_ff = m.expert_ff;
            out_geom.ssm_state_size = m.ssm_state_size;
            out_geom.ssm_k_heads = m.ssm_k_heads;
            out_geom.ssm_v_heads = m.ssm_v_heads;
            out_geom.ssm_d_conv = m.ssm_d_conv;
            out_geom.ssm_conv_channels = m.ssm_conv_channels;
            out_geom.ssm_value_dim = m.ssm_value_dim;
            out_geom.idx_q_heads = m.index_heads;
            out_geom.idx_key_dim = m.index_dim;
            out_geom.hc = m.residual_streams;
            out_geom.hc_lr = m.residual_rank;
            out_geom.gdn_gate_silu = m.gdn_gate_silu != 0;
            out_geom.attn_mechanism =
                m.indexed_attention ? AttentionMechanism::IndexedSparse : AttentionMechanism::DenseCausal;
            if (m.indexed_attention)
                out_geom.sync_qwen4exp();
            out_k_active = m.active_experts;
            err.clear();
            return true;
        }
        if (desc.attn.n_embd <= 0 || desc.attn.n_layers <= 0) {
            err = "ABI runner: invalid model descriptor dimensions";
            return false;
        }
        out_geom.apply_descriptor(desc);
        out_k_active = desc.moe.k_active_experts > 0 ? desc.moe.k_active_experts : 8;
        err.clear();
        return true;
    }

    bool configure_weight_skip_set(bool native_pack, bool has_shards, const std::string& /*pack_dir*/,
                                   std::set<std::string>& inout_skip, std::string& err) const override {
        inout_skip.insert(tmap_.output_tensor);
        if (has_shards) {
            if (!native_pack && !tmap_.ple_key_tensor.empty()) {
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

    bool build_execution_plan(const ModelDescriptor& desc, const core::ModelGeometry& geom,
                              const memory::ExecutionPlan& mem_plan, RunnerExecutionPlan& out_exec,
                              std::string& err) const override {
        out_exec = {};
        out_exec.runner_id = runner_id_;
        out_exec.arch_id = arch_id_;
        out_exec.prefill_strategy = PrefillStrategy::SequentialVerifier;
        out_exec.required_core_caps = core_caps_;
        out_exec.required_ops = required_ops_;
        if (extension_) {
            guild_runner_model_v1_t m{};
            bool have_model = false;
            if (desc.runner_model) {
                m = *desc.runner_model;
                have_model = true;
            } else if (desc.attn.n_embd > 0 && desc.attn.n_layers > 0 && desc.moe.n_routed_experts > 0) {
                DescReaderCtx ctx{&desc, arch_id_};
                auto reader = descriptor_model_reader(ctx);
                m.struct_size = sizeof(m);
                m.version = GUILD_RUNNER_MODEL_VERSION_V1;
                have_model = (extension_->describe(&reader, &m, nullptr) == GUILD_STATUS_OK);
            }
            if (have_model) {
                out_exec.prefill_strategy = static_cast<PrefillStrategy>(m.prefill_strategy);
                if (out_exec.prefill_strategy == PrefillStrategy::LayerWiseBatched) {
                    const bool batched_ok = (geom.hc == 1 && geom.ssm_state_size == 128 && geom.ssm_d_conv == 4 &&
                                             geom.head_dim == 256 && mem_plan.kv_format == memory::KvPrecision::FP16);
                    if (!batched_ok) {
                        out_exec.prefill_strategy = PrefillStrategy::SequentialVerifier;
                        out_exec.required_core_caps &= ~uint32_t(GUILD_CORE_CAP_BATCHED_PREFILL);
                    } else {
                        out_exec.required_core_caps |= GUILD_CORE_CAP_BATCHED_PREFILL;
                    }
                }
                out_exec.requires_ple = m.requires_ple != 0;
                out_exec.supports_mtp = m.supports_mtp != 0;
            }
        }
        if (mem_plan.kv_mode == memory::KvMode::HostOnly) {
            out_exec.required_core_caps |= GUILD_CORE_CAP_HOST_ONLY_KV;
        }
        err.clear();
        return true;
    }

    bool validate_numerical(const float* values, size_t count, const std::string& label,
                            std::string& err) const override {
        if (extension_ && extension_->validate_numerical) {
            guild_error_info_t error{};
            const auto status = extension_->validate_numerical(values, count, &error);
            if (status != GUILD_STATUS_OK)
                err = error.message;
            return status == GUILD_STATUS_OK;
        }
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
        d.next_ext = extension_ ? &*extension_ : nullptr;
        return d;
    }

private:
    std::shared_ptr<void> library_keepalive_;
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
    std::optional<guild_runner_model_ext_v1_t> extension_;
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
#if !defined(GUILD_NO_BUILTIN_QWEN_RUNNERS)
    guild_host_api_v1_t host{};
    host.struct_size = sizeof(host);
    host.negotiate_abi = &kernels::negotiate_abi_version;
    host.register_runner = [](guild_core_host_t h, const guild_runner_desc_v1_t* d,
                              guild_error_info_t* e) -> guild_status_t {
        if (!h || !d)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        std::string err;
        const guild_status_t st = reinterpret_cast<ModelRunnerRegistry*>(h)->register_abi_runner(*d, err);
        if (st != GUILD_STATUS_OK && e) {
            e->struct_size = sizeof(*e);
            e->code = st;
            std::snprintf(e->message, sizeof(e->message), "%s", err.c_str());
        }
        return st;
    };
    guild_register_bundled_qwen_runners(reinterpret_cast<guild_core_host_t>(this), &host, nullptr);
#endif
}

guild_status_t ModelRunnerRegistry::register_runner(std::unique_ptr<IModelRunner> runner, std::string& err) {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    if (!runner || !runner->runner_id() || !runner->runner_id()[0] || !runner->arch_id() || !runner->arch_id()[0] ||
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
        if (to_lower_str(existing->arch_id()) == to_lower_str(runner->arch_id()) &&
            existing->priority() == runner->priority()) {
            err = std::string("duplicate runner registration for architecture: ") + runner->arch_id();
            return GUILD_STATUS_ERR_DUPLICATE_REGISTRATION;
        }
    }
    runners_.push_back(std::move(runner));
    std::stable_sort(runners_.begin(), runners_.end(),
                     [](const auto& a, const auto& b) { return a->priority() > b->priority(); });
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t ModelRunnerRegistry::register_abi_runner(const guild_runner_desc_v1_t& desc, std::string& err,
                                                        std::shared_ptr<void> keepalive) {
    if (desc.struct_size < sizeof(guild_runner_desc_v1_t)) {
        err = "runner ABI descriptor struct_size is too small";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (kernels::negotiate_abi_version(desc.abi_version, nullptr) != GUILD_STATUS_OK) {
        err = "runner ABI version is incompatible with Guild Core ABI v1";
        return GUILD_STATUS_ERR_ABI_MISMATCH;
    }
    if (!desc.arch_id || !desc.arch_id[0] || !desc.runner_id || !desc.runner_id[0] || !desc.package_id ||
        !desc.package_id[0] || desc.n_required_op_kinds > 1024 ||
        (desc.n_required_op_kinds > 0 && !desc.required_op_kinds) || desc.n_required_package_deps > 1024 ||
        (desc.n_required_package_deps > 0 && !desc.required_package_deps)) {
        err = "runner ABI descriptor missing arch_id, runner_id, package_id, or valid dependency arrays";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < desc.n_required_op_kinds; ++i) {
        if (!desc.required_op_kinds || desc.required_op_kinds[i] <= GUILD_OP_INVALID ||
            desc.required_op_kinds[i] >= GUILD_OP_COUNT) {
            err = "runner ABI descriptor declares invalid required operation kind";
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
    }
    for (size_t i = 0; i < desc.n_required_package_deps; ++i) {
        if (!desc.required_package_deps || !desc.required_package_deps[i] || !desc.required_package_deps[i][0]) {
            err = "runner ABI descriptor declares null or empty package dependency";
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        }
    }
    if (desc.next_ext) {
        uint32_t size = 0;
        std::memcpy(&size, desc.next_ext, sizeof(size));
        if (size < sizeof(guild_runner_model_ext_v1_t)) {
            err = "runner model extension is truncated";
            return GUILD_STATUS_ERR_ABI_MISMATCH;
        }
        guild_runner_model_ext_v1_t ext{};
        std::memcpy(&ext, desc.next_ext, sizeof(ext));
        if (ext.tag != GUILD_RUNNER_MODEL_TAG_V1 || ext.version != GUILD_RUNNER_MODEL_VERSION_V1 || !ext.recognize ||
            !ext.describe) {
            err = "unsupported runner model extension or missing callbacks";
            return GUILD_STATUS_ERR_ABI_MISMATCH;
        }
    }
    return register_runner(std::make_unique<AbiModelRunnerAdapter>(desc, std::move(keepalive)), err);
}

guild_status_t ModelRunnerRegistry::unregister_runner(const std::string& runner_id, std::string& err) {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : runners_) {
        if (runner_id == r->runner_id()) {
            const auto lit = package_leases_.find(r->package_id());
            if (r.use_count() > 1 || (lit != package_leases_.end() && lit->second > 0)) {
                err = "cannot unregister model runner '" + runner_id + "' while its package holds active leases";
                return GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY;
            }
        }
    }
    const size_t before = runners_.size();
    runners_.erase(std::remove_if(runners_.begin(), runners_.end(),
                                  [&](const std::shared_ptr<IModelRunner>& r) { return runner_id == r->runner_id(); }),
                   runners_.end());
    if (runners_.size() == before) {
        err = "model runner not found: " + runner_id;
        return GUILD_STATUS_ERR_MISSING_RUNNER;
    }
    err.clear();
    return GUILD_STATUS_OK;
}

guild_status_t ModelRunnerRegistry::unregister_package(const std::string& package_id, std::string& err) {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    if (active_package_leases(package_id) > 0) {
        err = "cannot unregister runner package '" + package_id + "' while active sessions are running";
        return GUILD_STATUS_ERR_ACTIVE_SESSION_BUSY;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t before = runners_.size();
    runners_.erase(
        std::remove_if(runners_.begin(), runners_.end(),
                       [&](const std::shared_ptr<IModelRunner>& r) { return package_id == r->package_id(); }),
        runners_.end());
    disabled_packages_.erase(package_id);
    if (runners_.size() == before) {
        err = "runner package not found: " + package_id;
        return GUILD_STATUS_ERR_MISSING_RUNNER;
    }
    err.clear();
    return GUILD_STATUS_OK;
}

void ModelRunnerRegistry::set_package_enabled(const std::string& package_id, bool enabled) {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    std::lock_guard<std::mutex> lock(mutex_);
    if (enabled)
        disabled_packages_.erase(package_id);
    else
        disabled_packages_.insert(package_id);
}

bool ModelRunnerRegistry::is_package_enabled(const std::string& package_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return disabled_packages_.count(package_id) == 0;
}

void ModelRunnerRegistry::acquire_package_lease(const std::string& package_id) const {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    if (package_id.empty())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    ++package_leases_[package_id];
}

void ModelRunnerRegistry::release_package_lease(const std::string& package_id) const {
    if (package_id.empty())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = package_leases_.find(package_id);
    if (it != package_leases_.end()) {
        if (--it->second <= 0)
            package_leases_.erase(it);
    }
}

int64_t ModelRunnerRegistry::active_package_leases(const std::string& package_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = package_leases_.find(package_id);
    int64_t count = it != package_leases_.end() ? it->second : 0;
    for (const auto& r : runners_)
        if (package_id == r->package_id())
            count += r.use_count() - 1;
    return count;
}

guild_status_t ModelRunnerRegistry::resolve_for_gguf(const GgufFile& gguf,
                                                     std::shared_ptr<const IModelRunner>* out_runner,
                                                     std::string& err) const {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    if (out_runner)
        *out_runner = nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : runners_) {
        if (disabled_packages_.count(r->package_id()) > 0)
            continue;
        if (r->matches_gguf(gguf)) {
            if (out_runner)
                *out_runner = r;
            err.clear();
            return GUILD_STATUS_OK;
        }
    }
    const MetaValue* arch = gguf.get("general.architecture");
    if (!arch || arch->type != MetaType::STRING || arch->s.empty()) {
        err = "missing general.architecture";
        return GUILD_STATUS_ERR_MISSING_RUNNER;
    }
    err = "architecture is '" + arch->s + "', no compatible model runner is registered";
    return GUILD_STATUS_ERR_MISSING_RUNNER;
}

guild_status_t ModelRunnerRegistry::resolve_by_arch(const std::string& arch_name,
                                                    std::shared_ptr<const IModelRunner>* out_runner,
                                                    std::string& err) const {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    if (out_runner)
        *out_runner = nullptr;
    if (arch_name.empty()) {
        err = "architecture name cannot be empty";
        return GUILD_STATUS_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : runners_) {
        if (disabled_packages_.count(r->package_id()) > 0)
            continue;
        if (r->matches_arch_name(arch_name)) {
            if (out_runner)
                *out_runner = r;
            err.clear();
            return GUILD_STATUS_OK;
        }
    }
    err = "no model runner registered for architecture '" + arch_name + "'";
    return GUILD_STATUS_ERR_MISSING_RUNNER;
}

std::shared_ptr<const IModelRunner> ModelRunnerRegistry::find_by_id(const std::string& runner_id) const {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& r : runners_) {
        if (runner_id == r->runner_id())
            return r;
    }
    return nullptr;
}

guild_status_t ModelRunnerRegistry::validate_runner_requirements(
    const IModelRunner& runner, const ModelDescriptor& desc, const core::ModelGeometry& geom,
    const memory::ExecutionPlan& mem_plan, const kernels::KernelRegistry& kernel_reg,
    const guild_device_capabilities_v1_t& caps, RunnerExecutionPlan& out_exec, std::string& err) const {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    out_exec.bindings.clear();
    if (!runner.build_execution_plan(desc, geom, mem_plan, out_exec, err)) {
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }
    if ((out_exec.required_core_caps & caps.core_cap_flags) != out_exec.required_core_caps) {
        err = std::string("runner '") + runner.runner_id() +
              "' requires Guild Core capabilities not present on this host/device";
        return GUILD_STATUS_ERR_CAPABILITY_MISMATCH;
    }
    out_exec.resolved_kernel_ids.clear();
    out_exec.resolved_package_ids.clear();
    out_exec.resolved_kernel_ids.reserve(out_exec.required_ops.size());
    for (guild_op_kind_t op : out_exec.required_ops) {
        kernels::KernelSelectionQuery q;
        q.op_kind = op;
        q.dim_n = geom.n_embd;
        q.dim_k = geom.n_ff;
        kernels::KernelBinding kdesc;
        const guild_status_t st = kernel_reg.select(q, caps, &kdesc, err);
        if (st != GUILD_STATUS_OK || !kdesc) {
            out_exec.bindings.clear();
            return st != GUILD_STATUS_OK ? st : GUILD_STATUS_ERR_MISSING_KERNEL;
        }
        out_exec.resolved_kernel_ids.emplace_back(kdesc->impl_id);
        out_exec.bindings.push_back(kdesc);
        if (kdesc->package_id && kdesc->package_id[0]) {
            const std::string pkg_id = kdesc->package_id;
            if (std::find(out_exec.resolved_package_ids.begin(), out_exec.resolved_package_ids.end(), pkg_id) ==
                out_exec.resolved_package_ids.end()) {
                out_exec.resolved_package_ids.push_back(pkg_id);
            }
        }
    }
    err.clear();
    return GUILD_STATUS_OK;
}

std::vector<std::shared_ptr<const IModelRunner>> ModelRunnerRegistry::list_runners() const {
    abi::ProviderTransaction transaction(abi::provider_lifecycle_mutex());
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<const IModelRunner>> out;
    out.reserve(runners_.size());
    for (const auto& r : runners_)
        out.push_back(r);
    return out;
}

std::vector<RunnerInfoSnapshot> ModelRunnerRegistry::list_runner_snapshots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<RunnerInfoSnapshot> out;
    out.reserve(runners_.size());
    for (const auto& r : runners_) {
        const auto d = r->abi_descriptor();
        RunnerInfoSnapshot snap;
        snap.runner_id = r->runner_id();
        snap.arch_id = r->arch_id();
        snap.package_id = r->package_id();
        snap.runner_version = r->runner_version();
        snap.required_core_caps = d.required_core_caps;
        if (d.required_op_kinds && d.n_required_op_kinds > 0) {
            snap.required_ops.assign(d.required_op_kinds, d.required_op_kinds + d.n_required_op_kinds);
        }
        if (d.required_package_deps && d.n_required_package_deps > 0) {
            for (size_t i = 0; i < d.n_required_package_deps; ++i) {
                if (d.required_package_deps[i])
                    snap.required_package_deps.emplace_back(d.required_package_deps[i]);
            }
        }
        snap.enabled = disabled_packages_.count(snap.package_id) == 0;
        const auto lit = package_leases_.find(snap.package_id);
        snap.active_package_leases = (lit != package_leases_.end()) ? lit->second : 0;
        out.push_back(std::move(snap));
    }
    return out;
}

} // namespace guild::model
