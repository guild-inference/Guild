#include "guild/model/archetype.hpp"

#include <algorithm>
#include <cstring>

namespace guild::model {
namespace {

int64_t get_meta_i64(const GgufFile& gguf, const std::string& key, int64_t fallback) {
    if (const MetaValue* v = gguf.get(key)) {
        if (v->is_num()) {
            return (int64_t) v->u;
        }
    }
    return fallback;
}

double get_meta_double(const GgufFile& gguf, const std::string& key, double fallback) {
    if (const MetaValue* v = gguf.get(key)) {
        return v->num();
    }
    return fallback;
}

std::string get_meta_string(const GgufFile& gguf, const std::string& key, const std::string& fallback) {
    if (const MetaValue* v = gguf.get(key)) {
        if (v->type == MetaType::STRING) return v->s;
    }
    return fallback;
}

std::string get_arch_tag(const GgufFile& gguf) {
    return get_meta_string(gguf, "general.architecture", "");
}

// -----------------------------------------------------------------------------
// Qwen4Exp (Qwen3.8-Flash-Next)
// -----------------------------------------------------------------------------
class Qwen4ExpArchetype final : public IModelArchetype {
public:
    ModelArchetype archetype() const override { return ModelArchetype::Qwen4Exp; }
    const char* name() const override { return "qwen4exp"; }

    bool matches(const GgufFile& gguf) const override {
        const std::string arch = get_arch_tag(gguf);
        if (arch == "qwen4exp") return true;
        return gguf.get("qwen4exp.expert_count") != nullptr;
    }

    bool describe(const GgufFile& gguf, ModelDescriptor& out, std::string& /*err*/) const override {
        out.archetype = ModelArchetype::Qwen4Exp;
        out.arch_name = "qwen4exp";
        out.file_path = gguf.path();
        out.name = get_meta_string(gguf, "general.name", "Qwen3.8-Flash-Next");

        // Attention / recurrence geometry
        out.attn.n_embd = get_meta_i64(gguf, "qwen4exp.embedding_length", 2560);
        out.attn.n_layers = get_meta_i64(gguf, "qwen4exp.block_count", 48);
        out.attn.n_heads = get_meta_i64(gguf, "qwen4exp.attention.head_count", 24);
        out.attn.n_kv_heads = get_meta_i64(gguf, "qwen4exp.attention.head_count_kv", 2);
        out.attn.head_dim = get_meta_i64(gguf, "qwen4exp.attention.key_length", 256);
        out.attn.context_length = get_meta_i64(gguf, "qwen4exp.context_length", 262144);
        out.attn.vocab_size = get_meta_i64(gguf, "qwen4exp.vocab_size", 151936);

        out.attn.pattern = AttentionPattern::HybridGDN;
        out.attn.full_attn_interval = 4; // Every 4th layer is QSA full attention

        // MoE geometry
        out.moe.n_routed_experts = get_meta_i64(gguf, "qwen4exp.expert_count", 512);
        out.moe.k_active_experts = get_meta_i64(gguf, "qwen4exp.expert_used_count", 10);
        out.moe.expert_dim_ff = get_meta_i64(gguf, "qwen4exp.feed_forward_length", 640);
        out.moe.n_shared_experts = 1;
        out.moe.shared_dim_ff = 2560;

        // RoPE configuration
        out.rope.type = get_meta_string(gguf, "qwen4exp.rope.scaling.type", "none");
        out.rope.freq_base = get_meta_double(gguf, "qwen4exp.rope.freq_base", 1000000.0);
        out.rope.factor = get_meta_double(gguf, "qwen4exp.rope.scaling.factor", 1.0);
        out.rope.orig_ctx = get_meta_double(gguf, "qwen4exp.rope.scaling.original_context_length", 0.0);

        return true;
    }
};

// -----------------------------------------------------------------------------
// Qwen35MoE / Ornith
// -----------------------------------------------------------------------------
class Qwen35MoEArchetype final : public IModelArchetype {
public:
    ModelArchetype archetype() const override { return ModelArchetype::Qwen35MoE; }
    const char* name() const override { return "qwen35moe"; }

    bool matches(const GgufFile& gguf) const override {
        const std::string arch = get_arch_tag(gguf);
        if (arch == "qwen35moe" || arch == "qwen2moe" || arch == "ornith") return true;
        return gguf.get("qwen35moe.expert_count") != nullptr || gguf.get("qwen2moe.expert_count") != nullptr;
    }

    bool describe(const GgufFile& gguf, ModelDescriptor& out, std::string& /*err*/) const override {
        out.archetype = ModelArchetype::Qwen35MoE;
        std::string p = "qwen35moe";
        if (gguf.get("qwen2moe.expert_count") != nullptr || get_arch_tag(gguf) == "qwen2moe") p = "qwen2moe";
        out.arch_name = p;
        out.file_path = gguf.path();
        out.name = get_meta_string(gguf, "general.name", "Qwen3.5-MoE");

        out.attn.n_embd = get_meta_i64(gguf, p + ".embedding_length", 2048);
        out.attn.n_layers = get_meta_i64(gguf, p + ".block_count", 28);
        out.attn.n_heads = get_meta_i64(gguf, p + ".attention.head_count", 16);
        out.attn.n_kv_heads = get_meta_i64(gguf, p + ".attention.head_count_kv", 4);
        out.attn.head_dim = get_meta_i64(gguf, p + ".attention.key_length", 128);
        out.attn.context_length = get_meta_i64(gguf, p + ".context_length", 32768);
        out.attn.vocab_size = get_meta_i64(gguf, p + ".vocab_size", 151936);

        out.attn.pattern = AttentionPattern::Standard;
        out.attn.full_attn_interval = 1;

        out.moe.n_routed_experts = get_meta_i64(gguf, p + ".expert_count", 64);
        out.moe.k_active_experts = get_meta_i64(gguf, p + ".expert_used_count", 8);
        out.moe.expert_dim_ff = get_meta_i64(gguf, p + ".feed_forward_length", 1408);
        out.moe.n_shared_experts = get_meta_i64(gguf, p + ".expert_shared_count", 0);

        out.rope.type = get_meta_string(gguf, p + ".rope.scaling.type", "none");
        out.rope.freq_base = get_meta_double(gguf, p + ".rope.freq_base", 1000000.0);
        out.rope.factor = get_meta_double(gguf, p + ".rope.scaling.factor", 1.0);
        return true;
    }
};

// -----------------------------------------------------------------------------
// GLM (GLM-5.3-Flash)
// -----------------------------------------------------------------------------
class GLMArchetype final : public IModelArchetype {
public:
    ModelArchetype archetype() const override { return ModelArchetype::GLM; }
    const char* name() const override { return "glm"; }

    bool matches(const GgufFile& gguf) const override {
        const std::string arch = get_arch_tag(gguf);
        return arch == "glm" || arch == "chatglm" || arch == "glm4" || arch == "glm5";
    }

    bool describe(const GgufFile& gguf, ModelDescriptor& out, std::string& /*err*/) const override {
        out.archetype = ModelArchetype::GLM;
        out.arch_name = "glm";
        out.file_path = gguf.path();
        out.name = get_meta_string(gguf, "general.name", "GLM-5.3-Flash");

        const std::string p = "glm";
        out.attn.n_embd = get_meta_i64(gguf, p + ".embedding_length", 4096);
        out.attn.n_layers = get_meta_i64(gguf, p + ".block_count", 40);
        out.attn.n_heads = get_meta_i64(gguf, p + ".attention.head_count", 32);
        out.attn.n_kv_heads = get_meta_i64(gguf, p + ".attention.head_count_kv", 2);
        out.attn.head_dim = get_meta_i64(gguf, p + ".attention.key_length", 128);
        out.attn.context_length = get_meta_i64(gguf, p + ".context_length", 131072);
        out.attn.vocab_size = get_meta_i64(gguf, p + ".vocab_size", 151552);

        out.attn.pattern = AttentionPattern::HybridState;
        out.attn.full_attn_interval = get_meta_i64(gguf, p + ".full_attention_interval", 1);

        out.moe.n_routed_experts = get_meta_i64(gguf, p + ".expert_count", 128);
        out.moe.k_active_experts = get_meta_i64(gguf, p + ".expert_used_count", 8);
        out.moe.expert_dim_ff = get_meta_i64(gguf, p + ".feed_forward_length", 1024);
        out.moe.n_shared_experts = get_meta_i64(gguf, p + ".expert_shared_count", 1);

        out.rope.type = get_meta_string(gguf, p + ".rope.scaling.type", "none");
        out.rope.freq_base = get_meta_double(gguf, p + ".rope.freq_base", 10000.0);
        return true;
    }
};

// -----------------------------------------------------------------------------
// DeepSeek (V2 / V3 MoE)
// -----------------------------------------------------------------------------
class DeepSeekArchetype final : public IModelArchetype {
public:
    ModelArchetype archetype() const override { return ModelArchetype::DeepSeek; }
    const char* name() const override { return "deepseek"; }

    bool matches(const GgufFile& gguf) const override {
        const std::string arch = get_arch_tag(gguf);
        return arch == "deepseek" || arch == "deepseek2" || arch == "deepseek3";
    }

    bool describe(const GgufFile& gguf, ModelDescriptor& out, std::string& /*err*/) const override {
        out.archetype = ModelArchetype::DeepSeek;
        const std::string p = get_arch_tag(gguf);
        out.arch_name = p.empty() ? "deepseek" : p;
        out.file_path = gguf.path();
        out.name = get_meta_string(gguf, "general.name", "DeepSeek-MoE");

        out.attn.n_embd = get_meta_i64(gguf, out.arch_name + ".embedding_length", 2048);
        out.attn.n_layers = get_meta_i64(gguf, out.arch_name + ".block_count", 27);
        out.attn.n_heads = get_meta_i64(gguf, out.arch_name + ".attention.head_count", 16);
        out.attn.n_kv_heads = get_meta_i64(gguf, out.arch_name + ".attention.head_count_kv", 16);
        out.attn.head_dim = get_meta_i64(gguf, out.arch_name + ".attention.key_length", 128);
        out.attn.context_length = get_meta_i64(gguf, out.arch_name + ".context_length", 163840);
        out.attn.vocab_size = get_meta_i64(gguf, out.arch_name + ".vocab_size", 102400);

        out.attn.pattern = AttentionPattern::Standard;
        out.attn.full_attn_interval = 1;

        out.moe.n_routed_experts = get_meta_i64(gguf, out.arch_name + ".expert_count", 64);
        out.moe.k_active_experts = get_meta_i64(gguf, out.arch_name + ".expert_used_count", 6);
        out.moe.expert_dim_ff = get_meta_i64(gguf, out.arch_name + ".feed_forward_length", 1408);
        out.moe.n_shared_experts = get_meta_i64(gguf, out.arch_name + ".expert_shared_count", 2);

        out.rope.type = get_meta_string(gguf, out.arch_name + ".rope.scaling.type", "none");
        out.rope.freq_base = get_meta_double(gguf, out.arch_name + ".rope.freq_base", 10000.0);
        return true;
    }
};

// -----------------------------------------------------------------------------
// Mixtral (8x7B / 8x22B)
// -----------------------------------------------------------------------------
class MixtralArchetype final : public IModelArchetype {
public:
    ModelArchetype archetype() const override { return ModelArchetype::Mixtral; }
    const char* name() const override { return "mixtral"; }

    bool matches(const GgufFile& gguf) const override {
        const std::string arch = get_arch_tag(gguf);
        if (arch == "mixtral") return true;
        if (arch == "llama") {
            return gguf.get("llama.expert_count") != nullptr;
        }
        return false;
    }

    bool describe(const GgufFile& gguf, ModelDescriptor& out, std::string& /*err*/) const override {
        out.archetype = ModelArchetype::Mixtral;
        out.arch_name = "mixtral";
        out.file_path = gguf.path();
        out.name = get_meta_string(gguf, "general.name", "Mixtral-MoE");

        const std::string p = get_arch_tag(gguf) == "llama" ? "llama" : "mixtral";
        out.attn.n_embd = get_meta_i64(gguf, p + ".embedding_length", 4096);
        out.attn.n_layers = get_meta_i64(gguf, p + ".block_count", 32);
        out.attn.n_heads = get_meta_i64(gguf, p + ".attention.head_count", 32);
        out.attn.n_kv_heads = get_meta_i64(gguf, p + ".attention.head_count_kv", 8);
        out.attn.head_dim = get_meta_i64(gguf, p + ".attention.key_length", 128);
        out.attn.context_length = get_meta_i64(gguf, p + ".context_length", 32768);
        out.attn.vocab_size = get_meta_i64(gguf, p + ".vocab_size", 32000);

        out.attn.pattern = AttentionPattern::Standard;
        out.attn.full_attn_interval = 1;

        out.moe.n_routed_experts = get_meta_i64(gguf, p + ".expert_count", 8);
        out.moe.k_active_experts = get_meta_i64(gguf, p + ".expert_used_count", 2);
        out.moe.expert_dim_ff = get_meta_i64(gguf, p + ".feed_forward_length", 14336);
        out.moe.n_shared_experts = 0;

        out.rope.type = get_meta_string(gguf, p + ".rope.scaling.type", "none");
        out.rope.freq_base = get_meta_double(gguf, p + ".rope.freq_base", 1000000.0);
        return true;
    }
};

}  // namespace

ArchetypeRegistry& ArchetypeRegistry::instance() {
    static ArchetypeRegistry reg;
    return reg;
}

ArchetypeRegistry::ArchetypeRegistry() {
    register_archetype(std::make_unique<Qwen4ExpArchetype>());
    register_archetype(std::make_unique<Qwen35MoEArchetype>());
    register_archetype(std::make_unique<GLMArchetype>());
    register_archetype(std::make_unique<DeepSeekArchetype>());
    register_archetype(std::make_unique<MixtralArchetype>());
}

void ArchetypeRegistry::register_archetype(std::unique_ptr<IModelArchetype> arch) {
    if (arch) {
        registry_.push_back(std::move(arch));
    }
}

const IModelArchetype* ArchetypeRegistry::find(ModelArchetype type) const {
    for (const auto& arch : registry_) {
        if (arch->archetype() == type) return arch.get();
    }
    return nullptr;
}

const IModelArchetype* ArchetypeRegistry::find(const std::string& name) const {
    const ModelArchetype type = archetype_from_string(name);
    if (type != ModelArchetype::Unknown) {
        return find(type);
    }
    for (const auto& arch : registry_) {
        if (std::strcmp(arch->name(), name.c_str()) == 0) return arch.get();
    }
    return nullptr;
}

const IModelArchetype* ArchetypeRegistry::detect(const GgufFile& gguf) const {
    for (const auto& arch : registry_) {
        if (arch->matches(gguf)) return arch.get();
    }
    return nullptr;
}

bool ArchetypeRegistry::describe_gguf(const GgufFile& gguf, ModelDescriptor& out, std::string& err) const {
    const IModelArchetype* arch = detect(gguf);
    if (!arch) {
        err = "Unsupported model architecture: " + get_arch_tag(gguf);
        return false;
    }
    return arch->describe(gguf, out, err);
}

}  // namespace guild::model
