#include "guild/model/model_descriptor.hpp"

#include <algorithm>
#include <cstring>

namespace guild::model {

const char* archetype_to_string(ModelArchetype arch) {
    switch (arch) {
    case ModelArchetype::Qwen4Exp:  return "qwen4exp";
    case ModelArchetype::Qwen35MoE: return "qwen35moe";
    case ModelArchetype::GLM:       return "glm";
    case ModelArchetype::DeepSeek:  return "deepseek";
    case ModelArchetype::Mixtral:   return "mixtral";
    case ModelArchetype::Unknown:
    default:                        return "unknown";
    }
}

ModelArchetype archetype_from_string(const std::string& name) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    if (s == "qwen4exp") return ModelArchetype::Qwen4Exp;
    if (s == "qwen35moe" || s == "qwen2moe" || s == "ornith") return ModelArchetype::Qwen35MoE;
    if (s == "glm" || s == "chatglm" || s == "glm4" || s == "glm5") return ModelArchetype::GLM;
    if (s == "deepseek" || s == "deepseek2" || s == "deepseek3") return ModelArchetype::DeepSeek;
    if (s == "mixtral" || s == "mistral-moe") return ModelArchetype::Mixtral;
    return ModelArchetype::Unknown;
}

const char* attention_pattern_to_string(AttentionPattern pattern) {
    switch (pattern) {
    case AttentionPattern::Standard:   return "standard";
    case AttentionPattern::HybridGDN:  return "hybrid_gdn";
    case AttentionPattern::HybridState: return "hybrid_state";
    default:                           return "unknown";
    }
}

uint64_t ModelDescriptor::estimate_routed_weights_bytes(int bytes_per_elem_weight) const {
    const int64_t total_experts = moe.total_routed_experts(attn.n_layers);
    if (total_experts <= 0) return 0;
    if (moe.expert_blob_bytes > 0) {
        return (uint64_t) total_experts * (uint64_t) moe.expert_blob_bytes;
    }
    // Standard SwiGLU expert: gate_proj, up_proj, down_proj
    // gate: [ff, embd], up: [ff, embd], down: [embd, ff]
    const uint64_t elems_per_expert = 3ULL * (uint64_t) moe.expert_dim_ff * (uint64_t) attn.n_embd;
    return (uint64_t) total_experts * elems_per_expert * (uint64_t) std::max(1, bytes_per_elem_weight);
}

uint64_t ModelDescriptor::estimate_kv_bytes(int64_t context_tokens, int bytes_per_elem_kv) const {
    if (context_tokens <= 0 || attn.head_dim <= 0 || attn.n_kv_heads <= 0) return 0;
    const uint64_t bytes_per_token_layer = 2ULL * (uint64_t) attn.n_kv_heads * (uint64_t) attn.head_dim *
                                           (uint64_t) std::max(1, bytes_per_elem_kv);
    return (uint64_t) context_tokens * (uint64_t) attn.n_full_attn_layers() * bytes_per_token_layer;
}

}  // namespace guild::model
