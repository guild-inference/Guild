#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace guild::model {

enum class ModelArchetype {
    Unknown = 0,
    Qwen4Exp,    // Qwen3.8-Flash-Next / qwen4exp hybrid GDN+QSA MoE
    Qwen35MoE,   // Ornith / qwen35moe
    GLM,         // GLM-5.3-Flash hybrid attention
    DeepSeek,    // DeepSeek V2/V3 style MoE with MLA
    Mixtral      // Mixtral standard MoE
};

const char* archetype_to_string(ModelArchetype arch);
ModelArchetype archetype_from_string(const std::string& name);

enum class AttentionPattern {
    Standard = 0,  // Full attention on all layers
    HybridGDN,     // GDN recurrence + periodic QSA (e.g. Qwen3.8-Flash-Next)
    HybridState    // Generic attention + state / recurrence (e.g. GLM)
};

const char* attention_pattern_to_string(AttentionPattern pattern);

struct MoEGeometry {
    int64_t n_routed_experts = 0;   // e.g. 512 for Qwen3.8-Flash-Next
    int64_t k_active_experts = 0;   // e.g. 10 active per token
    int64_t expert_dim_ff = 0;      // e.g. 640 intermediate dim
    int64_t n_shared_experts = 0;   // e.g. 1 shared expert
    int64_t shared_dim_ff = 0;      // intermediate dim of shared expert
    int64_t expert_blob_bytes = 0;  // packed or quantized blob size per expert if known

    int64_t total_routed_experts(int64_t n_layers) const {
        return n_layers * n_routed_experts;
    }
};

struct AttentionGeometry {
    int64_t n_embd = 0;             // Embedding dimension (e.g. 2560)
    int64_t n_layers = 0;           // Layer count (e.g. 48)
    int64_t n_heads = 0;            // Query attention heads (e.g. 24)
    int64_t n_kv_heads = 0;         // Key/Value attention heads (e.g. 2)
    int64_t head_dim = 0;           // Head dimension (e.g. 256)
    int64_t context_length = 0;     // Max context window (e.g. 262144)
    int64_t vocab_size = 0;         // Vocabulary size (e.g. 151936)

    AttentionPattern pattern = AttentionPattern::Standard;
    int64_t full_attn_interval = 1; // e.g. 4 for Qwen3.8 (every 4th layer is full attention)

    // Derived layer counts
    int64_t n_full_attn_layers() const {
        if (full_attn_interval <= 1) return n_layers;
        return n_layers / full_attn_interval;
    }
    int64_t n_recurrent_layers() const {
        return n_layers - n_full_attn_layers();
    }
    bool is_full_attn_layer(int64_t layer) const {
        if (full_attn_interval <= 1) return true;
        return layer % full_attn_interval == (full_attn_interval - 1);
    }
};

struct RoPEConfig {
    std::string type = "none";      // "none", "linear", "yarn"
    double freq_base = 10000.0;
    double factor = 1.0;
    double orig_ctx = 0.0;
};

struct ModelDescriptor {
    std::string name;               // Model name or architecture tag
    std::string file_path;          // Source GGUF / shard path
    ModelArchetype archetype = ModelArchetype::Unknown;
    std::string arch_name;          // GGUF general.architecture string

    AttentionGeometry attn;
    MoEGeometry moe;
    RoPEConfig rope;

    // Sizing and memory estimation
    uint64_t estimate_routed_weights_bytes(int bytes_per_elem_weight = 2) const;
    uint64_t estimate_kv_bytes(int64_t context_tokens, int bytes_per_elem_kv = 2) const;
};

}  // namespace guild::model
