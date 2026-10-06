#pragma once

#include "guild/model/model_descriptor.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace guild::models {

struct ModelFile {
    std::string name;             // e.g. "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
    std::string url;              // remote URL or relative download path
    uint64_t size_bytes{0};       // expected size in bytes
    std::string sha256;           // expected SHA-256 hash (hex string), optional
    std::string role{"shard"};    // "primary", "shard", "dense_pack", "mtp", "tokenizer", "profile"
    std::string blob_id;          // digest or content address in local blob store
    std::string local_path;       // resolved local file path if present on disk
};

struct ModelManifest {
    std::string name;                     // canonical name: "qwen3.8-flash-next"
    std::vector<std::string> aliases;     // ["qwen", "qwen3.8", "qwen-flash", "unsloth-ud-iq4_xs"]
    std::string architecture;             // "qwen4exp"
    std::string quantization;             // "UD-IQ4_XS"
    std::string source;                   // "unsloth/Qwen3.8-Flash-Next-GGUF"
    std::string description;              // "Unsloth UD-IQ4_XS ~4-bit dynamic quant"
    uint64_t expected_size_bytes{0};      // expected total disk size
    uint32_t context_length{262144};      // native context length

    // Optional auxiliary assets
    std::string tokenizer_source;
    std::string mtp_source;
    std::string expert_profile_source;

    // Model geometry parameters to construct ModelDescriptor
    int64_t n_layers{0};
    int64_t n_embd{0};
    int64_t n_heads{0};
    int64_t n_kv_heads{0};
    int64_t head_dim{0};
    int64_t vocab_size{0};
    int64_t n_routed_experts{0};
    int64_t k_active_experts{0};
    int64_t expert_dim_ff{0};
    int64_t n_shared_experts{0};
    int64_t shared_dim_ff{0};
    int64_t full_attn_interval{1};
    int64_t expert_blob_bytes{0};

    // Extra arbitrary metadata
    std::map<std::string, std::string> metadata;

    // Files composing the model
    std::vector<ModelFile> files;

    // Methods
    std::string to_json() const;
    static std::optional<ModelManifest> from_json(const std::string& json_str);

    guild::model::ModelDescriptor to_descriptor() const;
    uint64_t total_size_bytes() const;
    bool has_file(const std::string& filename) const;
    const ModelFile* find_file_by_name(const std::string& filename) const;
    const ModelFile* find_file_by_role(const std::string& role) const;

    bool matches_name_or_alias(const std::string& query) const;
};

} // namespace guild::models
