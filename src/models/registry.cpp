#include "guild/models/registry.hpp"

#include <filesystem>

namespace guild::models {

namespace fs = std::filesystem;

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry reg;
    return reg;
}

ModelRegistry::ModelRegistry() {
    init_builtins();
}

void ModelRegistry::init_builtins() {
    // 1. Qwen3.8-Flash-Next UD-IQ4_XS
    {
        ModelManifest m;
        m.name = "qwen3.8-flash-next";
        m.aliases = {
            "qwen",
            "qwen3.8",
            "qwen-flash",
            "unsloth-ud-iq4_xs",
            "UD-IQ4_XS",
            "Qwen3.8-Flash-Next"
        };
        m.architecture = "qwen4exp";
        m.quantization = "UD-IQ4_XS";
        m.source = "unsloth/Qwen3.8-Flash-Next-GGUF";
        m.description = "Qwen3.8-Flash-Next UD-IQ4_XS (~4-bit dynamic Unsloth MoE, 512x10 experts)";
        m.context_length = 262144;

        m.n_layers = 48;
        m.n_embd = 2560;
        m.n_heads = 24;
        m.n_kv_heads = 2;
        m.head_dim = 256;
        m.vocab_size = 151936;
        m.n_routed_experts = 512;
        m.k_active_experts = 10;
        m.expert_dim_ff = 640;
        m.n_shared_experts = 1;
        m.shared_dim_ff = 2560;
        m.full_attn_interval = 4;
        m.expert_blob_bytes = 2421813;

        ModelFile s1;
        s1.name = "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
        s1.size_bytes = 10946624;
        s1.sha256 = "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4";
        s1.role = "primary";
        s1.url = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
        m.files.push_back(s1);

        ModelFile s2;
        s2.name = "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf";
        s2.size_bytes = 49835229856;
        s2.sha256 = "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7";
        s2.role = "shard";
        s2.url = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf";
        m.files.push_back(s2);

        ModelFile s3;
        s3.name = "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf";
        s3.size_bytes = 43836407744;
        s3.sha256 = "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833";
        s3.role = "shard";
        s3.url = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf";
        m.files.push_back(s3);

        m.expected_size_bytes = m.total_size_bytes();
        builtins_.push_back(std::move(m));
    }

    // 2. Ornith 1.5 35B
    {
        ModelManifest m;
        m.name = "ornith:35b";
        m.aliases = {
            "ornith",
            "ornith-1.5-35b",
            "ornith-35b"
        };
        m.architecture = "qwen35moe";
        m.quantization = "Q4_K_M";
        m.source = "ukisai/Ornith-1.5-35B-GGUF";
        m.description = "Ornith 1.5 35B MoE (Q4_K_M, 256 routed experts, 8 active)";
        m.context_length = 262144;

        m.n_layers = 40;
        m.n_embd = 2048;
        m.n_heads = 16;
        m.n_kv_heads = 2;
        m.head_dim = 256;
        m.vocab_size = 248320;
        m.n_routed_experts = 256;
        m.k_active_experts = 8;
        m.expert_dim_ff = 512;
        m.full_attn_interval = 4;
        m.n_shared_experts = 1;
        m.shared_dim_ff = 512;
        m.expert_blob_bytes = 1907936;

        ModelFile s1;
        s1.name = "Ornith-1.5-35B-Q4_K_M.gguf";
        s1.size_bytes = 21713463040;
        s1.role = "primary";
        s1.url = "https://huggingface.co/ukisai/Ornith-1.5-35B-GGUF/resolve/main/Ornith-1.5-35B-Q4_K_M.gguf";
        if (std::filesystem::exists("/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf")) {
            s1.local_path = "/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf";
        }
        m.files.push_back(s1);

        if (std::filesystem::exists("/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b")) {
            m.metadata["pack_dir"] = "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b";
            m.metadata["tokenizer_dir"] = "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b/tokenizer";
        }

        m.expected_size_bytes = m.total_size_bytes();
        builtins_.push_back(std::move(m));
    }
}

std::vector<ModelManifest> ModelRegistry::list_available() const {
    return builtins_;
}

std::vector<ModelManifest> ModelRegistry::list_installed(const ModelStore& store) const {
    return store.list_manifests();
}

std::optional<ModelManifest> ModelRegistry::find_builtin(const std::string& name_or_alias) const {
    for (const auto& m : builtins_) {
        if (m.matches_name_or_alias(name_or_alias)) {
            return m;
        }
    }
    return std::nullopt;
}

bool ModelRegistry::is_installed(const std::string& name_or_alias, const ModelStore& store) const {
    return store.has_manifest(name_or_alias);
}

std::optional<ModelManifest> ModelRegistry::resolve(const std::string& name_or_query, const ModelStore& store) const {
    // 1. Check local store manifests first
    auto installed = store.get_manifest(name_or_query);
    if (installed.has_value()) {
        return installed;
    }

    // 2. Check built-in catalog
    auto builtin = find_builtin(name_or_query);
    if (builtin.has_value()) {
        return builtin;
    }

    // 3. Check if query is an existing file or directory on disk
    std::error_code ec;
    if (fs::exists(name_or_query, ec) && fs::is_regular_file(name_or_query, ec)) {
        ModelManifest m;
        fs::path p(name_or_query);
        m.name = p.stem().string();
        m.architecture = "unknown";
        m.quantization = "unknown";
        m.description = "Local file model: " + name_or_query;

        ModelFile f;
        f.name = p.filename().string();
        f.local_path = fs::canonical(p).string();
        f.size_bytes = fs::file_size(p, ec);
        f.role = "primary";
        m.files.push_back(f);
        m.expected_size_bytes = f.size_bytes;
        return m;
    }

    return std::nullopt;
}

} // namespace guild::models
