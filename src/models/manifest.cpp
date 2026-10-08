#include "guild/models/manifest.hpp"
#include "guild/model/archetype.hpp"
#include "guild/server/json.hpp"

#include <algorithm>
#include <cctype>

namespace guild::models {

using namespace guild::server::json;

namespace {

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

} // namespace

std::string ModelManifest::to_json() const {
    JsonValue root;
    root["name"] = name;

    JsonValue aliases_arr(std::vector<JsonValue>{});
    for (const auto& a : aliases) {
        aliases_arr.arr_val.push_back(JsonValue(a));
    }
    root["aliases"] = std::move(aliases_arr);

    root["architecture"] = architecture;
    root["quantization"] = quantization;
    root["source"] = source;
    root["description"] = description;
    root["expected_size_bytes"] = static_cast<int64_t>(expected_size_bytes);
    root["context_length"] = static_cast<int64_t>(context_length);

    if (!tokenizer_source.empty()) root["tokenizer_source"] = tokenizer_source;
    if (!mtp_source.empty()) root["mtp_source"] = mtp_source;
    if (!expert_profile_source.empty()) root["expert_profile_source"] = expert_profile_source;

    // Geometry
    JsonValue geom;
    geom["n_layers"] = n_layers;
    geom["n_embd"] = n_embd;
    geom["n_heads"] = n_heads;
    geom["n_kv_heads"] = n_kv_heads;
    geom["head_dim"] = head_dim;
    geom["vocab_size"] = vocab_size;
    geom["n_routed_experts"] = n_routed_experts;
    geom["k_active_experts"] = k_active_experts;
    geom["expert_dim_ff"] = expert_dim_ff;
    geom["n_shared_experts"] = n_shared_experts;
    geom["shared_dim_ff"] = shared_dim_ff;
    geom["full_attn_interval"] = full_attn_interval;
    geom["expert_blob_bytes"] = expert_blob_bytes;
    root["geometry"] = std::move(geom);

    if (!metadata.empty()) {
        JsonValue meta;
        for (const auto& [k, v] : metadata) {
            meta[k] = v;
        }
        root["metadata"] = std::move(meta);
    }

    JsonValue files_arr(std::vector<JsonValue>{});
    for (const auto& f : files) {
        JsonValue f_obj;
        f_obj["name"] = f.name;
        f_obj["url"] = f.url;
        f_obj["size_bytes"] = static_cast<int64_t>(f.size_bytes);
        if (!f.sha256.empty()) f_obj["sha256"] = f.sha256;
        f_obj["role"] = f.role;
        if (!f.blob_id.empty()) f_obj["blob_id"] = f.blob_id;
        if (!f.local_path.empty()) f_obj["local_path"] = f.local_path;
        files_arr.arr_val.push_back(std::move(f_obj));
    }
    root["files"] = std::move(files_arr);

    return root.serialize_pretty();
}

std::optional<ModelManifest> ModelManifest::from_json(const std::string& json_str) {
    std::string err;
    auto root_opt = JsonValue::parse(json_str, err);
    if (!root_opt.has_value() || !root_opt->is_object()) {
        return std::nullopt;
    }

    const auto& root = *root_opt;
    ModelManifest m;
    m.name = root["name"].as_string();
    if (m.name.empty()) return std::nullopt;

    if (root.contains("aliases") && root["aliases"].is_array()) {
        for (const auto& a : root["aliases"].arr_val) {
            if (a.is_string() && !a.as_string().empty()) {
                m.aliases.push_back(a.as_string());
            }
        }
    }

    m.architecture = root["architecture"].as_string();
    m.quantization = root["quantization"].as_string();
    m.source = root["source"].as_string();
    m.description = root["description"].as_string();
    m.expected_size_bytes = static_cast<uint64_t>(root["expected_size_bytes"].as_int(0));
    m.context_length = static_cast<uint32_t>(root["context_length"].as_int(262144));

    m.tokenizer_source = root["tokenizer_source"].as_string();
    m.mtp_source = root["mtp_source"].as_string();
    m.expert_profile_source = root["expert_profile_source"].as_string();

    if (root.contains("geometry") && root["geometry"].is_object()) {
        const auto& g = root["geometry"];
        m.n_layers = g["n_layers"].as_int(0);
        m.n_embd = g["n_embd"].as_int(0);
        m.n_heads = g["n_heads"].as_int(0);
        m.n_kv_heads = g["n_kv_heads"].as_int(0);
        m.head_dim = g["head_dim"].as_int(0);
        m.vocab_size = g["vocab_size"].as_int(0);
        m.n_routed_experts = g["n_routed_experts"].as_int(0);
        m.k_active_experts = g["k_active_experts"].as_int(0);
        m.expert_dim_ff = g["expert_dim_ff"].as_int(0);
        m.n_shared_experts = g["n_shared_experts"].as_int(0);
        m.shared_dim_ff = g["shared_dim_ff"].as_int(0);
        m.full_attn_interval = g["full_attn_interval"].as_int(1);
        m.expert_blob_bytes = g["expert_blob_bytes"].as_int(0);
    }

    if (root.contains("metadata") && root["metadata"].is_object()) {
        for (const auto& kv : root["metadata"].obj_val) {
            if (kv.second.is_string()) {
                m.metadata[kv.first] = kv.second.as_string();
            }
        }
    }

    if (root.contains("files") && root["files"].is_array()) {
        for (const auto& item : root["files"].arr_val) {
            if (!item.is_object()) continue;
            ModelFile f;
            f.name = item["name"].as_string();
            f.url = item["url"].as_string();
            f.size_bytes = static_cast<uint64_t>(item["size_bytes"].as_int(0));
            f.sha256 = item["sha256"].as_string();
            f.role = item["role"].as_string("shard");
            f.blob_id = item["blob_id"].as_string();
            f.local_path = item["local_path"].as_string();
            m.files.push_back(std::move(f));
        }
    }

    if (m.expected_size_bytes == 0) {
        m.expected_size_bytes = m.total_size_bytes();
    }

    return m;
}

guild::model::ModelDescriptor ModelManifest::to_descriptor() const {
    guild::model::ModelDescriptor desc;
    desc.name = name;
    desc.arch_name = architecture;
    desc.archetype = guild::model::archetype_from_string(architecture);
    if (desc.archetype == guild::model::ModelArchetype::Unknown) {
        if (architecture == "qwen4exp") desc.archetype = guild::model::ModelArchetype::Qwen4Exp;
        else if (architecture == "qwen35moe") desc.archetype = guild::model::ModelArchetype::Qwen35MoE;
    }

    const ModelFile* prim = find_file_by_role("primary");
    if (!prim) prim = find_file_by_role("shard");
    if (prim && !prim->local_path.empty()) {
        desc.file_path = prim->local_path;
    } else if (!files.empty() && !files[0].local_path.empty()) {
        desc.file_path = files[0].local_path;
    } else {
        desc.file_path = name;
    }

    desc.attn.n_layers = n_layers;
    desc.attn.n_embd = n_embd;
    desc.attn.n_heads = n_heads;
    desc.attn.n_kv_heads = n_kv_heads;
    desc.attn.head_dim = head_dim;
    desc.attn.context_length = context_length;
    desc.attn.vocab_size = vocab_size;
    desc.attn.full_attn_interval = full_attn_interval;
    if (full_attn_interval > 1) {
        desc.attn.pattern = guild::model::AttentionPattern::HybridGDN;
    } else {
        desc.attn.pattern = guild::model::AttentionPattern::Standard;
    }
    if (desc.archetype == guild::model::ModelArchetype::Qwen4Exp) {
        desc.attn.mechanism = guild::model::AttentionMechanism::IndexedSparse;
    } else {
        desc.attn.mechanism = guild::model::AttentionMechanism::DenseCausal;
    }

    desc.moe.n_routed_experts = n_routed_experts;
    desc.moe.k_active_experts = k_active_experts;
    desc.moe.expert_dim_ff = expert_dim_ff;
    desc.moe.n_shared_experts = n_shared_experts;
    desc.moe.shared_dim_ff = shared_dim_ff;
    desc.moe.expert_blob_bytes = expert_blob_bytes;

    return desc;
}

uint64_t ModelManifest::total_size_bytes() const {
    uint64_t total = 0;
    for (const auto& f : files) {
        total += f.size_bytes;
    }
    return total;
}

bool ModelManifest::has_file(const std::string& filename) const {
    return find_file_by_name(filename) != nullptr;
}

const ModelFile* ModelManifest::find_file_by_name(const std::string& filename) const {
    for (const auto& f : files) {
        if (f.name == filename) return &f;
    }
    return nullptr;
}

const ModelFile* ModelManifest::find_file_by_role(const std::string& role) const {
    for (const auto& f : files) {
        if (f.role == role) return &f;
    }
    return nullptr;
}

bool ModelManifest::matches_name_or_alias(const std::string& query) const {
    std::string q = to_lower(query);
    if (to_lower(name) == q) return true;
    for (const auto& a : aliases) {
        if (to_lower(a) == q) return true;
    }
    return false;
}

} // namespace guild::models
