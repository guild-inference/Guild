#include "guild/models/store.hpp"
#include "guild/models/sha256.hpp"
#include "guild/server/json.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace guild::models {

namespace fs = std::filesystem;
using namespace guild::server::json;

namespace {

std::string get_env_var(const char* name) {
    const char* val = std::getenv(name);
    return val ? std::string(val) : "";
}

} // namespace

std::string ModelStore::default_data_dir() {
    std::string custom = get_env_var("GUILD_DATA_DIR");
    if (!custom.empty()) return custom;

    std::string xdg = get_env_var("XDG_DATA_HOME");
    if (!xdg.empty()) {
        return (fs::path(xdg) / "guild").string();
    }

    std::string home = get_env_var("HOME");
    if (!home.empty()) {
        return (fs::path(home) / ".local" / "share" / "guild").string();
    }

    return "./.guild_data";
}

bool ModelStore::is_safe_name(const std::string& name) {
    if (name.empty()) return false;
    if (name == "." || name == "..") return false;
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos) return false;
    if (name.find("..") != std::string::npos) return false;
    for (char c : name) {
        if (static_cast<unsigned char>(c) < 32) return false;
        if (c == '*' || c == '?' || c == '<' || c == '>' || c == '|' || c == '\"') return false;
    }
    return true;
}

ModelStore::ModelStore(const StoreOptions& options) {
    if (!options.custom_data_dir.empty()) {
        root_dir_ = options.custom_data_dir;
    } else {
        root_dir_ = default_data_dir();
    }
}

std::string ModelStore::models_dir() const {
    return (fs::path(root_dir_) / "models").string();
}

std::string ModelStore::manifests_dir() const {
    return (fs::path(models_dir()) / "manifests").string();
}

std::string ModelStore::blobs_dir() const {
    return (fs::path(models_dir()) / "blobs").string();
}

std::string ModelStore::refs_dir() const {
    return (fs::path(models_dir()) / "refs").string();
}

std::string ModelStore::index_path() const {
    return (fs::path(models_dir()) / "index.json").string();
}

bool ModelStore::init() {
    std::error_code ec;
    fs::create_directories(manifests_dir(), ec);
    fs::create_directories(blobs_dir(), ec);
    fs::create_directories(refs_dir(), ec);
    load_blob_index();
    return !ec;
}

void ModelStore::load_blob_index() const {
    blob_index_.clear();
    std::ifstream f(index_path());
    if (!f.is_open()) return;

    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string err;
    auto val = JsonValue::parse(buffer.str(), err);
    if (!val.has_value() || !val->is_object()) return;

    const auto& root = *val;
    if (root.contains("blobs") && root["blobs"].is_object()) {
        for (const auto& kv : root["blobs"].obj_val) {
            if (kv.second.is_object()) {
                BlobInfo bi;
                bi.blob_id = kv.first;
                bi.size_bytes = static_cast<uint64_t>(kv.second["size_bytes"].as_int(0));
                bi.ref_count = static_cast<int>(kv.second["ref_count"].as_int(1));
                bi.path = kv.second["path"].as_string();
                blob_index_[kv.first] = bi;
            }
        }
    }
}

void ModelStore::save_blob_index() const {
    JsonValue root;
    JsonValue blobs_obj;
    for (const auto& [id, info] : blob_index_) {
        JsonValue b;
        b["size_bytes"] = static_cast<int64_t>(info.size_bytes);
        b["ref_count"] = info.ref_count;
        b["path"] = info.path;
        blobs_obj[id] = std::move(b);
    }
    root["blobs"] = std::move(blobs_obj);

    std::ofstream f(index_path());
    if (f.is_open()) {
        f << root.serialize_pretty() << "\n";
    }
}

std::vector<ModelManifest> ModelStore::list_manifests() const {
    std::vector<ModelManifest> manifests;
    std::error_code ec;
    if (!fs::exists(manifests_dir(), ec)) return manifests;

    for (const auto& entry : fs::directory_iterator(manifests_dir(), ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            std::ifstream file(entry.path());
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                auto m = ModelManifest::from_json(buffer.str());
                if (m.has_value()) {
                    manifests.push_back(std::move(*m));
                }
            }
        }
    }
    return manifests;
}

std::optional<ModelManifest> ModelStore::get_manifest(const std::string& name_or_alias) const {
    if (is_safe_name(name_or_alias)) {
        fs::path p = fs::path(manifests_dir()) / (name_or_alias + ".json");
        if (fs::exists(p)) {
            std::ifstream file(p);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                auto m = ModelManifest::from_json(buffer.str());
                if (m.has_value()) return m;
            }
        }
    }

    // Check aliases
    for (const auto& m : list_manifests()) {
        if (m.matches_name_or_alias(name_or_alias)) {
            return m;
        }
    }

    return std::nullopt;
}

bool ModelStore::has_manifest(const std::string& name_or_alias) const {
    return get_manifest(name_or_alias).has_value();
}

bool ModelStore::save_manifest(const ModelManifest& manifest) {
    if (!is_safe_name(manifest.name)) return false;
    init();

    fs::path p = fs::path(manifests_dir()) / (manifest.name + ".json");
    std::ofstream file(p);
    if (!file.is_open()) return false;

    file << manifest.to_json() << "\n";
    return true;
}

bool ModelStore::remove_model(const std::string& name_or_alias) {
    auto m_opt = get_manifest(name_or_alias);
    if (!m_opt.has_value()) return false;

    const auto& m = *m_opt;
    init();

    // 1. Remove manifest file
    fs::path man_path = fs::path(manifests_dir()) / (m.name + ".json");
    std::error_code ec;
    fs::remove(man_path, ec);

    // 2. Remove model refs folder
    fs::path ref_path = fs::path(refs_dir()) / m.name;
    fs::remove_all(ref_path, ec);

    // 3. Unregister and delete blobs if ref_count reaches 0
    for (const auto& f : m.files) {
        if (!f.blob_id.empty()) {
            unregister_blob(f.blob_id);
        }
    }

    save_blob_index();
    return true;
}

std::string ModelStore::get_blob_path(const std::string& blob_id) const {
    if (!is_safe_name(blob_id)) return "";
    return (fs::path(blobs_dir()) / blob_id).string();
}

bool ModelStore::has_blob(const std::string& blob_id) const {
    std::string path = get_blob_path(blob_id);
    if (path.empty()) return false;
    return fs::exists(path);
}

bool ModelStore::register_blob(const std::string& blob_id, uint64_t size_bytes, const std::string& rel_or_abs_path) {
    load_blob_index();
    auto it = blob_index_.find(blob_id);
    if (it != blob_index_.end()) {
        it->second.ref_count++;
    } else {
        BlobInfo bi;
        bi.blob_id = blob_id;
        bi.size_bytes = size_bytes;
        bi.ref_count = 1;
        bi.path = rel_or_abs_path;
        blob_index_[blob_id] = bi;
    }
    save_blob_index();
    return true;
}

bool ModelStore::unregister_blob(const std::string& blob_id) {
    load_blob_index();
    auto it = blob_index_.find(blob_id);
    if (it == blob_index_.end()) return false;

    it->second.ref_count--;
    if (it->second.ref_count <= 0) {
        std::string path = get_blob_path(blob_id);
        if (!path.empty() && fs::exists(path)) {
            // Safe removal: verify path is strictly inside blobs_dir()
            fs::path bp = fs::canonical(blobs_dir());
            fs::path fp = fs::canonical(path);
            if (fp.string().find(bp.string()) == 0) {
                std::error_code ec;
                fs::remove(fp, ec);
            }
        }
        blob_index_.erase(it);
    }
    save_blob_index();
    return true;
}

bool ModelStore::import_model_directory(const ModelManifest& base_manifest,
                                        const std::string& source_dir,
                                        bool use_symlinks,
                                        std::string& err_msg) {
    if (!is_safe_name(base_manifest.name)) {
        err_msg = "Invalid or unsafe model name: " + base_manifest.name;
        return false;
    }

    fs::path src(source_dir);
    if (!fs::exists(src)) {
        err_msg = "Source path does not exist: " + source_dir;
        return false;
    }

    init();
    ModelManifest imported = base_manifest;
    fs::path m_ref_dir = fs::path(refs_dir()) / imported.name;
    std::error_code ec;
    fs::create_directories(m_ref_dir, ec);

    for (auto& file_spec : imported.files) {
        fs::path candidate = src / file_spec.name;
        if (!fs::exists(candidate)) {
            // Search inside source_dir recursively if needed
            bool found = false;
            for (const auto& entry : fs::recursive_directory_iterator(src, ec)) {
                if (entry.is_regular_file() && entry.path().filename() == file_spec.name) {
                    candidate = entry.path();
                    found = true;
                    break;
                }
            }
            if (!found) {
                err_msg = "Missing required file in source directory: " + file_spec.name;
                return false;
            }
        }

        fs::path dest = m_ref_dir / file_spec.name;
        fs::remove(dest, ec);

        if (use_symlinks) {
            fs::create_symlink(fs::canonical(candidate), dest, ec);
            if (ec) {
                // If symlink failed (e.g. filesystem limitation), try hardlink or copy
                ec.clear();
                fs::create_hard_link(candidate, dest, ec);
                if (ec) {
                    err_msg = "Failed to link " + file_spec.name + ": " + ec.message();
                    return false;
                }
            }
        } else {
            fs::create_hard_link(candidate, dest, ec);
            if (ec) {
                fs::copy_file(candidate, dest, fs::copy_options::overwrite_existing, ec);
                if (ec) {
                    err_msg = "Failed to copy " + file_spec.name + ": " + ec.message();
                    return false;
                }
            }
        }

        file_spec.local_path = dest.string();
        file_spec.blob_id = file_spec.name;
        register_blob(file_spec.blob_id, file_spec.size_bytes, dest.string());
    }

    // Save manifest
    if (!save_manifest(imported)) {
        err_msg = "Failed to write manifest file into store";
        return false;
    }

    return true;
}

bool ModelStore::verify_model(const ModelManifest& manifest, bool verify_hashes, std::string& err_msg) const {
    for (const auto& f : manifest.files) {
        if (f.local_path.empty()) {
            err_msg = "File has no local path configured: " + f.name;
            return false;
        }

        if (!fs::exists(f.local_path)) {
            err_msg = "File does not exist: " + f.local_path;
            return false;
        }

        std::error_code ec;
        uint64_t actual_size = fs::file_size(f.local_path, ec);
        if (ec) {
            err_msg = "Cannot read size of " + f.local_path + ": " + ec.message();
            return false;
        }

        if (f.size_bytes > 0 && actual_size != f.size_bytes) {
            err_msg = "File size mismatch for " + f.name + ": expected " +
                      std::to_string(f.size_bytes) + " bytes, got " +
                      std::to_string(actual_size) + " bytes";
            return false;
        }

        if (verify_hashes && !f.sha256.empty()) {
            std::string actual_hash = Sha256::hash_file(f.local_path);
            if (actual_hash != f.sha256) {
                err_msg = "SHA-256 hash mismatch for " + f.name + ": expected " +
                          f.sha256 + ", got " + actual_hash;
                return false;
            }
        }
    }
    return true;
}

} // namespace guild::models
