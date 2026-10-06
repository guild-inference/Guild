#pragma once

#include "guild/models/manifest.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace guild::models {

struct StoreOptions {
    std::string custom_data_dir;
};

struct BlobInfo {
    std::string blob_id;        // SHA-256 or digest
    uint64_t size_bytes{0};
    int ref_count{1};
    std::string path;
};

class ModelStore {
public:
    explicit ModelStore(const StoreOptions& options = {});

    // Root store directories
    std::string data_dir() const { return root_dir_; }
    std::string models_dir() const;
    std::string manifests_dir() const;
    std::string blobs_dir() const;
    std::string refs_dir() const;

    // Initialization
    bool init();

    // Manifest operations
    std::vector<ModelManifest> list_manifests() const;
    std::optional<ModelManifest> get_manifest(const std::string& name_or_alias) const;
    bool has_manifest(const std::string& name_or_alias) const;
    bool save_manifest(const ModelManifest& manifest);
    bool remove_model(const std::string& name_or_alias);

    // Blob operations
    std::string get_blob_path(const std::string& blob_id) const;
    bool has_blob(const std::string& blob_id) const;
    bool register_blob(const std::string& blob_id, uint64_t size_bytes, const std::string& rel_or_abs_path);
    bool unregister_blob(const std::string& blob_id);

    // Model import without copying 94 GiB
    bool import_model_directory(const ModelManifest& base_manifest,
                                const std::string& source_dir,
                                bool use_symlinks,
                                std::string& err_msg);

    // Validation
    bool verify_model(const ModelManifest& manifest, bool verify_hashes, std::string& err_msg) const;

    // Path sanitization helper
    static bool is_safe_name(const std::string& name);
    static std::string default_data_dir();

private:
    std::string root_dir_;
    mutable std::map<std::string, BlobInfo> blob_index_;

    void load_blob_index() const;
    void save_blob_index() const;
    std::string index_path() const;
};

} // namespace guild::models
