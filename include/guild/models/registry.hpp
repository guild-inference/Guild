#pragma once

#include "guild/models/manifest.hpp"
#include "guild/models/store.hpp"

#include <optional>
#include <string>
#include <vector>

namespace guild::models {

class ModelRegistry {
public:
    static ModelRegistry& instance();

    // Query manifests
    std::vector<ModelManifest> list_available() const;
    std::vector<ModelManifest> list_installed(const ModelStore& store) const;

    // Lookup: installed first, then built-in, then local path
    std::optional<ModelManifest> resolve(const std::string& name_or_query, const ModelStore& store) const;

    // Lookup strictly among built-in manifests
    std::optional<ModelManifest> find_builtin(const std::string& name_or_alias) const;

    // Check if installed in store
    bool is_installed(const std::string& name_or_alias, const ModelStore& store) const;

private:
    ModelRegistry();
    std::vector<ModelManifest> builtins_;

    void init_builtins();
};

} // namespace guild::models
