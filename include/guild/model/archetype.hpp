#pragma once

#include "guild/model/model_descriptor.hpp"
#include "guild/artifact/gguf_reader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace guild::model {

/// Interface for model architecture adapters.
/// Each architecture describes its own GGUF tensor names, metadata decoding,
/// and model-specific geometry, delegating runtime tiering and scheduling to
/// generic Guild MoE runtime.
class IModelArchetype {
public:
    virtual ~IModelArchetype() = default;

    /// Primary archetype identifier.
    virtual ModelArchetype archetype() const = 0;

    /// Canonical architecture name (e.g. "qwen4exp", "qwen35moe", "glm").
    virtual const char* name() const = 0;

    /// Returns true if this adapter recognizes the GGUF file metadata.
    virtual bool matches(const GgufFile& gguf) const = 0;

    /// Extracts architecture-specific metadata from GGUF into ModelDescriptor.
    virtual bool describe(const GgufFile& gguf, ModelDescriptor& out, std::string& err) const = 0;
};

/// Registry of known model archetypes.
class ArchetypeRegistry {
public:
    static ArchetypeRegistry& instance();

    /// Register a new architecture adapter.
    void register_archetype(std::unique_ptr<IModelArchetype> arch);

    /// Find an archetype adapter by enum.
    const IModelArchetype* find(ModelArchetype type) const;

    /// Find an archetype adapter by canonical name or alias.
    const IModelArchetype* find(const std::string& name) const;

    /// Auto-detect matching archetype adapter from a loaded GGUF file.
    const IModelArchetype* detect(const GgufFile& gguf) const;

    /// Auto-detect and populate a ModelDescriptor from a GGUF file.
    bool describe_gguf(const GgufFile& gguf, ModelDescriptor& out, std::string& err) const;

    /// Read-only access to registered adapters.
    const std::vector<std::unique_ptr<IModelArchetype>>& archetypes() const {
        return registry_;
    }

private:
    ArchetypeRegistry();
    std::vector<std::unique_ptr<IModelArchetype>> registry_;
};

}  // namespace guild::model
