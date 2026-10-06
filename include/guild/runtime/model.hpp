#pragma once

#include "guild/model/model_descriptor.hpp"
#include "guild/memory/plan.hpp"
#include "guild/runtime/types.hpp"
#include "guild/runtime/tokenizer.hpp"

#include <memory>
#include <string>

namespace guild::runtime {

class GuildSession;

class GuildModel {
public:
    // Loads the model weights, expert tiers, KV staging, MTP, and capture graphs into memory.
    // Keeps all weights and GPU contexts resident for multiple requests/sessions.
    static std::unique_ptr<GuildModel> load(
        const ModelPaths& paths,
        const model::ModelDescriptor& desc,
        const memory::ExecutionPlan& plan,
        std::string& error_msg
    );

    ~GuildModel();

    GuildModel(const GuildModel&) = delete;
    GuildModel& operator=(const GuildModel&) = delete;

    // Creates an isolated inference session with its own KV cache and conversation state
    std::unique_ptr<GuildSession> create_session(int64_t max_context = 0);

    const model::ModelDescriptor& descriptor() const { return desc_; }
    const memory::ExecutionPlan& plan() const { return plan_; }
    const ModelPaths& paths() const { return paths_; }
    const RuntimeConfig& config() const { return config_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    Tokenizer& tokenizer() { return tokenizer_; }

    bool is_ready() const { return ready_; }

    struct Impl;
    Impl* impl() { return impl_.get(); }
    const Impl* impl() const { return impl_.get(); }

private:
    GuildModel();
    model::ModelDescriptor desc_;
    memory::ExecutionPlan plan_;
    ModelPaths paths_;
    RuntimeConfig config_;
    Tokenizer tokenizer_;
    bool ready_ = false;
    std::unique_ptr<Impl> impl_;
};

} // namespace guild::runtime
