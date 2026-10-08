#pragma once

#include "guild/server/engine.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "guild/runtime/types.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/memory/plan.hpp"

#include <memory>
#include <string>

namespace guild::server {

struct NativeInferenceEngineOptions {
    std::string model_name = "Qwen3.8-Flash-Next";
    guild::runtime::ModelPaths paths;
    guild::model::ModelDescriptor desc;
    guild::memory::ExecutionPlan plan;
};

class NativeInferenceEngine : public IInferenceEngine {
public:
    explicit NativeInferenceEngine(NativeInferenceEngineOptions options);
    ~NativeInferenceEngine() override;

    bool init(std::string& error_msg);

    std::string model_name() const override { return options_.model_name; }
    int64_t max_context() const override { return options_.plan.context_length; }
    bool is_ready() const override { return ready_ && model_ != nullptr && model_->is_ready() && session_ != nullptr; }

    bool generate(const InferenceRequest& req, GenerationResult& result) override;
    bool generate_stream(const InferenceRequest& req,
                         StreamCallback on_token,
                         GenerationResult& result) override;
    void stop() override;

    guild::runtime::GuildModel* model() { return model_.get(); }
    const guild::runtime::GuildModel* model() const { return model_.get(); }
    guild::runtime::GuildSession* session() { return session_.get(); }
    const guild::runtime::GuildSession* session() const { return session_.get(); }

private:
    NativeInferenceEngineOptions options_;
    std::unique_ptr<guild::runtime::GuildModel> model_;
    std::unique_ptr<guild::runtime::GuildSession> session_;
    bool ready_ = false;
};

} // namespace guild::server
