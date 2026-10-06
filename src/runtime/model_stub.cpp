#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

namespace guild::runtime {

struct GuildModel::Impl {};

GuildModel::GuildModel() : impl_(std::make_unique<Impl>()) {}
GuildModel::~GuildModel() = default;

std::unique_ptr<GuildModel> GuildModel::load(
    const ModelPaths& paths,
    const model::ModelDescriptor& desc,
    const memory::ExecutionPlan& plan,
    std::string& error_msg
) {
    (void) paths;
    (void) desc;
    (void) plan;
    error_msg = "Guild GPU runtime is disabled in CPU-only build";
    return nullptr;
}

std::unique_ptr<GuildSession> GuildModel::create_session(int64_t max_context) {
    (void) max_context;
    return nullptr;
}

} // namespace guild::runtime
