#include "../check.hpp"
#include "gguf_fixture.hpp"
#include "../../src/runtime/model_impl.hpp"
#include <unistd.h>

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("guild-ple-binding-" + std::to_string(getpid()));
    fs::create_directories(root);
    const std::string path = (root / "metadata.gguf").string();
    fixture::write(path, {fixture::str("general.architecture", "qwen4exp")}, {});
    guild::runtime::GuildModel::Impl model;
    model.desc.archetype = guild::model::ModelArchetype::Qwen4Exp;
    std::string err;
    CHECK(!guild::runtime::bind_required_ple(model, {path}, err));
    CHECK(err.find("required PLE table") != std::string::npos);
    CHECK(model.ple_required && !model.ss.ple.ready());
    // Even a supplied external table cannot bypass missing required metadata.
    model.paths.ple_gguf = (root / "absent-table.gguf").string();
    CHECK(!guild::runtime::bind_required_ple(model, {path}, err));
    CHECK(err.find("required PLE metadata") != std::string::npos);
    CHECK(!model.ss.ple.ready());
    fs::remove_all(root);
    std::puts("ple_binding_failure_test: mandatory PLE cannot silently become an ablated model");
}
