#include "../check.hpp"
#include "gguf_fixture.hpp"
#include "guild/runtime/model.hpp"
#include "guild/server/native_engine.hpp"

#include <filesystem>
#include <fstream>
#include <unistd.h>

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("guild-load-failure-" + std::to_string(getpid()));
    fs::create_directories(root);
    guild::server::NativeInferenceEngineOptions opts;
    opts.plan.context_length = 64;
    opts.plan.mtp_spec_tokens = 2;
    auto rejects = [&]() {
        std::string err = "old error";
        auto model = guild::runtime::GuildModel::load(opts.paths, opts.desc, opts.plan, err);
        CHECK(!model && !err.empty() && err != "old error");
        guild::server::NativeInferenceEngine engine(opts);
        CHECK(!engine.init(err));
        CHECK(!engine.is_ready());
        guild::server::GenerationResult result;
        result.text = "stale completion";
        guild::server::InferenceRequest req;
        req.prompt_tokens = {1};
        CHECK(!engine.generate(req, result));
        CHECK(result.finish_reason == "error" && !result.error_message.empty());
        CHECK(result.text.empty() && result.tokens.empty() && result.completion_tokens == 0);
    };
    rejects(); // absent pack
    opts.paths.pack_dir = root.string();
    { std::ofstream index(root / "index.txt"); index << "not a pack\n"; }
    rejects(); // absent primary file
    opts.paths.primary_model_path = (root / "absent.gguf").string();
    rejects();
    opts.paths.primary_model_path = (root / "model-00001-of-00002.gguf").string();
    fixture::write(opts.paths.primary_model_path, fixture::split_keys(0, 2, 1), {{"token_embd.weight", {2560, 16}, 8}});
    rejects(); // incomplete split must not fall back to its first shard
    opts.paths.primary_model_path = (root / "wrong-architecture.gguf").string();
    fixture::write(opts.paths.primary_model_path, {fixture::str("general.architecture", "mixtral")},
                   {{"token_embd.weight", {2560, 16}, 8}});
    rejects();
    fs::remove_all(root);
    std::puts("runtime_load_failure_test: unavailable/malformed/partial models cannot become ready or generate");
}
