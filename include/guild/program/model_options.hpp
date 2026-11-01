// include/guild/program/model_options.hpp - the model-loading options of the command-line runners (guild-generate,
// decision_run): their parsing, their usage text, and the load itself (descriptor from the GGUF, the plan for this
// machine, the model and a session).  Model-agnostic: every runner that loads through GuildModel shares it.
#pragma once

#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace guild::program {

struct ModelOptions {
    std::string pack_dir;
    std::string native_path;
    std::string profile_path;
    std::string mtp_dir;
    std::string tokenizer_dir; ///< default: <pack_dir>/tokenizer
    std::string drama3_dir;
    int64_t max_context = 262144;
    std::string kv_precision = "fp16";
    bool kv_host_only = false;
    int resident_budget_gib = 0;
    std::string expert_cache = "auto";
    double pcie_fraction = -1.0;
    std::string prefill = "auto";
    int64_t prefill_rows = 0; ///< --prefill-rows: compute the prompt chunks as chunks of this many tokens (0: off)
    int spec = 4;
    float spec_min_p = 0.5f;
    std::string decode_strategy;
    std::string rope_scaling;
    bool mtp_optional = false;
    std::string ple_gguf, ple_io;
    double rope_scale = 0;
    int pool_workers = 0;
};

/// argv[i] as a model option: consumes it and its value (advancing i) and returns 1; 0 when it is not one; 2 when its
/// value is invalid (the message is printed, prefixed by `program` where guild-generate's was).  A missing value
/// exits with status 2, as guild-generate always did.
int parse_model_option(int argc, char** argv, int& i, ModelOptions& o, const char* program);

/// The options' lines for a runner's --help.
const char* model_options_usage();

struct LoadedModel {
    std::unique_ptr<runtime::GuildModel> model;
    std::unique_ptr<runtime::GuildSession> session;
    int64_t context = 0; ///< the context the session holds (at most --max-context)
};

/// Loads the model the options name and a session on it.  0 on success; 2 for options the plan refuses; 1 when the
/// load fails.  Messages go to stderr, prefixed by `program`.
int load_model(ModelOptions& o, const char* program, LoadedModel& out);

} // namespace guild::program
