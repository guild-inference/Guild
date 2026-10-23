#pragma once

#include <string>
#include <vector>

namespace guild::cli {

struct TuiOptions {
    std::string initial_model;
    std::string data_dir;
    std::string gexec_path;
    std::string decode_strategy = "adaptive";
    int spec_max_draft = 4;
    int max_tokens = 256;
    int64_t prefill_batch = 0;
    float temperature = 0.0f;
    bool force_mock = false;
    int mock_delay_ms = 0;
    int override_cols = 0;
    int override_rows = 0;
    // Optional semicolon-separated commands/keys for deterministic automated testing
    std::vector<std::string> script_commands;
    bool dump_screen_on_exit = false;
};

int run_tui(const TuiOptions& opts);
int cmd_tui(int argc, char** argv);

} // namespace guild::cli
