#include "guild/cli/tui.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void require_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "tui_test failure: " << msg << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    // 1. Test Chat + Reasoning + Telemetry in 132x38 split-sidebar mode
    {
        std::ostringstream captured;
        std::streambuf* old_cout = std::cout.rdbuf(captured.rdbuf());

        guild::cli::TuiOptions opts;
        opts.initial_model = "qwen3.8-flash-next";
        opts.force_mock = true;
        opts.override_cols = 132;
        opts.override_rows = 38;
        opts.dump_screen_on_exit = true;
        opts.script_commands = {
            "/strategy adaptive",
            "/draft 4",
            "Explain sparse MoE routing.",
        };

        int rc = guild::cli::run_tui(opts);
        std::cout.rdbuf(old_cout);
        require_true(rc == 0, "run_tui returned non-zero in test 1");

        const std::string screen = captured.str();
        require_true(screen.find("GUILD TUI") != std::string::npos, "missing GUILD TUI header");
        require_true(screen.find("HARDWARE MONITOR") != std::string::npos, "missing HARDWARE MONITOR section");
        require_true(screen.find("KV & EXPERT RESIDENCY") != std::string::npos, "missing KV & EXPERT RESIDENCY");
        require_true(screen.find("THROUGHPUT & SPECULATION") != std::string::npos, "missing THROUGHPUT & SPECULATION");
        require_true(screen.find("GEXEC OPTIMIZATION") != std::string::npos, "missing GEXEC OPTIMIZATION");
        require_true(screen.find("You > Explain sparse MoE routing.") != std::string::npos, "missing user prompt");
        require_true(screen.find("[Reasoning (Ctrl+R to fold)]") != std::string::npos, "missing reasoning header");
        require_true(screen.find("Guild > Guild native TUI session active") != std::string::npos, "missing answer");
    }

    // 2. Test Reasoning Collapse (Ctrl+R) and Model Selector View (F2)
    {
        std::ostringstream captured;
        std::streambuf* old_cout = std::cout.rdbuf(captured.rdbuf());

        guild::cli::TuiOptions opts;
        opts.initial_model = "ornith:35b";
        opts.force_mock = true;
        opts.override_cols = 120;
        opts.override_rows = 34;
        opts.dump_screen_on_exit = true;
        opts.script_commands = {
            "Hello Ornith",
            "WAIT:20",
            "KEY:CTRL_R",
        };

        int rc = guild::cli::run_tui(opts);
        std::cout.rdbuf(old_cout);
        require_true(rc == 0, "run_tui returned non-zero in test 2");

        const std::string screen = captured.str();
        require_true(screen.find("Reasoning collapsed") != std::string::npos, "missing collapsed reasoning banner");
    }

    // 3. Test Generation Cancellation (Esc during slow streaming)
    {
        std::ostringstream captured;
        std::streambuf* old_cout = std::cout.rdbuf(captured.rdbuf());

        guild::cli::TuiOptions opts;
        opts.initial_model = "qwen3.8-flash-next";
        opts.force_mock = true;
        opts.mock_delay_ms = 50;
        opts.override_cols = 120;
        opts.override_rows = 32;
        opts.dump_screen_on_exit = true;
        opts.script_commands = {
            "Stream a long response",
            "WAIT:60",
            "KEY:ESC",
            "WAIT:80",
        };

        int rc = guild::cli::run_tui(opts);
        std::cout.rdbuf(old_cout);
        require_true(rc == 0, "run_tui returned non-zero in test 3");

        const std::string screen = captured.str();
        require_true(screen.find("Generation cancelled by user") != std::string::npos, "missing cancel banner");
    }

    // 4. Test Narrow Terminal Resize (80x24) and Telemetry View (F4)
    {
        std::ostringstream captured;
        std::streambuf* old_cout = std::cout.rdbuf(captured.rdbuf());

        guild::cli::TuiOptions opts;
        opts.initial_model = "ornith:35b";
        opts.force_mock = true;
        opts.override_cols = 80;
        opts.override_rows = 24;
        opts.dump_screen_on_exit = true;
        opts.script_commands = {
            "KEY:F4",
        };

        int rc = guild::cli::run_tui(opts);
        std::cout.rdbuf(old_cout);
        require_true(rc == 0, "run_tui returned non-zero in test 4");

        const std::string screen = captured.str();
        require_true(screen.find("Deep-Dive Runtime, Speculative & GEXEC Telemetry") != std::string::npos,
                     "missing F4 telemetry view");
    }

    std::cout << "All Guild TUI tests passed.\n";
    return 0;
}
