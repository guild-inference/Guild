#include "guild/cli/hardware.hpp"
#include "guild/cli/ansi.hpp"

#include <cstdio>
#include <iostream>

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "guild_cli_test failed at line %d: %s\n", __LINE__, #cond); \
        return 1; \
    } \
} while (0)

int main() {
    const auto hw = guild::cli::detect_hardware();

    // Check CPU detection
    CHECK(!hw.cpu_model.empty());
    CHECK(hw.cpu_physical_cores > 0);
    CHECK(hw.cpu_logical_threads > 0);
    CHECK(hw.cpu_logical_threads >= hw.cpu_physical_cores);

    // Check RAM detection
    CHECK(hw.ram_total_gib > 0.0);

    // Check summaries
    const std::string s_cpu = hw.summary_cpu();
    CHECK(!s_cpu.empty());
    const std::string s_ram = hw.summary_ram();
    CHECK(!s_ram.empty());
    const std::string s_gpu = hw.summary_gpu();
    CHECK(!s_gpu.empty());

    std::cout << "guild_cli_test detected hardware:\n"
              << "  CPU: " << s_cpu << " (" << hw.cpu_physical_cores << " cores / "
              << hw.cpu_logical_threads << " threads)\n"
              << "  RAM: " << s_ram << "\n"
              << "  GPU: " << s_gpu << "\n";

    std::cout << "guild_cli_test: all checks PASSED\n";
    return 0;
}
