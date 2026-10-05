#pragma once

#include <cstdint>
#include <string>

namespace guild::cli {

struct HardwareProfile {
    std::string cpu_model;
    int cpu_sockets = 1;
    int cpu_physical_cores = 0;
    int cpu_logical_threads = 0;

    double ram_total_gib = 0.0;
    double ram_free_gib = 0.0;

    int gpu_count = 0;
    std::string gpu_name;
    double gpu_vram_gib = 0.0;
    bool has_cuda = false;

    bool isa_avx512 = false;
    bool isa_avx2 = false;
    bool isa_vnni = false;
    bool isa_vbmi = false;

    std::string summary_cpu() const;
    std::string summary_gpu() const;
    std::string summary_ram() const;
};

HardwareProfile detect_hardware();

}  // namespace guild::cli
