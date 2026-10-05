#pragma once

#include <cstdint>
#include <string>

namespace guild::hardware {

struct HardwareInfo {
    std::string cpu_model;
    int cpu_sockets = 1;
    int cpu_physical_cores = 0;
    int cpu_logical_threads = 0;

    uint64_t ram_total_bytes = 0;
    uint64_t ram_free_bytes = 0;

    int gpu_count = 0;
    std::string gpu_name;
    uint64_t gpu_vram_bytes = 0;
    bool has_cuda = false;

    bool isa_avx512 = false;
    bool isa_avx2 = false;
    bool isa_vnni = false;
    bool isa_vbmi = false;

    // Direct double fields preserved for compatibility
    double ram_total_gib = 0.0;
    double ram_free_gib = 0.0;
    double gpu_vram_gib = 0.0;

    void sync_gib() {
        if (ram_total_bytes > 0 && ram_total_gib == 0.0) {
            ram_total_gib = (double) ram_total_bytes / (1024.0 * 1024.0 * 1024.0);
        } else if (ram_total_gib > 0.0 && ram_total_bytes == 0) {
            ram_total_bytes = (uint64_t) (ram_total_gib * 1024.0 * 1024.0 * 1024.0);
        }
        if (ram_free_bytes > 0 && ram_free_gib == 0.0) {
            ram_free_gib = (double) ram_free_bytes / (1024.0 * 1024.0 * 1024.0);
        } else if (ram_free_gib > 0.0 && ram_free_bytes == 0) {
            ram_free_bytes = (uint64_t) (ram_free_gib * 1024.0 * 1024.0 * 1024.0);
        }
        if (gpu_vram_bytes > 0 && gpu_vram_gib == 0.0) {
            gpu_vram_gib = (double) gpu_vram_bytes / (1024.0 * 1024.0 * 1024.0);
        } else if (gpu_vram_gib > 0.0 && gpu_vram_bytes == 0) {
            gpu_vram_bytes = (uint64_t) (gpu_vram_gib * 1024.0 * 1024.0 * 1024.0);
        }
    }

    std::string summary_cpu() const;
    std::string summary_gpu() const;
    std::string summary_ram() const;
};

HardwareInfo detect_hardware();

}  // namespace guild::hardware
