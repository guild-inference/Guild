#include "guild/hardware/hardware_info.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/kernels/cpu/expert.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#if defined(GUILD_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace guild::hardware {

namespace {

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

void probe_linux_cpu(HardwareInfo& hw) {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) return;

    std::string line;
    std::set<std::string> physical_cores;
    std::set<std::string> physical_sockets;
    int logical_count = 0;
    std::string current_phys_id = "0";
    std::string current_core_id = "0";

    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            auto colon = line.find(':');
            if (colon != std::string::npos && hw.cpu_model.empty()) {
                hw.cpu_model = trim(line.substr(colon + 1));
            }
        } else if (line.rfind("processor", 0) == 0) {
            ++logical_count;
        } else if (line.rfind("physical id", 0) == 0) {
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                current_phys_id = trim(line.substr(colon + 1));
                physical_sockets.insert(current_phys_id);
            }
        } else if (line.rfind("core id", 0) == 0) {
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                current_core_id = trim(line.substr(colon + 1));
                physical_cores.insert(current_phys_id + ":" + current_core_id);
            }
        }
    }

    hw.cpu_logical_threads = logical_count;
    hw.cpu_physical_cores = (int) physical_cores.size();
    hw.cpu_sockets = std::max(1, (int) physical_sockets.size());
    if (hw.cpu_physical_cores == 0) hw.cpu_physical_cores = hw.cpu_logical_threads;
}

void probe_linux_ram(HardwareInfo& hw) {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) return;

    std::string line;
    uint64_t total_kb = 0, avail_kb = 0;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            std::istringstream iss(line.substr(9));
            iss >> total_kb;
        } else if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream iss(line.substr(13));
            iss >> avail_kb;
        }
    }
    hw.ram_total_bytes = total_kb * 1024ULL;
    hw.ram_free_bytes = avail_kb * 1024ULL;
    hw.ram_total_gib = (double) total_kb / (1024.0 * 1024.0);
    hw.ram_free_gib = (double) avail_kb / (1024.0 * 1024.0);
}

void probe_gpu(HardwareInfo& hw) {
#if defined(GUILD_ENABLE_CUDA)
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaSuccess && count > 0) {
        hw.gpu_count = count;
        hw.has_cuda = true;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            hw.gpu_name = prop.name;
            hw.gpu_vram_bytes = prop.totalGlobalMem;
            hw.gpu_vram_gib = (double) prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        }
        return;
    }
#endif
    // Basic fallback: check nvidia-smi if runtime not linked
    std::FILE* pipe = popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null", "r");
    if (pipe) {
        char buf[256];
        if (std::fgets(buf, sizeof(buf), pipe)) {
            std::string s(buf);
            auto comma = s.find(',');
            if (comma != std::string::npos) {
                hw.gpu_count = 1;
                hw.has_cuda = true;
                hw.gpu_name = trim(s.substr(0, comma));
                double mib = std::atof(trim(s.substr(comma + 1)).c_str());
                hw.gpu_vram_bytes = (uint64_t) (mib * 1024.0 * 1024.0);
                hw.gpu_vram_gib = mib / 1024.0;
            }
        }
        pclose(pipe);
    }
}

}  // namespace

HardwareInfo detect_hardware() {
    HardwareInfo hw;
    probe_linux_cpu(hw);
    probe_linux_ram(hw);
    probe_gpu(hw);

    const auto feat = kernels::cpu::cpu_features();
    hw.isa_avx512 = feat.avx512f && feat.avx512bw && feat.avx512vl;
    hw.isa_vnni = feat.avx512_vnni;
    hw.isa_vbmi = feat.avx512_vbmi;
    hw.isa_avx2 = kernels::cpu::cpu_avx2_ok();

    hw.sync_gib();
    return hw;
}

std::string HardwareInfo::summary_cpu() const {
    std::string s;
    if (cpu_sockets > 1) {
        s += std::to_string(cpu_sockets) + "x ";
    }
    std::string name = cpu_model;
    const std::string intel = "Intel(R) ";
    const std::string reg = "(R)";
    const std::string tm = "(TM)";
    size_t pos;
    while ((pos = name.find(intel)) != std::string::npos) name.erase(pos, intel.length());
    while ((pos = name.find(reg)) != std::string::npos) name.erase(pos, reg.length());
    while ((pos = name.find(tm)) != std::string::npos) name.erase(pos, tm.length());
    auto at = name.find(" CPU @");
    if (at != std::string::npos) name = name.substr(0, at);
    at = name.find(" @");
    if (at != std::string::npos) name = name.substr(0, at);

    s += trim(name);
    return s.empty() ? "x86_64 CPU" : s;
}

std::string HardwareInfo::summary_gpu() const {
    if (gpu_name.empty()) return "None (CPU-only)";
    std::string name = gpu_name;
    const std::string nvid = "NVIDIA GeForce ";
    const std::string nvid2 = "NVIDIA ";
    size_t pos;
    if ((pos = name.find(nvid)) != std::string::npos) name.erase(pos, nvid.length());
    else if ((pos = name.find(nvid2)) != std::string::npos) name.erase(pos, nvid2.length());

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f GiB", gpu_vram_gib);
    return trim(name) + " · " + buf;
}

std::string HardwareInfo::summary_ram() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f GiB", ram_total_gib);
    return std::string(buf);
}

}  // namespace guild::hardware
