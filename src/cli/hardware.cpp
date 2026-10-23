#include "guild/hardware/hardware_info.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/kernels/cpu/expert.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include "guild/platform/device_policy.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <intrin.h>
#include <windows.h>
#endif

#if defined(GUILD_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace guild::hardware {

namespace {

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

#if defined(_WIN32)
void probe_windows_cpu_and_ram(HardwareInfo& hw) {
    int cpuInfo[4] = {0, 0, 0, 0};
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = static_cast<unsigned int>(cpuInfo[0]);
    char brand[64] = {0};
    if (nExIds >= 0x80000004) {
        __cpuid(reinterpret_cast<int*>(brand), 0x80000002);
        __cpuid(reinterpret_cast<int*>(brand + 16), 0x80000003);
        __cpuid(reinterpret_cast<int*>(brand + 32), 0x80000004);
        hw.cpu_model = trim(std::string(brand));
    }

    SYSTEM_INFO sysInfo{};
    GetSystemInfo(&sysInfo);
    hw.cpu_logical_threads = static_cast<int>(sysInfo.dwNumberOfProcessors);
    if (hw.cpu_logical_threads <= 0) {
        hw.cpu_logical_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    if (len > 0) {
        std::vector<uint8_t> buffer(len);
        if (GetLogicalProcessorInformationEx(
                RelationAll, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &len)) {
            int phys_cores = 0;
            int packages = 0;
            size_t offset = 0;
            while (offset < len) {
                auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
                if (info->Size == 0)
                    break;
                if (info->Relationship == RelationProcessorCore)
                    ++phys_cores;
                else if (info->Relationship == RelationProcessorPackage)
                    ++packages;
                offset += info->Size;
            }
            if (phys_cores > 0)
                hw.cpu_physical_cores = phys_cores;
            if (packages > 0)
                hw.cpu_sockets = packages;
        }
    }
    if (hw.cpu_physical_cores <= 0)
        hw.cpu_physical_cores = hw.cpu_logical_threads;

    MEMORYSTATUSEX memStatus{};
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        hw.ram_total_bytes = static_cast<uint64_t>(memStatus.ullTotalPhys);
        hw.ram_free_bytes = static_cast<uint64_t>(memStatus.ullAvailPhys);
        hw.ram_total_gib = static_cast<double>(hw.ram_total_bytes) / (1024.0 * 1024.0 * 1024.0);
        hw.ram_free_gib = static_cast<double>(hw.ram_free_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
}
#else
void probe_linux_cpu(HardwareInfo& hw) {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open())
        return;

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
    if (hw.cpu_physical_cores == 0)
        hw.cpu_physical_cores = hw.cpu_logical_threads;
}

void probe_linux_ram(HardwareInfo& hw) {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open())
        return;

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
#endif

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
            hw.gpu_cc_major = prop.major;
            hw.gpu_cc_minor = prop.minor;
        }
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b > 0) {
            hw.gpu_vram_free_bytes = static_cast<uint64_t>(free_b);
            hw.gpu_vram_free_gib = static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
        }
        return;
    }
#endif
    // Fallback: query nvidia-smi if CUDA runtime is not linked
#if defined(_WIN32)
    std::FILE* pipe = _popen(
        "nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits 2>NUL", "r");
#else
    std::FILE* pipe = popen(
        "nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits 2>/dev/null",
        "r");
#endif
    if (pipe) {
        char buf[256];
        if (std::fgets(buf, sizeof(buf), pipe)) {
            std::string s(buf);
            std::vector<std::string> parts;
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                parts.push_back(trim(item));
            }
            if (parts.size() >= 2 && !parts[0].empty()) {
                hw.gpu_count = 1;
                hw.has_cuda = true;
                hw.gpu_name = parts[0];
                double total_mib = std::atof(parts[1].c_str());
                hw.gpu_vram_bytes = (uint64_t) (total_mib * 1024.0 * 1024.0);
                hw.gpu_vram_gib = total_mib / 1024.0;
                if (parts.size() >= 3) {
                    double free_mib = std::atof(parts[2].c_str());
                    hw.gpu_vram_free_bytes = (uint64_t) (free_mib * 1024.0 * 1024.0);
                    hw.gpu_vram_free_gib = free_mib / 1024.0;
                }
                if (parts.size() >= 4) {
                    auto dot = parts[3].find('.');
                    if (dot != std::string::npos) {
                        hw.gpu_cc_major = std::atoi(parts[3].substr(0, dot).c_str());
                        hw.gpu_cc_minor = std::atoi(parts[3].substr(dot + 1).c_str());
                    }
                }
            }
        }
#if defined(_WIN32)
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    }
}

} // namespace

HardwareInfo detect_hardware() {
    platform::configure_device_runtime_policy();
    HardwareInfo hw;
#if defined(_WIN32)
    probe_windows_cpu_and_ram(hw);
#else
    probe_linux_cpu(hw);
    probe_linux_ram(hw);
#endif
    probe_gpu(hw);

    const auto feat = kernels::cpu::cpu_features();
    hw.isa_avx512 = feat.os_avx512 && feat.avx512f && feat.avx512bw && feat.avx512dq && feat.avx512vl;
    hw.isa_vnni = hw.isa_avx512 && feat.avx512_vnni;
    hw.isa_vbmi = hw.isa_avx512 && feat.avx512_vbmi;
    hw.isa_bmi2 = feat.bmi2;
    hw.isa_avx512cd = hw.isa_avx512 && feat.avx512cd;
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
    while ((pos = name.find(intel)) != std::string::npos)
        name.erase(pos, intel.length());
    while ((pos = name.find(reg)) != std::string::npos)
        name.erase(pos, reg.length());
    while ((pos = name.find(tm)) != std::string::npos)
        name.erase(pos, tm.length());
    auto at = name.find(" CPU @");
    if (at != std::string::npos)
        name = name.substr(0, at);
    at = name.find(" @");
    if (at != std::string::npos)
        name = name.substr(0, at);

    s += trim(name);
    return s.empty() ? "x86_64 CPU" : s;
}

std::string HardwareInfo::summary_gpu() const {
    if (gpu_name.empty())
        return "None (CPU-only)";
    std::string name = gpu_name;
    const std::string nvid = "NVIDIA GeForce ";
    const std::string nvid2 = "NVIDIA ";
    size_t pos;
    if ((pos = name.find(nvid)) != std::string::npos)
        name.erase(pos, nvid.length());
    else if ((pos = name.find(nvid2)) != std::string::npos)
        name.erase(pos, nvid2.length());

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f GiB", gpu_vram_gib);
    return trim(name) + " · " + buf;
}

std::string HardwareInfo::summary_ram() const {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f GiB", ram_total_gib);
    return std::string(buf);
}

} // namespace guild::hardware
