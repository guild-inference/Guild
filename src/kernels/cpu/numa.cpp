// src/kernels/cpu/numa.cpp - the CPU experts' NUMA split.  Read numa.hpp first.
#include "guild/kernels/cpu/numa.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <sched.h>
#endif

namespace guild::kernels::cpu {
namespace {

#if defined(__linux__)
// "0-15,32-47" -> {0..15, 32..47}
std::vector<int> parse_cpulist(const std::string& text) {
    std::vector<int> cpus;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        if (item.empty() || item[0] < '0' || item[0] > '9')
            continue;
        const auto dash = item.find('-');
        const int a = std::atoi(item.c_str());
        const int b = dash == std::string::npos ? a : std::atoi(item.c_str() + dash + 1);
        for (int c = a; c <= b && c - a < 65536; ++c)
            cpus.push_back(c);
    }
    return cpus;
}
#endif

NumaTopology detect() {
    NumaTopology t;
#if defined(__linux__)
    std::vector<std::vector<int>> nodes;
    for (int n = 0; n < 64; ++n) {
        std::ifstream f("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (!f.is_open())
            continue;
        std::string line;
        std::getline(f, line);
        auto cpus = parse_cpulist(line);
        if (cpus.empty())
            continue; // a memory-only node has no threads to give work to
        if ((int) nodes.size() != n)
            return t; // a gap in the numbering: keep the single-node answer rather than guess
        nodes.push_back(std::move(cpus));
    }
    if (nodes.size() < 2)
        return t;
    t.nodes = (int) nodes.size();
    for (int n = 0; n < t.nodes; ++n)
        for (int c : nodes[(size_t) n]) {
            if ((size_t) c >= t.cpu_node.size())
                t.cpu_node.resize((size_t) c + 1, -1);
            t.cpu_node[(size_t) c] = n;
        }
    t.node_cpus = std::move(nodes);
#endif
    return t;
}

} // namespace

const NumaTopology& numa_topology() {
    static const NumaTopology topology = detect();
    return topology;
}

bool numa_split_requested() {
    static const bool on = [] {
        const char* v = std::getenv("GUILD_EXPERT_NUMA");
        if (v && std::strcmp(v, "split") == 0)
            return numa_topology().nodes == 2;
        if (v && *v)
            return false;
        return numa_topology().nodes == 2;
    }();
    return on;
}

int current_numa_node() {
#if defined(__linux__)
    const int cpu = sched_getcpu();
    const auto& t = numa_topology();
    if (cpu >= 0 && (size_t) cpu < t.cpu_node.size())
        return t.cpu_node[(size_t) cpu];
#endif
    return -1;
}

int numa_node0_ranges(const guild_cpu_expert_format_v1_t& f, bool gate_up_interleaved, uint64_t ranges[3][2]) {
    if (f.n_ff <= 0 || f.n_embd <= 0 || f.gu_row == 0 || f.d_row == 0 || f.bytes == 0)
        return 0;
    const uint64_t ff_half = (uint64_t) numa_half_rows((int) f.n_ff);
    const uint64_t embd_half = (uint64_t) numa_half_rows((int) f.n_embd);
    int n = 0;
    if (gate_up_interleaved) {
        ranges[n][0] = 0;
        ranges[n++][1] = 2 * ff_half * f.gu_row;
    } else {
        ranges[n][0] = 0;
        ranges[n++][1] = ff_half * f.gu_row;
        ranges[n][0] = f.up_off;
        ranges[n++][1] = f.up_off + ff_half * f.gu_row;
    }
    ranges[n][0] = f.down_off;
    ranges[n++][1] = f.down_off + embd_half * f.d_row;
    return n;
}

} // namespace guild::kernels::cpu
