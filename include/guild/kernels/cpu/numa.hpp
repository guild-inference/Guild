// include/guild/kernels/cpu/numa.hpp - the CPU experts' NUMA split.
//
// On a two-socket PC the resident expert arena used to be page-interleaved across both nodes, and every pool
// thread read every expert's rows, so about half of all expert bytes crossed the socket link.  Measured on the
// dual Xeon Gold 6130 (32 threads, one per physical core, streaming reads): 125 GB/s from interleaved pages
// against 214 GB/s when each thread reads only its own node's memory.
//
// The split keeps one copy of every expert and gives each node one half of its rows: node 0 holds the first half
// of the gate rows, the first half of the up rows and the first half of the down rows, node 1 the rest.  The pool
// gives node n's threads the row ranges that node n holds (and lets an idle node take the other's work at the
// end), so every row is computed by the same arithmetic as before and the outputs are bitwise unchanged.
#pragma once

#include "guild/abi/cpu_expert_abi.h"

#include <cstdint>
#include <vector>

namespace guild::kernels::cpu {

/// The NUMA nodes that have CPUs (Linux `/sys/devices/system/node`); one node everywhere else.
struct NumaTopology {
    int nodes = 1;
    std::vector<int> cpu_node;               ///< by logical CPU id; -1 when unknown
    std::vector<std::vector<int>> node_cpus; ///< logical CPU ids of each node
};
const NumaTopology& numa_topology();

/// Whether the expert arena and the pool use the two-node row split: `GUILD_EXPERT_NUMA=split`, or unset on a
/// machine with exactly two nodes.  `interleave` and `process` turn it off.
bool numa_split_requested();

/// The node of the CPU the calling thread is on now, or -1.
int current_numa_node();

/// The first row of node 1's half of `rows` rows, on the pool's four-row task boundaries.
inline int numa_half_rows(int rows) {
    return rows >= 8 ? (rows / 4 / 2) * 4 : rows / 2;
}

/// Node 0's byte ranges of one native expert blob (`[begin, end)` pairs, up to three); the rest belongs to node 1.
/// `gate_up_interleaved` is the GEXEC layout (gate row r at 2r, up row r at 2r + 1).
int numa_node0_ranges(const guild_cpu_expert_format_v1_t& f, bool gate_up_interleaved, uint64_t ranges[3][2]);

} // namespace guild::kernels::cpu
