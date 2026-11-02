// include/guild/core/cpu_moe.hpp - feed-forward networks on the CPU for the engines that keep their routed experts in
// RAM (GLM-5.3-Flash, Nemotron-H): the worker pool, the experts' rows split across NUMA nodes, and the expert phases.
//
// Weights stay in their GGUF formats and every product is ggml-cpu's dot product (or guild-kernels-cpu's multi-token
// kernels, bitwise the same per token), so a token's FFN output is llama.cpp's CPU backend's whatever the window.
// Two activations: SwiGLU, silu(gate) * up with an optional clamp (GLM), and the ungated relu(up)^2 (Nemotron-H).
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace guild::core::cpu_moe {

/// A thread that runs one task at a time for the calling thread: a pipelined window's GPU launches, so that the
/// calling thread starts the other group's CPU experts at once.
class Launcher {
public:
    Launcher(int cpu, int device);
    ~Launcher();
    void post(std::function<void()> f);
    void wait();

private:
    void loop(int cpu, int device);
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<void()> task_;
    bool pending_ = false, stop_ = false;
    std::thread th_;
};

/// Pins the calling thread to `cpu` (the core the pool leaves it) for one call into an engine and gives it back its
/// affinity afterwards.  It runs a share of every phase; unpinned, the scheduler sometimes kept it on a hyper-thread
/// sibling of a worker or on the other node for a whole run (2x Xeon Gold 6130: CPU experts 33 -> 41 ms a token).
class PinCaller {
public:
    explicit PinCaller(int cpu);
    ~PinCaller();
    PinCaller(const PinCaller&) = delete;
    PinCaller& operator=(const PinCaller&) = delete;

private:
    alignas(8) unsigned char saved_[128]; ///< a cpu_set_t
    bool pinned_ = false;
};

/// The pool: a worker per CPU of the node lists it is given (the calling thread joins every phase as node 0's) and a
/// queue per NUMA node.  A queue is one 64-bit word on a cache line of its own - the phase's generation, its item
/// count and the next item - so a worker claims an item of the phase it woke for with a compare-and-swap, or none: one
/// that wakes late never takes the next phase's items.  Workers count the items they finish on their node's line, and
/// the calling thread returns once every item is done, without waiting for the workers that found none (a worker the
/// scheduler parked behind another process holds up only an item it has claimed).
class Pool {
public:
    /// cpus[n]: the logical CPUs of node n's workers (a worker per CPU)
    explicit Pool(const std::vector<std::vector<int>>& cpus) {
        for (int node = 0; node < (int) cpus.size(); ++node)
            for (int cpu : cpus[(size_t) node])
                workers_.emplace_back([this, node, cpu] { loop(node, cpu); });
    }
    ~Pool() {
        stop_.store(true);
        publish(0, 0);
        { std::lock_guard<std::mutex> lk(mu_); }
        cv_.notify_all();
        for (auto& w : workers_)
            w.join();
    }
    int size() const {
        return (int) workers_.size() + 1;
    }
    /// fn(i) for i in [0, n), any node
    void run(int n, const std::function<void(int)>& fn) {
        run_nodes(n, 0, [&](int, int i) { fn(i); });
    }
    /// fn(node, i) for i in [0, n0) on node 0's queue and [0, n1) on node 1's; a worker drains its node's queue
    /// first, then helps the other
    void run_nodes(int n0, int n1, const std::function<void(int, int)>& fn) {
        if (n0 + n1 <= 0)
            return;
        constexpr int kMax = (int) kIdx; // a queue's items in one phase; more go in several phases
        if (n0 > kMax || n1 > kMax) {
            for (int b = 0; b < n0 || b < n1; b += kMax)
                run_nodes(std::clamp(n0 - b, 0, kMax), std::clamp(n1 - b, 0, kMax),
                          [&](int node, int i) { fn(node, b + i); });
            return;
        }
        fn_.p = &fn;
        done_[0].v.store(0, std::memory_order_relaxed);
        done_[1].v.store(0, std::memory_order_relaxed);
        const uint64_t g = publish(n0, n1);
        if (sleepers_.v.load(std::memory_order_seq_cst) > 0) {
            {
                std::lock_guard<std::mutex> lk(mu_);
            }
            cv_.notify_all();
        }
        work(0, g);
        while (done_[0].v.load(std::memory_order_acquire) + done_[1].v.load(std::memory_order_acquire) < n0 + n1)
            relax();
    }

private:
    // a queue word: generation (24 bits) | item count (20) | next item (20)
    static constexpr uint64_t kIdx = (1u << 20) - 1;
    static uint64_t gen_of(uint64_t w) {
        return w >> 40;
    }
    static uint64_t n_of(uint64_t w) {
        return (w >> 20) & kIdx;
    }
    static uint64_t idx_of(uint64_t w) {
        return w & kIdx;
    }
    static void relax() {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    static void pin(int cpu);
    /// the calling thread's: the next generation (never 0, which no phase has) on both queues
    uint64_t publish(int n0, int n1) {
        gen_ = gen_ % ((1u << 24) - 1) + 1;
        q_[0].w.store(gen_ << 40 | (uint64_t) n0 << 20, std::memory_order_seq_cst);
        q_[1].w.store(gen_ << 40 | (uint64_t) n1 << 20, std::memory_order_seq_cst);
        return gen_;
    }
    /// the items of generation g: node's queue first, then the other's
    void work(int node, uint64_t g) {
        for (int q : {node, 1 - node}) {
            std::atomic<uint64_t>& w = q_[q].w;
            uint64_t v = w.load(std::memory_order_acquire);
            while (gen_of(v) == g && idx_of(v) < n_of(v)) {
                if (!w.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel, std::memory_order_acquire))
                    continue;
                (*fn_.p)(q, (int) idx_of(v));
                done_[node].v.fetch_add(1, std::memory_order_release);
                v = w.load(std::memory_order_acquire);
            }
        }
    }
    void loop(int node, int cpu) {
        pin(cpu);
        uint64_t seen = 0;
        std::atomic<uint64_t>& w = q_[node].w;
        for (;;) {
            // spin a while for the next phase (a layer's phases follow each other within microseconds), then sleep
            uint64_t g = gen_of(w.load(std::memory_order_acquire));
            for (int s = 0; g == seen && s < 200000; ++s) {
                relax();
                g = gen_of(w.load(std::memory_order_acquire));
            }
            if (g == seen) {
                std::unique_lock<std::mutex> lk(mu_);
                sleepers_.v.fetch_add(1, std::memory_order_seq_cst);
                while ((g = gen_of(w.load(std::memory_order_seq_cst))) == seen)
                    cv_.wait(lk);
                sleepers_.v.fetch_sub(1, std::memory_order_relaxed);
            }
            if (stop_.load())
                return;
            seen = g;
            work(node, g);
        }
    }
    // what workers read or write on every item, a cache line each (the two nodes' sockets share none)
    struct alignas(128) Word {
        std::atomic<uint64_t> w{0};
    };
    struct alignas(128) Count {
        std::atomic<int> v{0};
    };
    struct alignas(128) Fn {
        const std::function<void(int, int)>* p = nullptr;
    };
    Word q_[2];
    Count done_[2];
    Fn fn_;
    Count sleepers_;
    uint64_t gen_ = 0;
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::thread> workers_;
};

/// A pool of a worker per physical core, grouped by NUMA node (two groups when `split` and the machine has two
/// nodes), less the first core, which the calling thread takes (caller_cpu) and a thread budget (`threads` > 0)
/// trims alike on both nodes; and a launcher on that core's other hyper-thread.
struct Workers {
    std::unique_ptr<Pool> pool;
    std::unique_ptr<Launcher> launcher;
    int caller_cpu = -1;
    bool split = false; ///< two NUMA nodes
};
Workers make_workers(bool numa_split, int threads);

/// An FFN on the CPU in its GGUF formats: gate/up rows of n_embd values, down rows of n_ff (one expert's slices of the
/// stacked tensors, or a layer's own FFN).  Row r of a matrix is at base[node] + r * row bytes, node = r >= split:
/// two NUMA nodes each hold half the rows (node 1's base is offset back by its first row so a row keeps its global
/// index); unsplit, both bases are one.
struct CpuFfn {
    enum Act : int {
        SwiGLU = 0,
        ReLU2 = 1
    };
    int act = SwiGLU;
    int gu_type = -1, d_type = -1;
    int64_t n_ff = 0;
    const uint8_t* gate[2] = {nullptr, nullptr}; ///< null for ReLU2
    const uint8_t* up[2] = {nullptr, nullptr};
    const uint8_t* down[2] = {nullptr, nullptr};
    int64_t gu_split = 0, d_split = 0; ///< the first gate/up row and the first down row of node 1
    size_t gu_row = 0, d_row = 0;
    float lim = 0.f; ///< SwiGLU's clamp (0: none)
};

/// Copies the FFNs' rows into memory bound to each NUMA node (gate/up rows [n h, (n + 1) h) and down rows
/// [n dh, (n + 1) dh) on node n; one node: all of them, for the NL4 repack) and points them there.  `nl4`: IQ3_S
/// gate/up rows repacked as NL4 (glm5_nl4.hpp).  The arenas are the caller's to munmap (arena_bytes).
bool place_ffns(Pool& pool, const std::vector<CpuFfn*>& all, int64_t n_embd, bool split, bool nl4, uint8_t* arena[2],
                size_t arena_bytes[2], std::string& err);

/// a GGUF type's activation type for gate/up rows (NL4 rows are IQ3_S's: Q8_K)
int act_type(int gu_type);

/// The expert phases of one layer for a window of T tokens: the routed experts (`ids` / `w`: n_used a token, into
/// `experts`) and, when `own` is given, an FFN every token goes through with weight 1.
///   activations: each token's row quantized once per activation type;
///   gate/up over (FFN, row block): an FFN's tokens share the weight decode (multi-token kernels from two tokens up),
///     and the item that completes a 256-row block of an FFN's hidden activations quantizes it;
///   down over row blocks of the output, each summing its rows over every unit of every token, so no two threads
///     write one row; in one pool phase with the gate/up items (GUILD_CPU_MOE_FLOW=0: two phases), a down item waiting
///     for a group only until that group's hidden rows are quantized.
/// Q8_0 rows take q8_0_rows512.hpp's AVX-512 kernel when the CPU has it, in every window: each (token, row) result is
/// bitwise ggml-cpu's one-token vec_dot.
/// x: T x n_embd floats (host), y: T x n_embd (host), written.  `single`: every (token, row) dot product as ggml-cpu's
/// one-token vec_dot computes it, so that a token's output does not depend on the other tokens of its window (a verify
/// window's rows equal decode steps; guild-kernels-cpu's multi-token kernels are bitwise vec_dot per token only within
/// one row partition, and the partition follows the window's experts).
class Experts {
public:
    void run(Pool& pool, int64_t n_embd, int T, int max_t, const int32_t* ids, const float* w, int n_used,
             const std::vector<CpuFfn>& experts, const CpuFfn* own, const float* x, float* y, bool single = false);
    /// GUILD_GLM_TIMING-style phase totals, milliseconds: activations, gate/up, hidden, down, and gate/up and down as
    /// one phase (GUILD_CPU_MOE_FLOW); and the weight bytes the gate/up and down phases read (each FFN's rows once)
    double phase_ms[5] = {0, 0, 0, 0, 0};
    double weight_bytes = 0, gu_bytes = 0, down_bytes = 0; ///< both phases', the gate/up phase's, the down phase's
    /// The rows of every group a down item takes (4-64; the last items of each node take a quarter).  16: with 64 a
    /// phase's last items kept a few threads busy while the rest waited (2x Xeon Gold 6130, 32 threads: a thread's
    /// last item ended 35 us before the phase on average, 16 us with 16 rows), before those quarter items; Nemotron
    /// 3.5's Q8_0 rows are still ~1% faster with 16, GLM-5.3-Flash's IQ4_XS rows ~1% with 64 (glm5.cpp).
    int down_rows = 16;
    /// the gate/up, down and one-phase item times summed over the threads ([1], [3], [4]): busy_ms / (phase_ms * the
    /// pool's threads) is the share of the threads' time spent in items
    double busy_ms[5] = {0, 0, 0, 0, 0};

private:
    std::vector<uint8_t> actbuf_, hbuf_, hqbuf_, pkbuf_;
    std::vector<int> qneed_;
    std::unique_ptr<std::atomic<int>[]> qcnt_;
    /// the one phase's per-group counts of quantized blocks and ready flags, a cache line each: a waiting down item
    /// reads its group's flag while gate/up items count other groups' blocks
    struct alignas(128) Line {
        std::atomic<int> v{0};
    };
    std::unique_ptr<Line[]> gflow_;
    size_t qcnt_cap_ = 0, gflow_cap_ = 0;
};

} // namespace guild::core::cpu_moe
