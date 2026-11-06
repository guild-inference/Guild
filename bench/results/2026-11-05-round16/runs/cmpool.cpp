#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
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
    static void pin(int cpu) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); pthread_setaffinity_np(pthread_self(), sizeof(set), &set); }
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
class PoolF {
public:
    /// cpus[n]: the logical CPUs of node n's workers (a worker per CPU)
    explicit PoolF(const std::vector<std::vector<int>>& cpus) {
        for (int node = 0; node < (int) cpus.size(); ++node)
            for (int cpu : cpus[(size_t) node])
                workers_.emplace_back([this, node, cpu] { loop(node, cpu); });
    }
    ~PoolF() {
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
    static void pin(int cpu) { cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set); pthread_setaffinity_np(pthread_self(), sizeof(set), &set); }
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
                const uint64_t t = w.fetch_add(1, std::memory_order_acq_rel);
                if (idx_of(t) < n_of(t)) {
                    (*fn_.p)(q, (int) idx_of(t));
                    done_[node].v.fetch_add(1, std::memory_order_release);
                }
                if (gen_of(t) != g)
                    break;
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
static void spin_ns(int ns) { auto t0 = std::chrono::steady_clock::now(); while (std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() < ns) {} }
template <class P> double bench(P& p, int n, int work_ns, int iters) {
    std::vector<int> hit(n * 2);
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it)
        p.run_nodes(n, n, [&](int node, int i) { spin_ns(work_ns); hit[node * n + i]++; });
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    for (int i = 0; i < 2 * n; ++i) if (hit[i] != iters) { std::printf("BAD item %d ran %d times\n", i, hit[i]); std::exit(1); }
    return us;
}
int main(int argc, char** argv) {
    const int n = std::atoi(argv[1]), work = std::atoi(argv[2]), iters = argc > 3 ? std::atoi(argv[3]) : 20000;
    std::vector<std::vector<int>> cpus(2);
    for (int c = 1; c < 16; ++c) cpus[0].push_back(c);
    for (int c = 16; c < 32; ++c) cpus[1].push_back(c);
    { cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
    double a, b, a2, b2;
    { Pool p(cpus); bench(p, n, work, 200); a = bench(p, n, work, iters); }
    { PoolF p(cpus); bench(p, n, work, 200); b = bench(p, n, work, iters); }
    { Pool p(cpus); a2 = bench(p, n, work, iters); }
    { PoolF p(cpus); b2 = bench(p, n, work, iters); }
    std::printf("items %d x2, work %d ns: CAS %.1f / %.1f us, fetch_add %.1f / %.1f us a phase (ideal %.1f)\n", n, work, a, a2, b, b2, 2.0 * n * work / 32 / 1e3);
}
