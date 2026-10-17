// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "guild/kernels/cpu/pool.hpp"
#include "guild/core/progress.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "pool_affinity_win.hpp"
#else
#include <pthread.h>
#include <sched.h>
#include "pool_affinity_linux.hpp"
#endif

namespace guild::kernels::cpu {

namespace {
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
double thread_cpu_ms() {
#if defined(__linux__)
    timespec value{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value);
    return double(value.tv_sec) * 1000.0 + double(value.tv_nsec) / 1e6;
#elif defined(_WIN32)
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user))
        throw std::runtime_error("cannot read thread CPU time");
    ULARGE_INTEGER k{}, u{};
    k.LowPart=kernel.dwLowDateTime;k.HighPart=kernel.dwHighDateTime;
    u.LowPart=user.dwLowDateTime;u.HighPart=user.dwHighDateTime;
    return double(k.QuadPart + u.QuadPart) / 10000.0; // 100 ns units
#else
    return double(std::clock()) * 1000.0 / CLOCKS_PER_SEC;
#endif
}
}  // namespace

CpuTopology detect_cpu_topology(bool skip_first, PoolAffinity affinity) {
    CpuTopology topo;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        // Preserve group identity even when detailed core topology is unavailable.
        const WORD groups = GetActiveProcessorGroupCount();
        for (WORD group = 0; group < groups; ++group) {
            const DWORD count = GetActiveProcessorCount(group);
            if (count == 0 || count == (DWORD) -1 || count > 64) continue;
            for (DWORD i = 0; i < count; ++i) topo.worker_cores.push_back((int) group * 64 + (int) i);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }
    std::vector<char> buf(len);
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                         (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        const char* p = buf.data();
        const char* end = p + len;
        struct CoreDesc {
            uint8_t efficiency = 0;
            bool has_smt = false;
            std::vector<int> lps;
        };
        std::vector<CoreDesc> descs;
        while (p < end) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
            if (e->Relationship == RelationProcessorCore) {
                CoreDesc cd;
                cd.efficiency = e->Processor.EfficiencyClass;
                cd.has_smt = (e->Processor.Flags & LTP_PC_SMT) != 0;
                for (WORD group = 0; group < e->Processor.GroupCount; ++group) {
                    const GROUP_AFFINITY& g = e->Processor.GroupMask[group];
                    for (int bit = 0; bit < 64; ++bit) {
                        if (g.Mask & (KAFFINITY(1) << bit)) {
                            cd.lps.push_back((int) (g.Group * 64 + bit));
                        }
                    }
                }
                if (!cd.lps.empty()) {
                    descs.push_back(std::move(cd));
                }
            }
            p += e->Size;
        }

        uint8_t min_eff = 255, max_eff = 0;
        for (const auto& c : descs) {
            min_eff = (std::min)(min_eff, c.efficiency);
            max_eff = (std::max)(max_eff, c.efficiency);
        }

        topo.is_hybrid = (max_eff > min_eff);
        if (topo.is_hybrid) {
            for (const auto& c : descs) {
                if (c.efficiency == max_eff) {
                    topo.p_cores++;
                    topo.p_threads += (int) c.lps.size();
                } else {
                    topo.e_cores++;
                }
            }
        } else {
            topo.p_cores = (int) descs.size();
            for (const auto& c : descs) topo.p_threads += (int) c.lps.size();
        }

        if (affinity == PoolAffinity::All || !topo.is_hybrid) {
            // #642 (from Hardin22's fork): on a hybrid CPU the P-cores first (the host takes the first of them), so a
            // pool smaller than the core count (setup's --pool-workers for a hybrid CPU) runs on the P-cores and the
            // first E-cores rather than on whatever the OS numbered first.  All cores alike: the order is unchanged.
            if (topo.is_hybrid)
                std::stable_sort(descs.begin(), descs.end(),
                                 [](const CoreDesc& x, const CoreDesc& y) { return x.efficiency > y.efficiency; });
            for (const auto& c : descs) topo.worker_cores.push_back(c.lps[0]);
            if (skip_first && !topo.worker_cores.empty()) {
                topo.host_core = topo.worker_cores.front();
                topo.worker_cores.erase(topo.worker_cores.begin());
            }
            return topo;
        }

        // Hybrid CPU with Auto or PCores affinity:
        // Prioritize Performance cores:
        // 1. Primary logical processor of each P-core (avoids SMT resource contention)
        // 2. SMT sibling logical processors of P-cores
        // 3. E-cores (only as overflow in Auto mode)
        std::vector<int> p_primaries;
        std::vector<int> p_siblings;
        std::vector<int> e_cores;

        for (const auto& c : descs) {
            if (c.efficiency == max_eff) {
                p_primaries.push_back(c.lps[0]);
                for (size_t s = 1; s < c.lps.size(); ++s) {
                    p_siblings.push_back(c.lps[s]);
                }
            } else {
                for (int lp : c.lps) e_cores.push_back(lp);
            }
        }

        if (skip_first && !p_primaries.empty()) {
            topo.host_core = p_primaries.front();
            p_primaries.erase(p_primaries.begin());
        }

        for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
        for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
        if (affinity != PoolAffinity::PCores) {
            for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
        }
        return topo;
    }
#else
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE (issue #40): SMT siblings share a core's
    // load/store bandwidth, so a worker on each would put two workers on one core, as the Windows branch above
    // explains.  sysfs names each CPU's (package, core); the first allowed CPU of each pair is kept, so a taskset
    // that leaves out the first sibling still gets its core.  Without sysfs every allowed CPU counts, as before.
    auto topo_read = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    auto cap_read = [](int cpu) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };

    std::vector<int> allowed;
    std::vector<unsigned long> allowed_mask;
    if (detail::get_thread_affinity(allowed_mask, &allowed) != 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }

    struct CoreLinux {
        int cpu = -1;
        long pkg = -1;
        long core = -1;
        long cap = -1;
        bool is_sibling = false;
    };
    std::vector<CoreLinux> all_cpus;
    std::vector<std::pair<long, long>> seen_phys;
    long max_cap = 0, min_cap = 1000000;

    for (int cpu : allowed) {
        CoreLinux cl;
        cl.cpu = cpu;
        cl.pkg = topo_read(cpu, "physical_package_id");
        cl.core = topo_read(cpu, "core_id");
        cl.cap = cap_read(cpu);
        if (cl.cap > 0) {
            max_cap = (std::max)(max_cap, cl.cap);
            min_cap = (std::min)(min_cap, cl.cap);
        }
        if (cl.pkg >= 0 && cl.core >= 0) {
            const std::pair<long, long> key{cl.pkg, cl.core};
            if (std::find(seen_phys.begin(), seen_phys.end(), key) != seen_phys.end()) {
                cl.is_sibling = true;
            } else {
                seen_phys.push_back(key);
            }
        }
        all_cpus.push_back(cl);
    }

    topo.is_hybrid = (max_cap > 0 && max_cap > min_cap);
    if (topo.is_hybrid) {
        for (const auto& cl : all_cpus) {
            if (cl.cap == max_cap) {
                if (!cl.is_sibling) topo.p_cores++;
                topo.p_threads++;
            } else {
                if (!cl.is_sibling) topo.e_cores++;
            }
        }
    } else {
        topo.p_cores = (int) seen_phys.size();
        topo.p_threads = (int) all_cpus.size();
    }

    if (affinity == PoolAffinity::All || !topo.is_hybrid) {
        if (topo.is_hybrid)   // #642: the P-cores first (see the Windows branch)
            std::stable_sort(all_cpus.begin(), all_cpus.end(),
                             [](const CoreLinux& x, const CoreLinux& y) { return x.cap > y.cap; });
        for (const auto& cl : all_cpus) {
            if (!cl.is_sibling) topo.worker_cores.push_back(cl.cpu);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }

    // Hybrid CPU on Linux:
    std::vector<int> p_primaries;
    std::vector<int> p_siblings;
    std::vector<int> e_cores;

    for (const auto& cl : all_cpus) {
        if (cl.cap == max_cap) {
            if (!cl.is_sibling) p_primaries.push_back(cl.cpu);
            else p_siblings.push_back(cl.cpu);
        } else {
            e_cores.push_back(cl.cpu);
        }
    }

    if (skip_first && !p_primaries.empty()) {
        topo.host_core = p_primaries.front();
        p_primaries.erase(p_primaries.begin());
    }

    for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
    for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
    if (affinity != PoolAffinity::PCores) {
        for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
    }
    return topo;
#endif
    return topo;
}

std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity) {
    return detect_cpu_topology(skip_first, affinity).worker_cores;
}

namespace {

bool pin_this_thread(int core, [[maybe_unused]] int worker = -1) {
    if (core < 0) return false;
#if defined(_WIN32)
    return detail::set_thread_group_affinity(core, worker);
#else
    const int error = detail::pin_thread_to_cpu(core);
    if (error != 0)
        std::fprintf(stderr, "guild cpu pool: affinity for worker %d (CPU %d) failed: %d; previous affinity kept\n",
                     worker, core, error);
    return error == 0;
#endif
}

}  // namespace

ThreadAffinity pin_current_thread(int core) {
    if (core < 0) return {};
#if defined(_WIN32)
    ThreadAffinity previous;
    ULONG target = 0;
    if (!detail::get_thread_cpu_sets(previous.cpu_sets) || !detail::cpu_set_for_core(core, target) ||
        !SetThreadSelectedCpuSets(GetCurrentThread(), &target, 1)) {
        std::fprintf(stderr, "guild cpu pool: host CPU Set selection for processor %d failed: %lu; previous placement kept\n",
                     core, (unsigned long) GetLastError());
        return {};
    }
    previous.valid = true;
    return previous;
#else
    ThreadAffinity previous;
    int error = detail::get_thread_affinity(previous.mask);
    if (error == 0) error = detail::pin_thread_to_cpu(core);
    if (error != 0) {
        std::fprintf(stderr, "guild cpu pool: host affinity for CPU %d failed: %d; previous affinity kept\n", core, error);
        return {};
    }
    previous.valid = true;
    return previous;
#endif
}

ThreadAffinity pin_current_thread_if_allowed(int core) {
    if (core < 0) return {};
#if defined(_WIN32)
    GROUP_AFFINITY group{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &group) ||
        unsigned(core / 64) != group.Group || !(group.Mask & (KAFFINITY(1) << (core % 64)))) return {};
    std::vector<unsigned long> selected;
    ULONG target=0;
    if (!detail::get_thread_cpu_sets(selected) || !detail::cpu_set_for_core(core,target)) return {};
    if (!selected.empty() && std::find(selected.begin(),selected.end(),target)==selected.end()) return {};
#else
    std::vector<unsigned long> mask;
    std::vector<int> allowed;
    if (detail::get_thread_affinity(mask,&allowed)!=0 ||
        std::find(allowed.begin(),allowed.end(),core)==allowed.end()) return {};
#endif
    return pin_current_thread(core);
}

void restore_thread_affinity(const ThreadAffinity& previous) {
    if (!previous.valid) return;
#if defined(_WIN32)
    // Clearing an originally empty selection restores process-default/all-group eligibility without
    // turning the caller's implicit Windows 11 affinity into an explicit single-group hard mask.
    if (!SetThreadSelectedCpuSets(GetCurrentThread(), previous.cpu_sets.empty() ? nullptr : previous.cpu_sets.data(),
                                 (ULONG) previous.cpu_sets.size()))
        std::fprintf(stderr, "guild cpu pool: host CPU Set restoration failed: %lu\n",
                     (unsigned long) GetLastError());
#else
    const int error = detail::set_thread_affinity(previous.mask);
    if (error != 0)
        std::fprintf(stderr, "guild cpu pool: host affinity restoration failed: %d\n", error);
#endif
}

namespace {
std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_.load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_);
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works, PoolAffinity affinity)
    : host_works_(host_works), affinity_(affinity), topo_(detect_cpu_topology(true, affinity)) {
    if (const char* e = std::getenv("GUILD_POOL_SPIN_US"))   // a test knob; see kSpinBeforeSleep
        spin_before_sleep_ = std::chrono::microseconds((std::max)(0, std::atoi(e)));
    if (n_workers > 0) {
        n_ = n_workers;
    } else if (topo_.is_hybrid && affinity_ != PoolAffinity::All) {
        n_ = (std::max)(1, topo_.p_cores - 1);
    } else {
        n_ = (int) topo_.worker_cores.size();
    }
    if (n_ < 1) n_ = 1;
    if (const char* path = std::getenv("GUILD_CPU_PROFILE")) {
        profile_file_ = std::fopen(path, "w");
        if (!profile_file_) throw std::runtime_error("cannot open GUILD_CPU_PROFILE");
        profile_times_.resize(size_t(n_) + 1);
        std::fputs("start_ns,end_ns,gu_type,down_type,jobs,entries,whole,gu_thread_cpu_ms,quant_thread_cpu_ms,down_thread_cpu_ms\n", profile_file_);
    }
    scratch_.resize((size_t) n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    guild::core::diag_pool_fn().store(&diag_active_pool);
    split_.resize((size_t) kMaxSplit);
    split_multi_.resize((size_t) kMaxSplitMulti);
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) topo_.worker_cores.size() ? topo_.worker_cores[(size_t) i] : -1) : -1;
        threads_.emplace_back([this, i, core] {
            pin_this_thread(core, i);
            worker(i);
        });
    }
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
    for (auto& t : threads_) t.join();
    if (profile_file_) std::fclose(profile_file_);
}

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction.  Counting only
    // on the RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first
    // `run()` - which waits for `parked_ == n_` before publishing - then deadlocks.  It deadlocks on the very
    // first call, which is the good case; a version that deadlocked on the second would be far worse.
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        // Park: wait for work.  `_mm_pause` rather than a bare spin because it yields the pipeline to the
        // sibling hyperthread; `epoch_` is bumped once per LAYER, not once per expert, so most of these
        // iterations are spent here with nothing to do.
        //
        // **AND NOTHING ELSE HAPPENS IN HERE.**  This loop used to do `pauses_.fetch_add(1)` on every iteration
        // - a locked read-modify-write, five workers against one cache line - so the workers spent their wait
        // invalidating each other's caches and the very line the host writes to publish work.  The counter was
        // diagnostic and nothing branched on it.  See the note on the atomics in pool.hpp.
        //
        // After `kSpinBeforeSleep` with no work the worker sleeps instead (issue #4).  The clock is read once
        // every 1024 pauses, so the spin itself is unchanged.
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            if (std::chrono::steady_clock::now() - parked_at < spin_before_sleep_) continue;
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.  Only
        // this epoch's jobs: if the host has already moved on, the claims fail and the worker parks again.
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, scratch_[(size_t) id], seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        std::fprintf(stderr, "guild: expert pool batch of %d jobs is out of range\n", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    publish();
    return e;
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0;
    std::chrono::steady_clock::time_point t0{};
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u) t0 = now;
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "guild: the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again (issue #29)\n",
                         what, parked_.load(), n_);
            guild::core::release_gpu_waits(stderr);   // #267: the GPU may be spinning on this layer's flag
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0, seen = 0;
    std::chrono::steady_clock::time_point t0{};
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;                                // `>=`: never a wait that an overshoot outlives
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u || d != seen) { t0 = now; seen = d; }     // progress restarts the clock
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "guild: the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again (issue #29)\n",
                         d, n, parked_.load(), n_);
            guild::core::release_gpu_waits(stderr);   // #267
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch, uint32_t epoch) {
    (void) id;
    for (;;) {
        const int ci = claim(epoch);
        if (ci < 0) break;
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        const double profile_start = profile_file_ ? thread_cpu_ms() : 0;
        if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(jobs_[e].blob, *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(jobs_[e].blob, split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ == 9) {
            native_quant_act(*nfmt_, qact_x_ + (size_t) i * (size_t) nfmt_->n_embd,
                             qact_dst_ + (size_t) i * kNativeActBytes);
        } else if (mode_ == 10) {
            const int K = comb_k_;
            const int N = comb_n_embd_;
            const float* p_base = comb_parts_ + (size_t) i * (size_t) K * (size_t) N;
            const float* w_base = comb_weights_ + (size_t) i * (size_t) K;
            float* dst = comb_out_ + (size_t) i * (size_t) N;
            uint32_t nonfinite = 0;
            if(backend_table_.combine) {
                if(backend_table_.combine(p_base,w_base,dst,K,N)!=GUILD_STATUS_OK)
                    comb_nonfinite_.store(true,std::memory_order_relaxed);
            } else {
            for (int d = 0; d < N; ++d) {
                float acc = 0.0f;
                for (int r = 0; r < K; ++r) {
                    const float pv = p_base[(size_t) r * (size_t) N + d];
                    uint32_t bits;
                    std::memcpy(&bits, &pv, sizeof(bits));
                    nonfinite |= (bits & 0x7f800000u) == 0x7f800000u;
                    acc = std::fma(pv, w_base[r], acc);
                }
                dst[d] = acc;
            }
            if (nonfinite) comb_nonfinite_.store(true, std::memory_order_relaxed);
            }
        } else if (mode_ == 8) {
            // Whole multi-token native expert per task: gate/up -> quant_h -> down in thread-local L1/L2 cache.
            const ExpertJobMulti& mj = mjobs_[i];
            const int nt = mj.nt;
            const void* const* nact = mj.nact_ext ? mj.nact_ext : mj.nact;
            float* const* out = mj.out_ext ? mj.out_ext : mj.out;
            alignas(64) thread_local float ff_buf[kMaxMultiExt][FF];
            alignas(64) thread_local ActQ a2_buf[kMaxMultiExt];
            alignas(64) thread_local uint8_t hq_buf[kMaxMultiExt][kNativeHBytes];
            const bool time_phases = (id < 0);
            double cpu_a = 0, cpu_b = 0;
            std::chrono::steady_clock::time_point t_a, t_b, t_c, t_d;
            if (time_phases) t_a = std::chrono::steady_clock::now();
            if (q2_native_kernels(nfmt_->gu_type)) {
                thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                const int nbk = (int) (nfmt_->n_embd / 64);
                for (int t0 = 0; t0 < nt; t0 += MAXT) {
                    const int m = (std::min)(MAXT, nt - t0);
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < m; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    q2_rows_any(mj.blob, nfmt_->gu_row, nbk, mj.act + t0, m, gp, 0, (int) nfmt_->n_ff);
                    q2_rows_any(mj.blob + nfmt_->up_off, nfmt_->gu_row, nbk, mj.act + t0, m, up, 0, (int) nfmt_->n_ff);
                    for (int t = 0; t < m; ++t)
                        for (int r = 0; r < (int) nfmt_->n_ff; ++r)
                            ff_buf[t0 + t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                }
            } else {
                float* ff[kMaxMultiExt];
                for (int t = 0; t < nt; ++t) ff[t] = ff_buf[t];
                native_gu_rows(*nfmt_, mj.blob, nact, nt, ff, 0, (int) nfmt_->n_ff);
            }
            if (time_phases) t_b = std::chrono::steady_clock::now();
            if (profile_file_) cpu_a = thread_cpu_ms();
            if (q2_native_kernels(nfmt_->d_type)) {
                const ActQ* a2[kMaxMultiExt];
                for (int t = 0; t < nt; ++t) {
                    act_quant_any(ff_buf[t], (int) nfmt_->n_ff, a2_buf[t]);
                    a2[t] = &a2_buf[t];
                }
                if (time_phases) t_c = std::chrono::steady_clock::now();
                if (profile_file_) cpu_b = thread_cpu_ms();
                for (int t0 = 0; t0 < nt; t0 += MAXT) {
                    const int m = (std::min)(MAXT, nt - t0);
                    q2_rows_any(mj.blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2 + t0,
                                m, out + t0, 0, (int) nfmt_->n_embd);
                }
            } else {
                const void* hq[kMaxMultiExt];
                for (int t = 0; t < nt; ++t) {
                    native_quant_h(*nfmt_, ff_buf[t], hq_buf[t]);
                    hq[t] = hq_buf[t];
                }
                if (time_phases) t_c = std::chrono::steady_clock::now();
                if (profile_file_) cpu_b = thread_cpu_ms();
                native_down_rows(*nfmt_, mj.blob, hq, nt, out, 0, (int) nfmt_->n_embd);
            }
            if (time_phases) {
                t_d = std::chrono::steady_clock::now();
                host_gu_ms_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
                host_q_ms_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
                host_down_ms_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
            }
            if (profile_file_) {
                auto& p = profile_times_[size_t(id + 1)];
                p.gu += cpu_a - profile_start;
                p.quant += cpu_b - cpu_a;
                p.down += thread_cpu_ms() - cpu_b;
            }
        } else if (mode_ == 7) {
            // One independent expert intermediate per task, no cross-task writes.
            auto& sb=split_multi_[i];
            for (int t=0;t<mjobs_[i].nt;++t)
                if (q2_native_kernels(nfmt_->d_type)) act_quant_any(sb.ff[t],int(nfmt_->n_ff),sb.a2[t]);
                else native_quant_h(*nfmt_,sb.ff[t],sb.hq[t]);
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? (int) nfmt_->n_ff : (int) nfmt_->n_embd;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 5 && q2_native_kernels(nfmt_->gu_type)) {
                    // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt_->n_embd / 64);
                    q2_rows_any(mjobs_[e].blob, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mjobs_[e].blob + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r = r0; r < r1; ++r)
                            sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                } else if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (q2_native_kernels(nfmt_->d_type)) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mjobs_[e].blob, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mjobs_[e].blob, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        if (profile_file_ && mode_ != 8) {
            auto& p = profile_times_[size_t(id + 1)];
            const double value = thread_cpu_ms() - profile_start;
            if (mode_ == 5) p.gu += value;
            else if (mode_ == 6) p.down += value;
            else if (mode_ == 7 || mode_ == 9) p.quant += value;
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    wait_parked("before a phase");
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    if (host_works_) drain(-1, host_scratch_, e);
    wait_done(n_tasks);
    wait_parked("after a phase");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = 3 * threads;
    mrows_ = (int64_t) n * FF;
    run_phase(3, mtasks_);
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    mrows_ = (int64_t) n * H;
    run_phase(4, mtasks_);
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::quant_act_multi_native(const NativeFmt& f, const float* x, uint8_t* dst, int n_tok) {
    if (n_tok <= 0) return;
    if (n_tok < 8 || n_ <= 1) {
        for (int t = 0; t < n_tok; ++t)
            native_quant_act(f, x + (size_t) t * (size_t) f.n_embd, dst + (size_t) t * kNativeActBytes);
        return;
    }
    nfmt_ = &f;
    qact_x_ = x;
    qact_dst_ = dst;
    run_phase(9, n_tok);
    mode_ = 0;
}

bool ExpertPool::combine_multi_native(const float* parts, const float* weights, float* out,
                                      int n_tok, int k, int n_embd) {
    if (n_tok <= 0 || k <= 0 || n_embd <= 0) return false;
    comb_parts_ = parts;
    comb_weights_ = weights;
    comb_out_ = out;
    comb_k_ = k;
    comb_n_embd_ = n_embd;
    comb_nonfinite_.store(false, std::memory_order_relaxed);
    run_phase(10, n_tok);
    mode_ = 0;
    return !comb_nonfinite_.load(std::memory_order_relaxed);
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n, bool parallel_quant) {
    if (n <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    if (profile_file_) std::fill(profile_times_.begin(), profile_times_.end(), ProfileTimes{});
    const int threads = n_ + (host_works_ ? 1 : 0);
    static const bool whole_expert = [] {
        const char* v = std::getenv("GUILD_POOL_WHOLE_EXPERT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (parallel_quant && whole_expert && n >= threads) {
        mjobs_ = jobs;
        nfmt_ = &f;
        host_gu_ms_ = 0.0;
        host_q_ms_ = 0.0;
        host_down_ms_ = 0.0;
        const auto a = std::chrono::steady_clock::now();
        run_phase(8, n);
        const auto d = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double, std::milli>(d - a).count();
        const double hsum = host_gu_ms_ + host_q_ms_ + host_down_ms_;
        if (hsum > 0.0) {
            ms_multi_gu += wall * (host_gu_ms_ / hsum);
            ms_multi_q += wall * (host_q_ms_ / hsum);
            ms_multi_down += wall * (host_down_ms_ / hsum);
        } else {
            ms_multi_gu += wall * 0.65;
            ms_multi_down += wall * 0.35;
        }
        multi_bytes += (int64_t) n * (int64_t) f.bytes;
        mode_ = 0;
        ms_drain_ += std::chrono::duration<double, std::milli>(d - t0).count();
        if (profile_file_) report_profile(t0, f, jobs, n, true);
        return;
    }
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        const int threads = n_ + (host_works_ ? 1 : 0);
        mtasks_ = 3 * threads;
        mrows_ = (int64_t) nb * f.n_ff;
        const auto a = std::chrono::steady_clock::now();
        run_phase(5, mtasks_);
        const auto b = std::chrono::steady_clock::now();
        if (parallel_quant && nb >= 32) run_phase(7,nb);
        else {
            const double cpu_start = profile_file_ ? thread_cpu_ms() : 0;
            for (int e = 0; e < nb; ++e)
                for (int t = 0; t < mjobs_[e].nt; ++t)
                    if (q2_native_kernels(f.d_type)) act_quant_any(split_multi_[(size_t) e].ff[t], (int) f.n_ff, split_multi_[(size_t) e].a2[t]);
                    else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
            if (profile_file_) profile_times_[0].quant += thread_cpu_ms() - cpu_start;
        }
        const auto c = std::chrono::steady_clock::now();
        mrows_ = (int64_t) nb * f.n_embd;
        run_phase(6, mtasks_);
        const auto d = std::chrono::steady_clock::now();
        ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
        ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
        ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (profile_file_) report_profile(t0, f, jobs, n, false);
}

void ExpertPool::report_profile(std::chrono::steady_clock::time_point start, const NativeFmt& f,
                                ExpertJobMulti* jobs, int n, bool whole) {
    ProfileTimes sum;
    for (const auto& p : profile_times_) { sum.gu += p.gu; sum.quant += p.quant; sum.down += p.down; }
    int entries = 0;
    for (int i = 0; i < n; ++i) entries += jobs[i].nt;
    const auto end = std::chrono::steady_clock::now();
    std::fprintf(profile_file_, "%lld,%lld,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f\n",
        (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count(),
        (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count(),
        f.gu_type, f.d_type, n, entries, int(whole), sum.gu, sum.quant, sum.down);
    std::fflush(profile_file_);
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(jobs[i].blob, *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // Wait for every worker to be parked BEFORE touching the batch, so the publish below is the only thing
    // that can move a worker into the drain loop.
    //
    // THE THREE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is
    // slow at the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_a = std::chrono::steady_clock::now();
    wait_parked("before a batch");
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    const uint32_t e = begin_batch(n);   // done, then head (release), then the epoch: the batch is described first

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) _mm_pause();`.  The host is pinned to core 0 - the core
    // `physical_cores(true)` deliberately keeps the five workers off - so for the whole drain that core was
    // idle while five cores did six cores' worth of work.  Measured before the change: 33.7 GB/s against
    // 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const int ci = claim(e);
            if (ci < 0) break;
            hstate_.store(ci, std::memory_order_relaxed);
            hstate_ms_.store(now_ms(), std::memory_order_relaxed);
            const ExpertJob& j = jobs_[ci];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    wait_done(n);
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    wait_parked("after a batch");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace guild::kernels::cpu
