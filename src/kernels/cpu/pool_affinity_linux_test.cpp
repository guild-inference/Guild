#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "guild/kernels/cpu/pool.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include "pool_affinity_linux.hpp"

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

namespace cpu = guild::kernels::cpu;

class DynamicCpuSet {
public:
    explicit DynamicCpuSet(size_t count) : count_(count), bytes_(CPU_ALLOC_SIZE(count)), set_(CPU_ALLOC(count)) {
        if (set_)
            CPU_ZERO_S(bytes_, set_);
    }

    ~DynamicCpuSet() {
        if (set_)
            CPU_FREE(set_);
    }

    DynamicCpuSet(const DynamicCpuSet&) = delete;
    DynamicCpuSet& operator=(const DynamicCpuSet&) = delete;

    cpu_set_t* get() {
        return set_;
    }
    const cpu_set_t* get() const {
        return set_;
    }
    size_t bytes() const {
        return bytes_;
    }
    size_t count() const {
        return count_;
    }
    explicit operator bool() const {
        return set_ != nullptr;
    }

private:
    size_t count_;
    size_t bytes_;
    cpu_set_t* set_;
};

static size_t initial_cpu_count() {
    const long configured = sysconf(_SC_NPROCESSORS_CONF);
    return (std::max)(size_t(128), configured > 0 ? (size_t) configured : size_t(0));
}

// Query with CPU_ALLOC rather than cpu_set_t: this must observe CPUs beyond CPU_SETSIZE too.
static bool get_affinity(std::vector<int>& ids) {
    size_t count = initial_cpu_count();
    for (int attempt = 0; attempt < 8; ++attempt) {
        DynamicCpuSet set(count);
        if (!set)
            return false;
        const int error = pthread_getaffinity_np(pthread_self(), set.bytes(), set.get());
        if (error == 0) {
            ids.clear();
            for (size_t i = 0; i < set.count(); ++i)
                if (CPU_ISSET_S(i, set.bytes(), set.get()))
                    ids.push_back((int) i);
            return !ids.empty();
        }
        if (error != EINVAL || count > (size_t(1) << 20)) {
            std::fprintf(stderr, "pthread_getaffinity_np failed: %s\n", std::strerror(error));
            return false;
        }
        count *= 2;
    }
    std::fprintf(stderr, "pthread_getaffinity_np still needs a larger CPU set\n");
    return false;
}

static bool set_affinity(const std::vector<int>& ids) {
    if (ids.empty())
        return false;
    const int highest = *std::max_element(ids.begin(), ids.end());
    if (highest < 0)
        return false;
    const size_t count = (std::max)(initial_cpu_count(), (size_t) highest + 1);
    DynamicCpuSet set(count);
    if (!set)
        return false;
    for (int id : ids) {
        if (id < 0 || (size_t) id >= count)
            return false;
        CPU_SET_S((size_t) id, set.bytes(), set.get());
    }
    const int error = pthread_setaffinity_np(pthread_self(), set.bytes(), set.get());
    if (error != 0) {
        std::fprintf(stderr, "pthread_setaffinity_np failed: %s\n", std::strerror(error));
        return false;
    }
    return true;
}

class OriginalAffinity {
public:
    bool capture() {
        return get_affinity(ids_);
    }
    const std::vector<int>& ids() const {
        return ids_;
    }
    bool restore() {
        return ids_.empty() || set_affinity(ids_);
    }
    ~OriginalAffinity() {
        if (!ids_.empty())
            set_affinity(ids_);
    }

private:
    std::vector<int> ids_;
};

static bool expect_affinity(const std::vector<int>& expected, const char* label) {
    std::vector<int> actual;
    if (!get_affinity(actual)) {
        std::fprintf(stderr, "Linux affinity test: %s: cannot query current affinity\n", label);
        return false;
    }
    if (actual != expected) {
        std::fprintf(stderr, "Linux affinity test: %s: expected %zu CPUs, observed %zu\n", label, expected.size(),
                     actual.size());
        return false;
    }
    return true;
}

static bool round_trip_full_mask(const std::vector<int>& allowed) {
    const int target = allowed.front();
    const auto previous = cpu::pin_current_thread(target);
    const bool pinned = previous.valid && expect_affinity({target}, "pin from full allowed mask");
    cpu::restore_thread_affinity(previous);
    return pinned && expect_affinity(allowed, "restore full allowed mask");
}

static bool topology_respects_current_mask() {
    std::vector<int> allowed;
    if (!get_affinity(allowed))
        return false;
    const auto topology = cpu::detect_cpu_topology(false);
    if (topology.worker_cores.empty())
        return false;
    for (int core : topology.worker_cores) {
        if (!std::binary_search(allowed.begin(), allowed.end(), core)) {
            std::fprintf(stderr, "Linux affinity test: topology selected disallowed CPU %d\n", core);
            return false;
        }
    }
    return true;
}

static bool round_trip_sparse_nested(const std::vector<int>& allowed) {
    if (allowed.size() < 2) {
        std::puts("SKIP sparse/nested affinity: fewer than two allowed CPUs");
        return true;
    }
    const std::vector<int> sparse{allowed.front(), allowed.back()};
    if (!set_affinity(sparse) || !expect_affinity(sparse, "establish sparse mask") || !topology_respects_current_mask())
        return false;

    const auto outer = cpu::pin_current_thread(sparse.front());
    const bool outer_pinned = outer.valid && expect_affinity({sparse.front()}, "outer pin");
    if (!outer_pinned) {
        cpu::restore_thread_affinity(outer);
        return false;
    }

    const auto inner = cpu::pin_current_thread(sparse.back());
    const bool inner_pinned = inner.valid && expect_affinity({sparse.back()}, "inner pin");
    cpu::restore_thread_affinity(inner);
    const bool outer_restored = expect_affinity({sparse.front()}, "restore inner pin to outer pin");
    cpu::restore_thread_affinity(outer);
    const bool sparse_restored = expect_affinity(sparse, "restore outer pin to sparse mask");
    const bool full_restored = set_affinity(allowed) && expect_affinity(allowed, "restore full mask after sparse test");
    return inner_pinned && outer_restored && sparse_restored && full_restored;
}

static bool round_trip_high_cpu(const std::vector<int>& allowed, bool require_high) {
    const auto it = std::find_if(allowed.begin(), allowed.end(), [](int id) { return id >= 64; });
    if (it == allowed.end()) {
        if (require_high) {
            std::fprintf(stderr, "Linux affinity test: --require-high-cpu requested, but no allowed CPU ID is >= 64\n");
            return false;
        }
        std::puts("SKIP high-CPU affinity path: no allowed CPU ID is >= 64 (use --require-high-cpu to require it)");
        return true;
    }
    const int target = allowed.front() == *it ? (allowed.size() > 1 ? allowed.back() : *it) : allowed.front();
    if (target == *it) {
        if (require_high) {
            std::fprintf(
                stderr,
                "Linux affinity test: --require-high-cpu needs at least two allowed CPUs to test restoration\n");
            return false;
        }
        std::puts(
            "SKIP high-CPU affinity round trip: only one CPU is allowed, so no meaningful pin transition is possible");
        return true;
    }

    // Restrict the caller to one high-numbered CPU, then pin to a different originally allowed CPU.
    // Restoring the high-only mask catches truncation even when the original process mask is sparse.
    const std::vector<int> high_only{*it};
    if (!set_affinity(high_only) || !expect_affinity(high_only, "establish high-only mask") ||
        !topology_respects_current_mask())
        return false;
    const auto previous = cpu::pin_current_thread(target);
    const bool pinned = previous.valid && expect_affinity({target}, "pin away from high-only mask");
    cpu::restore_thread_affinity(previous);
    const bool restored = pinned && expect_affinity(high_only, "restore high-only mask");
    if (restored)
        std::printf("High-CPU affinity round trip: saved CPU %d, pin target %d\n", *it, target);
    return restored;
}

static bool negative_cpu_is_noop() {
    std::vector<int> before;
    if (!get_affinity(before))
        return false;
    const auto invalid = cpu::pin_current_thread(-1);
    std::vector<int> after;
    const bool unchanged = get_affinity(after) && before == after;
    cpu::restore_thread_affinity(invalid);
    return !invalid.valid && unchanged && expect_affinity(before, "negative CPU no-op");
}

static bool worker_thread_can_pin(int target) {
    bool ok = false;
    std::thread worker([&] {
        std::vector<int> before;
        if (!get_affinity(before))
            return;
        std::vector<unsigned long> saved;
        if (cpu::detail::get_thread_affinity(saved) != 0)
            return;
        const int pin_error = cpu::detail::pin_thread_to_cpu(target);
        const bool pinned =
            pin_error == 0 && expect_affinity({target}, "worker thread helper pin") && sched_getcpu() == target;
        const int restore_error = cpu::detail::set_thread_affinity(saved);
        ok = pinned && restore_error == 0 && expect_affinity(before, "worker thread helper restore");
    });
    worker.join();
    return ok;
}

static bool failed_pin_preserves_affinity() {
    std::vector<int> before;
    if (!get_affinity(before))
        return false;
    // An ID outside the caller's current mask may still be online and legal, so it is not a
    // reliable failure case. INT_MAX is rejected before allocating or changing any CPU set.
    const int error = cpu::detail::pin_thread_to_cpu(INT_MAX);
    std::vector<int> after;
    const bool helper_unchanged = get_affinity(after) && before == after;
    if (error == 0) {
        std::fprintf(stderr, "Linux affinity test: pin_thread_to_cpu(INT_MAX) unexpectedly succeeded\n");
        return false;
    }
    const auto invalid = cpu::pin_current_thread(INT_MAX);
    cpu::restore_thread_affinity(invalid);
    std::vector<int> after_public;
    const bool public_unchanged = get_affinity(after_public) && before == after_public;
    if (!helper_unchanged || invalid.valid || !public_unchanged)
        std::fprintf(stderr,
                     "Linux affinity test: failed pin changed the observed affinity or returned a valid snapshot\n");
    return helper_unchanged && !invalid.valid && public_unchanged;
}

static bool request_pin_respects_restricted_mask(const std::vector<int>& allowed) {
    if (allowed.size() < 2)
        return true;
    const int reserved = allowed.front(), request_cpu = allowed.back();
    if (!set_affinity({request_cpu}))
        return false;
    // Model load may reserve a CPU that a later request thread cannot use.
    const auto skipped = cpu::pin_current_thread_if_allowed(reserved);
    const bool kept = !skipped.valid && expect_affinity({request_cpu}, "request pin must not widen caller mask");
    cpu::restore_thread_affinity(skipped);
    if (!set_affinity(allowed))
        return false;
    const auto pinned = cpu::pin_current_thread_if_allowed(reserved);
    const bool selected = pinned.valid && expect_affinity({reserved}, "eligible request pin");
    cpu::restore_thread_affinity(pinned);
    return kept && selected && expect_affinity(allowed, "request pin restores caller mask");
}

static long sysfs_topology(int cpu, const char* what) {
    char path[96];
    std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
    long v = -1;
    if (std::FILE* f = std::fopen(path, "r")) {
        if (std::fscanf(f, "%ld", &v) != 1)
            v = -1;
        std::fclose(f);
    }
    return v;
}

// ExpertPool::helper_cpus' source: detect_cpu_topology(true)'s host_siblings are exactly the other allowed CPUs with
// the host core's package and core ids, none of them a worker core; under a mask of one CPU a core there are none;
// and pin_current_thread_to puts a new thread on exactly that list (an empty or negative list changes nothing).
static bool host_siblings_and_helper_pin(const std::vector<int>& allowed) {
    const auto topo = cpu::detect_cpu_topology(true);
    if (topo.host_core < 0) {
        std::puts("SKIP host siblings: no host core");
        return true;
    }
    const long pkg = sysfs_topology(topo.host_core, "physical_package_id");
    const long core = sysfs_topology(topo.host_core, "core_id");
    std::vector<int> expected;
    if (pkg >= 0 && core >= 0)
        for (int c : allowed)
            if (c != topo.host_core && sysfs_topology(c, "physical_package_id") == pkg &&
                sysfs_topology(c, "core_id") == core)
                expected.push_back(c);
    if (topo.host_siblings != expected) {
        std::fprintf(stderr, "Linux affinity test: host core %d: %zu siblings listed, sysfs has %zu\n", topo.host_core,
                     topo.host_siblings.size(), expected.size());
        return false;
    }
    for (int c : topo.host_siblings)
        if (std::find(topo.worker_cores.begin(), topo.worker_cores.end(), c) != topo.worker_cores.end()) {
            std::fprintf(stderr, "Linux affinity test: host sibling %d is a worker core\n", c);
            return false;
        }
    std::vector<int> firsts = topo.worker_cores; // one CPU a core: the topology's own choice, and the host
    firsts.push_back(topo.host_core);
    std::sort(firsts.begin(), firsts.end());
    if (!set_affinity(firsts))
        return false;
    const bool none = cpu::detect_cpu_topology(true).host_siblings.empty();
    if (!set_affinity(allowed) || !none) {
        if (!none)
            std::fprintf(stderr, "Linux affinity test: host siblings listed under a one-CPU-a-core mask\n");
        return false;
    }
    const std::vector<int> target = topo.host_siblings.empty() ? std::vector<int>{allowed.back()} : topo.host_siblings;
    bool pinned = false, refused = false;
    std::thread([&] {
        std::vector<int> before;
        refused = get_affinity(before) && !cpu::pin_current_thread_to({}) && !cpu::pin_current_thread_to({-1}) &&
                  expect_affinity(before, "refused helper pin");
        pinned = cpu::pin_current_thread_to(target) && expect_affinity(target, "helper pin");
    }).join();
    if (topo.host_siblings.empty())
        std::puts("SKIP host siblings: the host core has no SMT sibling here (the helper pin checked on one CPU)");
    return pinned && refused && expect_affinity(allowed, "a helper's pin leaves its creator's affinity");
}

// ExpertPool::helper_cpus' policy (it also decides Drama 3's request pin): the host siblings with the workers pinned
// one to each other core (the default pool), none with the workers unpinned or with fewer or more workers than those
// cores (GUILD_POOL_WORKERS).
static bool helper_cpus_policy(const std::vector<int>& allowed) {
    const auto topo = cpu::detect_cpu_topology(true);
    const int cores = (int) topo.worker_cores.size();
    auto check = [&](int workers, bool pin, const std::vector<int>& expected, const char* label) {
        const cpu::ExpertPool pool(workers, pin, true);
        if (pool.helper_cpus() == expected)
            return true;
        std::fprintf(stderr, "Linux affinity test: %s (%d workers, pin %d): %zu helper CPUs, expected %zu\n", label,
                     pool.workers(), (int) pin, pool.helper_cpus().size(), expected.size());
        return false;
    };
    bool ok = check(0, true, topo.host_siblings, "default pool") && check(0, false, {}, "workers unpinned") &&
              check(cores + 1, true, {}, "more workers than cores");
    if (ok && cores >= 2)
        ok = check(cores - 1, true, {}, "fewer workers than cores");
    if (ok && topo.host_siblings.empty())
        std::puts("SKIP helper CPUs: the host core has no SMT sibling here (only the empty cases checked)");
    return ok && expect_affinity(allowed, "pools leave their creator's affinity");
}

// SpawnAffinity: a thread started by a thread pinned to one CPU inherits that CPU; started inside a SpawnAffinity of
// the mask taken before the pin (current_thread_affinity) it has that mask, and its creator has its pin back when the
// scope ends; a SpawnAffinity of the creator's own mask, or an invalid one, changes nothing.  This is what
// FileExpertSource::fill_many does for its threads under a request thread pinned to the pool's host core.
static bool spawn_affinity_leaves_the_pin(const std::vector<int>& allowed) {
    if (allowed.size() < 2) {
        std::puts("SKIP spawn affinity: fewer than two allowed CPUs");
        return true;
    }
    const cpu::ThreadAffinity before = cpu::current_thread_affinity();
    const std::vector<int> one{allowed.front()};
    if (!before.valid || !set_affinity(one))
        return false;
    std::vector<int> inherited, spawned;
    bool got_inherited = false, got_spawned = false, during = false;
    std::thread([&] { got_inherited = get_affinity(inherited); }).join();
    {
        const cpu::SpawnAffinity spawn(before);
        during = expect_affinity(allowed, "the creator inside a SpawnAffinity");
        std::thread([&] { got_spawned = get_affinity(spawned); }).join();
    }
    const bool back = expect_affinity(one, "a SpawnAffinity gives its creator its pin back");
    bool unchanged = false;
    {
        const cpu::SpawnAffinity own(cpu::current_thread_affinity());
        const cpu::SpawnAffinity invalid{cpu::ThreadAffinity{}};
        unchanged = expect_affinity(one, "a SpawnAffinity of the creator's own mask, or an invalid one");
    }
    unchanged = unchanged && expect_affinity(one, "after a SpawnAffinity that changed nothing");
    if (!set_affinity(allowed))
        return false;
    if (!got_inherited || inherited != one) {
        std::fprintf(stderr, "Linux affinity test: a new thread of a pinned thread has %zu CPUs, expected 1\n",
                     inherited.size());
        return false;
    }
    if (!got_spawned || spawned != allowed) {
        std::fprintf(stderr, "Linux affinity test: a thread started in a SpawnAffinity has %zu CPUs, expected %zu\n",
                     spawned.size(), allowed.size());
        return false;
    }
    return during && back && unchanged && expect_affinity(allowed, "restore after the spawned threads");
}

int main(int argc, char** argv) {
    bool require_high = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--require-high-cpu") == 0)
            require_high = true;
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    OriginalAffinity original;
    if (!original.capture()) {
        std::fprintf(stderr, "Linux affinity test: cannot capture original thread affinity\n");
        return 1;
    }
    const auto allowed = original.ids();

    bool ok = round_trip_full_mask(allowed) && worker_thread_can_pin(allowed.back());
    if (ok)
        ok = round_trip_sparse_nested(allowed);
    if (ok)
        ok = round_trip_high_cpu(allowed, require_high);
    if (ok)
        ok = negative_cpu_is_noop();
    if (ok)
        ok = failed_pin_preserves_affinity();
    if (ok)
        ok = request_pin_respects_restricted_mask(allowed);
    if (ok)
        ok = host_siblings_and_helper_pin(allowed);
    if (ok)
        ok = helper_cpus_policy(allowed);
    if (ok)
        ok = spawn_affinity_leaves_the_pin(allowed);
    const bool restored = original.restore() && expect_affinity(allowed, "final original affinity cleanup");
    if (!ok || !restored)
        return 1;

    std::printf("Linux affinity test: full, sparse/nested, worker helper, invalid-target, failed-pin, host-sibling, "
                "helper-pin, helper-CPU policy, spawn-affinity and cleanup checks passed (%zu allowed CPUs)\n",
                allowed.size());
    return 0;
}

#else

int main() {
    std::puts("SKIP Linux affinity test: Linux is required");
    return 0;
}

#endif
