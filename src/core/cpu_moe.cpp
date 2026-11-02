// src/core/cpu_moe.cpp - feed-forward networks on the CPU (include/guild/core/cpu_moe.hpp).
#include "guild/core/cpu_moe.hpp"

#include "guild/core/glm5_nl4.hpp"
#include "guild/core/iq4xs_rows512.hpp"
#include "guild/core/q8_0_rows512.hpp"
#include "guild/kernels/cpu/expert.hpp"
#include "guild/kernels/cpu/features.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"
#include "guild/kernels/cpu/iq_avx512.hpp"
#include "guild/kernels/cpu/kq_avx2.hpp"
#include "guild/kernels/cpu/numa.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace guild::core::cpu_moe {

namespace kcpu = guild::kernels::cpu;

namespace {

/// asks for `bytes` from p (a row two rows ahead of the one-token dot products, which do not ask themselves)
inline void ask(const uint8_t* p, size_t bytes) {
#if defined(__x86_64__)
    for (size_t o = 0; o < bytes; o += 64)
        _mm_prefetch((const char*) p + o, _MM_HINT_T0);
#else
    (void) p;
    (void) bytes;
#endif
}

const ggml_type_traits_cpu* cpu_traits(int type) {
    return ggml_get_type_traits_cpu((ggml_type) type);
}

#if defined(__x86_64__)
/// ggml's Q8_K block (ggml-common.h)
struct Q8KBlock {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(Q8KBlock) == 292, "block_q8_K");

/// quantize_row_q8_K_ref (the CPU backend's Q8_K from_float, scalar) with AVX-512, the same bytes: the scale from the
/// first value of the largest magnitude, nearest_int (round half to even, which a conversion in the default rounding
/// mode does) of the same single-precision product, and a block of zeros left as the reference leaves it (d and qs
/// written, bsums not).  A block with a NaN goes to the reference.  One Xeon Gold 6130 core: 4.8 ns a value for the
/// reference, 0.7 for this.
__attribute__((target("avx512f,avx512bw"))) void q8k_row_avx512(const float* x, void* vy, int64_t k) {
    static const ggml_from_float_t ref = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    Q8KBlock* y = (Q8KBlock*) vy;
    for (int64_t i = 0; i < k / 256; ++i, x += 256, ++y) {
        __m512 v[16];
        __m512 m = _mm512_setzero_ps();
        __mmask16 nan = 0;
        for (int j = 0; j < 16; ++j) {
            v[j] = _mm512_loadu_ps(x + 16 * j);
            nan |= _mm512_cmp_ps_mask(v[j], v[j], _CMP_UNORD_Q);
            m = _mm512_max_ps(m, _mm512_abs_ps(v[j]));
        }
        if (nan) {
            ref(x, y, 256);
            continue;
        }
        const float amax = _mm512_reduce_max_ps(m);
        if (amax == 0.f) {
            y->d = 0.f;
            std::memset(y->qs, 0, sizeof y->qs);
            continue;
        }
        int at = 0;
        const __m512 va = _mm512_set1_ps(amax);
        for (int j = 0; j < 16; ++j) {
            const __mmask16 e = _mm512_cmp_ps_mask(_mm512_abs_ps(v[j]), va, _CMP_EQ_OQ);
            if (e) {
                at = 16 * j + __builtin_ctz((unsigned) e);
                break;
            }
        }
        const float iscale = -127.f / x[at];
        const __m512 vs = _mm512_set1_ps(iscale);
        const __m512i lim = _mm512_set1_epi32(127);
        for (int j = 0; j < 16; ++j) {
            const __m512i q = _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(vs, v[j])), lim);
            const __m128i q8 = _mm512_cvtepi32_epi8(q);
            _mm_storeu_si128((__m128i*) (y->qs + 16 * j), q8);
            y->bsums[j] = (int16_t) _mm512_reduce_add_epi32(_mm512_cvtepi8_epi32(q8));
        }
        y->d = 1.f / iscale;
    }
}
#endif

/// a type's from_float (q8k_row_avx512 for Q8_K where the CPU has AVX-512BW)
ggml_from_float_t from_float_of(int type) {
#if defined(__x86_64__)
    if (type == (int) GGML_TYPE_Q8_K && kcpu::cpu_avx512bw_ok() &&
        ggml_row_size(GGML_TYPE_Q8_K, 256) == sizeof(Q8KBlock))
        return &q8k_row_avx512;
#endif
    return cpu_traits(type)->from_float;
}

/// ggml-cpu's tables (the FP16 -> FP32 conversion its dot products read for block scales, the i-quant grids): without
/// them a single-token Q8_0 vec_dot returns 0.  Every entry point that leads to a dot product calls this.
void ggml_cpu_ready() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

/// `p` moved back by `rows` rows of `row` bytes (the base that puts row `rows` at `p`)
inline const uint8_t* rebase(const uint8_t* p, int64_t rows, size_t row) {
    return (const uint8_t*) ((uintptr_t) p - (uintptr_t) rows * row);
}

} // namespace

int act_type(int gu_type) {
    return gu_type == kcpu::kIq3sNl4Type ? (int) GGML_TYPE_Q8_K : (int) cpu_traits(gu_type)->vec_dot_type;
}

// ---- Launcher, PinCaller, Pool ------------------------------------------------------------------------------------

Launcher::Launcher(int cpu, int device) : th_([this, cpu, device] { loop(cpu, device); }) {
}
Launcher::~Launcher() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    th_.join();
}
void Launcher::post(std::function<void()> f) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        task_ = std::move(f);
        pending_ = true;
    }
    cv_.notify_all();
}
void Launcher::wait() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return !pending_; });
}
void Launcher::loop(int cpu, int device) {
#if defined(__linux__)
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
    (void) cpu;
    cudaSetDevice(device);
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return pending_ || stop_; });
        if (stop_)
            return;
        std::function<void()> f = std::move(task_);
        lk.unlock();
        f();
        lk.lock();
        pending_ = false;
        cv_.notify_all();
    }
}

PinCaller::PinCaller(int cpu) {
#if defined(__linux__)
    static_assert(sizeof(cpu_set_t) <= sizeof(saved_), "cpu_set_t");
    cpu_set_t* saved = reinterpret_cast<cpu_set_t*>(saved_);
    if (cpu < 0 || pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), saved) != 0)
        return;
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    pinned_ = pthread_setaffinity_np(pthread_self(), sizeof(one), &one) == 0;
#else
    (void) cpu;
#endif
}
PinCaller::~PinCaller() {
#if defined(__linux__)
    if (pinned_)
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), reinterpret_cast<cpu_set_t*>(saved_));
#endif
}

void Pool::pin(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void) cpu;
#endif
}

Workers make_workers(bool numa_split, int threads) {
    ggml_cpu_ready();
    Workers w;
    const kcpu::NumaTopology& topo = kcpu::numa_topology();
    w.split = topo.nodes == 2 && numa_split && kcpu::numa_split_requested(); // GUILD_EXPERT_NUMA
    std::vector<std::vector<int>> cpus(w.split ? 2 : 1);
    const int ncpu = (int) std::thread::hardware_concurrency();
    auto first_sibling = [](int cpu) { // the lowest CPU sharing this one's core
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        int first = cpu;
        if (f >> first)
            return first;
        return cpu;
    };
    for (int cpu = 0; cpu < ncpu; ++cpu) {
        if (first_sibling(cpu) != cpu)
            continue; // a hyper-thread sibling
        const int node = w.split && cpu < (int) topo.cpu_node.size() && topo.cpu_node[(size_t) cpu] == 1 ? 1 : 0;
        cpus[(size_t) node].push_back(cpu);
    }
    // the calling thread runs on the first core (PinCaller); a thread budget trims both nodes alike
    if (!cpus[0].empty()) {
        w.caller_cpu = cpus[0].front();
        cpus[0].erase(cpus[0].begin());
    }
    if (threads > 0)
        for (auto& c : cpus)
            c.resize(std::min(c.size(), (size_t) std::max(1, threads / (int) cpus.size())));
    w.pool = std::make_unique<Pool>(cpus);
    // the launcher beside the calling thread, on its core's other hyper-thread when it has one
    int sib = -1;
    std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(w.caller_cpu) + "/topology/thread_siblings_list");
    std::string list;
    if (w.caller_cpu >= 0 && std::getline(f, list))
        for (size_t i = 0; i < list.size();) {
            const size_t j = list.find_first_of(",-", i);
            const int c = std::atoi(list.substr(i, j == std::string::npos ? std::string::npos : j - i).c_str());
            if (c != w.caller_cpu) {
                sib = c;
                break;
            }
            if (j == std::string::npos)
                break;
            i = j + 1;
        }
    int dev = 0;
    cudaGetDevice(&dev);
    w.launcher = std::make_unique<Launcher>(sib, dev);
    return w;
}

// ---- the experts' rows on the NUMA nodes ----------------------------------------------------------------------------

bool place_ffns(Pool& pool, const std::vector<CpuFfn*>& all, int64_t N, bool split, bool nl4, uint8_t* arena[2],
                size_t arena_bytes[2], std::string& err) {
    const int nn = split ? 2 : 1;
    const int64_t dh = split ? N / 2 : N;
    auto h_of = [&](const CpuFfn& f) { return split ? f.n_ff / 2 : f.n_ff; };
    auto repacked = [&](const CpuFfn& f) { return nl4 && f.gu_type == GGML_TYPE_IQ3_S; };
    auto gu_row_of = [&](const CpuFfn& f) { return repacked(f) ? (size_t) (N / 256) * kcpu::kIq3sNl4Block : f.gu_row; };
    auto n_gu = [](const CpuFfn& f) { return f.act == CpuFfn::ReLU2 ? (size_t) 1 : (size_t) 2; }; // up, or gate + up
    size_t node_bytes[2] = {0, 0};
    std::vector<std::array<size_t, 2>> at; // per FFN in `all`: the offset of its part in each arena
    for (const CpuFfn* fp : all) {
        const CpuFfn& f = *fp;
        const size_t gr = gu_row_of(f);
        const int64_t h = h_of(f);
        at.push_back({node_bytes[0], node_bytes[1]});
        node_bytes[0] += n_gu(f) * (size_t) h * gr + (size_t) dh * f.d_row;
        if (nn == 2)
            node_bytes[1] += n_gu(f) * (size_t) (f.n_ff - h) * gr + (size_t) (N - dh) * f.d_row;
    }
    for (int node = 0; node < nn; ++node) {
#if defined(__linux__)
        void* p = mmap(nullptr, node_bytes[node], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            err = "no memory for the experts (" + std::to_string(node_bytes[node] >> 30) + " GiB on node " +
                  std::to_string(node) + ")";
            return false;
        }
        unsigned long mask = 1ul << node;
        if (split && syscall(SYS_mbind, p, node_bytes[node], 2 /* MPOL_BIND */, &mask, 64ul, 0u) != 0)
            std::fprintf(stderr, "guild: mbind to node %d failed; the experts' halves go where they land\n", node);
        madvise(p, node_bytes[node], MADV_HUGEPAGE);
        arena[node] = (uint8_t*) p;
        arena_bytes[node] = node_bytes[node];
#else
        err = "the experts' NUMA placement needs Linux";
        return false;
#endif
    }
    const int n_all = (int) all.size();
    std::atomic<bool> repack_ok{true};
    pool.run_nodes(n_all, nn == 2 ? n_all : 0, [&](int node, int i) {
        CpuFfn& f = *all[(size_t) i];
        const int64_t h = h_of(f);
        const int64_t r0 = node ? h : 0, rg = node ? f.n_ff - h : h;
        const int64_t d0 = node ? dh : 0, rd = node ? N - dh : dh;
        const size_t gr = gu_row_of(f), ng = n_gu(f);
        uint8_t* dst = arena[node] + at[(size_t) i][(size_t) node];
        if (repacked(f)) {
            for (int64_t r = 0; r < rg; ++r) {
                bool ok = kcpu::iq3s_nl4_repack(f.gate[0] + (size_t) (r0 + r) * f.gu_row, (int) (N / 256),
                                                dst + (size_t) r * gr);
                ok &= kcpu::iq3s_nl4_repack(f.up[0] + (size_t) (r0 + r) * f.gu_row, (int) (N / 256),
                                            dst + (size_t) (rg + r) * gr);
                if (!ok)
                    repack_ok = false;
            }
        } else if (ng == 2) {
            std::memcpy(dst, f.gate[0] + (size_t) r0 * f.gu_row, (size_t) rg * f.gu_row);
            std::memcpy(dst + (size_t) rg * gr, f.up[0] + (size_t) r0 * f.gu_row, (size_t) rg * f.gu_row);
        } else {
            std::memcpy(dst, f.up[0] + (size_t) r0 * f.gu_row, (size_t) rg * f.gu_row);
        }
        std::memcpy(dst + ng * (size_t) rg * gr, f.down[0] + (size_t) d0 * f.d_row, (size_t) rd * f.d_row);
    });
    if (!repack_ok) {
        err = "an IQ3_S expert row does not repack (a grid value outside the NL4 codes); run with GUILD_GLM_NL4=0";
        return false;
    }
    // the pointers, once every part is copied (the copies read the mapping through gate[0] and friends)
    for (int i = 0; i < n_all; ++i) {
        CpuFfn& f = *all[(size_t) i];
        const size_t gr = gu_row_of(f), ng = n_gu(f);
        const int64_t h = h_of(f);
        for (int node = 0; node < 2; ++node) {
            const int src = node < nn ? node : 0;
            const int64_t r0 = src ? h : 0, rg = src ? f.n_ff - h : h, d0 = src ? dh : 0;
            const uint8_t* base = arena[src] + at[(size_t) i][(size_t) src];
            if (ng == 2) {
                f.gate[node] = rebase(base, r0, gr);
                f.up[node] = rebase(base + (size_t) rg * gr, r0, gr);
            } else {
                f.gate[node] = nullptr;
                f.up[node] = rebase(base, r0, gr);
            }
            f.down[node] = rebase(base + ng * (size_t) rg * gr, d0, f.d_row);
        }
        if (repacked(f))
            f.gu_type = kcpu::kIq3sNl4Type;
        f.gu_row = gr;
        f.gu_split = h;
        f.d_split = dh;
    }
    return true;
}

// ---- the expert phases --------------------------------------------------------------------------------------------

void Experts::run(Pool& pool, int64_t N, int T, int max_t, const int32_t* ids, const float* wts, int n_used,
                  const std::vector<CpuFfn>& experts, const CpuFfn* own, const float* x, float* y, bool single,
                  int t_first) {
    ggml_cpu_ready();
    // Groups: one FFN slice and the tokens it serves (a routed expert and the tokens routed to it, or the layer's own
    // FFN for every token).  Units: (group, token) pairs, each with its hidden activation.
    struct Group {
        CpuFfn f;
        std::vector<int> units;          // indices into `unit_t` / `unit_w`
        std::vector<int> gu_c, d_c;      // the units' chunks for gate/up and for down: [c[i], c[i + 1])
        std::vector<size_t> gu_pk, d_pk; // q8_0_rows512's rows: each chunk's packed activations in pkbuf_ (else empty)
    };
    std::vector<Group> groups;
    std::vector<int> unit_t, unit_g;
    std::vector<float> unit_w;
    t_first = std::clamp(t_first, 0, T);
    if (ids) {
        // the experts in the order the window's tokens first name them, those a computed token names
        std::vector<int> group_of(experts.size(), -1);
        std::vector<int> order;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < n_used; ++j) {
                const int ex = ids[t * n_used + j];
                if (group_of[(size_t) ex] == -1) {
                    group_of[(size_t) ex] = -2;
                    order.push_back(ex);
                }
            }
        for (int t = t_first; t < T; ++t)
            for (int j = 0; j < n_used; ++j)
                group_of[(size_t) ids[t * n_used + j]] = -3;
        for (int ex : order)
            if (group_of[(size_t) ex] == -3) {
                group_of[(size_t) ex] = (int) groups.size();
                Group gr;
                gr.f = experts[(size_t) ex];
                groups.push_back(std::move(gr));
            }
        for (int t = t_first; t < T; ++t)
            for (int j = 0; j < n_used; ++j) {
                const int gi = group_of[(size_t) ids[t * n_used + j]];
                groups[(size_t) gi].units.push_back((int) unit_t.size());
                unit_t.push_back(t);
                unit_g.push_back(gi);
                unit_w.push_back(wts[t * n_used + j]);
            }
    }
    if (own) {
        Group gr;
        gr.f = *own;
        const int gi = (int) groups.size();
        for (int t = t_first; t < T; ++t) {
            gr.units.push_back((int) unit_t.size());
            unit_t.push_back(t);
            unit_g.push_back(gi);
            unit_w.push_back(1.f);
        }
        groups.push_back(std::move(gr));
    }
    const int U = (int) unit_t.size();

    // the quantized inputs: one per (token, activation type)
    std::map<int, size_t> act_off; // (vec_dot_type * max_t + t) -> offset
    std::vector<size_t> act_at((size_t) U);
    size_t total = 0;
    for (int u = 0; u < U; ++u) {
        const int at = act_type(groups[(size_t) unit_g[(size_t) u]].f.gu_type);
        const int key = at * max_t + unit_t[(size_t) u];
        auto it = act_off.find(key);
        if (it == act_off.end()) {
            it = act_off.emplace(key, total).first;
            total += ggml_row_size((ggml_type) at, N);
        }
        act_at[(size_t) u] = it->second;
    }
    static const bool timing = std::getenv("GUILD_GLM_TIMING") != nullptr || std::getenv("GUILD_MOE_TIMING") != nullptr;
    auto tp = std::chrono::steady_clock::now();
    auto lap = [&](int k) {
        if (!timing)
            return;
        const auto now = std::chrono::steady_clock::now();
        phase_ms[k] += std::chrono::duration<double, std::milli>(now - tp).count();
        tp = now;
    };
    if (timing)
        for (const Group& gr : groups) {
            const double gu = (double) gr.f.n_ff * (double) gr.f.gu_row * (gr.f.act == CpuFfn::ReLU2 ? 1 : 2),
                         dn = (double) N * (double) gr.f.d_row;
            gu_bytes += gu;
            down_bytes += dn;
            weight_bytes += gu + dn;
        }
    // a few rows or chunks (a decode step's) on this thread, a prompt window's across the pool
    auto each = [&](int n, size_t bytes, const std::function<void(int)>& f) {
        if (bytes < (size_t) 1 << 18)
            for (int i = 0; i < n; ++i)
                f(i);
        else
            pool.run(n, f);
    };
    // the buffers only grow: resize() zero-fills what it adds, megabytes on this thread after a smaller window
    auto grow = [](std::vector<uint8_t>& v, size_t n) {
        if (v.size() < n)
            v.resize(n);
    };
    grow(actbuf_, total + 64);
    const std::vector<std::pair<int, size_t>> act_rows(act_off.begin(), act_off.end());
    each((int) act_rows.size(), total, [&](int i) {
        const int at = act_rows[(size_t) i].first / max_t, t = act_rows[(size_t) i].first % max_t;
        from_float_of(at)(x + (size_t) t * N, actbuf_.data() + act_rows[(size_t) i].second, N);
    });
    std::vector<size_t> h_off((size_t) U + 1, 0), hq_off((size_t) U + 1, 0);
    for (int u = 0; u < U; ++u) {
        const CpuFfn& f = groups[(size_t) unit_g[(size_t) u]].f;
        h_off[(size_t) u + 1] = h_off[(size_t) u] + (size_t) f.n_ff;
        const int dt = (int) cpu_traits(f.d_type)->vec_dot_type;
        hq_off[(size_t) u + 1] = hq_off[(size_t) u] + ggml_row_size((ggml_type) dt, f.n_ff);
    }
    grow(hbuf_, h_off[(size_t) U] * 4);
    grow(hqbuf_, hq_off[(size_t) U] + 64);
    float* hv = (float*) hbuf_.data();

    // the multi-token row kernel for a weight type, or null (ggml-cpu's single-token dot then)
    using RowsFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
    auto multi = [](int type) -> RowsFn {
        if (kcpu::cpu_avx512_ok() && kcpu::iq512_supported(type))
            return &kcpu::iq512_rows;
        if (kcpu::cpu_avx2_ok() && kcpu::iq256_supported(type))
            return &kcpu::iq256_rows;
        if (kcpu::cpu_avx2_ok() && kcpu::kq256_supported(type))
            return &kcpu::kq256_rows;
        return nullptr;
    };
    // IQ4_XS down rows two to an AVX-512BW register for any number of tokens, one-token windows included: each
    // (row, token) is ggml-cpu's dot to the bit (iq4xs_rows512_test).  All 32 cores of 2x Xeon Gold 6130 streaming
    // GLM-5.3-Flash's 2,048-value rows: 161 -> 193 GB/s at one token, 91 -> 162 at two.  GUILD_IQ4XS_512=0 (read each
    // call, for A/Bs in one process): ggml-cpu's dot, or the multi-token kernel above.
    const char* x512 = std::getenv("GUILD_IQ4XS_512");
    const bool iq4xs = iq4xs_rows512_ok() && !(x512 && x512[0] == '0');
    constexpr int RB = 64, MT = kQ8Rows512MaxT > kcpu::MAXT ? kQ8Rows512MaxT : kcpu::MAXT;
    // a down item's rows of every group (down_rows, at most RB)
    const int DRB = std::clamp(down_rows, 4, RB) / 4 * 4;
    // and the last items of each node's queue, about one a worker, are split in four, so that the phase ends with
    // short items (2x Xeon Gold 6130: CPU experts -0.9 ms a 2-token step)
    constexpr int kTail = 16, GTR = RB / 4;
    const int DTR = DRB / 4;

    // Q8_0 rows go through q8_0_rows512 in any window (each token's result is bitwise ggml-cpu's dot product), a chunk
    // of up to kQ8Rows512MaxT tokens at a time, as even as they come, packed once: for gate/up here, for down with the
    // hidden activations.  Other rows keep chunks of kcpu::MAXT tokens.
    const bool q8 = q8_0_rows512_ok();
    auto chunks = [](size_t units, size_t most, bool even, std::vector<int>& c) {
        const size_t k = (units + most - 1) / most;
        c.assign(1, 0);
        for (size_t i = 1; i <= k; ++i)
            c.push_back((int) (even ? units * i / k : std::min(units, i * most)));
    };
    size_t pk_bytes = 0;
    std::vector<std::pair<int, int>> gu_pack, d_pack; // (group, chunk)
    for (int gi = 0; gi < (int) groups.size(); ++gi) {
        Group& gr = groups[(size_t) gi];
        const bool pg = q8 && gr.f.gu_type == (int) GGML_TYPE_Q8_0, pd = q8 && gr.f.d_type == (int) GGML_TYPE_Q8_0;
        chunks(gr.units.size(), pg ? kQ8Rows512MaxT : kcpu::MAXT, pg, gr.gu_c);
        chunks(gr.units.size(), pd ? kQ8Rows512MaxT : kcpu::MAXT, pd, gr.d_c);
        for (size_t c = 0; pg && c + 1 < gr.gu_c.size(); ++c) {
            gr.gu_pk.push_back(pk_bytes);
            pk_bytes += q8_0_packed_bytes((int) N, gr.gu_c[c + 1] - gr.gu_c[c]);
            gu_pack.emplace_back(gi, (int) c);
        }
        for (size_t c = 0; pd && c + 1 < gr.d_c.size(); ++c) {
            gr.d_pk.push_back(pk_bytes);
            pk_bytes += q8_0_packed_bytes((int) gr.f.n_ff, gr.d_c[c + 1] - gr.d_c[c]);
            d_pack.emplace_back(gi, (int) c);
        }
    }
    grow(pkbuf_, pk_bytes + 64);
    uint8_t* pk = pkbuf_.data() + (64 - (uintptr_t) pkbuf_.data() % 64) % 64;
    each((int) gu_pack.size(), pk_bytes, [&](int i) {
        const Group& gr = groups[(size_t) gu_pack[(size_t) i].first];
        const int c = gu_pack[(size_t) i].second, c0 = gr.gu_c[(size_t) c], nt = gr.gu_c[(size_t) c + 1] - c0;
        const void* act[MT];
        for (int k = 0; k < nt; ++k)
            act[k] = actbuf_.data() + act_at[(size_t) gr.units[(size_t) (c0 + k)]];
        q8_0_pack(act, nt, (int) N, pk + gr.gu_pk[(size_t) c]);
    });
    lap(0);
    // phase 1: gate/up, each row block on the queue of the node that holds it
    struct Item {
        int gi, r0, r1; // group, rows
    };
    // A node's threads take the items in queue order, so the items next to each other in a queue run at the same
    // time.  In each group's order, sixteen threads read sixteen adjacent row blocks of one expert at once, which a
    // DRAM channel serves slower than the same reads apart: 32 threads over 2x Xeon Gold 6130 read 137 KiB items at
    // 193 GB/s side by side, 214-216 spread, and 70 KiB items at 175 against 208.  So the queue takes the groups in
    // turn, a group's blocks from its two halves alternately (GUILD_CPU_MOE_SPREAD=0, read each call: group by group).
    const char* spr = std::getenv("GUILD_CPU_MOE_SPREAD");
    const bool spread = !(spr && spr[0] == '0');
    // the k-th of n blocks in the spread order: 0, h, 1, h + 1, ... (h = ceil(n / 2))
    auto spread_at = [](size_t k, size_t n) { return k % 2 == 0 ? k / 2 : (n + 1) / 2 + k / 2; };
    std::vector<Item> items[2];
    for (int node = 0; node < 2; ++node) {
        std::vector<std::vector<Item>> per(groups.size());
        size_t most = 0;
        for (int gi = 0; gi < (int) groups.size(); ++gi) {
            const CpuFfn& f = groups[(size_t) gi].f;
            const int64_t a = node ? f.gu_split : 0, b = node ? f.n_ff : f.gu_split;
            for (int64_t r = a; r < b; r += RB)
                per[(size_t) gi].push_back({gi, (int) r, (int) std::min<int64_t>(r + RB, b)});
            most = std::max(most, per[(size_t) gi].size());
        }
        if (!spread) {
            for (const auto& v : per)
                items[node].insert(items[node].end(), v.begin(), v.end());
            continue;
        }
        for (size_t k = 0; k < most; ++k)
            for (const auto& v : per)
                if (k < v.size())
                    items[node].push_back(v[spread_at(k, v.size())]);
    }
    for (auto& v : items) {
        const size_t k0 = v.size() > (size_t) kTail ? v.size() - kTail : 0;
        std::vector<Item> tail;
        for (size_t k = k0; k < v.size(); ++k)
            for (int r = v[k].r0; r < v[k].r1; r += GTR)
                tail.push_back({v[k].gi, r, std::min(r + GTR, v[k].r1)});
        v.resize(k0);
        v.insert(v.end(), tail.begin(), tail.end());
    }
    // The hidden activations' quantization: from_float quantizes each 256-row block on its own (Q8_K's blocks are
    // 256, Q8_0's 32), so the item that completes a group's block of 256 rows quantizes that block for the group's
    // units, and the bytes are those of a whole-row from_float.  qneed: the items that write each (group, block);
    // qcnt: how many have.
    constexpr int QB = 256;
    bool fuse = true;
    for (const Group& gr : groups) {
        const int dt = (int) cpu_traits(gr.f.d_type)->vec_dot_type;
        if (gr.f.n_ff % QB != 0 || QB % ggml_blck_size((ggml_type) dt) != 0)
            fuse = false;
    }
    std::vector<int> qbase(groups.size() + 1, 0);
    if (fuse) {
        for (size_t gi = 0; gi < groups.size(); ++gi)
            qbase[gi + 1] = qbase[gi] + (int) (groups[gi].f.n_ff / QB);
        const size_t nb = (size_t) qbase.back();
        qneed_.assign(nb, 0);
        for (int node = 0; node < 2; ++node)
            for (const Item& it : items[node])
                for (int b = it.r0 / QB; b <= (it.r1 - 1) / QB; ++b)
                    ++qneed_[(size_t) (qbase[(size_t) it.gi] + b)];
        if (qcnt_cap_ < nb) {
            qcnt_.reset(new std::atomic<int>[nb]);
            qcnt_cap_ = nb;
        }
        for (size_t k = 0; k < nb; ++k)
            qcnt_[k].store(0, std::memory_order_relaxed);
    }
    // One phase for gate/up and down (GUILD_CPU_MOE_FLOW=0, read each call: two phases): each queue holds its gate/up
    // items and then its down items, and a down item, which sums its rows over every group in order, waits for a group
    // only until that group's hidden rows are quantized.  The threads that end the gate/up items early start down
    // items instead of waiting for the last gate/up item; the arithmetic and its order are the two phases', so the
    // bytes are the same.  A gate/up item waits on nothing, and a queue's down items come after all its gate/up items,
    // so a waiting down item waits only on items already taken.
    const char* fl = std::getenv("GUILD_CPU_MOE_FLOW");
    const int G = (int) groups.size();
    const bool flow = fuse && !single && G > 0 && !(fl && fl[0] == '0');
    if (flow) {
        if (gflow_cap_ < (size_t) (2 * G)) {
            gflow_.reset(new Line[(size_t) (2 * G)]);
            gflow_cap_ = (size_t) (2 * G);
        }
        for (int k = 0; k < 2 * G; ++k) // [0, G): a group's 256-row blocks quantized; [G, 2G): its down rows ready
            gflow_[(size_t) k].v.store(0, std::memory_order_relaxed);
    }
    auto pack_down = [&](const Group& gr, int c) {
        const int c0 = gr.d_c[(size_t) c], nt = gr.d_c[(size_t) c + 1] - c0;
        const void* hq[MT];
        for (int k = 0; k < nt; ++k)
            hq[k] = hqbuf_.data() + hq_off[(size_t) gr.units[(size_t) (c0 + k)]];
        q8_0_pack(hq, nt, (int) gr.f.n_ff, pk + gr.d_pk[(size_t) c]);
    };
    // with the timing, each phase's item times summed over the threads too (busy_ms)
    auto run_timed = [&](int k, int n0, int n1, const std::function<void(int, int)>& f) {
        if (!timing) {
            pool.run_nodes(n0, n1, f);
            return;
        }
        std::atomic<int64_t> ns{0};
        pool.run_nodes(n0, n1, [&](int node, int i) {
            const auto a = std::chrono::steady_clock::now();
            f(node, i);
            ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - a).count(),
                std::memory_order_relaxed);
        });
        busy_ms[k] += (double) ns.load() * 1e-6;
    };
    auto gu_item = [&](int node, int i) {
        const Item& it = items[node][(size_t) i];
        const int gi = it.gi;
        const Group& gr = groups[(size_t) gi];
        const CpuFfn& f = gr.f;
        const int r0 = it.r0, r1 = it.r1, nr = r1 - r0;
        const bool gated = f.act != CpuFfn::ReLU2;
        const uint8_t* wg = f.gate[node];
        const uint8_t* wu = f.up[node];
        const bool nl4 = f.gu_type == kcpu::kIq3sNl4Type;
        const ggml_type_traits_cpu* tr = nl4 ? nullptr : cpu_traits(f.gu_type);
        const RowsFn mt = !nl4 && !single && gr.units.size() >= 2 ? multi(f.gu_type) : nullptr;
        float gbuf[MT][RB], ubuf[MT][RB];
        for (size_t c = 0; c + 1 < gr.gu_c.size(); ++c) {
            const size_t c0 = (size_t) gr.gu_c[c];
            const int nt = gr.gu_c[c + 1] - (int) c0;
            const void* act[MT];
            float* go[MT];
            float* uo[MT];
            for (int k = 0; k < nt; ++k) {
                act[k] = actbuf_.data() + act_at[(size_t) gr.units[c0 + k]];
                go[k] = gbuf[k] - r0; // the kernels index outputs by row
                uo[k] = ubuf[k] - r0;
            }
            if (!gr.gu_pk.empty()) {
                if (gated)
                    q8_0_rows512(wg, f.gu_row, (int) N, pk + gr.gu_pk[c], nt, go, r0, r1, wu + (size_t) r0 * f.gu_row,
                                 f.gu_row);
                q8_0_rows512(wu, f.gu_row, (int) N, pk + gr.gu_pk[c], nt, uo, r0, r1);
            } else if (nl4) {
                if (gated)
                    glm5cpu::nl4_rows(wg, f.gu_row, (int) N, act, nt, go, r0, r1);
                glm5cpu::nl4_rows(wu, f.gu_row, (int) N, act, nt, uo, r0, r1);
            } else if (mt && nt >= 2) {
                if (gated)
                    mt(f.gu_type, wg, f.gu_row, (int) N, act, nt, go, r0, r1);
                mt(f.gu_type, wu, f.gu_row, (int) N, act, nt, uo, r0, r1);
            } else {
                for (int k = 0; k < nt; ++k)
                    for (int r = r0; r < r1; ++r) {
                        if (k == 0 && r + 2 < r1) { // the DRAM stream, once: the rows after next
                            if (gated)
                                ask(wg + (size_t) (r + 2) * f.gu_row, f.gu_row);
                            ask(wu + (size_t) (r + 2) * f.gu_row, f.gu_row);
                        }
                        if (gated)
                            tr->vec_dot((int) N, &go[k][r], 0, wg + (size_t) r * f.gu_row, 0, act[k], 0, 1);
                        tr->vec_dot((int) N, &uo[k][r], 0, wu + (size_t) r * f.gu_row, 0, act[k], 0, 1);
                    }
            }
            for (int k = 0; k < nt; ++k) {
                float* h = hv + h_off[(size_t) gr.units[c0 + k]];
                if (!gated) {
                    for (int j = 0; j < nr; ++j) { // relu(up)^2, as torch's relu then square
                        const float r = std::max(ubuf[k][j], 0.f);
                        h[r0 + j] = r * r;
                    }
                    continue;
                }
                for (int j = 0; j < nr; ++j) {
                    float gt = gbuf[k][j], up = ubuf[k][j];
                    if (f.lim > 0.f) {
                        gt = std::min(gt, f.lim);
                        up = std::clamp(up, -f.lim, f.lim);
                    }
                    h[r0 + j] = gt / (1.f + std::exp(-gt)) * up;
                }
            }
        }
        if (fuse) {
            const int dt = (int) cpu_traits(f.d_type)->vec_dot_type;
            const size_t qb_bytes = ggml_row_size((ggml_type) dt, QB);
            const ggml_from_float_t from_float = from_float_of(dt);
            for (int b = r0 / QB; b <= (r1 - 1) / QB; ++b) {
                const size_t k = (size_t) (qbase[(size_t) gi] + b);
                if (qcnt_[k].fetch_add(1, std::memory_order_acq_rel) + 1 != qneed_[k])
                    continue;
                for (int u : gr.units)
                    from_float(hv + h_off[(size_t) u] + (size_t) b * QB,
                               hqbuf_.data() + hq_off[(size_t) u] + (size_t) b * qb_bytes, QB);
                // the group's last block: its packed down chunks, then its down rows may start
                if (flow && gflow_[(size_t) gi].v.fetch_add(1, std::memory_order_acq_rel) + 1 == (int) (f.n_ff / QB)) {
                    for (int c = 0; c < (int) gr.d_pk.size(); ++c)
                        pack_down(gr, c);
                    gflow_[(size_t) (G + gi)].v.store(1, std::memory_order_release);
                }
            }
        }
    };
    if (!flow) {
        run_timed(1, (int) items[0].size(), (int) items[1].size(), gu_item);
        lap(1);
    }
    // the hidden activations, unless gate/up quantized them, and the down chunks q8_0_rows512 reads packed
    auto quantize = [&](int u) {
        const CpuFfn& f = groups[(size_t) unit_g[(size_t) u]].f;
        from_float_of((int) cpu_traits(f.d_type)->vec_dot_type)(hv + h_off[(size_t) u],
                                                                hqbuf_.data() + hq_off[(size_t) u], f.n_ff);
    };
    if (flow) {
        // packed and quantized by the gate/up items
    } else if (!fuse) {
        // a task per down chunk: its units quantized, then packed
        std::vector<std::pair<int, int>> tasks;
        for (int gi = 0; gi < (int) groups.size(); ++gi)
            for (int c = 0; c + 1 < (int) groups[(size_t) gi].d_c.size(); ++c)
                tasks.emplace_back(gi, c);
        pool.run((int) tasks.size(), [&](int i) {
            const Group& gr = groups[(size_t) tasks[(size_t) i].first];
            const int c = tasks[(size_t) i].second;
            for (int k = gr.d_c[(size_t) c]; k < gr.d_c[(size_t) c + 1]; ++k)
                quantize(gr.units[(size_t) k]);
            if (!gr.d_pk.empty())
                pack_down(gr, c);
        });
    } else if (!d_pack.empty()) {
        each((int) d_pack.size(), pk_bytes,
             [&](int i) { pack_down(groups[(size_t) d_pack[(size_t) i].first], d_pack[(size_t) i].second); });
    }
    lap(2);
    // phase 2: down rows, summed over each token's units; the first half of the rows on node 0 (where the split puts
    // them), the second on node 1
    const int64_t half = (N / 2 + RB - 1) / RB * RB;
    const int64_t d0[2] = {0, half}, d1[2] = {half, N};
    int dmain[2], dn[2]; // a node's items of DRB rows, and of all
    for (int node = 0; node < 2; ++node) {
        const int64_t rows = d1[node] - d0[node];
        dmain[node] = (int) std::max<int64_t>(0, rows / DRB - kTail);
        dn[node] = dmain[node] + (int) ((rows - (int64_t) dmain[node] * DRB + DTR - 1) / DTR);
    }
    // single: each unit's rows kept, then each token's units summed in the order a one-token window sums them (its
    // routing slots, then its own FFN): the group order follows the window's other tokens
    std::vector<std::vector<int>> token_units;
    if (single) {
        token_units.resize((size_t) T);
        for (int u = 0; u < U; ++u)
            token_units[(size_t) unit_t[(size_t) u]].push_back(u);
    }
    // spread: the items next to each other in the queue take row blocks apart, and each starts at its own group (a
    // group's rows kept, then all of them summed in group order: the same sums as group by group)
    size_t max_units = 0;
    for (const Group& gr : groups)
        max_units = std::max(max_units, gr.units.size());
    auto down_item = [&](int node, int i) {
        const int m = dmain[node];
        const int ib = spread && i < m ? (int) spread_at((size_t) i, (size_t) m) : i;
        const int r0 = (int) d0[node] + (ib < m ? ib * DRB : m * DRB + (ib - m) * DTR);
        const int r1 = (int) std::min<int64_t>(r0 + (ib < m ? DRB : DTR), d1[node]), nr = r1 - r0;
        for (int t = 0; t < T; ++t)
            std::fill(y + (size_t) t * N + r0, y + (size_t) t * N + r1, 0.f);
        thread_local std::vector<float> ub; // single: [unit][row of the item]
        if (single)
            ub.resize((size_t) U * DRB);
        const size_t NG = groups.size();
        const size_t g0 = spread && !single && NG > 0 ? (size_t) i % NG : 0;
        thread_local std::vector<float> gbufs; // g0 > 0: [group][unit of the group][row]
        if (g0 != 0)
            gbufs.resize(NG * max_units * RB);
        float dbuf_own[MT][RB];
        for (size_t gs = 0; gs < NG; ++gs) {
            const size_t gi = (g0 + gs) % NG;
            const Group& gr = groups[gi];
            // the next group taken, for the stream hints
            const Group* gnext = gs + 1 < NG ? &groups[(gi + 1) % NG] : nullptr;
            if (flow)
                while (gflow_[(size_t) G + gi].v.load(std::memory_order_acquire) == 0)
                    __builtin_ia32_pause();
            const CpuFfn& f = gr.f;
            const ggml_type_traits_cpu* tr = cpu_traits(f.d_type);
            const bool x4 = iq4xs && f.d_type == (int) GGML_TYPE_IQ4_XS;
            const RowsFn mt = !single && gr.units.size() >= 2 ? multi(f.d_type) : nullptr;
            for (size_t c = 0; c + 1 < gr.d_c.size(); ++c) {
                const size_t c0 = (size_t) gr.d_c[c];
                const int nt = gr.d_c[c + 1] - (int) c0;
                // a group's rows go to its slots while the sums wait for group order (g0 > 0)
                float(*dbuf)[RB] =
                    g0 != 0 ? reinterpret_cast<float(*)[RB]>(gbufs.data() + (gi * max_units + c0) * RB) : dbuf_own;
                const void* hq[MT];
                float* dout[MT];
                for (int k = 0; k < nt; ++k) {
                    hq[k] = hqbuf_.data() + hq_off[(size_t) gr.units[c0 + k]];
                    dout[k] = dbuf[k] - r0;
                }
                // the block lies on one side of this group's split (the split is at `half` or past N)
                const uint8_t* wd = f.down[r0 >= f.d_split];
                if (!gr.d_pk.empty()) {
                    // after the group's last chunk, the next group's rows are the ones to ask for
                    const CpuFfn* nf = c + 2 == gr.d_c.size() && gnext ? &gnext->f : nullptr;
                    q8_0_rows512(wd, f.d_row, (int) f.n_ff, pk + gr.d_pk[c], nt, dout, r0, r1,
                                 nf ? nf->down[r0 >= nf->d_split] + (size_t) r0 * nf->d_row : nullptr,
                                 nf ? nf->d_row : 0);
                } else if (x4) {
                    // after the group's last chunk, the next group's rows are the ones to ask for
                    const CpuFfn* nf = c + 2 == gr.d_c.size() && gnext ? &gnext->f : nullptr;
                    iq4xs_rows512(wd, f.d_row, (int) f.n_ff, hq, nt, dout, r0, r1,
                                  nf ? nf->down[r0 >= nf->d_split] + (size_t) r0 * nf->d_row : nullptr,
                                  nf ? nf->d_row : 0);
                } else if (mt && nt >= 2) {
                    mt(f.d_type, wd, f.d_row, (int) f.n_ff, hq, nt, dout, r0, r1);
                } else {
                    // the DRAM stream, once: the rows after next, then the next group's first rows
                    const CpuFfn* nf = gnext ? &gnext->f : nullptr;
                    for (int k = 0; k < nt; ++k)
                        for (int r = r0; r < r1; ++r) {
                            if (k == 0 && r + 2 < r1)
                                ask(wd + (size_t) (r + 2) * f.d_row, f.d_row);
                            else if (k == 0 && nf && c + 2 == gr.d_c.size())
                                ask(nf->down[r0 >= nf->d_split] + (size_t) (r0 + r + 2 - r1) * nf->d_row, nf->d_row);
                            tr->vec_dot((int) f.n_ff, &dout[k][r], 0, wd + (size_t) r * f.d_row, 0, hq[k], 0, 1);
                        }
                }
                for (int k = 0; k < nt; ++k) {
                    const int u = gr.units[c0 + k];
                    if (single) {
                        std::copy(dbuf[k], dbuf[k] + nr, ub.data() + (size_t) u * DRB);
                        continue;
                    }
                    if (g0 != 0)
                        continue; // summed in group order below
                    float* out = y + (size_t) unit_t[(size_t) u] * N + r0;
                    const float w = unit_w[(size_t) u];
                    for (int j = 0; j < nr; ++j)
                        out[j] += w * dbuf[k][j];
                }
            }
        }
        if (g0 != 0) // the sums in group order, as a group by group item adds them
            for (size_t gi = 0; gi < NG; ++gi) {
                const Group& gr = groups[gi];
                for (size_t k = 0; k < gr.units.size(); ++k) {
                    const int u = gr.units[k];
                    float* out = y + (size_t) unit_t[(size_t) u] * N + r0;
                    const float w = unit_w[(size_t) u];
                    const float* d = gbufs.data() + (gi * max_units + k) * RB;
                    for (int j = 0; j < nr; ++j)
                        out[j] += w * d[j];
                }
            }
        if (single)
            for (int t = 0; t < T; ++t) {
                float* out = y + (size_t) t * N + r0;
                for (int u : token_units[(size_t) t]) {
                    const float w = unit_w[(size_t) u];
                    const float* d = ub.data() + (size_t) u * DRB;
                    for (int j = 0; j < nr; ++j)
                        out[j] += w * d[j];
                }
            }
    };
    if (!flow) {
        run_timed(3, dn[0], dn[1], down_item);
        lap(3);
        return;
    }
    const int ngu[2] = {(int) items[0].size(), (int) items[1].size()};
    run_timed(4, ngu[0] + dn[0], ngu[1] + dn[1], [&](int node, int i) {
        if (i < ngu[node])
            gu_item(node, i);
        else
            down_item(node, i - ngu[node]);
    });
    lap(4);
}

} // namespace guild::core::cpu_moe
