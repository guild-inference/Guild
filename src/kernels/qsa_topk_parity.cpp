// src/kernels/qsa_topk_parity.cpp - the QSA top-k the engine dispatches (qsa_block_topk) against the original kernel
// (qsa_block_topk_ref): the selected cells must be identical (GPU, synthetic, no model).
//
// A case is a context under a capacity (the engine's --max-context; max_blocks = capacity / 4 + 2, and without an
// active-block count the dispatch follows it), with consecutive queries at the context's end so that the tail block
// holds 0 to 3 cells.  Three score sets per case: continuous, 64 levels (many ties at the threshold) and all equal
// (the rule's ties go to the lowest index).  The dispatch is called as a decode window calls it (no block count) and as
// the prompt path does (with the call's largest block count).  The reference and the dispatched kernel are also timed.
//
// --ties (also part of --selftest): the selection rule itself, on scores built to tie, as a decode window calls it.  A
// host reference sorts every cell by (key descending, index ascending), the documented rule, written independently
// of the kernels' threshold search.  The reference kernel and the dispatched one at each capacity that picks another
// kernel (register, wide), each launched again and again and on two streams at once, must all give its exact ids.
//
// --scores (also part of --selftest): the block scores the top-k reads.  A decode window's call (no block count: the
// multi-query kernel) against the prompt path's per-query warp kernel (a block count), bit for bit, for 1 to 8 queries
// at contexts whose tail block holds 0 to 3 cells.
//
// Usage: qsa_topk_parity --selftest
//        qsa_topk_parity --ties [MAX_CONTEXT=1048576]
//        qsa_topk_parity --scores
//        qsa_topk_parity CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] [COUNT=1]   (COUNT=0: no block count)
#include "guild/kernels/qsa.hpp"
#include "guild/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace k = guild::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

bool run_case(int64_t ctx, int64_t nq, int reps, int64_t capacity, bool counted) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    std::mt19937 rng((uint32_t) (ctx * 31 + capacity));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> scores((size_t) (nq * max_blocks)), tied(scores.size()), equal(scores.size(), 0.25f);
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        float* row = scores.data() + i * max_blocks;
        for (int64_t b = 0; b < max_blocks; ++b)
            row[b] = nd(rng);
        // the tail block as the scorers leave it: far above the others when it holds cells
        if (st[k::kStepNKv] % s.idx_block != 0)
            row[st[k::kStepNBid]] += 1e9f;
        for (int64_t b = 0; b < max_blocks; ++b)
            tied[(size_t) (i * max_blocks + b)] =
                row[b] > 1e8f ? row[b] : std::floor(row[b] * 8.0f) / 8.0f; // ~64 levels
    }
    int32_t *d_steps = nullptr, *ids_ref = nullptr, *ids_new = nullptr;
    float* d_scores = nullptr;
    ck(cudaMalloc(&d_steps, steps.size() * 4), "malloc");
    ck(cudaMalloc(&d_scores, scores.size() * 4), "malloc");
    ck(cudaMalloc(&ids_ref, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&ids_new, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "upload");
    const int64_t active = counted ? steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1 : -1;
    std::vector<int32_t> a((size_t) (nq * cap)), b(a.size());
    auto same = [&](const std::vector<float>& sc) -> int64_t {
        ck(cudaMemcpy(d_scores, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "upload");
        ck(cudaMemset(ids_ref, 0xff, a.size() * 4), "memset");
        ck(cudaMemset(ids_new, 0xff, a.size() * 4), "memset");
        k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr);
        k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active);
        ck(cudaDeviceSynchronize(), "sync");
        ck(cudaMemcpy(a.data(), ids_ref, a.size() * 4, cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(b.data(), ids_new, b.size() * 4, cudaMemcpyDeviceToHost), "down");
        int64_t n = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
            n += std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap);
        }
        return n;
    };
    const int64_t same_tied = same(tied), same_equal = same(equal), same_cont = same(scores); // d_scores: continuous
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto timed = [&](auto f) {
        ck(cudaEventRecord(e0), "record");
        for (int r = 0; r < reps; ++r)
            f();
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "time");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return ms / (float) reps;
    };
    const float t_ref =
        timed([&] { k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr); });
    const float t_new =
        timed([&] { k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active); });
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_steps);
    cudaFree(d_scores);
    cudaFree(ids_ref);
    cudaFree(ids_new);
    const bool ok = same_cont == nq && same_tied == nq && same_equal == nq;
    std::printf("%s context %lld under a capacity of %lld (%lld blocks), %lld queries %s: reference %.3f ms, "
                "dispatched %.3f ms (%.1fx); identical ids: continuous %lld/%lld, 64 levels %lld/%lld, equal scores "
                "%lld/%lld\n",
                ok ? "PASS" : "FAIL", (long long) ctx, (long long) capacity, (long long) max_blocks, (long long) nq,
                counted ? "with their block count" : "without a block count", t_ref, t_new, t_ref / t_new,
                (long long) same_cont, (long long) nq, (long long) same_tied, (long long) nq, (long long) same_equal,
                (long long) nq);
    return ok;
}

// ---- --ties ----
uint32_t host_key(float x) { // the kernels' order_key: -0 is +0, NaN is the lowest key
    const float v = x + 0.0f;
    if (!(v == v))
        return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// The documented rule for one query: cell j < n_kv scores row[min(j / r, n_bid)]; the `width` cells with the largest
// keys, ties to the lower index, written ascending.
void host_select(const float* row, int64_t n_kv, int64_t n_bid, int64_t width, int64_t r, int32_t* out) {
    if (n_kv <= width) {
        for (int64_t j = 0; j < n_kv; ++j)
            out[j] = (int32_t) j;
        return;
    }
    std::vector<uint64_t> v((size_t) n_kv);
    for (int64_t j = 0; j < n_kv; ++j)
        v[(size_t) j] =
            ((uint64_t) host_key(row[std::min(j / r, n_bid)]) << 32) | (uint64_t) (0xffffffffu - (uint32_t) j);
    std::nth_element(v.begin(), v.begin() + (width - 1), v.end(), std::greater<uint64_t>());
    std::vector<int32_t> sel((size_t) width);
    for (int64_t i = 0; i < width; ++i)
        sel[(size_t) i] = (int32_t) (0xffffffffu - (uint32_t) (v[(size_t) i] & 0xffffffffu));
    std::sort(sel.begin(), sel.end());
    std::copy(sel.begin(), sel.end(), out);
}

// a block score of pattern p; most patterns tie by construction
float tie_score(int p, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::normal_distribution<float> nd(0.f, 1.f);
    const float x = u(rng);
    switch (p) {
    case 0: return 0.25f;                             // all equal
    case 1: return x < 0.006f ? 0.5f + u(rng) : 0.0f; // zeros, ~0.6% positive: the threshold among zeros
    case 2: return x < 0.006f ? 0.5f + u(rng) : (u(rng) < 0.5f ? -0.0f : 0.0f);  // the same with -0 and +0 mixed
    case 3: return x < 0.01f ? 1.5f : x < 0.03f ? 1.0f : x < 0.5f ? 0.5f : 0.0f; // four levels
    case 4: { // eight values that differ in the last byte only
        uint32_t b = 0x3f800000u + (uint32_t) (rng() % 8u);
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    }
    case 5: // NaN, +-inf and three levels
        if (x < 0.1f)
            return std::nanf("");
        if (x < 0.1015f)
            return INFINITY;
        if (x < 0.11f)
            return -INFINITY;
        return x < 0.12f ? 2.0f : x < 0.5f ? 1.0f : 0.0f;
    case 6: { // the scorer's shape: sum over 4 heads of relu
        float sc = 0.f;
        for (int h = 0; h < 4; ++h) {
            const float d = nd(rng) - 1.0f;
            sc += d > 0.f ? d : 0.f;
        }
        return sc;
    }
    default: return nd(rng); // continuous
    }
}
constexpr int kTiePatterns = 8;

bool run_ties(int64_t ctx, int64_t nq, int reps) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t r = s.idx_block, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    // the capacities the engine could run this context under; each picks its kernel by its block count
    std::vector<int64_t> caps{ctx};
    for (int64_t c : {(int64_t) 135160, (int64_t) 262144, (int64_t) 1048576})
        if (c > ctx)
            caps.push_back(c);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / r);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const int64_t nb_max = ctx / r + 1;
    int32_t* d_steps = nullptr;
    ck(cudaMalloc(&d_steps, steps.size() * 4), "malloc");
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "upload");
    cudaStream_t sa, sb;
    ck(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking), "stream");
    ck(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking), "stream");
    int32_t *d_a = nullptr, *d_b = nullptr;
    ck(cudaMalloc(&d_a, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&d_b, (size_t) (nq * cap) * 4), "malloc");
    bool ok = true;
    int64_t checked = 0;
    for (int p = 0; p < kTiePatterns; ++p) {
        std::mt19937 rng((uint32_t) (ctx * 131 + p));
        std::vector<float> base((size_t) (nq * nb_max));
        for (auto& x : base)
            x = tie_score(p, rng);
        std::vector<int32_t> want((size_t) (nq * cap), -1);
        for (int64_t i = 0; i < nq; ++i) {
            const int32_t* st = steps.data() + i * k::kStepCount;
            float* row = base.data() + i * nb_max;
            if (st[k::kStepNKv] % r != 0) // the tail block as the scorers leave it
                row[st[k::kStepNBid]] += 1e9f;
            host_select(row, st[k::kStepNKv], st[k::kStepNBid], st[k::kStepWidth], r, want.data() + i * cap);
        }
        for (const int64_t capacity : caps) {
            const int64_t max_blocks = capacity / r + 2;
            // blocks past a query's last (and a weightless tail) hold values no kernel may select
            std::vector<float> sc((size_t) (nq * max_blocks), 1e30f);
            for (int64_t i = 0; i < nq; ++i) {
                const int32_t* st = steps.data() + i * k::kStepCount;
                const int64_t nb = st[k::kStepNBid] + 1;
                std::copy(base.begin() + i * nb_max, base.begin() + i * nb_max + nb, sc.begin() + i * max_blocks);
                if (st[k::kStepNKv] % r == 0)
                    sc[(size_t) (i * max_blocks + st[k::kStepNBid])] = 1e30f;
            }
            float* d_sc = nullptr;
            ck(cudaMalloc(&d_sc, sc.size() * 4), "malloc");
            ck(cudaMemcpy(d_sc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "upload");
            ck(cudaDeviceSynchronize(), "upload"); // the streams below do not wait for the legacy stream's copy
            std::vector<int32_t> got((size_t) (nq * cap));
            auto check = [&](int32_t* d, const char* what) {
                ck(cudaMemcpy(got.data(), d, got.size() * 4, cudaMemcpyDeviceToHost), "down");
                for (int64_t i = 0; i < nq; ++i) {
                    const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
                    if (!std::equal(want.begin() + i * cap, want.begin() + i * cap + w, got.begin() + i * cap)) {
                        int64_t j = 0;
                        while (want[(size_t) (i * cap + j)] == got[(size_t) (i * cap + j)])
                            ++j;
                        std::printf("FAIL ties: context %lld capacity %lld pattern %d query %lld %s: id %lld is %d, "
                                    "the rule gives %d\n",
                                    (long long) ctx, (long long) capacity, p, (long long) i, what, (long long) j,
                                    got[(size_t) (i * cap + j)], want[(size_t) (i * cap + j)]);
                        ok = false;
                        return;
                    }
                }
                ++checked;
            };
            ck(cudaMemsetAsync(d_a, 0xff, (size_t) (nq * cap) * 4, sa), "memset");
            k::qsa_block_topk_ref(d_sc, d_steps, nq, max_blocks, cap, s, d_a, sa);
            ck(cudaStreamSynchronize(sa), "sync");
            check(d_a, "reference kernel");
            for (int rep = 0; rep < reps; ++rep) { // the decode window's call, again and again
                ck(cudaMemsetAsync(d_a, 0xff, (size_t) (nq * cap) * 4, sa), "memset");
                k::qsa_block_topk(d_sc, d_steps, nq, max_blocks, cap, s, d_a, sa, 0);
                ck(cudaStreamSynchronize(sa), "sync");
                check(d_a, "dispatched kernel");
            }
            for (int rep = 0; rep < reps; ++rep) { // two at once on two streams
                ck(cudaMemsetAsync(d_a, 0xff, (size_t) (nq * cap) * 4, sa), "memset");
                ck(cudaMemsetAsync(d_b, 0xff, (size_t) (nq * cap) * 4, sb), "memset");
                k::qsa_block_topk(d_sc, d_steps, nq, max_blocks, cap, s, d_a, sa, 0);
                k::qsa_block_topk(d_sc, d_steps, nq, max_blocks, cap, s, d_b, sb, 0);
                ck(cudaDeviceSynchronize(), "sync");
                check(d_a, "dispatched kernel, stream a");
                check(d_b, "dispatched kernel, stream b");
            }
            cudaFree(d_sc);
        }
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_steps);
    cudaStreamDestroy(sa);
    cudaStreamDestroy(sb);
    std::printf("%s ties: context %lld (%lld blocks), %lld queries, %d patterns x %zu capacities: %lld launches match "
                "the host rule\n",
                ok ? "PASS" : "FAIL", (long long) ctx, (long long) (ctx / r), (long long) nq, kTiePatterns, caps.size(),
                (long long) checked);
    return ok;
}

// ---- --scores ----
bool run_scores(int64_t ctx, int64_t nq, int64_t capacity) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2;
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const int64_t n_bid_max = steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)];
    std::mt19937 rng((uint32_t) (ctx * 7 + nq));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> pooled((size_t) ((n_bid_max + 1) * s.idx_dim)), dead((size_t) s.idx_dim),
        q((size_t) (nq * s.idx_n_head * s.idx_dim));
    for (auto& x : pooled)
        x = nd(rng);
    for (auto& x : dead)
        x = nd(rng);
    for (auto& x : q)
        x = 0.3f * nd(rng);
    float *d_pooled = nullptr, *d_dead = nullptr, *d_q = nullptr, *d_a = nullptr, *d_b = nullptr;
    int32_t* d_steps = nullptr;
    ck(cudaMalloc(&d_pooled, pooled.size() * 4), "malloc");
    ck(cudaMalloc(&d_dead, dead.size() * 4), "malloc");
    ck(cudaMalloc(&d_q, q.size() * 4), "malloc");
    ck(cudaMalloc(&d_steps, steps.size() * 4), "malloc");
    ck(cudaMalloc(&d_a, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMalloc(&d_b, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMemcpy(d_pooled, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(d_dead, dead.data(), dead.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemset(d_a, 0xff, (size_t) (nq * max_blocks) * 4), "memset");
    ck(cudaMemset(d_b, 0xff, (size_t) (nq * max_blocks) * 4), "memset");
    k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_a, nullptr, 0);             // decode
    k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_b, nullptr, n_bid_max + 1); // prompt
    ck(cudaDeviceSynchronize(), "scores");
    std::vector<float> a((size_t) (nq * max_blocks)), b(a.size());
    ck(cudaMemcpy(a.data(), d_a, a.size() * 4, cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(b.data(), d_b, b.size() * 4, cudaMemcpyDeviceToHost), "download");
    bool ok = true;
    for (int64_t i = 0; i < nq && ok; ++i) {
        const int64_t nb = steps[(size_t) (i * k::kStepCount + k::kStepNBid)] + 1;
        for (int64_t j = 0; j < nb; ++j) {
            const size_t at = (size_t) (i * max_blocks + j);
            if (std::memcmp(&a[at], &b[at], 4) != 0) {
                std::printf("FAIL scores: context %lld, %lld queries: query %lld block %lld is %.9g, the warp kernel "
                            "gives %.9g\n",
                            (long long) ctx, (long long) nq, (long long) i, (long long) j, a[at], b[at]);
                ok = false;
                break;
            }
        }
    }
    for (void* x : {(void*) d_pooled, (void*) d_dead, (void*) d_q, (void*) d_steps, (void*) d_a, (void*) d_b})
        cudaFree(x);
    return ok;
}

bool scores_suite() {
    bool ok = true;
    int64_t n = 0;
    for (const int64_t ctx :
         {(int64_t) 2100, (int64_t) 9001, (int64_t) 32770, (int64_t) 65539, (int64_t) 131072, (int64_t) 262143})
        for (int64_t nq = 1; nq <= 8; ++nq, ++n)
            ok = run_scores(ctx, nq, 262144) && ok;
    std::printf("%s scores: %lld decode-shaped calls (1-8 queries, 2,100 to 262,143 cells) bitwise equal to the warp "
                "kernel\n",
                ok ? "PASS" : "FAIL", (long long) n);
    return ok;
}

bool ties_suite(int64_t max_ctx) {
    bool ok = true;
    // 16K to 256K blocks; contexts whose last queries leave 0 to 3 cells in the tail
    for (const int64_t ctx :
         {(int64_t) 65539, (int64_t) 131071, (int64_t) 135160, (int64_t) 262144, (int64_t) 524289, (int64_t) 1048576}) {
        if (ctx > max_ctx)
            continue;
        ok = run_ties(ctx, 4, 4) && ok;
        ok = run_ties(ctx, 1, 2) && ok;
    }
    return ok;
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        // {context, capacity}: both sides of the register kernel's reach (33,792 blocks = 135,168 cells), short
        // contexts under a long capacity (fewer blocks than the kernel's threads; a prompt batch below the dispatch's
        // 4,608 blocks), and the long contexts themselves
        const int64_t cases[][2] = {{9000, 9000},     {135160, 135160}, {135176, 135176},
                                    {2100, 262144},   {9000, 262144},   {32768, 262144},
                                    {262144, 262144}, {300001, 524288}, {524288, 524288}};
        bool ok = ties_suite(1048576);
        ok = scores_suite() && ok;
        for (const auto& c : cases) {
            ok = run_case(c[0], 8, 3, c[1], false) && ok; // a decode window's call
            ok = run_case(c[0], 64, 3, c[1], true) && ok; // the prompt path's
        }
        std::printf("%s\n", ok ? "qsa_topk_parity: PASS" : "qsa_topk_parity: FAIL");
        return ok ? 0 : 1;
    }
    if (argc > 1 && std::strcmp(argv[1], "--scores") == 0) {
        const bool ok = scores_suite();
        return ok ? 0 : 1;
    }
    if (argc > 1 && std::strcmp(argv[1], "--ties") == 0) {
        const bool ok = ties_suite(argc > 2 ? std::atoll(argv[2]) : 1048576);
        std::printf("%s\n", ok ? "qsa_topk_parity --ties: PASS" : "qsa_topk_parity --ties: FAIL");
        return ok ? 0 : 1;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: qsa_topk_parity --selftest | --ties [MAX_CONTEXT] | --scores | CONTEXT "
                             "[QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] "
                             "[COUNT=1]\n");
        return 2;
    }
    const int64_t ctx = std::atoll(argv[1]);
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : ctx;
    if (ctx < nq || nq <= 0 || reps <= 0 || capacity < ctx) {
        std::fprintf(stderr, "qsa_topk_parity: CONTEXT >= QUERIES > 0, REPS > 0, CAPACITY >= CONTEXT\n");
        return 2;
    }
    return run_case(ctx, nq, reps, capacity, argc <= 5 || std::atoi(argv[5]) != 0) ? 0 : 1;
}
