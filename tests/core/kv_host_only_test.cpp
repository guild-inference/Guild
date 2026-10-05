// Bounded shared staging at the full 262K context, including graph replay and different layers sharing storage.
#include "strata/core/layer.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace c = strata::core;
namespace k = strata::kernels;
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
};
struct State {
    c::QsaState s;
    Device<uint8_t> arena;
    State(const c::ModelGeometry& g, int64_t ctx, const c::QsaState* share = nullptr)
        : arena(c::qsa_state_bytes(g, ctx, share == nullptr)) {
        require(c::qsa_state_init(g, ctx, arena.p, s, share) != 0, "state init failed");
        c::qsa_state_zero(s, g, nullptr);
    }
    ~State() { c::qsa_state_release_host(s); }
};

void run(int fmt) {
    const c::ModelGeometry g{};
    const auto shape = k::qsa_real_shapes();
    constexpr int64_t ctx = 262144;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, shape);
    const int nq = k::kVerifyMaxT;
    const size_t row = shape.n_head * shape.head_dim;
    c::qsa_set_kv_int8(fmt == k::kKvInt8);
    c::qsa_set_kv_q4(fmt == k::kKvQ4);
    c::qsa_set_kv_resident(0);
    c::qsa_set_kv_host_only(false);
    State ref(g, ctx);
    require(ref.s.kv_mode == 0, "--kv-resident 0 must still mean full VRAM");
    c::qsa_set_kv_host_only(true);
    State a(g, ctx), b(g, ctx, &a.s);
    require(a.s.kv_mode == 3 && b.s.kv_mode == 3, "host-only mode was not selected");
    require(a.s.n_slots * shape.page_size == c::qsa_kv_staging_cells(g), "wrong staging bound");
    require(a.s.k_pool == b.s.k_pool && a.s.k_q == b.s.k_q && a.s.k_q4 == b.s.k_q4,
            "layers did not share the staging pool");
    require(a.s.host_allocation != b.s.host_allocation, "authoritative KV must be private to each layer");
    const uint64_t staged = a.s.n_slots * k::kv_block_bytes(shape, fmt);
    require(staged <= 33 * 1048576ull, "staging grew with context");
    require(c::qsa_state_bytes(g, ctx, true) - c::qsa_state_bytes(g, ctx, false) < staged + 65 * 1048576ull,
            "each layer allocated an extra context-sized KV pool");

    Device<float> kc(shape.n_head_kv * shape.head_dim), vc(shape.n_head_kv * shape.head_dim);
    Device<int32_t> step(k::kStepCount), steps(nq * k::kStepCount), ids(nq * cap);
    Device<float> q(nq * row), scratch(nq * k::qsa_decode_attn_scratch_floats(cap, shape));
    Device<float> out(nq * row), expected(nq * row);
    std::mt19937 rng(91);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    std::vector<float> keys(shape.n_head_kv * shape.head_dim), values(keys.size()), query(nq * row);
    std::vector<int32_t> selections(nq * cap), records(nq * k::kStepCount);
    std::set<int32_t> cells;
    for (int t = 0; t < nq; ++t) {
        std::set<int32_t> blocks;
        while (blocks.size() < size_t(cap / 4)) blocks.insert(int32_t(rng() % (ctx / 4 - 1)));
        int64_t at = t * cap;
        for (int32_t block : blocks) for (int j = 0; j < 4; ++j) selections[at++] = 4 * block + j;
        for (int j = 0; j < cap % 4; ++j) selections[at++] = int32_t(ctx - 4 + j);
        cells.insert(selections.begin() + t * cap, selections.begin() + (t + 1) * cap);
        records[t * k::kStepCount] = int32_t(ctx - 1);
        records[t * k::kStepCount + 1] = int32_t(ctx);
        records[t * k::kStepCount + 2] = int32_t(ctx / 4);
        records[t * k::kStepCount + 3] = int32_t(cap);
    }
    for (float& x : query) x = random(rng);
    check(cudaMemcpy(q.p, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(ids.p, selections.data(), selections.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    check(cudaMemcpy(steps.p, records.data(), records.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    auto append = [&](c::QsaState& s, const k::KvHostPools* host) {
        if (fmt == k::kKvF16) k::kv_append_step(s.k_pool, s.v_pool, s.page_table, step.p, kc.p, vc.p, shape, nullptr, host);
        else if (fmt == k::kKvInt8) k::kv_append_q8_step(s.k_q, s.v_q, s.k_scale, s.v_scale, s.page_table, step.p,
                                                       kc.p, vc.p, shape, nullptr, host);
        else k::kv_append_q4_step(s.k_q4, s.v_q4, s.page_table, step.p, kc.p, vc.p, shape, nullptr, host);
    };
    // Fill only the selected cells, at scattered addresses across the full context. Distinct layers detect stale
    // residency maps after the other layer overwrites the shared buffer; varying cells detect incorrect mapping.
    for (int layer = 0; layer < 2; ++layer) {
        auto& state = layer == 0 ? a.s : b.s;
        for (int32_t cell : cells) {
            for (float& x : keys) x = random(rng);
            for (float& x : values) x = random(rng) + 2.f * layer;
            const int32_t st[k::kStepCount] = {cell, cell + 1, (cell + 1) / 4, 0};
            check(cudaMemcpy(step.p, st, sizeof(st), cudaMemcpyHostToDevice));
            check(cudaMemcpy(kc.p, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice));
            check(cudaMemcpy(vc.p, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
            append(state, &state.host);
        }
    }
    auto copy_reference = [&](const c::QsaState& state) {
        const auto host = state.host;
        const size_t rows = ctx * shape.n_head_kv;
        if (fmt == k::kKvF16) {
            check(cudaMemcpy(ref.s.k_pool, host.k_pool, rows * shape.head_dim * 2, cudaMemcpyDefault));
            check(cudaMemcpy(ref.s.v_pool, host.v_pool, rows * shape.head_dim * 2, cudaMemcpyDefault));
        } else if (fmt == k::kKvInt8) {
            check(cudaMemcpy(ref.s.k_q, host.k_q, rows * shape.head_dim, cudaMemcpyDefault));
            check(cudaMemcpy(ref.s.v_q, host.v_q, rows * shape.head_dim, cudaMemcpyDefault));
            check(cudaMemcpy(ref.s.k_scale, host.k_scale, rows * (shape.head_dim / k::KV_Q8_GROUP) * 2, cudaMemcpyDefault));
            check(cudaMemcpy(ref.s.v_scale, host.v_scale, rows * (shape.head_dim / k::KV_Q8_GROUP) * 2, cudaMemcpyDefault));
        } else {
            const size_t bytes = rows * k::kv_q4_bytes_per_head(int(shape.head_dim));
            check(cudaMemcpy(ref.s.k_q4, host.k_q4, bytes, cudaMemcpyDefault));
            check(cudaMemcpy(ref.s.v_q4, host.v_q4, bytes, cudaMemcpyDefault));
        }
    };
    std::vector<float> got(nq * row), want(nq * row);
    auto compare = [&](int count) {
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(got.data(), out.p, count * row * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(want.data(), expected.p, count * row * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count * row; ++i) require(std::isfinite(got[i]), "nonfinite attention");
        require(std::memcmp(got.data(), want.data(), count * row * sizeof(float)) == 0, "attention differs bitwise");
    };
    for (int pass = 0; pass < 3; ++pass) {
        auto& state = pass == 1 ? b.s : a.s;
        copy_reference(state);
        for (int count = 1; count <= nq; ++count) {
            k::qsa_decode_attn_batch(q.p, c::qsa_attn_pools(ref.s), ids.p, steps.p, cap, shape, scratch.p, expected.p, count, nullptr);
            c::qsa_kv_resolve(state, g, ids.p, steps.p, count, cap, nullptr);
            k::qsa_decode_attn_batch(q.p, c::qsa_attn_pools(state), ids.p, steps.p, cap, shape, scratch.p, out.p, count, nullptr);
            compare(count);
        }
    }
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    check(cudaStreamCreate(&stream));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    c::qsa_kv_resolve(a.s, g, ids.p, steps.p, nq, cap, stream);
    k::qsa_decode_attn_batch(q.p, c::qsa_attn_pools(a.s), ids.p, steps.p, cap, shape, scratch.p, out.p, nq, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        c::qsa_kv_resolve(b.s, g, ids.p, steps.p, nq, cap, nullptr); // poison a's previously valid slots
        check(cudaDeviceSynchronize());
        check(cudaGraphLaunch(exec, stream));
        compare(nq);
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    const auto counters = k::kv_stream_counters(a.s.map);
    require(counters.calls == 18 && counters.misses > 0 && counters.misses <= counters.lookups && counters.overflow == 0,
            "transient staging counters were reset or overflowed");
    std::printf("kv_host_only_test: format %d, 262144 cells, %.3f MiB shared staging: bitwise parity (1..8 rows, layers, graphs)\n",
                fmt, double(staged) / 1048576.0);
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        const uint64_t pinned_before = c::qsa_kv_host_bytes();
        run(k::kKvF16); run(k::kKvInt8); run(k::kKvQ4);
        require(c::qsa_kv_host_bytes() == pinned_before, "host KV allocations were not released");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kv_host_only_test: FAIL: %s\n", e.what()); return 1;
    }
    std::puts("kv_host_only_test: PASS");
    return 0;
}
