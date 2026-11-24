// tests/core/pool_rows_slots_test.cpp - the tables pool_rows512 hands out (include/guild/core/pool_rows.hpp) stay
// independent of each other.  Fake base tables that write their own marker values, wrapped one after another: each
// wrapped table reaches its own base for a format the wrappers do not reroute (Q8_0 gate/up and down) and not for
// one they do (IQ4_NL down; Q4_K gate/up with GUILD_KQ_ROWS512), also after later ones were wrapped; bases wrapped
// from several threads at once take one slot each; a base with the same functions as one already wrapped reuses its
// slot; with every slot taken a new base comes back unchanged, while the slots taken keep working.  Then pool_rows512
// on real tables: the library's table (the registry's default CPU expert kernel, the one kq_rows512_test compares
// the rows with, and the one ExpertPool binds) is wrapped, every copy of it into one slot; with interleaved gate/up
// rows, and for a GEXEC table (fused or interleaved), a table like an external package's (the library's with a
// gate_up or quant_act of its own) or a table already wrapped, it comes back unchanged.  No model needed.  Exit 77
// without AVX-512BW, or with GUILD_POOL_ROWS512=0 or GUILD_IQ_GGML_ORDER=0 (no wrapping to test).
#include "guild/core/glm5_nl4.hpp"
#include "guild/core/iq4nl_rows512.hpp"
#include "guild/core/kq_rows512.hpp"
#include "guild/core/pool_rows.hpp"
#include "guild/kernels/cpu/native_expert.hpp"

#if GUILD_BUNDLED_CPU_ARITHMETIC && defined(GUILD_NATIVE_EXPERTS)
#include "guild/kernels/cpu/gexec_expert.hpp"
#include "guild/kernels/cpu/pool.hpp"
#include "guild/kernels/registry.hpp"
#endif

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Table = guild_cpu_expert_ext_v1_t;
using Fmt = guild_cpu_expert_format_v1_t;
namespace detail = guild::core::detail;
constexpr int kSlots = detail::kPoolRowsSlots;

int g_fail = 0;
#define EXPECT(c)                                                                                                      \
    do {                                                                                                               \
        if (!(c) && g_fail++ < 20)                                                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                                   \
    } while (0)

float gu_marker(int k) {
    return 1000.f + (float) k;
}
float down_marker(int k) {
    return 2000.f + (float) k;
}

// fake base k: every row it is asked for gets its marker
template <int K> void fake_quant(const Fmt*, const float*, void*) {
}
template <int K>
void fake_gate_up(const Fmt*, const uint8_t*, const void* const*, int32_t nt, float* const* ff, int32_t r0,
                  int32_t r1) {
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r)
            ff[t][r] = gu_marker(K);
}
template <int K>
void fake_down(const Fmt*, const uint8_t*, const void* const*, int32_t nt, float* const* out, int32_t r0, int32_t r1) {
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r)
            out[t][r] = down_marker(K);
}
template <int K> Table fake_table() {
    Table t{};
    t.struct_size = sizeof(Table);
    t.tag = GUILD_CPU_EXPERT_TAG_V1;
    t.version = 1;
    t.quant_act = fake_quant<K>;
    t.quant_h = fake_quant<K>;
    t.gate_up = fake_gate_up<K>;
    t.down = fake_down<K>;
    return t;
}
template <int... K> std::vector<Table> fake_tables(std::integer_sequence<int, K...>) {
    return {fake_table<K>()...};
}

bool same(const Table& a, const Table& b) {
    return a.struct_size == b.struct_size && a.tag == b.tag && a.version == b.version && a.quant_act == b.quant_act &&
           a.quant_h == b.quant_h && a.gate_up == b.gate_up && a.down == b.down;
}
// `w` wraps a base whose other fields are `base`'s
bool wraps(const Table& w, const Table& base) {
    return w.gate_up != base.gate_up && w.down != base.down && w.struct_size == base.struct_size && w.tag == base.tag &&
           w.version == base.version && w.quant_act == base.quant_act && w.quant_h == base.quant_h;
}

// One expert of format (gu, d) in zeros, zero activations (every dot 0), and output rows to call a table with.
struct Expert {
    Fmt f{};
    std::vector<uint8_t> blob, act, h;
    Expert(int gu, int d, int n_embd, int n_ff) {
        std::string err;
        if (!guild::kernels::cpu::native_fmt(gu, d, n_embd, n_ff, f, err)) {
            std::printf("FAIL native_fmt(%d, %d): %s\n", gu, d, err.c_str());
            std::exit(1);
        }
        blob.assign(f.bytes, 0);
        act.assign(f.act_bytes * 3, 0);
        h.assign(f.h_bytes * 3, 0);
    }
    // rows [r0, r1) of 3 tokens: gate_up (down false) or down; every row asked for must be `want` (or, want_not:
    // anything but it), every other row untouched
    bool run(const Table& t, bool down, int r0, int r1, float v, bool want_not = false) const {
        const int rows = down ? (int) f.n_embd : (int) f.n_ff;
        std::vector<float> o((size_t) 3 * rows, -7.f);
        float* op[3];
        const void* xp[3];
        for (int i = 0; i < 3; ++i) {
            op[i] = o.data() + (size_t) i * rows;
            xp[i] = down ? (const void*) (h.data() + (size_t) i * f.h_bytes)
                         : (const void*) (act.data() + (size_t) i * f.act_bytes);
        }
        if (down)
            t.down(&f, blob.data(), xp, 3, op, r0, r1);
        else
            t.gate_up(&f, blob.data(), xp, 3, op, r0, r1);
        for (int i = 0; i < 3; ++i)
            for (int r = 0; r < rows; ++r) {
                const float y = op[i][r];
                const bool in = r >= r0 && r < r1;
                if (in ? (want_not ? y == v : y != v) : y != -7.f)
                    return false;
            }
        return true;
    }
};

bool env_off(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && std::atoi(e) == 0;
}

} // namespace

int main() {
    using namespace guild::core;
    if (!glm5cpu::nl4_rows512_ok() || !cpu_moe::iq4nl_rows512_ok()) {
        std::printf("skip: no AVX-512BW\n");
        return 77;
    }
    if (env_off("GUILD_POOL_ROWS512") || env_off("GUILD_IQ_GGML_ORDER")) {
        std::printf("skip: GUILD_POOL_ROWS512=0 or GUILD_IQ_GGML_ORDER=0 (the pool's tables are not wrapped)\n");
        return 77;
    }
    const std::vector<Table> F = fake_tables(std::make_integer_sequence<int, kSlots + 2>{});
    // Q8_0 rows go to the base; IQ4_NL down rows (an even number of blocks) and, unless GUILD_KQ_ROWS512=0, Q4_K
    // gate/up rows go to the AVX-512BW rows
    const Expert q8(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 128), nl(GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, 256, 128);
    const Expert q4k(GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, 256, 128);
    const bool q4k_rerouted =
        !env_off("GUILD_KQ_ROWS512") && cpu_moe::kq_rows512_ok() && cpu_moe::kq_rows512_supported(GGML_TYPE_Q4_K, 256);
    int used = 0; // slots taken by this process so far

    // 1. The library's table and the others, through pool_rows512
#if GUILD_BUNDLED_CPU_ARITHMETIC && defined(GUILD_NATIVE_EXPERTS)
    const auto registry_table = [](const char* id, Table& t) {
        const auto b = guild::kernels::KernelRegistry::instance().find_by_id(id);
        if (!b || !b->next_ext)
            return false;
        std::memcpy(&t, b->next_ext, sizeof(t)); // as model.cpp binds a selected kernel's table
        return true;
    };
    Table lib{};
    EXPECT(registry_table("cpu.expert_pool.split_multi_native", lib));
    const Table w_lib = pool_rows512(lib, false);
    ++used;
    EXPECT(wraps(w_lib, lib));
    EXPECT(same(pool_rows512(lib, false), w_lib)); // a second model on the same table: the same slot
    {
        guild::kernels::cpu::ExpertPool pool(1, false, true); // binds the library's table at construction
        const Table p = pool.native_table();
        EXPECT(p.gate_up == lib.gate_up && p.down == lib.down);
        const Table w = pool_rows512(p, false);
        EXPECT(w.gate_up == w_lib.gate_up && w.down == w_lib.down);
    }
    EXPECT(same(pool_rows512(lib, true), lib)); // interleaved gate/up rows: unchanged
    // a table already wrapped is not the library's
    EXPECT(same(pool_rows512(w_lib, false), w_lib));
    EXPECT(same(detail::pool_rows512_slot(w_lib), w_lib));
    // the formats it does not reroute: what the library computes (zeros here), and the rerouted ones too
    EXPECT(q8.run(w_lib, false, 0, 128, 0.f) && q8.run(w_lib, true, 3, 255, 0.f) && nl.run(w_lib, true, 5, 6, 0.f));
    // the GEXEC tables (gexec_expert.cpp), fused and interleaved, as the registry holds them and as built
    using guild::kernels::cpu::kGExecFusedAvx2ImplId, guild::kernels::cpu::kGExecFusedAvx512ImplId;
    using guild::kernels::cpu::kGExecInterleavedAvx2ImplId, guild::kernels::cpu::kGExecInterleavedAvx512ImplId;
    const guild_cpu_backend_ext_v2_t* gexec[4] = {guild::kernels::cpu::guild_cpu_gexec_fused_avx2_table_v2(),
                                                  guild::kernels::cpu::guild_cpu_gexec_fused_avx512_table_v2(),
                                                  guild::kernels::cpu::guild_cpu_gexec_interleaved_avx2_table_v2(),
                                                  guild::kernels::cpu::guild_cpu_gexec_interleaved_avx512_table_v2()};
    const char* gexec_id[4] = {kGExecFusedAvx2ImplId, kGExecFusedAvx512ImplId, kGExecInterleavedAvx2ImplId,
                               kGExecInterleavedAvx512ImplId};
    for (int i = 0; i < 4; ++i) {
        EXPECT(same(pool_rows512(gexec[i]->native, false), gexec[i]->native));
        EXPECT(same(pool_rows512(gexec[i]->native, true), gexec[i]->native));
        Table g{};
        if (registry_table(gexec_id[i], g)) {
            EXPECT(same(pool_rows512(g, false), g));
        } else {
            std::printf("FAIL: the registry has no %s\n", gexec_id[i]);
            ++g_fail;
        }
    }
    // an external package's table: the library's arithmetic behind functions of its own (kcpu's plugin.cpp counts
    // its gate_up calls), or a quantizer of its own
    Table ext = lib;
    ext.gate_up = fake_gate_up<0>;
    EXPECT(same(pool_rows512(ext, false), ext));
    ext = lib;
    ext.down = fake_down<0>;
    EXPECT(same(pool_rows512(ext, false), ext));
    ext = lib;
    ext.quant_act = fake_quant<0>;
    EXPECT(same(pool_rows512(ext, false), ext));
    ext = lib;
    ext.quant_h = fake_quant<0>;
    EXPECT(same(pool_rows512(ext, false), ext));
#endif
    // a fake table is not the library's either
    EXPECT(same(pool_rows512(F[0], false), F[0]));

    // 2. Fakes wrapped one after another, each through its own base
    const Table w0 = detail::pool_rows512_slot(F[0]);
    ++used;
    EXPECT(wraps(w0, F[0]));
    EXPECT(q8.run(w0, false, 0, 128, gu_marker(0)) && q8.run(w0, true, 3, 255, down_marker(0)));
    const Table w1 = detail::pool_rows512_slot(F[1]);
    ++used;
    EXPECT(wraps(w1, F[1]) && w1.gate_up != w0.gate_up && w1.down != w0.down);
    EXPECT(q8.run(w1, false, 0, 128, gu_marker(1)) && q8.run(w1, true, 3, 255, down_marker(1)));
    // w0 after w1 was wrapped: still its own base (one process-wide base sent it to F[1])
    EXPECT(q8.run(w0, false, 5, 6, gu_marker(0)) && q8.run(w0, true, 0, 256, down_marker(0)));
    // the rerouted formats do not reach the base
    EXPECT(nl.run(w0, true, 0, 256, down_marker(0), true) && nl.run(w1, true, 3, 255, down_marker(1), true));
    EXPECT(nl.run(w0, true, 0, 256, 0.f) && nl.run(w1, true, 3, 255, 0.f));
    if (q4k_rerouted)
        EXPECT(q4k.run(w0, false, 0, 128, 0.f) && q4k.run(w1, false, 3, 127, 0.f));
    else
        EXPECT(q4k.run(w0, false, 0, 128, gu_marker(0)));
    // the same base again: the same slot
    EXPECT(same(detail::pool_rows512_slot(F[0]), w0) && same(detail::pool_rows512_slot(F[1]), w1));
    Table f0_copy = F[0];
    f0_copy.quant_act = fake_quant<5>; // the slot is the gate_up and down; the rest is the base's own
    const Table w0c = detail::pool_rows512_slot(f0_copy);
    EXPECT(w0c.gate_up == w0.gate_up && w0c.down == w0.down && w0c.quant_act == fake_quant<5>);

    // 3. Two new bases from eight threads at once: one slot each
    Table w2{}, w3{};
    {
        constexpr int kThreads = 8, kIters = 200;
        std::vector<std::vector<Table>> got(kThreads);
        std::vector<std::thread> th;
        for (int i = 0; i < kThreads; ++i)
            th.emplace_back([&, i] {
                for (int k = 0; k < kIters; ++k)
                    got[(size_t) i].push_back(detail::pool_rows512_slot(F[(size_t) (2 + (i + k) % 2)]));
            });
        for (auto& t : th)
            t.join();
        w2 = detail::pool_rows512_slot(F[2]);
        w3 = detail::pool_rows512_slot(F[3]);
        used += 2;
        EXPECT(wraps(w2, F[2]) && wraps(w3, F[3]) && w2.gate_up != w3.gate_up);
        for (int i = 0; i < kThreads; ++i)
            for (int k = 0; k < kIters; ++k)
                EXPECT(same(got[(size_t) i][(size_t) k], (i + k) % 2 ? w3 : w2));
        EXPECT(q8.run(w2, false, 0, 128, gu_marker(2)) && q8.run(w3, true, 0, 256, down_marker(3)));
    }

    // 4. Every slot taken: a new base comes back unchanged, the slots taken keep their bases
    int wrapped = 0, unchanged = 0;
    for (int k = 4; k < kSlots + 2; ++k) {
        const Table w = detail::pool_rows512_slot(F[(size_t) k]);
        if (same(w, F[(size_t) k])) {
            ++unchanged;
            EXPECT(q8.run(w, false, 0, 128, gu_marker(k)));
        } else {
            EXPECT(unchanged == 0 && wraps(w, F[(size_t) k]));
            EXPECT(q8.run(w, false, 0, 128, gu_marker(k)) && q8.run(w, true, 0, 256, down_marker(k)));
            ++wrapped;
        }
    }
    EXPECT(wrapped == kSlots - used && unchanged == kSlots + 2 - 4 - wrapped);
    EXPECT(same(detail::pool_rows512_slot(F[0]), w0) && same(detail::pool_rows512_slot(F[3]), w3));
    EXPECT(q8.run(w0, false, 0, 128, gu_marker(0)) && q8.run(w1, true, 0, 256, down_marker(1)));
    EXPECT(q8.run(w2, true, 3, 255, down_marker(2)) && q8.run(w3, false, 5, 6, gu_marker(3)));
#if GUILD_BUNDLED_CPU_ARITHMETIC && defined(GUILD_NATIVE_EXPERTS)
    EXPECT(same(pool_rows512(lib, false), w_lib)); // a model on the library's table still gets its slot
#endif

    std::printf("%s: %d slots (%d taken here, %d bases refused), %d failures\n", g_fail ? "FAIL" : "ok", kSlots,
                wrapped + used, unchanged, g_fail);
    return g_fail ? 1 : 0;
}
