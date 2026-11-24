// src/core/pool_rows.cpp - see include/guild/core/pool_rows.hpp.
#include "guild/core/pool_rows.hpp"

#include "guild/abi/cpu_backend_abi.h"
#include "guild/core/glm5_nl4.hpp"
#include "guild/core/iq4nl_rows512.hpp"
#include "guild/core/iq4xs_rows512.hpp"
#include "guild/core/kq_rows512.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

#if GUILD_BUNDLED_CPU_ARITHMETIC && defined(GUILD_NATIVE_EXPERTS)
extern "C" const guild_cpu_backend_ext_v2_t* guild_cpu_backend_table_v2();
#endif

namespace guild::core {
namespace {

bool on() {
    static const bool v = [] {
        const char* p = std::getenv("GUILD_POOL_ROWS512");
        const char* o = std::getenv("GUILD_IQ_GGML_ORDER");
        return (p == nullptr || std::atoi(p) != 0) && (o == nullptr || std::atoi(o) != 0) &&
               glm5cpu::nl4_rows512_ok() && cpu_moe::iq4nl_rows512_ok();
    }();
    return v;
}

// Q4_K gate/up and Q4_K / Q6_K down rows (Drama 3, Ornith) on cpu_moe::kq_rows512: two rows a pass at every token
// count, where the library takes ggml's dot one row at a time at one token (native_expert.cpp) and its AVX2-pair
// kernel from two.  Every (row, token) is ggml's one-token dot either way (kq_rows512_test), so only the speed
// changes.  GUILD_KQ_ROWS512=0 (read once): the library's rows, as before.
bool kq_on() {
    static const bool v = [] {
        const char* e = std::getenv("GUILD_KQ_ROWS512");
        return (e == nullptr || std::atoi(e) != 0) && cpu_moe::kq_rows512_ok();
    }();
    return v;
}

// IQ4_XS gate/up rows (Qwen3.8-Flash-Next's layer 2, the one gate/up layer that is not IQ3_S) on
// cpu_moe::iq4xs_rows512, GLM's lean rows: bitwise ggml-cpu's one-token dot at any token count (iq4xs_rows512_test),
// as the library's rows are (ggml's vec_dot one row at a time below two tokens, its AVX-512 i-quant rows from two),
// so only the speed changes.  GUILD_POOL_IQ4XS=0 (read once): the library's rows, as before.
bool iq4xs_on() {
    static const bool v = [] {
        const char* e = std::getenv("GUILD_POOL_IQ4XS");
        return (e == nullptr || std::atoi(e) != 0) && cpu_moe::iq4xs_rows512_ok();
    }();
    return v;
}

constexpr int32_t kIq4XsType = 23; // GGML_TYPE_IQ4_XS

// gate and up rows [r0, r1) of every token, then the library's SwiGLU expression on them (native_gu_rows); the
// formats not rerouted through `base`.  The rows read gate row r at r * gu_row and up row r at up_off past it, the
// [gate | up] layout of the library's own rows.
void gate_up(const guild_cpu_expert_ext_v1_t& base, const guild_cpu_expert_format_v1_t* f, const uint8_t* b,
             const void* const* x, int32_t nt, float* const* ff, int32_t r0, int32_t r1) {
    if (f->gu_type == 12 && kq_on() && cpu_moe::kq_rows512_supported(12, (int) f->n_embd)) {
        cpu_moe::q4k_gu_rows512(b, f->gu_row, f->up_off, (int) f->n_embd, x, nt, ff, r0, r1);
        return;
    }
    const bool nl4 = f->gu_type == guild::kernels::cpu::kIq3sNl4Type;
    const bool xs = f->gu_type == kIq4XsType && iq4xs_on();
    if ((!nl4 && !xs) || f->n_embd % 256 != 0) {
        base.gate_up(f, b, x, nt, ff, r0, r1);
        return;
    }
    thread_local std::vector<float> buf;
    const size_t per = (size_t) f->n_ff;
    if (buf.size() < 2 * per * (size_t) nt)
        buf.resize(2 * per * (size_t) nt);
    std::vector<float*> g((size_t) nt), u((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        g[(size_t) t] = buf.data() + (size_t) (2 * t) * per;
        u[(size_t) t] = buf.data() + (size_t) (2 * t + 1) * per;
    }
    if (nl4) {
        glm5cpu::nl4_rows(b, f->gu_row, (int) f->n_embd, x, nt, g.data(), r0, r1);
        glm5cpu::nl4_rows(b + f->up_off, f->gu_row, (int) f->n_embd, x, nt, u.data(), r0, r1);
    } else {
        // the gate rows' last pair asks for the first up rows
        const uint8_t* up = b + f->up_off;
        cpu_moe::iq4xs_rows512(b, f->gu_row, (int) f->n_embd, x, nt, g.data(), r0, r1, up + (size_t) r0 * f->gu_row,
                               f->gu_row);
        cpu_moe::iq4xs_rows512(up, f->gu_row, (int) f->n_embd, x, nt, u.data(), r0, r1);
    }
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            const float gv = g[(size_t) t][r], uv = u[(size_t) t][r];
            ff[t][r] = (gv / (1.f + std::exp(-gv))) * uv;
        }
}

void down(const guild_cpu_expert_ext_v1_t& base, const guild_cpu_expert_format_v1_t* f, const uint8_t* b,
          const void* const* h, int32_t nt, float* const* out, int32_t r0, int32_t r1) {
    if ((f->d_type == 12 || f->d_type == 14) && kq_on() && cpu_moe::kq_rows512_supported(f->d_type, (int) f->n_ff)) {
        cpu_moe::kq_rows512(f->d_type, b + f->down_off, f->d_row, (int) f->n_ff, h, nt, out, r0, r1);
        return;
    }
    if (f->d_type != 20 || f->n_ff % 64 != 0) { // IQ4_NL rows of an even number of blocks
        base.down(f, b, h, nt, out, r0, r1);
        return;
    }
    cpu_moe::iq4nl_rows512(b + f->down_off, f->d_row, (int) f->n_ff, h, nt, out, r0, r1);
}

// Per-model wrappers.  The C table has no user pointer, so each slot has its own pair of functions, and slot s's
// leave the formats they do not reroute to g_base[s]: two models' tables share a base only when their bases are the
// same functions (every model on the library's table shares slot 0).  g_base[s] is written once, under g_mu, before
// the first table naming slot s is returned (and so before any pool can call it), and never again; the calls read it
// without the lock.  With one base for the process, a model loaded later sent an earlier one's rows to its table.
constexpr int kSlots = detail::kPoolRowsSlots;
guild_cpu_expert_ext_v1_t g_base[kSlots]{};
int g_used = 0; // slots taken, under g_mu
std::mutex g_mu;

template <int S>
void gate_up_slot(const guild_cpu_expert_format_v1_t* f, const uint8_t* b, const void* const* x, int32_t nt,
                  float* const* ff, int32_t r0, int32_t r1) {
    gate_up(g_base[S], f, b, x, nt, ff, r0, r1);
}

template <int S>
void down_slot(const guild_cpu_expert_format_v1_t* f, const uint8_t* b, const void* const* h, int32_t nt,
               float* const* out, int32_t r0, int32_t r1) {
    down(g_base[S], f, b, h, nt, out, r0, r1);
}

using GateUpFn = decltype(guild_cpu_expert_ext_v1_t::gate_up);
using DownFn = decltype(guild_cpu_expert_ext_v1_t::down);

template <int... S> constexpr std::array<GateUpFn, sizeof...(S)> gate_up_slots(std::integer_sequence<int, S...>) {
    return {{&gate_up_slot<S>...}};
}
template <int... S> constexpr std::array<DownFn, sizeof...(S)> down_slots(std::integer_sequence<int, S...>) {
    return {{&down_slot<S>...}};
}
constexpr std::array<GateUpFn, kSlots> kGateUp = gate_up_slots(std::make_integer_sequence<int, kSlots>{});
constexpr std::array<DownFn, kSlots> kDown = down_slots(std::make_integer_sequence<int, kSlots>{});

bool wrapped(const guild_cpu_expert_ext_v1_t& t) {
    for (int s = 0; s < kSlots; ++s)
        if (t.gate_up == kGateUp[(size_t) s] || t.down == kDown[(size_t) s])
            return true;
    return false;
}

// Whether `base` is the bundled guild-kernels-cpu table, the one the rows above are proven against
// (kq_rows512_test compares pool_rows512's table with it; iq4nl/iq4xs_rows512_test and glm5_nl4_test compare the rows
// with ggml-cpu's dot, which that table's rows reproduce): all four functions its own, the quantizers too, as the
// rows read the activations they make.  The registry's default CPU expert kernel (split_multi_native) carries it and
// ExpertPool binds it at construction.  A GEXEC table (gexec_expert.cpp), an external CPU package's (its own build,
// even of the same sources) or one already wrapped is not.
bool library_table(const guild_cpu_expert_ext_v1_t& base) {
#if GUILD_BUNDLED_CPU_ARITHMETIC && defined(GUILD_NATIVE_EXPERTS)
    const guild_cpu_expert_ext_v1_t& lib = guild_cpu_backend_table_v2()->native;
    return base.quant_act == lib.quant_act && base.quant_h == lib.quant_h && base.gate_up == lib.gate_up &&
           base.down == lib.down;
#else
    (void) base;
    return false;
#endif
}

} // namespace

namespace detail {

guild_cpu_expert_ext_v1_t pool_rows512_slot(const guild_cpu_expert_ext_v1_t& base) {
    if (!on() || !base.gate_up || !base.down || wrapped(base))
        return base;
    std::lock_guard<std::mutex> lock(g_mu);
    int s = 0;
    while (s < g_used && (g_base[s].gate_up != base.gate_up || g_base[s].down != base.down))
        ++s;
    if (s == g_used) {
        if (g_used == kSlots) {
            std::fprintf(stderr,
                         "guild experts: the pool's AVX-512BW rows already serve %d CPU expert tables; this one keeps "
                         "its own rows\n",
                         kSlots);
            return base;
        }
        g_base[s] = base;
        ++g_used;
    }
    guild_cpu_expert_ext_v1_t t = base;
    t.gate_up = kGateUp[(size_t) s];
    t.down = kDown[(size_t) s];
    return t;
}

} // namespace detail

guild_cpu_expert_ext_v1_t pool_rows512(const guild_cpu_expert_ext_v1_t& base, bool gate_up_interleaved) {
    if (gate_up_interleaved || !library_table(base))
        return base;
    return detail::pool_rows512_slot(base);
}

} // namespace guild::core
