// src/core/pool_rows.cpp - see include/guild/core/pool_rows.hpp.
#include "guild/core/pool_rows.hpp"

#include "guild/core/glm5_nl4.hpp"
#include "guild/core/iq4nl_rows512.hpp"
#include "guild/kernels/cpu/iq_avx2.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace guild::core {
namespace {

guild_cpu_expert_ext_v1_t g_base{}; // what the wrappers leave to

bool on() {
    static const bool v = [] {
        const char* p = std::getenv("GUILD_POOL_ROWS512");
        const char* o = std::getenv("GUILD_IQ_GGML_ORDER");
        return (p == nullptr || std::atoi(p) != 0) && (o == nullptr || std::atoi(o) != 0) &&
               glm5cpu::nl4_rows512_ok() && cpu_moe::iq4nl_rows512_ok();
    }();
    return v;
}

// gate and up rows [r0, r1) of every token, then the library's SwiGLU expression on them (native_gu_rows)
void gate_up(const guild_cpu_expert_format_v1_t* f, const uint8_t* b, const void* const* x, int32_t nt,
             float* const* ff, int32_t r0, int32_t r1) {
    if (f->gu_type != guild::kernels::cpu::kIq3sNl4Type || f->n_embd % 256 != 0) {
        g_base.gate_up(f, b, x, nt, ff, r0, r1);
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
    glm5cpu::nl4_rows(b, f->gu_row, (int) f->n_embd, x, nt, g.data(), r0, r1);
    glm5cpu::nl4_rows(b + f->up_off, f->gu_row, (int) f->n_embd, x, nt, u.data(), r0, r1);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            const float gv = g[(size_t) t][r], uv = u[(size_t) t][r];
            ff[t][r] = (gv / (1.f + std::exp(-gv))) * uv;
        }
}

void down(const guild_cpu_expert_format_v1_t* f, const uint8_t* b, const void* const* h, int32_t nt, float* const* out,
          int32_t r0, int32_t r1) {
    if (f->d_type != 20 || f->n_ff % 64 != 0) { // IQ4_NL rows of an even number of blocks
        g_base.down(f, b, h, nt, out, r0, r1);
        return;
    }
    cpu_moe::iq4nl_rows512(b + f->down_off, f->d_row, (int) f->n_ff, h, nt, out, r0, r1);
}

} // namespace

guild_cpu_expert_ext_v1_t pool_rows512(const guild_cpu_expert_ext_v1_t& base) {
    if (!on() || !base.gate_up || !base.down)
        return base;
    g_base = base;
    guild_cpu_expert_ext_v1_t t = base;
    t.gate_up = gate_up;
    t.down = down;
    return t;
}

} // namespace guild::core
