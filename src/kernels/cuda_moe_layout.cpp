// Host storage descriptions and arena layout only; no numerical fallback.
#include "guild/kernels/dequant_bf16.hpp"
#include "guild/kernels/iq_kernels.hpp"
#include "guild/kernels/s2_expert_grouped.hpp"
#include "guild/kernels/shared_expert.hpp"
#include "guild/kernels/gr.hpp"
#include "guild/kernels/ple.hpp"

namespace guild::kernels {
bool dequant_bf16_supported(int t) noexcept {
    switch (t) {
    case 2:
    case 6:
    case 7:
    case 8:
    case 20:
    case 11:
    case 12:
    case 13:
    case 14:
    case 23:
    case 42: return true;
    default: return false;
    }
}
bool iq_supported(int t) noexcept {
    switch (t) {
    case 16:
    case 17:
    case 18:
    case 20:
    case 21:
    case 22:
    case 23:
    case 29:
    case 42:
    case 11:
    case 12:
    case 13:
    case 7:
    case 6:
    case 8: return true;
    default: return false;
    }
}
bool embed_type_supported(int t) noexcept {
    return iq_supported(t) || t == 30;
}
size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
    case 16: return size_t(n / 256) * 66;
    case 17: return size_t(n / 256) * 74;
    case 18: return size_t(n / 256) * 98;
    case 20: return size_t(n / 32) * 18;
    case 21: return size_t(n / 256) * 110;
    case 22: return size_t(n / 256) * 82;
    case 29: return size_t(n / 256) * 56;
    case 23: return size_t(n / 256) * 136;
    case 11: return size_t(n / 256) * 110;
    case 42: return size_t(n / 64) * 18;
    case 12: return size_t(n / 256) * 144;
    case 13: return size_t(n / 256) * 176;
    case 7: return size_t(n / 32) * 24;
    case 6: return size_t(n / 32) * 22;
    case 8: return size_t(n / 32) * 34;
    case 30: return size_t(n) * 2;
    default: return 0;
    }
}
bool native_expert_supported(int gu, int down, int64_t n, int64_t ff) noexcept {
    int qg = 0, qd = 0;
    switch (gu) {
    case 16:
    case 17:
    case 18:
    case 21:
    case 22:
    case 23:
    case 29:
    case 12:
    case 13: qg = 256; break;
    case 42: qg = 64; break;
    case 6:
    case 8: qg = 32; break;
    }
    switch (down) {
    case 20:
    case 7:
    case 6:
    case 8: qd = 32; break;
    case 23: qd = 256; break;
    case 42: qd = 64; break;
    }
    return qg && qd && iq_supported(gu) && iq_supported(down) && n % qg == 0 && ff % qd == 0 && n % 256 == 0 &&
           (ff * n) % 256 == 0;
}
NativeExpertLayout native_expert_layout(int gu, int down, int64_t n, int64_t ff) {
    NativeExpertLayout l;
    l.gu_type = gu;
    l.d_type = down;
    l.n_embd = n;
    l.n_ff = ff;
    l.gu_row = iq_row_bytes(gu, n);
    l.d_row = iq_row_bytes(down, ff);
    l.up_off = size_t(ff) * l.gu_row;
    l.down_off = 2 * l.up_off;
    l.bytes = l.down_off + size_t(n) * l.d_row;
    return l;
}
size_t native_expert_scratch_bytes(int64_t cap, int64_t ff) {
    const size_t f = size_t(cap) * size_t(ff) * sizeof(float);
    return 3 * ((f + 255) & ~size_t(255)) + ((size_t(cap) * size_t(ff / 32) * 36 + 255) & ~size_t(255));
}
uint64_t moe_hit_grouped_scratch_bytes(int64_t hits, int64_t n, int64_t ff) {
    if (hits <= 0)
        return 0;
    const uint64_t gu = uint64_t(hits) * uint64_t(2 * ff) * 4;
    const uint64_t q8 = uint64_t(hits) * uint64_t(ff / 32) * 34;
    const uint64_t hs = uint64_t(hits) * uint64_t(ff / 32) * 4;
    const uint64_t xh = uint64_t(n / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}
uint64_t shared_expert_scratch_bytes(int64_t ff) {
    const uint64_t a = (uint64_t(ff) * 4 + 15) & ~15ull;
    const uint64_t q0 = (uint64_t(ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = (uint64_t(ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}
size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t dim = size_t(s.hc) * size_t(s.n_embd);
    const size_t sz[] = {dim * 4, dim * 2, size_t(s.hc_lr) * 2, dim * 4, size_t(s.hc_lr) * 4};
    size_t bytes = 0;
    void* ptr[5]{};
    for (int k = 0; k < 5; ++k) {
        if (base)
            ptr[k] = static_cast<unsigned char*>(base) + bytes;
        bytes += (sz[k] + 15) & ~size_t(15);
    }
    out.bytes = bytes;
    if (base) {
        out.xn = static_cast<float*>(ptr[0]);
        out.xq = static_cast<uint16_t*>(ptr[1]);
        out.lq = static_cast<uint16_t*>(ptr[2]);
        out.gated = static_cast<float*>(ptr[3]);
        out.lo = static_cast<float*>(ptr[4]);
    }
    return bytes;
}
uint64_t ple_block_scratch_bytes() {
    // Fixed Qwen4Exp PLE geometry: embd 2560, four streams, history nine.
    constexpr size_t f = size_t(5 * 10240 + 2560 + 4) * sizeof(float);
    constexpr size_t q = size_t(2560 / 32) * 34;
    constexpr size_t e = size_t(2560) * sizeof(uint16_t);
    return ((f + 15) & ~size_t(15)) + ((q + 15) & ~size_t(15)) + e + 256;
}
} // namespace guild::kernels
