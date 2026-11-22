// tests/core/drama3_fuse_test.cpp - Drama 3 tower's faster step paths (GUILD_DRAMA3_FUSE) against the paths they
// replace, bit for bit.
//
//     drama3_fuse_test
//
// The fused small kernels (drama3_tower_kernels.hpp) against the separate ones they replace (add, rms_norm, embed and
// guild-kernels-cuda's own native_quantize_q8_1, as linked, not a copy) on random rows, massive activations, zeros,
// subnormal products and blocks past FP16's range.  Exit 0 when everything matches, 1 on a difference, 77 without a
// GPU.
#include "guild/core/drama3_tower_kernels.hpp"
#include "guild/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace d3 = guild::core::drama3;

namespace {

int g_fail = 0;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        const cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) {                                                                                       \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                       \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

template <typename T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) {
        CK(cudaMalloc((void**) &p, count * sizeof(T) + 64));
        CK(cudaMemset(p, 0, count * sizeof(T) + 64));
    }
    ~Dev() {
        cudaFree(p);
    }
    void put(const std::vector<T>& v) {
        CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get(size_t count) const {
        std::vector<T> v(count);
        CK(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

// byte-compares two device buffers; counts and reports a difference
bool same(const void* a, const void* b, size_t bytes, const std::string& what, bool quiet = false) {
    std::vector<unsigned char> ha(bytes), hb(bytes);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost));
    size_t first = bytes, n = 0;
    for (size_t i = 0; i < bytes; ++i)
        if (ha[i] != hb[i]) {
            n += 1;
            first = first == bytes ? i : first;
        }
    if (n) {
        if (++g_fail <= 20)
            std::printf("DIFFER %s: %zu of %zu bytes, first at %zu\n", what.c_str(), n, bytes, first);
        return false;
    }
    if (!quiet)
        std::printf("same   %s (%zu bytes)\n", what.c_str(), bytes);
    return true;
}

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (u >> 16);
}

// ---- the fused kernels against the separate ones
struct Rows {
    std::vector<float> x, y, w;
};

enum class Kind {
    normal,
    massive,
    zeros,
    subnormal,
    huge,
    signed_zero
};

Rows make_rows(Kind kind, std::mt19937& rng) {
    const int D = d3::kFusedDim;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Rows r;
    r.x.resize(D);
    r.y.resize(D);
    r.w.resize(D);
    for (int i = 0; i < D; ++i) {
        r.x[i] = 2.0f * nd(rng);
        r.y[i] = nd(rng);
        r.w[i] = 1.0f + 0.2f * nd(rng);
    }
    switch (kind) {
    case Kind::normal:
        r.x[(size_t) (rng() % D)] = 400.0f * (1 + rng() % 30); // one massive activation, as the backbone has
        break;
    case Kind::massive: // |x| ~ 1e4 everywhere
        for (int i = 0; i < D; ++i)
            r.x[i] = 1.0e4f * nd(rng);
        break;
    case Kind::zeros:
        for (int i = 0; i < D; ++i)
            r.x[i] = r.y[i] = 0.0f;
        break;
    case Kind::subnormal: // products below FLT_MIN: whole blocks, and single elements of normal blocks
        for (int i = 0; i < D; ++i) {
            const int b = i / 32;
            if (b % 4 == 0)
                r.w[i] = 1.0e-39f * nd(rng);
            else if (b % 4 == 1 && i % 7 == 0)
                r.w[i] = 3.0e-40f;
            else if (b % 4 == 2)
                r.w[i] = 1.0e-37f * nd(rng); // normal products whose amax / 127 is subnormal
        }
        break;
    case Kind::huge: // products past FP16's range: sums past 65504, and d = amax / 127 past it too
        for (int i = 0; i < D; ++i) {
            const int b = i / 32;
            r.w[i] = b % 3 == 0 ? 3.0e4f * nd(rng) : b % 3 == 1 ? 2.0e7f * nd(rng) : r.w[i];
        }
        break;
    case Kind::signed_zero:
        for (int i = 0; i < D; ++i) {
            r.x[i] = i % 3 == 0 ? -0.0f : r.x[i];
            r.y[i] = i % 3 == 0 ? -0.0f : r.y[i];
            r.w[i] = i % 5 == 0 ? -0.0f : r.w[i];
        }
        break;
    }
    return r;
}

const char* kind_name(Kind k) {
    switch (k) {
    case Kind::normal: return "random";
    case Kind::massive: return "massive";
    case Kind::zeros: return "zeros";
    case Kind::subnormal: return "subnormal products";
    case Kind::huge: return "past FP16";
    case Kind::signed_zero: return "signed zeros";
    }
    return "?";
}

void test_add_norm(cudaStream_t st) {
    const int D = d3::kFusedDim, NB = D / 32;
    const float eps = 1e-6f;
    const size_t qbytes = guild::kernels::native_q8_1_bytes(D, 1);
    Dev<float> x0(D), x1(D), y(D), w(D), n0(D), n1(D);
    Dev<unsigned char> q0(qbytes), q1(qbytes);
    std::mt19937 rng(11);
    for (Kind kind : {Kind::normal, Kind::massive, Kind::zeros, Kind::subnormal, Kind::huge, Kind::signed_zero}) {
        const int trials = kind == Kind::normal || kind == Kind::massive ? 300 : 40;
        int bad = 0;
        for (int t = 0; t < trials; ++t) {
            const Rows r = make_rows(kind, rng);
            for (int variant = 0; variant < 4; ++variant) {
                // 0: x += y, norm, q8 (the ffn norm); 1: norm, q8 (attn norm); 2: x += y, norm out + q8 (final norm,
                // a head next); 3: x += y, norm out only (final norm, no head)
                const bool add = variant != 1, out = variant >= 2, q8 = variant != 3;
                x0.put(r.x);
                x1.put(r.x);
                y.put(r.y);
                w.put(r.w);
                CK(cudaMemset(n1.p, 0xff, D * 4));
                CK(cudaMemset(q1.p, 0xff, qbytes));
                if (add)
                    d3::add(x0.p, y.p, D, st);
                d3::rms_norm(x0.p, w.p, n0.p, D, eps, st);
                guild::kernels::native_quantize_q8_1(n0.p, q0.p, D, 1, st);
                d3::add_norm_q8(x1.p, add ? y.p : nullptr, w.p, eps, out ? n1.p : nullptr, q8 ? q1.p : nullptr, st);
                const std::string what = std::string("add_norm_q8 ") + kind_name(kind) + " variant " +
                                         std::to_string(variant) + " trial " + std::to_string(t);
                bool ok = same(x0.p, x1.p, (size_t) D * 4, what + " x", true);
                if (out)
                    ok &= same(n0.p, n1.p, (size_t) D * 4, what + " out", true);
                if (q8)
                    ok &= same(q0.p, q1.p, (size_t) NB * d3::kFusedQ81Bytes, what + " q8_1", true);
                bad += !ok;
            }
        }
        std::printf("%s add_norm_q8, %s: %d trials x 4 variants, %d differ\n", bad ? "DIFFER" : "same  ",
                    kind_name(kind), trials, bad);
    }
}

void test_embed_norm2(cudaStream_t st) {
    const int D = d3::kFusedDim, NB = D / 32, CS = 64, NCB = 10;
    const float eps = 1e-6f;
    std::mt19937 rng(12);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> table((size_t) CS * D), adds((size_t) CS * NCB * D);
    for (auto& v : table)
        v = to_bf16(0.05f * nd(rng));
    for (auto& v : adds)
        v = to_bf16(0.05f * nd(rng));
    Dev<uint16_t> dt(table.size()), da(adds.size());
    dt.put(table);
    da.put(adds);
    const size_t qbytes = guild::kernels::native_q8_1_bytes(2 * D, 1);
    Dev<float> e(D), h(D), w0(D), w1(D), c0(2 * D), c1(2 * D);
    Dev<unsigned char> q0(qbytes), q1(qbytes);
    int bad = 0, trials = 0;
    for (int t = 0; t < 120; ++t) {
        std::vector<float> hh(D), hw0(D), hw1(D);
        for (int i = 0; i < D; ++i) {
            hh[i] = 3.0f * nd(rng);
            hw0[i] = 1.0f + 0.2f * nd(rng);
            hw1[i] = 1.0f + 0.2f * nd(rng);
        }
        hh[(size_t) (rng() % D)] = 2.0e3f;
        h.put(hh);
        w0.put(hw0);
        w1.put(hw1);
        // row -1 (a text position: zeros), a row alone, a row with layer 0's eight adds, and adds alone
        for (int c = 0; c < 4; ++c) {
            const int row = c == 0 || c == 3 ? -1 : (int) (rng() % CS);
            d3::EmbedAdd add;
            if (c >= 2)
                for (int k = 2; k < NCB; ++k)
                    add.rows[add.n++] = k * CS + (int) (rng() % CS);
            CK(cudaMemset(c1.p, 0xff, 2 * D * 4));
            CK(cudaMemset(q1.p, 0xff, qbytes));
            d3::embed(dt.p, row, add, da.p, e.p, D, st);
            d3::rms_norm(e.p, w0.p, c0.p, D, eps, st);
            d3::rms_norm(h.p, w1.p, c0.p + D, D, eps, st);
            guild::kernels::native_quantize_q8_1(c0.p, q0.p, 2 * D, 1, st);
            d3::embed_norm2_q8(dt.p, row, add, da.p, h.p, w0.p, w1.p, eps, q1.p, st, c1.p);
            const std::string what = "embed_norm2_q8 case " + std::to_string(c) + " trial " + std::to_string(t);
            bool ok = same(c0.p, c1.p, (size_t) 2 * D * 4, what + " rows", true);
            ok &= same(q0.p, q1.p, (size_t) 2 * NB * d3::kFusedQ81Bytes, what + " q8_1", true);
            bad += !ok;
            ++trials;
        }
    }
    std::printf("%s embed_norm2_q8 (row -1, a row, a row + 8 adds, adds alone): %d trials, %d differ\n",
                bad ? "DIFFER" : "same  ", trials, bad);
}

} // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    if (guild::kernels::native_q8_1_bytes(32, 1) != (size_t) d3::kFusedQ81Bytes) {
        std::printf("FAIL: the native Q8_1 block is %zu bytes, the fused kernels write %d\n",
                    guild::kernels::native_q8_1_bytes(32, 1), d3::kFusedQ81Bytes);
        return 1;
    }
    cudaStream_t st = nullptr;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    test_add_norm(st);
    test_embed_norm2(st);
    (void) argc;
    (void) argv;
    CK(cudaStreamDestroy(st));
    std::printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
