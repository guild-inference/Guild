// tests/core/drama3_fuse_test.cpp - Drama 3 tower's faster step paths (GUILD_DRAMA3_FUSE) against the paths they
// replace, bit for bit.
//
//     drama3_fuse_test [logits.bin]
//
// The fused small kernels (drama3_tower_kernels.hpp) against the separate ones they replace (add, rms_norm, embed and
// guild-kernels-cuda's own native_quantize_q8_1, as linked, not a copy) on random rows, massive activations, zeros,
// subnormal products and blocks past FP16's range; the staged attention against attention() at contexts across both
// of its kernels; [w1|w3] as one 12288-row Q8_0 call against two 6144-row ones; the fast sampler against the reference
// on synthetic logits (ties, signed zeros, wide ranges, nine settings) and, given a GUILD_DRAMA3_DUMP_LOGITS file, on
// every row of it, plus its exp against std::exp on a stride of all float patterns.  Exit 0 when everything matches, 1
// on a difference, 77 without a GPU.
#include "guild/core/drama3_tower.hpp"
#include "guild/core/drama3_tower_kernels.hpp"
#include "guild/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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

// ---- the sampler: sample_fast against sample_reference, the same index for every draw
struct Setting {
    float temperature, top_p;
    int top_k;
};
const Setting kSettings[] = {{1.0f, 0.9f, 30},  {1.0f, 0.95f, 30}, {0.7f, 0.9f, 30},
                             {1.3f, 0.8f, 50},  {1.0f, 1.0f, 64},  {0.5f, 0.5f, 1},
                             {2.0f, 0.99f, 64}, {1e-7f, 0.9f, 30}, {1.0f, 1.0f, 4096}};

struct SampleCount {
    uint64_t draws = 0, fast = 0, differ = 0;
};

void compare_rows(const float* logits, int n, uint64_t seed, int draws, SampleCount& c) {
    for (const Setting& s : kSettings) {
        guild::core::Drama3Sampling sp;
        sp.temperature = s.temperature;
        sp.top_p = s.top_p;
        sp.top_k = s.top_k;
        sp.seed = seed;
        for (int d = 0; d < draws; ++d) {
            const uint64_t draw = seed * 31 + (uint64_t) d;
            const int a = d3::sample_reference(logits, n, sp, draw), b = d3::sample_fast(logits, n, sp, draw);
            c.draws += 1;
            if (b == -2)
                continue;
            c.fast += 1;
            if (a != b && ++c.differ <= 5)
                std::printf("DIFFER sample: n %d T %g top_p %g top_k %d draw %llu: %d vs %d\n", n, s.temperature,
                            s.top_p, s.top_k, (unsigned long long) draw, a, b);
        }
    }
}

void test_sampler(const char* dump) {
    if (!d3::sample_fast_ok()) {
        std::printf("skip   sampler: no fast path here (needs AVX-512 F/DQ/BW/VL and glibc's exp)\n");
        return;
    }
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    SampleCount c;
    std::vector<float> l(4097);
    for (int row = 0; row < 1500; ++row) {
        const int n = row % 2 ? 4096 : 4097, shape = row % 6;
        const float scale = shape == 0 ? 1.0f : shape == 1 ? 3.0f : 8.0f;
        for (int i = 0; i < n; ++i)
            l[i] = scale * nd(rng);
        if (shape == 3) // ties: logits on a 1/4 grid
            for (int i = 0; i < n; ++i)
                l[i] = std::round(l[i] * 4.0f) / 4.0f;
        if (shape == 4) { // one peaked logit, and signed zeros
            l[(size_t) (rng() % n)] = 25.0f;
            for (int i = 0; i < n; i += 5)
                l[i] = (i / 5) % 2 ? 0.0f : -0.0f;
        }
        if (shape == 5) // a wide range: most exps underflow
            for (int i = 0; i < n; ++i)
                l[i] = -1.0e3f * std::fabs(nd(rng)) + (i % 97 == 0 ? 50.0f : 0.0f);
        compare_rows(l.data(), n, (uint64_t) row, 4, c);
    }
    // a non-finite logit: both refuse
    for (const float bad : {NAN, INFINITY, -INFINITY}) {
        for (int i = 0; i < 4097; ++i)
            l[i] = nd(rng);
        l[1234] = bad;
        guild::core::Drama3Sampling sp;
        const int a = d3::sample_reference(l.data(), 4097, sp, 0), b = d3::sample_fast(l.data(), 4097, sp, 0);
        if (a != -1 || b != -1) {
            ++g_fail;
            std::printf("DIFFER sample of a non-finite row: %d and %d, not -1\n", a, b);
        }
    }
    g_fail += c.differ != 0;
    std::printf("%s sampler, synthetic rows: %llu draws, %llu on the fast path, %llu differ\n",
                c.differ ? "DIFFER" : "same  ", (unsigned long long) c.draws, (unsigned long long) c.fast,
                (unsigned long long) c.differ);
    if (dump) { // a GUILD_DRAMA3_DUMP_LOGITS file: per frame 4097 candidate logits, then 9 x 4096 code logits
        std::FILE* f = std::fopen(dump, "rb");
        if (!f) {
            ++g_fail;
            std::printf("FAIL: cannot read %s\n", dump);
            return;
        }
        std::vector<float> fr(4097 + 9 * 4096);
        SampleCount cd;
        uint64_t frames = 0;
        while (std::fread(fr.data(), 4, fr.size(), f) == fr.size()) {
            compare_rows(fr.data(), 4097, frames * 10, 4, cd);
            for (int h = 0; h < 9; ++h)
                compare_rows(fr.data() + 4097 + (size_t) h * 4096, 4096, frames * 10 + 1 + h, 4, cd);
            ++frames;
        }
        std::fclose(f);
        g_fail += cd.differ != 0 || frames == 0;
        std::printf("%s sampler, %llu dumped frames: %llu draws, %llu on the fast path, %llu differ\n",
                    cd.differ || !frames ? "DIFFER" : "same  ", (unsigned long long) frames,
                    (unsigned long long) cd.draws, (unsigned long long) cd.fast, (unsigned long long) cd.differ);
    }
    // the exp itself on every 97th negative float (the sampler's inputs are <= 0) and every 997th positive one
    const uint64_t neg = d3::exp_mismatches(0x80000000ull, 0xff800001ull, 97),
                   pos = d3::exp_mismatches(0x00000000ull, 0x7f800001ull, 997);
    g_fail += neg + pos != 0;
    std::printf("%s exp8 against std::exp: %llu negative and %llu positive float patterns differ\n",
                neg + pos ? "DIFFER" : "same  ", (unsigned long long) neg, (unsigned long long) pos);
}

// ---- attention_fast (staged K/V, two blocks a K/V head at short contexts) against attention, partials and output
void test_attention(cudaStream_t st) {
    const d3::QkvShape s{16, 2, 256, 64, 1e7f, 1e-6f};
    const int max_pos = 2048, Q = s.n_head * s.head_dim, KV = s.n_kv * s.head_dim;
    std::mt19937 rng(14);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> hk((size_t) max_pos * KV), hv(hk.size());
    for (auto& v : hk) {
        const __half h = __float2half(2.0f * nd(rng));
        std::memcpy(&v, &h, 2);
    }
    for (auto& v : hv) {
        const __half h = __float2half(nd(rng));
        std::memcpy(&v, &h, 2);
    }
    std::vector<float> hq((size_t) Q), hg((size_t) Q);
    for (auto& v : hq)
        v = 0.5f * nd(rng);
    for (auto& v : hg)
        v = nd(rng);
    Dev<uint16_t> k(hk.size()), v(hv.size());
    k.put(hk);
    v.put(hv);
    Dev<float> q(Q), g(Q), o0(Q), o1(Q);
    q.put(hq);
    g.put(hg);
    const size_t pf = d3::attn_part_floats(s, max_pos);
    Dev<float> p0(pf), p1(pf);
    int bad = 0;
    for (const int64_t n_pos : {1, 63, 64, 65, 130, 319, 320, 321, 565, 2000}) {
        const size_t used = (size_t) s.n_kv * (size_t) ((n_pos + 63) / 64) * 8 * (s.head_dim + 2);
        CK(cudaMemset(p0.p, 0, pf * 4));
        CK(cudaMemset(p1.p, 0xff, pf * 4));
        if (!d3::attention(q.p, k.p, v.p, g.p, p0.p, o0.p, s, n_pos, st) ||
            !d3::attention_fast(q.p, k.p, v.p, g.p, p1.p, o1.p, s, n_pos, st)) {
            ++g_fail;
            std::printf("FAIL: attention refused the tower's shape\n");
            return;
        }
        const std::string what = "attention n_pos " + std::to_string(n_pos);
        bool ok = same(p0.p, p1.p, used * 4, what + " partials", true);
        ok &= same(o0.p, o1.p, (size_t) Q * 4, what + " output", true);
        bad += !ok;
    }
    std::printf("%s attention_fast at n_pos 1, 63, 64, 65, 130, 319, 320, 321, 565, 2000: %d differ\n",
                bad ? "DIFFER" : "same  ", bad);
}

// ---- [w1|w3]: one 12288-row Q8_0 call against the two 6144-row calls (gkc's MMVQ sums every row alone)
void test_w13(cudaStream_t st) {
    const int D = 2048, F = 6144, rb = D / 32 * 34;
    std::mt19937 rng(15);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint8_t> w((size_t) 2 * F * rb);
    for (size_t b = 0; b < w.size() / 34; ++b) { // Q8_0 blocks: an FP16 scale and 32 int8
        const __half d = __float2half(0.002f * (1.0f + 0.3f * nd(rng)));
        std::memcpy(&w[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i)
            w[b * 34 + 2 + i] = (uint8_t) (int8_t) std::max(-127.0f, std::min(127.0f, std::round(40.0f * nd(rng))));
    }
    Dev<uint8_t> dw(w.size());
    dw.put(w);
    const size_t qbytes = guild::kernels::native_q8_1_bytes(D, 1);
    Dev<float> x(D), y0(2 * F), y1(2 * F);
    Dev<unsigned char> xq(qbytes);
    int bad = 0;
    for (int t = 0; t < 20; ++t) {
        std::vector<float> hx(D);
        for (auto& e : hx)
            e = nd(rng);
        x.put(hx);
        guild::kernels::native_quantize_q8_1(x.p, xq.p, D, 1, st);
        CK(cudaMemset(y1.p, 0xff, (size_t) 2 * F * 4));
        guild::kernels::native_mmvq(8, dw.p, xq.p, y0.p, D, F, 1, st);
        guild::kernels::native_mmvq(8, dw.p + (size_t) F * rb, xq.p, y0.p + F, D, F, 1, st);
        guild::kernels::native_mmvq(8, dw.p, xq.p, y1.p, D, 2 * F, 1, st);
        bad += !same(y0.p, y1.p, (size_t) 2 * F * 4, "w13 trial " + std::to_string(t), true);
    }
    std::printf("%s [w1|w3] as one 12288-row Q8_0 call against two 6144-row calls: 20 trials, %d differ\n",
                bad ? "DIFFER" : "same  ", bad);
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
    test_attention(st);
    test_w13(st);
    test_sampler(argc > 1 ? argv[1] : nullptr);
    CK(cudaStreamDestroy(st));
    std::printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
