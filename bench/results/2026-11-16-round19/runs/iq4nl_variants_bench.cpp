// nlb.cpp - IQ4_NL down-row kernel variants, timing only (scratch): which part of iq4nl_rows512's row2 costs what.
// Build: g++ -O3 -std=c++20 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -mf16c -ffp-contract=off nlb.cpp -lpthread
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#include <pthread.h>

constexpr size_t kBlock = 18;
struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
alignas(16) const int8_t kValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
alignas(16) const int8_t kAbs[16] = {127, 104, 83, 65, 49, 35, 22, 10, 1, 13, 25, 38, 53, 69, 89, 113};
inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline float hsum_ggml(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m512i decode2(const uint8_t* b0, const uint8_t* b1, __m512i values, __m512i m4) {
    const __m128i q0 = _mm_loadu_si128((const __m128i*) (b0 + 2));
    const __m128i q1 = _mm_loadu_si128((const __m128i*) (b1 + 2));
    const __m512i bits = _mm512_mask_broadcast_i32x4(_mm512_broadcast_i32x4(q0), 0xFF00, q1);
    const __m512i nib =
        _mm512_mask_blend_epi64(0xCC, _mm512_and_si512(bits, m4), _mm512_and_si512(_mm512_srli_epi16(bits, 4), m4));
    return _mm512_shuffle_epi8(values, nib);
}
inline __m512i decode2v(const uint8_t* b0, const uint8_t* b1, __m512i values, __m512i m4, __m512i shift) {
    const __m512i bits = _mm512_mask_broadcast_i32x4(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) (b0 + 2))),
                                                     0xFF00, _mm_loadu_si128((const __m128i*) (b1 + 2)));
    return _mm512_shuffle_epi8(values, _mm512_and_si512(_mm512_srlv_epi64(bits, shift), m4));
}

// V: 0 = v1 as shipped; 1 = v1 with d_y from a per-call table; 2 = 1 + weight scales by vpermps of one cvtph; 3 = no
// scale math (constant d; timing only); 4 = no decode (raw bytes; timing only); 5 = v1 without prefetch; 6 = 2 + srlv
// decode; 7 = 6 with the 4 weight-scale words assembled in a GPR
template <int NT, int V>
__attribute__((always_inline)) inline void row2(const uint8_t* row0, const uint8_t* row1, int nblocks, const BlockQ8_0* const* y, const float* dy, float* res0,
          float* res1, const char* pf0, const char* pf1, const __m512i* yb = nullptr) {
    const __m512i values = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) kValues));
    const __m512i m4 = _mm512_set1_epi8(0x0f), zero = _mm512_setzero_si512(), ones = _mm512_set1_epi16(1);
    const __m512i shift = _mm512_set_epi64(4, 4, 0, 0, 4, 4, 0, 0);
    const __m512i i1 = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0); // [x00 x8 | x10 x8]
    const __m512i i2 = _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2); // [x01 x8 | x11 x8]
    __m512 acc1[NT], acc2[NT];
    for (int t = 0; t < NT; ++t)
        acc1[t] = acc2[t] = _mm512_setzero_ps();
    for (int ib = 0; ib + 1 < nblocks; ib += 2) {
        if (V != 5 && V != 9 && (ib & 3) == 0) {
            _mm_prefetch(pf0 + (size_t) ib * kBlock, _MM_HINT_T0);
            _mm_prefetch(pf1 + (size_t) ib * kBlock, _MM_HINT_T0);
        }
        const uint8_t *a0 = row0 + (size_t) ib * kBlock, *a1 = row1 + (size_t) ib * kBlock;
        __m512i w1, w2;
        if constexpr (V >= 10) {
            w1 = w2 = zero;
        } else if constexpr (V == 4) {
            w1 = _mm512_loadu_si512(a0);
            w2 = _mm512_loadu_si512(a1);
        } else if constexpr (V == 6 || V == 7 || V >= 8) {
            w1 = decode2v(a0, a1, values, m4, shift);
            w2 = decode2v(a0 + kBlock, a1 + kBlock, values, m4, shift);
        } else {
            w1 = decode2(a0, a1, values, m4);
            w2 = decode2(a0 + kBlock, a1 + kBlock, values, m4);
        }
        __m512i aw1, aw2;
        __mmask64 n1, n2;
        if constexpr (V >= 10) {
            const __m512i absv = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) kAbs));
            const __m512i eight = _mm512_set1_epi8(8);
            const __m512i nb1 = _mm512_and_si512(_mm512_srlv_epi64(_mm512_mask_broadcast_i32x4(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) (a0 + 2))), 0xFF00, _mm_loadu_si128((const __m128i*) (a1 + 2))), shift), m4);
            const __m512i nb2 = _mm512_and_si512(_mm512_srlv_epi64(_mm512_mask_broadcast_i32x4(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) (a0 + kBlock + 2))), 0xFF00, _mm_loadu_si128((const __m128i*) (a1 + kBlock + 2))), shift), m4);
            aw1 = _mm512_shuffle_epi8(absv, nb1); aw2 = _mm512_shuffle_epi8(absv, nb2);
            n1 = _mm512_testn_epi8_mask(nb1, eight); n2 = _mm512_testn_epi8_mask(nb2, eight);
        } else {
            aw1 = _mm512_abs_epi8(w1); aw2 = _mm512_abs_epi8(w2);
            n1 = _mm512_movepi8_mask(w1); n2 = _mm512_movepi8_mask(w2);
        }
        uint16_t h00, h01, h10, h11;
        std::memcpy(&h00, a0, 2);
        std::memcpy(&h10, a1, 2);
        std::memcpy(&h01, a0 + kBlock, 2);
        std::memcpy(&h11, a1 + kBlock, 2);
        float x00 = 0, x10 = 0, x01 = 0, x11 = 0;
        __m512 X1, X2;
        if constexpr (V == 2 || V == 6 || V == 11) {
            const __m128i h = _mm_set_epi16(0, 0, 0, 0, (short) h11, (short) h01, (short) h10, (short) h00);
            const __m512 f = _mm512_castps128_ps512(_mm_cvtph_ps(h)); // [x00, x10, x01, x11]
            X1 = _mm512_permutexvar_ps(i1, f);
            X2 = _mm512_permutexvar_ps(i2, f);
        } else if constexpr (V == 7 || (V >= 8 && V != 11)) {
            const uint64_t hw = (uint64_t) h00 | ((uint64_t) h10 << 16) | ((uint64_t) h01 << 32) | ((uint64_t) h11 << 48);
            const __m512 f = _mm512_castps128_ps512(_mm_cvtph_ps(_mm_cvtsi64_si128((long long) hw)));
            X1 = _mm512_permutexvar_ps(i1, f);
            X2 = _mm512_permutexvar_ps(i2, f);
        } else {
            x00 = h2f(h00), x10 = h2f(h10), x01 = h2f(h01), x11 = h2f(h11);
        }
        for (int t = 0; t < NT; ++t) {
            const BlockQ8_0& yb1 = y[t][ib];
            const BlockQ8_0& yb2 = y[t][ib + 1];
            __m512i y1, y2;
            if constexpr (V == 8 || V == 9) {
                y1 = _mm512_load_si512(yb + (size_t) t * nblocks + ib);
                y2 = _mm512_load_si512(yb + (size_t) t * nblocks + ib + 1);
            } else {
                y1 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) yb1.qs));
                y2 = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) yb2.qs));
            }
            const __m512i p1 =
                _mm512_madd_epi16(_mm512_maddubs_epi16(aw1, _mm512_mask_sub_epi8(y1, n1, zero, y1)), ones);
            const __m512i p2 =
                _mm512_madd_epi16(_mm512_maddubs_epi16(aw2, _mm512_mask_sub_epi8(y2, n2, zero, y2)), ones);
            __m512 d1, d2;
            if constexpr (V == 3) {
                d1 = d2 = _mm512_set1_ps(1.0f);
            } else if constexpr (V == 2 || V == 6 || V == 7 || V >= 8) {
                d1 = _mm512_mul_ps(X1, _mm512_set1_ps(dy[t * nblocks + ib]));
                d2 = _mm512_mul_ps(X2, _mm512_set1_ps(dy[t * nblocks + ib + 1]));
            } else {
                const float dy1 = V == 0 ? h2f(yb1.d) : dy[t * nblocks + ib];
                const float dy2 = V == 0 ? h2f(yb2.d) : dy[t * nblocks + ib + 1];
                d1 = _mm512_insertf32x8(_mm512_set1_ps(dy1 * x00), _mm256_set1_ps(dy1 * x10), 1);
                d2 = _mm512_insertf32x8(_mm512_set1_ps(dy2 * x01), _mm256_set1_ps(dy2 * x11), 1);
            }
            acc1[t] = _mm512_fmadd_ps(d1, _mm512_cvtepi32_ps(p1), acc1[t]);
            acc2[t] = _mm512_fmadd_ps(d2, _mm512_cvtepi32_ps(p2), acc2[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m512 s = _mm512_add_ps(acc1[t], acc2[t]);
        res0[t] = hsum_ggml(_mm512_castps512_ps256(s));
        res1[t] = hsum_ggml(_mm512_extractf32x8_ps(s, 1));
    }
}

template <int NT, int V>
void rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const int nb = n / 32;
    const BlockQ8_0* y[NT];
    alignas(64) float dy[NT * 128];
    alignas(64) __m512i ybuf[(V == 8 || V == 9) ? NT * 64 : 1];
    for (int t = 0; t < NT; ++t) {
        y[t] = (const BlockQ8_0*) act[t];
        if (V != 0)
            for (int b = 0; b < nb; ++b)
                dy[t * nb + b] = h2f(y[t][b].d);
        if constexpr (V == 8 || V == 9)
            for (int b = 0; b < nb; ++b)
                ybuf[t * nb + b] = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) y[t][b].qs));
    }
    float a[NT], b[NT];
    for (int r = r0; r < r1; r += 2) {
        const uint8_t* row0 = w + (size_t) r * row_bytes;
        const bool two = r + 1 < r1;
        const char* pf = (const char*) row0 + 2 * row_bytes;
        row2<NT, V>(row0, two ? row0 + row_bytes : row0, nb, y, dy, a, b, pf, pf + row_bytes, ybuf);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = a[t];
            if (two)
                out[t][r + 1] = b[t];
        }
    }
}

using Fn = void (*)(const uint8_t*, size_t, int, const void* const*, float* const*, int, int);
template <int V> Fn pick(int nt) {
    switch (nt) {
    case 1: return rows<1, V>;
    case 2: return rows<2, V>;
    case 3: return rows<3, V>;
    default: return rows<4, V>;
    }
}
Fn pickv(int v, int nt) {
    switch (v) {
    case 0: return pick<0>(nt);
    case 1: return pick<1>(nt);
    case 2: return pick<2>(nt);
    case 3: return pick<3>(nt);
    case 4: return pick<4>(nt);
    case 5: return pick<5>(nt);
    case 6: return pick<6>(nt);
    case 7: return pick<7>(nt);
    case 8: return pick<8>(nt);
    case 9: return pick<9>(nt);
    case 10: return pick<10>(nt);
    default: return pick<11>(nt);
    }
}

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 1, nt = argc > 2 ? std::atoi(argv[2]) : 1;
    const int n = argc > 3 ? std::atoi(argv[3]) : 640;
    const size_t mb = argc > 4 ? std::atoi(argv[4]) : 128; // per thread; small = cache-resident
    const size_t rb = (size_t) (n / 32) * kBlock;
    const int rows_n = (int) ((mb << 20) / rb), item = 64;
    std::printf("n %d, %d threads, %d tokens, %zu MB a thread\n", n, threads, nt, mb);
    for (int v : {0, 6, 7, 10, 11}) {
        std::vector<double> gbs((size_t) threads);
        std::atomic<int> ready{0};
        std::vector<std::thread> th;
        for (int i = 0; i < threads; ++i)
            th.emplace_back([&, i] {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(i, &set);
                pthread_setaffinity_np(pthread_self(), sizeof set, &set);
                std::mt19937 r((unsigned) i);
                std::vector<uint8_t> w(rb * rows_n);
                for (auto& x : w)
                    x = (uint8_t) r();
                for (size_t b = 0; b < w.size() / kBlock; ++b) {
                    const uint16_t d = 0x2c00 + (r() & 0x3ff);
                    std::memcpy(w.data() + b * kBlock, &d, 2);
                }
                std::vector<BlockQ8_0> a((size_t) nt * (n / 32));
                for (auto& blk : a) {
                    blk.d = 0x2c00 + (r() & 0x3ff);
                    for (auto& q : blk.qs)
                        q = (int8_t) ((int) (r() % 255) - 127);
                }
                std::vector<float> o((size_t) rows_n * nt);
                const void* act[8];
                float* out[8];
                for (int t = 0; t < nt; ++t) {
                    act[t] = a.data() + (size_t) t * (n / 32);
                    out[t] = o.data() + (size_t) t * rows_n;
                }
                std::vector<int> order;
                for (int q = 0; q < rows_n; q += item)
                    order.push_back(q);
                std::shuffle(order.begin(), order.end(), r);
                const Fn f = pickv(v, nt);
                ready.fetch_add(1);
                while (ready.load() < threads) {}
                const int reps = mb < 16 ? 200 : 3;
                const auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < reps; ++k)
                    for (int q0 : order)
                        f(w.data(), rb, n, act, out, q0, std::min(q0 + item, rows_n));
                gbs[(size_t) i] = reps * (double) w.size() /
                                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            });
        for (auto& t : th)
            t.join();
        double s = 0;
        for (double g : gbs)
            s += g;
        static const char* names[] = {"v1", "v1 dy-table", "dy-table + permute scales", "no scale math (timing)",
                                      "no decode (timing)", "v1 no prefetch", "6: permute scales + srlv decode",
                                      "7: 6 with GPR-built scale words", "8: 7 + pre-broadcast activations", "9: 8 without prefetch", "10: 7 + lean decode (abs table, testn)", "11: 6 + lean decode"};
        std::printf("  %-36s %7.2f GB/s\n", names[v], s * 1e-9);
    }
}
