// src/kernels/mmvf_rows_parity.cpp - the FP32-activation BF16 products with R rows a block against the one-row
// kernels, bitwise.
//
//     GUILD_MMVF_ROWS=0 build/mmvf_rows_parity --dump ref.bin   # the one-row kernels
//     build/mmvf_rows_parity --compare ref.bin                  # the default (rows) kernels, compared
//     GUILD_MMVF_ROWS=<0|1|2|4|8> build/mmvf_rows_parity --bench   # per-call times of the models' shapes
//
// WHY IT EXISTS.  bf16_gemv_fp32_mmvf and bf16_gemv_fp32_mmvf_multi (the routers, 2560 -> 512 for up to a verify
// group's tokens) run a kernel with R output rows a block when n_in is 2048, 2560 or 4096 and there are at least 64
// rows (native_bf16.cu, bf16_f32_mmvf_rows_kernel), and say every output is bitwise the one-row kernel's.  The
// switch that turns them off is an environment variable read once per process, so the two runs are two processes,
// joined by a file (registered as two tests with a fixture).  Each run also checks its own multi-column calls
// against its single-column calls.
//
// THE CASES.  The routers' shape (2560 -> 512) and the other row counts the kernel takes (partial last blocks of
// 1, 2, 4 and 8 rows, 64 rows), the three pair counts (n_in 2048, 2560, 4096), shapes that stay on the one-row
// kernels (one row, n_in 1536), padded activation and output strides; every column count from 1 to 8.  The data is
// random, with zeros of both signs and subnormals in the weights and the activations; the outputs' padding is a NaN
// pattern that must survive.
#include "guild/kernels/bf16_gemv.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    int n_in, n_out, pad_x, pad_y;
};

const Case CASES[] = {
    {2560, 512, 0, 0}, {2560, 512, 2, 3}, {2560, 2560, 0, 0}, {2560, 128, 0, 0}, {2560, 64, 0, 0},  {2560, 513, 0, 1},
    {2560, 515, 4, 0}, {2560, 517, 0, 0}, {2560, 519, 0, 0},  {2048, 256, 0, 0}, {2048, 509, 0, 5}, {4096, 67, 2, 0},
    {4096, 512, 0, 0}, {2560, 1, 0, 0},   {2560, 63, 0, 0},   {1536, 100, 0, 0},
};
constexpr int MAX_T = 8;
constexpr uint32_t SENTINEL = 0x7fc0beefu;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

uint16_t bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    return (uint16_t) ((i + ((i >> 16) & 1u) + 0x7FFFu) >> 16);
}

float f32(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// one case's outputs: for each column count T = 1..8, the multi call's [T][ldy] buffer, then (T = 1..8) the T single
// calls' buffer
std::vector<uint32_t> run_case(const Case& c, int index, bool& self_ok) {
    std::mt19937 rng(1234u + (uint32_t) index);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t ldx = c.n_in + c.pad_x, ldy = c.n_out + c.pad_y;
    std::vector<uint16_t> w((size_t) c.n_out * c.n_in);
    for (auto& v : w) {
        const uint32_t r = rng() % 1000u;
        v = r == 0 ? 0x8000u : r == 1 ? 0x0000u : r == 2 ? (uint16_t) (0x0001u + rng() % 0x7fu) : bf16(nd(rng) * 0.05f);
    }
    std::vector<float> x((size_t) MAX_T * ldx);
    for (auto& v : x) {
        const uint32_t r = rng() % 1000u;
        v = r == 0 ? -0.0f : r == 1 ? 0.0f : r == 2 ? f32(0x00000001u + rng() % 0x7fffffu) : nd(rng);
    }
    uint16_t* dw;
    float *dx, *dy;
    const size_t ybytes = (size_t) MAX_T * ldy * 4;
    ck(cudaMalloc(&dw, w.size() * 2), "malloc w");
    ck(cudaMalloc(&dx, x.size() * 4), "malloc x");
    ck(cudaMalloc(&dy, ybytes), "malloc y");
    ck(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "upload w");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "upload x");
    std::vector<uint32_t> fill((size_t) MAX_T * ldy, SENTINEL), out;
    std::vector<uint32_t> multi((size_t) MAX_T * ldy), single((size_t) MAX_T * ldy);
    for (int T = 1; T <= MAX_T; ++T) {
        ck(cudaMemcpy(dy, fill.data(), ybytes, cudaMemcpyHostToDevice), "fill y");
        guild::kernels::bf16_gemv_fp32_mmvf_multi(dx, ldx, dw, dy, ldy, c.n_in, c.n_out, T, nullptr);
        ck(cudaDeviceSynchronize(), "multi");
        ck(cudaMemcpy(multi.data(), dy, ybytes, cudaMemcpyDeviceToHost), "download multi");
        ck(cudaMemcpy(dy, fill.data(), ybytes, cudaMemcpyHostToDevice), "fill y");
        for (int t = 0; t < T; ++t)
            guild::kernels::bf16_gemv_fp32_mmvf(dx + (size_t) t * ldx, dw, dy + (size_t) t * ldy, c.n_in, c.n_out,
                                                nullptr);
        ck(cudaDeviceSynchronize(), "single");
        ck(cudaMemcpy(single.data(), dy, ybytes, cudaMemcpyDeviceToHost), "download single");
        if (multi != single) {
            std::printf("  n_in %d n_out %d T %d: multi-column != single-column\n", c.n_in, c.n_out, T);
            self_ok = false;
        }
        for (int t = 0; t < MAX_T; ++t)
            for (int64_t j = 0; j < ldy; ++j) {
                const uint32_t v = multi[(size_t) t * ldy + j];
                const bool written = t < T && j < c.n_out;
                if (written == (v == SENTINEL)) {
                    std::printf("  n_in %d n_out %d T %d: output %d,%lld %s\n", c.n_in, c.n_out, T, t, (long long) j,
                                written ? "not written" : "written outside the outputs");
                    self_ok = false;
                    t = MAX_T;
                    break;
                }
            }
        out.insert(out.end(), multi.begin(), multi.end());
    }
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dy);
    return out;
}

int bench() {
    const char* rows = std::getenv("GUILD_MMVF_ROWS");
    std::printf("GUILD_MMVF_ROWS=%s\n", rows ? rows : "(default)");
    struct B {
        int n_in, n_out;
    };
    for (const B b : {B{2560, 512}, B{2560, 128}, B{2560, 2560}, B{2048, 256}}) {
        // 8 copies of the weight, used in turn, so a call does not find the last one's rows in the 2 MB L2
        const int copies = 8;
        uint16_t* dw;
        float *dx, *dy;
        const size_t wn = (size_t) b.n_out * b.n_in;
        ck(cudaMalloc(&dw, wn * 2 * copies), "malloc w");
        ck(cudaMalloc(&dx, (size_t) MAX_T * b.n_in * 4), "malloc x");
        ck(cudaMalloc(&dy, (size_t) MAX_T * b.n_out * 4), "malloc y");
        ck(cudaMemset(dw, 0x3c, wn * 2 * copies), "set w");
        ck(cudaMemset(dx, 0x3c, (size_t) MAX_T * b.n_in * 4), "set x");
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int T : {1, 2, 4}) {
            const int reps = 400;
            for (int i = 0; i < 20; ++i)
                guild::kernels::bf16_gemv_fp32_mmvf_multi(dx, b.n_in, dw + (i % copies) * wn, dy, b.n_out, b.n_in,
                                                          b.n_out, T, s);
            cudaEventRecord(e0, s);
            for (int i = 0; i < reps; ++i)
                guild::kernels::bf16_gemv_fp32_mmvf_multi(dx, b.n_in, dw + (i % copies) * wn, dy, b.n_out, b.n_in,
                                                          b.n_out, T, s);
            cudaEventRecord(e1, s);
            ck(cudaEventSynchronize(e1), "bench");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = ms * 1000.0 / reps;
            std::printf("  %4d x %4d  T %d: %7.2f us a call, %6.1f GB/s of weights\n", b.n_out, b.n_in, T, us,
                        (double) wn * 2 / us / 1e3);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaStreamDestroy(s);
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dy);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--bench")
        return bench();
    if ((mode != "--dump" && mode != "--compare") || argc < 3) {
        std::printf("usage: %s --dump FILE | --compare FILE | --bench (see the file's header)\n", argv[0]);
        return 2;
    }
    bool self_ok = true;
    std::vector<std::vector<uint32_t>> outs;
    for (size_t i = 0; i < sizeof CASES / sizeof CASES[0]; ++i)
        outs.push_back(run_case(CASES[i], (int) i, self_ok));
    if (mode == "--dump") {
        FILE* f = std::fopen(argv[2], "wb");
        if (!f) {
            std::printf("cannot write %s\n", argv[2]);
            return 2;
        }
        for (const auto& o : outs)
            std::fwrite(o.data(), 4, o.size(), f);
        std::fclose(f);
        std::printf(self_ok ? "mmvf_rows_parity: reference written (%zu cases)\n"
                            : "mmvf_rows_parity: reference written (%zu cases), multi != single: FAILED\n",
                    outs.size());
        return self_ok ? 0 : 1;
    }
    FILE* f = std::fopen(argv[2], "rb");
    if (!f) {
        std::printf("cannot read %s (run --dump first)\n", argv[2]);
        return 2;
    }
    int bad = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
        std::vector<uint32_t> ref(outs[i].size());
        const bool read = std::fread(ref.data(), 4, ref.size(), f) == ref.size();
        const bool same = read && ref == outs[i];
        std::printf("  %4d -> %4d (x pad %d, y pad %d), T 1..8: %s\n", CASES[i].n_in, CASES[i].n_out, CASES[i].pad_x,
                    CASES[i].pad_y, same ? "bit-identical" : "*** DIFFERS ***");
        bad += same ? 0 : 1;
    }
    std::fclose(f);
    if (!self_ok)
        ++bad;
    std::printf(bad ? "mmvf_rows_parity: %d FAILURES\n" : "mmvf_rows_parity OK\n", bad);
    return bad ? 1 : 0;
}
