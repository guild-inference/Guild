// src/kernels/mmvq_staged_parity.cpp - native_mmvq's staged rows against its other kernels, bitwise.
//
//     GUILD_MMVQ_STAGED=0 build/mmvq_staged_parity --dump ref.bin   # the kernels without the staged rows
//     build/mmvq_staged_parity --compare ref.bin                    # the default (staged) kernels, compared
//
// WHY IT EXISTS.  On CUDA, native_mmvq computes Q4_K, Q6_K and Q8_0 with a warp per row read through shared memory
// (native_mmvq.cu, "Staged rows"), and says each output is bitwise the ncols = 1 kernel's at every column count.
// mmvq_multi_parity cannot see that: it compares T columns against T single columns, and both now take the staged
// kernel.  The switch that turns the staged kernel off is an environment variable read once per process, so the
// two runs are two processes, joined by a file (registered as two tests with a fixture).
//
// THE CASES.  Rows of one piece, of a partial last piece of blocks (n_in not a multiple of the 8 or 32 blocks a
// warp stages), rows that start inside a 16-byte piece (Q6_K and Q8_0 rows of 2100 and 340 bytes), matrices whose
// end is not 16-byte aligned (the last piece read in 2-byte parts), fewer rows than a block's 8 warps, Q6_K rows of
// 9 to 12 blocks (staged as one piece a row) and of 13, and the shapes of the models' products; every column count
// from 1 to 8 on the same Q8_1 bytes.  The data is random with
// finite fp16 scales (mmvq_multi_parity's THE DATA), and a non-finite output fails the test.
#include "guild/kernels/iq_kernels.hpp"
#include "guild/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    int type; // stable GGML type id
    int n_in, n_out;
    int block_elems, block_bytes;
    int scale_at[2]; // byte offsets of the block's fp16 scales; -1 = none
};

const Case CASES[] = {
    {"Q4_K 512 (2 blocks)", 12, 512, 512, 256, 144, {0, 2}},
    {"Q4_K 2048", 12, 2048, 1000, 256, 144, {0, 2}},
    {"Q4_K 2304 (9 blocks)", 12, 2304, 7, 256, 144, {0, 2}},
    {"Q4_K 4096", 12, 4096, 2048, 256, 144, {0, 2}},
    {"Q6_K 512", 14, 512, 13, 256, 210, {208, -1}},
    {"Q6_K 2048", 14, 2048, 512, 256, 210, {208, -1}},
    {"Q6_K 2304 (9 blocks)", 14, 2304, 1003, 256, 210, {208, -1}},
    {"Q6_K 2560 (2100 B rows)", 14, 2560, 1001, 256, 210, {208, -1}},
    {"Q6_K 2560 x 4099", 14, 2560, 4099, 256, 210, {208, -1}},
    {"Q6_K 3072 (12 blocks)", 14, 3072, 777, 256, 210, {208, -1}},
    {"Q6_K 3328 (13 blocks)", 14, 3328, 700, 256, 210, {208, -1}},
    {"Q6_K 4352 (17 blocks)", 14, 4352, 64, 256, 210, {208, -1}},
    {"Q8_0 320 (340 B rows)", 8, 320, 999, 32, 34, {0, -1}},
    {"Q8_0 640", 8, 640, 2560, 32, 34, {0, -1}},
    {"Q8_0 2560", 8, 2560, 517, 32, 34, {0, -1}},
    {"Q8_0 6144", 8, 6144, 256, 32, 34, {0, -1}},
    {"Q8_0 10240", 8, 10240, 9, 32, 34, {0, -1}},
};
constexpr int MAX_T = 8;

uint16_t sane_half(std::mt19937& rng) { // a normal fp16 in +-[2^-10, 2^-5)
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess)
        return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// Every case at every column count 1..MAX_T, outputs appended to `out` as raw bits.
bool run_all(std::vector<uint32_t>& out) {
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream"))
        return false;
    for (const Case& c : CASES) {
        const std::size_t wbytes = guild::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
        std::mt19937 rng(77u + (unsigned) c.n_in * 3u + (unsigned) c.n_out);
        std::vector<uint8_t> w(wbytes);
        for (auto& b : w)
            b = (uint8_t) (rng() & 0xff);
        const std::size_t n_blocks = wbytes / (std::size_t) c.block_bytes;
        for (std::size_t k = 0; k < n_blocks; ++k)
            for (int at : c.scale_at)
                if (at >= 0) {
                    const uint16_t h = sane_half(rng);
                    std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
                }
        std::vector<float> x((std::size_t) MAX_T * c.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x)
            v = nd(rng);
        void *dw = nullptr, *xq = nullptr;
        float *dx = nullptr, *dy = nullptr;
        // the weights in an allocation of exactly their size: a read past the matrix would be past the allocation
        if (!ck(cudaMalloc(&dw, wbytes), "malloc w") ||
            !ck(cudaMalloc(&xq, guild::kernels::native_q8_1_bytes(c.n_in, MAX_T)), "malloc xq") ||
            !ck(cudaMalloc(&dx, x.size() * 4), "malloc x") ||
            !ck(cudaMalloc(&dy, (std::size_t) MAX_T * c.n_out * 4), "malloc y") ||
            !ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w") ||
            !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
            return false;
        guild::kernels::quantize_q8_1_rows(dx, MAX_T, c.n_in, xq, s);
        for (int T = 1; T <= MAX_T; ++T) {
            if (!ck(cudaMemset(dy, 0xff, (std::size_t) T * c.n_out * 4), "clear y"))
                return false;
            try {
                guild::kernels::native_mmvq(c.type, dw, xq, dy, c.n_in, c.n_out, T, s);
            } catch (const std::exception& e) {
                std::printf("%s T=%d: %s\n", c.name, T, e.what());
                return false;
            }
            const std::size_t at = out.size();
            out.resize(at + (std::size_t) T * c.n_out);
            if (!ck(cudaStreamSynchronize(s), "mmvq") ||
                !ck(cudaMemcpy(out.data() + at, dy, (std::size_t) T * c.n_out * 4, cudaMemcpyDeviceToHost), "read y"))
                return false;
        }
        cudaFree(dw);
        cudaFree(xq);
        cudaFree(dx);
        cudaFree(dy);
    }
    cudaStreamDestroy(s);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 2 ? argv[1] : "";
    if (mode != "--dump" && mode != "--compare") {
        std::printf("usage: %s --dump FILE | --compare FILE (see the file's header)\n", argv[0]);
        return 2;
    }
    std::vector<uint32_t> got;
    if (!run_all(got))
        return 1;
    long long nonfinite = 0;
    for (uint32_t b : got) {
        float f;
        std::memcpy(&f, &b, 4);
        nonfinite += !std::isfinite(f);
    }
    if (nonfinite) {
        std::printf("mmvq_staged_parity: FAILED, %lld of %zu outputs are not finite\n", nonfinite, got.size());
        return 1;
    }
    if (mode == "--dump") {
        FILE* f = std::fopen(argv[2], "wb");
        if (!f || std::fwrite(got.data(), 4, got.size(), f) != got.size() || std::fclose(f) != 0) {
            std::printf("mmvq_staged_parity: cannot write %s\n", argv[2]);
            return 1;
        }
        std::printf("mmvq_staged_parity: %zu outputs written to %s\n", got.size(), argv[2]);
        return 0;
    }
    std::vector<uint32_t> ref(got.size());
    FILE* f = std::fopen(argv[2], "rb");
    const std::size_t n = f ? std::fread(ref.data(), 4, ref.size() + 1, f) : 0;
    if (f)
        std::fclose(f);
    if (n != got.size()) {
        std::printf("mmvq_staged_parity: %s holds %zu outputs, this run has %zu\n", argv[2], n, got.size());
        return 1;
    }
    // per case and column count, so a failure says where
    std::size_t at = 0;
    long long diff = 0;
    for (const Case& c : CASES)
        for (int T = 1; T <= MAX_T; ++T) {
            long long d = 0;
            for (std::size_t i = 0; i < (std::size_t) T * c.n_out; ++i)
                d += got[at + i] != ref[at + i];
            if (d)
                std::printf("%-26s T=%d: %lld of %d outputs differ\n", c.name, T, d, T * c.n_out);
            diff += d;
            at += (std::size_t) T * c.n_out;
        }
    std::printf("mmvq_staged_parity: %s, %zu outputs compared, %lld differ bitwise\n", diff ? "FAILED" : "ok",
                got.size(), diff);
    return diff ? 1 : 0;
}
