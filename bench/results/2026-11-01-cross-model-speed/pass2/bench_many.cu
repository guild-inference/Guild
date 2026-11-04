// bench_many.cu - native_mmvq with 128 columns on the prefill's projection shapes: microseconds per call, and every
// output against single-column calls bit for bit.  GUILD_MMVQ_MANY_XR picks the kernel (0: the many kernels).
#include "guild/kernels/native_mmvq.hpp"
#include "guild/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    using namespace guild::kernels;
    struct Shape { const char* name; int type, n_in, n_out, block_elems, block_bytes; };
    const Shape shapes[] = {
        {"Q8_0 2048x8192", 8, 2048, 8192, 32, 34}, {"Q8_0 2048x4096", 8, 2048, 4096, 32, 34},
        {"Q8_0 4096x2048", 8, 4096, 2048, 32, 34}, {"Q8_0 2048x512", 8, 2048, 512, 32, 34},
        {"Q4_K 2048x8192", 12, 2048, 8192, 256, 144}, {"Q4_K 4096x2048", 12, 4096, 2048, 256, 144},
        {"Q6_K 2048x8192", 14, 2048, 8192, 256, 210}, {"Q6_K 2048x512", 14, 2048, 512, 256, 210},
    };
    const int T = 128;
    cudaStream_t s;
    cudaStreamCreate(&s);
    std::mt19937 rng(1);
    for (const Shape& c : shapes) {
        const size_t wbytes = (size_t) c.n_out * (c.n_in / c.block_elems) * c.block_bytes;
        std::vector<uint8_t> w(wbytes);
        for (auto& b : w) b = (uint8_t) (rng() & 0xff);
        // finite fp16 scales: Q8_0 d at 0, Q4_K dm at 0, Q6_K d at 208
        const int at = c.type == 14 ? 208 : 0;
        for (size_t k = 0; k < wbytes / c.block_bytes; ++k) {
            const uint16_t h = (uint16_t) (0x1800 + (rng() & 0x3ff)) | (uint16_t) ((rng() & 1) << 15);
            std::memcpy(&w[k * c.block_bytes + at], &h, 2);
            if (c.type == 12) std::memcpy(&w[k * c.block_bytes + 2], &h, 2);
        }
        std::vector<float> x((size_t) T * c.n_in);
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng);
        void *dw, *xq;
        float *dx, *ym, *ys;
        const size_t qcol = (size_t) c.n_in / 32 * 36;
        cudaMalloc(&dw, wbytes);
        cudaMalloc(&xq, qcol * T);
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&ym, (size_t) T * c.n_out * 4);
        cudaMalloc(&ys, (size_t) T * c.n_out * 4);
        cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        quantize_q8_1_rows(dx, T, c.n_in, xq, s);
        native_mmvq(c.type, dw, xq, ym, c.n_in, c.n_out, T, s);
        for (int j = 0; j < T; ++j)
            native_mmvq(c.type, dw, (const uint8_t*) xq + (size_t) j * qcol, ys + (size_t) j * c.n_out, c.n_in, c.n_out, 1, s);
        cudaStreamSynchronize(s);
        std::vector<uint32_t> a((size_t) T * c.n_out), b(a.size());
        cudaMemcpy(a.data(), ym, a.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), ys, b.size() * 4, cudaMemcpyDeviceToHost);
        long diff = 0;
        for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int reps = 20;
        cudaEventRecord(e0, s);
        for (int r = 0; r < reps; ++r) native_mmvq(c.type, dw, xq, ym, c.n_in, c.n_out, T, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1000.0 * ms / reps;
        std::printf("%-16s T=%d: %8.1f us  %6.2f TMAC/s  %ld of %zu outputs differ from one-column calls (%s)\n", c.name, T,
                    us, 1e-6 * (double) c.n_in * c.n_out * T / us, diff, a.size(), cudaGetErrorString(cudaGetLastError()));
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(ym); cudaFree(ys);
    }
    return 0;
}
