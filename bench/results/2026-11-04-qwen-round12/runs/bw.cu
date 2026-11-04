#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>
// pinned (cudaHostRegister of malloc'd, first-touched here) H2D: one 1 GiB buffer in 2.4 MB copies, the way the
// prompt path streams expert blobs
int main(int argc, char** argv) {
    const size_t n = 1ull << 30, blk = argc > 2 ? (size_t) atoll(argv[2]) : 2457600;
    char* h = (char*) aligned_alloc(4096, n);
    memset(h, 1, n);
    cudaHostRegister(h, n, cudaHostRegisterDefault);
    void* d; cudaMalloc(&d, 16 * blk);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int rep = 0; rep < 3; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t o = 0; o + blk <= n; o += blk)
            cudaMemcpyAsync((char*) d + (o / blk % 16) * blk, h + o, blk, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("%s: %.2f GB/s\n", argc > 1 ? argv[1] : "", (n / blk * blk) / sec / 1e9);
    }
}
