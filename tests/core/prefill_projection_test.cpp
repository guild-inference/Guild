#include "../check.hpp"
#include "guild/kernels/bf16_gemv.hpp"
#include "guild/kernels/bf16_bits.hpp"
#include "guild/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace guild::kernels;
template <class T> T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    CHECK(cudaMalloc(&p, v.size() * sizeof(T)) == cudaSuccess);
    CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    return p;
}
int main() {
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);
    std::mt19937 rng(78291);
    std::normal_distribution<float> random(0.f, .1f);
    const int widths[] = {1, 2, 3, 4, 7, 8, 9, 15, 17, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512};
    for (int n : {64, 512, 2048}) {
        std::vector<float> x(512 * n);
        for (auto& v : x)
            v = random(rng);
        auto* dx = upload(x);
        for (int outputs : {1, 257}) {
            std::vector<uint16_t> w(size_t(n) * outputs);
            for (auto& v : w)
                v = bf16_from_f32(random(rng));
            auto* dw = upload(w);
            std::vector<float> ref(512 * outputs), got(ref.size());
            auto* y = upload(got);
            for (int t = 0; t < 512; ++t)
                bf16_gemv_fp32_mmvf(dx + size_t(t) * n, dw, y + size_t(t) * outputs, n, outputs, stream);
            CHECK(cudaMemcpyAsync(ref.data(), y, ref.size() * 4, cudaMemcpyDeviceToHost, stream) == cudaSuccess);
            CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
            for (int count : widths) {
                CHECK(cudaMemsetAsync(y, 0xff, got.size() * 4, stream) == cudaSuccess);
                bf16_gemv_fp32_mmvf_cols(dx, dw, y, n, outputs, count, stream);
                CHECK(cudaMemcpyAsync(got.data(), y, size_t(count) * outputs * 4, cudaMemcpyDeviceToHost, stream) ==
                      cudaSuccess);
                CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
                CHECK(std::memcmp(ref.data(), got.data(), size_t(count) * outputs * 4) == 0);
            }
            CHECK(cudaFree(y) == cudaSuccess);
            CHECK(cudaFree(dw) == cudaSuccess);
        }
        const int heads = 3;
        std::vector<uint16_t> wa(n * heads), wb(n * heads);
        for (auto& v : wa)
            v = bf16_from_f32(random(rng));
        for (auto& v : wb)
            v = bf16_from_f32(random(rng));
        std::vector<float> dt(heads, .01f), a(heads, -.1f), ref(512 * heads * 2), got(ref.size());
        auto* dwa = upload(wa);
        auto* dwb = upload(wb);
        auto* ddt = upload(dt);
        auto* da = upload(a);
        auto* out = upload(got);
        for (int t = 0; t < 512; t += 8)
            gdn_ab_multi(dx + size_t(t) * n, dwa, dwb, ddt, da, out + size_t(t) * heads,
                         out + 512 * heads + size_t(t) * heads, n, heads, 8, stream);
        CHECK(cudaMemcpyAsync(ref.data(), out, ref.size() * 4, cudaMemcpyDeviceToHost, stream) == cudaSuccess);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        for (int count : widths) {
            CHECK(cudaMemsetAsync(out, 0xff, got.size() * 4, stream) == cudaSuccess);
            gdn_ab_prefill(dx, dwa, dwb, ddt, da, out, out + 512 * heads, n, heads, count, stream);
            CHECK(cudaMemcpyAsync(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost, stream) == cudaSuccess);
            CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
            CHECK(std::memcmp(ref.data(), got.data(), count * heads * 4) == 0);
            CHECK(std::memcmp(ref.data() + 512 * heads, got.data() + 512 * heads, count * heads * 4) == 0);
        }
        for (void* p : {static_cast<void*>(dwa), static_cast<void*>(dwb), static_cast<void*>(ddt),
                        static_cast<void*>(da), static_cast<void*>(out), static_cast<void*>(dx)})
            CHECK(cudaFree(p) == cudaSuccess);
    }
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::cout << "prefill_projection_test: BF16 and GDN A/B bitwise parity at 19 widths through 512 PASS\n";
}
