#include "../check.hpp"
#include "guild/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    unsigned long long *status, *host;
    CHECK(cudaMalloc(&status, sizeof(*status)) == cudaSuccess);
    CHECK(cudaMallocHost(&host, sizeof(*host)) == cudaSuccess);
    for (size_t count : {size_t(1), size_t(63), size_t(64), size_t(65), size_t(255), size_t(257),
                         size_t(248320), size_t(8 * 248320)}) {
        std::vector<float> values(count, std::numeric_limits<float>::max());
        values[0] = -0.0f;
        values[count / 2] = std::numeric_limits<float>::denorm_min();
        float* device;
        CHECK(cudaMalloc(&device, count * sizeof(float)) == cudaSuccess);
        CHECK(cudaMemcpyAsync(device, values.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
        guild::kernels::finite_first_bad(device, count, status, stream);
        CHECK(cudaMemcpyAsync(host, status, sizeof(*host), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
        CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
        CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess);
        auto check = [&](unsigned long long expected) {
            CHECK(cudaMemcpyAsync(device, values.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess);
            CHECK(cudaGraphLaunch(exec, stream) == cudaSuccess);
            CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
            CHECK(*host == expected);
        };
        check(~0ull);
        // The last, potentially unselected logit must be checked, even in row eight.
        for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
            values.back() = bad;
            check(count - 1);
            values[0] = bad;
            check(0); // first-index reporting is deterministic with multiple bad blocks
            values[0] = 0;
            values.back() = 0;
            check(~0ull); // a replay clears the previous error
        }
        const uint32_t signaling_nan = 0x7f800001;
        std::memcpy(&values.back(), &signaling_nan, sizeof(signaling_nan));
        check(count - 1);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaFree(device);
    }
    cudaFreeHost(host);
    cudaFree(status);
    cudaStreamDestroy(stream);
    std::puts("device_finite_test: all vocabulary rows, NaN/Inf, first index and graph replay PASS");
}
