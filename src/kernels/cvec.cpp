// Host control-vector storage; the arithmetic is dispatched through CDM1.
#include "guild/kernels/cvec.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace guild::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16; // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec; // the description; its device pointers are the uploading device's
bool g_on_host = false;
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables {
    float* dir = nullptr;
    float* s = nullptr;
    int* on = nullptr;
};
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices)
        d = 0;
    return d;
}
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr)
        return true;
    const int flag = g_on_host ? 1 : 0;
    if (cudaMalloc(&t.dir, g_dir_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&t.s, g_s_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&t.on, sizeof(int)) != cudaSuccess ||
        cudaMemcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.on, &flag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        if (t.dir)
            cudaFree(t.dir);
        if (t.s)
            cudaFree(t.s);
        if (t.on)
            cudaFree(t.on);
        err = "control vector: device allocation failed";
        t = DevTables{};
        return false;
    }
    return true;
}

} // namespace

const Cvec& cvec() {
    return g_cvec;
}

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) {
        err = "control vector: unsupported n_embd";
        return false;
    }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) {
        err = "control vector: bad table sizes";
        return false;
    }
    int prev = 0; // a new vector replaces the old one on every device
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr)
            continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaFree(g_dev[d].dir);
        cudaFree(g_dev[d].s);
        cudaFree(g_dev[d].on);
        g_dev[d] = DevTables{};
    }
    cudaSetDevice(prev);
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err))
        return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l)
        g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) {
    return !g_cvec.loaded() || upload_here(err);
}

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host)
        return;
    int prev = 0;
    cudaGetDevice(&prev);
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr)
            continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize(); // nothing in flight may still read the flag
        cudaMemcpy(g_dev[d].on, &v, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prev);
    g_on_host = on;
}

bool cvec_enabled() {
    return g_cvec.loaded() && g_on_host;
}

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1)
        return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr)
        throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    cvec_launch_kernel(R, t.dir, t.s, t.on, g_cvec.mode, layer, T, g_cvec.n_embd, g_cvec.hc, r_ld, bo, bo_ld, inj,
                       inj_ld, write, stream);
}

} // namespace guild::kernels
