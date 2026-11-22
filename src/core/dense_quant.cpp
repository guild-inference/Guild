// src/core/dense_quant.cpp - see include/guild/core/dense_quant.hpp.
#include "guild/core/dense_quant.hpp"

#include "guild/artifact/gguf_reader.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

namespace guild::core {

Imatrix::Imatrix() = default;
Imatrix::~Imatrix() = default;
Imatrix::Imatrix(const std::string& path) : f_(std::make_unique<GgufFile>(path)) {
}

std::vector<float> Imatrix::weights(const std::string& name, int64_t n_in) const {
    if (!f_)
        return {};
    const TensorInfo* sums = f_->find(name + ".in_sum2");
    const TensorInfo* counts = f_->find(name + ".counts");
    // F32 (type 0) tensors: one sum a column, one count for the matrix
    if (!sums || !counts || sums->type != 0 || counts->type != 0 || sums->elements() != (uint64_t) n_in ||
        counts->elements() != 1)
        return {};
    float c = 0.0f;
    std::memcpy(&c, f_->tensor_data(*counts), 4);
    if (!(c > 0.0f))
        return {};
    std::vector<float> w((size_t) n_in);
    std::memcpy(w.data(), f_->tensor_data(*sums), (size_t) n_in * 4);
    for (float& v : w)
        v /= c;
    return w;
}

int dense_map_type(const std::string& map, const std::string& name) {
    // the kind: the name without "blk.N." and ".weight"
    const size_t dot = name.rfind("blk.", 0) == 0 ? name.find('.', 4) : std::string::npos;
    std::string kind = dot == std::string::npos ? name : name.substr(dot + 1);
    if (kind.size() > 7 && kind.compare(kind.size() - 7, 7, ".weight") == 0)
        kind.resize(kind.size() - 7);
    for (size_t at = 0; at < map.size();) {
        const size_t end = std::min(map.find(',', at), map.size()), eq = map.find('=', at);
        if (eq != std::string::npos && eq < end && map.compare(at, eq - at, kind) == 0) {
            const std::string v = map.substr(eq + 1, end - eq - 1);
            return v == "q6_k" ? 14 : v == "q5_k" ? 13 : v == "q4_k" ? 12 : v == "q8_0" ? 8 : 0;
        }
        at = end + 1;
    }
    return 0;
}

std::vector<uint8_t> q8_0_requant(const uint8_t* q8, int64_t n_in, int64_t n_out, int type, const float* imatrix) {
    const size_t q_row = ggml_row_size((ggml_type) type, n_in), q8_row = (size_t) (n_in / 32) * 34;
    std::vector<uint8_t> out(q_row * (size_t) n_out);
    constexpr int64_t kRows = 64;
    std::atomic<int64_t> next{0};
    auto work = [&] {
        std::vector<float> buf((size_t) kRows * n_in);
        for (;;) {
            const int64_t r0 = next.fetch_add(kRows);
            if (r0 >= n_out)
                return;
            const int64_t nr = std::min(kRows, n_out - r0);
            for (int64_t r = 0; r < nr; ++r) {
                const uint8_t* q = q8 + (size_t) (r0 + r) * q8_row;
                float* f = buf.data() + (size_t) r * n_in;
                for (int64_t b = 0; b < n_in / 32; ++b) {
                    ggml_fp16_t h;
                    std::memcpy(&h, q + b * 34, 2);
                    const float d = ggml_fp16_to_fp32(h);
                    for (int k = 0; k < 32; ++k)
                        f[b * 32 + k] = d * (float) (int8_t) q[b * 34 + 2 + k];
                }
            }
            ggml_quantize_chunk((ggml_type) type, buf.data(), out.data() + (size_t) r0 * q_row, 0, nr, n_in, imatrix);
        }
    };
    const unsigned n = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned i = 1; i < n; ++i)
        th.emplace_back(work);
    work();
    for (auto& t : th)
        t.join();
    return out;
}

} // namespace guild::core
