// src/core/dense_quant.cpp - see include/guild/core/dense_quant.hpp.
#include "guild/core/dense_quant.hpp"

#include "guild/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstring>

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

} // namespace guild::core
