// include/guild/core/dense_quant.hpp - what the load-time re-quantization of dense matrices reads besides the matrix:
// an importance matrix and a per-kind type map (Qwen's --dense / NativeDense, GLM-5.3-Flash's Glm5Dense).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace guild {
class GgufFile;
}

namespace guild::core {

/// llama.cpp's importance matrix in its GGUF form (llama-imatrix --output-format gguf; Unsloth publishes the ones its
/// quants were made with): per tensor `<name>.in_sum2` and `<name>.counts`.  `weights` is in_sum2 / counts, each
/// input column's mean squared activation: the quant_weights llama-quantize --imatrix passes to ggml_quantize_chunk.
class Imatrix {
public:
    Imatrix();
    ~Imatrix();
    /// throws std::runtime_error on an unreadable file
    explicit Imatrix(const std::string& path);
    /// the weights of matrix `name` ([n_in]); empty when the file does not list it (or lists another width)
    std::vector<float> weights(const std::string& name, int64_t n_in) const;

private:
    std::unique_ptr<GgufFile> f_;
};

/// The ggml type a kind=type,... map ("ssm_out=q5_k,attn_output=q8_0") gives tensor `name` ("blk.N.<kind>.weight"):
/// 14 q6_k, 13 q5_k, 12 q4_k, 8 q8_0; 0 when the map does not name its kind.
int dense_map_type(const std::string& map, const std::string& name);

} // namespace guild::core
