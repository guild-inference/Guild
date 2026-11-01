// include/guild/core/dense_mm.hpp - GGUF weight matrices on the GPU and their products with a window of tokens, for
// the engines that read their dense weights straight from a GGUF (GLM-5.3-Flash, Nemotron-H).
//
// The activations are quantized to Q8_1 (and, for wide windows, to llama.cpp's MMQ layout too); a product takes the
// staged one/two-token kernels where they apply (bitwise native_mmvq's), MMVQ up to kMmvqCols tokens, and MMQ beyond
// that for the matrices mark() admits.  Every column of an MMVQ product is bitwise a one-column product, so a token's
// result does not depend on the window (native_mmvq_set_multi_exact's default).
#pragma once

#include "guild/core/glm5_kernels.hpp"
#include "guild/kernels/native_mmvq.hpp"
#if defined(GUILD_GLM_MMQ)
#include "guild/prefill/moe_mmq.hpp"
#endif

#include "ggml.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace guild::core {

/// A weight matrix in its GGUF form: n_out rows of n_in values
struct GgufMat {
    int type = -1;
    int64_t n_in = 0, n_out = 0;
    const uint8_t* host = nullptr;
    void* dev = nullptr;
    size_t bytes = 0;
    std::vector<uint8_t> owned; ///< host bytes of a re-quantized copy (until uploaded)
    bool mmq = false;           ///< prompt windows multiply it with llama.cpp's MMQ (int8 tiles) instead of MMVQ
    size_t row_bytes() const {
        return ggml_row_size((ggml_type) type, n_in);
    }
};

class DenseProducts {
public:
    // MMVQ reads the weights once for up to 8 columns; beyond that each column costs about as much again (GTX 1070:
    // 0.20 ms for one column of a 4096 x 8192 Q8_0 matrix, 0.31 for 8, 2.75 for 64), so wider windows use MMQ
    static constexpr int kMmvqCols = 8;
    static constexpr int kQ8_0 = 8, kQ6_K = 14;

    /// Two activation buffers (xq[0], xq[1]) for windows of up to max_t tokens of up to `widest` values; with
    /// `allow_mmq` (and MMQ built) their MMQ twins and the launch tables.  alloc: the engine's device allocator.
    bool init(cudaStream_t st, int64_t widest, int max_t, bool allow_mmq, const std::function<void*(size_t)>& alloc,
              std::string& err) {
        st_ = st;
        const size_t qbytes = guild::kernels::native_q8_1_bytes((int) widest, max_t);
        if (!(xq[0] = alloc(qbytes)) || !(xq[1] = alloc(qbytes))) {
            err = "out of GPU memory for the activations";
            return false;
        }
#if defined(GUILD_GLM_MMQ)
        namespace mmq = guild::prefill::mmq;
        if (allow_mmq && mmq::built()) {
            const size_t mb = mmq::q8_bytes(max_t, widest);
            std::vector<int32_t> bounds((size_t) 2 * (max_t + 1));
            for (int t = 0; t <= max_t; ++t) {
                bounds[(size_t) 2 * t] = 0;
                bounds[(size_t) 2 * t + 1] = t;
            }
            if (!(xqm_[0] = alloc(mb)) || !(xqm_[1] = alloc(mb)) || !(bounds_ = (int32_t*) alloc(bounds.size() * 4)) ||
                !(ids_ = (int32_t*) alloc((size_t) max_t * 4))) {
                err = "out of GPU memory for the MMQ activations";
                return false;
            }
            cudaMemcpy(bounds_, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice);
            mmq::iota(ids_, max_t, st);
            ctx_ = std::make_unique<mmq::Context>();
        }
#else
        (void) allow_mmq;
#endif
        return true;
    }
    bool mmq_on() const {
#if defined(GUILD_GLM_MMQ)
        return ctx_ != nullptr;
#else
        return false;
#endif
    }
    /// Admits m to MMQ: on the GPU, Q8_0 (or Q6_K when the build has its MMQ), rows a multiple of 256 values (MMQ
    /// steps through a row 256 values at a time: a narrower one reads past it) and a tile that fits.  Q6_K takes the
    /// activations Q8_0 does: llama.cpp quantizes both to q8_1 in the same D4 layout.
    bool mark(GgufMat& m) const {
#if defined(GUILD_GLM_MMQ)
        namespace mmq = guild::prefill::mmq;
        const bool type_ok = m.type == kQ8_0 || (m.type == kQ6_K && mmq::supported(kQ6_K));
        m.mmq = ctx_ && m.dev && type_ok && m.n_in % 256 == 0 && mmq::fits(m.type, m.n_out);
#else
        m.mmq = false;
#endif
        return m.mmq;
    }

    /// x [T][n] -> dst (xq[0] or xq[1]), and its MMQ twin for a wide window
    void quant(const float* x, int64_t n, int T, void* dst) {
        guild::kernels::native_quantize_q8_1(x, dst, (int) n, T, st_);
#if defined(GUILD_GLM_MMQ)
        if (T > kMmvqCols && ctx_)
            guild::prefill::mmq::quantize(x, nullptr, dst == xq[0] ? xqm_[0] : xqm_[1], kQ8_0, n, n, T, st_);
#endif
    }
    /// out [T][m.n_out] = the window xq_ (xq[0] or xq[1], quantized by quant()) times m's rows; returns the path taken
    const char* mm(const GgufMat& m, const void* xq_, float* out, int T) {
#if defined(GUILD_GLM_MMQ)
        if (T > kMmvqCols && m.mmq && ctx_ && (xq_ == xq[0] || xq_ == xq[1])) {
            namespace mmq = guild::prefill::mmq;
            mmq::Product p;
            p.w = m.dev;
            p.type = m.type;
            p.w_rows = m.n_out;
            p.w_cols = m.n_in;
            p.expert_bytes = mmq::matrix_bytes(m.type, m.n_out, m.n_in);
            p.n = 1;
            p.xq = xq_ == xq[0] ? xqm_[0] : xqm_[1];
            p.bounds = bounds_ + 2 * T;
            p.ids = ids_;
            p.total_rows = T;
            p.max_rows = T;
            p.dst = out;
            p.ld_dst = m.n_out;
            ctx_->run(p, st_);
            return "mmq";
        }
#endif
        // one or two tokens of Q6_K / Q8_0: the staged kernels (bitwise native_mmvq's, fewer load instructions)
        if (m.type == kQ6_K && glm5k::q6k_mmvq_supported((int) m.n_in, (int) m.n_out, T))
            glm5k::q6k_mmvq(m.dev, xq_, out, (int) m.n_in, (int) m.n_out, T, st_);
        else if (m.type == kQ8_0 && glm5k::q8_0_mmvq_supported((int) m.n_in, (int) m.n_out, T))
            glm5k::q8_0_mmvq(m.dev, xq_, out, (int) m.n_in, (int) m.n_out, T, st_);
        else if (m.type == kQ8_0 && glm5k::q8_0_narrow_supported((int) m.n_in, T))
            glm5k::q8_0_narrow(m.dev, xq_, out, (int) m.n_in, (int) m.n_out, T, st_);
        else
            guild::kernels::native_mmvq(m.type, m.dev, xq_, out, (int) m.n_in, (int) m.n_out, T, st_);
        return "mmvq";
    }

    void* xq[2] = {nullptr, nullptr}; ///< the activations (Q8_1, MMVQ's layout)

private:
    cudaStream_t st_ = nullptr;
    void* xqm_[2] = {nullptr, nullptr};
    int32_t *bounds_ = nullptr, *ids_ = nullptr;
#if defined(GUILD_GLM_MMQ)
    std::unique_ptr<guild::prefill::mmq::Context> ctx_;
#endif
};

} // namespace guild::core
