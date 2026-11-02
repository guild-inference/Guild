#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace guild::artifact {
class GPackReader;
}

namespace guild::core {
class WeightTable;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// With layer_hi >= 0, only the `blk.<l>.` matrices with layer_lo <= l < layer_hi are uploaded (a layer
    /// split's stage holds its own layers' projections, not the whole model's); the others keep data == nullptr.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t layer_lo = 0, int64_t layer_hi = -1);
    bool load_gpack(const guild::artifact::GPackReader& gpack, WeightTable& table, std::string& err);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key, std::set<std::string>& out,
                             std::string& err);
    /// Layer split: load only blocks [lb, le) (every other `blk.N.` projection belongs to another GPU's stage; the
    /// PLE tensors are loaded everywhere).  Process-wide, read by the next `load`; (-1, -1) = all layers.
    static void set_layer_range(int lb, int le);
    /// Q8_0 matrices of 2048 or more rows (rows of whole 256-value blocks) uploaded as Q6_K (--dense q6_k; also
    /// GUILD_DENSE=q6_k).  Process-wide, read by the next `load`.
    static void set_q6k(bool on);
    /// --hc q8_0 (also GUILD_HC=q8_0): every layer's hyper-connection down and up matrix (`blk.N.hc_{attn,ffn}_
    /// {down,up}.weight`) from the GGUF's own Q8_0 blocks instead of the pack's bf16 rounding of them: `served_names`
    /// adds them to the arena's skip set, `load` uploads them (WeightRef::hc_q8).  The caller checks that the CUDA
    /// module reads them (fused_gr_q8_supported).  Process-wide; set before `served_names`.
    static void set_hc_q8(bool on);
    static bool hc_q8();
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    uint64_t weight_bytes() const {
        return bytes_;
    }
    size_t tensor_count() const {
        return weights_.size();
    }
    /// matrices uploaded as Q6_K out of Q8_0 (GUILD_DENSE=q6_k)
    size_t requantized() const {
        return requantized_;
    }
    /// hyper-connection matrices uploaded as the GGUF's Q8_0 (--hc q8_0), and their bytes (part of weight_bytes)
    size_t hc_q8_count() const {
        return hc_q8_count_;
    }
    uint64_t hc_q8_bytes() const {
        return hc_q8_bytes_;
    }

private:
    std::vector<void*> weights_;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    size_t requantized_ = 0;
    size_t hc_q8_count_ = 0;
    uint64_t hc_q8_bytes_ = 0;
};
} // namespace guild::core
