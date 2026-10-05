// include/guild/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "guild/core/weights.hpp"
#include "guild/model/model_descriptor.hpp"

#include <cstdint>
#include <string>

namespace guild::core {

/// Architecture-specific parameters for the qwen4exp hybrid GDN+QSA MoE architecture.
struct Qwen4ExpGeometry {
    // GDN recurrence parameters (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA indexer heads (12 layers)
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // Gated residual, on every layer
    int64_t hc = 4;
    int64_t hc_lr = 320;
};

/// The model's geometry. Generic MoE and attention parameters are kept at the top level,
/// while architecture-specific fast paths (such as Qwen4Exp's GDN and gated residual)
/// are grouped in dedicated archetype structures.
struct ModelGeometry {
    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t qsa_interval = 4;      ///< attention interval: layers 3, 7, ... 47 for interval 4

    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;

    int64_t n_expert = 512;
    int64_t n_ff = 640;

    // Archetype-specific geometry
    Qwen4ExpGeometry qwen4exp;

    // Direct accessors preserved for kernel compatibility
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;
    int64_t ssm_value_dim = 6144;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;
    int64_t hc = 4;
    int64_t hc_lr = 320;

    void sync_qwen4exp() {
        ssm_state_size = qwen4exp.ssm_state_size;
        ssm_k_heads = qwen4exp.ssm_k_heads;
        ssm_v_heads = qwen4exp.ssm_v_heads;
        ssm_d_conv = qwen4exp.ssm_d_conv;
        ssm_conv_channels = qwen4exp.ssm_conv_channels;
        ssm_value_dim = qwen4exp.ssm_value_dim;
        idx_q_heads = qwen4exp.idx_q_heads;
        idx_key_dim = qwen4exp.idx_key_dim;
        hc = qwen4exp.hc;
        hc_lr = qwen4exp.hc_lr;
    }

    void apply_descriptor(const model::ModelDescriptor& desc) {
        if (desc.attn.n_embd > 0) n_embd = desc.attn.n_embd;
        if (desc.attn.n_layers > 0) n_layers = desc.attn.n_layers;
        if (desc.attn.full_attn_interval > 0) qsa_interval = desc.attn.full_attn_interval;
        if (desc.attn.n_heads > 0) n_head = desc.attn.n_heads;
        if (desc.attn.n_kv_heads > 0) n_head_kv = desc.attn.n_kv_heads;
        if (desc.attn.head_dim > 0) head_dim = desc.attn.head_dim;
        if (desc.moe.n_routed_experts > 0) n_expert = desc.moe.n_routed_experts;
        if (desc.moe.expert_dim_ff > 0) n_ff = desc.moe.expert_dim_ff;
        sync_qwen4exp();
    }

    int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const { return qsa_interval > 0 ? n_layers / qsa_interval : n_layers; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};

/// True for the full-attention layers.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    if (g.qsa_interval <= 1) return true;
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace guild::core
