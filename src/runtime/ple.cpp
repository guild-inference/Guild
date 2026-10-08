// Reconnect the PLE setup from the working driver at 028e3e6^.
#include "model_impl.hpp"
#include "guild/artifact/gguf_reader.hpp"

namespace guild::runtime {

bool bind_required_ple(GuildModel::Impl& m, const std::vector<std::string>& shards, std::string& err) {
    using namespace guild::core;
    using namespace guild::kernels;
    std::string table_path = m.paths.ple_gguf;
    {
        const guild::GgufModel artifact(shards);
        size_t shard = 0;
        const auto* table = artifact.find("per_layer_token_embd.weight", &shard);
        const bool required = table != nullptr || m.wt.find("blk.1.ple_key.weight") != nullptr ||
                              m.desc.archetype == model::ModelArchetype::Qwen4Exp;
        m.ple_required = required;
        if (!required) return true;
        if (!table && table_path.empty()) {
            err = "required PLE table per_layer_token_embd.weight is absent";
            return false;
        }
        if (table_path.empty()) table_path = artifact.shard(shard).path();
        const auto constants = ple_artifact_consts();
        auto scalar = [&](const char* suffix, uint64_t want) {
            const std::string name = "qwen4exp.ple." + std::string(suffix);
            const auto* v = artifact.meta().get(name);
            if (v && v->is_num() && v->u == want) return true;
            err = "required PLE metadata differs from the supported contract: " + name;
            return false;
        };
        auto array = [&](const char* suffix, const uint64_t* want, size_t count) {
            const std::string name = "qwen4exp.ple." + std::string(suffix);
            const auto* v = artifact.meta().get(name);
            if (!v || v->type != MetaType::ARRAY || v->count != count || v->items.size() != count) {
                err = "required PLE metadata is missing or malformed: " + name;
                return false;
            }
            for (size_t i = 0; i < count; ++i) {
                if (!v->items[i].is_num() || v->items[i].u != want[i]) {
                    err = "required PLE hash/layout metadata differs: " + name;
                    return false;
                }
            }
            return true;
        };
        const uint64_t layer = 1;
        if (!scalar("ngram_size", NGRAM_SIZE) || !scalar("heads_per_ngram", HEADS_PER_NGRAM) ||
            !scalar("conv_kernel", PLE_CONV_KERNEL) || !scalar("eos_token_id", PLE_EOS_TOKEN_ID) ||
            !array("layers", &layer, 1) || !array("layer_multipliers", constants.mult, NGRAM_SIZE) ||
            !array("head_offsets", constants.offset, PLE_N_HEADS) ||
            !array("head_vocab_sizes", constants.vocab, PLE_N_HEADS)) return false;
    }
    if (m.g.n_embd != NG_N_EMBD || m.g.hc != NG_HC) {
        err = "required PLE geometry is not supported by the existing kernels";
        return false;
    }
    auto tensor = [&](const char* suffix, int64_t ne0, int64_t ne1, WeightKind kind) -> const WeightRef* {
        const std::string name = std::string("blk.1.") + suffix;
        const auto* w = m.wt.find(name);
        if (!w || !w->data || w->ne0 != ne0 || (ne1 && w->ne1 != ne1) || w->kind != kind) {
            err = "required PLE tensor has missing bytes, shape or engine form: " + name;
            return nullptr;
        }
        return w;
    };
    const auto* key = m.wt.find("blk.1.ple_key.weight");
    const auto* value = tensor("ple_value.weight", NG_N_EMBD, NG_N_EMBD, WeightKind::Bf16InF32);
    const auto* nk = tensor("ple_norm_key.weight", NG_HC_DIM, 0, WeightKind::F32);
    const auto* nq = tensor("ple_norm_query.weight", NG_HC_DIM, 0, WeightKind::F32);
    const auto* nc = tensor("ple_norm_conv.weight", NG_HC_DIM, 0, WeightKind::F32);
    const auto* conv = m.wt.find("blk.1.ple_conv1d.weight");
    if (!key || !value || !nk || !nq || !nc || !conv) {
        if (err.empty()) err = "required blk.1.ple_* tensor is absent";
        return false;
    }
    if (key->ne0 != NG_N_EMBD || key->ne1 != NG_HC_DIM) {
        err = "required PLE key has incompatible dimensions";
        return false;
    }
    if (!key->quantized()) {
        if (!key->data || key->kind != WeightKind::Bf16InF32 || key->bytes != uint64_t(NG_N_EMBD) * NG_HC_DIM * 2) {
            err = "required PLE key is not resident BF16";
            return false;
        }
        m.ss.ple.w.key_bf16 = static_cast<const uint16_t*>(key->data);
    } else if (key->native_data && key->native_q8_1 &&
               (key->native_type == 42 || key->native_type == 18 || key->native_type == 23 || key->native_type == 8)) {
        m.ss.ple.w.key_native_data = key->native_data;
        m.ss.ple.w.key_native_type = key->native_type;
        m.ss.ple.w.key_native_q8_1 = key->native_q8_1;
    } else if (key->data && key->code_bits == 2 && key->code_bias == -1 && key->group_elems == 64 &&
               key->codes_bytes + key->scales_bytes == key->bytes) {
        m.ss.ple.w.key_codes = static_cast<const uint8_t*>(key->data);
        m.ss.ple.w.key_scales = reinterpret_cast<const float*>(m.ss.ple.w.key_codes + key->codes_bytes);
    } else {
        err = "required PLE key has no compatible projection";
        return false;
    }
    const bool f16 = conv->kind == WeightKind::F16InF32 ||
                     (conv->kind == WeightKind::Verbatim && !conv->quantized());
    if (!f16 || !conv->data || conv->ne0 != PLE_CONV_KERNEL || conv->ne1 != NG_HC_DIM ||
        conv->bytes != uint64_t(PLE_CONV_KERNEL) * NG_HC_DIM * 2) {
        err = "required PLE convolution must have the existing F16 engine form";
        return false;
    }
    if (!m.ple_table.open(table_path, err)) return false;
    m.ss.ple.consts = ple_artifact_consts();
    for (int h = 0; h < PLE_N_HEADS; ++h) {
        if (m.ss.ple.consts.offset[h] + m.ss.ple.consts.vocab[h] > m.ple_table.rows()) {
            err = "required PLE table is too small for its hash vocabulary";
            return false;
        }
    }
    m.ple_emb_host.resize(NG_N_EMBD);
    if (cudaMalloc(&m.ple_emb_dev, NG_N_EMBD * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&m.ple_scratch, ple_run_scratch_bytes()) != cudaSuccess) {
        err = "required PLE buffers could not be allocated";
        return false;
    }
    auto& p = m.ss.ple;
    p.w.value_bf16 = static_cast<const uint16_t*>(value->data);
    p.w.norm_key = static_cast<const float*>(nk->data);
    p.w.norm_query = static_cast<const float*>(nq->data);
    p.w.norm_conv = static_cast<const float*>(nc->data);
    p.w.conv1d_f16 = static_cast<const uint16_t*>(conv->data);
    p.table = &m.ple_table;
    p.token = &m.ss.ple_token;
    p.prev = m.ss.ple_prev;
    p.hist = m.ss.ple_hist;
    p.emb_host = m.ple_emb_host.data();
    p.emb_dev = m.ple_emb_dev;
    p.scratch = static_cast<float*>(m.ple_scratch);
    if (!p.ready()) { err = "required PLE binding is incomplete"; return false; }
    std::fprintf(stderr, "guild runtime: required PLE connected (%s, %llu rows)\n", table_path.c_str(),
                 static_cast<unsigned long long>(m.ple_table.rows()));
    return true;
}

} // namespace guild::runtime
