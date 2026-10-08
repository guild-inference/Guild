#pragma once

#include "gguf_fixture.hpp"
#include "guild/kernels/cpu/native_expert.hpp"
#include "guild/runtime/types.hpp"
#include <algorithm>
#include <sstream>

namespace fixture {

// Small, zero-weight hybrid model to exercise the real verifier/CPU handshake.
// Uses the existing non-HC geometry; it is not an Ornith quality fixture.
inline guild::runtime::ModelPaths runtime_pack(const std::filesystem::path& root) {
    constexpr uint64_t N = 256, FF = 256, V = 16;
    std::vector<Tensor> tensors = {{"token_embd.weight", {N, V}, 8}, {"output.weight", {N, V}, 8},
                                  {"output_norm.weight", {N}, 0}};
    auto tensor = [&](const std::string& name, std::vector<uint64_t> shape, uint32_t type) {
        tensors.push_back({name, std::move(shape), type});
    };
    for (int l = 0; l < 4; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        for (auto name : {"attn_norm.weight", "post_attention_norm.weight"}) tensor(p + name, {N}, 0);
        tensor(p + "ffn_gate_inp.weight", {N, 2}, 30);
        tensor(p + "ffn_gate_inp_shexp.weight", {N}, 30);
        for (auto role : {"gate", "up", "down"}) {
            tensor(p + "ffn_" + role + "_shexp.weight", {N, FF}, 8);
            tensor(p + "ffn_" + role + "_exps.weight", {N, FF, 2}, 8);
        }
        if (l != 3) {
            tensor(p + "attn_qkv.weight", {N, 8192}, 8);
            tensor(p + "attn_gate.weight", {N, 4096}, 8);
            tensor(p + "ssm_out.weight", {4096, N}, 8);
            tensor(p + "ssm_conv1d.weight", {4, 8192}, 0);
            for (auto name : {"ssm_alpha.weight", "ssm_beta.weight"}) tensor(p + name, {N, 32}, 30);
            tensor(p + "ssm_norm.weight", {128}, 0);
            tensor(p + "ssm_a", {32}, 0);
            tensor(p + "ssm_dt.bias", {32}, 0);
        } else {
            tensor(p + "attn_q.weight", {N, 4096}, 8);
            tensor(p + "attn_k.weight", {N, 256}, 8);
            tensor(p + "attn_v.weight", {N, 256}, 8);
            tensor(p + "attn_output.weight", {2048, N}, 8);
            for (auto name : {"attn_q_norm.weight", "attn_k_norm.weight"}) tensor(p + name, {256}, 0);
        }
    }
    const auto model_path = root / "synthetic.gguf";
    const auto written = write(model_path, {
        str("general.architecture", "qwen35moe"), str("general.name", "synthetic-failure-test"),
        u32("qwen35moe.block_count", 4), u32("qwen35moe.embedding_length", N),
        u32("qwen35moe.expert_count", 2), u32("qwen35moe.expert_used_count", 1),
        u32("qwen35moe.expert_feed_forward_length", FF), u32("qwen35moe.attention.head_count", 8),
        u32("qwen35moe.attention.head_count_kv", 1), u32("qwen35moe.attention.key_length", 256),
        u32("qwen35moe.full_attention_interval", 4), u32("qwen35moe.context_length", 64)}, tensors);
    guild::GgufFile gguf(model_path.string());
    std::ofstream dense(root / "dense.bin", std::ios::binary);
    std::ostringstream rows, experts;
    uint64_t src_off = 0, pool = 0, expert_off = 0;
    std::fstream payload(model_path, std::ios::in | std::ios::out | std::ios::binary);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        const auto* info = gguf.find(t.name);
        const uint64_t bytes = guild::tensor_payload_bytes(*info);
        std::vector<uint8_t> data(size_t(bytes), 0);
        if (t.type == 0 && t.name.find("norm.weight") != std::string::npos) {
            const float one = 1;
            for (size_t j = 0; j < data.size(); j += 4) std::memcpy(data.data() + j, &one, 4);
        }
        payload.seekp(std::streamoff(written.data_start + written.offsets[i]));
        payload.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        if (t.shape.size() == 3) continue;
        const bool quant = t.type == 8;
        const uint64_t ne0 = t.shape[0], ne1 = t.shape.size() > 1 ? t.shape[1] : 1;
        rows << t.name << " 0 " << (quant ? 0 : t.type == 30 ? 4 : 2) << ' '
             << src_off << ' ' << (quant ? 0 : bytes) << ' ' << pool << ' ' << (quant ? 0 : bytes)
             << ' ' << ne0 << ' ' << ne1 << ' ' << (quant ? 8 : 0)
             << " 0 0 0 0 0 0 0 0 0\n";
        if (!quant) {
            dense.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
            src_off += bytes;
            pool += (bytes + 255) & ~uint64_t(255);
        }
    }
    payload.close();
    dense.close();
    guild::kernels::cpu::NativeFmt fmt;
    std::string err;
    if (!guild::kernels::cpu::native_fmt(8, 8, N, FF, fmt, err)) throw std::runtime_error(err);
    for (int l = 0; l < 4; ++l) {
        experts << l << " 8 8 " << expert_off << ' ' << fmt.bytes;
        for (auto role : {"gate", "up", "down"}) {
            const auto* t = gguf.find("blk." + std::to_string(l) + ".ffn_" + role + "_exps.weight");
            experts << ' ' << gguf.data_start() + t->offset;
        }
        experts << '\n';
        expert_off += 2 * fmt.bytes;
    }
    { std::ofstream index(root / "index.txt"); index << "# align 256 pool " << pool << " tensors " << tensors.size() << '\n' << rows.str(); }
    { std::ofstream layout(root / "native_experts.txt"); layout << "# guild native experts v3: (n_expert 2)\n" << experts.str(); }
    guild::runtime::ModelPaths paths;
    paths.pack_dir = root.string();
    paths.primary_model_path = model_path.string();
    return paths;
}

} // namespace fixture
