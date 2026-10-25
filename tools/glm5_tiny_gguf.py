"""tools/glm5_tiny_gguf.py - a tiny random GLM-5.3-Flash GGUF for parity tests without the 157 GB model.

The same tensors and metadata as the real model (KDA and DSA layers, mHC, the indexer, MoE with a shared expert, the
NextN block), small dimensions and Q8_0 / F32 weights, so llama.cpp and Guild can both run it:

    python tools/glm5_tiny_gguf.py tiny.gguf <llama.cpp>/gguf-py [glm5next|glm5-next]
    tests/core/glm5_run: build-cuda12/glm5_run tiny.gguf <ids> 1 --logits ours.f32

The indexer keeps the top 64 cells (16 pools), so a prompt of more than 67 tokens takes the sparse path.
"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[2])
import gguf
from gguf import GGMLQuantizationType as Q

out, arch = sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else "glm5next"
rng = np.random.default_rng(7)
N, H, DK, DCONV = 512, 8, 128, 4
D = H * DK
QL, KL, QK, VD = 256, 128, 64, 64
IH, ID, TOPK, KP = 4, 128, 64, 4
HC = 4
E, EU, FF, FFE = 16, 4, 1024, 256
V = 512
kinds = ["kda", "dsa", "kda", "dsa"]          # main layers; the NextN block (layer 4) is DSA
NL = len(kinds) + 1

w = gguf.GGUFWriter(out, arch)
A = arch
w.add_uint32(f"{A}.block_count", NL)
w.add_uint32(f"{A}.context_length", 8192)
w.add_uint32(f"{A}.embedding_length", N)
w.add_uint32(f"{A}.feed_forward_length", FF)
w.add_uint32(f"{A}.attention.head_count", H)
w.add_array(f"{A}.attention.head_count_kv", [0 if k == "kda" else 1 for k in kinds] + [1])
w.add_float32(f"{A}.attention.layer_norm_rms_epsilon", 1e-5)
w.add_float32(f"{A}.attention.layer_norm_epsilon", 1e-6)
w.add_uint32(f"{A}.attention.q_lora_rank", QL)
w.add_uint32(f"{A}.attention.kv_lora_rank", KL)
w.add_uint32(f"{A}.attention.key_length", KL)
w.add_uint32(f"{A}.attention.value_length", KL)
w.add_uint32(f"{A}.attention.key_length_mla", QK)
w.add_uint32(f"{A}.attention.value_length_mla", VD)
w.add_uint32(f"{A}.rope.dimension_count", 0)
w.add_uint32(f"{A}.ssm.conv_kernel", DCONV)
w.add_uint32(f"{A}.kda.head_dim", DK)
w.add_float32(f"{A}.kda.gate_lower_bound", -5.0)
w.add_uint32(f"{A}.attention.indexer.head_count", IH)
w.add_uint32(f"{A}.attention.indexer.key_length", ID)
w.add_uint32(f"{A}.attention.indexer.top_k", TOPK)
w.add_uint32(f"{A}.attention.indexer.kpool", KP)
w.add_uint32(f"{A}.hyper_connection.count", HC)
w.add_uint32(f"{A}.hyper_connection.sinkhorn_iterations", 20)
w.add_float32(f"{A}.hyper_connection.epsilon", 1e-6)
w.add_uint32(f"{A}.expert_count", E)
w.add_uint32(f"{A}.expert_used_count", EU)
w.add_uint32(f"{A}.expert_group_count", 1)
w.add_uint32(f"{A}.expert_group_used_count", 1)
w.add_uint32(f"{A}.expert_gating_func", 2)
w.add_uint32(f"{A}.expert_feed_forward_length", FFE)
w.add_uint32(f"{A}.expert_shared_feed_forward_length", FFE)
w.add_uint32(f"{A}.expert_shared_count", 1)
w.add_uint32(f"{A}.leading_dense_block_count", 1)
w.add_float32(f"{A}.expert_weights_scale", 2.5)
w.add_bool(f"{A}.expert_weights_norm", True)
w.add_array(f"{A}.swiglu_clamp_exp", [10.0] * NL)
w.add_array(f"{A}.swiglu_clamp_shexp", [10.0] * NL)
w.add_uint32(f"{A}.nextn_predict_layers", 1)
w.add_uint32(f"{A}.vocab_size", V)
# a byte-level vocabulary (the ids are all that matter here)
b2u = {}
bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
toks = [chr(c) for c in cs] + [f"<tok{i}>" for i in range(V - 256 - 3)] + ["<|endoftext|>", "<|user|>", "<|observation|>"]
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("glm4")
w.add_token_list(toks)
w.add_token_types([1] * (V - 3) + [3, 3, 3])
w.add_token_merges([toks[ord('a') - 33 + 0] + " " + toks[ord('b') - 33]])
w.add_eos_token_id(V - 3)
w.add_eot_token_id(V - 2)
w.add_eom_token_id(V - 1)

def t(name, shape_ggml, std=0.05, q=Q.Q8_0, value=None):
    shape = tuple(reversed(shape_ggml))
    a = np.full(shape, value, dtype=np.float32) if value is not None else (rng.standard_normal(shape) * std).astype(np.float32)
    if q == Q.F32:
        w.add_tensor(name, a)
    else:
        qd = gguf.quants.quantize(a, q)
        w.add_tensor(name, qd, raw_shape=qd.shape, raw_dtype=q)

def norm(name, n):
    t(name, (n,), q=Q.F32, value=None, std=0.1)

t("token_embd.weight", (N, V), std=0.5)
t("output.weight", (N, V), std=0.1)
w.add_tensor("output_norm.weight", (1.0 + rng.standard_normal(N) * 0.1).astype(np.float32))
for l in range(NL):
    p = f"blk.{l}."
    w.add_tensor(p + "attn_norm.weight", (1.0 + rng.standard_normal(N) * 0.1).astype(np.float32))
    w.add_tensor(p + "ffn_norm.weight", (1.0 + rng.standard_normal(N) * 0.1).astype(np.float32))
    kind = kinds[l] if l < len(kinds) else "dsa"
    if l < len(kinds):
        for hc in ("attn", "ffn"):
            t(p + f"hc_{hc}_fn.weight", (HC * N, (2 + HC) * HC), std=0.02)
            t(p + f"hc_{hc}_base.weight", ((2 + HC) * HC,), q=Q.F32, std=0.5)
            t(p + f"hc_{hc}_scale.weight", (3,), q=Q.F32, std=0.5)
    if kind == "kda":
        for x in ("q", "k", "v"):
            t(p + f"attn_{x}.weight", (N, D), std=0.04)
            t(p + f"ssm_conv1d_{x}.weight", (DCONV, 1, D), q=Q.F32, std=0.4)
        t(p + "ssm_f_a.weight", (N, DK), std=0.04)
        t(p + "ssm_f_b.weight", (DK, D), std=0.08)
        t(p + "ssm_g_a.weight", (N, DK), std=0.04)
        t(p + "ssm_g_b.weight", (DK, D), std=0.08)
        t(p + "ssm_beta.weight", (N, H), std=0.04)
        w.add_tensor(p + "ssm_a", (-np.exp(rng.standard_normal(H) * 0.5)).astype(np.float32))
        t(p + "ssm_dt.bias", (D,), q=Q.F32, std=0.5)
        w.add_tensor(p + "ssm_norm.weight", (1.0 + rng.standard_normal(DK) * 0.1).astype(np.float32))
        t(p + "attn_output.weight", (D, N), std=0.03)
    else:
        t(p + "attn_q_a.weight", (N, QL), std=0.04)
        w.add_tensor(p + "attn_q_a_norm.weight", (1.0 + rng.standard_normal(QL) * 0.1).astype(np.float32))
        t(p + "attn_q_b.weight", (QL, H * QK), std=0.06)
        t(p + "attn_kv_a_mqa.weight", (N, KL), std=0.04)
        w.add_tensor(p + "attn_kv_a_norm.weight", (1.0 + rng.standard_normal(KL) * 0.1).astype(np.float32))
        t(p + "attn_k_b.weight", (QK, KL, H), std=0.08)
        t(p + "attn_v_b.weight", (KL, VD, H), std=0.08)
        t(p + "attn_output.weight", (H * VD, N), std=0.04)
        t(p + "indexer.k_norm.weight", (ID,), q=Q.F32, std=0.2)
        t(p + "indexer.k_norm.bias", (ID,), q=Q.F32, std=0.2)
        t(p + "indexer.proj.weight", (N, IH), q=Q.F32, std=0.05)
        t(p + "indexer.attn_k.weight", (N, ID), std=0.04)
        t(p + "indexer.attn_q_b.weight", (QL, IH * ID), std=0.06)
        t(p + "indexer_compressor_gate.weight", (N, ID), std=0.04)
        t(p + "indexer_compressor_ape.weight", (ID, KP), q=Q.F32, std=0.5)
    if l == 0:
        t(p + "ffn_gate.weight", (N, FF), std=0.04)
        t(p + "ffn_up.weight", (N, FF), std=0.04)
        t(p + "ffn_down.weight", (FF, N), std=0.03)
    else:
        t(p + "ffn_gate_inp.weight", (N, E), q=Q.F32, std=0.05)
        t(p + "exp_probs_b.bias", (E,), q=Q.F32, std=0.05)
        t(p + "ffn_gate_exps.weight", (N, FFE, E), std=0.04)
        t(p + "ffn_up_exps.weight", (N, FFE, E), std=0.04)
        t(p + "ffn_down_exps.weight", (FFE, N, E), std=0.05)
        t(p + "ffn_gate_shexp.weight", (N, FFE), std=0.04)
        t(p + "ffn_up_shexp.weight", (N, FFE), std=0.04)
        t(p + "ffn_down_shexp.weight", (FFE, N), std=0.05)
    if l == NL - 1:
        t(p + "nextn.eh_proj.weight", (2 * N, N), std=0.03)
        w.add_tensor(p + "nextn.enorm.weight", (1.0 + rng.standard_normal(N) * 0.1).astype(np.float32))
        w.add_tensor(p + "nextn.hnorm.weight", (1.0 + rng.standard_normal(N) * 0.1).astype(np.float32))
        w.add_tensor(p + "nextn.shared_head_norm.weight", (1.0 + rng.standard_normal(N) * 0.1).astype(np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
