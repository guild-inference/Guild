"""tools/ornith_mtp_rt.py - the MTP draft layer's runtime files for a qwen35moe GGUF (Ornith-1.5-35B), from the
`blk.<n_layer>.*` / `blk.<n_layer>.nextn.*` layer the GGUF ships and the engine's main loader skips.

    python tools/ornith_mtp_rt.py --gguf Ornith-1.5-35B-Q4_K_M.gguf --out <dir>

then run the engine with `--mtp <dir>`.  The layer (llama.cpp's qwen35moe nextn, vLLM's Qwen3_5MTP):

    x = eh_proj([enorm(embed(tok)), hnorm(h)])  ->  pre-norm full-attention block (its own K/V)  ->  pre-norm MoE
    (256 experts, top 8, shared expert with a sigmoid gate)  ->  shared_head_norm  ->  the main model's head

Writes the files tools/mtp_rt.py writes for Qwen's drafter, which `MtpDrafter` reads (see include/guild/core/mtp.hpp):
  mtp.txt       `arch qwen35moe` and the experts' format: native GGUF blobs, not the Q2_0 form
  experts.bin   one blob per expert, [gate rows | up rows | down rows]: gate/up as stored (Q4_K), down re-quantized
                from Q6_K to Q8_0 (ggml's reference rounding), which the grouped GPU expert kernels take
  dense.bin     eh_proj split into its embedding and hidden halves (fc_embedding / fc_hidden), the attention and
                shared-expert projections as Q8_0, the router and the shared gate as BF16, the norms as F32 (the
                GGUF's norm weights are already the multipliers)
  dense.txt     one line per tensor: name kind rows cols offset bytes
  draft_vocab.bin  the draft head's token subset (--draft-vocab, by default data/draft_vocab.bin: Ornith's
                tokenizer is Qwen's, byte for byte, so the subsets built for Qwen's drafter apply)

The draft layer only proposes tokens; the verify window decides every emitted one, so the re-quantization changes
how often a draft is accepted, never the output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

from mtp_rt import q8_0  # noqa: E402

Q4_K, Q6_K, Q8_0, F32 = 12, 14, 8, 0


def field_int(r: gguf.GGUFReader, key: str) -> int:
    f = r.fields[key]
    return int(f.parts[f.data[0]][0])


def floats(t) -> np.ndarray:
    """A tensor as float32 rows (outputs x inputs)."""
    data = np.asarray(t.data)
    if int(t.tensor_type) == F32:
        return data.astype(np.float32).reshape(-1, int(t.shape[0]))
    return quants.dequantize(data, t.tensor_type).astype(np.float32).reshape(-1, int(t.shape[0]))


def bf16(x: np.ndarray) -> bytes:
    """Round to nearest even, as torch's float -> bfloat16."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft-vocab", default=str(Path(__file__).resolve().parents[1] / "data" / "draft_vocab.bin"),
                    help="token subset for the draft head (int32 ids), or 'none' for the whole head")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = gguf.GGUFReader(a.gguf)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    if arch != "qwen35moe":
        sys.exit(f"{a.gguf}: architecture {arch}, this tool reads qwen35moe (Qwen's drafter: tools/mtp_rt.py)")
    n_layer = field_int(r, f"{arch}.block_count")
    if field_int(r, f"{arch}.nextn_predict_layers") < 1:
        sys.exit(f"{a.gguf}: no nextn (MTP) layer")
    L = n_layer - 1  # the nextn layer is the last block
    tens = {t.name: t for t in r.tensors}
    p = f"blk.{L}."
    need = lambda n: tens.get(p + n) or sys.exit(f"{a.gguf}: {p + n} is missing")  # noqa: E731

    # ---- the routed experts: native blobs
    gate, up, down = need("ffn_gate_exps.weight"), need("ffn_up_exps.weight"), need("ffn_down_exps.weight")
    if int(gate.tensor_type) != Q4_K or int(up.tensor_type) != Q4_K:
        sys.exit("the draft layer's gate/up experts are not Q4_K; extend this tool for their type")
    n_embd, n_ff, n_expert = (int(v) for v in gate.shape)
    g_raw, u_raw = np.asarray(gate.data), np.asarray(up.data)  # (n_expert, n_ff, row bytes)
    d_raw = np.asarray(down.data)  # (n_expert, n_embd, row bytes)
    blob = 0
    with open(out / "experts.bin", "wb") as f:
        for e in range(n_expert):
            if int(down.tensor_type) == Q8_0:
                dq = d_raw[e].tobytes()
            else:
                dq = q8_0(quants.dequantize(d_raw[e], down.tensor_type).astype(np.float32))
            b = g_raw[e].tobytes() + u_raw[e].tobytes() + dq
            blob = blob or len(b)
            assert len(b) == blob, (e, len(b), blob)
            f.write(b)

    # ---- the dense tensors
    eh = floats(need("nextn.eh_proj.weight"))  # (n_embd, 2 n_embd): [embedding | hidden] inputs
    assert eh.shape == (n_embd, 2 * n_embd), eh.shape
    dense = [
        ("fc_embedding.weight", "q8_0", eh[:, :n_embd]),
        ("fc_hidden.weight", "q8_0", eh[:, n_embd:]),
        ("pre_fc_norm_embedding.weight", "f32", floats(need("nextn.enorm.weight"))),
        ("pre_fc_norm_hidden.weight", "f32", floats(need("nextn.hnorm.weight"))),
        ("input_layernorm.weight", "f32", floats(need("attn_norm.weight"))),
        ("self_attn.q_proj.weight", "q8_0", floats(need("attn_q.weight"))),
        ("self_attn.k_proj.weight", "q8_0", floats(need("attn_k.weight"))),
        ("self_attn.v_proj.weight", "q8_0", floats(need("attn_v.weight"))),
        ("self_attn.o_proj.weight", "q8_0", floats(need("attn_output.weight"))),
        ("self_attn.q_norm.weight", "f32", floats(need("attn_q_norm.weight"))),
        ("self_attn.k_norm.weight", "f32", floats(need("attn_k_norm.weight"))),
        ("post_attention_layernorm.weight", "f32", floats(need("post_attention_norm.weight"))),
        ("mlp.gate.weight", "bf16", floats(need("ffn_gate_inp.weight"))),
        ("mlp.shared_expert.gate_proj.weight", "q8_0", floats(need("ffn_gate_shexp.weight"))),
        ("mlp.shared_expert.up_proj.weight", "q8_0", floats(need("ffn_up_shexp.weight"))),
        ("mlp.shared_expert.down_proj.weight", "q8_0", floats(need("ffn_down_shexp.weight"))),
        ("mlp.shared_expert_gate.weight", "bf16", floats(need("ffn_gate_inp_shexp.weight"))),
        ("shared_head_norm.weight", "f32", floats(need("nextn.shared_head_norm.weight"))),
    ]
    lines, off = [], 0
    with open(out / "dense.bin", "wb") as f:
        for name, kind, x in dense:
            x = np.ascontiguousarray(x, dtype=np.float32)
            rows, cols = (1, x.size) if kind == "f32" else x.shape
            raw = x.tobytes() if kind == "f32" else bf16(x) if kind == "bf16" else q8_0(x)
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{name} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_vocab = int(tens["output.weight"].shape[1]) if "output.weight" in tens else 0
    if a.draft_vocab != "none":
        ids = np.fromfile(a.draft_vocab, dtype=np.int32)
        if ids.size == 0 or ids.min() < 0 or ids.max() >= n_vocab:
            sys.exit(f"{a.draft_vocab}: token ids outside this model's {n_vocab}-token vocabulary")
        ids.tofile(out / "draft_vocab.bin")
        print(f"draft_vocab.bin: {ids.size} of {n_vocab} tokens from {a.draft_vocab}")
    else:
        (out / "draft_vocab.bin").unlink(missing_ok=True)
    d_type = Q8_0
    (out / "mtp.txt").write_text(
        f"arch {arch}\nexperts native\nexpert_gu_type {int(gate.tensor_type)}\nexpert_d_type {d_type}\n"
        f"n_embd {n_embd}\nn_ff {n_ff}\nn_expert {n_expert}\nexpert_bytes {blob}\n", encoding="utf-8")
    print(f"experts.bin {n_expert} x {blob} B, dense.bin {off} B, {len(lines)} tensors -> {out}")
    for line in lines:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
