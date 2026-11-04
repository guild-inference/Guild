"""tools/drama3_tower_rt.py - the runtime files of Fish Audio Drama 3's codebook tower.

    python tools/drama3_tower_rt.py --model <drama-3 dir> --out <dir>

Drama 3 (fishaudio/drama-3) emits one audio frame per backbone step: the backbone's head picks the frame's main
token (end of speech, or <|semantic:c0|>), then the tower predicts codes 1..9 of the frame.  The tower is 13
pre-norm blocks over time, each with its own K/V cache (fish_speech/models/text2semantic/fish_omni/tower.py):

    shared layer i (4):    e = semantic_embedding[c0] (+ sum of the previous frame's codes 2..9 embeddings, i = 0)
    residual layer j (9):  e = codebook_embedding[code j]          -> head j: logits of code j + 1
    every layer:           h = final_layernorm(block(eh_proj([enorm(e), hnorm(h)])))
    block:                 x += wo(attention(x) * sigmoid(gate));  x += w2(silu(w1 x) * w3 x)
    attention:             wqkv rows [Q 16x256 | gate 16x256 | K 2x256 | V 2x256], q/k RMS-normed per head,
                           interleaved (pairwise) RoPE on the first 64 channels, base 1e7, causal over the frames

e is zero on rows whose main token is not audio.  The backbone's input on an audio row is
(embed(main) + codebook_embeddings[4096 + code 1]) / sqrt(2), so that table's codebook-1 rows are written too.

Files (read by the engine's Drama 3 tower, include/guild/core/drama3_tower.hpp):
  tower.txt      the geometry and the token ids, one `key value` per line
  dense.bin      the tensors, 64-byte aligned
  dense.txt      one line per tensor: name kind rows cols offset bytes (kind q8_0, bf16 or f32)

The projections and heads are Q8_0 (ggml's reference rounding, as tools/mtp_rt.py), the embedding tables BF16 as
the checkpoint stores them, the norms F32 as their multipliers 1 + w.  `head.cand` is the backbone's lm_head over
the main token's candidates, eos then the 4,096 semantic tokens (ascending ids, as the reference samples them).
Needs torch (the checkpoint is BF16 safetensors).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mtp_rt import q8_0  # noqa: E402

ALIGN = 64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    from safetensors import safe_open

    cfg = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    ad, tc = cfg["audio_decoder_config"], cfg["text_config"]
    if ad.get("decoder_type") != "codebook_mtp_shared" or ad.get("vq_feature_placement") != "backbone_partial" or \
            ad.get("vq_backbone_codebooks") != 1 or ad.get("use_moe"):
        print("unsupported audio_decoder_config", file=sys.stderr)
        return 1
    dim, ncb, vocab = ad["dim"], ad["num_codebooks"], ad["vocab_size"]
    n_shared = ad["shared_enrichment_layers"]
    s0, s1, eos = cfg["semantic_start_token_id"], cfg["semantic_end_token_id"], cfg["eos_token_id"]
    if s1 - s0 + 1 != vocab or eos >= s0 or dim != tc["hidden_size"]:
        print("unexpected token band or dims", file=sys.stderr)
        return 1

    index = json.loads((args.model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    handles: dict[str, object] = {}

    def get(name: str) -> np.ndarray:
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(args.model / shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(name).to(torch.float32).numpy()

    def bf16_bytes(x: np.ndarray) -> bytes:
        u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()

    tensors: list[tuple[str, str, np.ndarray]] = []

    def layer(prefix: str, src: str, emb_name: str) -> None:
        p = f"{src}."
        tensors.append((prefix + "embedding", "bf16", get(p + emb_name + ".weight")))
        for n in ("enorm", "hnorm", "final_layernorm"):
            tensors.append((prefix + n, "f32", 1.0 + get(p + n + ".weight")))
        tensors.append((prefix + "eh_proj", "q8_0", get(p + "eh_proj.weight")))
        tl = p + "transformer_layer."
        tensors.append((prefix + "attention_norm", "f32", 1.0 + get(tl + "attention_norm.weight")))
        tensors.append((prefix + "ffn_norm", "f32", 1.0 + get(tl + "ffn_norm.weight")))
        tensors.append((prefix + "q_norm", "f32", 1.0 + get(tl + "attention.q_norm.weight")))
        tensors.append((prefix + "k_norm", "f32", 1.0 + get(tl + "attention.k_norm.weight")))
        for n in ("wqkv", "wo"):
            tensors.append((prefix + n, "q8_0", get(tl + f"attention.{n}.weight")))
        for n in ("w1", "w2", "w3"):
            tensors.append((prefix + n, "q8_0", get(tl + f"feed_forward.{n}.weight")))

    for i in range(n_shared):
        layer(f"shared.{i}.", f"audio_decoder_mtp_shared.layers.{i}", "semantic_embedding")
    for j in range(ncb - 1):
        layer(f"residual.{j}.", f"audio_decoder_mtp.layers.{j}", "codebook_embedding")
        tensors.append((f"head.{j}", "q8_0", get(f"audio_decoder_mtp.heads.{j}.weight")))
    tensors.append(("codebook_embeddings", "bf16", get("audio_decoder.codebook_embeddings.weight")))
    lm_head = get("lm_head.weight")
    cand = np.concatenate(([eos], np.arange(s0, s1 + 1)))
    tensors.append(("head.cand", "q8_0", lm_head[cand]))
    del lm_head

    args.out.mkdir(parents=True, exist_ok=True)
    lines, off = [], 0
    with open(args.out / "dense.bin", "wb") as f:
        for name, kind, x in tensors:
            x = np.ascontiguousarray(x, dtype=np.float32)
            rows, cols = (1, x.size) if kind == "f32" else x.shape
            raw = x.tobytes() if kind == "f32" else bf16_bytes(x) if kind == "bf16" else q8_0(x)
            pad = (-off) % ALIGN
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{name} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
    (args.out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "arch": "drama3-tower", "dim": dim, "n_head": ad["n_head"], "n_kv_head": ad["n_local_heads"],
        "head_dim": ad["head_dim"], "ffn": ad["intermediate_size"], "rope_dim": int(ad["head_dim"] * ad["rotary_percent"]),
        "rope_base": ad["rope_base"], "norm_eps": ad["norm_eps"], "n_shared": n_shared, "n_codebooks": ncb,
        "codebook_size": vocab, "semantic_start": s0, "semantic_end": s1, "eos": eos,
        "backbone_eps": tc["rms_norm_eps"],
    }
    (args.out / "tower.txt").write_text("".join(f"{k} {v}\n" for k, v in meta.items()), encoding="utf-8")
    print(f"wrote {len(tensors)} tensors, {off / 2**20:.0f} MiB, to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
