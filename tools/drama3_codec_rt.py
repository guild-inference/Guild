"""tools/drama3_codec_rt.py - the runtime files of Fish Audio Drama 3's audio codec (decode path).

    python tools/drama3_codec_rt.py --codec <drama-3 dir>/codec.pth --out <dir>

Drama 3 (and S2-Pro) decode their 10 codebooks per frame with fish-speech's "modded DAC" (fish_speech/models/dac/
modded_dac.py, rvq.py, configs/modded_dac_vq.yaml): 44.1 kHz, 2048 samples per frame.  Decode, in order:

  z = sum over the 10 codebooks of out_proj_i(codebook_i[c_i])     (codebook 0: 4096 entries, 1..9: 1024)
  post_module: 8 causal transformer layers (dim 1024, 16 heads of 64, window 128, pairwise RoPE base 1e4,
               RMSNorm eps 1e-5, layer scales, SwiGLU 3072), then RMSNorm
  upsample x2: [transposed conv k2 s2 -> ConvNeXt block (depthwise k7, LayerNorm, 1024 -> 4096 GELU -> 1024, gamma)]
  decoder: conv k7 1024 -> 1536; 4 blocks [Snake, transposed conv k=2s (s = 8, 8, 4, 2), 3 residual units
           (Snake, conv k7 dilation 1/3/9, Snake, conv k1) at 768, 384, 192, 96 channels]; Snake; conv k7 -> 1; tanh

Every convolution is causal (zeros on the left, transposed convolutions trimmed on the right).  Weight norm is folded
in fp32 (w = v * g / ||v||, the norm over all but axis 0), as fish-speech's server does before casting.

Files (read by include/guild/core/drama3_codec.hpp):
  codec.txt   the geometry, one `key value` per line
  dense.bin   float32 tensors, 64-byte aligned
  dense.txt   one line per tensor: name f32 rows cols offset bytes

Layouts, for the decoder's GEMMs (activations are [channels][time]):
  conv  `<name>.w`  [k][out][in] (one [out][in] matrix per tap), `<name>.b` [out]
  convT `<name>.w`  [out][k][in], `<name>.b` [out]
  vq    `vq.<i>`    [entries][1024]: out_proj applied to every codebook row, bias included (a frame's z is the sum
                    of ten rows)
Needs torch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ALIGN = 64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--codec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch

    sd = torch.load(str(args.codec), map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    def fold(prefix: str) -> torch.Tensor:
        """weight-normed weight at prefix (parametrizations or old weight_g/weight_v API), fp32 folding"""
        if prefix + ".parametrizations.weight.original0" in sd:
            g = sd[prefix + ".parametrizations.weight.original0"].float()
            v = sd[prefix + ".parametrizations.weight.original1"].float()
        else:
            g, v = sd[prefix + ".weight_g"].float(), sd[prefix + ".weight_v"].float()
        norm = v.reshape(v.shape[0], -1).norm(dim=1).reshape(-1, *([1] * (v.dim() - 1)))
        return v * (g / norm)

    out: list[tuple[str, np.ndarray]] = []

    def put(name: str, x: torch.Tensor) -> None:
        a = x.detach().to(torch.float32).contiguous().numpy()
        out.append((name, a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)))

    def conv(name: str, w: torch.Tensor, b: torch.Tensor) -> None:  # w [out, in, k] -> [k][out][in]
        put(name + ".w", w.permute(2, 0, 1).reshape(-1, w.shape[1]))
        put(name + ".b", b)

    def conv_t(name: str, w: torch.Tensor, b: torch.Tensor) -> None:  # w [in, out, k] -> [out][k][in]
        put(name + ".w", w.permute(1, 2, 0).reshape(-1, w.shape[0]))
        put(name + ".b", b)

    # the quantizer's decode tables: out_proj(codebook row) + bias for every row of every codebook
    qs = ["quantizer.semantic_quantizer.quantizers.0"] + [f"quantizer.quantizer.quantizers.{i}" for i in range(9)]
    for i, q in enumerate(qs):
        cb = sd[q + ".codebook.weight"].float()                      # [n, 8]
        w = fold(q + ".out_proj")[:, :, 0]                          # [1024, 8]
        put(f"vq.{i}", cb @ w.T + sd[q + ".out_proj.bias"].float())  # [n, 1024]

    pm = "quantizer.post_module"
    n_layers = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith(pm + ".layers."))
    for l in range(n_layers):
        p = f"{pm}.layers.{l}."
        for n in ("attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"):
            put(f"post.{l}.{n.split('.')[1]}", sd[p + n + ".weight"].float())
        for n in ("attention_norm", "ffn_norm"):
            put(f"post.{l}.{n}", sd[p + n + ".weight"].float())
        put(f"post.{l}.attn_scale", sd[p + "attention_layer_scale.gamma"].float())
        put(f"post.{l}.ffn_scale", sd[p + "ffn_layer_scale.gamma"].float())
    put("post.norm", sd[pm + ".norm.weight"].float())

    for u in range(2):
        p = f"quantizer.upsample.{u}."
        conv_t(f"up.{u}.convt", sd[p + "0.conv.weight"].float(), sd[p + "0.conv.bias"].float())
        put(f"up.{u}.dw.w", sd[p + "1.dwconv.conv.weight"].float()[:, 0, :])
        put(f"up.{u}.dw.b", sd[p + "1.dwconv.conv.bias"].float())
        put(f"up.{u}.ln.w", sd[p + "1.norm.weight"].float())
        put(f"up.{u}.ln.b", sd[p + "1.norm.bias"].float())
        put(f"up.{u}.pw1.w", sd[p + "1.pwconv1.weight"].float())
        put(f"up.{u}.pw1.b", sd[p + "1.pwconv1.bias"].float())
        put(f"up.{u}.pw2.w", sd[p + "1.pwconv2.weight"].float())
        put(f"up.{u}.pw2.b", sd[p + "1.pwconv2.bias"].float())
        put(f"up.{u}.gamma", sd[p + "1.gamma"].float())

    conv("dec.in", fold("decoder.model.0.conv"), sd["decoder.model.0.conv.bias"].float())
    rates = []
    for blk in range(1, 5):
        p = f"decoder.model.{blk}.block."
        put(f"dec.{blk}.snake", sd[p + "0.alpha"].float().reshape(-1))
        w = fold(p + "1.conv")
        rates.append(w.shape[2] // 2)
        conv_t(f"dec.{blk}.convt", w, sd[p + "1.conv.bias"].float())
        for r in range(3):
            q = f"{p}{2 + r}.block."
            put(f"dec.{blk}.ru{r}.snake1", sd[q + "0.alpha"].float().reshape(-1))
            conv(f"dec.{blk}.ru{r}.conv1", fold(q + "1.conv"), sd[q + "1.conv.bias"].float())
            put(f"dec.{blk}.ru{r}.snake2", sd[q + "2.alpha"].float().reshape(-1))
            conv(f"dec.{blk}.ru{r}.conv2", fold(q + "3.conv"), sd[q + "3.conv.bias"].float())
    put("dec.out.snake", sd["decoder.model.5.alpha"].float().reshape(-1))
    conv("dec.out", fold("decoder.model.6.conv"), sd["decoder.model.6.conv.bias"].float())

    args.out.mkdir(parents=True, exist_ok=True)
    lines, off = [], 0
    with open(args.out / "dense.bin", "wb") as f:
        for name, a in out:
            raw = np.ascontiguousarray(a, dtype=np.float32).tobytes()
            pad = (-off) % ALIGN
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{name} f32 {a.shape[0]} {a.shape[1]} {off} {len(raw)}")
            off += len(raw)
    (args.out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"arch": "drama3-codec", "sample_rate": 44100, "frame_samples": 2048, "n_codebooks": len(qs),
            "post_layers": n_layers, "post_dim": 1024, "post_heads": 16, "post_window": 128, "post_rope_base": 10000,
            "post_eps": 1e-5, "rates": ",".join(map(str, rates))}
    (args.out / "codec.txt").write_text("".join(f"{k} {v}\n" for k, v in meta.items()), encoding="utf-8")
    print(f"wrote {len(out)} tensors, {off / 2**20:.0f} MiB, to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
