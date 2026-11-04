"""tools/drama3_backbone_gguf.py - Fish Audio Drama 3's text backbone as a qwen35moe GGUF.

    python tools/drama3_backbone_gguf.py --model <drama-3 dir> --llama-cpp <llama.cpp checkout> --out <file.gguf>

Drama 3 (`FishOmniModel`, fishaudio/drama-3) is a Qwen3.5-MoE backbone (40 layers, 30 Gated DeltaNet + 10 gated
full attention, 256 experts top 8 with a gated shared expert, hidden 2048) plus a codebook tower.  This writes the
backbone with llama.cpp's own Qwen3.5-MoE converter (`conversion.qwen.Qwen3_5MoeTextModel`), so the tensor
transforms (the V-head reorder of the linear attention, the norm offset, A_log -> -exp) are exactly the ones Ornith's
GGUF went through.  What differs from a Qwen3.5 checkpoint is handled here:

  * the config nests the backbone under `text_config` and names its FP8 scheme `bsgemm-moe`; it is the same
    128 x 128 block scale (`weight_scale_inv`) as the converter's `fp8` method, which dequantizes the experts;
  * the tower's tensors (`audio_decoder*`) are left out: tools/drama3_tower_rt.py writes them;
  * there is no MTP layer.

The vocabulary keeps all 252,288 rows: the 4,096 `<|semantic:N|>` tokens at 248099..252194 are the audio band.
The output is BF16 by default (lossless from the FP8 experts); quantize it with llama-quantize, then pack it with
tools/iq_pack.py.  Needs torch and transformers (the converter does).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, type=Path, help="the fishaudio/drama-3 download")
    ap.add_argument("--llama-cpp", required=True, type=Path, help="a llama.cpp checkout with conversion/qwen.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "q8_0"])
    args = ap.parse_args()

    cfg = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "fish_omni":
        print(f"{args.model}: model_type {cfg.get('model_type')!r}, not fish_omni", file=sys.stderr)
        return 1
    quant = cfg.get("quantization_config") or {}
    if quant and (quant.get("quant_method") != "bsgemm-moe" or quant.get("weight_block_size") != [128, 128]):
        print(f"unsupported quantization_config {quant}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(args.llama_cpp / "gguf-py"))
    sys.path.insert(0, str(args.llama_cpp))
    import gguf  # noqa: E402
    import torch  # noqa: E402
    from conversion.qwen import Qwen3_5MoeTextModel  # noqa: E402

    hparams = {**cfg, **cfg["text_config"]}
    hparams["architectures"] = ["Qwen3_5MoeForCausalLM"]
    hparams["model_type"] = "qwen3_5_moe_text"
    hparams.pop("text_config", None)
    hparams.pop("audio_decoder_config", None)
    if quant:
        hparams["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}

    class Drama3Backbone(Qwen3_5MoeTextModel):
        model_arch = gguf.MODEL_ARCH.QWEN35MOE
        no_mtp = True

        @classmethod
        def filter_tensors(cls, item):
            if item[0].startswith("audio_decoder"):
                return None
            return super().filter_tensors(item)

    ftype = {"bf16": gguf.LlamaFileType.MOSTLY_BF16, "f16": gguf.LlamaFileType.MOSTLY_F16,
             "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}[args.outtype]
    with torch.inference_mode():
        model = Drama3Backbone(args.model, ftype, args.out, hparams=hparams, model_name="Fish Audio Drama 3")
        model.write()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
