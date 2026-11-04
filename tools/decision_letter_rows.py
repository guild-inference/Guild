"""tools/decision_letter_rows.py - the option letters' rows of a decision checkpoint's output head, for decision_run.

    python tools/decision_letter_rows.py --model <HF checkpoint dir> --out <pack>/tokenizer/decision_letter_rows.f32

Reads decision_config.json's letter_token_ids (A..Z) and those rows of lm_head.weight from the checkpoint's
safetensors (bf16, as the model ships), and writes them as float32 [26][n_embd], little-endian.  decision_run then
reads the letters' logits as the checkpoint's own package does - the final hidden state times these exact rows - not
through the GGUF's quantized head.  No dependencies beyond the standard library.
"""
import argparse
import json
import struct
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tensor", default="lm_head.weight")
    a = ap.parse_args()
    letters = json.loads((a.model / "decision_config.json").read_text())["letter_token_ids"]
    index = a.model / "model.safetensors.index.json"
    shard = a.model / (json.loads(index.read_text())["weight_map"][a.tensor] if index.exists() else "model.safetensors")
    with open(shard, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        meta = header[a.tensor]
        if meta["dtype"] not in ("BF16", "F32", "F16"):
            sys.exit(f"{a.tensor}: unsupported dtype {meta['dtype']}")
        rows, cols = meta["shape"]
        esize = 4 if meta["dtype"] == "F32" else 2
        base = 8 + hlen + meta["data_offsets"][0]
        out = bytearray()
        for t in letters:
            if not 0 <= t < rows:
                sys.exit(f"letter id {t} outside the head's {rows} rows")
            f.seek(base + t * cols * esize)
            raw = f.read(cols * esize)
            if meta["dtype"] == "F32":
                out += raw
            elif meta["dtype"] == "BF16":  # bf16 is the high half of a float32
                for i in range(0, len(raw), 2):
                    out += b"\x00\x00" + raw[i:i + 2]
            else:
                for (h,) in struct.iter_unpack("<e", raw):
                    out += struct.pack("<f", h)
    a.out.write_bytes(bytes(out))
    print(f"{a.out}: {len(letters)} rows of {cols} from {shard.name} ({meta['dtype']})")


if __name__ == "__main__":
    main()
