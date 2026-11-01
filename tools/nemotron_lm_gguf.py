"""tools/nemotron_lm_gguf.py - Nemotron 3.5 Transcribe's language model (with its MTP block) as a GGUF, through
llama.cpp's own Nemotron-H converter.

    python tools/nemotron_lm_gguf.py --model <NVIDIA-Nemotron-3.5-Transcribe-30B-A4B dir> --llama-cpp <llama.cpp dir>
                                     --out <file.gguf> [--outtype q8_0|bf16] [--work <dir>]

The checkpoint keeps the language model in NeMo's names (llm.model.*, llm.mtp.*, llm.lm_head) with the routed
experts as (experts, in, out) blocks.  This feeds llama.cpp's NemotronHModel converter the Hugging Face Nemotron-H
names it reads (backbone.*, mtp.*, one up_proj / down_proj per expert, transposed: transformers' own mapping,
conversion_mapping.py "nemotron3_5_transcribe") without copying the 60 GB of weights: the converter's tensor index
is replaced by lazy views of the checkpoint.  The MTP block becomes the GGUF's last block (llama.cpp's layout:
nextn_predict_layers 1).  The config is the checkpoint's llm_backbone/config.json; the tokenizer its tokenizer.json
without the audio placeholder token (id 131072, past the language model's vocabulary).
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--llama-cpp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", default="q8_0")
    ap.add_argument("--work", help="where the converter's config and tokenizer folder goes (default: next to --out)")
    a = ap.parse_args()
    src = Path(a.model)
    work = Path(a.work) if a.work else Path(a.out).with_suffix(".hf")
    work.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "llm_backbone" / "config.json").read_text())
    (work / "config.json").write_text(json.dumps(cfg, indent=1))
    tok = json.loads((src / "tokenizer.json").read_text())
    audio_ids = [t["id"] for t in tok.get("added_tokens", []) if t["content"] == "<|audio|>"]
    tok["added_tokens"] = [t for t in tok.get("added_tokens", []) if t["content"] != "<|audio|>"]
    vocab = tok.get("model", {}).get("vocab")
    if isinstance(vocab, dict):
        vocab.pop("<|audio|>", None)
    assert all(i >= cfg["vocab_size"] for i in audio_ids), "the audio token is inside the vocabulary"
    (work / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
    tcfg = json.loads((src / "tokenizer_config.json").read_text())
    tcfg.pop("audio_token", None)
    for k in ("extra_special_tokens", "model_specific_special_tokens"):
        if isinstance(tcfg.get(k), dict):
            tcfg[k].pop("audio_token", None)
    for k, v in list(tcfg.get("added_tokens_decoder", {}).items()):
        if v.get("content") == "<|audio|>":
            del tcfg["added_tokens_decoder"][k]
    (work / "tokenizer_config.json").write_text(json.dumps(tcfg, ensure_ascii=False, indent=1))
    for extra in ("chat_template.jinja", "generation_config.json"):
        if (src / extra).exists():
            shutil.copy(src / extra, work / extra)

    sys.path.insert(0, a.llama_cpp)
    sys.path.insert(1, str(Path(a.llama_cpp) / "gguf-py"))
    import gguf  # noqa: F401
    import convert_hf_to_gguf as conv
    from conversion.base import LazyTorchTensor
    from conversion.nemotron import NemotronHModel

    checkpoint = src / "model.safetensors"

    def rename(name):
        """-> the Hugging Face Nemotron-H name of an llm.* tensor, or None"""
        if name.endswith("._extra_state") or not name.startswith("llm."):
            return None
        name = name.replace("._fp32_params.", ".")
        if name == "llm.model.embed_tokens.weight":
            return "backbone.embeddings.weight"
        if name == "llm.model.norm.weight":
            return "backbone.norm_f.weight"
        if name == "llm.lm_head.weight":
            return "lm_head.weight"
        m = re.match(r"llm\.model\.layers\.(\d+)\.(.*)", name)
        if m:
            return f"backbone.layers.{m.group(1)}.{m.group(2)}"
        if name.startswith("llm.mtp."):
            return name[len("llm."):]
        raise ValueError(f"unmapped tensor {name}")

    def index_tensors(self, remote_hf_model_id=None):
        tensors = {}
        st = gguf.utility.SafetensorsLocal(checkpoint)
        with st as part:
            for name in part.keys():
                new = rename(name)
                if new is None:
                    continue
                data = part[name]
                if new.endswith(("experts.gate_and_up_projs", "experts.down_projs")):
                    which = "up_proj" if new.endswith("gate_and_up_projs") else "down_proj"
                    base = new.rsplit(".", 1)[0]
                    for e in range(data.shape[0]):  # (experts, in, out) -> one [out, in] matrix per expert
                        gen = lambda data=data, e=e: LazyTorchTensor.from_local_tensor(data)[e].T  # noqa: E731
                        if t := self.filter_tensors((f"{base}.{e}.{which}.weight", gen)):
                            tensors[t[0]] = t[1]
                    continue
                gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
                if t := self.filter_tensors((new, gen)):
                    tensors[t[0]] = t[1]
        return tensors

    NemotronHModel.index_tensors = index_tensors
    sys.argv = ["convert_hf_to_gguf.py", str(work), "--outtype", a.outtype, "--outfile", a.out]
    conv.main()


if __name__ == "__main__":
    main()
