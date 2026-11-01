"""tools/nemotron_audio_rt.py - the runtime files of Nemotron 3.5 Transcribe's audio model.

    python tools/nemotron_audio_rt.py --model <NVIDIA-Nemotron-3.5-Transcribe-30B-A4B dir> --out <dir>

The audio model turns a log-mel spectrogram into one 2,688-wide embedding per 8 frames (80 ms), which take the
places of the prompt's audio tokens (transformers' Nemotron3_5TranscribeAudioModel and multi_modal_projector):

    speech:   per-feature normalized mel -> 8 frames stacked (1,024) -> Linear 1,280 -> LayerNorm
              -> 32 pre-norm blocks (q/k LayerNorm per head, RoPE on the first half of each head, base 1e4, GELU MLP)
              -> LayerNorm -> asr_norm (LayerNorm)
    speakers: raw mel -> 8 frames stacked -> Linear 512 -> LayerNorm -> 31 pre-norm blocks (RoPE on whole heads)
              -> LayerNorm -> Linear 192 -> sub-pixel conv (k 3, x8 frames) -> ReLU, Linear, ReLU, Linear 8 speakers
              -> sigmoid, mean over each 8 frames, > 0.5 -> speaker_norm (LayerNorm over 8) @ a sinusoidal table
    out:      Linear 2,688 (projector) of (asr_norm output + speaker term)

Files (read by include/guild/audio/encoder.hpp):
  audio.txt        the geometry, one `key value` per line
  audio.bin        the tensors, 64-byte aligned
  audio_index.txt  one line per tensor: name kind rows cols offset bytes (kind bf16 or f32)
  prompt.json      the token ids around the audio of the default transcription request (the checkpoint's processor:
                   chat template, thinking off, the verbatim prompt) and the end-of-turn ids: the runner reads these
                   instead of tokenizing (Guild's tokenizer has no rule for this model's pre-tokenization)

Matrices are BF16 as the checkpoint stores them (the reference upcasts them to float32: the audio model runs in
float32); norms, biases, the STFT window and the mel filter bank (BF16 in the checkpoint) as F32.  Needs torch and
safetensors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

ASR = "perception.encoder.asr_encoder."
DIAR = "perception.encoder.diarization_model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    st = safe_open(str(src / "model.safetensors"), "pt")

    entries = []
    blob = bytearray()

    def put(name, t, kind):
        t = t.detach()
        rows, cols = (t.shape[0], t[0].numel()) if t.dim() > 1 else (1, t.numel())
        data = (t.to(torch.bfloat16).view(torch.int16) if kind == "bf16" else t.float()).contiguous().numpy().tobytes()
        while len(blob) % 64:
            blob.append(0)
        entries.append((name, kind, rows, cols, len(blob), len(data)))
        blob.extend(data)

    def mat(name, key):
        t = st.get_tensor(key)
        assert t.dtype == torch.bfloat16, key
        put(name, t, "bf16")

    def vec(name, key):
        put(name, st.get_tensor(key), "f32")

    vec("frontend.window", "perception.preprocessor.featurizer.window")
    vec("frontend.filters", "perception.preprocessor.featurizer.fb")

    def encoder(prefix, tag, n_layers, qk_norm):
        mat(f"{tag}.stack", prefix + "pre_encode.proj.weight")
        vec(f"{tag}.in_norm.w", prefix + "embed_norm.weight")
        vec(f"{tag}.in_norm.b", prefix + "embed_norm.bias")
        for i in range(n_layers):
            p, q = f"{prefix}layers.{i}.", f"{tag}.{i}."
            vec(q + "norm1.w", p + "norm1.weight")
            vec(q + "norm1.b", p + "norm1.bias")
            mat(q + "qkv", p + "attn.w_qkv.weight")  # rows [query | key | value]
            if qk_norm:
                for n in ("q_norm", "k_norm"):
                    vec(q + n + ".w", p + f"attn.{n}.weight")
                    vec(q + n + ".b", p + f"attn.{n}.bias")
            mat(q + "out", p + "attn.out_proj.weight")
            vec(q + "out.b", p + "attn.out_proj.bias")
            vec(q + "norm2.w", p + "norm2.weight")
            vec(q + "norm2.b", p + "norm2.bias")
            mat(q + "fc1", p + "ffn.net.0.weight")
            vec(q + "fc1.b", p + "ffn.net.0.bias")
            mat(q + "fc2", p + "ffn.net.3.weight")
            vec(q + "fc2.b", p + "ffn.net.3.bias")
        vec(f"{tag}.out_norm.w", prefix + "final_norm.weight")
        vec(f"{tag}.out_norm.b", prefix + "final_norm.bias")

    pe = cfg["pe_encoder_config"]
    asr_cfg = pe["asr_encoder_cfg"]
    diar_enc = pe["diarization_model_cfg"]["encoder"]
    n_asr = int(asr_cfg["n_layers"])
    n_diar = sum(1 for k in st.keys() if k.startswith(DIAR + "encoder.layers.") and k.endswith(".attn.w_qkv.weight"))
    encoder(ASR, "asr", n_asr, True)
    vec("asr_norm.w", "perception.encoder.asr_norm.weight")
    vec("asr_norm.b", "perception.encoder.asr_norm.bias")
    encoder(DIAR + "encoder.", "diar", n_diar, False)
    sm = DIAR + "sortformer_modules."
    mat("diar.proj", sm + "encoder_proj.weight")
    vec("diar.proj.b", sm + "encoder_proj.bias")
    w = st.get_tensor(sm + "subpixel_upsample.weight")  # [out, in, k]
    put("diar.upsample", w.reshape(w.shape[0], -1), "bf16")
    vec("diar.upsample.b", sm + "subpixel_upsample.bias")
    mat("diar.dense", sm + "first_hidden_to_hidden.weight")
    vec("diar.dense.b", sm + "first_hidden_to_hidden.bias")
    mat("diar.out", sm + "single_hidden_to_spks.weight")
    vec("diar.out.b", sm + "single_hidden_to_spks.bias")
    vec("speaker_norm.w", "perception.encoder.diar_norm.weight")
    vec("speaker_norm.b", "perception.encoder.diar_norm.bias")
    mat("projector", "perception.proj.weight")
    vec("projector.b", "perception.proj.bias")

    d_asr = int(asr_cfg["d_model"])
    d_diar = st.get_tensor(DIAR + "encoder.embed_norm.weight").numel()
    geo = {
        "sample_rate": 16000, "n_fft": 512, "hop": 160, "win": 400, "n_mels": 128, "stack": 8,
        "preemphasis": 0.97, "log_guard": 2.0 ** -24,
        "asr_layers": n_asr, "asr_dim": d_asr, "asr_heads": int(asr_cfg["n_heads"]),
        "asr_ff": d_asr * int(asr_cfg["ff_expansion"]), "asr_rope_fraction": float(asr_cfg["rotary_fraction"]),
        "asr_rope_base": float(asr_cfg["rope_base"]),
        "diar_layers": n_diar, "diar_dim": d_diar, "diar_heads": int(diar_enc.get("n_heads", 8)),
        "diar_ff": st.get_tensor(DIAR + "encoder.layers.0.ffn.net.0.bias").numel(), "diar_rope_base": 10000.0,
        "diar_hidden": st.get_tensor(sm + "encoder_proj.bias").numel(), "speakers": 8,
        "diar_chunk": 500, "speaker_threshold": 0.5,
        "out_dim": st.get_tensor("perception.proj.bias").numel(), "ln_eps": 1e-5,
    }
    with open(out / "audio.txt", "w") as f:
        for k, v in geo.items():
            f.write(f"{k} {v}\n")
    with open(out / "audio.bin", "wb") as f:
        f.write(blob)
    with open(out / "audio_index.txt", "w") as f:
        for e in entries:
            f.write(" ".join(str(x) for x in e) + "\n")
    # the default request's ids around the audio placeholders (one placeholder per `stack` mel frames)
    import numpy as np
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(src))
    ids = proc.apply_transcription_request(np.zeros(16000, dtype=np.float32))["input_ids"][0].tolist()
    aid = proc.audio_token_id
    first, last = ids.index(aid), len(ids) - 1 - ids[::-1].index(aid)
    gen = json.loads((src / "generation_config.json").read_text())
    eos = gen.get("eos_token_id", [])
    prompt = {"prompt": proc.default_transcription_prompt, "prefix_ids": ids[:first], "suffix_ids": ids[last + 1:],
              "eos_ids": eos if isinstance(eos, list) else [eos]}
    (out / "prompt.json").write_text(json.dumps(prompt, indent=1))
    print(f"{len(entries)} tensors, {len(blob) / 2**20:.1f} MiB -> {out}")


if __name__ == "__main__":
    main()
