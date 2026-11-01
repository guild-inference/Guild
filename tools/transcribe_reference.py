"""tools/transcribe_reference.py - the checkpoint's transformers implementation transcribing clips on the CPU, greedy
(Nemotron 3.5 Transcribe; the reference of tests/data/nemotron_transcribe).

    python tools/transcribe_reference.py --model <checkpoint> --clips clips.jsonl --dtype bf16|fp32 --out ref.jsonl
                                         [--dump <dir>] [--speaker model|one] [--weights-q8] [--ids a,b]
                                         [--max-new 256]

Per clip, one line: the prompt ids, the generated ids, the text (special tokens skipped), and each step's margin
between the two largest logits.  --dump also saves the clip's log-mel features, the speech encoder's output, the
speaker model's logits, the audio model's output and the projected audio embeddings (npz), which
tools/nemotron_audio_parity.py reads.  --speaker one replaces the speaker model's output with speaker 0 active on
every frame (whether transcription needs the speaker model); --weights-q8 rounds the language model's matrices to
Q8_0 first (the weights' share of a Q8_0 GGUF's error, with fp32 activations).  The audio model and the projector
stay fp32 in both dtypes (the checkpoint's _keep_in_fp32_modules_strict); --dtype sets the language model's.
transformers' audio loader reads the WAV files (librosa: no resampling at 16 kHz).
"""
import argparse, json, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--clips", required=True)
ap.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
ap.add_argument("--out", required=True)
ap.add_argument("--dump")
ap.add_argument("--speaker", choices=("model", "one"), default="model")
ap.add_argument("--ids")
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--threads", type=int, default=32)
ap.add_argument("--weights-q8", action="store_true",
                help="round the language model's matrices (not the embedding) to Q8_0 first: blocks of 32 along a row, "
                     "scale max|w|/127, round to nearest; the audio model stays as it is")
a = ap.parse_args()
torch.set_num_threads(a.threads)
from transformers import AutoModelForMultimodalLM, AutoProcessor

t0 = time.time()
processor = AutoProcessor.from_pretrained(a.model)
model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
model.eval()
if a.weights_q8:
    n = 0
    with torch.no_grad():
        for name, p in list(model.model.language_model.named_parameters()) + list(model.lm_head.named_parameters()):
            if p.dim() < 2 or "norm" in name or "embeddings" in name or name.endswith("mixer.gate.weight") or p.shape[-1] % 32:
                continue
            w = p.data.float().reshape(-1, 32)
            d = w.abs().amax(dim=1, keepdim=True) / 127.0
            q = torch.where(d > 0, torch.round(w / torch.where(d > 0, d, torch.ones_like(d))), torch.zeros_like(w))
            p.data.copy_((q * d).reshape(p.shape).to(p.dtype)); n += 1
    print(f"rounded {n} language-model matrices to Q8_0", flush=True)
print(f"loaded in {time.time() - t0:.1f} s; LM {model.model.language_model.dtype}, audio "
      f"{model.model.audio_tower.dtype}", flush=True)

cap = {}
tower = model.model.audio_tower


def on_tower(mod, args, out):
    cap["audio"] = out.last_hidden_state[0].float().numpy().copy()
    cap["asr"] = out.asr_last_hidden_state[0].float().numpy().copy()
    cap["speaker_logits"] = out.speaker_logits[0].float().numpy().copy()


def on_diar(mod, args, out):
    if a.speaker == "one":
        lg = torch.full_like(out.logits, -30.0)
        lg[..., 0] = 30.0
        out.logits = lg
    return out


tower.register_forward_hook(on_tower)
tower.diarization_model.register_forward_hook(on_diar)
model.model.multi_modal_projector.register_forward_hook(
    lambda mod, args, out: cap.__setitem__("embeds", out[0].float().numpy().copy()))

clips = [json.loads(l) for l in open(a.clips)]
want = set(a.ids.split(",")) if a.ids else None
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out)}
root = os.path.dirname(os.path.abspath(a.clips))
if a.dump:
    os.makedirs(a.dump, exist_ok=True)
with open(a.out, "a") as f, torch.no_grad():
    for c in clips:
        if c["id"] in done or (want and c["id"] not in want):
            continue
        cap.clear()
        t0 = time.time()
        inputs = processor.apply_transcription_request(os.path.join(root, c["wav"]))
        t1 = time.time()
        out = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False, return_dict_in_generate=True,
                             output_logits=True)
        t2 = time.time()
        n = inputs["input_ids"].shape[1]
        gen = out.sequences[0, n:].tolist()
        margins = []
        for lg in out.logits:
            top = torch.topk(lg[0].float(), 2).values
            margins.append(round(float(top[0] - top[1]), 4))
        text = processor.decode(out.sequences[0, n:], skip_special_tokens=True)
        line = {"id": c["id"], "set": c["set"], "seconds": c["seconds"], "dtype": a.dtype, "speaker": a.speaker,
                "prompt_ids": inputs["input_ids"][0].tolist(), "mel_frames": int(inputs["input_features_mask"].sum()),
                "ids": gen, "text": text, "margins": margins, "features_ms": round(1000 * (t1 - t0), 1),
                "generate_ms": round(1000 * (t2 - t1), 1)}
        f.write(json.dumps(line) + "\n")
        f.flush()
        if a.dump:
            np.savez(os.path.join(a.dump, c["id"] + ".npz"), mel=inputs["input_features"][0].float().numpy(),
                     mel_mask=inputs["input_features_mask"][0].numpy(), **cap)
        print(f"{c['id']}: {len(gen)} tokens, {t2 - t1:.1f} s: {text}", flush=True)
