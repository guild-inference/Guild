# Whisper-small transcripts of WAVs and their word error rate against the text (lowercase, no punctuation)
import re, sys, wave
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
M = "/mnt/models-ssd/Strata-data/models/whisper-small"
proc = WhisperProcessor.from_pretrained(M)
model = WhisperForConditionalGeneration.from_pretrained(M).eval()
text = open(sys.argv[1]).read()
def words(s):
    return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()
def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / len(ref)
ref = words(text)
for path in sys.argv[2:]:
    with wave.open(path) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        if w.getnchannels() > 1: x = x.reshape(-1, w.getnchannels()).mean(1)
    if sr != 16000:
        t = np.arange(0, len(x) / sr, 1 / 16000); x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    hyp_parts = []
    for c in range(0, len(x), 16000 * 28):   # 28 s windows
        f = proc(x[c : c + 16000 * 30], sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(f, language="en", task="transcribe", max_new_tokens=220)
        hyp_parts.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
    hyp = " ".join(hyp_parts)
    print(f"{path.split('/')[-1]}\t{wer(ref, words(hyp))*100:.1f}%\t{hyp.strip()}", flush=True)
