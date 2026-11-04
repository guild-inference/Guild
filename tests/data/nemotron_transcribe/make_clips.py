"""make_clips.py <data dir> <out dir>: the gate's clips as 16 kHz PCM16 WAV files, and clips.jsonl.

<data dir> holds LibriSpeech/test-clean and LibriSpeech/test-other (openslr.org/12: test-clean.tar.gz,
test-other.tar.gz; CC BY 4.0) and voxpopuli-en_accented-test0.parquet (facebook/voxpopuli on the Hugging Face hub,
en_accented test split; CC0).  The same seed picks the same clips: the output's clips.jsonl equals this fixture's.

50 LibriSpeech test-clean utterances (at most 2 a speaker, 30 s or shorter, 3 of
25 s or longer and 5 more of 20 s or longer), 5 test-other
utterances and 5 VoxPopuli accented-English test utterances (five accents), chosen with a fixed seed."""
import io, json, os, random, sys
import pyarrow.parquet as pq
import soundfile as sf

data, out = sys.argv[1], sys.argv[2]
os.makedirs(out + "/wav", exist_ok=True)


def libri(split):
    rows = []
    root = f"{data}/LibriSpeech/{split}"
    for spk in sorted(os.listdir(root)):
        for ch in sorted(os.listdir(f"{root}/{spk}")):
            d = f"{root}/{spk}/{ch}"
            for line in open(f"{d}/{spk}-{ch}.trans.txt"):
                uid, text = line.rstrip("\n").split(" ", 1)
                info = sf.info(f"{d}/{uid}.flac")
                rows.append({"id": uid, "speaker": spk, "path": f"{d}/{uid}.flac", "text": text,
                             "seconds": info.frames / info.samplerate})
    return rows


rng = random.Random(20261031)
clean = [r for r in libri("test-clean") if r["seconds"] <= 30.0]
rng.shuffle(clean)
picked, per_spk = [], {}
for floor, upto in ((25.0, 3), (20.0, 8), (0.0, 50)):  # 3 of 25-30 s, 5 more of 20 s or longer, then any
    for r in clean:
        if len(picked) >= upto or r in picked or r["seconds"] < floor:
            continue
        if per_spk.get(r["speaker"], 0) >= 2:
            continue
        picked.append(r)
        per_spk[r["speaker"]] = per_spk.get(r["speaker"], 0) + 1
other = [r for r in libri("test-other") if r["seconds"] <= 30.0]
rng.shuffle(other)
seen = set()
for r in other:
    if r["speaker"] not in seen and len(seen) < 5:
        seen.add(r["speaker"])
        picked.append(dict(r, set="test-other"))
t = pq.read_table(f"{data}/voxpopuli-en_accented-test0.parquet").to_pylist()
rng.shuffle(t)
for acc in ("en_de", "en_fr", "en_pl", "en_fi", "en_es"):
    r = next(x for x in t if x["accent"] == acc and x["is_gold_transcript"] and
             sf.info(io.BytesIO(x["audio"]["bytes"])).duration <= 30.0)
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="int16")
    picked.append({"id": "voxpopuli-" + r["audio_id"].replace(":", ""), "set": "voxpopuli-" + acc, "audio": (a, sr),
                   "text": r["raw_text"], "normalized_text": r["normalized_text"], "seconds": len(a) / sr})
with open(f"{out}/clips.jsonl", "w") as f:
    for r in picked:
        r.setdefault("set", "test-clean")
        if "audio" in r:
            a, sr = r.pop("audio")
        else:
            a, sr = sf.read(r.pop("path"), dtype="int16")
        assert sr == 16000
        sf.write(f"{out}/wav/{r['id']}.wav", a, sr, subtype="PCM_16")
        r["wav"] = f"wav/{r['id']}.wav"
        r.pop("speaker", None)
        f.write(json.dumps(r) + "\n")
print(len(picked), "clips;", sum(r["seconds"] for r in picked) / 60, "minutes")
