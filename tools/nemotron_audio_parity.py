"""tools/nemotron_audio_parity.py - Guild's audio front end and audio model against the reference, stage by stage.

    python tools/nemotron_audio_parity.py --ref <reference dump dir> --got <audio_encode out dir> [--json report.json]

Only the valid frames and tokens are compared.  The reference dump is the transformers reference's (one <id>.npz per clip: mel, mel_mask, asr, speaker_logits,
audio, embeds); audio_encode writes <id>.{mel,asr,spk,audio,emb}.f32.  Per stage: the largest absolute difference
over all clips, the largest relative to the stage's largest magnitude, and the clip it is in; the speaker
activity (sigmoid, mean over a token's 8 frames, > 0.5) is compared as a count of tokens and speakers that differ.
"""
import argparse
import json
import os

import numpy as np

STAGES = [("mel", "mel", "log-mel (valid frames)"), ("asr", "asr", "speech encoder output"),
          ("speaker_logits", "spk", "speaker logits"), ("audio", "audio", "audio model output"),
          ("embeds", "emb", "embeddings (projector)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--got", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    ids = sorted(f[:-4] for f in os.listdir(a.ref) if f.endswith(".npz"))
    worst = {k: (0.0, 0.0, None) for k, _, _ in STAGES}
    act_diff = act_total = 0
    rows = []
    for cid in ids:
        r = np.load(os.path.join(a.ref, cid + ".npz"))
        frames = int(r["mel_mask"].sum())
        row = {"id": cid, "frames": frames}
        for key, ext, _ in STAGES:
            path = os.path.join(a.got, f"{cid}.{ext}.f32")
            if not os.path.exists(path):
                row[key] = None
                continue
            ref = r[key].astype(np.float32)
            got = np.fromfile(path, dtype=np.float32).reshape(-1, ref.shape[-1])
            # the valid part: the reference pads one group of 8 frames more when the frames fill whole groups (it
            # counts the centered STFT's masked last frame), which the audio model masks out and the prompt drops
            tokens = (frames + 7) // 8
            n = {"mel": frames, "speaker_logits": tokens * 8}.get(key, tokens)
            ref, got = ref[:n], got[:n]
            d = float(np.abs(got - ref).max())
            rel = d / max(float(np.abs(ref).max()), 1e-30)
            row[key] = {"max_abs": d, "max_rel": rel}
            if d > worst[key][0]:
                worst[key] = (d, rel, cid)
            if key == "speaker_logits":
                def activity(lg):
                    p = 1 / (1 + np.exp(-lg.astype(np.float64)))
                    return p.reshape(-1, 8, p.shape[-1]).mean(1) > 0.5
                ar, ag = activity(ref), activity(got)
                row["activity_diff"] = int((ar != ag).sum())
                act_diff += row["activity_diff"]
                act_total += ar.size
        rows.append(row)
    print(f"{len(ids)} clips")
    for key, _, label in STAGES:
        d, rel, cid = worst[key]
        print(f"  {label:28s} max |diff| {d:.3e}  (relative to its largest value {rel:.2e}; {cid})")
    print(f"  speaker activity: {act_diff} of {act_total} token x speaker values differ")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"clips": rows, "worst": worst, "activity_diff": act_diff, "activity_total": act_total}, f,
                      indent=1)


if __name__ == "__main__":
    main()
