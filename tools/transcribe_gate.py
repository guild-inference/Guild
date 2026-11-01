"""tools/transcribe_gate.py - transcribe_run's transcripts against the reference's (Nemotron 3.5 Transcribe).

    python tools/transcribe_gate.py --ref reference_fp32.jsonl --got transcribe_run.jsonl --clips clips.jsonl
                                    [--thresholds gate.json] [--json report.json]
    python tools/transcribe_gate.py --ref reference_fp32.jsonl --noise reference_bf16.jsonl --clips clips.jsonl
                                    --write-thresholds gate.json

One line per clip in each file ({"id", "text"}, and "ids").  Per clip: whether the transcript is the reference's
(the same text), and its word error rate against the clip's own transcript (clips.jsonl "text": LibriSpeech's and
VoxPopuli's), both normalized the same way (lower case, punctuation but apostrophes removed, spaces collapsed).

The gate (the user's, 2026-10-31): the reference is the checkpoint's transformers implementation in float32, greedy;
the bounds come from the reference's own bfloat16 run (--noise), written once to a thresholds file:
  1. at least as many transcripts identical to the reference's as the bf16 run has, and at least `min_identical_rate`
     (0.95, the task's) of the clips, whichever is lower;
  2. the overall WER (words in error over words, all clips) no further from the reference's than the bf16 run's is,
     or 0.2 points (the task's), whichever is larger.
Exit status 0 when it passes.
"""
import argparse
import json
import re
import sys


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["id"], r)  # the first run of a clip
    return out


def norm(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return text.split()


def edits(ref, hyp):
    """word-level Levenshtein distance"""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def score(run, clips):
    """-> (overall WER in percent, per clip errors and words)"""
    errs = words = 0
    per = {}
    for cid, c in clips.items():
        if cid not in run:
            continue
        gold = norm(c["text"])
        e = edits(gold, norm(run[cid]["text"]))
        per[cid] = (e, len(gold))
        errs += e
        words += len(gold)
    return 100.0 * errs / max(words, 1), per


def compare(ref, got, clips):
    same = [cid for cid in ref if cid in got and ref[cid]["text"] == got[cid]["text"]]
    diff = [cid for cid in ref if cid in got and ref[cid]["text"] != got[cid]["text"]]
    missing = [cid for cid in ref if cid not in got]
    wer_ref, _ = score(ref, clips)
    wer_got, _ = score(got, clips)
    return {"clips": len(ref), "identical": len(same), "different": diff, "missing": missing, "wer_ref": wer_ref,
            "wer_got": wer_got, "wer_diff": abs(wer_got - wer_ref)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--got")
    ap.add_argument("--noise")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--write-thresholds")
    ap.add_argument("--thresholds")
    ap.add_argument("--json")
    a = ap.parse_args()
    ref, clips = load(a.ref), load(a.clips)
    if a.write_thresholds:
        s = compare(ref, load(a.noise), clips)
        th = {"reference": a.ref.split("/")[-1], "noise": a.noise.split("/")[-1], "min_identical_rate": 0.95,
              "noise_identical": s["identical"], "min_wer_diff": 0.2, "noise_wer_diff": round(s["wer_diff"], 4),
              "noise_run": {k: s[k] for k in ("clips", "identical", "different", "wer_ref", "wer_got")}}
        th["min_identical"] = min(s["identical"], int(-(-0.95 * s["clips"] // 1)))
        th["max_wer_diff"] = max(th["noise_wer_diff"], 0.2)
        with open(a.write_thresholds, "w", encoding="utf-8") as f:
            json.dump(th, f, indent=1)
        print(json.dumps(th, indent=1))
        return 0
    got = load(a.got)
    s = compare(ref, got, clips)
    print(f"clips: {s['clips']}; identical transcripts: {s['identical']} ({100.0 * s['identical'] / s['clips']:.1f}%)")
    print(f"WER: reference {s['wer_ref']:.2f}%, got {s['wer_got']:.2f}% (difference {s['wer_diff']:.2f} points)")
    for cid in s["different"]:
        print(f"  differs: {cid}\n    ref: {ref[cid]['text']}\n    got: {got[cid]['text']}")
    failures = [f"{cid}: no transcript" for cid in s["missing"]]
    if a.thresholds:
        th = json.load(open(a.thresholds))
        print(f"bounds: identical >= {th['min_identical']}, WER difference <= {th['max_wer_diff']:.2f} points")
        if s["identical"] < th["min_identical"]:
            failures.append(f"{s['identical']} identical transcripts, the bound is {th['min_identical']}")
        if s["wer_diff"] > th["max_wer_diff"]:
            failures.append(f"WER {s['wer_diff']:.2f} points from the reference's, the bound is {th['max_wer_diff']:.2f}")
    for f in failures:
        print("FAIL " + f)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": s, "failures": failures}, f, indent=1)
    print("PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
