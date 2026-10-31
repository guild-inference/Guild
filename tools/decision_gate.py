"""tools/decision_gate.py - decision_run's answers against a reference's (the checkpoint's own package).

    python tools/decision_gate.py --ref reference_fp32.jsonl --got decision_run.jsonl
                                  [--thresholds gate.json] [--requests requests.jsonl] [--json report.json]
    python tools/decision_gate.py --ref reference_fp32.jsonl --noise reference_bf16.jsonl --write-thresholds gate.json

Both files hold one line per request ({"id", "answers"}; decision_run --detail and the reference harness also give
each question's prompt "tokens").  Per question: the top answer (a choice's option, a noul's side, a score's most
likely level), the largest absolute probability difference max|dp| against the reference, the reference's margin
between its top two probabilities, and whether the prompt tokens are the same.

The gate (the user's, 2026-10-31): the reference is the package in fp32; the bound is the package's own bf16 run's
distance from it (--noise), written once to a thresholds file (--write-thresholds) so the gate is a fixed test:
  1. the top answer equals the reference's wherever the reference's top two probabilities differ by `margin` or more
     (0.05); near-ties may flip, and every flip is listed with its margin;
  2. (a) no more questions with max|dp| > 0.01 than the bf16 run has, and (b) a 95th percentile of max|dp| (nearest
     rank) no larger than the bf16 run's;
  3. the same prompt tokens wherever both sides have them.
With --requests, the requests' "expected" answers (JevBench's gold) are scored too.  Exit status 0 when it passes.
"""
import argparse
import json
import math
import sys


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def distribution(ans):
    """-> option -> probability of one answer"""
    if ans["type"] == "noul":
        return {"true": ans["noul"], "false": 1.0 - ans["noul"]}
    return ans["probabilities"]


def p95(values):
    """the nearest-rank 95th percentile"""
    v = sorted(values)
    return v[max(0, math.ceil(0.95 * len(v)) - 1)] if v else float("nan")


def compare(ref, got, gold=None):
    """-> (per-question rows, missing answers)"""
    rows, missing = [], []
    for rid, r in ref.items():
        g = got.get(rid)
        if g is None or "answers" not in g:
            missing.append(f"{rid}: {(g or {}).get('error', 'no line')}")
            continue
        for k, ra in r["answers"].items():
            ga = g["answers"].get(k)
            if ga is None:
                missing.append(f"{rid}/{k}: no answer")
                continue
            rp, gp = distribution(ra), distribution(ga)
            srt = sorted(rp.values(), reverse=True)
            rq, gq = r.get("questions", {}).get(k), g.get("questions", {}).get(k)
            toks = rq["tokens"] == gq["tokens"] if rq and gq and "tokens" in rq and "tokens" in gq else None
            row = {"id": rid, "q": k, "type": ra["type"], "top_ref": max(rp, key=rp.get), "top_got": max(gp, key=gp.get),
                   "margin": srt[0] - srt[1] if len(srt) > 1 else 1.0,
                   "max_abs_diff": max(abs(rp[o] - gp.get(o, float("nan"))) for o in rp), "tokens_equal": toks,
                   "n_tokens": len(rq["tokens"]) if rq and "tokens" in rq else None}
            if gold and rid in gold and k == "q":  # JevBench: a noul's gold is yes / no, a score's its level's index
                gv = gold[rid]
                row["gold"] = {"yes": "true", "no": "false"}.get(gv, str(gv)) if ra["type"] == "noul" else str(gv)
            rows.append(row)
    return rows, missing


def summary(rows):
    d = [r["max_abs_diff"] for r in rows]
    return {"questions": len(rows), "over_0_01": sum(x > 0.01 for x in d), "p95": p95(d), "max": max(d) if d else None,
            "median": sorted(d)[len(d) // 2] if d else None,
            "flips": [(r["id"], r["q"], round(r["margin"], 4)) for r in rows if r["top_ref"] != r["top_got"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--got")
    ap.add_argument("--noise", help="the reference's own lower-precision run (bf16), to derive the thresholds")
    ap.add_argument("--write-thresholds")
    ap.add_argument("--thresholds")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--requests")
    ap.add_argument("--json")
    a = ap.parse_args()
    ref = load(a.ref)

    if a.write_thresholds:
        rows, missing = compare(ref, load(a.noise))
        if missing:
            sys.exit("the noise run lacks answers: " + "; ".join(missing[:5]))
        s = summary(rows)
        th = {"reference": a.ref.split("/")[-1], "noise": a.noise.split("/")[-1], "margin": a.margin,
              "max_over_0_01": s["over_0_01"], "max_p95": s["p95"],
              "noise_run": {k: s[k] for k in ("questions", "over_0_01", "p95", "median", "max", "flips")}}
        with open(a.write_thresholds, "w", encoding="utf-8") as f:
            json.dump(th, f, indent=1)
        print(json.dumps(th, indent=1))
        return 0

    th = json.load(open(a.thresholds)) if a.thresholds else {"margin": a.margin, "max_over_0_01": None, "max_p95": None}
    gold = {}
    if a.requests:
        for r in load(a.requests).values():
            if "expected" in r:
                gold[r["id"]] = r["expected"]
    rows, missing = compare(ref, load(a.got), gold)
    s = summary(rows)
    failures = list(missing)
    print(f"questions: {s['questions']} in {len(ref)} requests")
    print(f"max|dp|: median {s['median']:.5f}, 95th (nearest rank) {s['p95']:.5f}, max {s['max']:.5f}; "
          f"over 0.01: {s['over_0_01']}")
    print("top-answer flips (reference margin): " + (", ".join(f"{i}/{q} {m}" for i, q, m in s["flips"]) or "none"))
    for r in rows:
        if r["top_ref"] != r["top_got"] and r["margin"] >= th["margin"]:
            failures.append(f"{r['id']}/{r['q']}: top {r['top_got']} against {r['top_ref']}, the reference's margin "
                            f"{r['margin']:.4f} >= {th['margin']}")
        if r["tokens_equal"] is False:
            failures.append(f"{r['id']}/{r['q']}: prompt tokens differ")
    if th.get("max_over_0_01") is not None:
        print(f"bound (the noise run's): over 0.01 <= {th['max_over_0_01']}, 95th <= {th['max_p95']:.5f}")
        if s["over_0_01"] > th["max_over_0_01"]:
            failures.append(f"{s['over_0_01']} questions over 0.01, the bound is {th['max_over_0_01']}")
        if s["p95"] > th["max_p95"]:
            failures.append(f"95th percentile {s['p95']:.5f}, the bound is {th['max_p95']:.5f}")
    tok = [r for r in rows if r["tokens_equal"] is not None]
    if tok:
        print(f"prompt tokens equal: {sum(r['tokens_equal'] for r in tok)} of {len(tok)}")
    for group in ("easy", "original", "hard"):
        g_rows = [r for r in rows if "gold" in r and r["id"].startswith(group)]
        if g_rows:
            print(f"jevbench {group}: reference {sum(r['top_ref'] == r['gold'] for r in g_rows)} / got "
                  f"{sum(r['top_got'] == r['gold'] for r in g_rows)} correct of {len(g_rows)}")
    for f in failures:
        print("FAIL " + f)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "summary": s, "thresholds": th, "failures": failures}, f, indent=1)
    print("PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
