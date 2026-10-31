"""tools/decision_gate.py - decision_run's answers against a reference's (the checkpoint's own package).

    python tools/decision_gate.py --ref reference.jsonl --got decision_run.jsonl [--requests requests.jsonl]
                                  [--tol 0.01] [--json report.json]

Both files hold one line per request ({"id", "answers"}; decision_run --detail and the reference harness also give
each question's prompt "tokens").  For every question of every request in the reference:

  * the top answer (a choice's option, a noul's side, a score's most likely level) must be the same;
  * every probability must be within --tol of the reference's (default 0.01, absolute);
  * the prompt tokens, when both sides have them, must be equal.

With --requests, the requests' "expected" answers (JevBench's gold) are scored for both sides too.  Exit status 0
when every question passes.
"""
import argparse
import json
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
    """-> (option -> probability, top option) of one answer"""
    t = ans["type"]
    if t == "noul":
        p = {"true": ans["noul"], "false": 1.0 - ans["noul"]}
    else:
        p = ans["probabilities"]
    return p, max(p, key=p.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--got", required=True)
    ap.add_argument("--requests")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--json")
    a = ap.parse_args()
    ref, got = load(a.ref), load(a.got)
    gold = {}
    if a.requests:
        for r in load(a.requests).values():
            if "expected" in r:
                gold[r["id"]] = r["expected"]

    rows, failures = [], []
    for rid, r in ref.items():
        g = got.get(rid)
        if g is None or "answers" not in g:
            failures.append(f"{rid}: missing ({(g or {}).get('error', 'no line')})")
            continue
        for k, ra in r["answers"].items():
            ga = g["answers"].get(k)
            if ga is None:
                failures.append(f"{rid}/{k}: no answer")
                continue
            rp, rtop = distribution(ra)
            gp, gtop = distribution(ga)
            diff = max(abs(rp[o] - gp.get(o, float("nan"))) for o in rp)
            toks = None
            rq, gq = r.get("questions", {}).get(k), g.get("questions", {}).get(k)
            if rq and gq and "tokens" in rq and "tokens" in gq:
                toks = rq["tokens"] == gq["tokens"]
            row = {"id": rid, "q": k, "type": ra["type"], "top_ref": rtop, "top_got": gtop, "max_abs_diff": diff,
                   "tokens_equal": toks, "n_tokens": len(rq["tokens"]) if rq and "tokens" in rq else None}
            if rid in gold and k == "q":  # JevBench: a noul's gold is yes / no, a score's its level's index
                g_val = gold[rid]
                row["gold"] = {"yes": "true", "no": "false"}.get(g_val, str(g_val)) if ra["type"] == "noul" else str(g_val)
            rows.append(row)
            why = []
            if rtop != gtop:
                why.append(f"top {gtop} against {rtop}")
            if not diff <= a.tol:
                why.append(f"max |dp| {diff:.4f}")
            if toks is False:
                why.append("prompt tokens differ")
            if why:
                failures.append(f"{rid}/{k}: " + ", ".join(why))

    diffs = sorted(r["max_abs_diff"] for r in rows)
    def pct(q):
        return diffs[min(len(diffs) - 1, int(q * len(diffs)))] if diffs else float("nan")
    tok_checked = [r for r in rows if r["tokens_equal"] is not None]
    print(f"questions: {len(rows)} in {len(ref)} requests; same top answer: "
          f"{sum(r['top_ref'] == r['top_got'] for r in rows)}; max |dp| <= {a.tol}: "
          f"{sum(r['max_abs_diff'] <= a.tol for r in rows)}")
    print(f"max |dp|: median {pct(0.5):.5f}, 90th {pct(0.9):.5f}, 99th {pct(0.99):.5f}, max {diffs[-1] if diffs else float('nan'):.5f}")
    if tok_checked:
        print(f"prompt tokens equal: {sum(r['tokens_equal'] for r in tok_checked)} of {len(tok_checked)}")
    for group in ("easy", "original", "hard"):
        g_rows = [r for r in rows if "gold" in r and r["id"].startswith(group)]
        if g_rows:
            print(f"jevbench {group}: reference {sum(r['top_ref'] == r['gold'] for r in g_rows)} / guild "
                  f"{sum(r['top_got'] == r['gold'] for r in g_rows)} correct of {len(g_rows)}")
    for f in failures:
        print("FAIL " + f)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "failures": failures, "tol": a.tol}, f, indent=1)
    print("PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
