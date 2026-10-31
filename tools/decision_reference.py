"""tools/decision_reference.py - the HF reference for the StartLux-Decision gate: the checkpoint's own package
(startlux_decision/, Apache-2.0) with its transformers model, on the CPU (no GPU here holds the 65 GiB of bf16
weights).  It wrote tests/data/startlux_decision/reference_{fp32,bf16}.jsonl (torch 2.14.1 CPU, transformers
5.20.2; the long requests with --no-share).

    python tools/decision_reference.py --model <dir> --requests requests.jsonl --out ref-bf16.jsonl
                                       [--dtype bf16|fp32] [--ids a,b] [--no-share] [--weights-q8]

bf16 is the package's own setting (weights in bf16, autocast to bf16).  fp32 loads the weights in float32 and turns
the autocast off, for a check of how much bf16 itself moves the probabilities.

One line out per request: the answers decide() returns, and per question the rendered prompt's token ids, the
option letters' raw logits, the type's temperature and the probabilities (for the questions the model reads in one
pass; a wide choice's rounds and a single option are only in the answers).  Requests already in --out are skipped,
so the run can be restarted.
"""
import argparse
import contextlib
import json
import os
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--weights-q8", action="store_true",
                    help="round every matrix weight to Q8_0 (blocks of 32 along a row, scale max|w|/127, round to "
                         "nearest) before running: the weights' share of a Q8_0 GGUF's error, with fp32 activations")
    ap.add_argument("--no-share", action="store_true",
                    help="read every prompt whole: the package's shared-prefix path (_rows_cache) fails on "
                         "transformers 5.20, whose linear-attention cache keeps its states in dicts")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.model)
    import startlux_decision.model as M
    from startlux_decision import jevfmt as J

    # The package's load_model without device_map={"": device}: that needs accelerate, and on the CPU it places
    # nothing anyway.  Otherwise the same call: the checkpoint's own class, bf16, grouped experts.
    def load_cpu(path, device):
        import transformers
        config = transformers.AutoConfig.from_pretrained(path)
        extra = {"experts_implementation": "grouped_mm"} if getattr(config.get_text_config(), "num_experts", 0) else {}
        dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
        model = getattr(transformers, config.architectures[0]).from_pretrained(path, dtype=dtype, **extra)
        return model, getattr(model.model, "language_model", model.model)
    M.load_model = load_cpu
    if a.dtype == "fp32":
        torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()

    t0 = time.time()
    m = M.StartLuxDecision(a.model, device="cpu", graphs=False, images=False)
    if a.no_share:
        m.share = False
    if a.weights_q8:
        n = 0
        with torch.no_grad():
            for name, p in m.body.named_parameters():
                if p.dim() < 2 or "norm" in name or p.shape[-1] % 32:
                    continue
                w = p.data.float().reshape(-1, 32)
                d = w.abs().amax(dim=1, keepdim=True) / 127.0
                q = torch.where(d > 0, torch.round(w / torch.where(d > 0, d, torch.ones_like(d))), torch.zeros_like(w))
                p.data.copy_((q * d).reshape(p.shape).to(p.dtype))
                n += 1
            # the letters' head rows, as Q8_0 too
            w = m.letter_rows.reshape(-1, 32)
            d = w.abs().amax(dim=1, keepdim=True) / 127.0
            m.letter_rows = (torch.round(w / torch.where(d > 0, d, torch.ones_like(d))) * d).reshape(m.letter_rows.shape)
        print(f"weights rounded to Q8_0: {n} matrices", flush=True)
    print(f"loaded in {time.time() - t0:.1f} s, letters {m.letters[:3]}..., temperatures {m.temperature}", flush=True)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            done.add(json.loads(line)["id"])
    want = set(a.ids.split(",")) if a.ids else None
    with open(a.out, "a", encoding="utf-8") as f:
        for line in open(a.requests, encoding="utf-8"):
            req = json.loads(line)
            if req["id"] in done or (want and req["id"] not in want):
                continue
            # decide()'s own steps (model.py StartLuxDecision.decide), with the letter logits kept: _split answers
            # single options and wide choices, _logits reads the rest in one eager pass, _answer formats them
            t0 = time.time()
            detail = {}
            try:
                answers, rows, tokens = m._split(req["state"], req["questions"])
                if rows:
                    logits, n = m._logits([r for _, r in rows])
                    tokens += n
                    for (k, row), z in zip(rows, logits):
                        temp = m.temperature.get(row["type"], 1.0)
                        p = torch.softmax(z / temp, -1).tolist()
                        answers[k] = m._answer(row, p, req["questions"][k])
                        order = [o["id"] for o in row["options"]]
                        ids, _ = J.render_ids(row, m.tok, order, max_length=m.max_length)
                        detail[k] = {"type": row["type"], "options": order, "tokens": ids,
                                     "letter_logits": z.tolist(), "temperature": temp, "probs": p}
            finally:
                m._kept = m._media = None
            usage = {"input_tokens": tokens, "output_tokens": 0}
            t_decide = time.time() - t0
            rec = {"id": req["id"], "dtype": a.dtype, "shared_prefix_path": not a.no_share, "answers": answers, "usage": usage, "questions": detail,
                   "seconds": round(t_decide, 2)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{req['id']}: {usage['input_tokens']} tokens, {t_decide:.1f} s", flush=True)


if __name__ == "__main__":
    main()
