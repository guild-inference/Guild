#!/usr/bin/env python3
"""The time to read one long prompt (GEN, max_new 1, no prefix cache) with --batch N slots carved."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/home/ubuntu/work/par/guild/tools")
from batch_test import Engine, tokenizer  # noqa: E402

exe, cfg_path, batch, n_tok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = json.loads(Path(cfg_path).read_text())
tok = tokenizer(cfg["tokenizer"])
text = ("The history of mathematics is long and full of surprising turns. Early civilizations counted with tally "
        "marks, then with symbols, and later with place-value systems that made arithmetic far easier. ")
ids = tok.encode(text, parse_special=False)
ids = (ids * (n_tok // len(ids) + 1))[:n_tok]
eng = Engine(exe, cfg, batch if batch >= 2 else 0, {"GUILD_NO_KV_CACHE": "1"}, [])
out = eng.lines()
for rep in range(2):
    eng.send(f"GEN 1 {','.join(map(str, ids))}")
    for line in out:
        if line.startswith("DONE"):
            f = line.split()
            print(f"batch {batch} slots {eng.info.get('batch_slots', 0)}: {len(ids)} tokens read in {float(f[3]):.0f} ms",
                  flush=True)
            break
        if line.startswith("ERR"):
            print(line)
            break
eng.send("QUIT")
eng.p.wait(timeout=180)
