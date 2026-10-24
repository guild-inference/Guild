"""Greedy and sampled requests through guild-generate --serve: each decode strategy must give the same tokens.

    python strategy_check.py raw/o-st-normal.json <build>/guild-generate     (Ornith paths as on this PC)
"""
import json, subprocess, sys
report, exe = sys.argv[1], sys.argv[2]
d = json.load(open(report))
code = d["prompts"]["code"]
ids = code["token_ids"] if isinstance(code, dict) and "token_ids" in code else code
base = ",".join(str(i) for i in ids)
rep = ",".join(str(i) for i in (ids * 12)[:512])  # a repeating prompt, where lookup drafts get accepted
cmd = [exe, "--serve", "--native", "/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf",
       "--pack", "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b",
       "--tokenizer", "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b/tokenizer",
       "--max-context", "4096", "--kv", "fp16", "--kv-host-only", "--resident-budget-gib", "24", "--spec", "4"]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
while not p.stdout.readline().startswith("READY"):
    pass
def gen(line):
    p.stdin.write(line + "\n"); p.stdin.flush()
    toks = []
    while True:
        l = p.stdout.readline()
        if l.startswith("T "): toks.append(l.split()[1])
        elif l.startswith("DONE") or l.startswith("ERR"): return toks, l.strip()
ok = True
for name, prompt in (("code", base), ("repeating", rep)):
    for samp in ("temperature=0", "temperature=0.8 top_p=0.95 seed=12345", "temperature=1.0 top_k=40 seed=7"):
        outs = {}
        for st in ("normal", "adaptive", "lookup"):
            outs[st] = gen(f"GEN 160 {samp} decode_strategy={st} {prompt}")
        same = outs["normal"][0] == outs["adaptive"][0] == outs["lookup"][0]
        ok &= same
        print(f"{name:9s} {samp:40s} {'identical' if same else 'DIFFERENT'} ({len(outs['normal'][0])} tokens; {outs['adaptive'][1][:60]})")
p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=60)
sys.exit(0 if ok else 1)
