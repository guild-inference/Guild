"""Ornith with its MTP drafter through guild-generate --serve: greedy and sampled requests, short and long prompts,
and strategy changes between requests, against decode_strategy=normal.

    python mtp_check.py raw/o-st-normal.json <build>/guild-generate /mnt/models-ssd/Strata-data/mtp/ornith-rt
"""
import json, subprocess, sys, time
report, exe, mtp = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(report))
ids = d["prompts"]["code"]
prose = d["prompts"]["prose"]
long_ids = (prose + ids) * 40
long_ids = long_ids[:3000]
cmd = [exe, "--serve", "--native", "/home/ubuntu/models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf",
       "--pack", "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b",
       "--tokenizer", "/mnt/models-ssd/Strata-data/packs/ornith-1.5-35b/tokenizer",
       "--max-context", "4096", "--kv", "fp16", "--kv-host-only", "--resident-budget-gib", "24", "--spec", "4",
       "--mtp", mtp]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
while not p.stdout.readline().startswith("READY"):
    pass
def gen(line):
    t0 = time.time()
    p.stdin.write(line + "\n"); p.stdin.flush()
    toks = []
    while True:
        l = p.stdout.readline()
        if l.startswith("T "): toks.append(l.split()[1])
        elif l.startswith("DONE") or l.startswith("ERR"):
            return toks, l.strip(), time.time() - t0
ok = True
for name, prompt in (("code", ids), ("long 3000", long_ids)):
    pl = ",".join(map(str, prompt))
    for samp in ("temperature=0", "temperature=0.8 top_p=0.95 seed=12345"):
        outs = {st: gen(f"GEN 128 {samp} decode_strategy={st} {pl}") for st in ("normal", "auto", "normal", "auto")}
        res = {st: gen(f"GEN 128 {samp} decode_strategy={st} {pl}") for st in ("normal", "auto")}
        same = res["normal"][0] == res["auto"][0] and len(res["auto"][0]) == 128
        ok &= same
        f = res["auto"][1].split()
        print(f"{name:10s} {samp:40s} {'identical' if same else 'DIFFERENT'}; auto {res['auto'][2]:.2f} s, normal {res['normal'][2]:.2f} s; {res['auto'][1][:70]}")
p.stdin.write("QUIT\n"); p.stdin.flush(); p.stdin.close(); p.wait(timeout=120)
print("exit", p.returncode)
sys.exit(0 if ok else 1)
