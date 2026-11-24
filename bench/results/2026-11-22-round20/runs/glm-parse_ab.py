#!/usr/bin/env python3
"""parse_ab.py <glm-ab.out> [base id line file]: the glm5_run --prompts --ab generations, per prompt and arm."""
import hashlib
import re
import statistics
import sys

out = open(sys.argv[1]).read().splitlines()
base_ids = None
if len(sys.argv) > 2:
    base_ids = open(sys.argv[2]).read().splitlines()[-1].strip()

gen_re = re.compile(
    r"^gen (\S+) (\S+) round (\d+): prompt (\d+) tokens in (\d+) ms \(logits ([0-9a-f]+)\); (\d+) tokens, (\d+) steps, "
    r"(\d+) drafts accepted \(([0-9.]+)%; the second too: (\d+)\); per step: draft ([0-9.]+) ms \(median ([0-9.]+), "
    r"90th ([0-9.]+)\), verify ([0-9.]+), accept ([0-9.]+), CPU experts ([0-9.]+), step median ([0-9.]+); "
    r"([0-9.]+) tok/s")
ids_re = re.compile(r"^ids (\S+) (\S+) round (\d+): ?(.*)$")
gens, ids = [], {}
for line in out:
    m = gen_re.match(line)
    if m:
        g = m.groups()
        gens.append(dict(prompt=g[0], arm=g[1], round=int(g[2]), n_prompt=int(g[3]), prompt_ms=float(g[4]),
                         logits=g[5], tokens=int(g[6]), steps=int(g[7]), accepted=int(g[8]), acc_pct=float(g[9]),
                         draft=float(g[11]), verify=float(g[14]), accept=float(g[15]), cpu=float(g[16]),
                         step_med=float(g[17]), toks=float(g[18])))
        continue
    m = ids_re.match(line)
    if m:
        ids[(m.group(1), m.group(2), int(m.group(3)))] = m.group(4).strip()

short = {"GUILD_GLM_ATTEND3=1,GUILD_GLM_FUSE2=0xffffffff": "A new",
         "GUILD_GLM_ATTEND3=1,GUILD_GLM_FUSE2=0xfffffffb": "B fuse-off",
         "GUILD_GLM_ATTEND3=0,GUILD_GLM_FUSE2=0xfffffffb": "C both-off"}
print("generations in run order:")
for i, g in enumerate(gens):
    key = (g["prompt"], g["arm"], g["round"])
    h = hashlib.md5(ids.get(key, "").encode()).hexdigest()[:8]
    print(f"{i:2d} {g['prompt']:5s} r{g['round']} {short.get(g['arm'], g['arm']):10s} prompt {g['prompt_ms']:7.0f} ms "
          f"logits {g['logits'][:8]} | steps {g['steps']:3d} acc {g['accepted']:3d} ({g['acc_pct']:.1f}%) "
          f"draft {g['draft']:.3f} verify {g['verify']:.2f} accept {g['accept']:.2f} cpu {g['cpu']:.1f} "
          f"step med {g['step_med']:.2f} | {g['toks']:.2f} tok/s | ids md5 {h}")

print()
for p in sorted({g["prompt"] for g in gens}):
    print(f"prompt {p}:")
    arms = sorted({g["arm"] for g in gens if g["prompt"] == p})
    ref = None
    for a in arms:
        gs = [g for g in gens if g["prompt"] == p and g["arm"] == a]
        sm = [g["step_med"] for g in gs]
        ts = [g["toks"] for g in gs]
        pm = [g["prompt_ms"] for g in gs]
        dv = [g["draft"] for g in gs]
        vv = [g["verify"] for g in gs]
        print(f"  {short.get(a, a):10s} step med {' '.join(f'{x:.2f}' for x in sm)} (mean {statistics.mean(sm):.2f}) "
              f"| tok/s {' '.join(f'{x:.2f}' for x in ts)} (mean {statistics.mean(ts):.2f}) "
              f"| prompt ms {' '.join(f'{x:.0f}' for x in pm)} (mean {statistics.mean(pm):.0f}) "
              f"| draft {statistics.mean(dv):.3f} verify {statistics.mean(vv):.2f} "
              f"| accepted {' '.join(str(g['accepted']) for g in gs)}")
    # ids equality across arms and rounds
    allids = {k: v for k, v in ids.items() if k[0] == p}
    distinct = set(allids.values())
    logits = {g["logits"] for g in gens if g["prompt"] == p}
    print(f"  ids: {len(allids)} generations, {len(distinct)} distinct id lines; prompt logits hashes: {len(logits)} "
          f"distinct")
    if base_ids is not None and p == "short":
        same = [v == base_ids for v in allids.values()]
        print(f"  short vs base id line (md5 {hashlib.md5(base_ids.encode()).hexdigest()}): {sum(same)} of "
              f"{len(same)} equal")
    # paired differences per round: each arm against C
    rounds = sorted({g["round"] for g in gens if g["prompt"] == p})
    by = {(g["arm"], g["round"]): g for g in gens if g["prompt"] == p}
    names = {v: k for k, v in short.items()}
    for x, y in (("A new", "C both-off"), ("B fuse-off", "C both-off"), ("A new", "B fuse-off")):
        if names.get(x) in arms and names.get(y) in arms:
            d = [by[(names[x], r)]["step_med"] - by[(names[y], r)]["step_med"] for r in rounds]
            dt = [by[(names[x], r)]["toks"] / by[(names[y], r)]["toks"] * 100 - 100 for r in rounds]
            dp = [by[(names[x], r)]["prompt_ms"] - by[(names[y], r)]["prompt_ms"] for r in rounds]
            print(f"  {x} - {y}: step med {' '.join(f'{v:+.2f}' for v in d)} ms (mean {statistics.mean(d):+.2f}); "
                  f"tok/s {' '.join(f'{v:+.2f}%' for v in dt)} (mean {statistics.mean(dt):+.2f}%); prompt "
                  f"{' '.join(f'{v:+.0f}' for v in dp)} ms (mean {statistics.mean(dp):+.0f})")
