# per arm: total tokens / total decode seconds over every prompt and run, and the geometric mean of the per-prompt
# medians; tokens a window from the engine logs
import glob, json, math, re, sys, collections
arms = collections.defaultdict(list)
for f in sorted(glob.glob('b-six_*_*.json')):
    m = re.match(r'b-six_(.+)_(\d)_(def|more)\.json', f)
    if not m:
        continue
    arm = m.group(1)
    d = json.load(open(f))
    for r in d['runs']:
        arms[arm].append((r['prompt'], r['generated'], r['decode_ms'], r['decode_tok_s']))
for arm, runs in arms.items():
    tok = sum(r[1] for r in runs); ms = sum(r[2] for r in runs)
    by = collections.defaultdict(list)
    for p, g, d, t in runs:
        by[p].append(t)
    geo = math.exp(sum(math.log(sorted(v)[len(v) // 2]) for v in by.values()) / len(by))
    print(f"{arm:6s} runs {len(runs):3d}  aggregate {tok / ms * 1000:6.2f} tok/s  geomean {geo:6.2f}  " +
          ' '.join(f"{p} {sum(v) / len(v):.1f}" for p, v in sorted(by.items())))
