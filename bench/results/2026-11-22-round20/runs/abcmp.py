#!/usr/bin/env python3
"""abcmp.py [ab dir]: the integration A/B (ab.sh) per model: each run's bits against round 19's, then base (b) against
int (i) arms with the per-pair changes (pairs: b1/i1 and i2/b2), and Drama 3's GUILD_DRAMA3_FUSE arms."""
import glob, hashlib, json, os, re, statistics, subprocess, sys

O = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/lanes/r20/int/ab'
BL = '/home/ubuntu/lanes/r20/base20/bl'
bad = []


def mean(x):
    return statistics.mean(x) if x else float('nan')


def pct(a, b):  # change from a to b in %
    return (b - a) / a * 100.0 if a else float('nan')


def rep_pairs(name, b1, i1, i2, b2, unit, lower_better):
    pr1, pr2 = pct(b1, i1), pct(b2, i2)
    m = pct(mean([b1, b2]), mean([i1, i2]))
    word = 'faster' if (m < 0) == lower_better else 'slower'
    print(f'  {name:28s} base {b1:9.2f} {b2:9.2f} | int {i1:9.2f} {i2:9.2f} {unit} | pairs {pr1:+6.2f}% {pr2:+6.2f}% '
          f'| mean {m:+6.2f}% ({word})')


def ex(p):
    return os.path.exists(p)


# ---- Qwen / Ornith decode
def dec(kind, prefix, ids_ok):
    print(f'== {kind} decode (tok/s, prose / code)')
    v = {}
    for a in ('b1', 'i1', 'i2', 'b2'):
        lb = f'{prefix}-{a}'
        if not ex(f'{O}/{lb}.out'):
            return
        m = re.search(r'SUMMARY (\{.*\})', open(f'{O}/{lb}.out').read())
        s = json.loads(m.group(1)) if m else {}
        d = json.load(open(f'{O}/b-{lb}.json'))
        ids = [hash(tuple(x['output_ids'])) % 100000 for x in d['runs']]
        lg = open(f'{O}/b-{lb}.log', errors='replace').read() if ex(f'{O}/b-{lb}.log') else ''
        lay = 'chunks of 8192 tokens' in lg if kind == 'qwen' else True
        gc0 = 'GPU cache 0 slots' in lg if kind == 'qwen' else True
        ok = ids == ids_ok and lay and gc0
        if not ok:
            bad.append(f'{kind} {lb}: ids {ids} layout {lay} cache0 {gc0}')
        win = re.findall(r'([0-9.]+) ms/window', lg)
        v[a] = (s.get('prose', float('nan')), s.get('code', float('nan')), ok)
        print(f'  {lb:8s} prose {v[a][0]:6.2f} code {v[a][1]:6.2f} ids {"ok" if ok else "DIFFER " + str(ids)}'
              + (f' ms/window {" ".join(win)}' if win else ''))
    rep_pairs('prose tok/s', v['b1'][0], v['i1'][0], v['i2'][0], v['b2'][0], 'tok/s', False)
    rep_pairs('code tok/s', v['b1'][1], v['i1'][1], v['i2'][1], v['b2'][1], 'tok/s', False)


dec('qwen', 'q', [38747, 38747, 41916, 41916])
dec('ornith', 'o', [30936, 30936, 65886, 65886])

# ---- Drama 3
want_md5 = {'11': '6e04c9b1', '12': 'f84288c9'}
for s in ('11', '12'):
    runs = sorted(glob.glob(f'{O}/d{s}-*.log'), key=os.path.getmtime)
    if not runs:
        continue
    print(f'== Drama 3 seed {s} (frames/s, run order)')
    arms = {}
    for p in runs:
        lb = os.path.basename(p)[:-4]
        t = open(p, errors='replace').read()
        m = re.search(r'([0-9.]+) frames/s', t)
        fs = float(m.group(1)) if m else float('nan')
        w = f'{O}/{lb}.wav'
        h = hashlib.md5(open(w, 'rb').read()).hexdigest()[:8] if ex(w) else 'none'
        if h != want_md5[s]:
            bad.append(f'drama3 {lb}: md5 {h}')
        arm = re.sub(r'\d+$', '', lb.split('-', 1)[1])
        arms.setdefault(arm, []).append(fs)
        pm = re.search(r'prompt[^\n]*?([0-9.]+) ms', t)
        print(f'  {lb:14s} {fs:6.2f} md5 {h} {"ok" if h == want_md5[s] else "DIFFER"}')
    b = arms.get('b', [])
    for arm, xs in arms.items():
        print(f'  arm {arm:6s} mean {mean(xs):6.2f} ({" ".join(f"{x:.2f}" for x in xs)}) vs base {pct(mean(b), mean(xs)):+6.2f}%')

# ---- Nemotron
def nem_load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    tx = [(r['id'], r['ids'], r['text']) for r in rows]
    med = {}
    for cid in dict.fromkeys(r['id'] for r in rows):
        rr = [r for r in rows if r['id'] == cid and r['run'] >= 1]
        med[cid] = {k: statistics.median(r[k] for r in rr) for k in ('prefill_ms', 'decode_ms', 'total_ms')}
    return tx, med


if ex(f'{O}/n-b1.jsonl'):
    print('== Nemotron (medians of repeats 2-6, ms)')
    ref, _ = nem_load(f'{BL}/nem.jsonl')
    v = {}
    for a in ('b1', 'i1', 'i2', 'b2'):
        tx, med = nem_load(f'{O}/n-{a}.jsonl')
        if tx != ref:
            bad.append(f'nemotron n-{a}: transcripts differ')
        v[a] = med
    for cid in v['b1']:
        for k in ('prefill_ms', 'decode_ms', 'total_ms'):
            rep_pairs(f'{cid[:12]} {k}', v['b1'][cid][k], v['i1'][cid][k], v['i2'][cid][k], v['b2'][cid][k], 'ms', True)

# ---- StartLux
if ex(f'{O}/s-b1.jsonl'):
    print('== StartLux (mean ms of 5 repeats)')
    v = {}
    for a in ('b1', 'i1', 'i2', 'b2'):
        o = subprocess.run(['python3', '/home/ubuntu/lanes/r20/win-combine/tools/cmp_sl.py', f'{BL}/sl.jsonl',
                            f'{O}/s-{a}.jsonl'], capture_output=True, text=True).stdout
        if 'DIFFER' in o or 'missing' in o:
            bad.append(f'startlux s-{a}: {o.strip().splitlines()[-1]}')
        v[a] = {j['id']: j['timing']['mean_ms'] for j in map(json.loads, open(f'{O}/s-{a}.jsonl')) if j['id'] != 'warmup'}
    for rid in v['b1']:
        rep_pairs(rid, v['b1'][rid], v['i1'][rid], v['i2'][rid], v['b2'][rid], 'ms', True)

# ---- batch slots
for n in ('b3', 'b7'):
    if not ex(f'{O}/{n}-b1.out'):
        continue
    print(f'== batch slots {n}')
    ref = open(f'{BL}/{"b3.out" if n == "b3" else "b6.out"}').read()
    k_ref = ref.count('IDENTICAL')
    v = {}
    for a in ('b1', 'i1', 'i2', 'b2'):
        t = open(f'{O}/{n}-{a}.out').read()
        k = t.count('IDENTICAL')
        texts = [l for l in t.splitlines() if l.startswith('slot ') or l.startswith("    '") or l.startswith('    "')]
        rtexts = [l for l in ref.splitlines() if l.startswith('slot ') or l.startswith("    '") or l.startswith('    "')]
        if k != k_ref or 'DIFFER' in t or texts != rtexts:
            bad.append(f'slots {n}-{a}: {k} of {k_ref} IDENTICAL, texts {"equal" if texts == rtexts else "DIFFER"}')
        m = re.search(r'aggregate ([0-9.]+) tok/s overall, ([0-9.]+) tok/s from the first batch token', t)
        v[a] = (float(m.group(1)), float(m.group(2))) if m else (float('nan'),) * 2
    rep_pairs('tok/s overall', v['b1'][0], v['i1'][0], v['i2'][0], v['b2'][0], 'tok/s', False)
    rep_pairs('tok/s from first batch tok', v['b1'][1], v['i1'][1], v['i2'][1], v['b2'][1], 'tok/s', False)

# ---- Qwen prompts
known = {'w16': 86768, 'p1k': 86557, 'p2k': 81685, 'p4k': 59650, 'p3k': 95757, 'p6k': 60700, 'p7600': 77730,
         'ctx32k': 3242}
if ex(f'{O}/b-p-b1.json'):
    print('== Qwen prompts (prompt ms, GUILD_NO_KV_CACHE=1)')
    v = {}
    for a in ('b1', 'i1', 'i2', 'b2'):
        d = json.load(open(f'{O}/b-p-{a}.json'))
        v[a] = {}
        for r in d['runs']:
            h = hash(tuple(r['output_ids'])) % 100000
            if r['prompt'] in known and known[r['prompt']] != h:
                bad.append(f'prompts p-{a} {r["prompt"]}: hash {h}')
            v[a][r['prompt']] = r['prompt_ms']
    for p in v['b1']:
        rep_pairs(f'{p} ({known.get(p, "?")})', v['b1'][p], v['i1'][p], v['i2'][p], v['b2'][p], 'ms', True)

print()
print('BITS:', 'all runs identical to round 19' if not bad else f'{len(bad)} DIFFER')
for b in bad:
    print('  DIFFER', b)
