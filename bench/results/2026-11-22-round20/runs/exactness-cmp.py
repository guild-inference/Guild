#!/usr/bin/env python3
# cmp.py [chk dir]: lane20/int's exactness check (check.sh) against base20/bl and the known hashes; PASS / FAIL lines.
import json, re, sys, os, hashlib, subprocess
B = '/home/ubuntu/lanes/r20/base20/bl'
C = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/lanes/r20/int/chk'
res = []
def rep(name, ok, detail=''):
    res.append(ok); print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
def last_ids(p):
    for l in open(p):
        if l.startswith('ids'): return l.strip()
def mib_lines(p, pat=r'MiB|GPU cache|chunks of'):
    # the auto cache's would-be size moves with the window graphs' VRAM (win-combine's smaller host-combine graphs:
    # 45 -> 47 here); it is a note, not a failure, while the decision ('none is kept') is the same
    out = []
    for l in open(p, errors='replace'):
        if re.search(pat, l) and 'loaded in' not in l:
            l = re.sub(r'\d+\.\d+ s\b', 'T s', l.strip())
            l = re.sub(r'\(\d+ MiB/s\)', '(R MiB/s)', l)  # a file read speed (the page cache), not a size
            m = re.search(r'a GPU cache of (\d+) slots would hold ([0-9.]+)%', l)
            if m:
                notes.add(f'{os.path.basename(p)}: would-be GPU cache {m.group(1)} slots')
                l = l.replace(m.group(0), 'a GPU cache of N slots would hold X%')
            out.append(l)
    return out
notes = set()
def ex(p): return os.path.exists(p)
# Qwen / Ornith decode ids
for k, f in (('qwen ids', 'q1.summary'), ('ornith ids', 'o1.summary')):
    if ex(f'{C}/{f}'): rep(k, last_ids(f'{C}/{f}') == last_ids(f'{B}/{f}'), last_ids(f'{C}/{f}'))
# VRAM / layout lines
for k, f in (('qwen vram lines', 'b-q1.log'), ('ornith vram lines', 'b-o1.log'), ('drama3 vram lines', 'd11.log'), ('startlux vram lines', 'sl.log')):
    if ex(f'{C}/{f}'):
        a, b = mib_lines(f'{B}/{f}'), mib_lines(f'{C}/{f}')
        rep(k, a == b, f'{len(b)} lines' + ('' if a == b else ' | base: ' + ' || '.join(x for x in a if x not in b)[:600] + ' | int: ' + ' || '.join(x for x in b if x not in a)[:600]))
# Drama 3 md5s, every arm
want = {'11': '6e04c9b1', '12': 'f84288c9'}
for f in sorted(os.listdir(C)):
    m = re.match(r'd(11|12)(-\w+)?\.summary$', f)
    if m:
        s = open(f'{C}/{f}').read().split()
        rep(f'drama3 {f}', bool(s) and s[-1] == want[m.group(1)], ' '.join(s))
# tower bits
for f in sorted(os.listdir(C)):
    if f.startswith('tower_'):
        t = open(f'{C}/{f}').read()
        rep(f'drama3 {f}', '8d3fea1524964286' in t and 'rc 0' in t, ' '.join(re.findall(r'bits [0-9a-f]+', t))[:80])
# Nemotron transcripts
if ex(f'{C}/nem.jsonl'):
    a = [(j['id'], j['ids'], j['text']) for j in map(json.loads, open(f'{B}/nem.jsonl'))]
    b = [(j['id'], j['ids'], j['text']) for j in map(json.loads, open(f'{C}/nem.jsonl'))]
    rep('nemotron transcripts', a == b, f'{sum(x == y for x, y in zip(a, b))} of {len(a)} lines')
# StartLux answers
if ex(f'{C}/sl.jsonl'):
    o = subprocess.run(['python3', '/home/ubuntu/lanes/r20/win-combine/tools/cmp_sl.py', f'{B}/sl.jsonl', f'{C}/sl.jsonl'], capture_output=True, text=True).stdout.strip()
    rep('startlux answers', 'DIFFER' not in o and 'missing' not in o and ' of ' in o, o.splitlines()[-1] if o else 'no output')
# batch slots
def slot_texts(t):
    # each 'slot N: ...' line with the quoted text line after it, the timings left out
    L = t.splitlines()
    return [(L[i], L[i + 1] if i + 1 < len(L) else '') for i in range(len(L)) if L[i].startswith('slot ')]
for f, n0 in (('b6.out', 6), ('b3.out', 3), ('btl.out', 3)):
    if ex(f'{C}/{f}'):
        t = open(f'{C}/{f}').read()
        bt = open(f'{B}/{f}').read() if ex(f'{B}/{f}') else None
        n = bt.count('IDENTICAL') if bt else n0
        k = t.count('IDENTICAL'); ok = k == n and 'DIFFER' not in t
        msg = f'{k} of {n} IDENTICAL'
        if bt:
            same = slot_texts(t) == slot_texts(bt); ok = ok and same
            msg += ', slot texts ' + ('equal to base20' if same else 'DIFFER from base20')
        rep(f'slots {f}', ok, msg)
# prompt hashes
known = {'pT': {'w16': 86768, 'p1k': 86557, 'p2k': 81685, 'p4k': 59650, 'p3k': 95757, 'p6k': 60700},
         'pD': {'w16': 86768, 'p7600': 77730}, 'pB': {'w16': 86768, 'p4k': 59650, 'ctx32k': 3242},
         'pM': {'w16': 78230, 'p2k': 26759, 'x10700': 86410, 'p4k': 89156, 'p3k': 18125}}
for L, k in known.items():
    p = f'{C}/pr/b-{L}.json'
    if ex(p):
        d = json.load(open(p))
        got = {r['prompt']: hash(tuple(r['output_ids'])) % 100000 for r in d['runs']}
        bad = {n: (h, got.get(n)) for n, h in k.items() if got.get(n) != h}
        rep(f'prompt hashes {L}', not bad, ' '.join(f'{n} {got.get(n)}' for n in k) + (f' | expected {bad}' if bad else ''))
        lg = f'{C}/pr/b-{L}.log'
        if ex(lg) and L != 'pM':
            t = open(lg).read(); rep(f'prompt layout {L}', 'chunks of 8192 tokens' in t and 'GPU cache 0 slots' in t, '')
# ctest
if ex(f'{C}/ctest.log'):
    t = open(f'{C}/ctest.log').read()
    m = re.search(r'(\d+)% tests passed, (\d+) tests failed out of (\d+)', t)
    fails = re.findall(r'\d+ - (\S+) \((Failed|Timeout|SEGFAULT|Subprocess aborted|Exception)', t)
    rep('ctest', m is not None and all(f[0] == 'ple_parity' for f in fails), (m.group(0) if m else 'no summary') + ' ' + str(fails))
for n in sorted(notes): print("NOTE", n)
print(f"{sum(res)} of {len(res)} checks pass")
