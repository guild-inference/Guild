"""A request sequence through every prefix-cache path; prints outputs, reused tokens and wall time per request."""
import subprocess, sys, threading, queue, time, json
exe, out = sys.argv[1], sys.argv[2]; M = '/mnt/models-ssd/Strata-data'
p = subprocess.Popen([exe, '--serve', '--native', f'{M}/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf',
    '--pack', f'{M}/packs/unsloth-ud-iq4_xs', '--max-context', '32768', '--kv', 'fp16', '--kv-host-only', '--spec', '4',
    '--mtp', f'{M}/mtp/rt', '--expert-cache', '0', '--resident-budget-gib', '56'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
q = queue.Queue(); threading.Thread(target=lambda: [q.put(l.rstrip()) for l in p.stdout], daemon=True).start()
def gen(ids, n=16):
    t0 = time.perf_counter(); p.stdin.write(f"GEN {n} {','.join(map(str, ids))}\n"); p.stdin.flush(); toks = []
    while True:
        l = q.get()
        if l.startswith('T '): toks.append(int(l[2:]))
        elif l.startswith('DONE'): f = l.split(); return toks, int(f[8]), round((time.perf_counter() - t0) * 1000)
        elif l.startswith('ERR'): raise SystemExit(l)
while not q.get().startswith('READY'): pass
base = [760, 6511, 315, 9625, 13, 220, 1957, 4387, 11, 323, 279, 3363]
P = (base * 30)[:300]
res = {}
res['A cold'] = a = gen(P)
res['B continue'] = gen(P + a[0] + base[:7])
res['C branch after A'] = gen(P + a[0] + base[3:11])
res['D branch inside P'] = gen(P[:200] + base[5:12] * 3)
res['E unrelated'] = gen([1000 + i for i in range(40)])
res['F A again'] = gen(P)
for k, v in res.items(): print(f'{k:20s} reused {v[1]:4d}  {v[2]:5d} ms  out {v[0][:6]}...')
json.dump(res, open(out, 'w')); p.stdin.close(); p.wait()
