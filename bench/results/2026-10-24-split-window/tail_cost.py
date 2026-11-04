"""Wall time of a request that reuses a cached 299-token prefix and prefills a new tail of k tokens, then emits one."""
import subprocess, sys, threading, queue, time, statistics, os, json
exe, out = sys.argv[1], sys.argv[2]; M = '/mnt/models-ssd/Strata-data'
p = subprocess.Popen([exe, '--serve', '--native', f'{M}/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf',
    '--pack', f'{M}/packs/unsloth-ud-iq4_xs', '--max-context', '32768', '--kv', 'fp16', '--kv-host-only', '--spec', '4',
    '--mtp', f'{M}/mtp/rt', '--expert-cache', '0', '--resident-budget-gib', '56'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
q = queue.Queue(); threading.Thread(target=lambda: [q.put(l.rstrip()) for l in p.stdout], daemon=True).start()
def gen(ids, n=1):
    t0 = time.perf_counter(); p.stdin.write(f"GEN {n} {','.join(map(str, ids))}\n"); p.stdin.flush(); toks = []
    while True:
        l = q.get()
        if l.startswith('T '): toks.append(int(l[2:]))
        elif l.startswith('DONE'): return toks, int(l.split()[8]), (time.perf_counter() - t0) * 1000
        elif l.startswith('ERR'): raise SystemExit(l)
while not q.get().startswith('READY'): pass
base = [760, 6511, 315, 9625, 13, 220, 1957, 4387, 11, 323, 279, 3363]
P = (base * 30)[:300]; gen(P)
res = {}
for k in [2, 4, 8, 16, 32, 64, 128]:
    ts, outs = [], []
    for r in range(3):
        tail = [(1000 + 37 * (i + 7 * r + k)) % 90000 + 200 for i in range(k)]
        o, reused, ms = gen(P + tail); ts.append(ms); outs.append(o)
        assert reused >= 299, reused
    res[k] = (statistics.median(ts), outs)
    print(f'{os.environ.get("GUILD_PREFILL_VERIFY_MAX","0"):>5} tail {k:4d}: {statistics.median(ts):7.0f} ms')
json.dump(res, open(out, 'w')); p.stdin.close(); p.wait()
