"""kw.py <sqlite>: per decode window of the last two requests (verify graph launches), each kernel name's time,
count, and the grid shape most seen"""
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1])
L = c.execute("select r.start,r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like 'cudaGraphLaunch%' order by r.start").fetchall()
K = c.execute("select k.start,k.end,k.correlationId,s.value,k.gridX,k.gridY,k.gridZ,k.blockX from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
by = collections.defaultdict(set)
for k in K: by[k[2]].add(k[3])
ver = [s for s, cid in L if 'wait_flag_ge_kernel' in by[cid]]
# requests: verify launches < 400 ms apart
reqs = [[ver[0]]]
for t in ver[1:]:
    (reqs[-1].append(t) if t - reqs[-1][-1] < 400e6 else reqs.append([t]))
for ri, r in enumerate(reqs[-2:]):
    t0, t1 = r[0], r[-1]
    nwin = len(r) - 1
    tot = collections.Counter(); cnt = collections.Counter(); shape = collections.defaultdict(collections.Counter)
    for k in K:
        if t0 <= k[0] < t1:
            tot[k[3]] += k[1] - k[0]; cnt[k[3]] += 1; shape[k[3]][(k[4], k[5], k[6], k[7])] += 1
    print(f"request {ri}: {nwin} windows, {(t1-t0)/nwin/1e6:.2f} ms a window, kernels {sum(tot.values())/nwin/1e6:.2f} ms")
    for n, v in tot.most_common(28):
        print(f"  {v/nwin/1e3:8.1f} us  x{cnt[n]/nwin:6.1f}  {n[:44]:44s} {shape[n].most_common(1)[0][0]}")
