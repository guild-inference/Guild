import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1])
L = c.execute("select r.start,r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like 'cudaGraphLaunch%' order by r.start").fetchall()
K = c.execute("select k.start,k.end,k.correlationId,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.streamId,k.dynamicSharedMemory,k.staticSharedMemory,k.registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
by = collections.defaultdict(set)
for k in K: by[k[2]].add(k[3])
ver = [s for s, cid in L if 'wait_flag_ge_kernel' in by[cid]]
t0, t1 = ver[len(ver)//4], ver[-1]
nwin = sum(1 for v in ver if t0 <= v < t1)
tot = collections.Counter(); cnt = collections.Counter(); mx=collections.defaultdict(list)
name = sys.argv[2] if len(sys.argv) > 2 else 'native_mmvq_staged_kernel'
for k in K:
    if t0 <= k[0] < t1 and k[3] == name:
        key = (k[4], k[5], k[6], k[7], k[8], k[9], k[11])
        tot[key] += k[1] - k[0]; cnt[key] += 1; mx[key].append(k[1]-k[0])
print(nwin, 'windows')
for key, v in tot.most_common(30):
    ds = sorted(mx[key])
    print(f"  grid {key[:3]} blk {key[3]} stream {key[4]} dsmem {key[5]} regs {key[6]}: {v/nwin/1e3:8.1f} us a window, x{cnt[key]/nwin:5.1f}, median {ds[len(ds)//2]/1e3:6.1f} us")
