import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1])
K = c.execute("select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.registersPerThread from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
# decode: kernels after the prompt: the last 60% of the timeline by count
n0 = int(len(K) * 0.1)
steps = 34
tot = collections.Counter(); cnt = collections.Counter(); sh = collections.defaultdict(collections.Counter)
for k in K[n0:]:
    tot[k[2]] += k[1]-k[0]; cnt[k[2]] += 1; sh[k[2]][(k[3],k[4],k[5],k[6],k[7])] += 1
span = (K[-1][1] - K[n0][0])
print(f"span {span/1e6:.0f} ms, kernels {sum(tot.values())/1e6:.0f} ms")
for n, v in tot.most_common(40):
    print(f"  {v/1e6:8.1f} ms  x{cnt[n]:7d}  {v/cnt[n]/1e3:7.1f} us  {n[:40]:40s} {sh[n].most_common(1)[0][0]}")
