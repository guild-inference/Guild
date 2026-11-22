"""winb.py <sqlite> [first_frac]: decode windows between sampler_greedy_kernel launches (< 150 ms apart) in the
trace's later part; per window: span, GPU busy (union), idle, and time by kernel class."""
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1]); frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.55
K = c.execute("select k.start,k.end,k.streamId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start").fetchall()
t_lo = K[int(len(K) * frac)][0]
marks = [i for i, r in enumerate(K) if r[3] == 'sampler_greedy_kernel' and r[0] >= t_lo]
cls = lambda n: ('wait CPU experts' if n.startswith('wait_flag') else 'dense mmvq' if 'mmvq' in n else
                 'hc read' if n.startswith('gr_') else 'GDN' if n.startswith('gdn') else
                 'draft head k4_mm/d8_mm' if n in ('k4_mm_k', 'd8_mm_k', 'q8_act_k') else
                 'bf16 gemv (router/indexer/PLE)' if n.startswith('bf16') else
                 'attention' if n.startswith('attn') or n in ('append', 'kv_append_kernel', 'resolve_kernel', 'block_scores_multi_kernel', 'block_topk_split_kernel', 'block_topk_merge_kernel', 'head_norm_rope_k') else
                 'routing/doorbell/copies' if n in ('route', 'doorbell_publish_kernel', 'copy_from_mapped_kernel', 'combine', 'gate', 'apply') else 'other')
per = collections.Counter(); nw = 0; span = busy = 0; spans = []
for a, b in zip(marks, marks[1:]):
    t0, t1 = K[a][1], K[b][1]
    if t1 - t0 > 150e6: continue
    nw += 1; span += t1 - t0; spans.append((t1 - t0) / 1e6)
    ce = t0; bz = 0
    for r in K[a + 1:b + 1]:
        per[cls(r[3])] += r[1] - r[0]
        s, e = max(r[0], ce), r[1]
        if e > s: bz += e - s; ce = e
    busy += bz
print(f"{nw} windows: span {span/nw/1e6:.2f} ms, GPU busy (union) {busy/nw/1e6:.2f}, idle {(span-busy)/nw/1e6:.2f}")
for k, v in per.most_common():
    print(f"  {v/nw/1e6:7.2f} ms  {k}")
spans.sort(); print("span p10/p50/p90 %.1f / %.1f / %.1f" % (spans[len(spans)//10], spans[len(spans)//2], spans[9*len(spans)//10]))
