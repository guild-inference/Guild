# Sixteenth round: the CPU pool's fixed cost a layer, the NUMA split under a full page cache, and Drama 3's frame

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap, PCIe 3.0 x16), CUDA 12. Branch `perf/next`, the pinned kernel revisions (guild-kernels-cpu
`9b7281a`, guild-kernels-cuda `de65d60`). Unless a section says otherwise, the A/Bs compare the build before this
round's changes (`gpri`, round 15's head) with the build after them.

## Drama 3: a layer's CPU experts cost 50 us before any work

Drama 3 decodes one token a frame, and its 40 layers send 8 routed experts each to the CPU pool (32 threads, rows
split by NUMA node). `GUILD_CPU_PROFILE` gives each layer's pool time. The table is from a run with a GPU expert cache
(the section below), which leaves the CPU a different number of experts in each layer (`runs/cprof_old.table`):

| Experts on the CPU | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Median us a layer | 61.7 | 64.7 | 72.5 | 81.6 | 93.7 | 104.0 | 115.6 | 125.8 |

An expert is 1.8-2.0 MB (Q4_K gate/up, Q4_K or Q6_K down), so one expert is about 2 us of reading for 32 threads.

**Where the 50 us went.** `tools/cpu_expert_workload` with Drama 3's layer 1 and an instrumented copy of the pool
(`runs/workload_inst.diff`, `runs/pool_inst.diff`): one expert, one token, 32 threads, 3,000 calls. Per thread, a
call held:

| Part | us |
| --- | ---: |
| gate/up rows | 6 |
| waiting for the expert's gate/up before its down rows | 4 |
| down rows | 5.4 |
| claiming tasks (5-6 claims) | 16-18 |
| adding to `done` | 3.5-4.7 |

The first task started 6.4 us after the publish, and every thread finished its last at about 47 us.

The claims were the cost, for three reasons:
- **CAS retries.** A claim was a CAS loop on a queue head that the node's 16 threads share.
- **`numa_first_` and `numa_batch_` shared node 1's `head1_` line.** Every claim reads them, so node 0's claims
  pulled that line across the sockets.
- **The other node's head.** Every claim after a thread's own gate/up tasks read it again, a remote line under
  constant writes.

**The change** (`src/kernels/cpu/pool.cpp`, `pool.hpp`):
- **One `fetch_add` a claim.** The claim reads the head first: the batch must be the worker's own and have an index
  left. A worker is out of the park while it claims, so the host cannot publish another batch before the add lands,
  and a late worker still fails its read (issue #29's case).
- **Hot fields on their own lines.** The NUMA fields get their own line. `hstate_`, which the host writes on every job
  it claims, moves off the line of the `wstate_` pointer that every worker reads at every claim.
- **Fewer shared reads and writes.** `claim_numa` remembers the queue parts it found exhausted, and `done_` is added
  once a drain.

Every task computes the same rows, so the results are the same.

One expert, 32 threads, us a call, for each step of the change:

| Pool | us a call |
| --- | ---: |
| before | 52.0 |
| claims by `fetch_add` | 45.7 |
| + the NUMA fields on their own line | 33.8-36.4 |
| + exhausted parts remembered, `done` once a drain | 34.5-34.9 |

Without the instrumentation, before against after, 2 runs each:

| Experts | Before | After |
| ---: | ---: | ---: |
| 1 | 56.6 / 49.7 | 30.1 / 31.3 |
| 4 | 64.9 / 59.4 | 50.8 / 54.4 |
| 8 | 88.0 / 87.0 | 88.3 / 89.9 |

At 8 experts the reading takes the claims' place: the engine's 8-expert layers still take 126 us
(`GUILD_CPU_PROFILE`, 125.9 against 125.8). The engine gains in the pool's small phases: a layer's activation
quantization and its sums.

**In the engine** (`runs/dpool.*`, `runs/batch1.*`): every run's outputs are the same in both arms.

Drama 3 (the text of round 15), frames/s:

| Seed | Before | After | WAV md5, both arms |
| --- | ---: | ---: | --- |
| 11 | 46.45 / 46.47 | 47.67 / 46.97 | `6e04c9b1` |
| 12 | 46.40 / 46.17 | 47.36 / 46.52 | `f84288c9` |

Qwen3.8 (setup's flags, DFlash) and Ornith-1.5 (MTP, `--spec 4`), tok/s, 2 rounds:

| Model | prose, before | prose, after | code, before | code, after |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 | 49.10 / 49.14 | 49.18 / 49.12 | 61.55 / 61.79 | 61.92 / 61.77 |
| Ornith-1.5 | 80.53 / 79.81 | 80.59 / 80.12 | 93.27 / 92.28 | 93.22 / 93.18 |

Both models are within their run-to-run spread.

**GLM's pool** (`cpu_moe::Pool`, which Nemotron and StartLux use too) already keeps its words and counters on
separate 128-byte lines. Its claims by `fetch_add` instead of CAS (`runs/cmpool.cpp`: 2 x 16 to 2 x 256 items of 1-5
us) measured within the run-to-run spread of the CAS, so it is unchanged.

## The NUMA split under a full page cache

During this round Qwen3.8 decoded 8-12% slower than round 15's runs of the same binary. CPU expert time rose from
27-37 to 40-54 ms a window. The cause was where the engine's memory was: 29 GB on node 0 and 70 GB on node 1
(`numastat -p`), where the split asks for about half on each.

The split places each node's rows by first touch from that node's threads. Node 0 had 30 GB free, and 155 GB of its
memory was page cache (other models' files). With Linux's default `zone_reclaim_mode` 0, a node short of free memory
hands a local allocation to the other node rather than reclaim its own cache. The resident arena (28 GB a node) still
fit, but the IQ3_S repack (another 19 GB a node) went to node 1.

**The fix** (`NodeBind`, `guild/kernels/cpu/numa.hpp`): the touching threads bind to their node (`MPOL_BIND`) for the
touch. The kernel then reclaims that node's cache. A node binds only when its free memory and file pages
(`/sys/devices/system/node/nodeN/meminfo`) hold its share with 4 GiB to spare. Otherwise it falls back as before,
rather than risk the OOM killer. The repack's load line says when a node was short.

Qwen3.8, the same flags, `gbind` (the fix) against `gpool` (the pool change only). Before each run, other models'
files were read into node 0's page cache (`numactl -m 0 cat`):

| Before each run | Arm | Node 0 free | prose tok/s | code tok/s | Sampled node-0 pages on node 0 | Engine, node 0 / node 1 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 42 GB read | bound | 28.1 GB | 49.41 | 61.96 | 1536/1536 | |
| 42 GB read | not bound | 31.1 GB | 45.87 | 55.17 | 1536/1536 | |
| 111 GB read | bound | 1.4 GB | 49.24 / 48.89 | 62.02 / 62.17 | 1536/1536 | 48.2 / 51.4 GB |
| 111 GB read | not bound | 1.0-1.4 GB | 44.79 / 43.80 | 53.93 / 53.55 | 24 and 3 of 1536 | 16.9 / 82.7, 10.7 / 88.8 GB |

- **The 42 GB arms.** The repack went to node 1 without the bind; the resident arena's sample cannot show it.
- **The 111 GB arms.** The resident arena moved too.
- **Size of the effect.** Bound decode is 8-16% faster, and its tokens are the same.
- **The 42 GB round 2.** It is left out: the bound run before it had already freed node 0, so its unbound arm had 50
  GB free and matched the bound one (`runs/qbind.progress`).

Ornith and Drama 3 split 18 GB of experts; their loads' samples found every page on its node in this round's runs.
GLM's arenas (`cpu_moe`) were already bound per node.

## Drama 3: a GPU expert cache, and where a frame goes

**An expert profile.** Drama 3's routing on 8 texts unlike the bench's (`runs/trace.sh`, `runs/texts.txt`; 2,486
tokens) is concentrated: the hottest 10% of (layer, expert) pairs serve 41.6% of lookups, and 20% serve 59.9%. With
that profile and the local kernel branch that lets the GPU compute Q4_K and Q6_K experts (`perf/q6k-experts`, round
15), `--expert-cache auto` kept 1,004 slots (1.9 GB) of 10,240:

| Arm | frames/s |
| --- | ---: |
| no cache | 45.46 / 45.33 |
| cache | 45.60 / 45.64 |

The cache arm's WAV differs (the GPU's rounding); its run is `runs/cab.*`. The CPU's layers got cheaper (the table
above is that run), but the frame barely moved. Not shipped.

**Why.** nsys (`runs/d3nsys.sh`; 180 frames, 24.2 ms a frame under the profiler, against 21.3 without it). A frame is
the backbone's step, then the tower's first layers, then the next step:

| Part | ms a frame |
| --- | ---: |
| backbone step, first to last kernel | 14.7 |
| of which its kernels | 7.9 |
| of which waiting for the CPU experts (`wait_flag`) | 5.2 |
| of which gaps | 1.6 |
| between steps | 7.6 |
| of which tower kernels | 5.6 |
| tower kernels during the step | 2.0 |

- **What runs between steps.** The tower's `step_finish` (codes 2-9, 8 layers, a host sample after each head) runs
  beside the step, but only about 3 of its layers fit. The other ~5 run after the step, then `step_begin`'s 5 layers
  (the main token, codes 0-1) before the next step can start.
- **The tower's size.** 13 layers of 78 MB and 10 heads of 8.9 MB, all Q8_0: about 1.1 GB a frame, the largest read
  on the GPU.
- **What that means.** The frame is set by the GPU's reads, the CPU experts' 5.2 ms and the tower's sequential
  samples. Moving expert work to the GPU trades one for another.

A K-quant tower would read 23-47% fewer bytes (Q6_K to Q4_K), but it changes the audio, so it is not done here.

## Files

`runs/`:
- **The pool.** `workload_inst.diff`, `pool_inst.diff` (the instrumented pool), `dpool.*`, `batch1.*` (Drama 3
  profiles, then Ornith and Qwen A/Bs, then nsys), `qpool.*` (Qwen's pool A/B taken under the full page cache before
  its cause was known: both arms slow, the same as each other), `cmpool.cpp`.
- **The NUMA split.** `qnuma.sh`, `qnuma.mon`, `qbind.*`, `qbind2.*`, `qb*_[34].mon`, `qb.sh`.
- **Drama 3.** `trace.sh`, `texts.txt`, `cab.*`, `cprof_old.table`, `d3nsys.sh`.
