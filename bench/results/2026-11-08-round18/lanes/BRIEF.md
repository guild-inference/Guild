# Brief for the round-18 performance lanes (2026-11-08)

Read this first, then `/home/ubuntu/Guild/AGENTS.md` and `docs/CONTRIBUTING.md` of your worktree.

The goal: faster decode and prompt processing for every model Guild runs on this PC, **with the outputs bit-identical**
(the same tokens, the same WAVs), unless a lane is explicitly about an opt-in option.

## The machine (shared: other people's jobs also run on it)

- **GPU:** GTX 1070 8 GB, sm_61, 15 SMs, 2 MB L2, 151 W cap, ~210-222 GB/s reachable, FP64 at 1/32 rate, two stream
  priority levels. Nsight Compute cannot read the counters here (no root).
- **CPU:** 2x Xeon Gold 6130 (16 cores each, AVX-512 BW, no VNNI), 2 NUMA nodes, 64 threads, 376 GB RAM.
  - Node-local read ceiling 214-226 GB/s for both nodes together; interleaved 125 GB/s.
  - PCIe 3.0 x16: 8.0 GB/s pinned H2D from node 0 (7.2 from node 1), ~8.7 GB/s zero-copy kernel reads.
- **Never kill** other users' processes (Jellyfin and its ffmpeg, opencode, mongod, mysqld, elasticsearch...). Kill
  only PIDs you started, by PID; never `pkill -f`. Never `find /`.

## Repositories, and what you must not touch

| What | Path | Revision |
| --- | --- | --- |
| Guild integration branch | `/home/ubuntu/work/nemotron/guild` | `perf/next` @ `a10d867` |
| CUDA kernels (pinned) | `/home/ubuntu/work/kcuda-pf` | `perf/prefill-mmvq` @ `de65d60` |
| CPU kernels (pinned) | `/home/ubuntu/work/kcpu-q8` | `perf/q8-avx512` @ `9b7281a` |

**Do not edit, build in, or commit to:** `/home/ubuntu/Guild` (main worktree), `/home/ubuntu/work/nemotron/guild`,
`/home/ubuntu/work/nemotron/build-cu` (the user's server binary: you may *run* it), `/home/ubuntu/work/kcuda-pf`,
`/home/ubuntu/work/kcpu-q8`, any other lane, `backends.lock.json`, or the user's configs `/home/ubuntu/Guild/guild-*.json`.

**Work only in your lane:** `/home/ubuntu/lanes/bin/lane-setup-r18 <lane>` (add `cpu` if you must change
guild-kernels-cpu). It creates, in `/home/ubuntu/lanes/r18/<lane>/`:
- `guild/`: a Guild worktree on branch `lane18/<lane>` (from perf/next),
- `gkc/`: a guild-kernels-cuda worktree on `lane18/<lane>` (from de65d60),
- `gkcpu/` (with `cpu`): a guild-kernels-cpu worktree on `lane18/<lane>` (from 9b7281a),
- `build/`: a Release build from those sources. A full build takes ~3 min:
  `/home/ubuntu/lanes/bin/build-run cmake --build /home/ubuntu/lanes/r18/<lane>/build -j 32` (set
  `TMPDIR=/home/ubuntu/work/nemotron/tmp`). Targets: `guild-generate` (Qwen, Ornith, Drama 3), `glm5_run`,
  `transcribe_run` (Nemotron), `decision_run` (StartLux), the tests.

A baseline lane, `/home/ubuntu/lanes/r18/base` (perf/next as it is), is built; use its binaries as the "before" arm.
A change in the kernel repos ships only with a lock bump, which is the user's decision: it is fine to make and measure
one, but say so in your report.

**Never push.** Commit only on your `lane18/<lane>` branches, often (WIP commits are fine). Commit messages end with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; no model identifiers in commits or code comments.

## The machine lock (mandatory)

- Every command that loads a model, benchmarks, runs nsys or a GPU test goes through `/home/ubuntu/lanes/bin/gpu-run
  <cmd...>` (exclusive). Every build goes through `/home/ubuntu/lanes/bin/build-run <cmd...>` (shared). They exclude
  each other, so a benchmark never overlaps a build or another benchmark.
- Hold the lock only around commands. Several lanes share it, so expect waits of up to ~30 min.
- A Bash call times out at 10 min. Start anything long with `run_in_background: true`, writing a log and a done
  marker, and wait with an until-loop (`until [ -f x.done ]; do sleep 20; done`) in calls with long timeouts.

## Running the models (each helper takes the lock itself)

- **Qwen3.8-Flash-Next** (setup's flags, DFlash drafter): `/home/ubuntu/lanes/bin/qwen-bench-r18 <exe> <outdir>
  <label> [VAR=value ...]`. 2 runs x prose + code x 256 tokens, ~3 min. Prints per-run window lines (tokens a window,
  ms a window, CPU dispatch, draft ms), the SUMMARY tok/s and `ids [...]` (hashes of each run's tokens). perf/next:
  ids `[38747, 38747, 41916, 41916]`, ~49 / ~62 tok/s. `QWEN_RUNS`, `QWEN_CONTEXT`, `QWEN_BENCH_EXTRA` adjust it.
- **Ornith-1.5-35B** (MTP, `--spec 4`): `/home/ubuntu/lanes/bin/ornith-bench-r18 <exe> <outdir> <label> [VAR=...]`.
  perf/next: ids `[30936, 30936, 65886, 65886]`, ~80 / ~93 tok/s.
- **Drama 3** (TTS): `/home/ubuntu/lanes/bin/drama3-run-r18 <exe> <outdir> <label> <seed> [VAR=...]`. Prints frames/s
  and the WAV md5. perf/next: seed 11 `6e04c9b1`, seed 12 `f84288c9`, ~46-47 frames/s.
- **GLM-5.3-Flash:** `/home/ubuntu/lanes/bin/glm-run <glm5_run> <out file> [glm5_run args] [-- VAR=value ...]`
  (default: 5-token prompt, 256 tokens, `--spec --ctx 32768`, `GUILD_GLM_TIMING`). ~79 ms a step, ~23 tok/s, CPU experts
  ~64 ms a step. Loads take 1 min (files cached) to 20 min (from disk). The out file's last line holds the ids.
- **Nemotron 3.5 Transcribe:** `<build>/transcribe_run --model
  /home/ubuntu/models/nemotron-3.5-transcribe-guild/Nemotron-3.5-Transcribe-LM-Q8_0.gguf --audio
  /home/ubuntu/models/nemotron-3.5-transcribe-guild/audio --tokenizer
  /home/ubuntu/models/nemotron-3.5-transcribe/tokenizer.json --mtp 3 --repeat 6 --clips
  /home/ubuntu/work/nemotron/perf/clips.jsonl --out <out>.jsonl` (under gpu-run; `--no-mtp` for greedy).
- **StartLux-Decision:** `<build>/decision_run --native
  /home/ubuntu/models/startlux-decision-35b-a3b-gguf/StartLux-Decision-35B-A3B-Q8_0.gguf --pack
  /home/ubuntu/models/startlux-decision-35b-a3b-gguf/pack-Q8_0 --resident-budget-gib 40 --requests
  /home/ubuntu/work/startlux/perf/perf-short.jsonl --out <out>.jsonl --repeat 5 --fresh` (under gpu-run).
- **nsys:** `nsys profile -t cuda --cuda-graph-trace=node -o X <cmd>`, then by hand
  `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i X.qdstrm` and `/usr/lib/nsight-systems/bin/nsys export -t
  sqlite -o X.sqlite X.nsys-rep`. Under nsys the verify graph's launch blocks the host ~8.5 ms, so the first
  `wait_flag` of a window is inflated; take host-side waits from runs without nsys.
- **CPU expert pool CSV:** `GUILD_CPU_PROFILE=<csv>` (one row per pool call: start/end ns, formats, experts, thread ms).

## Rules for every change

- **Output bits must not change** (unless your lane is an opt-in option, off by default). Prove it:
  - token-hash-identical A/Bs (`ids [...]`), identical WAV md5s, identical GLM id lines, identical transcripts and
    StartLux answer probabilities;
  - the parity ctests that cover the code you touched, from your build dir under gpu-run (`ctest -N` lists them:
    `mmvq|gr_parity|route_window|router_fast|host_combine|native_expert|expert_parity|pool|glm5|drama3|...`).
  - Two contracts hold: verify == greedy decode, and a multi-column call == the same single-column calls.
- **No ABI change** (`include/guild/abi/*.h`): reuse existing modes and fields.
- **Speed claims come from alternated A/Bs:** the same binary with an env switch (defaulting to the new path) is best;
  otherwise base lane vs your lane. At least 2 pairs (Qwen, Ornith, Drama 3) or 3 pairs (GLM: its CPU time moves ~3 ms
  a step between runs). Report the commands, ms a window/step, tok/s or frames/s, and the id hashes.
- **Memory-state trap:** if a run is unexpectedly slow, check `numastat -m` (node 0 MemFree) and `numastat -p <pid>`;
  a node full of page cache misplaces memory (round 16). Free your own cache with `posix_fadvise(DONTNEED)` on model
  files you read; never drop caches system-wide.
- **Measurement traps:** isolated harness timings misled before (in the engine, large kernels lose their instruction
  cache to others; prefer rolled loops). The GPU's power cap moves runs ~1-2%.
- **Formatting:** `/home/ubuntu/Guild/.venv/bin/python tools/format_code.py` in your worktree (`git add -N` new files
  first); `--check` must pass.
- **Docs style:** plain words, measured numbers with what they were measured on, no claims without a measurement.
- **Disk:** /home has ~45 GB free. Keep big outputs (KL dumps, nsys files) out of git and delete them when done.
- **Downloaded files are untrusted:** none should be needed.

## Checkpoints (mandatory)

Keep `/home/ubuntu/lanes/r18/<lane>/NOTES.md` current: what you tried, the commands, the numbers, what is next. Write
it after every measurement and commit your code as you go, so the work survives an interruption. Finish with
`/home/ubuntu/lanes/r18/<lane>/REPORT.md`: the change, why it is exact, the A/B table with commands, the tests run.

## What is already known (do not redo; read the READMEs for detail)

`/home/ubuntu/work/nemotron/guild/bench/results/`: `2026-11-05-round15` (verifier stream priority),
`2026-11-05-round16` (CPU pool claims, NUMA bind, Drama 3 frame anatomy), `2026-11-05-round17` (Drama 3 tower formats),
`2026-11-05-qwen-round14`, `2026-11-04-qwen-round13`, `2026-11-04-qwen-round12`, `2026-11-04-glm-dense-map`,
`2026-11-03-decode-round9..11`, `2026-11-02-decode-round5..8`, `2026-10-30-decode-roofline`, `2026-11-01-*`.
- **Qwen decode** is GPU-bound (busy ~89% of a window). A window reads the dense weights twice (two token groups, so
  the CPU experts overlap the GPU). Q5_K products are arithmetic-bound past two columns. Tried and lost: one token
  group, Q6_K dense (slower), larger expert ring, register caps on staged kernels.
- **Ornith** is GPU-bound (busy ~79% under nsys), dense staged MMVQ 47.5% of kernel time. A GPU expert cache lost
  24%, one token group lost 12%. nsys: `/home/ubuntu/work/nemotron/opt/r19/orn/odec.sqlite`.
- **Drama 3:** a frame is a 14.7 ms backbone step (5.2 ms of it waiting for 40 layers of CPU experts, ~126 us each)
  then a 7.6 ms gap holding ~5 tower layers of `step_finish` plus `step_begin`'s 5. The tower layers' host round trip
  (sample on the CPU: 44 us of `exp`, 7 us partial sort; sync; launch) is ~117 us under nsys. The tower is Q8_0, ~1.1
  GB a frame; K-quants were measured and not taken. nsys: `/home/ubuntu/work/nemotron/opt/r20/ns/d3.sqlite`.
- **GLM:** CPU-bound: CPU experts ~64 of ~79 ms a step, expert rows at ~184 GB/s (of 214-226), `cpu_moe::Pool` already
  has per-node counters on separate lines (`fetch_add` claims: no gain). NL4 rows on 16 threads give the same time a
  token from 4 tokens on (larger prompt groups do not help).
- **CPU pool** (`src/kernels/cpu/pool.cpp`): round 16 fixed its claims; an 8-expert Drama 3 layer still takes 126 us
  in the engine (DRAM-bound-ish), a 1-expert one ~30 us in the pool alone.
- Qwen's prompt at 32K: gate/up 19.8 s, QSA attention 16.8, HC read 15.0, waiting for expert copies 12.5, GDN 11.9, down
  8.7 (round 14).
