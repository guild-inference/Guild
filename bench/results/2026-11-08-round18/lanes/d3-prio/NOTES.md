# d3-prio notes (round 18)

Lane: Drama 3 step_begin on a top-priority stream, codec paced/held, flush after step_begin.
Guild worktree: /home/ubuntu/lanes/r18/d3-prio/guild (branch lane18/d3-prio, from perf/next a10d867).
Build: /home/ubuntu/lanes/r18/d3-prio/build. Base arm: /home/ubuntu/lanes/r18/base/build.

## Implementation (commit c5562f5, WIP)
- Item 1: Drama3Tower gets prio_stream_ (cudaDeviceGetStreamPriorityRange's greatest, non-blocking); step_begin's
  launches (hidden copy, cand matvec, read_logits, layers 0..split with the residual head + host sample) go to it;
  step_finish and prefill keep stream_. matvec/layer_forward/step_layers/read_logits take the stream.
  set_step_priority(bool); load() defaults it from GUILD_DRAMA3_PRIO; synthesize sets it per call.
  prefill_chunk ends with cudaStreamSynchronize(stream_), so prefill -> first step_begin is ordered on the host.
- Item 2: Drama3Codec::set_hold(const std::atomic<bool>*, depth=2): gemm() keeps <= depth SGEMMs in flight (ring of
  4 events recorded after each GEMM, cudaEventSynchronize on GEMM issued-depth) and yields while *hold; hold_wait()
  also before the decoder's big snake / scale_add launches. nullptr = unpaced (today).
- Item 3: drama3.cpp: hold=true around step_begin; flush(false) after tower_step (the last frame flushes before
  the break as today). Pacing set before the decode loop, cleared (after joining codec_side) before the final
  flush(true) and on every return (guard).
- Item 4: GUILD_DRAMA3_PRIO=0: one tower stream, no pacing, flush before tower_step. GUILD_DRAMA3_CODEC_DEPTH
  (default 2; 0 = no in-flight bound, only the hold).

## Tests (lane build, under gpu-run; tests/t1.sh)
- drama3_tower_test (ref.npz.d, 40 teacher-forced steps): new line hashes every step's logits, layer outputs,
  frames and all K/V: PRIO=1, PRIO=0, unset all 8d3fea1524964286; every other line identical to the base build's
  test (cosines, KL 0.00147/0.00655, argmax 0.940, K/V 0 of 507 differ), PASS.
- drama3_codec_test rand.i32 chunks 7 and 16: lane == base (110.4 / 109.9 dB), PASS.

## A/B 1 (ab2.sh: one gpu-run for the batch, same binary, PRIO=1 default vs GUILD_DRAMA3_PRIO=0)
Runner d3run.sh = drama3-run-r18 without its own lock (the batch holds it). frames/s, md5:
| run | PRIO=1 | PRIO=0 |
| --- | ---: | ---: |
| s11 r1 | 46.00 6e04c9b1 | 47.60 6e04c9b1 |
| s12 r1 | 47.41 f84288c9 | 46.24 f84288c9 |
| s11 r2 | 47.68 6e04c9b1 | 47.32 6e04c9b1 |
| s12 r2 | 47.17 f84288c9 | 47.47 f84288c9 |
| mean | 47.07 | 47.16 |
(an earlier pair through drama3-run-r18, ab1.sh: s11 46.66 vs 47.08; killed after it, lock waits ~20 min a run)
CPU-profile runs (seed 11; the profile itself costs ~4%): PRIO=1 45.16, PRIO=0 44.63.
tools/phase.py (groups the CSV's pool calls 40 a step; position = step % 16, 15 = the flush step):
- PRIO=0: flush step between 7.06 ms (regular 3.93); pos 0 step 27.55 / between 17.51; pos 1 step 18.58.
  cycle 359.1 ms, ~30.3 ms over 16 regular frames.
- PRIO=1: flush step between 4.05; pos 0 step 27.01 / between 4.84; pos 1 step 24.89 / between 9.61.
  cycle 354.6 ms, ~25.8 ms over 16 regular frames.
So the between-steps share falls as planned, but the codec, now beside two backbone steps, costs those steps
nearly as much: net ~4.5 ms a cycle (0.28 ms a frame, ~1.3%), inside the run-to-run noise (+-1.5%).
Reading: step_finish already spans the whole backbone step (regular between = tower_join tail + step_begin), so
the GPU's idle time during the backbone step is used; the codec has no idle time to hide in and its blocks delay
each backbone kernel (no preemption).
Next: measure the codec chunk alone (OVERLAP=0 codec_ms) and a serialized variant (the chunk decoded right after
step_begin with the backbone waiting for it: only step_finish beside it).

## A/B 2 (ab3.sh, one gpu-run) and the codec alone
- GUILD_DRAMA3_OVERLAP=0 PRIO=0 (codec on the main thread, alone): codec 1302 ms for 30 chunks = ~43 ms a 16-frame
  chunk (2.7 ms a frame). So today's overlap already hides ~40% of it (cost ~30 ms a cycle, PRIO=1 ~26).
  Base nsys (/home/ubuntu/work/nemotron/opt/r20/ns/d3.sqlite): chunks 52-53 ms of kernel time under nsys; over the
  decode window non-codec kernels (without wait_flag_ge) busy 67.6%, codec 529 ms of which ~47% overlapped.
- GUILD_DRAMA3_CODEC_SYNC=1 (chunk right after step_begin, backbone waits): 44.86 / 45.19 / 44.73 / 44.62 frames/s
  (codec 1359 ms on the main thread): clearly worse; dropped.
| run | PRIO=1 | PRIO=0 |
| --- | ---: | ---: |
| s11 r3 | 47.90 6e04c9b1 | 46.35 6e04c9b1 |
| s12 r3 | 47.34 f84288c9 | 46.58 f84288c9 |
| s11 r4 | 46.87 6e04c9b1 | 46.43 6e04c9b1 |
| s12 r4 | 46.77 f84288c9 | 45.57 f84288c9 |
| mean | 47.22 | 46.23 (+2.1%) |
8 pairs (A/B 1 + 2): PRIO=1 47.15, PRIO=0 46.70: +0.96%, per-pair SD 2.2%, SE ~0.8%. Tower ms on the main
thread falls every run (s11 ~1905 vs ~2165).
Next: GUILD_DRAMA3_HOLD=between: hold the codec from the backbone's return (tower_join, where step_finish is the
critical path) through step_begin, so the codec runs only beside backbone steps (PRIO=1 pos 1 between 9.61 ms).

## A/B 3 (ab4.sh, one gpu-run): variants, 2 rounds x 2 seeds, round-robin, round 2 reversed
| arm | s11 r5 | s12 r5 | s11 r6 | s12 r6 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRIO=1 (default, depth 2) | 46.88 | 46.54 | 46.44 | 46.70 | 46.64 |
| HOLD=between | 47.03 | 47.00 | 47.11 | 47.94 | 47.27 |
| FINISH_PRIO=1 | 47.01 | 47.09 | 46.60 | 46.51 | 46.80 |
| CODEC_DEPTH=1 | 46.65 | 46.83 | 47.71 | 46.57 | 46.94 |
| CODEC_DEPTH=0 (spec's form: bound only while held) | 46.64 | 46.43 | 46.33 | 46.46 | 46.47 |
| PRIO=0 | 46.28 | 46.14 | 46.10 | 45.95 | 46.12 |
md5 6e04c9b1 (s11) / f84288c9 (s12) in every run of every arm.
All 12 PRIO=1/PRIO=0 pairs so far: +1.03% mean, SD 1.74, SE 0.50 (PRIO=1 46.98, PRIO=0 46.50).
Profiles (seed 11, phase.py cycle sums; regular frame ~20.5): PRIO=1 354.6 / 350.4 (two runs), HOLD=between 350.5
(pos 1 between 9.9 -> 6.6 but pos 2 step 16.5 -> 18.1), FINISH_PRIO 352.7 (pos 1 between 5.5, pos 2 step 18.3),
PRIO=0 359.1. The codec's cost moves between phases but its total stays ~22-26 ms a cycle.
Next: hb vs p1 vs p0, 3 rounds x 2 seeds, and mb3 (serve, chunk 16 vs 4096) per arm.

## A/B 5 (ab5.sh, one gpu-run, finished 09:54): hb / p1 / p0, 3 rounds x 2 seeds, order rotated; mb3 per arm
| arm | s11 r7 | s12 r7 | s11 r8 | s12 r8 | s11 r9 | s12 r9 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PRIO=1 | 46.79 | 46.53 | 46.43 | 46.40 | 46.43 | 46.93 | 46.59 |
| HOLD=between | 46.62 | 45.50 | 46.11 | 46.56 | 46.92 | 47.08 | 46.47 |
| PRIO=0 | 45.82 | 44.17 | 45.92 | 46.01 | 46.43 | 45.00 | 45.56 |
md5 6e04c9b1 / f84288c9 in every run (all 69 WAVs in ab/ of the lane: 38 x 6e04c9b1, 31 x f84288c9).
mb3 (serve, SAY chunk=16 vs chunk=4096, 3 each, 479 frames): PRIO=1 10456.7 vs 9586.1 ms = 1.82 ms a frame;
PRIO=0 10490.9 vs 9581.0 = 1.90; HOLD=between 10341.4 vs 9527.6 = 1.70 (scout base 1.90). Target <= 0.9: not met.
All pairs so far (A/B 1, 2, 3, 5): PRIO=1 vs PRIO=0 18 pairs +1.45% (SD 1.90, SE 0.45; 46.84 vs 46.19);
HOLD=between vs PRIO=0 10 pairs +2.20% (SE 0.44); HOLD=between vs PRIO=1 10 pairs +0.39% (SE 0.42): no gain.
Decision (takeover, 2026-11-08): PRIO=1 stays the default (clears the +1% stop rule; positions move: phase.py
flush-step between 7.06 -> 4.05 ms, pos 0 between 17.51 -> 4.84). HOLD=between stays opt-in (off). CODEC_SYNC and
FINISH_PRIO (both measured worse or equal) are removed (the uncommitted edit, finished and committed now).
cprof.py's position table is flat for PRIO=1 (its slow frames move between scout positions 14/15/0, so medians
miss them); phase.py (groups by step index) is the table to quote.

## Takeover, finishing (2026-11-08 ~10:00-)
- Committed the cleanup as 362150b (message amended with the final numbers: see git log) (removes CODEC_SYNC / FINISH_PRIO, HOLD=between opt-in; message finishes c5562f5's
  WIP). format_code.py --check: all 546 files formatted.
- The binary of 09:54 still holds the removed switches (strings), so it predates 362150b: rebuilding (build6.log),
  then ab6.sh under one gpu-run: tests/t2.sh (tower test PRIO=1/0/unset + base, codec test chunks 7/16, results in
  tests/final/) and p1 / hb / p0, 2 rounds x 2 seeds (r10, r11). The first build attempt waited 1 h for the lock
  without getting it (shared waiters starve behind exclusive ones) and was stopped; requeued.
- The REPORT.md write was refused by the harness (report files go into the lane's structured summary instead).

## A/B 6 (ab6.sh, final binary 362150b built 11:34, one gpu-run)
tests/t2.sh -> tests/final/: drama3_tower_test PRIO=1 / PRIO=0 / unset all 8d3fea1524964286, every other line equal
to the base build's test (diff empty: cosines, KL 0.00147 / 0.00655, argmax 0.940, K/V 0 of 507), PASS;
drama3_codec_test chunks 7 / 16: lane == base, 110.4 / 109.9 dB, worst sample error 0, PASS.
| run | PRIO=1 | HOLD=between | PRIO=0 |
| --- | ---: | ---: | ---: |
| s11 r10 | 47.82 | 47.03 | 47.57 |
| s12 r10 | 46.66 | 46.78 | 46.23 |
| s12 r11 | 46.75 | 46.85 | 46.46 |
| s11 r11 | 47.47 | 46.94 | 46.91 |
md5 6e04c9b1 / f84288c9 in every run; all 81 WAVs in ab/: 44 x 6e04c9b1 (seed 11), 37 x f84288c9 (seed 12).
Final binary alone: PRIO=1 vs PRIO=0 +0.82% (4 pairs, SE 0.15); HOLD=between vs PRIO=1 -0.57%.
All rounds: PRIO=1 vs PRIO=0 22 pairs +1.33% (SD 1.73, SE 0.37; 46.91 vs 46.30); HOLD=between vs PRIO=1 14 pairs
+0.11% (SE 0.34). Settled: PRIO=1 default (exact, +1.33%, positions move); HOLD=between opt-in, off; item 5 skipped.
Open: the HIP shim (include/guild/hip_compat/cuda_runtime.h) maps neither cudaDeviceGetStreamPriorityRange nor
cudaStreamCreateWithPriority; verify.cpp already uses both since round 15, the tower now does too (untested on HIP).
Lane done.
- 362150b was amended to 0841464 (same tree, final numbers in the message).
