# glm-pipe (round 18) notes

Lane: GLM decode pipeline. Worktree `guild/` on `lane18/glm-pipe` (from perf/next a10d867), build `build/`.
Before arm: `/home/ubuntu/lanes/r18/base/build/glm5_run`.

## Log

- 06:40 UTC: lane set up, first build OK (build0.log).
- Item 1 (side stream for group A's GPU shared expert): implementing. Design: st created at the greatest
  priority, `side` at the least; A's quant(xn -> xq_s) on st before ev_route (B's pre-mix rewrites xn), then
  gate|up, swiglu, quant, down on `side` with its own scratch (xq_s, xq2_s, a_s, b_s, c_s, for up to kSideT
  tokens), ev_join; ffn_end waits ev_join on st before the CPU rows' add / hc_post. Switch
  GUILD_GLM_SHEXP_STREAM (read each window).

### Item 1 measured (07:14 UTC): no gain, left off

`glm-run ab/glm5_run.s1 ab/s1.out 785,6722,315,9621,374 256 --ctx 32768 --spec --ab GUILD_GLM_SHEXP_STREAM=1,0
--reps 3 -- GUILD_GLM_KEEP_CACHE=1` (binary at f958540; load 947 s, files partly evicted from the page cache).

| round | SHEXP=1 step p50 / CPU p50 | SHEXP=0 step p50 / CPU p50 | diff |
| --- | --- | --- | --- |
| 1 | 85.5 / 71.1 (first decode, p90 117.7) | 79.2 / 64.7 | +6.3 |
| 2 | 80.4 / 67.2 | 79.6 / 65.8 | +0.8 |
| 3 | 80.6 / 67.6 | 80.4 / 67.5 | +0.2 |

Median +0.8 ms (worse); ids identical in every run. The step follows the CPU experts (step - CPU ~13 ms in both
arms), so taking 38 us/layer off the GPU half does not show. Default changed to off (GUILD_GLM_SHEXP_STREAM=1 opts
in); kept in the code for the combined run. glm5_run now prints GUILD_GLM_TIMING totals after every arm.

Items 2 (lean IQ4_XS down, GUILD_IQ4XS_LEAN), 3 (IQ4_XS gate/up through iq4xs_rows512, GUILD_IQ4XS_GU) and 4a
(kda_fg: f_b + gates + g_b in one launch, GUILD_GLM_FUSE2 bit 1) implemented; building.

### 07:30-08:05 UTC: items 2-5 implemented, tests

- Item 2 (lean IQ4_XS down, `GUILD_IQ4XS_LEAN`, default on): `iq4xs_rows512(..., lean)`; the lean loop is unrolled
  (`#pragma GCC unroll 8`, as the scout's x3 was measured), the first kernel's loop as it was.
  `iq4xs_rows512_test`: ok, 30672 results (lean and first, n 256/2048/4096, 1-8 tokens), 0 differ from ggml's dot.
- Item 3 (IQ4_XS gate/up through iq4xs_rows512, `GUILD_IQ4XS_GU`, default on), in gu_item.
- Item 4a (`GUILD_GLM_FUSE2` bit 1): `glm5k::kda_fg`, q8_0_narrow_k<.., Pair=true>: f_b's blocks write the gates
  (kda_gates' expression) and block 0 beta, g_b's blocks the output gate; one quantization of f_a|g_a (consecutive
  in the fgb product's output, Q8_1 blocks independent).  glm5_kernels_test: kda_fg bitwise for n_in 64/128/256,
  D 8192/1024, both one-wave settings.
- Item 4b (`GUILD_GLM_FUSE2` bit 2): hc_pre_k (attention and FFN inputs), kda_out_k, rms_norm_k (DSA q_a) and
  swiglu_clamp_k write their output's Q8_1 for windows of up to 8 tokens.  native_quantize_q8_1 is compiled with
  --use_fast_math (its PTX: abs/max/add .ftz, div.approx.ftz, roundf as add.rz.ftz + cvt.rzi), glm5.cu is not, so
  q8_1_warp spells those instructions in inline PTX.  glm5_kernels_test: every fused output and Q8_1 byte-equal to
  the unfused kernel + native_quantize_q8_1, T 1/2/8, including all-zero, subnormal and >65504-sum blocks.  PASS.
- Item 5 (`GUILD_CPU_MOE_REUSE`, default on): Experts keeps the per-call containers (groups with their vectors,
  units, the activation offsets as a small vector instead of a std::map, items, per-group item lists, the tail,
  qbase, token_units); =0 makes them afresh each call.  No arithmetic change.
- Runs queued: ab2 (7 arms: new / LEAN=0 / GU=0 / FUSE2=2 / FUSE2=1 / SHEXP=1 / all off, --reps 4, binary at
  1fd4904+4b = glm5_run.ab2), ab3 (new / REUSE=0 / all off, --reps 4, binary at 12ee0ce), then base for the ids;
  cross-model (Nemotron, StartLux base vs lane) and ctest -R 'glm5|pool|cpu_moe|iq4'.

### ab2 (08:28 UTC): seven arms, one process, --reps 4 (binary glm5_run.ab2 = 1fd4904 + 4b, before item 5)

`glm-run ab/glm5_run.ab2 ab/ab2.out 785,6722,315,9621,374 256 --ctx 32768 --spec --ab GUILD_GLM_SHEXP_STREAM=0,0,0,0,0,1,0
--ab GUILD_IQ4XS_LEAN=1,0,1,1,1,1,0 --ab GUILD_IQ4XS_GU=1,1,0,1,1,1,0 --ab GUILD_GLM_FUSE2=255,255,255,2,1,255,0
--reps 4 -- GUILD_GLM_KEEP_CACHE=1`; load 948 s; "A/B ids: identical in every run" (28 decodes).

Step p50 by round (arm order forward in rounds 1 and 3, reversed in 2 and 4):

| arm | r1 | r2 | r3 | r4 | mean | pairs vs new, r2-r4 (median) |
| --- | --- | --- | --- | --- | --- | --- |
| new (all on, side off) | 77.9 | 80.8 | 81.0 | 79.3 | 79.75 | |
| LEAN=0 | 77.8 | 81.5 | 82.0 | 79.6 | 80.23 | +0.7 +1.0 +0.3 (+0.70) |
| GU=0 | 78.9 | 80.8 | 81.4 | 79.7 | 80.20 | +0.0 +0.4 +0.4 (+0.40) |
| FUSE2=2 (4a off) | 79.0 | 81.0 | 80.9 | 79.7 | 80.15 | +0.2 -0.1 +0.4 (+0.20) |
| FUSE2=1 (4b off) | 79.1 | 80.7 | 80.9 | 79.6 | 80.08 | -0.1 -0.1 +0.3 (-0.10) |
| SHEXP=1 (side on) | 79.7 | 80.3 | 80.6 | 79.7 | 80.08 | -0.5 -0.4 +0.4 (-0.40) |
| all off | 80.9 | 80.8 | 81.2 | 80.8 | 80.93 | +0.0 +0.2 +1.5 (+0.20) |

The drift within the process (CPU experts p50 61 -> 68 -> 65 ms) is larger than the effects, and with seven arms
a run's position in its round moves it by up to 3 ms (round 1 rises monotonically 77.9 -> 80.9 in arm order).
Fits with a smooth drift (tools/abfit.py, polynomial of degree 3 / 4 in the run index): all off +0.51 / +1.00 ms
step, +0.96 / +0.95 CPU; LEAN=0 +0.29 / +0.45 step, +0.63 / +0.62 CPU; GU=0 +0.10 / +0.39 step, +0.51 CPU;
4a off -0.07 / +0.30; 4b off -0.24 / +0.19; side on -0.31 / +0.16 (+-0.2-0.35).  GUILD_GLM_TIMING per step (new
arm): activations 1.2-1.45 ms, hidden (the flow setup) 1.75-2.2 ms, the pool phase 59-72 ms, router waits
1.5-4.7 (KDA) + 1.1-1.9 (DSA) ms.

Stop rule (median of step p50 pair differences >= 0.5 ms): LEAN kept (+0.70); GU (+0.40), 4a (+0.20), 4b (-0.10)
and the side stream fail it; fusions (c)/(d) not started (neither (a) nor (b) reached 0.3 ms).

### ab3 (08:35 UTC): item 5 and all on vs all off, --reps 4 (binary glm5_run.ab3 = 12ee0ce)

`glm-run ab/glm5_run.ab3 ab/ab3.out 785,6722,315,9621,374 256 --ctx 32768 --spec --ab GUILD_GLM_SHEXP_STREAM=0,0,0
--ab GUILD_IQ4XS_LEAN=1,1,0 --ab GUILD_IQ4XS_GU=1,1,0 --ab GUILD_GLM_FUSE2=255,255,0 --ab GUILD_CPU_MOE_REUSE=1,0,0
--reps 4 -- GUILD_GLM_KEEP_CACHE=1`; load 328 s; ids identical in every run (12 decodes).

| arm | r1 | r2 | r3 | r4 | pairs vs all on, r2-r4 (median) | CPU p50 r1-r4 |
| --- | --- | --- | --- | --- | --- | --- |
| all on (REUSE=1) | 77.8 | 78.6 | 78.8 | 79.4 | | 58.5 62.9 63.2 64.4 |
| REUSE=0 | 77.8 | 78.8 | 79.3 | 79.9 | +0.2 +0.5 +0.5 (+0.50) | 59.4 64.1 65.5 66.7 |
| all off | 79.5 | 79.7 | 80.8 | 80.8 | +1.1 +2.0 +1.4 (+1.40) | 64.6 65.2 67.6 67.6 |

Summary lines: all on 78.8 ms p50, 23.32 tok/s; REUSE=0 79.2, 23.25; all off 80.5, 22.92.  Fit (degree 2/3):
REUSE=0 +0.28 +- 0.12 ms step, +1.42 CPU; all off +1.52 +- 0.12 step, +3.61 +- 0.5 CPU.  Per step, REUSE=1 cuts
the timed setup phases from 1.15-1.22 (activations) + 1.60-1.67 (hidden) ms to 0.83-1.03 + 0.93-1.09.  With the
CPU shorter the router waits grow (KDA 3.3-7.2 ms a step against 2.1-3.6 all off): the GPU half is the limit again
in part of the layers.  Item 5 kept (+0.50 median, the rule's threshold).  ab4 queued: GU, FUSE2 and the side
stream again with LEAN and REUSE on.

### Base binary for the ids (09:05 UTC)

`glm-run /home/ubuntu/lanes/r18/base/build/glm5_run ab/base.out -- GUILD_GLM_KEEP_CACHE=1` (5-token prompt, 256,
--ctx 32768 --spec): step 79.3 ms p50, 22.66 tok/s (one decode right after the load).  Its id line's md5 is
5714f4a1d575..., the same as every s1 / ab2 / ab3 arm's (and the scout's ab1.log).

IQ4_XS `--dram 32 1`: ggml vec_dot 142.0 GB/s, iq4xs_rows512 first 184.5, lean 190.9 (32 threads, 2,048-value rows).
Cross-model: Nemotron and StartLux base runs done; lane runs queued.

IQ4_XS `--dram 32 2`: ggml 88.5 GB/s, first 155.9, lean 165.6.  Nemotron transcribe_run (BRIEF command, --mtp 3
--repeat 6): base vs lane ids identical in all 12 records (ab/nem_*.jsonl).  StartLux lane run and ab4 waiting for
the lock (09:48 UTC).

### ab4 (10:14 UTC): GU, FUSE2 and the side stream again with LEAN and REUSE on (binary glm5_run.ab3 = 12ee0ce)

(Taken over after the usage-limit stop.)  `glm-run ab/glm5_run.ab3 ab/ab4.out 785,6722,315,9621,374 256 --ctx 32768
--spec --ab GUILD_GLM_SHEXP_STREAM=0,0,0,1,0 --ab GUILD_IQ4XS_LEAN=1,1,1,1,1 --ab GUILD_IQ4XS_GU=1,0,1,1,0 --ab
GUILD_GLM_FUSE2=255,255,0,255,0 --ab GUILD_CPU_MOE_REUSE=1,1,1,1,1 --reps 4 -- GUILD_GLM_KEEP_CACHE=1`; load 902 s;
"A/B ids: identical in every run" (20 decodes), id line md5 5714f4a1d575bd906a4f5b5421eac78d (= base).  Round 1's
first run is the first decode after the load (98.0 ms p50): round 1 skipped.

| arm | r2 | r3 | r4 | pairs vs all on (median) | fit r2-r4, deg 1 / 2 |
| --- | --- | --- | --- | --- | --- |
| all on (side off) | 79.2 | 79.3 | 79.7 | | |
| GU=0 | 79.4 | 79.9 | 80.3 | +0.2 +0.6 +0.6 (+0.60) | +0.50 / +0.48 |
| FUSE2=0 | 79.4 | 80.0 | 80.6 | +0.2 +0.7 +0.9 (+0.70) | +0.67 / +0.64 |
| SHEXP=1 | 78.8 | 80.0 | 80.5 | -0.4 +0.7 +0.8 (+0.70) | +0.47 / +0.45 |
| GU=0 and FUSE2=0 | 79.5 | 80.6 | 80.7 | +0.3 +1.3 +1.0 (+1.00) | +1.01 / +1.01 |

(The fit is tools/abfit.py on ab4.r24.out, the rounds 2-4 only.  Rounds 2 and 4 run the arms in reverse, so the
all-on arm runs last in two of the three rounds while the process drifts up ~1.5 ms: the pairs lean against it.)  Summary
lines (4 runs each, round 1 included): all on 79.6 ms p50; GU=0 79.7, 23.16 tok/s; FUSE2=0 79.8, 23.07; side on
79.6, 23.17; both off 80.3, 23.00.  With the CPU shorter (LEAN and REUSE on), GU and the fusions now clear the 0.5 ms
rule; the side stream still loses (+0.70 median, +0.45 fit; step - CPU -1.1 but CPU +0.6).

Decisions so far: side stream off (and now made at load only when GUILD_GLM_SHEXP_STREAM is set: commit 8987e69,
a default load allocates none of its ~160 KB); LEAN, REUSE kept; GU and FUSE2 kept pending ab5.

### ab5 queued (10:21 UTC): the final build (8987e69), ctests + one --ab, --reps 5

ab/ab5.sh: ctest -R 'glm5|pool|cpu_moe|iq4' + glm5_kernels_test, then arms all on / FUSE2=0 / FUSE2=2 (4a off) /
FUSE2=1 (4b off) / GU=0 / everything off (LEAN=0 GU=0 FUSE2=0 REUSE=0), GUILD_GLM_SHEXP_STREAM unset.  StartLux
lane run (xm.sh) still waiting for the lock.

### StartLux cross-check (10:52 UTC): identical

decision_run (BRIEF command, --repeat 5 --fresh), base vs lane build at 12ee0ce (cpu_moe.cpp as final): answers
byte-equal in all 4 records (md5 of the answers 537328d7cf48d560bdaeae58cbd95dee both; tools/xmcmp.py: "sl: 4
records, identical").  Nemotron: 12 records identical (md5 of the ids 1f7a49a5839ddf90b52343564a71df71 both).
Build of 8987e69 and ab5 still waiting for the lock (nem-decode holds it).

### 11:40 UTC: finishing commit 3a8f4df (comments only)

3a8f4df finishes 1fd4904 / 12ee0ce: the measured numbers in the switches' comments, no code change (the diff has
no non-comment line).  format_code.py --check: all 546 files formatted.  The waiting build (build5) compiles
3a8f4df, so ab5 runs the final code.

### ab5 (12:24 UTC): the final build (3a8f4df; GUILD_GLM_SHEXP_STREAM unset), one process, --reps 5

ab/ab5.sh under one gpu-run: ctest -R 'glm5|pool|cpu_moe|iq4' (7 pass, pool_stress skips itself; ab/ctest5.log),
glm5_kernels_test PASS (ab/t_glm5k5.log), then `GUILD_GLM_TIMING=1 GUILD_GLM_KEEP_CACHE=1 ab/glm5_run.ab5 <GLM>
785,6722,315,9621,374 256 --ctx 32768 --spec --ab GUILD_IQ4XS_LEAN=1,1,1,1,1,0 --ab GUILD_IQ4XS_GU=1,1,1,1,0,0 --ab
GUILD_GLM_FUSE2=255,0,2,1,255,0 --ab GUILD_CPU_MOE_REUSE=1,1,1,1,1,0 --reps 5`.  Load 996 s (node 0 MemFree 20.7 GB
before it; the experts 77.9 + 77.9 GiB, numastat -p 152 / 155 GB private on nodes 0 / 1).  "A/B ids: identical in
every run" (30 decodes), id line md5 5714f4a1d575bd906a4f5b5421eac78d (= base).  The first decode was not slow
this time (77.5).

| arm | r1 | r2 | r3 | r4 | r5 | pairs r2-r5 (median) | with r1 | fit deg 2, all / r2-r5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all on | 77.5 | 79.0 | 79.3 | 78.7 | 78.8 | | | |
| FUSE2=0 | 77.9 | 79.4 | 79.8 | 79.4 | 79.3 | +0.4 +0.5 +0.7 +0.5 (+0.50) | +0.50 | +0.47 / +0.54 |
| FUSE2=2 (4a off) | 78.3 | 79.1 | 79.6 | 79.3 | 79.1 | +0.1 +0.3 +0.6 +0.3 (+0.30) | +0.30 | +0.36 / +0.38 |
| FUSE2=1 (4b off) | 78.2 | 78.8 | 79.6 | 79.4 | 79.3 | -0.2 +0.3 +0.7 +0.5 (+0.40) | +0.50 | +0.34 / +0.43 |
| GU=0 | 78.6 | 78.9 | 79.8 | 79.7 | 79.2 | -0.1 +0.5 +1.0 +0.4 (+0.45) | +0.50 | +0.52 / +0.63 |
| all off | 79.6 | 79.8 | 81.6 | 81.5 | 80.0 | +0.8 +2.3 +2.8 +1.2 (+1.75) | +2.10 | +1.80 / +2.04 |

(fit +- 0.17-0.18.)  Summary lines (5 runs each): all on 78.9 ms p50, 23.28 tok/s; FUSE2=0 79.4, 23.25; 4a off
79.2, 23.21; 4b off 79.2, 23.28; GU=0 79.4, 23.26; all off 80.8, 22.79.  Per step, all on vs all off: activations
0.72-0.95 vs 1.04-1.18 ms, hidden setup 0.91-1.11 vs 1.55-1.75, KDA router waits 3.3-6.9 vs 1.7-4.2 (the GPU half
is the limit more often once the CPU is shorter); CPU experts +2.95 ms median all off.

Decisions (final):
- side stream: off (opt-in, made at load only when its variable is set).
- LEAN: kept (ab2 +0.70).
- GU: kept, borderline: ab2 +0.40 (before REUSE), ab4 +0.60, ab5 +0.45 (r2-r5) / +0.50 (all rounds); fits +0.48-0.63.
- FUSE2: kept as one item (both bits): ab4 +0.70, ab5 +0.50; each bit alone is under the rule (ab5 +0.30 / +0.40,
  ab2 +0.20 / -0.10), and the two do not add (they shorten the same GPU half, capped by the CPU).
- REUSE: kept (ab3 +0.50).
- Total, all on vs all off: ab3 +1.40 median (fit +1.52), ab5 +1.75 (fit +1.80 / +2.04): 22.79 -> 23.28 tok/s in ab5.

12:25 UTC: commit 268bcb6 corrects the GU / FUSE2 comments to both processes' ranges (comments only;
the build dir is 3a8f4df's code, unchanged by it).  format_code.py --check passes.
