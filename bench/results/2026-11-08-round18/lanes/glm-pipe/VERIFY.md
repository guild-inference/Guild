# glm-pipe (round 18): independent verification

Verifier, 2026-11-08. Branch `lane18/glm-pipe` at 268bcb6 (perf/next a10d867 + 6 commits); the kernel worktree `gkc/`
has no change against de65d60, so no lock bump is needed. I did not edit the lane's source or commit to its branches.
My files are in `/home/ubuntu/lanes/r18/glm-pipe/verify/`.

## Verdict

| item | lane's call | verdict |
| --- | --- | --- |
| 1. group A's shared expert on a side stream | not kept (opt-in) | not kept: agree. No gain in the lane's three processes. The opt-in code reviewed clean. |
| 2. lean IQ4_XS down rows (`GUILD_IQ4XS_LEAN`) | kept | **accept** |
| 3. layer 11's IQ4_XS gate/up through iq4xs_rows512 (`GUILD_IQ4XS_GU`) | kept | **accept** (borderline gain) |
| 4a + 4b. KDA gates in one launch, Q8_1 written by the producers (`GUILD_GLM_FUSE2`) | kept, as one item | **accept** as one item (borderline gain; I did not measure each bit) |
| 4c / 4d | not started | n/a |
| 5. cpu_moe::Experts keeps its containers (`GUILD_CPU_MOE_REUSE`) | kept | **accept** |

The total holds up. Everything on against everything off, in one process: +1.80 +- 0.30 ms a step (fit), with a pair
median of +2.00 ms. That is 23.32 against 22.84 tok/s (+2.1%). The ids are identical in all 30 decodes, and the id
line matches the base binary's.

## 1. Code review (diff perf/next...HEAD)

**Item 2, the lean rows (iq4xs_rows512.cpp `row2<NT, Lean>`).** I checked the lean path step by step against the
first one:
- **Nibbles.** `srlv_epi64` uses shifts (0,0,4,4,0,0,4,4) by 64-bit element, which is lanes 0/2 low and lanes 1/3
  high. Then `& 0x0f`. That is the same nibble as `blend(0xCC, x & 0xf, (x >> 4) & 0xf)`.
- **Magnitudes.** `pshufb(kAbs, nib)` equals `abs(pshufb(kValues, nib))` (nib < 16).
- **Sign.** `testn(nib, 8)` is set exactly for codes 0-7, which are the negative codebook values. So it equals
  `movepi8_mask(w)`.
- **Scales.** `shuffle_i64x2(sc16, sc16, 0x50)` puts row 0's eight scales in lanes 0-1 and row 1's in lanes 2-3. It
  reads only the defined low 256 bits of `castsi256_si512`. The byte pair (2h, 2h+1) then picks word h, which is the
  same value as the old `permutexvar` index h / 8+h.
- **Float steps.** The code after the sub-block loop is untouched.

`GUILD_IQ4XS_LEAN=0` calls `rows_any<false>`, the old body.

**Item 3, gate/up (cpu_moe.cpp gu_item).** `x4` needs `iq4xs_rows512_ok()` (AVX-512BW) and gate/up of type IQ4_XS,
so AVX2 machines and every other type keep the old path. With the switch off, `x4` is false and the old
`mt` / vec_dot chain runs unchanged. iq4xs_rows512 gives every (row, token) ggml's one-token dot, so this also holds
in `single` (verify) windows. Those used the vec_dot loop before.

**Item 4a, kda_fg.** `kda_fg` uses the same `q8_0_narrow_k` body, the same C choice
(`one_wave() && D >= 4096`) and the same grid per matrix, so each row's sum is unchanged. The gate epilogue is
character for character `kda_gates_k`'s expression on the value that used to go through `fpre`, and beta is the same
`sigmoidf_(bpre)`. One quantization of f_a|g_a gives the same blocks, because Q8_1 blocks are independent.
`native_q8_1_bytes` has no row padding. Nothing reads `a` (fpre) after the gates in attn_kda. The path is guarded
by T == 1, both matrices Q8_0, consecutive f_a/g_a and the supported n_in. The attention conversion to Q6_K touches
only wq/wk/wv/wo/q_b, so f_b and g_b stay Q8_0 and the fused path does run in the default config. In --spec, each
2-token verify group is one token.

**Item 4b, the Q8_1 producers.** I dumped the PTX of `native_quantize_q8_1_kernel` from the lane build's
native_mmvq.cu.o. Its sequence is:
- abs.ftz;
- max.ftz and add.ftz butterflies with offsets 16-8-4-2-1 and the same operand order;
- div.approx.ftz;
- q8_1_finite as an abs.ftz / setp.gt.ftz select;
- setp.eq.ftz for amax == 0;
- div.approx.ftz, then add.rz.ftz of copysign(0.5), then cvt.rzi.f32;
- the clamps, then cvt.rzi.ftz.s32;
- cvt.rn.f16 for d and sum.

`q8_1_warp` in glm5.cu compiles to the same sequence. The only differences are the non-ftz abs/compares on d, amax
and the clamped t. Those values are already flushed or integral at that point, so the results are the same.

I also diffed the base build's and the lane build's PTX for `swiglu_clamp_k`, `kda_out_k`, `rms_norm_k` and
`hc_pre_k`. In each, the instructions that compute the stored output (y / xn) are the base ones; the Q8_1 code comes
after them.

Layout: blocks are contiguous over the T rows, `(t * n + i) / 32 * 36`, as `native_q8_1_bytes` lays them out. For a
warp-uniform loop the kernels need n % 32 == 0, and every caller checks it. `rms_norm_q8_1_supported` mirrors the
`rms_norm` dispatcher (`rms_norm_k` only), so the norm's sum order is the same.

The `xn_q` handshake:
- `hc_pre_half` clears `xn_q` on entry.
- `hc_pre_half` sets it only in the fused branch, and only when its caller passes the buffer of a product that runs
  next. `layer_ffn_begin` passes q exactly when `gpu_ffn` holds; layer_attention always reaches attn_kda / attn_dsa.
- The first `quant_xn` consumes it and clears it.
- `mtp_window`'s attn_dsa has no pre-mix, so it sees nullptr and quantizes.
- Only the thread that runs `layer_attention` / `layer_ffn_begin` touches `xn_q`; the calling thread's
  `ffn_end` never does.

I found no stale read.

**Item 5, REUSE.**
- **Offsets.** Activation offsets were already given out in first-met order, so `act_off` as a first-met vector gives
  the same offsets. Only the order of the quantization jobs changes, and each job writes its own bytes.
- **Leftover entries.** `per` and `token_units` entries past this call's size are cleared and never read: the spread
  loop skips empty lists, and `token_units` is read for t < T only.
- **Groups.** Reused groups have every vector cleared and `f` reassigned.
- **Switch off.** `GUILD_CPU_MOE_REUSE=0` builds a fresh Scratch.
- **Threads.** `Experts::run` is never called concurrently on one instance: GLM calls it from ffn_end on the calling
  thread, and Nemotron calls it sequentially. `actbuf_` already required that.

**Item 1, the side stream (opt-in).**
- **Default load.** It creates nothing (`want_side` = the variable set at load).
- **Events.** A's input is quantized on st before `ev_route`. ffn_end waits on `ev_join` before the add, and it uses
  `io.side`, not the launcher-owned `side_grp`. The ring of 2 event pairs is safe: the next write of `xq_s` / `a_s`
  comes on st after that wait.
- **Shared scratch.** The kernels the side stream runs (glm5k staged / narrow, native_mmvq) use no global scratch.
- **Default path.** It changes in one way: st is now created at the greatest priority. With st the engine's only
  stream, that orders nothing. The lane's A/Bs cannot see it (every arm shares the load-time stream), but the ids
  match the base binary.

**Other models.** These are the callers of everything the lane changed:
- `Experts::run`: glm5.cpp and nemotron_h.cpp.
- `iq4xs_rows512`: cpu_moe.cpp only.
- `dense_mm.hpp`: glm5.cpp and nemotron_h.cpp. The new stream arguments default to `st_`.
- glm5.cu kernels: nemotron_h.cpp calls `rms_norm`, `route`, `add`, `argmax` and `f32_matvec`, with the new
  arguments null.

So Nemotron is the only other model on a touched path. StartLux, Qwen, Ornith and Drama 3 do not reach any changed
function. The lane ran StartLux anyway, and it was identical.

`git diff 3a8f4df 268bcb6` changes comments only (4 lines, all `//`).

## 2. Build and bitwise tests (lane build rebuilt at HEAD)

The build dir held 3a8f4df's objects, so I rebuilt it at 268bcb6:
`TMPDIR=/home/ubuntu/work/nemotron/tmp build-run cmake --build build -j 32`. glm5.cpp and cpu_moe.cpp were
recompiled and the binaries relinked at 13:16 UTC (verify/build.log).

Then, under one gpu-run (verify/gpu.sh):
- `ctest -R 'glm5|pool|cpu_moe|iq4'`: 7 pass, pool_stress skips itself (verify/ctest.log).
- `glm5_kernels_test`: PASS, 219 cases ok, including every kda_fg case (n_in 64/128/256, D 8192/1024, both one-wave
  settings) and every Q8_1-producer case at T 1/2/8 (verify/glm5k.log).
- `iq4xs_rows512_test`: "ok: 30672 results, 0 differ from ggml-cpu's dot" (lean and first, 1-8 tokens).

`iq4xs_rows512_test --dram`, 32 threads, 2,048-value rows:

| tokens | ggml vec_dot | first | lean |
| --- | --- | --- | --- |
| 1 | 158.1 GB/s | 190.9 | 197.8 |
| 2 | 92.1 | 159.5 | 173.2 |

The lane measured 184.5 -> 190.9 and 155.9 -> 165.6.

## 3. My A/B (GLM-5.3-Flash UD-IQ4_XS, GTX 1070 + 2x Xeon Gold 6130)

One process of the lane build at HEAD (its own copy, verify/glm5_run.v, deleted afterwards), GUILD_GLM_SHEXP_STREAM
unset (a default load). Six arms, `--reps 5`, arms alternated by glm5_run:

```
cd /home/ubuntu/Guild && env GUILD_GLM_TIMING=1 GLM5_RUN_STEPS=1 GUILD_GLM_KEEP_CACHE=1 glm5_run.v <GLM> \
  785,6722,315,9621,374 256 --ctx 32768 --spec --ab GUILD_IQ4XS_LEAN=1,0,0,1,1,1 --ab GUILD_IQ4XS_GU=1,0,1,0,1,1 \
  --ab GUILD_GLM_FUSE2=255,0,255,255,0,255 --ab GUILD_CPU_MOE_REUSE=1,0,1,1,1,0 --reps 5
```

The load took 876 s. Before it, node 0 had 69 GB free and node 1 33 GB. After it, numastat -p showed 165 / 140 GB
private on nodes 0 / 1.

The lock was held 14:12-14:38 UTC (26.5 min, including the tests and Nemotron). Output: "A/B ids: identical in every
run" (30 decodes). The id line's md5 is 5714f4a1d575bd906a4f5b5421eac78d, the same as the base binary's
(ab/base.out) and the same as every lane process (s1, ab2-ab5). Drafts: 117 of 138 accepted in every run.

Step p50 in ms, by round. Round 1 ran forward right after the load: the all-on arm came first at 77.7, and the rest of
round 1 rose with position. So the "r2-r5" columns, two reversed and two forward rounds, are the fair ones.

| arm | r1 | r2 | r3 | r4 | r5 | pairs r1-r5 (median) | pairs r2-r5 (median) | fit, all / r2-r5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all on | 77.7 | 79.3 | 79.6 | 78.9 | 79.1 | | | |
| all off | 78.7 | 81.3 | 82.1 | 80.9 | 80.6 | +1.0 +2.0 +2.5 +2.0 +1.5 (+2.00) | +2.00 | +1.80 / +2.00 (+-0.30) |
| LEAN=0 | 78.1 | 79.6 | 80.6 | 79.6 | 79.3 | +0.4 +0.3 +1.0 +0.7 +0.2 (+0.40) | +0.50 | +0.51 / +0.55 |
| GU=0 | 78.6 | 79.2 | 80.4 | 79.8 | 79.3 | +0.9 -0.1 +0.8 +0.9 +0.2 (+0.80) | +0.50 | +0.53 / +0.45 |
| FUSE2=0 | 78.8 | 79.1 | 80.4 | 80.1 | 79.3 | +1.1 -0.2 +0.8 +1.2 +0.2 (+0.80) | +0.50 | +0.60 / +0.50 |
| REUSE=0 | 79.0 | 79.1 | 81.0 | 80.9 | 78.9 | +1.3 -0.2 +1.4 +2.0 -0.2 (+1.30) | +0.60 | +0.84 / +0.75 |

The fit is verify/vfit.py: arm + round + slope x position in the round, by least squares. Each standard error is
0.30-0.31 ms.

CPU experts, fit over all rounds: all off +2.98, LEAN=0 +1.55, GU=0 +0.97, FUSE2=0 +0.55, REUSE=0 +2.31 ms a step
(+-0.7). The lean and REUSE gains show up directly in the CPU time; FUSE2 is GPU-side, as expected.

Wall time a step (decode ms / 139 steps), fit over all rounds / r2-r5:

| arm | all rounds | r2-r5 |
| --- | --- | --- |
| all off | +1.69 | +2.21 |
| LEAN=0 | +0.35 | +0.62 |
| GU=0 | +0.53 | +0.48 |
| FUSE2=0 | +0.21 | +0.33 |
| REUSE=0 | +0.69 | +0.85 |

Each is +-0.3-0.4 ms. Summary tok/s (mean of 5 runs): all on 23.32, all off 22.84, LEAN=0 23.22, GU=0 23.17,
FUSE2=0 23.27, REUSE=0 23.13.

I re-parsed the lane's raw logs with my own parser (verify/vparse.py). Its tables in NOTES.md are what the logs hold:
ab3 +1.1 +2.0 +1.4; ab4 GU +0.60, FUSE2 +0.70; ab5 +0.50 / +0.30 / +0.40 / +0.45 / +1.75. In ab2, "all off" (before
item 5 existed) gave only +0.20 median, below LEAN=0 alone (+0.70). That shows how noisy a 7-arm process is.

## 4. Other models: Nemotron 3.5 Transcribe

`transcribe_run` (the BRIEF's command: --mtp 3 --repeat 6, clips.jsonl), lane build then base build in the same
gpu-run session:
- **Ids.** 12 of 12 records are identical (ids and text). The ids' md5 (json of id + ids) is
  788b10876c7166c37681483d665e2b2f for both. They also equal the lane's earlier base run (ab/nem_base.jsonl).
- **Speed.** Not claimed, but: runs 1-5 of each clip were lane 208-210 against base 213-216 ms (5 s clip), and lane
  1123-1128 against base 1145-1150 ms (30 s clip).
- **First run.** The lane binary was slower only on its first decode after a cold 295 s load.

StartLux does not reach any changed function, so I did not re-run it. The lane's run was identical.

## Issues and notes

- **Borderline items.** Items 3, 4 and 5 each sit near the 0.5 ms rule. Items 3 and 4 also clear it in my process
  (pair medians +0.50 to +0.80, fits +0.45 to +0.60 +- 0.30). Across the lane's processes and mine, every item is
  positive in step p50.
- **Wall time.** Item 4's gain in wall time is smaller: +0.2-0.3 ms a step against +0.5-0.6 in step p50. The same
  holds in the lane's ab4/ab5: +0.29 / +0.28 median. That is about 0.1% tok/s.
- **Item 2.** It is the weakest by pair median: +0.40 over all my rounds, +0.50 over r2-r5, and +0.70 in the lane's
  single ab2. Its CPU-time drop (+1.0 to +1.6 ms with it off) and the kernel rate (+3.6% / +8.6% at 1 / 2 tokens)
  are direct.
- **The two fusion bits.** The lane keeps them as one switch; I measured only both together. By the lane's own data,
  each bit alone is under the rule (ab5 +0.30 / +0.40, ab2 +0.20 / -0.10). That is the lane's call, flagged for the
  user.
- **Item 1's code.** It stays in the branch, opt-in and unused. It is harmless by review, and a default load allocates
  nothing. The one default-path change, st's priority, is unmeasured in isolation but orders nothing with one stream.
  The user may prefer to drop it.
- **Commit message.** 3a8f4df's message quotes ab4's 0.6 / 0.7 ms; 268bcb6 corrects the comments.
- **REPORT.md.** It was not written (the lane says the harness refused it). NOTES.md holds the log.
- **Lane gap.** The lane's planned nsys check (gprof3 / gprof4: route A -> B, layer 11's idle) was not done.
