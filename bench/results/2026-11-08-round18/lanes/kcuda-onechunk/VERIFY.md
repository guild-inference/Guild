# kcuda-onechunk: independent verification (round 18)

Verifier's record, 2026-11-08, 13:00-14:45 UTC. I did not edit the lane's sources or commit to its branches. My
outputs are in `/home/ubuntu/lanes/r18/kcuda-onechunk/verify/`.

## Verdict

| Item | Verdict | Why |
| --- | --- | --- |
| 1. Q4_K one-chunk staged rows (8-16 blocks at 1-2 columns, 8-10 at 3) | accept | Same arithmetic in the same order (read and checked), 0 bitwise differences in 2,119,284 parity outputs, full token lists identical to the base lane. Qwen +4.0% prose / +3.8% code in my pairs. |
| 2. Q6_K one-chunk rows of exactly 8 blocks | accept | Exact: a row of exactly C blocks is one chunk in both kernels, and the 16-bit scale product gives the same int. Its engine gain is only measured together with item 1 (one switch turns off both): Ornith +3.4% / +4.3%. The harness numbers below are its only separate measurement. |
| 3. Q4_K at 4 columns as 2 + 2 staged calls, rows of 9-12 blocks | accept | Exact by the per-column contract, which the parity test checks. The lane had no engine A/B of it. My 2 Qwen pairs (GUILD_MMVQ_SPLIT4=0): code +0.5% (-0.34 ms a window), prose unchanged. Small, but the sign is the same in both pairs. |
| 4. Q5_K one-chunk (optional) | not kept, nothing to verify | Not attempted. |

Shipping needs the user to bump the lock of guild-kernels-cuda from de65d60 to 11936b5 (branch lane18/kcuda-onechunk,
not pushed). backends.lock.json is untouched in the main worktree and in the lane.

## 1. Diff review

The kernel diff is `git -C gkc diff de65d60...HEAD`: `src/math/native_mmvq.cu`, +56 / -4. The Guild diff
(`git -C guild diff perf/next...HEAD`) only touches tests: `mmvq_staged_parity.cpp` (7 new cases and a per-column
check), a CMake test registered with `GUILD_MMVQ_ONE_WIDE=0`, and 7 `q4w` shapes in `glm5_kernels_test`.

How I checked that the arithmetic is the same:
- **Q4_K product.** `Staged<Q4KTraits>::dot` is `Q4KTraits::dot`. The chunked kernel's non-XR term is
  `F::apply(wv, x, kqs)`, which is `dot(r, load_x(x, kqs))`, and its XR term is `F::dot(wv, Staged::get(...))`. The
  one-chunk kernel's terms are `Staged::dot(wv, F::load_x(...))` and `Staged::dot(wv, Staged::get(...))`. So both
  kernels use the same expression on the same operands.
- **Block order.** Q4_K has VDR 2 and QI 32, so T = 16 and BPI = 8. In the chunked kernel (C = 8) each chunk runs
  the i loop once, and acc[j][t] gets block tid/16 from chunk 0, then 8 + tid/16 from chunk 1; acc is reset only
  after the last chunk. In the one-chunk kernel (C = 16), i = 0 and i = 1 visit the same two blocks in that order,
  into acc[j][0], which starts at 0. The `kbx < nb` guard is the same, so the blocks past a 9-15-block row are
  skipped in both. Q6_K has T = 32 and BPI = 4. For nb = 8, C = 8 takes blocks b and 4 + b; C = 12 takes the same two,
  and its third step (8 + b) fails `kbx < nb`.
- **Add order.** The chunked kernel (unrolled, since NCOLS < RollChunked) computes
  `v = acc[0]; v += acc[1]; v += acc[2]; v += acc[3]`. The one-chunk kernel's running total computes
  ((s0 + s1) + s2) + s3. Both then call the same `warp_sum`.
- **Q6_K's 16-bit scale product on 8-block rows.** The dp4a of four 6-bit values (-32..31) and four int8 values lies
  in -16256..16384, so `short()` keeps it exactly, and the float expression is unchanged
  (`sumf += d8 * (int * sc)`, then `d * sumf`). Qwen's 2560-wide Q6_K head already ran these kernels.
- **Split.** The activation stride `(n_in / 32) * sizeof(Q81Block)` matches the kernels (`x + j * xb`,
  `xb = n_in / Q8K`) and `native_q8_1_bytes`, which has no per-column padding. The outputs go to `y + 2 * n_out`
  (layout `y[j * n_out + r]`). Both calls are on the same stream, and the inner calls run every refusal check, so a
  refused first call launches nothing and the caller's usual path writes all 4 columns.
- **Reads and races.** The one-chunk kernel reads `staged_wbytes<F, ROWC>` per warp. For a row offset o ≤ 15,
  ceil((o + rb) / 16) ≤ floor((rb + 30) / 16) pieces fit, the same bound the Q6_K one-chunk rows already relied on.
  Q4_K rows are a multiple of 16 bytes (144 x nb), so no row starts inside a piece. The new globals
  (`g_one_wide_off`, `g_split4_off`) are read once at load and never written, and `g_multi_exact` is thread_local. No
  shared-memory synchronization was changed. The HIP build does not compile any of this (it is all inside
  `#if !defined(GUILD_USE_HIP)`).
- **Switches.** `GUILD_MMVQ_ONE_WIDE=0` gives `one = false` for Q4_K (the old ROWC = 0 behaviour) and `nb > C` for
  Q6_K, and it turns the split off. The XR choice depends on `one`, so the old arm launches exactly the old
  instantiations. nsys of the parity binary confirms it (`verify/ns_par_{new,old}.sqlite`): the old arm launches no
  Q4_K C = 16 kernel and sends every Q4_K row through C = 8. The new arm launches C = 16 at (1, non-XR), (2, XR)
  and (3, XR); its 2-column launches are 6 shapes plus 3 split shapes x 2 = 12. `GUILD_MMVQ_SPLIT4=0` sends 4-column
  Q4_K back to `launch_multi`, as before.
- **Callers.** `staged_mmvq<Q4KTraits/Q6KTraits>` are reached only from `native_q4_k_mmvq` / `native_q6_k_mmvq`, which
  get their calls through the type dispatcher. In Guild those calls come from verify.cpp, layer.cpp, dense_mm.hpp,
  native_head.cpp, dflash.cpp, mtp.cpp, drama3_tower.cpp and cuda_dense_bridge.cpp. Which models have rows that
  change (GGUF tensor tables read with a small header parser):
  - Ornith and Drama 3 (Q4_K_M): Q4_K rows of 8 and 16 blocks and Q6_K rows of 8 blocks (head, qkv, attn_v).
    Changed.
  - Qwen: setup's map gives 10-block Q4_K rows. Changed (1-3 columns, and the split at 4).
  - GLM-5.3-Flash: dense Q8_0, converted to Q6_K at --ctx 32768 (wq/wk/wv 16 blocks, q_b 6, wo 32/64). Its head is
    Q6_K with 16 blocks. Every one of these rows takes the same kernel as before, so GLM's path does not change in
    its default configuration.
  - Nemotron (LM all Q8_0, audio encoder bf16/f32) and StartLux (Q8_0): not touched.

## 2. Build state and tests

- `build-fin/` is the lane's HEAD: gkc 11936b5, guild 94a91d4, both worktrees clean. Its `native_mmvq.cu.o`
  (11:35:22) is newer than the source (10:53:17), so I did not rebuild it. `build/` is still binary 1 (b7f7ab9,
  .o 06:52), and the lane's queued chain3 runs on it.
- `ctest -R 'mmvq|gr_parity|qwen|ornith|drama3|route_window'` on build-fin under the lock: 11/11 pass
  (`verify/ctest.log`).
- My own reference, `GUILD_MMVQ_STAGED=0 mmvq_staged_parity --dump verify/ref.bin`. Against it, with each column also
  checked against its 1-column call, these arms all give "2,119,284 outputs compared, 0 differ bitwise": default,
  `GUILD_MMVQ_ONE_WIDE=0`, `GUILD_MMVQ_SPLIT4=0` and `GUILD_MMVQ_ROW_CHUNK=0` (`verify/par_*.log`). The dumps from
  the nsys runs (new and old arm) are byte-identical to ref.bin.
- `tools/format_code.py --check` in the guild worktree: all 546 files formatted.
- Sanitizers: see section 5.

## 3. Engine A/Bs (mine)

Setup: build-fin/guild-generate, one lock hold 13:55:49-14:12:01 UTC (16 min). The helpers are the BRIEF's with only
`gpu-run` removed (`verify/*-nolock`, diffed). Node 0 had 86 GB free (numastat). I compared the full `output_ids`
lists of every run with the base lane's JSONs (`/home/ubuntu/Guild/b-q_base.json` and `b-o_base.json`, made by
`base/build/guild-generate`, sha db36d21d...). All of them are identical.

**Qwen3.8-Flash-Next** (qwen-bench-r18), run order N1 O1 S1 S2 O2 N2. N = default, O = `GUILD_MMVQ_ONE_WIDE=0`,
S = `GUILD_MMVQ_SPLIT4=0`. Each cell is the mean ms a window of 2 runs (the warm-up line left out), then tok/s.

| Arm | prose ms/win | prose tok/s | code ms/win | code tok/s | ids |
| --- | --- | --- | --- | --- | --- |
| N1 | 55.12 / 55.01 | 51.09 | 61.59 / 61.30 | 64.10 | [38747, 38747, 41916, 41916] |
| O1 | 57.47 / 57.18 | 49.07 | 63.85 / 63.67 | 61.77 | same |
| S1 | 55.09 / 54.90 | 51.15 | 61.83 / 61.80 | 63.71 | same |
| S2 | 54.92 / 55.07 | 51.15 | 61.74 / 61.45 | 63.94 | same |
| O2 | 57.25 / 57.20 | 49.16 | 63.84 / 63.55 | 61.83 | same |
| N2 | 55.29 / 54.90 | 51.06 | 61.37 / 61.22 | 64.25 | same |

- Items 1-3 together (N vs O): prose 57.28 -> 55.08 ms a window, 49.12 -> 51.08 tok/s (+4.0%); code 63.73 -> 61.37,
  61.80 -> 64.18 tok/s (+3.8%). This matches the lane's +4.0% / +3.9%.
- Split alone (N vs S): code 61.71 -> 61.37 ms a window, 63.83 -> 64.18 tok/s (+0.5%; pair 1 +0.6%, pair 2 +0.5%);
  prose 55.00 -> 55.08 (no change). So item 1 alone is about +4.0% prose and +3.3% code on Qwen.

**Ornith-1.5-35B** (ornith-bench-r18), order N1 O1 O2 N2:

| Arm | prose ms/win | prose tok/s | code ms/win | code tok/s | ids |
| --- | --- | --- | --- | --- | --- |
| N1 | 24.48 / 24.41 | 83.12 | 25.10 / 25.12 | 97.09 | [30936, 30936, 65886, 65886] |
| O1 | 25.35 / 25.33 | 80.17 | 26.23 / 26.22 | 92.98 | same |
| O2 | 25.36 / 25.26 | 80.29 | 26.19 / 26.17 | 93.13 | same |
| N2 | 24.52 / 24.57 | 82.78 | 25.17 / 25.10 | 97.00 | same |

- Items 1+2 (Ornith runs no 4-column Q4_K call): prose 25.33 -> 24.50 ms a window, 80.23 -> 82.95 tok/s (+3.4%);
  code 26.20 -> 25.12, 93.05 -> 97.05 tok/s (+4.3%).
- **This is smaller than the lane's claim** (+4.5% / +5.6%, 25.46 -> 24.36 and 26.40 -> 25.02 ms). The lane's runs
  loaded in 132 s (expert cache copied while other jobs ran); mine loaded in 8 s. The gain is real, and it is within
  the spec's expected +2.5-4%. Report +3.4% / +4.3% unless the lane's numbers are reproduced.

**Drama 3** (drama3-run-r18), seeds 11 and 12, both arms:

| Run | frames/s | md5 |
| --- | --- | --- |
| dN11 | 47.67 | 6e04c9b1 |
| dO11 | 47.36 | 6e04c9b1 |
| dO12 | 47.59 | f84288c9 |
| dN12 | 47.43 | f84288c9 |

The md5s equal the base values in both arms. Speed: 47.48 -> 47.55 frames/s, which is noise; it does not drop. The
lane's final-binary runs gave the same md5s at 47.70 / 47.05 (`ab/fin_batch.log`).

**GLM-5.3-Flash.** I did not run GLM myself: its default path runs no changed kernel (section 1), and a load costs up
to 10 min of lock time. The lane's final-binary run (`ab/glm_fin.out`, build-fin, 13:15 UTC) has an ids line that is
byte-identical to the base runs': md5 5714f4a1, the same as `ab/glm_base_ref.out` and the glm-pipe lane's
`ab/base.out`. 23.22 tok/s, 78.9 ms a step (median).

## 4. Points against the lane's summary

1. **Ornith gain overstated** by about 1 point (above).
2. **The split had no engine measurement** in the lane. Mine gives +0.5% on Qwen code only. The lane's
   "Qwen shapes at T4" harness numbers (qkv 177.3 -> 156.1 us) promise more than the engine shows, because the
   4-column windows are a small share.
3. **Item 2 is not separable** in the engine: one switch turns off items 1-3, and Ornith runs items 1+2. Its only
   separate evidence is harness numbers (the lane's, and mine in section 5).
4. **The parity test has no Q4_K rows of 11, 13, 14 or 15 blocks.** The kernel code is the same (nb is a runtime
   value), and every instantiation those rows would reach (C = 16 at (1, non-XR) and (2, XR)) is covered by the 9-,
   10-, 12- and 16-block cases, so I count this as low risk. No current model has such rows.
5. **Housekeeping:**
   - REPORT.md is missing (the lane says why).
   - chain3 is still queued on `build/` (binary 1, the split on every 8-16-block row). Its results are stale for the
     final rule, except where the lane's reasoning holds (no model in this round reaches a kernel that differs).
   - `build-fin/` (2.5 GB) should be deleted once it is no longer needed.
   - The 3-column one-chunk at 11-15 blocks and the split at 13 blocks are unmeasured (open points, as the lane
     says).

## 5. Sanitizers and harness (second lock hold, 14:39:32-14:40:02 UTC)

**Harness, item 2 alone.** `GLM5_SHAPES_ONLY=ornith build-fin/glm5_kernels_test shapes`, default vs
`GUILD_MMVQ_ONE_WIDE=0`, 3 alternated rounds, min of 3, us, old -> new (`verify/hz_{new,old}_{1,2,3}.txt`):

| Shape | T1 | T2 | T3 | T4 | T5 | T8 |
| --- | --- | --- | --- | --- | --- | --- |
| ornith head Q6_K 2048 x 248320 | 1946 -> 1939 | 2230 -> 2074 (-7%) | 2920 -> 2471 (-15%) | 3621 -> 2988 (-17%) | 4129 -> 3412 (-17%) | 6364 -> 4977 (-22%) |
| ornith qkv q6 Q6_K 2048 x 8192 | 69.8 -> 68.2 | 76.6 -> 73.4 | 92.9 -> 80.0 (-14%) | 123.7 -> 95.5 (-23%) | 141.1 -> 114.5 | 224.4 -> 167.9 |
| ornith qkv Q4_K 2048 x 8192 | 49.5 -> 48.2 | 62.9 -> 51.0 (-19%) | 80.9 -> 64.0 (-21%) | 102.3 -> 98.8 | 138.4 -> 138.6 | 249.9 -> 250.5 |
| ornith out Q4_K 4096 x 2048 | 27.5 -> 26.5 | 33.9 -> 27.7 (-18%) | 39.7 -> 40.2 (chunked in both) | 44.9 -> 45.1 | 57.3 -> 57.5 | 98.7 -> 98.1 |
| ornith v q6 Q6_K 2048 x 512 (below the staged threshold) | 8.0 -> 8.0 | 8.9 -> 8.9 | 10.4 -> 10.4 | 11.0 -> 11.0 | 12.1 -> 12.1 | 17.3 -> 17.3 |

These agree with the lane's harness figures. Rows that run the same kernel in both arms move 0-5% (for example ornith
gate at T4/T5, +4/+5%), so only the large moves count. Together with the Ornith engine pairs, they are enough for
me to accept item 2.

**Sanitizers: not run.** `/usr/bin/compute-sanitizer` is version 2022.4.1 (CUDA 12.0); the driver is 580.178
(CUDA 13.0). Without `--injection-path` it cannot find `libsanitizer-collection.so`. With `--injection-path
/usr/lib/nvidia-cuda-toolkit/compute-sanitizer`, memcheck, racecheck and initcheck all stop with "Target
application terminated before first instrumented API call" (`verify/san_*.log`). The checks for stale or
out-of-range reads and for races rest on the code review in section 1. The parity test helps only a little here:
its weights sit in exactly-sized allocations, but a read past an allocation inside the same page does not fault.

My lock time: 16 min 12 s + 30 s.
