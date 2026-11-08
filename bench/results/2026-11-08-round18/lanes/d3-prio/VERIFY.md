# d3-prio: independent verification (round 18)

Verifier's run of 2026-11-08: two gpu-run batches, 12:45-13:02 and 13:53-13:56 UTC, ~20 min of lock time in all. Lane HEAD
`0841464` on `lane18/d3-prio` (6 commits over perf/next `a10d867`); the kernel worktree `gkc/` is unchanged from
`de65d60` (empty diff), so no lock bump. Nothing was edited in the lane's worktrees; the scripts and outputs of this
check are in `/home/ubuntu/lanes/r18/d3-prio/verify/` (`batch.sh`, `batch2.sh`, `tests.sh`, `mb3v.sh`, `ab/`,
`tests/`, `mb3_*.sum`).

## Verdict

| Item | Verdict | Why |
| --- | --- | --- |
| 1. step_begin on a top-priority stream | accept | Exact (code read, tower test bits, 36 WAVs, serve audio). Gain measured: +0.87% frames/s vs PRIO=0, 10 pairs, SE 0.14, 9 of 10 positive; +0.82% vs the base build (8 pairs). This is less than the lane's +1.33% headline and below the stop rule's +1%. It agrees with the lane's final-binary +0.82%. |
| 2. Codec pacing (`set_hold`, depth 2) | accept, as part of the PRIO=1 bundle | Exact: only host waits; the same cuBLAS calls. Depth 0/2/4 give the same md5s. Its own share of the gain is not measured apart from the bundle: the lane's depth 0/1/2 arms are within noise. |
| 3. Chunk flushed after tower_step | accept, as part of the bundle | Exact: `pending` is not changed between the old flush point and the new one, and the `last` and EOS paths flush as before. The CLI md5s (chunk 16) and the serve-loop audio hashes (chunk 16 and chunk 4096) match. |
| 4. GUILD_DRAMA3_PRIO=0 restores the old path | accept | Code: one stream, no `set_hold`, flush before tower_step. Same md5s and tower bits, and the same speed as the base build (-0.03%, 8 pairs, SE 0.10). |
| HOLD=between (opt-in, off) | not a gain (lane: not kept) | Exact in my run (seed 12 f84288c9). There is no gain over the default in the lane's data; its code stays in the tree, off by default. |
| FINISH_PRIO (removed) | not a gain (lane: not kept) | Removed in 0841464. The binary has no such string. |
| 5. Verifier commit graph below step_begin | not done | Not implemented: the qwen-prompt lane owns verify.cpp. |

The lane's numbers check out against its own logs: 22 PRIO=1/PRIO=0 pairs in `ab/ab2-6.progress` give +1.33%
(SD 1.73, SE 0.37, 19 of 22 positive). The 81 WAVs are 44 x `6e04c9b1` and 37 x `f84288c9`. phase.py on its
`ab/prof_*.csv` gives the cycle sums it quotes (p0 359.12, p1 354.62, p1b 350.35, hb 350.51).

## 1. Diff review (exactness)

Files: `src/core/drama3_tower.cpp`, `include/guild/core/drama3_tower.hpp`, `src/core/drama3_codec.cu`,
`include/guild/core/drama3_codec.hpp`, `src/runtime/drama3.cpp`, `tests/core/drama3_tower_test.cpp` (a new FNV
line only). No kernel, cpu_moe, verify, prefill or pool code changed, and no ABI header changed.

**Tower streams.**
- Every launch reachable from `step_begin` goes to the same `st`: the hidden copy, the cand matvec, `read_logits`,
  and `step_layers(0, split)` (embed, `layer_forward`: rms_norm, matvec, qk_norm_rope, attention, add, swiglu,
  mmvq) plus the residual layer's head and its D2H copies. `grep stream_` in drama3_tower.cpp shows no stray
  `stream_` use left in that path.
- `step_finish` (stream_) and `prefill_chunk` (stream_) each end with `cudaStreamSynchronize`. `step_begin` ends
  with a synchronize on `st`.
- In `synthesize`, every `tower_step` comes after `tower_join()` (the `beside` job ends with step_finish's
  synchronize). The first step comes after `prefill`, which synchronizes on every chunk.
- So the two tower streams never hold work at once, and the buffers they share (h_, logits_, xq_, host_logits_,
  K/V) are ordered on the host.
- The backbone's `head_input(0)` is complete when `Verifier::run` returns: verify.cpp:2361 synchronizes `cs_`. The
  copy was on a non-blocking stream with no event wait before this change too, so its ordering is unchanged.
- Kernel choice does not depend on the stream. native_mmvq's dispatch state is `thread_local` or keyed by device
  (gkc `src/math/native_mmvq.cu:1264, 1965`), and step_begin and step_finish keep their threads (main and `beside`).
- The same kernels run, in the same order, on the same data.

**Codec pacing.**
- `gemm()` passes the same arguments to `cublasSgemm`, on the same handle and stream. The ring arithmetic is right
  for every clamped depth (0..4): at depth 4 it synchronizes the slot that the next record overwrites, before
  overwriting it.
- `issued`, `done` and `hold` change only while `codec_side` is idle: before the loop, and in `clear()` after
  `side->join()`. Beside's mutex orders those writes before the codec thread reads them.
- `hold_wait()` and the event records change only host timing.

**No deadlock.** The main thread waits for the codec in three places: `flush` → `codec_join`, the guard's
`clear()`, and the final `codec_join`. In all three the hold is false:
- `tower_step` clears it right after `step_begin`;
- HOLD=between sets it only when `!last`, so a `tower_step` always follows before the next flush;
- the store after the loop covers every `break`;
- the guard stores false before it joins, on every return.

**Chunk boundaries.**
- With PRIO=1 the full-chunk flush moves from before `tower_step` to after it, in the same iteration. Nothing
  touches `pending` in between (tower_step writes `frame`), so every chunk holds the same frames.
- The `last` frame still flushes before the `break`. EOS still breaks at the top of the loop, and `flush(true)`
  takes the rest.
- The only behaviour change is on cancel (`on_audio` returning false): one more tower step runs before the loop
  stops. No audio changes.

**Switch-off path.** With `GUILD_DRAMA3_PRIO=0`:
- `top_` is false, so step_begin uses `stream_`.
- The codec's `set_hold` is never called, so `hold` is nullptr: no events and no waits.
- The flush stays before `tower_step`, and HOLD=between and CODEC_DEPTH are ignored.
- The only leftovers are an idle second stream and 4 events created at load.

**Callers.**
- `Drama3Tower::step_begin`, `step_finish` and `prefill` are called only from `src/runtime/drama3.cpp` and
  `tests/core/drama3_tower_test.cpp` (`Drama3Tower::step` wraps both).
- `Drama3Codec::decode` is called only from `synthesize`. `encode_voice` → `Drama3Codec::encode` runs unpaced,
  because the guard clears the hold on every exit from `synthesize`.
- No other model reaches these files, so step 4 (one run per other model) does not apply.

**Issues (none of them affects output bits):**
- **HIP.** `Drama3Tower::load` now calls `cudaDeviceGetStreamPriorityRange` and `cudaStreamCreateWithPriority`.
  Neither HIP shim maps them: `include/guild/hip_compat/cuda_runtime.h` and
  `include/guild/platform/hip_compat/cuda_runtime.h`, both listed in CMakeLists for HIP builds.
  - verify.cpp:705-708 already calls both without a guard at perf/next, so HIP builds were already affected. The
    lane adds a second call site.
  - Fix before an AMD release: either two `#define`s in each shim (hipDeviceGetStreamPriorityRange,
    hipStreamCreateWithPriority), or the `#if defined(GUILD_USE_HIP) || defined(GUILD_HIP_GFX906)` fallback that
    `create_gdn_stream` uses (verify.cpp:174-184). Not tested here (no AMD card, no hipcc).
- **Robustness.** If the priority stream cannot be created, `Drama3Tower::load` fails outright, even though
  PRIO=0 does not need that stream. A fallback to `prio_stream_ = nullptr` would be enough:
  `set_step_priority` already handles nullptr.
- **Tests.** The drama3 tests are executables only (CMakeLists:487-490), not ctest entries, so `ctest -R drama3`
  runs nothing. They were run by hand.

Formatting: `tools/format_code.py --check` in the lane worktree reports all 546 files formatted (clang-format
18.1.8). The commit trailers are as the brief asks.

## 2. Build and bitwise tests

The lane build is up to date with HEAD:
- No file under guild/src, include, tests or gkc is newer than `build/guild-generate` (11:34).
- The worktree is clean.
- 362150b and 0841464 have the same tree.
- The binary has no CODEC_SYNC or FINISH_PRIO strings.

I did not rebuild. Tests were run under gpu-run (`verify/tests.sh`, outputs in `verify/tests/`), with the same
reference data as the lane:

| Test | Lane PRIO=1 | Lane PRIO=0 | Base |
| --- | --- | --- | --- |
| drama3_tower_test bits line | 8d3fea1524964286 | 8d3fea1524964286 | (no such line) |
| drama3_tower_test, every other line | = base (diff empty) | = base (diff empty) | KL 0.00147 / 0.00655, argmax 0.940, K/V 0 of 507, PASS |
| drama3_codec_test chunks 7 / 16 | = base (diff empty) | | 110.4 / 109.9 dB, worst error 0, PASS |

## 3. Verifier's A/B (one gpu-run, `verify/batch.sh`)

Setup:
- Arms: L1 = lane binary with the default (PRIO=1), L0 = the same binary with `GUILD_DRAMA3_PRIO=0`, B =
  `/home/ubuntu/lanes/r18/base/build/guild-generate`.
- 4 rounds x 2 seeds. The order turns between blocks (L1 L0 B / L0 B L1 / B L1 L0), and the seed order is reversed
  every other round.
- Runner: the lane's `d3run.sh`, which is drama3-run-r18's command inside one lock.

frames/s and WAV md5:

| Run | L1 (PRIO=1) | L0 (PRIO=0) | B (base) |
| --- | ---: | ---: | ---: |
| s11 v1 | 46.88 | 46.33 | 46.35 |
| s12 v1 | 46.81 | 46.18 | 46.26 |
| s12 v2 | 46.54 | 46.21 | 46.43 |
| s11 v2 | 46.60 | 46.34 | 46.21 |
| s11 v3 | 46.73 | 46.27 | 46.19 |
| s12 v3 | 47.28 | 47.38 | 47.48 |
| s12 v4 | 47.85 | 47.42 | 47.23 |
| s11 v4 | 47.85 | 47.25 | 47.35 |
| mean | 47.07 | 46.67 | 46.69 |

- md5: seed 11 `6e04c9b1` and seed 12 `f84288c9` in every run of every arm (24 of 24).
- Main-thread tower ms: L1 ~1898-1911 (s11) / ~1706-1715 (s12); L0 and B ~2161-2169 / ~1940-1946.

Paired:

| Comparison | Mean | SD | SE | Positive pairs |
| --- | ---: | ---: | ---: | --- |
| L1 vs L0 | +0.85% | 0.51 | 0.18 | 7 of 8 |
| L1 vs B | +0.82% | 0.60 | 0.21 | 7 of 8 |
| L0 vs B | -0.03% | 0.29 | 0.10 | 3 of 8 |

So the switch-off path is as fast as the base.

The other switches, one run each on the lane binary (bits only):
- `GUILD_DRAMA3_OVERLAP=0` (PRIO=1, codec and step_finish inline on the main thread): s11 `6e04c9b1`, 40.33 frames/s.
- `GUILD_DRAMA3_HOLD=between`: s12 `f84288c9`.
- `GUILD_DRAMA3_CODEC_DEPTH=0`: s12 `f84288c9`.
- `GUILD_DRAMA3_CODEC_DEPTH=4`: s11 `6e04c9b1`.

Serve loop (`verify/mb3v.sh`: the lane's mb3 serve loop, SAY chunk=16 vs chunk=4096 alternated x3). This copy keeps
the audio lines and hashes each request's audio:
- The audio is the same bits in both arms: chunk 16 `79c7bf4e` and chunk 4096 `5e7b9e7f` in all 6 requests of each
  arm.
- Codec cost (chunk-16 time minus chunk-4096 time, in one session): L1 10044.9 - 9195.5 ms = **1.77 ms a frame**;
  L0 10411.7 - 9485.3 = **1.93**. The lane measured 1.82 vs 1.90. The target of <= 0.9 is not met.
- The two sessions ran one after the other, not alternated. The ~3% gap between them, at chunk 4096 as well, is
  drift between the sessions, so only the in-session difference counts.

CPU-profile pair (seed 11, PRIO=1 run first, `tools/phase.py`):

| Position | PRIO=0 frame / step / between (ms) | PRIO=1 frame / step / between (ms) |
| --- | --- | --- |
| 15 (flush step) | 22.90 / 16.03 / 6.84 | 20.69 / 16.59 / 4.06 |
| 0 | 44.77 / 27.44 / 17.42 | 32.06 / 27.14 / 4.86 |
| 1 | 22.24 / 18.31 / 3.90 | 34.54 / 24.76 / 9.72 |
| regular (2..14) step | 16.0-16.3 | 16.6-16.8 |
| cycle sum | 350.55 | 355.78 |
| frames/s (with the profiler) | 45.65 | 44.97 |

- The per-position moves the lane quotes come out the same: flush-step between 7.06 → 4.05, pos 0 between
  17.51 → 4.84, pos 1 step 18.58 → 24.89 and between 3.95 → 9.61.
- The cycle sum does not show the lane's 359.1 → 354.6 / 350.4 in this one pair. PRIO=1's regular positions (2..14),
  where no codec runs in either arm, are ~0.5 ms a step slower in this profiled run. That points to drift between
  the two runs, not the change: the lane's own profiles show no such gap (p0 16.6, p1 16.5-16.6, p1b 16.3), and the
  8 unprofiled pairs favour PRIO=1.
- A second profiled pair in the reverse order is in batch 2 (below).

## 3b. Batch 2 (`verify/batch2.sh`, one gpu-run, 13:53-13:56 UTC)

The profiled pair in the reverse order (PRIO=0 first), then 2 more alternated pairs:

| Run | PRIO=1 | PRIO=0 |
| --- | ---: | ---: |
| s11, CPU profile | 46.13 | 44.80 |
| s12 v5 | 47.70 | 47.22 |
| s11 v5 | 47.60 | 47.20 |

md5s `6e04c9b1` / `f84288c9` in all 6 runs. In all 36 of my Drama 3 runs there are 20 x `6e04c9b1` (seed 11) and 16 x
`f84288c9` (seed 12), with no other value.

**Profiled pair, reverse order** (phase.py, PRIO=0 → PRIO=1):
- flush step between 7.05 → 4.04; pos 0 between 17.54 → 4.74; pos 1 step 18.43 → 24.48, between 3.97 → 9.95.
- Regular frames 20.48 → 20.04 ms. This time the first run of the pair is the slower one again, which confirms
  that batch 1's regular-frame gap was run-order drift.
- Cycle sum 357.39 → 346.96.

**Drift-free measure.** The excess of positions 15, 0, 1 and 2 over 4 regular frames (median of positions 3..13):

| Pair | PRIO=0 | PRIO=1 |
| --- | ---: | ---: |
| batch 1 | 29.94 ms a cycle | 25.51 |
| batch 2 | 29.71 | 26.43 |

So the change saves ~3.9 ms a 16-frame cycle, ~0.24 ms a frame (~1.1%). This matches the A/B, and the lane's
reading that most of the codec's ~30 ms comes back inside the next two backbone steps.

**All 10 unprofiled pairs, PRIO=1 vs PRIO=0:** +0.87% (SD 0.45, SE 0.14, 9 of 10 positive; 47.18 vs 46.78
frames/s). The two profiled pairs: -1.5% and +3.0%, which is order drift; their mean is +0.7%.

## Assessment

- The change is exact. I found no race, no stale read, and no path where the switch-off differs from perf/next.
- The gain is real but small: +0.87% in my 10 unprofiled pairs (SE 0.14), against the lane's +1.33% over 22.
  The drift-free per-cycle excess falls by ~3.9 ms (~1.1%).
- All 32 pairs together: about +1.2%.
- The judge's +2-5% did not happen: the codec's cost moves into the next two backbone steps, as the lane says.
- It is on by default with one env switch off. It costs nothing measurable when off, and it is safe to ship for
  CUDA. The HIP mapping (or guard) above should land before an AMD build is released.
