# nem-decode (round 18): independent verification

Verifier's files: `/home/ubuntu/lanes/r18/nem-decode/verify/` (vrun.sh, vsum.py, cmpids.py, gate/, ab/, progress).
Binaries: lane `build/transcribe_run` md5 4b28e9be (= bin/tr-final3, branch lane18/nem-decode @ 1e59836, built
09:54, no source newer than it, worktree clean, so no rebuild); base `/home/ubuntu/lanes/r18/base/build/transcribe_run`
md5 111cc2a7. One lock hold, 11:43:23-12:00:59 (17.6 min); at the start GPU 3%, 3 MiB used, no other compute process;
node 0 MemFree 32.2 GB.

## Verdict

| Item | Verdict | Why |
| --- | --- | --- |
| 1. Scan state stored by thread | accept | Same registers and instructions, a bijection of a private buffer's offsets; 60/60 gate clips in 4 modes; -32 ms (-1.4%) mtp3 30 s, greedy decode -5.0% |
| 2. Argmax on the GPU | accept | Same tie rule as std::max_element (checked on 1,792 verify rows); switch restores the old path; -38 ms (-1.7%) mtp3 30 s |
| 3. Draft head over 44,602 tokens | accept | Only drafts change (subset logits equal the head's in 93 of 93 passes); output 60/60; -75 ms (-3.4%) total, draft -92 ms on 30 s |
| 4. Prefix snapshot | reject (lane did not keep it) | Not exact by the lane's own check; fully reverted: `git diff perf/next...HEAD` has no prefix code |

## 1. Diff review (perf/next...HEAD; gkc unchanged at de65d60)

Files: CMakeLists.txt (a compile definition on transcribe_run only), data/nemotron_draft_vocab.bin,
include/guild/core/nemotron_h.hpp, nemotron_h_kernels.hpp, src/core/nemotron_h.cpp, nemotron_h.cu,
src/program/transcribe_run.cpp, tools/nemotron_draft_vocab.py. Every user of `NemotronHEngine` and of `nhk::` is in
nemotron_h.cpp/.cu and transcribe_run.cpp (grep over src, include, serve, tools, tests, bench); serve/ does not run the
Nemotron engine. No shared code (kernels repo, cpu_moe, verify.cpp, prefill.cpp, pool) is touched, so no other model's
path changes and no other-model runs are needed.

- **Item 1.** Thread tid = p*4+q of head h now keeps s[i] at h*8192 + i*256 + tid instead of (h*64+p)*128 + q*32 + i.
  For i in 0..31 and tid in 0..255 that is a bijection onto the same 8192 floats of the head. The arithmetic loop, the
  shuffles and y are unchanged. The snapshots use the same offset within each [H*P*N] slice. Other readers: reset
  memsets the whole buffer, accept copies whole slices (`ssm_snap + (n_keep-1)*ssm_floats`), and dump_row dumps mixer
  outputs, not states. The switch was removed as the spec asked; the old order is still in bin/tr-item1.
- **Item 2.** `argmax_rows_k`: in each thread the indices rise (float4 chunks then the tail), with strict >. Warp and
  block reductions take the larger value, or the lower index when values are equal. So the result is the first
  maximum, as std::max_element gives, for any values but NaN (+-0 count as equal in both; an all -inf row gives 0 in
  both). The vector path needs 16-byte alignment, and ld % 4 when rows > 1: logits are cudaMalloc'd with ld 131072,
  and the subset has rows = 1. d_arg grew to kMaxVerify (4) ints. h_arg is 64 B pinned (16 ids). `top[8]` in
  transcribe_run holds at most k+1 <= 4. All of it is on the engine's one stream and read after
  cudaStreamSynchronize: no race. With `GUILD_NEM_DEVICE_ARGMAX=0`, verify gets host logits and top=nullptr, and step
  gets host logits: exactly the old calls. The TRANSCRIBE_CHECK_SPEC greedy pass still uses host logits.
- **Item 3.** The draft head copies the subset's Q8_0 rows from `head.host` with the same padding as `mat()`. It is
  not MMQ-marked, and neither is `head`; drafts are always one column. It loads only when the MTP block loads (no VRAM
  change for --no-mtp). verify/head_rows keep the whole head. `GUILD_NEM_DRAFT_VOCAB=0` leaves dhead.dev null, so the
  old `mm(head)` + `glm5k::argmax` runs. data file md5 3685b6e5 = the scout's nemotron_draft_vocab_en.bin: 44,602
  ascending ids, 0-1255 and eos 11 in.
- **Item 4.** Reverted by 1e59836, together with 44d5187's sync. Nothing of it is left in the diff.
- `format_code.py --check`: all 546 files formatted (clang-format 18.1.8).

## 2. Bitwise tests (final binary, verifier's runs)

| Test | Result |
| --- | --- |
| Gate, 60 clips, greedy / mtp1 / mtp2 / mtp3 vs guild5-greedy.jsonl (transcribe_gate.py --same-as) | 60/60 each, WER 10.34%, PASS |
| The same four outputs vs the **base binary's** greedy on the 60 clips (my run) | ids equal on 60/60 in every mode |
| Final with items 2+3 off, mtp3, 60 clips vs base mtp3 | ids 60/60; drafted/accepted/steps equal on 60/60 clips (1338/1180/446 both): the switches restore the old drafts |
| Final (subset) mtp3 vs base mtp3 drafts | 1344/1178/448 vs 1338/1180/446; only 2094-142345-0054 (18/15 -> 21/14) and 2300-131720-0031 (33/33 -> 36/32) differ, as the lane reported |
| TRANSCRIBE_CHECK_SPEC=1 --mtp 3, 60 clips (448 windows) | 0 "window row" lines (GPU argmax == std::max_element on every row), 0 "check: step" lines, prompt logits again max \|d\| 0 on 60/60, ids 60/60 |
| GUILD_NEM_DRAFT_CHECK=1 --mtp 3, 2 perf clips | 93 passes, 0 of 44,602 subset logits differ in every pass, head's argmax in the subset 93/93 |
| ctest transcribe_gate_test | Fails on base as well: base greedy against reference_fp32 with gate.json gives 50 identical (bound 57), WER 10.34 vs 10.27, rc 1; identical to the lane's greedy, so it does not come from this lane |

## 3. A/B (alternated in one hold, 2 pairs, --repeat 6, perf clips, medians of runs 1-5 of both files)

Arms, mtp3: A = final; B = final + GUILD_NEM_DRAFT_VOCAB=0; C = final + GUILD_NEM_DRAFT_VOCAB=0 +
GUILD_NEM_DEVICE_ARGMAX=0 (item 1 alone); D = base. Greedy (--no-mtp): Ag = final; Cg = final +
GUILD_NEM_DEVICE_ARGMAX=0; Dg = base. Order A B C D A B C D, then Ag Cg Dg Ag Cg Dg. Id hashes: 35646 (30 s) and
25430 (5 s) in every run of every arm; 26 steps and 78/69 drafted/accepted (30 s), 5 and 15/15 (5 s) in every mtp3
arm.

| Arm | Clip | total | prefill | decode | draft | verify | per-pair total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 30 s | 2103.7 | 923.3 | 1003.2 | 137.1 | 863.2 | 2103.6 / 2105.2 |
| B | 30 s | 2178.8 | 915.6 | 1090.8 | 229.2 | 858.3 | 2179.9 / 2177.6 |
| C | 30 s | 2216.6 | 916.8 | 1128.0 | 229.3 | 895.5 | 2220.0 / 2216.4 |
| D | 30 s | 2248.7 | 918.0 | 1161.0 | 229.1 | 928.5 | 2252.8 / 2247.9 |
| A | 5 s | 577.9 | 349.2 | 186.2 | 24.7 | 161.1 | 582.6 / 575.9 |
| B | 5 s | 600.1 | 352.3 | 204.6 | 42.4 | 161.8 | 599.5 / 600.5 |
| C | 5 s | 608.1 | 352.9 | 212.8 | 42.5 | 169.9 | 607.2 / 610.4 |
| D | 5 s | 611.5 | 352.4 | 217.6 | 42.5 | 174.4 | 611.0 / 611.6 |
| Ag | 30 s | 2847.2 | 864.8 | 1807.9 | | | 2847.3 / 2847.0 |
| Cg | 30 s | 2875.2 | 867.2 | 1837.7 | | | 2875.0 / 2875.6 |
| Dg | 30 s | 2974.2 | 867.2 | 1935.4 | | | 2976.4 / 2967.8 |
| Ag | 5 s | 751.5 | 335.8 | 372.4 | | | 754.7 / 748.8 |
| Cg | 5 s | 761.0 | 338.4 | 379.0 | | | 762.1 / 758.9 |
| Dg | 5 s | 780.7 | 336.9 | 400.7 | | | 779.8 / 780.9 |

| Item | Verifier, 30 s | Lane's claim, 30 s | Verifier, 5 s |
| --- | --- | --- | --- |
| 3 (A vs B) | total -75.1 ms (-3.4%), decode -87.6 (-8.0%), draft 137.1 vs 229.2 | -89.7 (-4.1%), draft 136.7 vs 228.9 | -22.2 (-3.7%) |
| 2 mtp3 (B vs C) | -37.9 (-1.7%), verify -37.2 (-1.4 ms a step) | -39.5 (-1.8%), verify -36.4 | -8.0 (-1.3%) |
| 2 greedy (Ag vs Cg) | total -27.9 (-1.0%), decode -29.8 (-1.6%) | decode -29.4 (-1.6%) | -9.5 (-1.2%) |
| 1 mtp3 (C vs D) | -32.0 (-1.4%), verify -33.0 (-1.3 ms a step), prefill -1.2 | -39 (-1.7%), verify -35.9, prefill +2.6 | -3.4 (-0.5%) |
| 1 greedy (Cg vs Dg) | total -99.0 (-3.3%), decode -97.7 (-5.0%, -1.0 ms a token) | -114.8 (-3.9%), decode -5.9% | -19.8 (-2.5%) |
| lane vs base mtp3 (A vs D) | -145.0 (-6.4%), decode -13.6% | -163.4 (-7.3%), decode -13.7% | -33.5 (-5.5%) |
| lane vs base greedy (Ag vs Dg) | -126.9 (-4.3%), decode -6.6% | -130.5 (-4.4%) | -29.2 (-3.7%) |

On the 30 s clip the per-pair totals of every compared pair of arms do not overlap. My item-1 and lane-total gains
are a little smaller than the lane's. Part of that is the A arm's prefill (923.3 against 915.6-918.0 in the other
arms; its runs spread 913.8-934.3, so it is noise: the lane measured 916.2 vs 920.1). The decode gains match the
lane's within 1-2 points. For item 1 on the 5 s mtp3 clip the gain (-3.4 ms) is within the noise; the 30 s clip and
greedy show it clearly.

## Points for the user (not blocking)

- The default draft vocabulary is the source tree's absolute path, baked in at compile time. A binary run without
  that file warns and drafts over the whole head (same output, slower). An installer must ship the file and pass
  `--draft-vocab`. If the file exists but is malformed, load fails instead of falling back.
- VRAM: LM 2383 MiB with the subset (2261 without); the audio model still keeps its float32 matrices (4221 MiB) in
  every run here. That leaves 122 MiB less margin on a shared 8 GB card.
- tools/nemotron_draft_vocab.py was not run (no pyarrow), by the lane or by me. The shipped .bin equals the scout's
  file by md5. Its reproducibility is not shown, though this does not affect output.
- The fixed sizes (`top[8]`, h_arg 64 B, d_arg kMaxVerify) rely on kMaxVerify = 4 with no static_assert; the
  existing `d[8]` makes the same assumption.
- HIP was not built or run here. The argmax kernel's 32-lane reductions use the same `__shfl_xor_sync` pattern as the
  existing kernels in nemotron_h.cu.
- REPORT.md is still missing in the lane dir; the lane's NOTES.md and its structured summary have the numbers.
