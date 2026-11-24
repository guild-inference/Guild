# Round 20: faster Qwen prompts, GLM's DSA attention, Drama 3's tower and threads, shared batch drafts, and three fixes to code in main

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61, 15 SMs,
151 W cap, PCIe 3.0 x16), CUDA 12.0, GCC 13.3. Branch `lane20/int`: 44 commits on `perf/r19` 67a2802, which is in main
through PR #4. The measured binary is 90d1b04; the last commit, 257e64d, changes comments only. Kernels are at the
pinned revisions (guild-kernels-cpu `9b7281a`, guild-kernels-cuda `11936b5`); `backends.lock.json` is unchanged. Round
19's binary is 67a2802 built from the same pins. Its build-info reads e9e1996, the commit at configure time, but it
holds 99081ae's `GUILD_SPLIT_MIN_T`.

Benches (the command lines are in `runs/ab-arms.sh`):
- **Qwen decode.** `qwen-bench-r18`: Qwen3.8-Flash-Next UD-IQ4_XS with setup's flags (DFlash, `--spec 8`, the dense map,
  `--hc q8_0`, `--prefill auto`, `--prefill-mmq dense`), 32K context, `--resident-budget-gib 56`, prose and code, 2
  runs x 256 tokens.
- **Qwen prompts.** The same command with `--prompts pT.json` and `GUILD_NO_KV_CACHE=1`: prompts of 16, 1,008, 2,008,
  3,008, 4,008 and 6,008 tokens, one run each, 32 new tokens.
- **Ornith.** `ornith-bench-r18`: Ornith-1.5-35B Q4_K_M, its MTP layer, `--spec 4`, prose and code, 2 runs x 256.
- **Drama 3.** `drama3-run-r18`: Drama 3 Q4_K_M, `--resident-budget-gib 24`, one fixed text, seeds 11 and 12.
- **Nemotron.** `transcribe_run --mtp 3 --repeat 6` on a 5.0 s and a 29.6 s clip; medians of repeats 2-6.
- **StartLux.** `decision_run --repeat 5 --fresh` on 3 requests of 503, 1,470 and 940 tokens; mean of the 5 repeats.
- **Batch slots.** `tools/batch_test.py --batch N --n N --max-new 96` with the 32K Qwen config: 3 slots (they draft) and
  `--batch 7` (the engine runs 6 slots).
- **GLM.** `glm5_run --spec --ctx 32768`, 256 tokens, a 3,000-token prompt and a 5-token prompt, 3 rounds a load.

The integration A/B ran round 19 (b) and round 20 (i) in the order b1 i1 i2 b2, two alternated pairs, inside one gpu-run
hold per bench (Drama 3: one per seed; the batch slots: one per slot count), 10:16-11:16 UTC on 2026-11-23
(`runs/ab.sh`, `runs/ab-arms.log`). GLM ran next in three loads, round 19, round 20, round 19, 11:17-11:53 UTC
(`runs/glm2.sh`, `runs/glm.sh`).

Exactness: every run of the integration A/B and the GLM loads kept round 19's bits (`runs/abcmp.py` and the last line of
`runs/ab-summary.txt`; for GLM the parses named below):
- Qwen's decode ids `[38747, 38747, 41916, 41916]`, with "chunks of 8192 tokens" and "GPU cache 0 slots" in each log,
  and Ornith's `[30936, 30936, 65886, 65886]` (`runs/ab-decode-ids.txt`);
- Drama 3's WAV md5s, seed 11 `6e04c9b1` and seed 12 `f84288c9`, in every arm;
- Nemotron's ids and texts, and StartLux's answers, equal to round 19's;
- every batch slot IDENTICAL to its solo run, with round 19's slot texts;
- the prompt bench's output hashes (`runs/ab-qwen-prompts-table.txt`);
- GLM: one id line per prompt across all 24 generations of the three loads (`runs/glm-r19-load1-parse.txt`,
  `runs/glm-r20-parse.txt`, `runs/glm-r19-load2-parse.txt`).

Each lane's verifier checked its own lane's bits the same way (`runs/verify-*.txt`).

## Round 20 against round 19

From the integration A/B (`runs/ab-summary.txt`). "Pairs" are b1 against i1 and b2 against i2; changes are in % of round
19.

**Qwen decode** (`runs/ab-qwen-decode-*.out`):

| tok/s | round 19 | round 20 | pairs | mean |
| --- | ---: | ---: | ---: | ---: |
| prose | 51.53 / 51.42 | 52.02 / 52.07 | +0.96 / +1.27% | +1.12% |
| code | 65.03 / 65.16 | 65.35 / 65.77 | +0.50 / +0.94% | +0.72% |

Every run had the same windows: 91 prose windows of 2.81 tokens, 65 code windows of 3.94. Means of the 4 runs a round
(`runs/ab-qwen-decode-*.log`): a prose window 54.65 -> 54.05 ms, a code window 60.51 -> 60.07 ms; CPU dispatch 28.83 ->
27.94 and 34.13 -> 33.26 ms a window.

**Qwen prompts** (prompt ms, one run a prompt and arm; `runs/ab-qwen-prompts-table.txt`, taken from the runs' JSON;
their text output and logs are `runs/ab-qwen-prompts-*.out` and `runs/ab-qwen-prompts-*.log`):

| prompt | round 19 | round 20 | pairs | mean |
| --- | ---: | ---: | ---: | ---: |
| 16 tokens | 132 / 138 | 136 / 135 | +3.00 / -2.09% | +0.40% |
| 1,008 | 7,389 / 7,404 | 7,388 / 7,406 | -0.02 / +0.02% | 0.00% |
| 2,008 | 10,309 / 10,342 | 8,104 / 8,045 | -21.40 / -22.21% | -21.80% |
| 3,008 | 13,171 / 13,198 | 9,699 / 9,725 | -26.36 / -26.31% | -26.33% |
| 4,008 | 16,141 / 16,170 | 12,415 / 12,431 | -23.09 / -23.12% | -23.10% |
| 6,008 | 20,539 / 20,537 | 18,351 / 18,338 | -10.66 / -10.71% | -10.68% |

Changes 2 and 7 are the round's prompt changes. Their own verifiers measured -9.2% and -13.3% at 2,008 tokens, which
compound to -21%.

**Ornith decode** (`runs/ab-ornith-decode-*.out`):

| tok/s | round 19 | round 20 | pairs | mean |
| --- | ---: | ---: | ---: | ---: |
| prose | 82.79 / 82.55 | 83.24 / 83.46 | +0.54 / +1.10% | +0.82% |
| code | 96.92 / 96.66 | 98.17 / 97.71 | +1.29 / +1.09% | +1.19% |

The same windows in every run (126 prose windows of 2.03 tokens, 105 code windows of 2.44). A prose window 24.58 ->
24.38 ms, a code window 25.19 -> 24.90 ms (`runs/ab-ornith-decode-*.log`).

**Drama 3** (frames/s; `runs/ab-drama3-s11-*.log`, `runs/ab-drama3-s12-*.log`):

| seed | round 19 | round 20 | pairs | mean |
| --- | ---: | ---: | ---: | ---: |
| 11 | 47.06 / 47.07 | 50.19 / 50.25 | +6.65 / +6.76% | +6.70% |
| 12 | 48.12 / 46.99 | 50.02 / 50.18 | +3.95 / +6.79% | +5.35% |

The tower's time a run (the request thread's time in it) went from 1,892 / 1,894 to 1,607 / 1,607 ms (seed 11) and from
1,691 / 1,696 to 1,452 / 1,455 ms (seed 12): 286 of the 639 ms saved a run and 240 of the 456. The 86-row prompt went
from 520 / 525 to 472 / 471 ms and from 494 / 520 to 470 / 472 ms. The same holds ran round 20 with
`GUILD_DRAMA3_FUSE=0` (round 19's tower kernels and sampler): +5.34% (seed 11) and +3.77% (seed 12) over round 19, 80%
and 70% of the round's gain. So most of the gain does not come from Change 4. The tower's time still fell in that arm,
to 1,747 / 1,751 ms (seed 11) and 1,576 / 1,580 ms (seed 12); Change 10's `GUILD_HELPER_PIN` took 81-102 ms a run off
it in the second verifier's pairs. The other changes measured on Drama 3 are 3, 6, 9 (no cost) and 10; their own A/Bs
are in those sections, and Change 4 has the tower's arms.

**Nemotron** (ms, medians of repeats 2-6; `runs/ab-nemotron-*`):

| clip | phase | round 19 | round 20 | pairs | mean |
| --- | --- | ---: | ---: | ---: | ---: |
| 5.0 s | prefill | 347.3 / 347.9 | 336.8 / 335.9 | -3.02 / -3.45% | -3.24% |
| | decode | 179.4 / 178.8 | 179.7 / 179.2 | +0.17 / +0.22% | +0.20% |
| | total | 569.4 / 569.5 | 559.7 / 559.2 | -1.70 / -1.81% | -1.76% |
| 29.6 s | prefill | 912.5 / 921.8 | 883.7 / 878.7 | -3.16 / -4.68% | -3.92% |
| | decode | 965.6 / 1,018.2 | 965.7 / 963.5 | +0.01 / -5.37% | -2.75% |
| | total | 2,050.6 / 2,108.2 | 2,014.9 / 2,009.9 | -1.74 / -4.66% | -3.22% |

Change 6's Q8_0 pair scales alone measured -3.7% and -3.5% on these prefills. The 29.6 s clip's decode mean rests on one
slow round 19 run (1,018.2 ms); its other pair is +0.01%.

**StartLux** (mean ms of 5 repeats; `runs/ab-startlux-*`):

| request | tokens | round 19 | round 20 | pairs | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| lat-1q-500 | 503 | 1,098.75 / 1,095.73 | 1,087.33 / 1,090.89 | -1.04 / -0.44% | -0.74% |
| lat-3q-500 | 1,470 (898 shared) | 1,472.18 / 1,475.87 | 1,461.87 / 1,471.92 | -0.70 / -0.27% | -0.48% |
| prefill-1k | 940 | 2,080.04 / 2,057.06 | 2,029.45 / 2,042.54 | -2.43 / -0.71% | -1.57% |

No lane A/B'd StartLux's speed, so this change is not attributed to one. Its experts are Q8_0 on the library's rows,
which Change 6 does not change.

**Batch slots** (tok/s; `runs/ab-batch3-*`, `runs/ab-batch7-*`):

| slots | rate | round 19 | round 20 | pairs | mean |
| --- | --- | ---: | ---: | ---: | ---: |
| 3, drafting | overall | 54.7 / 55.2 | 56.5 / 56.2 | +3.29 / +1.81% | +2.55% |
| | from the first batch token | 59.1 / 59.5 | 61.2 / 60.9 | +3.55 / +2.35% | +2.95% |
| 6 (`--batch 7`) | overall | 67.6 / 67.7 | 68.2 / 68.0 | +0.89 / +0.44% | +0.67% |
| | from the first batch token | 70.9 / 70.9 | 71.4 / 71.3 | +0.71 / +0.56% | +0.63% |

3 slots: 95.38 / 94.59 -> 91.93 / 92.22 ms a window, of which drafts 11.48 / 11.27 -> 9.00 / 8.99 ms (Change 8); 46
windows and 164 of 247 drafts accepted in every run. 6 slots: 68.96 / 69.02 -> 68.38 / 68.59 ms a window. There the CPU
experts went from 53.47 / 53.35 to 51.93 / 51.98 ms and the GPU-reach wait from 15.13 / 15.62 to 16.31 / 16.35 ms: the
window is GPU-bound, so saved CPU time turns into wait.

**GLM-5.3-Flash** (`glm5_run`, three loads in the order of the rows; A and C are one load, alternated in-process as C A,
A C, C A). The raw outputs are `runs/glm-r19-load1.out`, `runs/glm-r20.out` and `runs/glm-r19-load2.out`, parsed by
`runs/glm-parse_ab.py` into `runs/glm-*-parse.txt`; the prompts are `runs/glm-prompts.txt` with `runs/glm-ids3k.txt`,
and the times `runs/glm-progress.txt`:

| load | 3,000-token prompt: tok/s by round | mean | step median ms | prompt ms | 5-token prompt: tok/s by round | mean |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| round 19, first | 21.08 20.90 20.76 | 20.91 | 86.90 | 38,926 | 23.33 23.06 22.87 | 23.09 |
| round 20, defaults (A) | 21.40 21.44 21.36 | 21.40 | 84.66 | 38,381 | 23.08 23.27 23.21 | 23.19 |
| round 20, round 19's GPU path (C) | 20.83 20.78 20.87 | 20.83 | 86.99 | 39,045 | 23.08 23.15 23.21 | 23.15 |
| round 19, second | 20.91 20.85 20.84 | 20.87 | 87.07 | 38,773 | 23.19 23.04 23.08 | 23.10 |

C is `GUILD_GLM_ATTEND3=0 GUILD_GLM_FUSE2=0xfffffffb` in round 20's binary. Step medians and prompt times are means of
the 3 rounds.
- 3,000-token prompt (decode at 3,000-3,256 cells): round 20 against both round 19 loads +2.44% (20.89 -> 21.40 tok/s),
  the step median -2.32 ms, the prompt -469 ms (-1.2%). A against C in one process: +2.75%, -2.33 ms, the prompt -665 ms
  (-1.7%).
- 5-token prompt: +0.40% against round 19's loads, +0.17% A against C. That is noise.
- The id lines: one per prompt in all 24 generations; round 20's 12 also checked by `runs/glm-r20-ids.txt` (PASS).
- The first load read GLM from a cold page cache (1,250 s); the others loaded in 92 s and 46 s (`runs/glm-loads.txt`).
  The VRAM lines are the same in all three loads: "8010 MiB free, 6140 MiB fixed, context 32768 tokens x 13184 B, 2 MiB
  left; 1 shared experts on the CPU" and "42 of 46 FFNs on the GPU" (`runs/glm-r20-vram.txt`).

**With experts read from the files** (the helper-threads lane's second verifier, on round 20's build 90d1b04 and round
19's; `runs/verify2-helper-threads.txt`, `runs/verify2-helper-threads-notes.txt`). Qwen with qwen-bench-r18's flags at
`--resident-budget-gib 24` (10,643 of 24,576 experts in RAM, the rest read from the files), 1 run x 64 tokens, 2 pairs
with round 19 first in each:

| tok/s | round 19 | round 20 |
| --- | ---: | ---: |
| prose | 6.43 / 7.16 | 13.80 / 11.24 |
| code | 12.41 / 12.85 | 18.86 / 15.58 |
| prose / code, decode unpinned (`GUILD_HOST_AFFINITY=0`), one run each | 13.35 / 17.92 | 13.43 / 18.10 |
| 3 batch slots, from the first batch token (2 pairs) | 8.8 / 8.7 | 17.5 / 17.9 |

Three more round 20 runs gave 13.51-13.88 prose and 18.92-19.04 code. The tokens are the same in every run, and every
batch slot is IDENTICAL to its solo run. Unpinned, both rounds run alike, so the difference comes from Correctness fix
(b). Drama 3 at `--resident-budget-gib 6`, round 20 only, `GUILD_HELPER_PIN=0` against the default, 2 pairs: 16.30 /
17.43 -> 17.64 / 17.49 frames/s, no loss.

**Exactness of the final build** (`runs/exactness-90d1b04-cmp.txt`, made by `runs/exactness-check2.sh` and
`runs/exactness-cmp.py`; the raw outputs are `runs/exactness-90d1b04-*`): 28 of 28 checks pass. They cover:
- Qwen's and Ornith's ids;
- the VRAM and layout lines of Qwen, Ornith, Drama 3 and StartLux;
- Drama 3's md5s, both seeds with the defaults and seed 11 with `GUILD_DRAMA3_FUSE=0`, `ks` and `ksawf`, and the tower's
  bits in those four modes;
- Nemotron's 12 transcripts and StartLux's 4 answers;
- the 6-slot, 3-slot and long-prompt batch runs;
- the prompt hashes of four prompt sets, one of them with a 31,918-token prompt and one through the MTP drafter, and the
  layout lines of the three without the MTP drafter;
- ctest: 126 of 133 pass and 6 skip on this PC (`iq_parity_fixtures`, `iq_parity`, `prefill_fused_moe_test`,
  `prefill_fused_iq_test`, `expert_multi_test`, `pool_stress`). The one failure is `ple_parity`, whose fixture GGUF is
  not on this PC (`runs/exactness-90d1b04-ctest.log`). `multi_model_isolation_test` and `pool_rows_slots_test` pass.

The one difference in the log lines is noted, not failed: the auto GPU cache's would-be size reads 57 slots against
round 19's 45. Neither build keeps a cache here (see Not checked here).

## Change 1: GLM-5.3-Flash's DSA attention two heads a warp (d9407a9, 2c8e551)

Past 2,048 cells of context, every query of GLM's 11 DSA layers attends the indexer's 2,048 selected cells on the GPU
(`runs/judge-lanes.txt`). In d9407a9, `mla_attend3_k` computes two heads a warp with the latent's 512 values fixed: a
cell's FP16 values are converted once for both heads, and the per-value bound checks are gone. `mla_reduce2_k` reads the
16 splits' maxima and sums once a block and issues a row's 16 loads together (windows of 1-3 tokens). 2c8e551 runs the
KDA short convolution and q and k's L2 norm as one kernel instead of two.

Bits: each head runs the old kernel's operations in the same order (the same fma chain per cell from +0, the same warp
sum, the maximum in cell order, the same `expf` corrections). The reduce keeps the split order. The fused convolution
keeps each channel's expression and the norm's order of sums; its values stay in registers instead of an exact FP32
round trip through memory. glm5.cu is built without `--use_fast_math`. `glm5_kernels_test` compares the old and new
kernels with memcmp (64 new checks). The verifier's own cross-check ran 3,312 attention and 1,290 convolution checks
with 0 differences.

Switches: `GUILD_GLM_ATTEND3` (default on; `0` restores round 19's kernels) and `GUILD_GLM_FUSE2` bit value 4 (default
on; `GUILD_GLM_FUSE2=0xfffffffb` restores the two launches).

The verifier ran one GLM process with the 3,000-token prompt and three arms in each of 3 rounds
(`runs/verify-glm-attend3.txt`):

| arm | step median ms | tok/s |
| --- | ---: | ---: |
| C, round 19's kernels | 86.52 | 20.91 |
| B, the attention and reduce | 84.57 | 21.40 |
| A, B and the one-launch convolution (the default) | 84.54 | 21.45 |

- B against C: -1.66 / -1.99 / -2.19 ms a step, mean -1.95 ms and +2.35% tok/s. The NextN draft went from 5.138 to 4.955
  ms a step and the verify from 81.19 to 79.34 ms.
- A against B: +0.25 / -0.09 / -0.25 ms (mean -0.03) at 3,000-3,256 cells, and a mean of -0.22 ms with the 5-token
  prompt. That is inside GLM's noise. The kernel alone goes from 7.46 to 5.79 us a call at one token.
- `glm5_kernels_test attend` (64 heads, 2,048 selected cells, the reduce included): 331.1 -> 244.0 us at one token,
  615.6 -> 412.6 us at two.
- On the integrated build, A against C: -2.33 ms and +2.75% (the GLM table above).

The sm_61 code of glm5.cu grows by 46 KB (430,720 -> 478,016 B; `runs/impl-glm-attend3.txt`), and GLM's VRAM line is
unchanged. Not measured end to end: contexts past 3,256 cells, and contexts larger than VRAM holds (the server's, up to
262,144 cells), where some cells' rows come from host memory. The verifier's bench of those staged shapes: 1 query of a
64-token window 1,134 -> 903 us, 8 queries 2,837 -> 1,962 us. Not taken: two attention variants the scout had measured
slower (Measured and not taken).

## Change 2: prompt chunks under 8,192 tokens lend their idle VRAM to the expert ring (de038b4, 347cccc, 859184b, aa57d12)

Qwen's prompt path carves its scratch for 8,192-token chunks. Below 4,096 tokens it took the routed walk: 8 ring slots,
with each layer's copies issued only after its routing (`runs/judge-lanes.txt`, `runs/impl-pf-tail-ring.txt`). A shorter
chunk now lays its scratch out for its own length and lends what it leaves idle as whole-blob ring slots: the region's
tail and R's rows past the chunk (de038b4), and the idle rows of five per-token buffers (347cccc). The implementer
counted 433 slots at 2,007 tokens, 302 at 4,007 and 171 at 6,007, where the ring had 32. Chunks of 1,536 tokens or more
whose ring reaches 192 slots then take the streamed walk, which issues the copies ahead. 859184b: with
`--prefill-mmq dense` (setup's flag on Pascal), dense MMQ products of the same rows (the GDN's qkv and gate, the shared
expert's gate and up) quantize those rows to q8_1 once. aa57d12 adds measured times to a comment.

Bits: each expert's MMQ multiplies the same blob bytes in the same groups of 16 experts, in id order. Only buffer
addresses and slots change; they stay 256-B aligned, so MMQ and cuBLAS pick the same kernels. A group can now be split
into parts, which is exact because MMQ on Pascal computes each row on its own. The q8_1 rows depend only on a row's
values and type.

Switches: `GUILD_PREFILL_TAIL_RING` (default on; `0` lends nothing and keeps the layout as carved. Change 7's block is
carved either way, so round 19's layout and ring also need `GUILD_PREFILL_ZERO_COPY=0`), `GUILD_PREFILL_TAIL_BUFS`
(default on; `0` lends only the region's and R's tails), `GUILD_PREFILL_TAIL_STREAM_MIN` (default 1536) and
`GUILD_PREFILL_SHARE_XQ` (default on).

The verifier ran the lane's build (before Change 7; the shared q8_1 rows on in both arms) off, on, on, off in one hold
(2 pairs), off = `GUILD_PREFILL_TAIL_RING=0` (`runs/verify-pf-tail-ring.txt`):

| prompt | off, ms | on, ms | change |
| --- | ---: | ---: | ---: |
| 1,008 tokens | 7,461 / 7,448 | 7,437 / 7,441 | -0.2% |
| 2,008 | 10,308 / 10,315 | 9,381 / 9,346 | -9.19% |
| 3,008 | 13,220 / 13,218 | 11,355 / 11,402 | -13.92% |
| 4,008 | 16,148 / 16,159 | 14,277 / 14,308 | -11.52% |
| 6,008 | 20,488 / 20,540 | 20,193 / 20,214 | -1.51% |

At 4,008 tokens the copy wait fell from 2.92 / 2.91 s to 0.97 / 0.95 s, and the hyper-connection read rose from 1.89 to
2.30 s, because the copies now run beside it. The shared q8_1 rows alone, one pair: 6,008 tokens 20,923 -> 20,869 ms,
7,600 tokens 24,102 -> 23,997 ms.

The path needs CUDA on Pascal, `--prefill-rows` (which `--prefill auto` picks on this card), no GPU expert cache, and no
peer or layer split. A server started with `--prefill 512` and no `--prefill-rows` takes neither this change nor Change
7. Not taken: the copy gate, the inline issuer and other stream thresholds (Measured and not taken).

## Change 3: the host combine reads the CPU's sums in place, and the drafter streams go first (5b2ce07, 181726f)

With the host combine, the CPU sums its experts' rows; Qwen, Ornith, Drama 3's backbone and the batch slots run this
way. Each post used to copy those sums from mapped memory into VRAM and then run the k = 1 combine to add the shared
expert. Now one kernel, `combine_sums_f4` (5b2ce07), reads the sums from mapped memory with float4 loads. That is one
kernel and one kernel boundary less per post: 96 a Qwen window, 80 an Ornith window, 40 a Drama 3 frame
(`runs/impl-win-combine.txt`). 181726f puts the DFlash and MTP drafter streams at the device's top priority and the
commit graph's kernel nodes at the least, so the work after a window wins dispatch.

Bits: the kernel does the bundled combine's `--use_fast_math` k = 1 operations (a flush-to-zero multiply, then a
flush-to-zero add) in the same order. `host_combine_parity` checks it bit for bit against the combine of the copied
sums. The priorities change scheduling only.

Switches: `GUILD_COMBINE_MAPPED` (default 1 with the bundled CUDA MoE backend, otherwise 0, since c412de7; 2 = gkc's
combine kernel on the mapped rows; 0 = the copy, then the combine) and `GUILD_WINDOW_PRIO` (default 1 on CUDA; 0 = round
19's priorities, which HIP builds always keep).

The verifier ran one process per model with the arms alternating by request, 4 pairs a prompt; cd = the copy and round
19's priorities, fp = the default (`runs/verify-win-combine.txt`):

| | cd, ms a window | fp, ms a window | tok/s |
| --- | ---: | ---: | ---: |
| Qwen prose | 54.750 | 54.460 | 51.39 -> 51.66 (+0.53%) |
| Qwen code | 60.565 | 60.273 | 65.03 -> 65.35 (+0.49%) |
| Ornith prose | 24.572 | 24.402 | 82.68 -> 83.25 (+0.70%) |
| Ornith code | 25.295 | 25.095 | 96.39 -> 97.15 (+0.79%) |

Two Qwen prose requests were disturbed; the two clean prose pairs were -0.26 ms each. Ornith was faster in 4 of 4 pairs
on both prompts. Drama 3, one process a run, 6 pairs against both switches off: -0.96, +0.55, +0.19, +0.23, +1.61 and
+2.72%, mean +0.73%, median +0.39%. With the defaults the tower took 10-30 ms longer a generation in every pair. The
verifier's reading, not traced: the commit, now at the least priority, runs beside it.

Not taken: `GUILD_COMBINE_MAPPED=2` as the default, and the lane's per-request A/B scaffolding (Measured and not taken).

## Change 4: Drama 3's tower step (0edf698, c553c5d, af21181, 557fa25, ee6e3db, 713b55c)

The tower runs `step_begin` (5 layers) on the critical path between two backbone steps, and `step_finish` (8 layers)
beside the next one (`runs/impl-d3-tower-fuse.txt`). Four faster paths, each with a letter of `GUILD_DRAMA3_FUSE`:
- k: the norms, adds and Q8_1 quantizes in fused 1024-thread kernels (0edf698);
- s: the sampler in AVX-512 with glibc's exp bits (c553c5d; af21181 moves its scratch from static TLS to the heap);
- a: attention with K and V staged in shared memory (557fa25);
- w: w1 and w3 as one 12,288-row Q8_0 matvec (ee6e3db).

713b55c runs a and w only in `step_begin`'s 5 layers; the letter f also puts them in `step_finish`'s 8.

Bits: each part gives the bits of the path it replaces. `drama3_tower_test` gives `8d3fea1524964286` with the default,
`0`, `ks` and `ksawf`, and the WAV md5s match in every arm. The fast sampler checks its exp against glibc's at load and
stays off where they differ.

Switch: `GUILD_DRAMA3_FUSE`, unset or 1 = `ksaw`; `0` = round 19's path; letters pick parts. HIP builds take only s
(dc65310).

The verifier ran 6 rounds of 10 runs, each round in one hold, 12 (round, seed) pairs (`runs/verify-d3-tower-fuse.txt`):
- the default against `0`: +1.75% (SE 0.30), 11 of 12 pairs at or above 0, the tower -0.45 ms a frame;
- the default against round 19: +1.56% (SE 0.25);
- `ks` against `0`: +1.88% (SE 0.44); `ksawf` against `0`: +1.78% (SE 0.23);
- the default against `ks`: -0.11% (SE 0.47). a and w cut the main thread's tower time by 0.10 ms a frame, and the rest
  of the frame grew about as much.

Per-arm means there: round 19 47.32, `0` 47.23, the default 48.05, `ks` 48.11, `ksawf` 48.07 frames/s.

The integration A/B ran these arms between its round 19 and round 20 runs, in the order b1 i1 ks1 ksawf1 f01 f02 ksawf2
ks2 i2 b2 (2 runs an arm a seed; `runs/ab-drama3-s*.log`):

| frames/s | seed 11 | seed 12 |
| --- | ---: | ---: |
| round 19 | 47.06 / 47.07 | 48.12 / 46.99 |
| round 20, `GUILD_DRAMA3_FUSE=0` | 49.56 / 49.60 | 49.47 / 49.23 |
| `ks` | 49.77 / 49.72 | 49.78 / 50.06 |
| `ksawf` | 49.89 / 49.72 | 49.56 / 49.96 |
| the default, `ksaw` | 50.19 / 50.25 | 50.02 / 50.18 |

The default led `0` by +1.29% (seed 11) and +1.52% (seed 12), `ks` by +0.95% and +0.36%, and `ksawf` by +0.83% and
+0.68%, so it stays `ksaw`. a and w are exact and cost nothing measured; whether they gain is not resolved (the
verifier's -0.11% ± 0.47). drama3_tower.cu's sm_61 code grows from 106,392 to 212,832 B (`runs/impl-d3-tower-fuse.txt`).
Not taken: f and `ks` as the default (Measured and not taken).

## Change 5: three fixes made while merging (dc65310, bf0084d, c412de7)

- dc65310 keeps glm5.cu's and the tower's inline PTX out of HIP builds: Correctness fixes (c).
- bf0084d: the tower runs [w1|w3] as one matvec only when `GUILD_DRAMA3_FUSE` has w. Before, it did so whenever the file
  put w3 right after w1. With this file that happens only after the w swap, so nothing changes here.
- c412de7: `GUILD_COMBINE_MAPPED` defaults to 1 only with the bundled CUDA MoE backend. `combine_sums_f4` spells that
  backend's flush-to-zero operations, and an external backend may be built without `--use_fast_math`, so with one the
  default stays the copy (0).

## Change 6: CPU expert rows (5bda3af, b952fde, b9ba668, ddcb19e, d1f96e4, 55d8ba4, 4f3ce6d)

Five items, each with its own switch:
- 5bda3af: Q4_K gate/up and Q4_K / Q6_K down rows two to an AVX-512BW pass (`kq_rows512`), used by Drama 3 and Ornith.
  `GUILD_KQ_ROWS512` (default 1).
- b952fde: the IQ4_NL down rows prefetch 6 rows ahead instead of 2. `GUILD_IQ4NL_PF` (default 6; 2 is round 19's).
- b9ba668: Qwen's IQ4_XS gate/up rows (its layer 2) on the AVX-512BW rows. `GUILD_POOL_IQ4XS` (default 1).
- d1f96e4: Guild's Q8_0 rows convert a block pair's four weight scales at once. They serve Nemotron's experts and GLM's
  CPU twins. `GUILD_Q8_PAIR` (default 1).
- ddcb19e: the pool counts parked workers as acks, each worker's state on its own 64-byte line. Off by default;
  `GUILD_POOL_ACK=1` opts in (Measured and not taken).

55d8ba4 adds comments, and 4f3ce6d runs `q8_0_rows512_test` at odd block counts and GLM's row lengths.

Bits: `kq_rows512_test` finds 173,808 results equal to ggml-cpu's dot and 2,453,760 equal to the library's table.
`q8_0_rows512_test` finds 271,728 results, 0 different, with `GUILD_Q8_PAIR` 0 and 1; `iq4nl_rows512_test` 20,448 and
`iq4xs_rows512_test` 30,672, 0 different (the fixer's runs, `runs/fix-cpu-rows.txt`). The prefetch distance and the pool
protocol change no arithmetic.

The verifier ran the same binary with arms by environment, each model in one hold (`runs/verify-cpu-rows.txt`):
- Drama 3, the kq rows on against off with acks off in both arms, 3 pairs: +0.06, +0.65 and +1.55%, mean +0.75%. The
  86-row prompt went from 490-497 to 465-469 ms (-5%).
- Nemotron, `GUILD_Q8_PAIR` 1 against 0, 2 pairs: prefill 346.2 -> 333.2 ms on the 5 s clip (-3.7%) and 909.6 -> 877.7
  ms on the 29.6 s clip (-3.5%); decode within 0.5%.
- Qwen, the prefetch and IQ4_XS items against `GUILD_IQ4NL_PF=2 GUILD_POOL_IQ4XS=0`, with the acks on in both arms (the
  lane's default then), 2 pairs: CPU dispatch -0.42% (prose) and -1.23% (code) a window; tok/s +0.0% and +0.1%. Qwen's
  window is GPU-bound here.
- Ornith, the kq rows and the acks against neither: no change over 2 pairs.

The implementer's profile found 13 of Drama 3's 40 layers 5-9 us slower with the kq rows while a busy unpinned thread
sat on a pool worker's SMT sibling (`runs/impl-cpu-rows.txt`). That led to Change 10.

## Change 7: the streamed prompt walk reads its experts in place (683f8dc, b0ce130, 3e55982, fedd0be, 100879d, d6a174a, 0d54ef4)

After Change 2, the streamed walk still gathered each group's experts from the ring slots into group buffers before MMQ.
That gather took 0.57-0.70 s a chunk and 2.8 s at 32K (the implementer). Now the ring's 32 slots and 16 more are one
block, zeroed once, with the slots 3,500,640 B apart (a stride computed from the pack). Each group of up to 16
consecutive ring entries runs MMQ in place: gate/up from the group's first slot with the slot stride, down from the same
pointer plus the layer's offset. The 16 extra slots double as the group buffers when a chunk takes the routed walk.
- 3e55982 lends the idle tails only to a chunk that then streams.
- fedd0be lets the in-place walk stream below 4,096 tokens once its ring holds 78 slots (the block's 48 and 30 lent,
  measured at a 3-slot server's 4,608-token layout); the gather walk keeps 192, a number never measured.
- b0ce130: shared q8_1 rows lost past the first slice fall back to each product's own.
- d6a174a fixes F1, a latent stale zero tail (below).
- 0d54ef4 gives the Stager its zero tails only where the block is carved.
- 100879d adds the measured prompt times to a comment.

Bits: MMQ reads the same bytes in place. Each blob is followed by 256 zero bytes, as in the zeroed group buffer, so
MMQ's over-read sees the same zeros. The Stager's DMA carries them; a DMA from a page-locked arena gets a 256-B memset.
F1: after the MTP drafter's draft_kv had written over the lent slots, a request that repeated the lent layout kept a
stale "tail is zero" mark. A pinned DMA could then skip its memset, and the over-read could see leftovers (NaN after a
bad scale). It needs a pack whose first and last layers' blobs are the same size, so it cannot happen with UD-IQ4_XS.
d6a174a clears the marks on that return.

Switches: `GUILD_PREFILL_ZERO_COPY` (default 1; 0 = separate slots and the gather), `GUILD_PREFILL_TAIL_LEND_STREAMED`
(default 1) and `GUILD_PREFILL_TAIL_WALK_RING` (by default 78 slots in place, 192 gathered). Change 2's whole path is
`GUILD_PREFILL_ZERO_COPY=0 GUILD_PREFILL_TAIL_LEND_STREAMED=0`.

The verifier measured on top of Change 2, off = `GUILD_PREFILL_ZERO_COPY=0`, one hold a prompt set, in the order off,
on, on, off (`runs/verify-pf-zero-copy.txt`):

| prompt | off, ms | on, ms | change |
| --- | ---: | ---: | ---: |
| 1,008 tokens | 7,458 / 7,452 | 7,456 / 7,448 | 0.0% |
| 2,008 | 9,334 / 9,414 | 8,135 / 8,111 | -13.3% |
| 3,008 | 11,427 / 11,459 | 9,782 / 9,755 | -14.6% |
| 4,008 | 14,484 / 14,361 | 12,464 / 12,477 | -13.5% |
| 6,008 | 20,214 / 20,258 | 18,385 / 18,404 | -9.1% |
| 7,600 | 24,106 / 24,111 | 23,181 / 23,159 | -3.9% |
| 9,001 (an 8,192-token chunk and 808) | 34,087 / 34,023 | 33,127 / 33,112 | -2.7% |
| 31,918 | 105,263 / 105,886 | 101,990 / 102,190 | -3.3% |

At 4,008 tokens the gather went from 606 to 0 ms, the copy wait from 940 / 948 to 51 / 50 ms and gate/up from 2,810 /
2,823 to 2,425 / 2,421 ms. At 31,918 tokens the copy wait rose from 2.49 / 2.42 to 4.33 / 4.26 s, because a slot is now
held until its group's down product and 8,192-token chunks have 48 slots; the prompt was still 3.48 s faster. At a
3-slot server's 4,608-token layout, chunks of 2.5K-4.1K tokens, which took the routed walk there, ran 5.0-11.4% faster
in place (the implementer, 2 runs an arm; `runs/impl-pf-zero-copy.txt`).

On a page-locked expert arena the walk reads in place too. That arena is `GUILD_PIN_EXPERTS=1`, or the default on a
one-NUMA-node PC whose RAM copy of the experts (`--resident-budget-gib`, or every expert if that is less) is 24 GiB or
less, where the automatic PCIe offload page-locks it; this two-node PC takes it only with the switch. The fixer, 2
pairs with `GUILD_PIN_EXPERTS=1`, the default against `GUILD_PREFILL_ZERO_COPY=0` (`runs/fix-pf-zero-copy.txt`): 2,008
tokens 8,647 -> 8,573 ms (-0.9%), 4,008 tokens 13,751 -> 12,460 ms (-9.4%). The verifier's pairs: +0.2% and -8.9%. A
2K chunk streams all 24,576 blobs (55.43 GiB) and is bound by the copies there (the copy wait 0.31 -> 1.65 s), so
reading in place has nothing to gain at 2K.

Not taken: the F2 DMA tail and `GUILD_PREFILL_RING=64` (Measured and not taken).

## Change 8: the drafting batch slots share one pass of the DFlash head (22ab72b, 01b6bfc, 4af3bad)

Batch slots ran one full DFlash draft per drafting slot: an 8-row block through the drafter's 5 layers, the 236 MB head
over 7 rows, a sync and a selector, although each slot keeps only 1-2 drafts (`runs/impl-dflash-slots.txt`). The draft
is now split into its layers, its head and its selector. Each slot's layers run as before; then one head pass reads all
the slots' rows (up to 8 a pass), with one sync, and each slot's selector walks its own rows. 22ab72b adds the shared
pass and its test, 01b6bfc uses it in the batch window, and 4af3bad makes the test portable.

Bits: the drafter's kernels compute each column the same way at any column count. `dflash_kernels_test` checks that the
shared pass gives the drafts and probabilities of one draft per slot, bit for bit, on random weights. The verifier's
check with the real drafter and 4 real context sets: 1,488 jobs, 0 different. The tokens come from the main model's
verify windows in any case: a draft difference would change only speed.

Switch: `GUILD_BATCH_DFLASH_SHARED_HEAD` (default 1; 0 = one draft per slot). With `GUILD_DFLASH_K4=0` the drafts fall
back to one draft per slot.

The verifier ran one process a run, off first, 2 pairs each (`runs/verify-dflash-slots.txt`):

| | window ms, off | window ms, on | drafts ms, off | drafts ms, on | tok/s from the first batch token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 slots | 93.55 / 94.65 | 91.05 / 90.92 | 11.36 / 11.33 | 9.03 / 9.00 | +2.5 / +3.7% |
| 4 slots (`GUILD_BATCH_DRAFT_SLOTS=4`) | 99.47 / 100.77 | 95.80 / 95.39 | 15.59 / 15.65 | 11.84 / 11.89 | +3.4 / +5.0% |

The solo path is unchanged: qwen-bench-r18 with `--runs 1`, 2 pairs against the lane's base (int's c412de7), prose +0.9
/ +0.3% and code +0.4 / +0.1%. Not taken: drafts at 4 slots by default, row-sized blocks and a 2-row block for one-token
drafters (Measured and not taken).

## Change 9: each model's CPU expert table its own (f5bf450, c26f6bb, 95b5898)

A correctness fix, described in Correctness fixes (a). It changes no switch. Its verifier measured no cost: Drama 3 seed
11, 4 pairs with the order balanced, 48.72 -> 48.74 frames/s (+0.04%; `runs/verify-pool-rows-iso.txt`).

## Change 10: Drama 3's request, tower and codec threads (a731b64, 4e2ad70, 2b08576, b57ebfa, f1262df, 90d1b04, 257e64d)

During a Drama 3 decode the request thread and its two busy helper threads were unpinned. The helpers are the tower's
`step_finish` beside the backbone and the codec. The OS often put them on SMT siblings of pool workers: the cpu-rows
lane saw the helpers on CPUs 35 and 36, the siblings of workers 3 and 4 (`runs/impl-cpu-rows.txt`).
- a731b64 names the helpers d3-tower and d3-codec.
- 4e2ad70 gives the expert pool its helper CPUs: the host core's free SMT siblings, CPU 32 here. There are none unless
  the workers are pinned one a core.
- 2b08576 pins the helpers to those CPUs and the request thread to the pool's host core, CPU 0.
- f1262df pins the request thread only when there is a helper CPU, every expert is in the RAM copy (or in a GPU cache of
  `--expert-cache N` slots; an auto cache is filled later and does not count), and `GUILD_HOST_AFFINITY` is not 0.
  Outside that layout the pin gained nothing or cost: the first verifier measured -0.37% over 6 pairs with the process
  under `taskset -c 0-31` and -1.64% over 4 pairs with 16 workers (`runs/verify-helper-threads.txt`).
- b57ebfa starts `fill_many`'s fetch threads on the load thread's mask: Correctness fixes (b). 90d1b04 and 257e64d
  change comments only.

Bits: only which CPU a thread runs on changes. The second verifier's 29 Drama 3 WAVs all match.

Switch: `GUILD_HELPER_PIN` (default 1; 0 = every thread where the OS puts it). `GUILD_HOST_AFFINITY=0` also skips the
request thread's pin.

The second verifier ran on the integrated build 90d1b04: drama3-run-r18, `GUILD_HELPER_PIN=0` first, 8 alternated pairs
over both seeds (`runs/verify2-helper-threads.txt`):
- +1.43% (se 0.48), 7 of 8 pairs above 0: 49.30 -> 50.00 frames/s. The tower took 81-102 ms less a run (mean 93 ms
  over the 8 pairs).
- The gain depends on where the OS had put the unpinned request thread in the off run. On the host core already (5
  pairs): +0.48%. On a worker's SMT sibling (3 pairs): +3.00%.
- The lane's runs (+2.61%), the first verifier's (+2.88%) and the fixer's (+2.21%), 6 pairs each, mostly had that thread
  on CPU 48, a worker's sibling. In one hold the lane's build and the integrated build gave +1.97% and +2.52% (4 pairs
  each), so the spread comes from the machine's state at the time, not from the build.
- The pins are hard: with a busy loop on CPU 32, the default ran 0.83% slower than `GUILD_HELPER_PIN=0` (the first
  verifier, 2 pairs).

Not taken: the request pin with experts read from the files, and other places for the helpers (Measured and not taken).

## Correctness fixes

### (a) Two models in one process (f5bf450, c26f6bb, 95b5898)

fc15389 (2026-11-02, in main) gave the expert pool its AVX-512BW rows through two wrapper functions over one
process-wide base table, which every model load replaced. The wrappers pass the formats they do not reroute to that
base. So once a second model loaded, the first model's rows of those formats ran on the second model's table.

`multi_model_isolation_test` loads Ornith on the library's table as Model A. Then it loads the same model with a GEXEC
artifact that selects the interleaved AVX2 table as Model B (gate row r at 2r, up row r at 2r + 1). On round 19's build
Model A's tokens change once B is loaded, and the test fails at its line 381 (`afterB_A == refA`). The first integration
check found it: 27 of 28 checks, with ctest's `multi_model_isolation_test` failing (`runs/exactness-c412de7-cmp.txt`,
`runs/exactness-c412de7-ctest.log`). The bisect (`runs/mmi-log.txt`, `runs/mmi-log2.txt`, `runs/mmi-run.sh`,
`runs/mmi-run2.sh`, and one `runs/mmi-<run>.txt` a run):

| build | result |
| --- | --- |
| round 19 (67a2802), twice | fails at line 381 |
| the pf-tail-ring and glm-attend3 lane builds | fail at line 381 |
| lane20/int at c412de7: defaults (3 runs); `GUILD_COMBINE_MAPPED=0`; `GUILD_WINDOW_PRIO=0`; the three prompt switches off; all five off; `GUILD_HOST_COMBINE=0` | fail at line 381 |
| the cpu-rows lane build | passes, but Model B gives 220 220 16 15 19 24 |
| lane20/int at 4f3ce6d (cpu-rows merged) | passes, but Model B gives 220 220 16 15 19 24 |
| the cpu-rows lane build with its four switches off | fails at line 381 |
| lane20/int at 95b5898 (the fix) | passes; Model A 271 248068 198 760 1156 369, Model B 781 513 5904 3808 660 14145 |

Without Change 6's new gate/up rows, Model B's reference tokens are 781 513 5904 3808 660 14145 in every run. Change 6
hid the first bug and added a second:
- 5bda3af rerouted every one of Ornith's formats, so Model A's rows no longer reached the shared base.
- The new Q4_K and IQ4_XS gate/up rows (5bda3af, b9ba668) read [gate | up] rows. On Model B's interleaved rows they
  computed other values. The test passed because it compared B only with B.

The fix, f5bf450: `pool_rows512(base, gate_up_interleaved)` wraps a table only when all four of its functions are the
bundled library's (`guild_cpu_backend_table_v2()`) and its gate/up rows cannot be interleaved. A GEXEC table, fused or
interleaved, or an external CPU package's table keeps its own rows. The wrappers have eight slots, a pair of functions
each, and every model on the library's table shares slot 0. c26f6bb adds `pool_rows_slots_test`, which wraps fake tables
that write markers ("ok: 8 slots (8 taken here, 3 bases refused), 0 failures", `runs/mmi-int95b-slots.txt`).
`multi_model_isolation_test` now also checks that B's pool table is its selected GEXEC table (`runs/mmi-int95b-1.txt`).
95b5898 fixes a comment.

Single-model bits are unchanged, because the library's table is still wrapped, into slot 0. The verifier stopped Qwen,
Ornith, Drama 3 and StartLux loads in gdb and found each pool bound to it (`runs/verify-pool-rows-iso.txt`). One thing
changes for users: a model bound to a GEXEC table or an external CPU package's table no longer gets the AVX-512BW rows.
It computes its own table's bits, likely more slowly; that is not measured, since no such model is used on this PC.

### (b) The text decode with experts read from the files (b57ebfa)

A new Linux thread inherits its creator's CPU mask. `FileExpertSource::fill_many` reads the experts the RAM copy does
not hold on up to 8 threads a layer: the thread that drives the layer and up to 7 it starts. `GuildSession::generate`
pins its decode thread to the pool's host core (`GUILD_HOST_AFFINITY`, default on), and a batch pins its host thread the
same way. So in round 19 the fetch threads of such a decode ran on CPU 0. This came to light because Change 10's first
request pin did the same to Drama 3 at `--resident-budget-gib 6`: 17.62 / 16.36 -> 7.06 / 6.99 frames/s (the first
verifier, `runs/verify-helper-threads.txt`).

b57ebfa records the loading thread's mask when the expert files are opened. `fill_many` starts its threads inside a
`SpawnAffinity` of that mask: the creator runs on that mask while it starts them, so they inherit it, and it gets its
own pin back before its own share of the copies. The bytes read are the same. A side effect: a CPU restriction applied
to the process after load, such as `taskset -a -p`, no longer reaches these threads; cpusets still do.

The second verifier's runs are in the "With experts read from the files" table above: Qwen at
`--resident-budget-gib 24`, 6.43 / 7.16 -> 13.80 / 11.24 tok/s prose and 12.41 / 12.85 -> 18.86 / 15.58 code, and 3
batch slots 8.8 / 8.7 -> 17.5 / 17.9 tok/s. A thread poller running every 0.5 ms caught 2,510 new threads in round 19's
pinned decode, all allowed CPU 0 only, and 4,018 in round 20's, all allowed CPUs 0-63.

### (c) HIP builds since 4f7523a (dc65310)

`cmake/hip_backend.cmake` compiles every `src/*.cu` for HIP. glm5.cu's `q8_1_warp` has used NVIDIA inline PTX since
4f7523a (2026-11-08, in main), which a HIP build cannot compile (read from the code; no HIP build was run), and round
20's tower adds more. dc65310 keeps the PTX to NVIDIA builds. On HIP, `q8_1_warp` writes the Q8_1 block in plain
operations (IEEE division, `roundf`, then the clamp); its bits against the CUDA path are not checked.
`GUILD_DRAMA3_FUSE` takes only s there. The CUDA SASS of glm5.cu, drama3_tower.cu and verify_combine.cu is the same
before and after (`runs/hip-fix-check.txt`). Not built with ROCm, which this PC does not have: the HIP branches were
only compiled with nvcc and `-D__HIP_PLATFORM_AMD__`.

## Measured and not taken, and options left off by default

- **`GUILD_POOL_ACK=1`** (ddcb19e, opt-in). With acks the host waits for every worker to ack every epoch, so a worker
  the OS deschedules or puts to sleep stalls the layer. ddcb19e names the PCs in that case: those with no free SMT
  sibling (SMT off, 6-core 6-thread parts, VMs) and those running a parallel build; none was run here. The verifier,
  bits unchanged (`runs/verify-cpu-rows.txt`):
  - two busy loops pinned to worker CPUs 7 and 23: Drama 3 21.32 / 16.17 frames/s with acks against 41.61 / 40.54
    without, two runs an arm (2.2x slower); Qwen prose / code 21.06 / 26.14 against 36.10 / 43.03 tok/s, one run an arm
    (-42% / -39%);
  - a task busy 2 ms every 20 ms on one worker CPU: Drama 3 -2.9% (2 pairs);
  - workers sleeping between frames (`GUILD_POOL_SPIN_US=1000`): Drama 3 -1.25% (2 pairs).

  On an idle machine the gain is not resolved: over 6 idle pairs (the implementer's, the verifier's and the fixer's),
  +0.32 frames/s, se 0.34 (+0.7%; `runs/fix-cpu-rows.txt`).
- **`GUILD_PREFILL_RING=64`** at the 8,192-token layout (the fixer, 2 alternated pairs, `runs/fix-pf-zero-copy.txt`).
  One 8,192-token chunk (the 9,001-token prompt): 33.16 -> 32.83 s (-1.0%), the copy wait 4.31 -> 3.79 s. It costs 107
  MiB of VRAM, and the auto GPU cache's would-be size drops from 57 to 11 slots, which can change decode text on cards
  that keep an auto cache (none is kept here). Not measured at 32K.
- **`GUILD_BATCH_DRAFT_SLOTS=4` with `GUILD_BATCH_DRAFT_BLOCK=rows`** (the implementer, `runs/impl-dflash-slots.txt`;
  the verifier did not re-measure it). At 4 slots: 61.9 / 62.1 against the default's 61.3 / 61.2 tok/s from the first
  batch token (+1.0 / +1.5%; by default drafts run only while 3 or fewer slots decode). But rows also applies at 3
  slots, where it lost 2.5 / 1.7% on the lane's base: acceptance fell from 66.4% to 61.8%, 48 windows instead of 46.
  With whole blocks, drafts at 4 slots still lose 0.3-0.5% to the default (60.9 / 61.3 against 61.2 / 61.5). A 2-row
  block only for a slot drafting one token lost 1.8 / 2.6% at 3 slots and was reverted.
- **Wrapping GEXEC fused tables with the AVX-512BW rows.** Not done: it needs a committed test per table, like
  `kq_rows512_test`'s table check. A scratch comparison by the pool-rows-iso implementer found the four GEXEC tables
  give the library's bits on their own layouts (Q4_K/Q4_K, Q4_K/Q6_K, IQ4_XS/Q8_0, IQ4_XS/IQ4_NL and IQ3_S/IQ4_NL at
  1-16 tokens, 0 different; `runs/impl-pool-rows-iso.txt`). Speed not measured: no GEXEC model is on this PC.
- **The F2 DMA tail**: a pinned blob's zero tail as a 256-B copy from pinned zeros on the copy stream, instead of a
  memset. +0.5% at 2K and +0.2% at 4K against the memset (the fixer's page-locked pairs of Change 7,
  `runs/fix-pf-zero-copy.txt`); dropped.
- **Change 2's copy gate**: a layer's copies wait for its attention half's hyper-connection read. The implementer, 2
  pairs, as a patch only: 3,008 tokens +1.9% (slower), 4,008 -1.3%, 6,008 -2.9%. It loses at 3K, so it is not on the
  branch. Issuing the copies inline (`GUILD_PREFILL_ISSUER=0`, one run against one default run): 2,008 tokens 11.50 s
  against 9.35 s, 4,008 16.02 s against 14.23 s. Streaming from 2,048 tokens instead of 1,536 would give up the streamed
  walk's -8.8% at 2K, and from 3,072 also its -11.5% at 3K (`runs/impl-pf-tail-ring.txt`).
- **`GUILD_COMBINE_MAPPED=2`** as the default (gkc's combine kernel on the mapped rows, no new device code): equal
  overall, but Ornith's 4-row windows ran 58-113 us slower than the copy in 4 of 4 measurements (the implementer,
  `runs/impl-win-combine.txt`). It stays an option. The lane's per-request A/B scaffolding (its third commit) is not
  merged.
- **`GUILD_DRAMA3_FUSE` with f, or `ks`, as the default.** f puts a and w in all 13 layers. In the integration A/B the
  default led `ksawf` by +0.83 / +0.68%; in the verifier's 12 pairs it trailed `ksawf` by 0.03% (SE 0.26), and `ksawf`
  against `ks` was -0.08% (SE 0.35): the tower 0.20 ms a frame shorter, the rest 0.22 ms longer. The default led `ks`
  by +0.95 / +0.36% in the integration A/B and trailed it by 0.11% (SE 0.47) in the verifier's pairs.
- **Drama 3's request pin with experts read from the files.** With `fill_many`'s threads already on the load's mask, the
  pin still cost 9.83% at `--resident-budget-gib 6` (the fixer, 4 pairs, 0 of 4 above 0; `runs/fix-helper-threads.txt`),
  so f1262df pins only with every expert in RAM. Other places for the helpers (the implementer,
  `runs/impl-helper-threads.txt`): the tower thread alone on CPU 32, or both helpers on other workers' siblings, gave no
  gain; with the request thread pinned, the helpers on other workers' siblings came 0.7% below both on CPU 32 (4 of 4
  rounds).
- **GLM attention variants the scout measured** (not built in the lane, `runs/impl-glm-attend3.txt`): a prefetching
  `mla_attend4_k` took 235.5 us at one token against 225.3 us without the prefetch; forcing 3 blocks an SM, 288.8 us.

## Not checked here

- **VNNI hosts.** On AVX512-VNNI CPUs (Ice Lake and later, Zen 4) the library takes other rows at 2 or more tokens, and
  the new kq and IQ4_XS rows were not checked bit for bit against them. `expert_multi_test` and `pool_stress` skip on
  this CPU, and `pool_test` is not registered here (CMake leaves it out without its `pack/full/experts.bin` fixture).
  Run `kq_rows512_test` and `pool_stress` on such a CPU before a release.
- **HIP and Windows builds.** Neither was compiled; there is no ROCm or MSVC here. dc65310's HIP branches were only
  type-checked. On Windows the helper CPUs are not detected, so Drama 3's threads stay unpinned there. MSVC builds
  compile the fast sampler out; a GCC or Clang build turns it off if the CRT's exp differs from glibc's.
- **The helper pin on other CPUs.** `GUILD_HELPER_PIN`'s default was measured only on this PC (31 pool workers on CPUs
  1-31, the host core's sibling CPU 32 free). On a PC with few cores the host thread runs a larger share of each pool
  phase (1/8 on an 8-core part against 1/32 here), and two busy helpers on its sibling could cost there; not measured
  (`runs/verify2-helper-threads.txt`, `runs/impl-helper-threads.txt`).
- **Configs that keep an auto GPU expert cache.** The cache is sized from the VRAM left after load, and round 20 leaves
  more: the prompt bench's GPU memory in use at the end of its runs (nvidia-smi, the server still loaded) reads 7,557
  MiB against round 19's 7,585 (`runs/ab-qwen-prompts-table.txt`), and the would-be cache size moved from 45 to 57
  slots (none is kept on this card; `runs/exactness-90d1b04-cmp.txt`). On a card that keeps an auto cache, decode text
  can differ from round 19's by the cache size, which decides the experts that run on the GPU; both builds compute
  exactly. Compare texts only at equal "GPU cache N slots".
- **GLM past 3,256 cells and GLM inside guild-generate** (the server's path, up to 262,144 cells): not run. The device
  code guild-generate loads takes 8.00 MiB of VRAM in both builds (71 -> 72 modules, 25,570,096 -> 25,741,088 B of
  cubin), measured at c412de7, after which no `.cu` file changed (`runs/modmem-c412de7.log`). The old and new GLM
  kernels are shown bit-equal for sm_61's ptxas only.
- **F1** cannot occur with UD-IQ4_XS, so its fix is only shown to change nothing here.
- Peer and multi-GPU paths were not run.

## Open levers (estimates, not measured)

From the judge's rejected list and numerics options (`runs/judge-rejected-numerics.txt`) and the lanes' open issues
(`runs/impl-*.txt`, `runs/verify2-helper-threads.txt`).

- **Prompt copies gated by size.** Gate a layer's copies only when the chunk is at least ~10 times the ring: it might
  keep the gate's 4K and 6K wins without its 3K loss.
- **More ring slots at whole 8,192-token chunks.** The copy wait rose from 2.49 / 2.42 to 4.33 / 4.26 s at 32K with 48
  slots (Change 7's verifier); `GUILD_PREFILL_RING=64` is the one measured point.
- **A persistent fetch-thread pool**, started once at load. Qwen at budget 24 starts about 1,000 threads a second, and
  CPU 32 is ~48% busy beside the pinned decode.
- **A pool protocol that tolerates descheduled workers** and still saves the wake's locked subtract.
- **Batch drafts.** All slots' drafter contexts in one product after a window: ~0.4 ms of a 3-slot window. The per-slot
  bodies' small kernels batched across slots: ~0.5 ms per extra body, but it needs per-slot scratch VRAM.
- **The library's Q8_0 pair scales** (a CPU lock bump). Nothing on this PC, where those rows already read 224 GB/s at
  1-2 tokens; +14-18% at 4 tokens on 8-core PCs.
- **GLM.** AVX-512 Q3_K rows for the NextN block: ~0.15-0.45 ms a step (0.2-0.6%). A faster `route_k`: ~0.4-0.5 ms a
  step at best. Multi-block top-k selection: only past ~64K cells (52 us at 128K, 96 us at 262K). The IQ4_XS rows' d
  vector: ~0.2 ms a step. 16-warp attention blocks for 2+ token windows: at most ~15 us a step.
- **Decode windows.** The next draft before the commit: ~0.07 ms a window. A speculative add_context in the last layer's
  bubble: ~0.4%. Ornith's MTP argmax in one pass: ~0.1 ms a window (0.4%), a CUDA lock bump. MTP step 0 reusing the
  catch-up's products: ~60 us a window.
- **The CPU rows' combine inside the pool.** One scout models +0.9% for Drama 3; the other measured the combine split
  over the down rows' writers slower. A barrier-free microbenchmark comes first.
- **Profile-ranked prefetch in the routed prompt walk.** The hit rate is unknown; count it first.
- **Numerics options** (they change bits, so each needs its own quality check):
  - GLM: group A's shared expert on its CPU twin in DSA layers past 2,048 cells (not written; the judge would make it an
    opt-in switch, `GUILD_GLM_SHEXP_DSA=1`): ~-1.5 ms a step at long context (~+1.6%).
  - Qwen prompts: the shared expert's down (K = 640) through MMQ: ~-0.4 s at 32K (~-0.4%).
  - Qwen prompts: a chunked (WY) delta rule for the GDN recurrence. The recurrence is 3.37 s of a ~106 s 32K prompt, so
    at most ~3%.
