# Eighteenth round: five lanes across the models, every output the same

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap, PCIe 3.0 x16), CUDA 12. Branch `perf/next`, the pinned kernel revisions (guild-kernels-cpu
`9b7281a`, guild-kernels-cuda `de65d60`), except the one lane that changes the CUDA kernels.

## How the round ran

1. **Scouts.** Six of them, one per area: Qwen decode, Qwen prompts, GLM, Drama 3, the other three models, and the
   CPU kernels. They measured where the time goes and proposed changes.
2. **Selection.** A judge re-derived the scouts' key evidence from the code and their logs and picked five lanes
   (`lanes/judge_summary.json`; the rules are in `lanes/BRIEF.md`).
3. **Implementation.** Each lane was built and measured in its own worktree, with every model run under one machine
   lock.
4. **Verification.** An independent verifier then tried to refute each lane: it reviewed the diff, re-ran the
   bitwise tests, ran its own A/B and ran the other models that share the touched code (`lanes/*/VERIFY.md`).

The lanes' notes are in `lanes/*/NOTES.md`.

## What changed, and the integration check

Four lanes are merged into perf/next. The integrated build was compared with perf/next as it was (`a10d867`), with
each model run alternated (`validate.sh`, `validate.progress`). Every output is the same in both builds: the token
hashes, Drama 3's WAV md5s, GLM's id line, Nemotron's transcripts and StartLux's answer probabilities.
`ctest` (32 tests: mmvq, gr_parity, route_window, router_fast, host_combine, native_expert, expert_parity, pool, glm5,
verify, dflash, qwen, prefill, mmq, iq4xs, cpu_moe, nemotron) passes, and so does `drama3_tower_test` against the
reference.

**Qwen3.8-Flash-Next prompts** (setup's flags, no prefix cache; one run each). The `qwen-prompt` lane's changes:
- **(A) Head-less windows.** A prompt of up to 512 tokens is read through the verifier in 8-token windows, and those
  windows now skip the head. The head (a 2560 x 248320 Q6_K product and a sample) only produced a token the ingest
  threw away. About 6.5 ms an 8-token window.
- **(C) A 32-slot streamed walk** for chunks of 4,096 tokens or more under `--prefill-rows`. The 8-slot routed walk
  waited for each copy and re-streamed most experts.
- **(D) Indexer block scores 8 queries at a time** from position 16,384.

| Prompt | Before | After | Change |
| --- | ---: | ---: | ---: |
| 100 tokens | 965 ms | 872 | -9.6% |
| 200 | 1,911 | 1,722 | -9.9% |
| 400 | 3,852 | 3,462 | -10.1% |
| 800 / 1,600 (the prompt path, unchanged) | 6,804 / 9,042 | 6,794 / 9,087 | |
| 4,000 | 16,516 | 16,512 | |
| 7,600 (one chunk) | 27,253 | 25,057 | -8.1% |
| 32,000 | 116,230 | 104,991 | -9.7% |

- **What the parts are worth.** The lane measured the 32K prompt with D off: 106.5 s, so D is about 0.6-0.8 s of its
  gain and C the rest.
- **(B) The shared expert beside the host's expert grouping** gained 0.6-0.9% at 1-4K, under its 1% rule. It stays
  opt-in (`GUILD_PREFILL_SHEXP_EARLY=1`).
- **(E) The head's fused hyper-connection read by default.** In nsys it shortens the step before the head from 943 to
  186 us at 8 rows. Decode tok/s did not resolve it.
- **E's bug.** Its verifier found that E, as the lane wrote it, changed the outputs of models with one residual
  stream: Ornith's tokens changed, and Drama 3's WAV changed and never reached end of speech. The fused read skipped
  the last layer's residual add when hc = 1. It now applies only when hc > 1, and the integration check has Ornith's
  and Drama 3's outputs identical.

Qwen decode (2 runs x prose + code x 256 tokens, 2 pairs; ids `[38747, 38747, 41916, 41916]` in every run):

| Arm | prose tok/s | code tok/s |
| --- | ---: | ---: |
| before | 49.23 / 49.20 | 62.02 / 61.86 |
| after | 49.55 / 49.41 | 62.56 / 61.98 |

**GLM-5.3-Flash** (`glm-pipe`):
- **Lean IQ4_XS down rows.** One variable shift splits the nibbles, the sign comes from a test mask and the scales
  from one shuffle a block. At 1 / 2 tokens on 32 threads the rows read 190.9 -> 197.8 / 159.5 -> 173.2 GB/s.
- **Layer 11's IQ4_XS gate/up** through the same AVX-512 rows instead of a per-token `vec_dot` loop.
- **The KDA gates in one launch,** and **Q8_1 activations written by the kernels that compute them.**
- **cpu_moe's containers kept between calls.**

Each item has a switch (`GUILD_IQ4XS_LEAN`, `GUILD_IQ4XS_GU`, `GUILD_GLM_FUSE2`, `GUILD_CPU_MOE_REUSE`). The lane's
verifier measured them all together at -1.80 +- 0.30 ms a step in one process. The integration check (5-token
prompt, 256 tokens, `--spec --ctx 32768`, 3 pairs of separate processes):

| Arm | Step p50 | tok/s | CPU experts a step |
| --- | --- | --- | --- |
| before | 78.5 / 80.2 / 79.7 ms | 23.42 / 22.88 / 23.17 | 62.3 / 66.9 / 64.2 ms |
| after | 77.5 / 78.3 / 78.9 | 23.70 / 23.50 / 23.29 | 58.4 / 61.2 / 62.9 |

That is -1.2 ms a step and +1.5% tok/s. With the CPU shorter, the GPU half of a layer is the limit more often (the
lane's KDA router waits grew). Group A's shared expert on a low-priority side stream gained nothing and is opt-in
only (`GUILD_GLM_SHEXP_STREAM`).

**Nemotron 3.5 Transcribe** (`nem-decode`):
- **The Mamba-2 scan's state stored by thread.** A warp's loads were 32 cache lines apart; the coalesced scan is 4x
  faster in isolation.
- **Argmax of the verify rows on the GPU,** instead of a 2 MB pageable logits copy and four host scans a step.
- **A draft head over 44,602 English tokens** of the 131,072 (`data/nemotron_draft_vocab.bin`; the verify head stays
  whole, so only drafts change).

The prefix snapshot (read the prompt's 43 fixed tokens once) is not exact: cpu_moe sums a token's experts in the
order the window's tokens first name them, so a window starting at token 43 rounds differently. It was reverted.

`--mtp 3 --repeat 6` on the perf clips, 2 pairs, medians of runs 1-5, transcripts identical:

| Clip | Before total | After total | Decode before / after | Draft before / after |
| --- | ---: | ---: | --- | --- |
| 5 s | 603 ms | 564 (-6.5%) | 214 / 179 ms | 42 / 24 ms |
| 30 s | 2,235 | 2,046 (-8.5%) | 1,153 / 967 | 229 / 134 |

All 60 gate clips are identical to the gate's greedy reference, greedy and with 1-3 drafts.

**Drama 3** (`d3-prio`):
- **`step_begin` on a top-priority stream.** The tower's first layers, on the critical path between backbone steps,
  get the top priority.
- **The codec paced and held.** The codec's chunk is flushed after `step_begin` instead of before, and kept to 2
  GEMMs in flight while `step_begin` runs.

Two rounds x seeds 11 / 12:

| Arm | frames/s | WAV md5 |
| --- | --- | --- |
| before | 47.33, 47.22, 47.37, 47.18 | `6e04c9b1` / `f84288c9` |
| after | 47.64, 47.79, 47.58, 47.83 | the same |

That is +0.9%, below the expected +2-5%: the codec's cost moves into the next backbone steps rather than going away.
`GUILD_DRAMA3_PRIO=0` restores the old path.

**Ornith-1.5 and StartLux** are unchanged in speed, with the same outputs. Ornith: 79.63 / 79.74 against 79.71 / 79.54
prose, 92.00 / 92.13 against 92.30 / 91.99 code, ids `[30936, 30936, 65886, 65886]`. StartLux: 1,081 against 1,083
ms for a question on ~500 tokens, answers byte-equal.

## Not merged: the CUDA kernel lane (needs a lock bump)

`kcuda-onechunk` changes guild-kernels-cuda (branch `lane18/kcuda-onechunk`, `11936b5` on `de65d60`):
- **One-chunk Q4_K rows.** A 2560-wide Q4_K row (10 blocks) ran as a full 8-block chunk plus a 2-block chunk that
  left three of its four warp steps idle. Rows of 8-16 blocks now run as one chunk.
- **One-chunk Q6_K rows** of exactly 8 blocks.
- **4-column Q4_K calls** run as two 2-column calls for rows of 9-12 blocks.

Same arithmetic in the same order: 0 of 2,119,284 parity outputs differ, and every column equals its 1-column call.
The verifier's A/B (same binary, `GUILD_MMVQ_ONE_WIDE=0` as the old path):

| Model | prose tok/s | code tok/s | ids |
| --- | --- | --- | --- |
| Qwen3.8 | 49.12 -> 51.08 (+4.0%) | 61.80 -> 64.18 (+3.8%) | `[38747, 38747, 41916, 41916]` |
| Ornith-1.5 | 80.23 -> 82.95 (+3.4%) | 93.05 -> 97.05 (+4.3%) | `[30936, 30936, 65886, 65886]` |

The 4-column split alone is +0.5% on Qwen's code windows. Drama 3's WAVs and GLM's ids are unchanged. Shipping it
means moving `backends.lock.json`'s CUDA revision to `11936b5`, which is the user's decision; its Guild-side tests
are on `lane18/kcuda-onechunk` too.

**Since 2026-11-15** the lock pins `11936b5` (pushed to guild-kernels-cuda as `perf/onechunk-mmvq`), with the four
Guild-side tests. The check on a fresh build from the pins is in
[2026-11-15-parallel](../2026-11-15-parallel/README.md#the-cuda-kernels-at-11936b5).

## Measured and not taken

- **Ideas the judge dropped,** with reasons, in `lanes/judge_summary.json`:
  - a one-graph DFlash draft, the router chain, PLE once a window;
  - the window-size rule (a loss again);
  - pre-queued tower layers (under the noise);
  - StartLux's last token in the batched prefill.
- **Numerics-changing options for the user** (not done), listed there too:
  - the drafter's Q4_K arithmetic for the verifier;
  - the verifier for a prompt's small last chunk;
  - GLM's `--dense q5_k`.

## Files

`validate.sh` and `validate.progress` (the integration check), `ctest.log`, `lanes/` (the brief, the judge's
summary, every lane's NOTES.md and VERIFY.md).
