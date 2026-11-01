# Nemotron 3.5 Transcribe: a speed pass on the GTX 1070 + dual Xeon

The same transcripts, faster: on a 29.6 s LibriSpeech clip with three MTP drafts, 3.56 s -> 2.28 s end to end
(real-time factor 0.120 -> 0.077); greedy 4.05 s -> 2.99 s; a 5 s clip 0.89 s -> 0.62 s.  Every change keeps the
arithmetic of every token: the 60 gate clips give the committed engine's token ids (`5ce7143`), greedy and with 1, 2
and 3 drafts, and every drafted row's logits are greedy's to the bit.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 without VNNI, 2 NUMA nodes), GTX 1070 8 GB, CUDA 12; the Q8_0 GGUF
(experts in RAM, split over both nodes).  Commits `ce3a168`, `37ab5fd`, `cc46ac9`, `d9e7472` on
`feat/nemotron-transcribe`.

## Where the time went

`5ce7143`, the 29.6 s clip (421 prompt tokens: the request's and one row per 80 ms of audio):
- **The prompt, 1.88 s**, of which the CPU experts took 1.57 s (`GUILD_MOE_TIMING`) and the GPU the rest, one after
  the other.  The experts ran at 344 GMAC/s: guild-kernels-cpu's AVX-2 Q8_0 kernel (`kq256_rows`), one thread
  16-17 GMAC/s, its accumulators kept in memory between blocks.  Not the memory: the two 256-row windows read the
  experts at ~40 GB/s.
- **Decode, 20 ms a token greedy:** the CPU experts (1.46 GB a token) ~9 ms and the GPU (~2 GB of dense rows and the
  head) ~11 ms, serial.  A verify window of 4 tokens took 43.6 ms: the CPU reads each token's six experts (the
  `single` rule keeps one-token arithmetic), ~1.2 GB a token, at 148 GB/s; the GPU waited for it.
- **The audio model, 167 ms**, of which ~35 ms widened its 1.4 GB of BF16 matrices to float32 before each product.

## What changed

**1. Q8_0 expert rows on AVX-512 (`ce3a168`, `cpu_moe`, shared with GLM).**  `q8_0_rows512` takes two rows a
register (row r in the low 256 bits, r + 1 in the high) against 1-16 tokens, every lane computed and summed as
ggml-cpu's AVX-2 `ggml_vec_dot_q8_0_q8_0` does, so each (row, token) is ggml's result to the bit
(`q8_0_rows512_test`: 120,768 results, 1-16 tokens, random bytes including -128, none differ).  Because it is
exact per token, the expert phases use it in `single` windows too.  `q8_0_rows512_test --bench` / `--dram`
(`dram.sh`):

| | 1 token | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `kq256_rows` (AVX-2), one thread, GMAC/s | 13.0 | 15.5 | 17.1 | 17.8 | 16.1 |
| `q8_0_rows512`, one thread, GMAC/s | 17.2 | 26.8 | 34.0 | 42.7 | 48.1 |
| `q8_0_rows512`, 32 threads each streaming 128 MB of its own rows, GB/s, no prefetch | 187 | 163 | | | |
| the same, asking for the next two rows while two are read | 215 | 212 | | | |

What it took in the engine (the 29.6 s clip's prompt, `prof1.sh` / `phases.sh`, ms a run):

| Step | Prompt | Its CPU experts |
| --- | ---: | ---: |
| `5ce7143` | 1878 | 1571 |
| the kernel, activations packed inside each call | 1385 | 1077 |
| each group's activations packed once a phase, in even chunks of up to 16 tokens (the packing was 10% of the cycles) | 1236 | 925 |
| + window pairs (item 2) | 988 | 960 |
| + the prefetch, then the next group's rows at the end of a down item | 960 | 920 |
| + a prompt window's activations quantized across the pool; the phases' buffers only grow (`resize()` zero-filled megabytes on the calling thread after every smaller window) | 909 | 874 |

The drafted decode's CPU experts: 148 GB/s -> 169 with the prefetch -> 185 with the next group's rows asked for
(the down items are 16 rows of every group, so each group's first rows were cold).

**2. Two windows at a time (`37ab5fd`).**  A prompt over 256 rows runs its windows in pairs, and a verify window of
2-4 tokens runs as two halves (the first the larger): at each MoE layer the calling thread runs one window's CPU
experts while the launcher thread enqueues the other's layers up to its next MoE layer.  The stream keeps every
layer's order, so each window computes what it computes alone: the verify halves write the whole window's Mamba
snapshots (the scan and the convolution now take a snapshot count) and a verify row stays a decode step's.
`GUILD_SPEC_SPLIT=0` is the one-window control.  The 29.6 s clip, `absplitk.sh` (3 runs + a warm-up, alternated
twice; before the launcher, which then took verify(4) 37.4 -> 36.2 ms):

| Drafts | Verify window, one / split, ms | Its CPU experts, one / split, ms | Decode, one / split, ms |
| ---: | ---: | ---: | ---: |
| 1 | 28.6 / 28.2 | 14.5 / 17.5 | 1570 / 1552 |
| 2 | 35.4 / 33.3 | 19.5 / 23.8 | 1384 / 1314 |
| 3 | 43.1 / 37.4 | 24.7 / 28.4 | 1366 / 1218 |

The halves read the experts the two share twice (+15-20% CPU time), and pay it back where the GPU part is larger
than that: from 3 tokens.  Window pairs: the 10 gate clips over 256 rows, prompt 10.1 s -> 8.7 s.

**3. The audio model keeps its float32 matrices (`cc46ac9`)** when 512 MiB of the GPU stay free after the language
model loads (2.9 GB more: the audio model 4.1 GB and the language model 2.1 GB on this 8 GB card).  The same SGEMMs on the same values (`audio_encode --keep-wide`:
every stage of 6 clips the same bytes); 59 -> 35 ms for 5 s of audio, 176 -> 154 ms for 29.6 s.

**4. `transcribe_run` reports** the draft, verify and accept milliseconds and the CPU experts' share (`d9e7472`);
`GUILD_MOE_TIMING` adds the expert bytes read and the CPU's waits for the GPU.

## The transcripts

- `gatecheck.sh`: the 60 gate clips greedy and with 1, 2, 3 drafts, every clip's ids against the committed engine's
  greedy run: 60 of 60 in each mode, on the final binary (2026-11-01 04:00 UTC).  So the gate result is unchanged
  (50 of 60 identical to the fp32 reference, the WER 0.07 points from its).
- `speccheck.sh` (`TRANSCRIBE_CHECK_SPEC`): with 1-3 drafts on both speed clips, every verified row's logits equal
  greedy's at that position bit for bit.

## Speed

`run_perf.sh`: `transcribe_run --repeat 6` on the README's two clips, the first run of each process dropped, mean
and standard deviation of 5; 2026-11-01 04:04 UTC, the machine quiet.  Decode tokens/s counts the tokens after the
first.

| Clip | Mode | End to end, `5ce7143` -> now | Real-time factor | Audio model | Prompt | Decode | Decode tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.0 s, 21 tokens | greedy | 1001 -> 784 ms (sd 4) | 0.157 | 34 ms | 534 -> 336 ms | 400 -> 406 ms | 49.3 |
| | 1 draft | 950 -> 702 ms (sd 8) | 0.140 | 34 ms | 563 -> 349 ms | 319 -> 311 ms | 64.3 |
| | 2 drafts | 920 -> 663 ms (sd 11) | 0.133 | 34 ms | 555 -> 350 ms | 297 -> 271 ms | 74.0 |
| | 3 drafts | 892 -> 624 ms (sd 20) | 0.125 | 34 ms | 560 -> 351 ms | 263 -> 231 ms | 87.1 |
| 29.6 s, 96 tokens | greedy | 4050 -> 2990 ms (sd 7) | 0.101 | 148 ms | 1884 -> 855 ms | 1972 -> 1965 ms | 48.3 |
| | 1 draft | 3750 -> 2637 ms (sd 9) | 0.089 | 149 ms | 1968 -> 911 ms | 1590 -> 1555 ms | 61.1 |
| | 2 drafts | 3610 -> 2385 ms (sd 15) | 0.081 | 148 ms | 1979 -> 913 ms | 1429 -> 1302 ms | 73.0 |
| | 3 drafts | 3562 -> 2282 ms (sd 25) | 0.077 | 149 ms | 1965 -> 913 ms | 1401 -> 1196 ms | 79.5 |

A drafted step on the 29.6 s clip now (ms): verify 28.3 / 32.9 / 36.6 for 2 / 3 / 4 tokens, of which the CPU
experts 17.7 / 23.7 / 29.0; a draft 2.6; the accept 0.8-1.5.  Greedy decode did not move (20 ms a token: the
CPU experts 8.8 ms, the rest the GPU's layers before and after them).

## Nothing else moved

`cpu_moe` is GLM-5.3-Flash's CPU expert code too (its CPU-side dense FFNs and shared-expert twins are Q8_0, so they
now take the new kernel).  The branch at `d9e7472` against `build-cuda12` (`feat/glm5` `196527e`), one machine lock,
`bench/results/2026-10-31-startlux-decision/regress.sh`, 2026-11-01 04:05-05:11 UTC:
- **ctest:** 128 tests, the same 3 failures as before: `ple_parity` (its fixtures are not on this machine) and the
  two gates (`decision_gate_test`, `transcribe_gate_test`), failing with their results unchanged.
- **Qwen3.8-Flash-Next:** qwen-bench, server flags, base / branch / branch / base, tok/s:
  - prose: 34.03 / 34.17 / 34.28 / 34.21;
  - code: 44.50 / 44.59 / 44.84 / 44.78.
  - All texts identical.
- **GLM-5.3-Flash:** `glm5_run --spec`, base / branch / branch / base / branch / base, median step 84.6 / 84.4 /
  84.7 / 84.5 / 84.5 / 84.1 ms (its CPU experts 69.3 / 69.0 / 69.7 / 69.3 / 69.3 / 68.0 ms); the same ids in all six,
  and in `guild-generate`'s GLM path.  `session_score_test` on GLM: the same two checkpoint failures as on the
  branch before this work, with the same numbers.

## What is left

- **The CPU experts' bytes** bound the drafted decode: ~1.2 GB a token at 185 GB/s against the ~215 GB/s every
  core streaming reaches; the rest is phase starts and ends (three a layer) and the CPU waiting for the other
  window's GPU layers (~3.6 ms a step, `GUILD_MOE_TIMING`).  Fewer bytes means experts on the GPU (other arithmetic: the transcripts would
  move) or a smaller model.
- **The draft head** reads all 131,072 rows of the Q8_0 output matrix (374 MB, 1.9 of a draft's 2.6 ms).  A token
  subset or a 4-bit copy for drafting would change which drafts are made, not the output.
- **The accept's MTP rows** could join the next draft's first step (one MTP pass instead of two, ~1 ms a step).
- **A prompt of one window** (most clips under 20 s) cannot overlap its GPU and CPU parts this way: splitting it
  would read every expert twice.
- Hyper-threads: 64 copies of the kernel bench run 8-9% more GMAC/s than 32 (8-16 tokens), but the launcher sits on
  the calling thread's sibling; not tried in the engine.

Files: `prof1.sh` (the prompt alone, 5 runs), `phases.sh` (the expert phases of the prompt and of a drafted run),
`absplitk.sh` (split against one window), `dram.sh` (the all-core kernel bench), `gatecheck.sh`, `speccheck.sh`,
`run_perf.sh`, `perf_{greedy,mtp1,mtp2,mtp3}.jsonl` (every run).
