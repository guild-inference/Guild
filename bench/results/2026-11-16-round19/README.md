# Round 19: where decode time goes on this PC, a faster IQ4_NL down kernel, and a split switch

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes, 376 GiB RAM), GTX 1070 8 GB (sm_61,
15 SMs, 151 W cap, PCIe 3.0 x16), CUDA 12. Branch `perf/r19` (from `perf/parallel` 138cc93). Kernels are at the
pinned revisions (guild-kernels-cpu `9b7281a`, guild-kernels-cuda `11936b5`). The base binary is `perf/parallel`
138cc93 built from the pins.

Benches:
- **Qwen.** `qwen-bench-r18`: Qwen3.8-Flash-Next UD-IQ4_XS with setup's flags (DFlash, `--spec 8`, the dense map,
  `--hc q8_0`), 32K context, prose and code, 2 runs x 256 tokens.
- **Ornith.** `ornith-bench-r18`: Ornith-1.5-35B Q4_K_M, its MTP layer, `--spec 4`.
- **Slots.** `tools/batch_test.py` for the batch slots.

Every A/B below kept the bench's ids (Qwen `[38747, 38747, 41916, 41916]`, Ornith `[30936, 30936, 65886, 65886]`).
Every slot run kept each slot's solo tokens.

## What limits decode here: the GPU

**Qwen, one request.** An nsys capture (CUDA, graph nodes) of the trace's later windows, mostly the code request
(`runs/nsys-qwen-window.txt`). The
tracing stretches a window from about 60 to 72 ms, so read the times as shares, not absolute values. GPU time per
window by kind:

| Kind | ms a window |
| --- | ---: |
| dense products (MMVQ) | 33.7 |
| hyper-connection reads | 13.7 |
| GDN (conv, alpha/beta, step) | 7.4 |
| DFlash draft head (`k4_mm`) | 3.6 |
| BF16 GEMVs (router, indexer) | 3.3 |
| routing, doorbells, copies | 2.9 |
| attention | 2.4 |
| other | 2.2 |
| **GPU waiting for the CPU's experts** | **2.1** |
| GPU busy (union of streams) | 59.0 of 72.1 |

The GPU waits for the CPU's experts 2.1 ms a window. The CPU's experts take 31-38 ms a window, and nearly all of
that runs beside GPU work.

Shares of the 256 GB/s card that the products reach (durations from the trace):
- **Large Q4_K matrices.** 180-190 GB/s at one column, ~177 at two, ~150 at three.
- **Q5_K `ssm_out`.** 171, 149, 134 and 116 GB/s at 1-4 columns.
- **The shared expert's Q8_0 matrices.** 1.7 MB each, at ~74 GB/s.
- **The hyper-connection read.** Norm, down and up, 7.0 MB a layer and group. `fused_gr_bench` without tracing gives
  47.7 / 51.6 / 57.2 / 71.6 us for 1-4 tokens, that is 146 down to 97 GB/s.

A speculative window reads all of these twice, once per row group. That is the price of running one group's CPU
experts beside the other group's GPU work.

**One group instead of two** (`GUILD_SPEC_SPLIT=0`, base binary):

| | prose tok/s | code tok/s | prose ms a window | CPU experts ms a window |
| --- | ---: | ---: | ---: | ---: |
| two groups (default) | 51.70 | 65.16 | 54.4 | 30.8 |
| one group | 40.00 | 44.70 | 70.3 | 26.8 |

Reading the weights once does not repay putting the CPU's experts back on the critical path.

**Ornith.** One group costs 12-17% (prose 82.8 / 82.5 -> 72.6 / 72.3, code 96.8 / 96.7 -> 80.7 / 80.2 tok/s). Its
window is 25.2 ms with 13.3 ms of CPU experts and 3.7 ms of MTP draft. In a split window the GPU is the bound here
too.

**Qwen's batch slots.** They are GPU-bound as well: see below, where faster CPU experts turn into CPU wait.

## Change 1: IQ4_NL down rows with the scales out of the token loop (e9e1996)

Qwen UD-IQ4_XS's routed experts run on the CPU on this card. Their gate/up rows (IQ3_S, repacked to NL4 at load) are
memory-bound: 200 / 185 GB/s at 1 / 2 tokens on 32 cores (`glm5_nl4_test --dram`). The IQ4_NL down rows were not.
On 32 cores they ran at 145 / 89 / 70 / 56 GB/s for 1-4 tokens. They are 35% of the expert bytes but 46% of the
pool's thread time (a `GUILD_CPU_PROFILE` of batch windows).

What cost the time: for every row pair, `iq4nl_rows512` converted each token's f16 scale again and built
d_y * d_x with two scalar multiplies, two broadcasts and an insert per block and token. A scratch microbenchmark
(`runs/iq4nl_variants_bench.cpp`, the kernel inlined as in the engine, 32 threads) took parts out one at a time:
- **No scale arithmetic** (timing only): +25 / +61 / +74% at 1 / 2 / 4 tokens.
- **No nibble decode** (timing only): +6 / +31 / +26%.

v2 keeps ggml's arithmetic and moves the scale work out of the token loop:
- Each token's block scales are converted once a call, into a table of floats.
- A block pair's four weight scales are converted by one `vcvtph2ps` and spread into the two blocks' scale vectors
  by two permutes. A token's d_y * d_x is then one multiply by its scale, broadcast from memory. That is ggml's
  single-precision product: the conversion is exact and the multiply rounds once.
- The high nibbles come from one 64-bit variable shift.
- For 2+ tokens, |w| comes from a shuffle of the codebook's magnitudes and the negative lanes from the code's bit 3.
  That takes an instruction off port 0, where the token loop's multiplies run.

The integer sums and the float order are unchanged. `iq4nl_rows512_test`, now with 640-value rows too, finds 0 of
20,448 results different from ggml-cpu's dot. `GUILD_IQ4NL_V2=0` restores v1.

| 640-value rows, GB/s (`runs/cpu-kernels.txt`) | 1 token | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: |
| 32 threads, v1 | 145.5 | 89.2 | 69.4 | 56.4 |
| 32 threads, v2 | 169.7 | 136.8 | 111.0 | 93.4 |
| 8 threads, v1 | 39.7 | 24.1 | 19.1 | 15.4 |
| 8 threads, v2 | 47.5 | 37.3 | 30.5 | 26.0 |

In the engine (two alternated pairs, `runs/q-*.summary`):
- **Qwen.** CPU experts fell from 31.1 to 29.1 ms a prose window and from 37.7 to 34.8 a code window. Speed went
  from 51.58 / 51.08 to 52.01 / 51.74 tok/s prose (+0.8 / +1.3%) and from 64.63 / 64.62 to 65.49 / 65.05 code
  (+1.3 / +0.7%). The window is GPU-bound, so most of the CPU time saved becomes slack.
- **Batch slots** (`runs/ab1-*.out`). 6 slots: CPU experts 58.5 -> 54.1 ms a window, GPU-reach wait 9.7 -> 14.0 ms,
  the window 69.6 -> 69.5 ms (70.2 -> 70.5 tok/s). 3 slots with drafts: CPU experts 66.1 -> 60.2 ms, wait
  10.9 -> 16.6 ms, the window 96.3 -> 95.6 ms (58.6 -> 59.1 tok/s).

The gain is for PCs whose CPU experts are the bound. On 8 of this PC's cores, v2 does 1.55-1.69x v1's work at 2-4
tokens and 1.2x at one. Here the GPU hides most of it.

## Change 2: `GUILD_SPLIT_MIN_T`, the fewest rows a speculative window splits at (99081ae)

A window splits from 2 rows on. Grouping does not change a row's bits, so the threshold is free to move. Same binary,
two runs each:

| Threshold | Qwen prose | Qwen code | Ornith prose | Ornith code |
| --- | ---: | ---: | ---: | ---: |
| 2 (default) | 51.85 / 51.80 | 65.05 / 65.13 | 82.83 / 82.52 | 96.81 / 96.67 |
| 3 | 52.18 / 52.30 | 65.45 / 65.52 | 81.57 / 80.88 | 95.83 / 95.51 |
| 4 | 51.28 / 51.38 | 64.74 / 64.74 | - | - |

At 3, Qwen gains 0.6-0.7% and Ornith loses 1-2%. The default stays 2. Qwen on a GPU-bound PC can set 3.

## Measured and not taken

- **The hyper-connection down kernel's shape** (guild-kernels-cuda; the diff is `runs/gr_down_q8_shapes.patch`).
  Rows per warp and warps per block became template parameters. A variant held each token's activations in registers
  for all of a warp's rows. Every lane's chunks keep their order, so the bits are the same (`gr_parity` OK).
  - `fused_gr_bench`, 1-4 tokens: 2x4 (today) 47.7 / 51.6 / 57.2 / 71.6 us, 2x4 in registers 45.5 / 50.5 / 56.1 /
    67.6, 1x8 45.4 / 50.3 / 57.4 / 67.1.
  - In the engine, two pairs: 2x4 in registers 51.30 / 51.75 prose, 64.77 / 64.88 code, against 51.65 / 51.74 and
    65.22 / 65.36. No gain, so it was not committed (taking it would have needed a CUDA lock bump).
- **Drafts for more than 3 decoding slots** (`GUILD_BATCH_DRAFT_SLOTS`, `runs/bdraft-*.out`). 6 slots: 77.3 / 77.4
  tok/s with the default (no drafts), 68.3 / 68.2 with drafts. 4 slots: 64.9 / 65.1 against 64.2 / 63.8. Each
  drafting slot runs its own DFlash pass, 13-16 ms of drafts a window in all, more than the extra tokens repay. The
  default stays 3.
- **v2's first form.** It staged a row pair's weight scales through memory: 2-byte stores, then 32-byte loads, which
  cannot be store-forwarded. It ran slower than v1: 67 against 144 GB/s at one token.
- **Pre-broadcast activations** and **dropping the prefetch**, in the microbenchmark: no gain, or a loss.

## Drama 3's CPU experts

Drama 3 decodes one backbone row a step, so its experts cannot hide behind a second group. Earlier rounds measured
5.2 ms a frame of waiting on them. One layer's 8 experts for one token through the pool (`cpu_expert_workload`, real
GGUF rows, 32 cores, `runs/cew-*.csv`):
- **Q4_K gate/up with Q6_K down:** 110 us.
- **Q4_K gate/up with Q4_K down:** 112 us.

Each layer reads 13.5-15.6 MiB, which would take 64-74 us at the machine's ~220 GB/s. These 8 experts fit in the
22 MiB L3 across iterations, so the engine, which reads them from DRAM, is slower than this. The kernels are
guild-kernels-cpu's AVX-512 Q4_K / Q6_K rows, which spread their scale work over 256 values. The rest is likely the
pool's fixed cost per phase at such small sizes, which was not measured. Not pursued this round.

## Open levers (estimates, not measured)

- **Batched DFlash drafting across slots.** The drafter's dense layers would be read once for every drafting slot.
  The Q4_K head is compute-bound, so drafts for 4-6 slots might come out ~5% ahead.
- **A two-stage DFlash head.** Draft logits for the first 4 positions, then the rest only while the window rule
  still accepts. The same drafts. Maybe ~1 ms a prose window, nothing for code.
- **The shared expert's gate and up in one launch.** About 1% of a window. Needs a CUDA lock bump.
- **The Q8_0 expert rows** (Qwen's Q8_0 down layers, ~12% of the pool's time). Build the weight-scale vector the way
  IQ4_NL v2 does. Needs a CPU lock bump.
- **The hyper-connection read.** 13.7 ms a window at ~100 GB/s. A real gain needs a new kernel design under the same
  bit-exact order, not a new shape.
