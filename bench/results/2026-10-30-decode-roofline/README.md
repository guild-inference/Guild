# Decode on the GTX 1070 + dual Xeon against its memory roofline, and what two rounds of changes moved

Measured 2026-10-30 and 2026-10-31 (UTC) on this PC: Ubuntu (kernel 6.8), two Xeon Gold 6130 (2 NUMA nodes,
32 cores / 64 threads, AVX-512 F/BW/DQ/VL without VNNI or VBMI), 376 GiB RAM, GTX 1070 8 GiB (sm_61, 15 SMs, 2 MB
L2), driver 580.178.04, CUDA 12.0.140, GCC 13.3.0, `kernel.numa_balancing` = 1. Release build of `guild-generate`
and `glm5_run` (`-DGUILD_EXPERIMENTAL_SM60=ON`, `CMAKE_CUDA_ARCHITECTURES=61`) with `GUILD_CUDA_SOURCE_DIR` on a
`guild-kernels-cuda` checkout of branch `perf/sm61-decode` and, from round 2, `GUILD_CPU_SOURCE_DIR` on
`guild-kernels-cpu` `5147f69`. Models: Qwen3.8-Flash-Next UD-IQ4_XS with its MTP draft layer, and GLM-5.3-Flash
UD-IQ4_XS with its NextN layer. Ceilings measured on this PC in earlier reports: GPU DRAM 210-222 GB/s, CPU DRAM
214-226 GB/s node-local and 125 GB/s interleaved, PCIe 3.0 x16 11.2 GB/s by DMA and 8.7 GB/s for kernels that read
mapped host memory.

| Binary | Guild `feat/glm5` | guild-kernels-cuda `perf/sm61-decode` | guild-kernels-cpu |
| --- | --- | --- | --- |
| pre-round-1 | `54bfc99` | `515e3d1` | `629817c` |
| round 1 | `ab02a18` | `5641a3d` | `629817c` |
| round 2, batch A ("round2a") | `f5a4c92` | `1d7cf15` | `5147f69` |
| final | `30676b5` | `b511c94` | `5147f69` |

`perf/sm61-decode` is local, not pushed, and `backends.lock.json` still pins guild-kernels-cuda `fe4c080`: a build
from the lock has none of the CUDA kernel changes below (nor those of
[2026-10-27-staged-mmvq](../2026-10-27-staged-mmvq/README.md)). The lock pins guild-kernels-cpu `5147f69` since
`fe3bd34`.

How it was measured:
- **Qwen:** `tools/bench_runtime.py` with the server's flags (`--spec 4 --spec-min-p 0.5 --expert-cache auto --kv
  fp16 --kv-host-only --prefill 512 --resident-budget-gib 56`, the expert profile, MTP), its prose and code prompts,
  3 runs x 256 tokens, 32768-token context unless said. A window is one verify graph of 1-4 tokens and the draft
  steps before it; ms a window, tokens a window, CPU dispatch and draft ms come from `GUILD_DECODE_TIMING`. Every run
  logged "GPU cache 0 slots" (`e9ee500`), so every routed expert ran on the CPU and texts compare across runs.
- **GLM:** `glm5_run <shard 1> 785,6722,315,9621,374 256 --ctx 32768 --spec` (the prompt "The capital of France is")
  with `GUILD_GLM_TIMING=1 GLM5_RUN_STEPS=1`. A step is a draft, a 2-token verify window and an accept; 117-118 of
  138 drafts are kept, 1.85 tokens a step.
- **Speed** comes from alternated A/Bs, the same binary with an environment switch where one exists, at least 2
  pairs for Qwen and 3 for GLM (GLM's CPU expert time moves ~3 ms a step between processes). Each change was
  measured on its own branch, re-measured by a second, independent run of its A/B and its bitwise tests, and then
  measured again after merging. Where those disagree, both are given.
- **Labels:** S1 and S1G are the baseline profiles of Qwen and GLM taken first; Q1-Q8 are the Qwen changes, G1-G4
  the GLM ones, C1-C2 the correctness work. Round 1 is Q1-Q4; round 2 is the rest, merged as batch A (G1, G2, C1, Q5,
  Q6) and batch B (Q7, Q8, C2, G3, G4). int, int2 and int3 are the measurements taken after merging round 1, batch A
  and batch B.

## Headline

| Model, context | pre-round-1 | round 1 | final | pre-round-1 -> final | Measured as |
| --- | --- | --- | --- | --- | --- |
| Qwen, prose prompt | 69.81 ms a window, 30.57 tok/s | 65.52, 32.56 | **63.33, 34.26** | -6.48 ms, +12.1% | final vs round 1: 3 alternated pairs; pre-round-1: 2 runs the same night, not alternated |
| Qwen, code prompt | 73.65, 41.38 | 68.59, 44.44 | **66.45, 44.80** | -7.20 ms, +8.3% | the same |
| Qwen, greedy (`--spec 0`), prose / code | 22.74 / 22.31 tok/s | 23.92 / 23.91 | **24.47 / 24.45** | +7.6% / +9.6% | one run each; pre-round-1 from S1, another session |
| Qwen, 31,918-token prompt | 78.26 ms, 32.07 tok/s | 72.93, 34.42 | **69.51, 34.42** | -8.75 ms, +7.3% | final vs round 1: 2 alternated pairs; pre-round-1 from S1 |
| Qwen, 127,898-token prompt | 80.60 ms, 29.96 tok/s (first request after the prefill: 133.15 ms, 19.04 tok/s) | - | **72.04, 34.15** | -8.56 ms, +14% | one process each, other days |
| GLM, 5-token prompt | as round 1 (no GLM code changed) | 21.36 tok/s, step p50 85.3 ms | **21.80, 84.6** | +2.1% | 3 alternated pairs |
| GLM, `token_embd` evicted from the page cache | as round 1 | 19.57 | **21.63** | +10.5% | 2 alternated pairs |
| GLM, 15,999-token prompt | as round 1 | 18.45 | **18.96** | +2.8% | one pair |
| GLM, 32,000-token prompt | does not fit (16,640 cells) | does not fit | **18.23** | new | one process |

Qwen's speculative text changed on purpose in round 2 (it now equals greedy decode, section "Correctness"), so its
tokens a window moved (prose 2.13 -> 2.17, code 3.05 -> 2.98, 32K 2.51 -> 2.39): ms a window is the fair measure,
and at 32K the shorter windows carry fewer tokens each, so tok/s did not move against round 1.

## 1. The roofline

### Qwen at short context

What one row group (one pass over the 48 layers for a group of 1-2 tokens) reads: 4.62 GB. Q8_0 products 2.841
(attn_qkv 1.003, attn_gate 0.602, ssm_out 0.602, attn_q 0.401, attn_output 0.201, attn_k/v 0.033), hyper-connection
BF16 1.278, shared expert 0.251, router BF16 0.126, PLE 0.066, indexer projections 0.039, GDN a/b 0.018. A window of
2-4 tokens runs two row groups so the CPU's experts for one overlap the GPU's work for the other, and so reads these
twice. With S1's window mix (prose 1.86 row groups and 2.64 draft steps a window, code 1.98 and 2.84) a window reads
10.47 GB (prose) and 11.10 GB (code) on the GPU: 49.8 and 52.9 ms at 210 GB/s.

"Before" is S1's nsys capture of the pre-round-1 binary (code prompt, prose in brackets; kernel time a window, the
nsys launch inflation removed for the waits). "Change" is what each change measured inside the engine, by nsys
unless said. No nsys capture of the final binary was taken; its window is measured without a profiler.

| Component | GB a window | Bound at 210 GB/s | Before | Of bound | Change, measured in the engine |
| --- | --- | ---: | --- | ---: | --- |
| Dense Q8_0 products (read by both row groups) | 5.61 (5.28) | 26.7 ms (25.1) | 28.91 ms (27.08) | 93% | none. Q3's fork overlaps them with the shared expert: their total rose 40.2 -> 42.1 ms a two-group window under nsys |
| Hyper-connection read (BF16, 13.3 MB a read, 2 reads a layer and row group) | 2.53 (2.37) | 12.0 (11.3) | 15.89 (14.81) | 76% | Q7: a read 78.9 -> 74.1 us at 1 token, 87.0 -> 81.4 us at 2, against 63.4 us at the bound (73% -> 78% at 2 tokens) |
| Draft: MTP layer + 106K-row Q6_K head, ~2.7 steps | 1.00 (0.93) | 4.8 (4.4) | 6.46 (6.01) | 74% | host timer, round 1 -> final: 7.13 -> 7.00 ms (6.67 -> 6.46); its head alone was already at 209 GB/s |
| Shared expert (side stream) | 0.50 (0.47) | 2.4 (2.2) | 5.58 (5.22), 1.9 (1.7) of it not overlapped | 42% | Q3: forked in the other group's post, so the top-k and the doorbell run alone |
| GDN step + commit (113 MB of state) | 0.45 (0.44) | 2.1 | 5.31 (4.93) | 41% | Q6: recurrence 38.6 -> 26.9 us a layer (2-token group), 26.7 -> 21.8 (1 token); commit graph 2.26 -> 1.75 ms a window |
| Verify head, Q6_K 521 MB at T columns | 0.52 | 2.5 | 4.42 (4.07) | 58% | Q1: a call at T = 2 / 3 / 4: 3.13 / 3.59 / 4.73 -> 2.44 / 2.66 / 3.00 ms (T = 4: 110 -> 174 GB/s) |
| Router BF16 GEMV, 2.62 MB | 0.25 (0.23) | 1.2 (1.1) | 2.00 (1.83) | 60% | Q3: 22.8 -> 16.7 us a group and layer at 2 columns (115 -> 157 GB/s) |
| Top-10 + doorbell (latency, no bytes to speak of) | - | - | 1.95 (1.83) | - | Q3: 12.6 + 7.8 -> 8.6 + 6.8 us a group and layer |
| QSA attention, indexer, PLE, zero-copy reads, combine, KV resolve, small | ~0.24 | ~1.1 | 4.13 (3.76) | - | none measurable at short context |
| GPU waiting for the CPU experts | - | 0 | 4.3 (5.6): window minus busy, no profiler | - | Q4's lane: 1.96 -> 0.34 ms a window by a host-side probe (another method, 575 windows; with its unmerged CPU-kernel half too) |
| **Window** | **11.10 (10.47)** | **52.9 (49.8)** | **75.2 (71.65)** | **70% (70%)** | **final 66.45 (63.33): 80% (79%) of S1's bound** |

The final's bound differs a little from S1's because its text, and with it the window mix, changed (above).

The CPU side: the routed experts read 3.62 GB a window on average (4.21 GB in a four-token window), 16.0 ms at
226 GB/s, against 42.8 ms of batch wall time (85 GB/s, 38% of the bound; S1's `GUILD_CPU_PROFILE`). They run behind
the GPU, so only the waits show. Round 1, whose only CPU-side change is Q4's, cut the host's CPU-expert time from
30.1-31.4 / 33.7-35.2 to 26.6-27.1 / 29.7-29.8 ms a window (prose / code, round 1's pairs), and the waits with it.
PCIe at short context: about 1 MB a token (activations out, expert sums in), and no KV misses (0.0-0.07 MiB a
window).

### Qwen at long context (`--max-context 262144`, the server's setting)

The KV staging pools are per layer at this setting (959.5 MiB of session and a 1,274 MiB reserve fit the 2,970 MiB
free at load), so only misses cross PCIe. The selection is capped at 2,051 cells a token, so attention reads ~50 MB
a token at any length; what grows is the indexer's pooled keys. Before: S1's one-row-group profile
(`GUILD_VERIFY_PROFILE=1 GUILD_SPEC_SPLIT=0`), ms a window.

| Item | 32K | 128K | Bound | Before | Change |
| --- | --- | --- | --- | --- | --- |
| Pooled indexer keys, 128 x L bytes a layer and pass | 49 MB | 196 MB | 0.23 / 0.93 ms | scores + top-k 1.13 / 3.17 ms | Q5's top-k: 0.89 -> 0.82 ms at 32K, 2.15 -> 1.63 ms at 128K (greedy) |
| Selected K/V for attention, 2,051 cells x 2 KiB x 12 layers | 50 MB a token | 50 MB a token | ~0.6 ms | 1.98 / 1.95 ms | none |
| KV misses over PCIe (93-97% of lookups hit) | 3.5-4.2 MiB | 10.5-11.6 MiB | 0.33-0.39 / 0.98-1.09 ms at 11.2 GB/s | kv-resolve 0.57-0.62 / 1.23-1.34 ms | none |
| Draft layer's attention over its 32K-cell window | - | - | - | draft 9.01 ms a window at 32K against 7.2 at short context | Q8: 8.94 -> 8.29 ms; round 1 -> final 8.94 -> 8.07 |

A shared staging pool, reset at every resolve, would re-read most of the selected blocks: forced with
`GUILD_KV_SHARE_STAGING=1` at 32K it moved 114.5 MiB a window over PCIe and cost +7 to +9 ms a window (S1).

### GLM per step

Bytes from the GGUF tensor sizes and the engine's formats and placement (attention and DSA q_b re-quantized to
Q6_K, routed gate/up as NL4); bounds 210 GB/s GPU, 226 GB/s CPU. "Round 1" is S1G's profile of the round-1 binary
and int3's three round-1 runs; round 1 changed no GLM code path, so it is also the pre-round-1 state.

| Side | Item | GB a step | Bound | Round 1 | Final |
| --- | --- | ---: | ---: | --- | --- |
| GPU | verify window: two 1-token groups, each reading every attention matrix; mHC, routers, shared experts, 2-column head | 12.88 | 61.3 ms | 75.65 ms busy (nsys, 81%); verify 80.3 ms by the host timer | verify 78.1 ms by the host timer (not profiled) |
| GPU | draft (NextN layer + 154,880-row head) and accept | 0.88 | 4.2 ms | 4.4 ms busy (nsys); draft 4.2, accept 2.0 ms by the host timer | draft 4.3 (3.8 with G2's subset), accept 2.1 ms |
| CPU | routed experts of 2 tokens 9.02, group B's shared-expert twins 1.12, dense FFNs of layers 0-2 0.96, NextN MoE 0.18 | 11.28 | 49.9 ms | 72.6 ms (`GUILD_GLM_TIMING`, 66-69% of bound) | 68.4 ms (73%); 67.7 with every shared expert on the GPU |
| Step | per token at 1.85 tokens a step | | 35.3 ms (28.3 tok/s, the GPU's) | 21.36 tok/s, 75% of the GPU bound | 21.80 tok/s, 77% |

Where round 1's CPU time above the bound went (S1G's throwaway counters, two runs): the expert kernels' own rate
13.2 ms (down items at 149-163 GB/s, gate/up NL4 at 192-199), pool phase overhead 8.6-8.9 ms (276 phases a step at
31-32 us, nearly all of it the slowest thread's tail), and 3.4-3.7 ms of `cpu_ffn` outside the phases (quantizing
the activations on the calling thread, setup). G1 took the overhead part; the kernels' rate is unchanged.

The two halves of a layer overlap: group A's experts on the CPU run beside group B's attention on the GPU, and the
GPU's verify work is ~75.7 ms a step. So CPU-only gains could shorten the step by only ~5 ms at short context (S1G),
and the final's verify (78.1 ms) is within ~2.5 ms of the GPU's own work.

At ~15K tokens the GPU sets the pace: the step minus its CPU-expert time is 32.6 ms against 13.2-13.6 at short
context (S1G, round 1), mostly the DSA attention and the indexer. G3's indexer kernels took `idx_score` 126.4 ->
29.7 us and `idx_select` 75.9 -> 16.2 us a call after a 15,999-token prompt (nsys, ~23 calls a step).

## 2. End to end

### Qwen, short context (int3, the same night; 3 runs x 256 tokens per bench)

| Binary | prose ms a window | prose tok/s | code ms a window | code tok/s | tokens a window (prose / code) | draft ms | Benches |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| pre-round-1 | 69.81 | 30.57 | 73.65 | 41.38 | 2.13 / 3.05 | 6.74 / 7.21 | 2, not alternated |
| round 1 | 65.52 | 32.56 | 68.59 | 44.44 | 2.13 / 3.05 | 6.67 / 7.13 | 3, alternated with final |
| round2a | 64.06 | 33.87 | 67.26 | 44.26 | 2.17 / 2.98 | 6.49 / 7.03 | 2, alternated with final |
| final | **63.33** | **34.26** | **66.45** | **44.80** | 2.17 / 2.98 | 6.46 / 7.00 | 3, alternated with round 1 (all 5: 63.35 / 34.24 / 66.46 / 44.79) |
| final, every round-2 switch set back | 65.53 | 32.55 | 68.59 | 44.43 | 2.13 / 3.05 | 6.66 / 7.14 | 1 |
| final, one row group (`GUILD_SPEC_SPLIT=0`) | 67.70 | 32.04 | 73.78 | 40.35 | 2.17 / 2.98 | | 1 |

- The ranges do not overlap: the slowest final run is 63.58 / 66.92 ms, the fastest round-1 run 65.39 / 68.47, the
  fastest round2a run 64.00 / 67.20.
- Round 1 against pre-round-1, alternated on 2026-10-30 (3 pairs): 69.96 -> 65.38 ms (prose) and 73.34 -> 68.43 ms
  (code), -4.58 / -4.91 ms, +7.0% / +7.2% tok/s, all 48 texts identical. The same binary with round 1's five
  switches set back ran like the pre-round-1 binary (70.78 / 73.76 ms).
- Round 2 against round 1, the same binary with every round-2 switch set back (`GUILD_IQ_GGML_ORDER=0
  GUILD_TOPK_SPLIT=0 GUILD_GDN_STEP_COLS=0 GUILD_GDN_FORK=0 GUILD_GDN_COMMIT_FORK=0 GUILD_HC_V2=0
  GUILD_ATTN_TUNED=0`): -2.15 / -2.07 ms, and that arm writes round 1's text.
- Per token: prose 32.8 -> 30.8 -> 29.2 ms, code 24.1 -> 22.5 -> 22.3 ms (pre-round-1, round 1, final). Code gains
  little in round 2 because its exact text takes 2.98 tokens a window against 3.05.
- Greedy decode (`--spec 0`, one run each): round 1 41.81 / 41.83 ms a token, final 40.87 / 40.90 (+2.3%); S1's
  pre-round-1 run 43.98 / 44.83.

### Qwen, long prompts (`--max-context 262144`)

| Prompt, binary | Prefill | ms a window | tokens a window | tok/s | draft ms | Greedy | Runs |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 31,918 tokens, pre-round-1 (S1) | 442 s | 78.26 | 2.51 | 32.07 (30.28 on the first run) | 9.01 | - | 2 |
| 31,918 tokens, round 1 (int3) | ~6.5 min | 72.93 | 2.51 | 34.42 | 8.94 | - | 2 processes x 2 runs |
| 31,918 tokens, final (int3) | ~6.5 min | **69.51** | 2.39 | **34.42** | 8.07 | 42.47-42.56 ms a token, 23.5 tok/s | 2 processes x 2 runs, alternated with round 1 |
| 127,898 tokens, pre-round-1 (S1), 256-token requests | ~1,820 s | 80.60; first request after the prefill 133.15 | 2.42 | 29.96; first request 19.04 | 8.36 | - | 2 requests |
| 127,898 tokens, final (int3), 128-token requests | 1,795 s | **72.04** | 2.46 | **34.15** | 7.94 | 44.83-45.14 ms a token, 22.2-22.3 tok/s | 2 speculative + 2 greedy requests, one process |

At 32K the final window is 3.42 ms shorter than round 1's (pairs -3.49, -3.34), 0.87 ms of it in the draft (Q8
alone measured 0.65 ms). At 128K the first decode after the prefill ran as fast as the later ones (greedy 45.14 then
44.83 ms a token); in S1 it ran 2.4x slower in its CPU experts (Q4 is the likely reason; no run isolated it).
Q5's lane, on round-1 kernels with its own fixes, measured 75.15 ms a window and 44.27 ms a greedy token at 128K.

### GLM (int3; 5-token prompt, 256 tokens, `--ctx 32768 --spec` unless said)

| Run | Binary | tok/s | CPU experts ms a step | step p10 / p50 / p90 ms | verify / draft / accept ms | Cells | Runs |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
| short | round 1 | 21.36 | 72.6 | 84.3 / 85.3 / 89.8 | 80.3 / 4.2 / 2.0 | 16,640 | 3, alternated |
| short | round2a | 21.98 | 67.3 | 83.3 / 83.9 / 84.7 | 78.0 / 4.1 / 1.9 | 16,640 | 2, alternated |
| short | final | **21.80** | 68.4 | 84.1 / 84.6 / 85.3 | 78.1 / 4.3 / 2.1 | 32,768 | 3 + 2, alternated |
| short | final, `GUILD_GLM_CTX_FIRST=16384` | 21.94 | 67.7 | 83.5 / 84.0 / 84.8 | 77.9 / 4.1 / 2.0 | 24,832 | 1 |
| `token_embd` evicted before the run | round 1 | 19.57 (30-31 stalled steps) | | p90 117-120 | draft 7.5-8.1 | 16,640 | 2, alternated |
| `token_embd` evicted before the run | final | **21.63** (0-1 stalled steps) | | p90 85-86 | draft 4.3 | 32,768 | 2, alternated |
| 15,999-token prompt | round 1 | 18.45 | 73.2 | 104.9 / 105.5 / 108.2 | 99.2 / 5.0 | 16,640 | 1 |
| 15,999-token prompt | final | **18.96** | 70.0 | 101.5 / 101.8 / 102.2 | 93.8 / 5.0 | 32,768 | 1 |
| 32,000-token prompt (prefill 688 s, 46.5 tok/s) | final | **18.23** | 70.0 | 101.2 / 102.0 / 102.8 | 94.8 / 5.0 | 32,768 | 1 |
| the same, old indexer kernels (`GUILD_GLM_IDX_LEGACY=1`, same snapshot) | final | 17.16 | 69.8 | 107.7 / 108.8 / 109.1 | 100.7 / 5.3 | 32,768 | 1 |

- Final against round 1 at short context: pairs +1.6%, +1.4%, +3.2%; the ranges do not overlap (21.72-21.86 against
  21.17-21.53). A second, independent pair the same night: 21.73 against 21.40 (+1.5%). In batch A's own
  measurement, when round 1 ran slower (CPU experts 74.8 ms), round2a against round 1 was 20.88 -> 22.03 tok/s
  (+5.5%).
- Final against round2a: -0.6% (2 pairs). That is G3's placement at `--ctx 32768`: 32,768 cells instead of 16,640,
  paid for with four shared experts on their CPU twins. `GUILD_GLM_CTX_FIRST=16384` gives round2a's speed back.
- Plain decode (no `--spec`, one run each): round 1 13.91 tok/s, round2a 14.49, final 13.65 (CPU experts 36.2 against
  32.1 ms a step: every plain token computes the four spilled shared experts on the CPU; this run was also the first
  load after a 128K Qwen run).

## 3. The changes

Every change below keeps output bits unless its section says otherwise, adds no mode or field to
`include/guild/abi`, and has a switch that restores the earlier path in the same binary unless "switch: none".

### Before round 1: six commits of 2026-10-27/28 (UTC) not described elsewhere

| Commit | What | Switch | Measured (Qwen, GTX 1070, same binary, alternated) | Bits |
| --- | --- | --- | --- | --- |
| Guild `e9ee500` | `--expert-cache auto` keeps no GPU expert cache below 5% of the experts (it held 113 slots, 0.46% of 24,576 experts, 0.3% of lookups, and every layer waited on its plan) | `--expert-cache N` | windows 74.1 -> 72.8 ms (prose), 79.5 -> 76.4 (code); code 37.9 -> 39.9 tok/s; 356 slots was slower than 113 | experts that leave the GPU round as the CPU does: the text changes from token 136 (prose) |
| Guild `1e8257e` | the CPU sums its experts' rows when it serves them all; the GPU adds the shared expert through the same combine kernel | `GUILD_HOST_COMBINE=0` | windows 1.2-1.4 ms shorter; prose 29.5 -> 30.0, code 40.0 -> 40.8 tok/s | `host_combine_parity` at 1-4 tokens (fails without the FTZ/DAZ modes the kernel's `-ftz=true` needs); outputs identical |
| Guild `54bfc99` | one PLE key projection a group for a quantized key (28 MB a token read once a group) | `GUILD_PLE_BATCH=0` | windows 71.4 -> 71.0 (prose), 75.1 -> 74.4 ms (code) | each column bitwise its single call; outputs identical |
| kernels `e6135e7` | direct down and column up kernels for the 1-2-token hyper-connection read before sm_70 | `GUILD_HC_DIRECT=0` | down 45 -> 34 us (192 GB/s), up 39 / 43 -> 33.5 / 35-37 us; prose 28.9 -> 29.9, code 39.1 -> 40.6 tok/s | `fused_gr_check` against the plain read at 1-8 tokens, `gr_parity_staged` (Guild `883cfd9`); outputs identical |
| kernels `7203269` | that down kernel as rounds of its ring, a third of the code | none | 2-token down kernel 43.2 -> 38.3 us in the engine (nsys); it was refetching its 2,800 straight-line instructions | `gr_parity_staged` |
| kernels `515e3d1` | the native top-10 router's ten rounds as a loop (1,518 -> 408 instructions) | none | 7.92 -> 7.35 us a 2-token launch back to back, ~2 us less after a large kernel | ids and weights of 65,536 rows equal; output unchanged |

### Round 1 (merged 2026-10-30 as Guild `8c74432`..`ab02a18`, kernels `89351ec`, `5641a3d`)

Measured together: -4.58 / -4.91 ms a window against pre-round-1 (above). The four changes' own A/Bs add up to
-5.0 to -5.9 ms expected.

**Q1, the verify head's Q6_K rows staged whole** (kernels `89351ec`, test Guild `8c74432`). The 2560-wide head is
10 Q6_K blocks a row, which the staged kernel read as chunks of 8 + 2. Rows of 9-12 blocks now run in their own
instantiation: the whole row in one chunk, a running total for the four accumulators (the same add order), each
row's address carried to the warp's next row, the scale product as an exact 16-bit multiply (dp4a lies in
-16256..16384), activation records whenever they fit 48 KiB, `__launch_bounds__(256, 1)`.
- Switch: `GUILD_MMVQ_ROW_CHUNK=0`; the 8-block kernels compile to byte-identical SASS against the previous object.
- Head harness, us at 1..8 columns: 2505 3224 3631 4780 4961 6141 7372 7627 -> 2429 2439 2707 3049 3477 3923 5451
  6069. In the engine (nsys): 3.13 / 3.59 / 4.73 -> 2.44 / 2.66 / 3.00 ms at 2 / 3 / 4 columns, ~1.47 ms a window.
- Window: -1.89 / -1.85 ms (lane, 2 pairs); the second run of the A/B: -1.55 / -1.52 ms (+2.6% tok/s).
- Bits: `mmvq_staged_parity` with Q6_K rows of 9, 10, 12 and 13 blocks and unaligned row ends, 578,772 outputs at
  1-8 columns, 0 differ; texts identical.

**Q2, group 1 keeps the shared KV staging map group 0 left** (Guild `6f64d47`). When the 12 QSA layers share one
VRAM staging pool, a window's second row group now reuses the first group's residency map in the same layer and
copies only its misses.
- Switch: `GUILD_KV_STAGING_REUSE=0`.
- With the shared pool forced (`GUILD_KV_SHARE_STAGING=1`, 6,952-token prompt): misses 13,393 -> 8,380 blocks a
  window (104.6 -> 65.5 MiB over PCIe), copy kernel 9.13 -> 5.79 ms a window (nsys), windows -3.5 ms (5 pairs; the
  second run of the A/B: -3.9 and -4.0 ms).
- **No effect at the server's settings**: at `--max-context 262144` this card gets per-layer pools, which never
  reset their map. It matters when VRAM is tighter, or with `GUILD_KV_STAGING_BUDGET_MIB` /
  `GUILD_MAX_CONCURRENT_SESSIONS`.
- Bits: `kv_stream_parity` section 5 (two layers sharing a pool in the verifier's order; fp16, int8, q4_0; clock and
  direct map; a stale-map control fails with 390 mismatches a format), `kv_host_only_test` at 262,144 cells; tokens
  identical across 15 reports.

**Q3, the router chain** (kernels `5641a3d`, Guild `fe06b33`, test `754fe5e`, `ab02a18`). A BF16 GEMV with two
output rows a block for the router-shaped products, and in a two-group window each shared expert forked in the other
group's post instead of before the router, which removes a false join (post(l, B) waited for A's next shared
expert) and leaves the top-k and the doorbell alone on the GPU.
- Switches: `GUILD_MMVF_ROWS=0` (one-row kernels), `GUILD_SH_FORK=0` (the old fork and join). Both opt-in on HIP,
  and `ab02a18` makes the fork opt-in on the gfx906 compat build too.
- Engine (nsys, per group and layer): GEMV 22.82 -> 16.74 us, top-k 12.64 -> 8.64, doorbell 7.78 -> 6.79; chain
  44.67 -> 33.38 us. `wait_flag` time inside a window 4.77 -> 2.54 ms.
- Window: -2.29 / -2.15 ms (lane, 2 pairs; each change alone gives about half); second run: -1.90 / -1.77 ms.
- Bits: `mmvf_rows_parity` at 1, 2, 4 and 8 rows a block against the one-row kernels (partial blocks, padded strides,
  signed zeros, subnormals, multi-column == single-column), `route_window_parity`, `router_fast_parity`,
  `router_top10_parity`; texts identical in every switch combination.

**Q4, the NUMA-split expert arenas out of NUMA balancing** (Guild `bc7f565`). After placement the resident arena and
the repacked IQ3_S rows get an `MPOL_LOCAL` policy, which the kernel's balancer skips. It had been migrating them:
20-40 thousand hint faults a second during decode, with bursts of 400 thousand (Q4's lane, `/proc/vmstat`), and ~9 GB
of pages moved during a load (S1).
- Switch: `GUILD_EXPERT_AUTONUMA=1`.
- Alone (2 pairs, the lane's CPU-kernel half on in both arms): 70.66 -> 68.93 ms (prose), 73.96 -> 72.65 (code);
  CPU dispatch 28.1 -> 23.4 and 32.1 -> 25.9 ms a window. GPU waits for the CPU 1.96 -> 0.34 ms a window by the
  lane's probe (that probe also had the lane's CPU-kernel half, which was not merged).
- In round 1's pairs: 83-133 thousand hint faults/s and 1.0-4.3 GB migrated in an 80 s bench with the arenas open to
  the balancer, 5-14 thousand/s and 5-70 MB with them closed. The windows also got steadier: prose 65.17-65.54 ms over
  12 runs against 69.43-71.22.
- An old-path run where the balancer did not fire ran like the new path, so a short A/B can miss or exaggerate it.
- Bits: page placement only; `pool_numa_test` (20,817 rows bitwise), texts identical.
- The lane's CPU-kernel half (guild-kernels-cpu `b7dbae1`) was not merged: "Tried and not kept".

### Round 2, Qwen

**Q5, the QSA indexer at long context** (kernels `dfe200e`, Guild `790939f`, test `e40660f`).
- *The 128K repeats.* With `--kv-host-only` the draft layer keeps its K/V as a ring over its 32K-cell window. A
  request resuming from the prefix checkpoint restored the main model but not that ring, so past 32K cells the drafts
  depended on the requests before. `790939f` refills the ring from the host copy after the restore (no switch).
  After it, 8 speculative repeats in one process at 128K are identical, tokens and window shapes (it was token 29
  and 214 in S1; at 64K the second request left the first at token 154).
- *The top-k.* Below sm_90 the decode top-k ran on one CTA a query (1-5 of 15 SMs, at least 5 passes). It now gives
  a query up to 8 CTAs over slices of at least 2,048 blocks, then merges their lists by the same rule (ties to the
  lowest index); integer histograms only, so no result depends on timing. Default on compute capability 6.x.
  Switch: `GUILD_TOPK_SPLIT=0`.
- Harness, ms a call at 262,144-cell capacity, 1 query: 0.024 -> 0.019 (2,600 cells), 0.026 -> 0.024 (32K), 0.074
  -> 0.032 (128K), 0.128 -> 0.038 (262K).
- Engine at 128K (arms alternated request by request in one process): speculative 76.16 -> 75.15 ms a window,
  greedy 44.80 -> 44.27 ms (second run: -1.07 / -0.46 ms). At 32K within the 0.1-0.3 ms noise.
- Bits: `qsa_topk_parity --ties` (ties of every kind at 16K-256K blocks against an independent host sort, both
  kernels, two streams at once) and `--scores`; tokens identical across arms.

**Q6, the GDN recurrence and commit** (kernels `1d7cf15`, Guild `f5a4c92`, test `4b4d728`). The verify recurrence and
the commit split each value head's 128 state columns over four blocks (192 blocks of 128 threads, 8 to an SM, against
48 blocks of 512 threads at 86 registers); the gated RMSNorm moves to a small pass with the old warp-sum order. A
GDN layer's conv and alpha/beta run on a second, highest-priority stream beside the gate product, and the commit's
small commits beside its state commits.
- Switches: `GUILD_GDN_STEP_COLS=0`, `GUILD_GDN_FORK=0`, `GUILD_GDN_COMMIT_FORK=0` (CUDA default, opt-in on HIP).
- Window: -0.85 / -1.08 ms (lane, 3 runs an arm; kernels alone -0.71 / -0.88); second run -0.93 / -1.06; after
  merging, -0.78 / -0.93.
- Bits: `gdn_step_parity`, 408 cases (head shapes 16/48, 8/8, 32/32, 2/4; T = 1-8; commit at n_keep 0..T), 0 differ
  from the old kernels; each run also checks that a T-token verify gives what T one-token steps commit. Texts
  identical.

**Q7, the hyper-connection read at one and two tokens** (kernels `962818f`; fix `b511c94`, test Guild `30676b5`).
The norm in one pass with its inputs held in registers, four extra rows of blocks that prefetch the first
`GUILD_HC_PF` bytes (4096) of every w_down / w_inject row into L2 while the norm waits, the column up kernel with
its next loads unconditional, and the direct down kernel refilling each ring slot right after use (80 registers at
one token instead of 104).
- Switch: `GUILD_HC_V2=0` (bit mask, default 23); `GUILD_HC_PF=<bytes>`, `0` turns the prefetch off.
- Engine (nsys medians, norm + down + up): 78.85 -> 74.05 us at 1 token, 86.95 -> 81.44 at 2; ~208 reads a window.
- Window: -0.63 / -0.60 ms (lane); second run -0.61 / -0.56; after merging, `GUILD_HC_V2=0` costs +0.81 / +0.83.
- Bits: `gr_parity`, `gr_parity_staged`, `gr_parity_staged_pf`; the engine's start-up check against the plain read
  at 1-8 tokens; 30 runs identical.

**Q8, the draft layer's attention at long context** (kernels `31ab7b9`). The draft layer's decode attention over its
32K-cell window: the chunk kernel at G = 12 pads the query rows in shared memory and folds 8 cells a step, the merge
splits each head over 4 blocks and loads 16 chunks' partials before folding them; the same operations in the same
order.
- Switch: `GUILD_ATTN_TUNED=0` (read once a process).
- Harness at 32K cells: chunk 661.6 -> 512.8 us, merge 230.7 -> 98.0 us. Engine at 32K: draft 8.94 -> 8.29 ms a
  window, 34.56 -> 34.88 tok/s (6 of 6 pairs); second run 34.38 -> 34.93 tok/s. At short context no change (32.51 /
  44.43 against 32.56 / 44.46 tok/s).
- Bits: 171 of 171 harness cases bit-identical (KV modes 0-3, G = 8 and 12, masked pages, batched queries); 9 KV and
  QSA ctests in both arms; 32K tokens identical to round 1.
- The rest of Q8 was a study, with no commit: the spec-min-p default and one-token windows (below and in "Tried and
  not kept").

### Round 2, GLM

**G1, the CPU expert pool** (Guild `0e47c38`, `078fe33`; switch: none, A/B against the round-1 binary). Each NUMA
node's queue is one 64-bit word on its own 128-byte line (generation, item count, next item), claimed by
compare-and-swap; done counters count items, so the caller returns when every item is done instead of waiting on an
all-worker barrier; the mutex and condvar are used only when a worker actually slept. The hidden activations are
quantized by the item that completes each 256-row block (the separate phase is gone), Q8_K quantization has an
AVX-512 path (0.7 instead of 4.8 ns a value, the same bytes as `quantize_row_q8_K_ref`), down items are 16 rows, and
the last 16 items of each queue are split in four.
- Lane, 6 + 9 runs in two batches: CPU experts 73.9 -> 68.1 ms a step with `0e47c38` alone, and 72.2 -> 67.8 with
  both commits (67.6 with `0e47c38` alone), p90 step ~91 -> ~85 ms, +3.9% and +2.3% tok/s; second run (both
  commits, 3 pairs) 73.3 -> 68.1 ms, +3.1%. Phase totals over a run (`0e47c38` alone): activations ~300 -> ~115 ms,
  hidden ~225 -> 1 ms, down ~4,080 -> ~3,630 ms.
- Bits: ids identical in every run; the AVX-512 Q8_K against the reference on 784,095 engine rows and 300,000 x 16
  test blocks (ties, infinities, NaNs, denormals): 0 differ; the fused quantization against a whole-row
  `from_float`: 12,767 calls, 0 bytes differ; a standalone stress test of the pool (200,000 phases in each of 6
  thread layouts): no error, no hang.

**G2, drafts over a token subset of the head** (Guild `00b72bd`, `1e95569`; opt-in). The NextN draft multiplies only
the subset's rows of the same Q6_K head, in place, and maps the argmax back to a token id on the GPU; verify still
multiplies the whole head. `data/draft_vocab_glm.bin` holds 117,354 of 154,880 ids (75.8%, 376 of 496 MiB): the
tokens covering 99.9% of each of three corpora (69.1M tokens: local code and docs, WikiText-103, 19 Gutenberg books)
plus the CJK, kana, Hangul and Cyrillic ranges, digits, control tokens and byte pieces.
- Switch: it runs only when `draft_vocab.bin` is in the tokenizer directory, `GUILD_GLM_DRAFT_VOCAB=<file>` is set or
  `glm5_run --draft-vocab <file>` is given; `GUILD_GLM_DRAFT_VOCAB=0` turns it off. 458 KiB of VRAM.
- Draft 4.215 -> 3.663 ms a step (every prompt); tok/s +0.64% (30 pairs, standard error 0.20%), +0.40% +- 0.26% in
  the second run, +0.51% +- 0.17% pooled. After merging: 4.3 -> 3.8 ms, the same ids.
- French loses 5 of 116 accepted drafts; Russian and Chinese lose none.
- Bits: `q6k_mmvq_rows` bitwise against `q6k_mmvq` (sorted and shuffled ids), the argmax over ids; with
  `GUILD_GLM_SHEXP_SPLIT=0` the subset, the whole head and plain decode give the same ids on 8 prompts. Under the
  default split a different accepted draft can change the text (one prompt of six, from token 131).

**G3, 32K tokens of context on the 8 GB card** (Guild `ba77145`, `21e0fc0`, tool `df29d37`). A token of context in
the 12 DSA layers took 19,968 B of VRAM and now takes 13,184 B: the indexer's per-token key and gate rows are rings of
512 rows (a window is at most 256 cells plus 3 of the open pool; snapshots keep the open pool's rows), the pooled keys
are stored as FP16 (they were already rounded to FP16 in an FP32 buffer, so the floats read are the same), and the
prompt windows' score rows are now counted. The context asked for, up to 32,768 tokens, is placed before the FFNs;
at `--ctx 32768` that leaves layers 42-44's and the NextN block's shared experts on their CPU twins ("8010 MiB free,
6214 MiB fixed, context 32768 tokens x 13184 B, 5 MiB left; 4 shared experts on the CPU"). The indexer's query is
staged in shared memory and its top-k found by four radix passes.
- Switches: `GUILD_GLM_CTX_FIRST=<tokens>` (`16384` gives 24,832 cells with every shared expert on the GPU;
  `--max-context 16640` is round 1's configuration); `GUILD_GLM_IDX_LEGACY=1` (the old indexer kernels, read every
  window).
- Indexer kernels (nsys, 16K): `idx_score` 126.4 -> 29.7 us, `idx_select` 75.9 -> 16.2 us; step p50 106.2 -> 102.7
  ms at 16K and 110.1 -> 103.5 ms at 32K (lane), 108.8 -> 102.0 ms at 32K after merging; second run at 16K -3.3 ms.
- The spill costs ~0.2-0.3 ms a step a shared expert (4: p50 +0.8 ms, -0.5% tok/s, measured with a throwaway
  switch).
- Bits: `glm5_kernels_test` (ring of 16 rows; legacy == new scores and selection at 300-33,000 pools with ties); every
  logit of 529 plain steps after a 2,051-token (lane) and a 2,302-token (second run) prompt equal to round 1's,
  including after a snapshot restore; ids at 5 and 15,999 prompt tokens equal to round 1's at round 1's placement.
  The 32K placement itself changes numerics (CPU twins): greedy decode differs from round2a's from id 61.

**G4, `token_embd` copied out of the GGUF mapping** (Guild `230ca2f`). `embed_rows` read each token's row of the
643 MiB Q8_0 table from the page cache; after other models evicted it, every first use of a token waited on the RAID0
disk (10-90 ms, once 221 ms). A thread started early in the load copies the table into anonymous memory with
`pread` (8 MiB chunks); the load joins it at the end.
- Switch: `GUILD_GLM_EMBED_COPY=0`.
- Table evicted before each run: 19.15 -> 20.76 tok/s (lane, 3 pairs), 19.29 -> 21.32 (second run), 19.57 -> 21.63
  after merging (+10.5%); stalled steps 30-32 -> 0-1. The copy takes 5.0-8.6 s on its own thread and the load waited
  0.00 s for it in every run; 643 MiB more RAM. A fully cold load was not measured.
- Bits: the same bytes and arithmetic; ids identical in all runs.

### Round 2, correctness

**C1, multi-token CPU expert rows in ggml-cpu's float order** (guild-kernels-cpu `5147f69`, Guild `c40d576`; tools
`adc80e5`, docs `7a1ea2a`). A token's routed experts that miss the GPU run on the CPU; a group of tokens sharing an
expert took the multi-token kernels, a lone token ggml-cpu's `vec_dot`. Their integer parts were the same, their
float steps were not (the format's constant folded into the block scale, their own 8-lane sum, one IQ4_NL chain
instead of two), so a verify window computed other bits than one-token decode. The multi-token kernels now take
ggml's float steps (one 8-lane fmadd of `d_x * d_y` a block, the constant times `hsum_float_8` at the end, IQ4_NL in
two chains), and so do GLM's NL4 rows.
- Switch: `GUILD_IQ_GGML_ORDER=0` (both repositories).
- Window: -0.74 / -0.40 ms (2 pairs), -0.79 / -0.50 in the second run, -0.75 / -0.41 after merging; one thread at
  4 tokens: gate/up 386.0 -> 352.1 us an expert, down within 2%.
- Bits: `iq_ggml_order_test` (every format, 1-8 tokens, AVX2 and AVX-512): 0 of 72 token-sets differ from `vec_dot`
  (1,145 differ in the old order); `native_expert_parity` on the model's layers 2, 3, 30, 47: 0 of 1,920 gate and
  0 of 7,680 down rows differ. Greedy text unchanged for Qwen.

**C2, the GEXEC CPU expert providers in the same order** (Guild `92277de`). The four GEXEC providers (the .gpack /
GEXEC flow; the GGUF + pack engine measured here runs the native provider) had the same old order; they now give
every token of a group `vec_dot`'s bits, and `kGExecExpertImplVersion` moves 5.0.0 -> 6.0.0 so a `.gexec` record
tuned on the old numerics no longer binds. `GUILD_GEXEC_IQ512=1` lets the test reach the 512-bit kernels on this CPU.
- Switch: `GUILD_IQ_GGML_ORDER=0`.
- Cost, one thread: 2-token IQ4_NL down rows 6-7.5% slower (192.9 -> 207.4 us an expert; the second run +3.5% to
  +11.3%), +1-3% for a whole expert; gate/up within noise.
- Bits: `gexec_optimizer_test --kernels`, every provider and format, 1-8 tokens, against `vec_dot` and against one
  token alone: 284 failed checks before, 1,459 passed and 0 failed after.
- It also showed that nothing else in a Qwen window breaks the contract at 8K and 32K: with the old order set back,
  speculative leaves greedy at token 44 of the 8K prompt (Q5's token); with the new order, not within 256 tokens.

### Two fixes found by the second runs (batch B)

- **`gexec_iq4nl_down_rows` below `GUILD_IQ_MT_MIN`** (Guild `acea395`): with `GUILD_IQ_MT_MIN` = 3 or 4 a group of
  2-3 tokens computed only token 0's down rows and left the others unwritten (the default, 2, sends only one-token
  groups there). Each token now runs the one-token kernels on its own activations. New ctests `gexec_kernels_mt3` and
  `gexec_kernels_mt4`: 1,343 passed and 116 failed before, 1,459 passed and 0 failed after.
- **`GUILD_HC_PF` clamped to one row** (kernels `b511c94`, ctest `gr_parity_staged_pf` in Guild `30676b5`): a value
  above a row's 20,480 bytes prefetched past each row, and past w_inject's allocation for its last row. It is now at
  most 160 lines, with a `static_assert` that a row is whole lines; the default (4096 B) is unchanged. A prefetch
  cannot change bits, so the new ctest also passes on the unclamped kernel: it shows the clamped path gives the same
  bits, not that the overrun is gone. compute-sanitizer 2022.4.1 does not run on driver 580, so the clamp rests on
  the code.

## 4. Correctness

### Qwen: speculative decode equals greedy decode

| Prompt | Before: first token where speculative leaves greedy | Final | Runs |
| --- | --- | --- | --- |
| prose / code, 256 tokens | 155 / 138 (S1, pre-round-1 and round 1); one row group against two: 199 / 138 | `--spec 4` == `--spec 0` == `GUILD_SPEC_SPLIT=0` | 3 runs each (int3), and again in an independent check |
| 8,040 tokens of real text | 44 (Q5, round-1 binaries) | identical | 128 tokens (batch A), 256 tokens with one row group too (C2) |
| 31,918 tokens | round 1's speculative text leaves the final greedy text at 165 | identical | 4 speculative runs against 2 greedy runs, 256 tokens |
| 127,898 tokens | 29, two row groups (Q5, round-1 binaries) | identical | 2 x 128 speculative against 2 x 128 greedy, one process |

The cause was C1's float order and nothing else on these prompts: with `GUILD_IQ_GGML_ORDER=0` the final binary
diverges at the old tokens again. On the short prompts the final greedy text equals round 1's greedy text; the
final speculative text equals round2a's. One row group (`GUILD_SPEC_SPLIT=0`) was not run at 128K: it is read once
at load, and another process needs another 30-minute prefill.

### 128K repeats

S1's two 256-token runs of the 128K prompt in one process differed from token 29 (another process: 214). The cause
was the draft layer's K/V ring on a checkpoint resume (Q5). Since `790939f`, repeats are identical (Q5: 8 in a
process; int3: the two speculative and the two greedy 128-token requests). The refill adds a host-to-device copy of
up to 32K cells of one layer and a stream sync a resume; 256 speculative tokens at 64K still took 7.3-7.4 s a
request.

### GLM: the shared-expert split

A 2-token verify window runs as two 1-token groups, and group B's shared expert runs on its CPU twin
(`GUILD_GLM_SHEXP_SPLIT`, on by default since `0430a78`, to even the GPU's and the CPU's halves of a layer). The twin
does not round as the GPU does (Q8_1 activations, the MMVQ lane order, the GPU's `expf`), so a window's second row is
not what decode computes. By default `--spec` leaves plain greedy decode from id 7 of the 5-token prompt (batch A,
16,640 cells; at 32,768 cells the final binary's `--spec` and greedy ids equal those runs' up to id 60). With
`GUILD_GLM_SHEXP_SPLIT=0` every row matches: 255 of 255 logits rows bitwise equal with oracle drafts (C1), and
speculative == plain for 256 ids in batch A, in batch B with the four spilled shared experts (in one process from a
snapshot) and in a second check (separate processes).

Cost of the exact path: 3.1% (C1, 3 pairs before G1: 21.49-21.57 against 20.81-20.85 tok/s, verify +4.1 ms a step);
about 5% after G1 (one run: 20.89 against 21.95-22.07). `GUILD_GLM_PIPE=0` (one group of two tokens) is not exact
either: the first window's first row is bitwise equal, its second row differs and so does every row after it (C1).
It is not the default.

### Outputs that changed on purpose

- Qwen's speculative text against round 1, from token 155 (prose) and 138 (code), at 32K from token 165: it now
  equals greedy decode.
- GLM's greedy text, from id 7 of the 5-token prompt: C1 changed how the prompt's prefill rounds at a near tie
  (37322 against 67326); with `GUILD_IQ_GGML_ORDER=0` the batch-A binary writes round 1's greedy ids.
- GLM at `--ctx 32768` (G3's placement): greedy decode differs from round2a's from id 61; `--spec` gave round2a's ids
  on the 5-token prompt and on 6 chat prompts x 256 tokens.
- Qwen with `e9ee500`: experts that leave the GPU cache are computed with the CPU's rounding.

### Tests

The final build's whole ctest suite: 122 tests, 117 passed, 4 skipped for hardware (`prefill_fused_moe_test` and
`prefill_fused_iq_test` need sm_80; `expert_multi_test` needs AVX512-VNNI/VBMI; `pool_stress`), 1 failed
(`ple_parity`: its fixtures `bench/micro/ple_in.bin` / `ple_out.bin` and a Q2_0 GGUF are not on this machine; no
change here touches PLE). `native_expert_parity` on the Qwen GGUF: 0 failures. `tools/format_code.py --check`: 501
files formatted.

## 5. Tried and not kept

| Item | Idea | Measured | Verdict |
| --- | --- | --- | --- |
| GLM, 2026-10-28 | 256-row gate/up items with the hidden quantization fused in | CPU experts 73.9 -> 83.6 ms a step, 21.4 -> 19.0 tok/s, same tokens | reverted; G1 fused it with 64-row items instead |
| Q1 | one chunk only for every column count | 1..8 columns 2463 2660 3208 4315 4870 6784 8238 6970 us: 6 and 7 columns regressed | not kept |
| Q1 | `__launch_bounds__(256, 3)` (80 registers); no min-blocks hint | 3 / 4 columns 3510 / 4740 us; without a hint 2888 / 3457 / 4401 us at 3 / 4 / 5 columns, against 2707 / 3049 / 3477 | `(256, 1)` kept |
| Q1 | the 32 KiB records rule for one-chunk rows; 12 warps a block | 4 / 5 / 6 columns 3581 / 4070 / 4666 against 3095 / 3524 / 3933 us; 3794 against 3818 us | not kept |
| Q2 | smaller per-layer pools (8 MiB a layer) | 674 misses (5.3 MiB) a window, 76.76 ms at 262144, tokens identical: no faster than the full per-layer pools the server gets | not committed |
| Q2 | the miss copy on the copy engine | the miss list is computed on the GPU; a DMA would need 24 host round trips a window, and the zero-copy kernel already reads at ~12 GB/s | not built |
| Q3 | the stored-ring doorbell (ABI mode 14) on CUDA | ring kernel 7.8 -> 5.8 us; windows prose 70.16 / 68.96 against 68.17 / 68.45 ms, code 71.94 / 72.18 against 72.47 / 71.11 | reverted |
| Q3 | one join event per group | prose 68.73 / 69.59 ms, slower than the deferred fork; ~96 streams inside a verify graph | dropped |
| Q4 | the pool's down tasks also doing the MoE combine | prose 69.46 -> 69.18, code 72.80 -> 72.70 ms (2 pairs), tokens identical | too small for a change in another file |
| Q4 | the CPU-kernel half: AVX2 Q8_K quantizer (4.70 -> 0.76 us per 2,560 values), IQ4_NL and Q8_0 down rows several at a time (one core 6.1 -> 9.0 and 9.9 -> 13.5 GB/s) | CPU dispatch -4 to -5 ms a window, but windows 68.92-69.08 / 72.67-72.70 ms without it against 68.86-69.00 / 72.52-72.62 with it | not merged (guild-kernels-cpu `b7dbae1` on a lane branch) |
| Q4 | the GEXEC fused tables; the repacked IQ3_S half scales by a dword broadcast; a 2048-byte prefetch | one core: IQ4_NL down 5.3 against 6.1 GB/s, Q8_0 6.0 against 9.9; 10.5 -> 8.9 GB/s at 1 token; 7.35 against 8.15 GB/s | slower |
| Q5 | a fewer-shuffle decode scorer | harness 111 -> 93 us (2 queries, 128K); engine scores + top-k 2.15 -> 2.14 ms, windows within noise | left out |
| Q5 | other slice aggregations for the split top-k | 128K, 1 query 0.047 against 0.033 ms | slower |
| Q6 | heads over 2 blocks with several columns a thread, persistent blocks, a state reload, a runtime head count, ILP loops | verify at 1 token 23.1, 23.1-23.2, 23.1, 19.1, 22.0-24.0 us against 18.8 for the kept kernel (harness) | slower |
| Q6 | a prefetching `gdn_ab`; the commit without the stream priority | 6.29 -> 6.7 us at 2 tokens; commit span 1.95 against 1.75 ms | slower |
| Q6 | skipping the commit when every draft is accepted | it needs a 113 MB second state write every window (0.54 ms) to save ~1.45 ms in full-accept windows only | not built |
| Q7 | the down kernel prefetching the up kernel's first w_up rows | up -0.90 / -0.74 us, down +1.02 / +0.96 us; windows 64.90 / 67.95 against 64.93 / 67.99 ms | removed |
| Q7 | an up-kernel grid dividing the 2,560 columns evenly | 30 blocks 38.56 us, 32 blocks 39.43, against 38.37 at 45 blocks | not kept |
| Q7 | the norm fused into the down kernel | 2.5x the activation loads in a kernel already bound by its loads, and no prefetch | not built |
| Q8 | a lower `--spec-min-p` | mean of five prompts with exact texts: 0.5 33.97, 0.4 33.79, 0.3 33.31, 0.2 33.03, 0.1 32.69, 0.0 32.70 tok/s; 0.4 +0.9% on prose, -2.5% on Chinese | 0.5 kept |
| Q8 | always drafting one token (no one-token windows) | prose -1.4%, code -0.6%, Chinese -4.7% | dropped |
| Q8 | separate thresholds (chain 0.3, first draft 0.5) | +0.2% prose, +0.6% code, -1.0% / -1.4% / -1.7% Chinese / Spanish / Japanese | dropped |
| Q8 | drafting while the CPU computes a one-token window's last layer | the draft needs that layer's output; hiding every such wait would be worth at most ~3% | not possible |
| G1 | a spinning launcher handoff; a 20,000-pause spin budget; no stealing from the other node; 32-row gate/up items; a 32-item tail | CPU +1.26 and +0.71 (two runs), +1.91, +0.64, +0.66 ms a step; -0.28 against -0.70 for the 16-item tail | not kept |
| G2 | smaller subsets (en 99%, 59,816 rows: -7.7% predicted); a 94,776-row subset in the engine; a fallback below a logit floor; k-means clusters of the head | draft 4.20 -> 3.29 ms but -0.36% tok/s (fewer drafts accepted); floor +0.21% offline; clusters -1.9% offline | not kept |
| G4 | 2, 4 or 8 parallel `pread`s; pre-touching the mapping | 5.06-6.01 s against 5.01-6.76 s for one reader on this RAID0; a pre-touched page cache can be reclaimed again (the load's own arenas did) | one copy thread kept |
| C1 | `GUILD_IQ_MT_MIN=1` (multi-token kernels for every group) | exact too, but greedy text changes (prose token 76) and decode costs 1-3% | not kept |
| C2 | IQ4_NL ggml-order variants for 2-token groups | one row a pass 219-227, the block pair unrolled 201-211, against 200-209 for the kept kernel (one chain: 192-198 us) | not kept |
| S1 | one row group a window (`GUILD_SPEC_SPLIT=0`) | final: 67.70 / 73.78 against 63.33 / 66.45 ms | two groups kept |

## 6. What remains

**Qwen, distance from the bound now** (short context; S1's before, the changes' measured after):
- Hyper-connection read: 74-81 us a read against 63.4 us at the bound, ~208 reads a window.
- Dense Q8_0 products: 93% of their bound, but read twice a window. One row group reads them once and is 4-7 ms a
  window slower, because the CPU experts (26-30 ms a window of host time) then run on the critical path.
- GDN: the commit runs at ~40 us a layer against ~30 us to read and write its 3 MiB of state at 210 GB/s; the norm
  pass Q6 added costs ~3.5 us a layer and row group (at most ~0.24 ms a window).
- Routing: the top-k and the doorbell are ~15 us of latency a group and layer, ~1.4 ms a window; the router GEMV runs
  at ~157 GB/s.
- Verify head: 3.00 ms at 4 columns against 2.48 at the bound; at 2 columns it is at the bound.
- One-token windows: 13.3% of windows over nine prompts (Q8), ~45 ms each, ~16 ms of it waiting for the CPU experts
  with nothing to overlap (S1); 9.1% of decode time.
- Long context: the QSA attention reads its selected cells at ~60 GB/s (1.95-1.98 ms a window at 32K and 128K), and
  the draft layer still costs 1.1-1.6 ms more a window at 32K than at short context.

**GLM:**
- CPU experts 68.4 ms a step against 49.9 at the bound (73%). What G1 left: ~0.8 ms a step of setup and ~0.8 ms of
  activation quantization on the calling thread, and the kernels' own rate (down rows at 149-163 GB/s, gate/up at
  192-199), which lives in guild-kernels-cpu.
- The verify's GPU work (~75.7 ms a step) is now within ~2.5 ms of the verify (78.1 ms), so further CPU gains show
  little until the GPU's share shrinks: the second group re-reads every attention matrix (~29 ms a step), mHC is 542
  small latency-bound kernels (3.8 ms), the KDA core and small products 7.4 ms, the routers 3.0 ms.
- Long context: `mla_attend_k` takes ~660 us a call after a 16K prompt (~44 us at short context); 128K needs
  host-resident latents (the FP16 latent is 1,024 of the 1,088 bytes a token takes in a DSA layer).

**Decisions left to the owner:**
- Ship the CUDA kernels: `perf/sm61-decode` (`b511c94`) is local, and `backends.lock.json` pins `fe4c080`.
- GLM's `GUILD_GLM_SHEXP_SPLIT` default: on (21.8 tok/s, `--spec` leaves greedy) or off (exact, ~5% slower).
- GLM's context default: 32,768 cells with four shared experts on the CPU (-0.6% at short context, 5 MiB of VRAM
  left after the load, outputs differ from round 1's) or `GUILD_GLM_CTX_FIRST=16384` (24,832 cells, all on the GPU).
- G2's draft subset: copying `data/draft_vocab_glm.bin` to the tokenizer directory as `draft_vocab.bin` turns it on
  (+0.4-0.6%, French -4% of accepted drafts; a bad file makes the load fail).
- Q4's CPU-kernel half (`b7dbae1`): CPU headroom only, no window gain at short context.

**Not covered:** HIP builds (Q3's, Q6's and Q7's paths are opt-in or not compiled there); the multi-slot batch graph
with Q6's fork; Ornith, which shares Q8's attention file (bit-equal in the harness, not run); a fully cold GLM load
with G4's copy; GLM beyond 32K; one row group at 128K; a memory checker on Q7's prefetch.

## Reproduce

Qwen (from the repository, the API server stopped; `QWEN` and `PACK` are the first GGUF shard and its pack):

```bash
.venv/bin/python tools/bench_runtime.py --exe build-cuda12/guild-generate --model $QWEN --pack $PACK \
  --profile data/expert-profile.bin --mtp <mtp dir> --context 32768 --resident-budget-gib 56 \
  --runs 3 --max-new 256 --out b-new.json
GUILD_HC_V2=0 .venv/bin/python tools/bench_runtime.py ... --out b-hc0.json       # one change set back
.venv/bin/python tools/bench_runtime.py ... --spec 0 --out b-greedy.json         # greedy, for the text check
# long prompts: --context 262144 --prompts <[{"name":..., "text":...}]> --timeout 3600
```

GLM (`M` is the first of the five shards):

```bash
GUILD_GLM_TIMING=1 GLM5_RUN_STEPS=1 build-cuda12/glm5_run $M 785,6722,315,9621,374 256 --ctx 32768 --spec
GUILD_GLM_SHEXP_SPLIT=0 build-cuda12/glm5_run $M 785,6722,315,9621,374 256 --ctx 32768 --spec \
  --ab GLM5_RUN_PLAIN=0,1 --reps 1                                    # exact path: speculative == plain
build-cuda12/glm5_run $M @prompt.ids 256 --ctx 32768 --spec --ab GUILD_GLM_IDX_LEGACY=0,1 --reps 1
```

Bitwise tests: `ctest -R "parity|gexec|glm5_kernels|reference|router|mmv" -E ple_parity` from `build-cuda12` (66
tests). The runs' logs and JSON are on this PC under `/home/ubuntu/lanes/` (`s1`, `s1g` for the baselines; `int`,
`int2`, `int3` for the merged rounds; one folder for each change), not in the repository.
