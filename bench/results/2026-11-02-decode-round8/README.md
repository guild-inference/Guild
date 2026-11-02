# Decode, eighth round: a DFlash 2 block drafter for Qwen3.8-Flash-Next

An opt-in DFlash 2 drafter (`GUILD_DFLASH=<checkpoint directory>`, z-lab/dflash's `DFlash2DraftModel`, arXiv
2602.06036) drafts for Qwen3.8-Flash-Next in place of its MTP layer. One pass of a five-layer non-causal transformer
over [anchor, mask x 7] proposes seven tokens. The drafter reads the target's layers 5, 15, 25, 35 and 45, with the
hyper-connection streams averaged. The verify window still decides every token: in every run below the texts equal
the MTP runs' token for token.

Against today's MTP setting, two alternated rounds on one binary measured:
- prose 40.0 -> 40.5 tok/s;
- code 50.2 -> 53.7 tok/s.

The drafter's own kernels cut a draft from 14.6 ms (MMVQ) and 8.6 ms (MMQ) to 6.3 ms. What limits a window now is
the verify window itself, and the draft's 7.4 ms in the engine.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes), GTX 1070 8 GB (sm_61, 15 SMs, 151 W
cap), CUDA 12. Branch `feat/nemotron-transcribe` (`ba4a941` and later), guild-kernels-cpu `9b7281a`, guild-kernels-cuda
`de65d60` (all local). Drafter checkpoint: incoai's DFlash 2 checkpoint for Qwen3.8-Flash-Next (bf16 safetensors,
1.2 GB).

## Results

`tools/bench_runtime.py` with the server's settings: `--dense q6_k --hc q8_0`, `--context 32768`,
`--resident-budget-gib 56`, 2 runs x 256 tokens a prompt. The MTP and DFlash runs alternated, 2 rounds, one binary.

| Setting | prose tok/s | code tok/s | prose: tokens, ms a window | code: tokens, ms a window |
| --- | --- | --- | --- | --- |
| MTP, `--spec 4 --spec-min-p 0.5` | 40.05 / 39.93 | 50.28 / 50.07 | 2.23, 55.6-55.8 (draft 5.7) | 2.98, 59.2-59.6 (draft 6.2) |
| DFlash, `--spec 8 --spec-min-p 0`, cumulative probability >= 0.1 | 40.55 / 40.40 | 53.78 / 53.57 | 2.51, 61.9-62.1 (draft 7.4) | 3.71, 68.9-69.4 (draft 7.4) |

- **Output.** The same 256 tokens in every run, both prompts. Neither setting keeps a GPU expert cache: 469 and
  441 slots were too few to pay for one.
- **VRAM.** The drafter takes 908 MiB; the MTP layer it replaces takes about 1.1 GB.
- **Time to first token.** Prose 73 against 61 ms, because the first window drafts too. Code 448 against 605 ms,
  because the 40-token prompt is read in windows of 8 instead of 4.

### Where DFlash gains and costs

- **Gain: more tokens a window.** DFlash fits more tokens in a window than MTP at the same number of rows: 2.51
  against 2.23 on prose, 3.71 against 2.98 on code.
- **Cost: the draft.** It takes 7.4 ms a window, against MTP's 5.7-6.2 ms.
- **Cost: larger windows.** The verify cost of a window grows with its rows. Measured with MTP drafts at
  `--spec-min-p 0`, draft excluded:

  | Rows | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | ms | 48.0 | 51.1 | 53.7 | 57.9 | 61.1 | 68.5 | 73.0 |

  Part of the step at 7 rows: the verify head's Q6_K product (staged MMVQ, isolated) takes 5.5 ms at 7 columns and
  6.1 at 8, against 3.0 at 4.

## How it is wired

- **Features.** The verifier records the drafter's features in its window graphs: the mean of the residual
  streams at each target layer, one kernel per group and layer (`Verifier::set_feature_layers`). The prompt path
  records them chunk by chunk (`Prefill::set_features`), for the prompt positions within the drafter's window of
  2048.
- **The drafter's context.** A position's features go through `fc` and `hidden_norm`, then into each layer's K/V,
  kept in fp16 rings of 2048 rows. The context follows the committed rows of every window. A request that reuses a
  prefix keeps the context when the drafter read the same token ids (the turns of a conversation). Before this, the
  bench's repeated prompt arrived with no context, which cost prose 3% of its tokens a window.
- **The session.** After a window, the session adds the committed rows' features. It then drafts the block after
  the window's last token and verifies the drafts while their cumulative probability stays >= 0.1
  (`GUILD_DFLASH_CUM`), up to `--spec` - 1. It reads the drafted tokens' PLE rows ahead, as the MTP path does. The
  first window drafts too, from the prompt's context.
- **The drafter's weights.**
  - Every matrix is quantized as Q8_0 at load, then kept as int8 rows with a scale a 32.
  - The head is the target's own, copied as Q4_K for all 248,320 rows.
  - The candidate selector's codebooks stay on the host, where the selector runs (7 x 16 candidates).

## The drafter's time

`dflash_check` replays drafting at every position of a dumped greedy text (prose, 257 drafts). GPU time a draft
(nsys), by step:

| Step | MMVQ | MMQ | Now |
| --- | ---: | ---: | ---: |
| The five layers' products (470 MB as Q8_0) | 4.03 ms (106K-row head build) | 3.66-3.80 ms | 2.4-2.7 ms (`d8_mm`) |
| The head (248,320 rows x 2560, Q4_K, 7 columns) | multi-column MMVQ, 3.71 ms at 106K rows | 2.62 ms | 2.4-2.6 ms (`k4_mm`) |
| Top 16 and log-sum-exp a row | 0.53 ms | 0.92 ms | 0.12 ms |
| Attention over the context | 0.72 ms | 0.72 ms | 0.36 ms |
| A draft, wall time | 14.58 ms (full head) | 8.62 ms | 6.34 ms |

Isolated kernel rates, on random data at 1 to 8 columns (`d8_mm` 15360 x 2560, `k4_mm` 248,320 x 2560):

| Columns | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `d8_mm` GB/s | 215 | 211 | 213 | 210 | 208 | 207 | 205 | 203 |
| `k4_mm` GB/s | 216 | 213 | 199 | 185 | 161 | 150 | 139 | 120 |

- **`d8_mm` is memory-bound.** A half-warp takes four rows. Its 16 lanes read the rows' 32-value blocks straight
  from memory, and the columns' int8 values from shared memory. MMQ reached 115-153 GB/s on these shapes: its grids
  are 40-240 blocks of 64-row tiles, and a tile waits on its loads.
- **`k4_mm` is compute-bound from three columns.** Unpacking the nibbles and the dot products at seven columns cost
  more than the bytes: 2.57 ms, against a 1.65 ms read.
- **Top 16.**
  1. Each of 64 blocks a row takes its part's max and sum of exponentials.
  2. Every logit at or above the row's 16th-largest block max is gathered. No top-16 member can fall below that.
  3. A block a row keeps the best 16.

  The previous kernel used a block a row and spent 0.92 ms.
- **Attention.** Split over the keys (runs of 256 keys, online softmax), then a merge, in place of one block per KV
  head.
- **The context path.** One product covers every layer's K and V, in batches of up to 64 rows with no host sync.
  Each committed row costs 0.04 ms, against 0.45 ms before.

In the engine a draft takes 7.4 ms a window. The verifier's asynchronous commit (about 2.5 ms of GDN replay and KV
appends) runs alongside it: the drafter's attention takes 0.57 ms there against 0.33 ms alone.

## Acceptance

Offline replays on the target's greedy texts. "Accepted" counts the leading drafts equal to the text, of 7.

| Text | Default numerics | `--dense q6_k --hc q8_0` |
| --- | ---: | ---: |
| prose (the bench's prompt) | 1.79 | 1.54 |
| code (the bench's prompt) | 4.67 | 2.46 |
| story | 1.40 | 1.10 |
| sql | 4.21 | 3.92 |

- **The CUDA drafter matches the PyTorch port.** On the first two texts the port gives 1.80 and 4.67, and their
  drafts agree at 97.7% of first positions. The default-numerics story and sql figures are the port's (Q8_0
  weights); the rest are the CUDA drafter's.
- **The two code texts differ.** The default-numerics answer opens with a doctest-style docstring, which is easy to
  draft. The `--dense q6_k` answer opens with a prose docstring instead.
- **Head precision, prose, 257 positions:** Q4_K 1.794, Q4_0 1.747, Q5_0 1.825 (22% more bytes).
- **A head over the 106,299-token MTP draft subset** accepts 1.47 against 1.80 on prose. 21 of the 256 generated
  tokens fall outside the subset, ordinary words among them: " ecosystems", " molecules", " precipitation". On the
  other texts: code 5, story 19 (" lantern", " keeper"), sql 20 (" VARCHAR", " PRIMARY").

## Window rules

The rule tried against the fixed threshold verifies the window size that maximizes 1 + the drafts' cumulative
probabilities, over the window's verify cost plus the draft's (EVICT, arXiv 2605.00342).

Simulated on the replay tables of the four `--dense q6_k` texts, with the measured verify costs above and a 7.4 ms
draft (tok/s):

| Rule | prose | code | story | sql |
| --- | ---: | ---: | ---: | ---: |
| cumulative >= 0.05 | 36.3 | 46.0 | 33.6 | 62.3 |
| cumulative >= 0.1 | 36.3 | 46.5 | 33.9 | 62.8 |
| cumulative >= 0.2 | 36.3 | 46.5 | 33.4 | 62.6 |
| cumulative >= 0.3 | 36.4 | 47.6 | 33.5 | 60.7 |
| cost rule, raw probabilities | 36.5 | 47.5 | 34.0 | 62.6 |
| cost rule, probabilities recalibrated per draft position on the other texts | 37.0 | 47.8 | 33.5 | 64.1 |

These texts are not the engine's live texts: the dump ran at 8K context and the texts part after 2 (prose) and 31
(code) tokens. Their absolute rates sit 10-15% under the live ones; the rules' order is what they show. The rules
differ by 3% at most.

In the engine (`GUILD_DFLASH_RULE=cost`) the rule learns each window size's verify cost from the windows it runs,
kept across requests, along with the draft's cost. Rows not yet seen get a line fitted through the measured ones.
Two alternated rounds against the 0.1 threshold, one binary, the same texts:

| tok/s | cumulative >= 0.1 | cost rule |
| --- | --- | --- |
| prose | 40.27 / 40.40 | 39.88 / 39.92 (2.46 against 2.51 tokens a window) |
| code | 53.57 / 53.53 | 54.13 / 54.19 (66.6 against 69.3 ms a window) |

On prose the rule drops drafts that the target accepts more often than their raw probabilities say. It stays
opt-in until the probabilities are calibrated (the simulation's best row).

## Where a window's time goes now

nsys of the engine with the drafter (prose, 60 windows; nsys adds about 8 ms of launch gap at each window's start).
GPU kernel time a window, summed over streams:

| Kernels | ms a window |
| --- | ---: |
| Dense products (MMVQ, staged; Q6_K, two token groups) | 34.8 |
| The hyper-connection read (`gr_down_q8` 6.6, `gr_up_q8` 5.0, norm 1.2) | 12.8 |
| GDN (projection 3.7, steps and commit replay 3.3, convolution 0.8, norm 0.2) | 8.0 |
| The drafter | 6.8 |
| BF16 products (routers and the like) | 3.7 |
| Waits on the CPU's experts | 1.0 |

- **The two token groups run on two streams and overlap.** The sum (74 ms) exceeds the window (69.7 ms under nsys).
- **The GDN gate/beta projection is off the critical path.** It takes 51 µs a launch at 480 KB, but runs on its own
  stream entirely under the dense products.
- **The hyper-connection read is latency-bound.** At two tokens the down kernel walks each row in 20 steps, with a
  barrier at each. Its latency is hidden by the other group's work, though: a deeper prefetch that keeps every bit
  (below) made the window slower.
- **The GPU is busy 86% of a window.** Most of the idle is nsys's own launch gap, so every millisecond of GPU work
  counts.

## Tried, not kept

| What | Result |
| --- | --- |
| The Q8_0 hyper-connection read with a 4-step weight ring and the activations fetched 2 steps ahead (bit for bit the read before it: `gr_parity --q8-bits`, 301,944 values) | MTP setting, 2 rounds: prose 40.25 / 39.98 -> 39.35 / 39.23, code 50.81 / 50.21 -> 49.42 / 49.28 tok/s (+1.1-1.5 ms a window); the same texts |
| The head copy as Q4_0 / Q5_0 through MMQ | 8.34 / 8.45 ms a draft against 6.59 (Q4_K through MMQ) |
| `k4_mm` with 2 rows a task; unpacking hoisted out of the column loop (153 registers, one block an SM); then capped at 128 registers; the columns swizzled in shared memory | 7.23; 6.52; 6.34-6.41; 6.40 ms a draft against 6.36 |
| `d8_mm` with 2 rows a task for the 2560-row products | 6.52 against 6.37 ms a draft |
| The drafter's products through MMQ (`GUILD_DFLASH_MMQ_MIN`) | 8.62 ms a draft (full head) against 6.34 with `d8_mm` / `k4_mm` |
| Without the drafter's context across a reused prefix | prose 2.42 tokens a window (38.9 tok/s) against 2.51 (39.7) |

## Open decisions

- **A frequency-ranked draft head of about 64K rows.** It would save about 1.9 ms a window: `k4_mm` scales with its
  rows, 2.57 ms for 248,320. The acceptance cost is not measured here. Published numbers: FR-Spec keeps 97.7-98.6%
  of the acceptance length at 64K rows; llama.cpp issue #25187 reports this model's MTP head keeping its acceptance
  at 65,536 rows. Building it needs a corpus, and it trades away drafts in the languages it leaves out: the same
  decision as the English MTP draft subset.
- **Whether DFlash becomes Qwen's default.** It needs the 1.2 GB checkpoint shipped (setup and packs) and the
  prompt path's features for a layer-split (multi-GPU) setup. Today it requires one stage.
- **`--dense q5_k` / `q4_k` for Qwen** (round 6, numerics) is still open.

## Tools

- `dflash_check DRAFTER GGUF VOCAB PREFIX B [REF]`: the drafter's acceptance and speed on a dumped text.
  `DFLASH_TABLE=file` writes every position's drafts for offline window rules.
- `qwen_feat_dump MODEL PACK PROFILE IDS N_NEW OUT DRAFTER`: the target's greedy text with the drafter's features.
  `QWEN_DUMP_DENSE` and `QWEN_DUMP_HC_Q8` give the server's `--dense` and `--hc q8_0`.
- `dflash_kernels_test`: the drafter's kernels against host references.
- `gr_parity --selftest --q8-bits FILE`: the Q8_0 hyper-connection read's outputs. It writes FILE when FILE is
  missing and compares bit for bit when it exists, for kernel changes that must keep the read's bits.
